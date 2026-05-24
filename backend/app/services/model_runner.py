"""
Load the trained XGBoost students (GPI, decline, stress) from backend-dev and
run pixel-level inference, then aggregate to a single bbox-level result that
matches the InferenceResult schema the frontend consumes.

Falls back to a deterministic `DemoModel` if any artifact fails to load — so
the live-inference demo still works in environments without the trained
pickles or rasterio.
"""

from __future__ import annotations

import logging
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

from app.config import settings

log = logging.getLogger(__name__)

# Tier cut-offs mirror backend-dev/config.py's BED_RISK_BINS, kept inline so we
# don't have to depend on backend-dev's `config` import at request time.
_RISK_BINS = [(0.0, 0.33, "low"), (0.33, 0.66, "moderate"), (0.66, 1.01, "high")]
_OOD_PVAL_HIGH = 0.05
_OOD_PVAL_LOW = 0.01

# Friendly labels for SHAP feature names → top_drivers strings.
_FRIENDLY = {
    "GB_ratio": "green/blue ratio (canopy density)",
    "B3": "green reflectance (biomass)",
    "red_edge_slope": "red-edge slope (canopy structure)",
    "turbidity": "turbidity (water clarity)",
    "NDAVI": "aquatic vegetation index",
    "WAVI": "water-adjusted vegetation index",
    "NDWI": "water/land discrimination",
    "depth_invariant": "Lyzenga depth-invariant",
    "SABI": "surface algal bloom index",
    "B8": "near-IR reflectance",
    "B11": "SWIR reflectance",
    "B12": "SWIR reflectance",
}


def _risk_tier(prob: float) -> str:
    for lo, hi, name in _RISK_BINS:
        if lo <= prob < hi:
            return name
    return "unknown"


def _friendly_driver(feature_name: str) -> str:
    for stem, phrase in _FRIENDLY.items():
        if stem in feature_name:
            stat = feature_name.split("_")[-1]
            return f"{phrase} ({stat})"
    return feature_name.replace("_", " ")


# ---------------------------------------------------------------------------
# Fallback model (kept around so the demo still runs without artifacts)
# ---------------------------------------------------------------------------

class DemoModel:
    def predict_payload(self, summary: Dict[str, float]) -> Dict[str, Any]:
        scene_count = summary.get("scene_count", 0)
        cloud = summary.get("mean_cloud_cover", 30.0)
        stability = max(0.1, 1.0 - cloud / 100.0)
        ndvi = 0.55
        risk = 1 / (1 + math.exp(-(1.2 - 2.0 * ndvi - 1.0 * stability)))
        confidence = max(0.35, min(0.96, 0.55 + 0.04 * scene_count + 0.15 * stability))
        eelgrass = max(5.0, min(92.0, 60 + 18 * ndvi + 1.5 * math.log1p(scene_count)))
        return {
            "eelgrass_pct_estimate": round(eelgrass, 2),
            "depletion_risk_90d": round(float(risk), 3),
            "confidence": round(float(confidence), 3),
            "top_drivers": ["NDVI-like proxy", "scene stability", "cloud cover"],
        }


# ---------------------------------------------------------------------------
# Real model service
# ---------------------------------------------------------------------------

class ModelService:
    def __init__(self) -> None:
        self.gpi_model = None
        self.decline_model = None
        self.stress_model = None
        self.stress_encoder = None
        self.feature_cols: Optional[List[str]] = None
        self.train_mean: Optional[np.ndarray] = None
        self.train_cov_inv: Optional[np.ndarray] = None
        self.shap_decline: Optional[pd.DataFrame] = None
        self.loaded = False
        self.demo = DemoModel()

        self._load()

    def _load(self) -> None:
        models_dir = Path(settings.models_dir)
        if not models_dir.exists():
            log.warning("MODELS_DIR does not exist: %s — falling back to DemoModel", models_dir)
            return

        try:
            with open(models_dir / "gpi_model.pkl", "rb") as f:
                self.gpi_model = pickle.load(f)
            with open(models_dir / "decline_model.pkl", "rb") as f:
                self.decline_model = pickle.load(f)
            with open(models_dir / "stress_model.pkl", "rb") as f:
                stress_bundle = pickle.load(f)
            self.stress_model, self.stress_encoder = stress_bundle
            with open(models_dir / "feature_cols.pkl", "rb") as f:
                self.feature_cols = pickle.load(f)
            with open(models_dir / "train_X_mean.pkl", "rb") as f:
                self.train_mean = np.asarray(pickle.load(f), dtype=float)
            with open(models_dir / "train_cov_inv.pkl", "rb") as f:
                self.train_cov_inv = np.asarray(pickle.load(f), dtype=float)
        except FileNotFoundError as exc:
            log.warning("Missing model artifact in %s (%s) — falling back to DemoModel", models_dir, exc)
            return
        except Exception:
            log.exception("Failed to load model artifacts from %s", models_dir)
            return

        shap_path = models_dir / "shap_decline.csv"
        if shap_path.exists():
            try:
                self.shap_decline = pd.read_csv(shap_path)
            except Exception:
                log.exception("Failed to read %s", shap_path)

        self.loaded = True
        log.info("Loaded XGBoost models from %s (%d features)", models_dir, len(self.feature_cols))

    # --- inference ---------------------------------------------------------

    def _align(self, features: pd.DataFrame) -> pd.DataFrame:
        """Reindex `features` to the model's training feature columns, filling
        any missing column with the training-set mean."""
        X = features.reindex(columns=self.feature_cols)
        if self.train_mean is not None and len(self.train_mean) == len(self.feature_cols):
            means = pd.Series(self.train_mean, index=self.feature_cols)
            X = X.fillna(means)
        return X.fillna(0.0)

    def _mahalanobis_pvals(self, X: pd.DataFrame) -> np.ndarray:
        if self.train_mean is None or self.train_cov_inv is None:
            return np.full(len(X), np.nan)
        dists = np.empty(len(X))
        mean = self.train_mean
        cov_inv = self.train_cov_inv
        for i, row in enumerate(X.values):
            try:
                dists[i] = mahalanobis(row, mean, cov_inv)
            except Exception:
                dists[i] = np.nan
        return 1.0 - chi2.cdf(dists ** 2, df=X.shape[1])

    def _confidence(self, pvals: np.ndarray, obs_ratio: np.ndarray) -> str:
        good = float(np.nanmean((pvals > _OOD_PVAL_HIGH) & (obs_ratio >= 0.5)))
        ok = float(np.nanmean((pvals > _OOD_PVAL_LOW) | (obs_ratio >= 0.25)))
        if good >= 0.5:
            return "high"
        if ok >= 0.5:
            return "medium"
        return "low"

    def _top_drivers(self, predominant_stress: Optional[str]) -> List[str]:
        drivers: List[str] = []
        if self.shap_decline is not None and not self.shap_decline.empty:
            top_feats = self.shap_decline.head(3)["feature"].tolist()
            drivers.extend(_friendly_driver(f) for f in top_feats)
        if predominant_stress and predominant_stress not in {"unknown", None}:
            drivers.append(f"dominant stress regime: {predominant_stress}")
        return drivers[:4] or ["model loaded — no SHAP available"]

    def predict_from_features(
        self,
        features: pd.DataFrame,
        meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run the three students on a pixel × window feature matrix and
        aggregate to one bbox-level result for the frontend.

        Aggregation: take the *latest* window per pixel, then weight by each
        pixel's valid_obs_count when collapsing to bbox averages.
        """
        if not self.loaded:
            return self.demo.predict_payload(meta)

        if features.empty:
            return self.demo.predict_payload(meta)

        # Keep the most recent window per pixel — this is what "decline now" means.
        latest = (
            features.sort_values("window_center")
            .groupby("pixel_id", sort=False)
            .tail(1)
            .reset_index(drop=True)
        )

        X = self._align(latest)
        pred_gpi = np.clip(self.gpi_model.predict(X), 0.0, 1.0)
        pred_decl = np.clip(self.decline_model.predict(X), 0.0, 1.0)
        stress_idx = self.stress_model.predict(X)
        stress_labels = self.stress_encoder.inverse_transform(stress_idx)

        pvals = self._mahalanobis_pvals(X)
        obs_ratio = (latest["valid_obs_count"].astype(float) / 6.0).clip(0, 1).values

        weights = latest["valid_obs_count"].astype(float).values
        if weights.sum() <= 0:
            weights = np.ones_like(weights)

        mean_gpi = float(np.average(pred_gpi, weights=weights))
        mean_decline = float(np.average(pred_decl, weights=weights))
        stress_series = pd.Series(stress_labels)
        dominant = stress_series.mode().iloc[0] if not stress_series.empty else None

        eelgrass_pct = round(mean_gpi * 100.0, 2)
        depletion_risk = round(mean_decline, 3)
        confidence = self._confidence(pvals, obs_ratio)
        confidence_score = {"low": 0.45, "medium": 0.72, "high": 0.9}[confidence]

        return {
            "eelgrass_pct_estimate": eelgrass_pct,
            "depletion_risk_90d": depletion_risk,
            "confidence": confidence_score,
            "confidence_tier": confidence,
            "risk_tier": _risk_tier(depletion_risk),
            "dominant_stress": dominant,
            "top_drivers": self._top_drivers(dominant),
            "pixels_scored": int(len(latest)),
            "scene_count": int(meta.get("scene_count", 0)),
        }

    # --- backward-compatible legacy entry point ---------------------------

    def predict(self, feats: Dict[str, float]) -> Dict[str, Any]:
        """Kept so any legacy callers still get a sensible payload."""
        return self.demo.predict_payload(feats)


model_service = ModelService()
