"""
pipeline.inference
------------------
Loads the trained student models and scores a target-region S2 feature CSV.
Produces two outputs:

    data/predictions/<region>_pixels.csv  — per pixel × window:
        predicted_GPI, predicted_decline_prob, predicted_stress,
        prob_<class>, risk_tier, spectral_distance, spectral_pval,
        valid_obs_ratio, confidence

    data/predictions/<region>_beds.csv    — per bed × window:
        bed_GPI, bed_decline_prob, high_risk_pixel_frac, dominant_stress,
        bed_risk_tier, bed_confidence, n_pixels

Usage:
    python -m pipeline.inference <features_csv> <region_id>
"""

from __future__ import annotations
import sys
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.spatial.distance import mahalanobis

from config import (
    MODELS_DIR, PREDICTIONS_DIR, REGIONS_FILE,
    DECLINE_PIXEL_THRESHOLD, BED_RISK_BINS,
    EXPECTED_PASSES_PER_WINDOW, OOD_PVAL_HIGH, OOD_PVAL_LOW,
)


# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------

def load_artifacts():
    def _load(fname):
        with open(MODELS_DIR / fname, 'rb') as f:
            return pickle.load(f)
    return (
        _load('gpi_model.pkl'),
        _load('decline_model.pkl'),
        *_load('stress_model.pkl'),
        _load('feature_cols.pkl'),
        _load('train_X_mean.pkl'),
        _load('train_cov_inv.pkl'),
    )


# ---------------------------------------------------------------------------
# OOD detection
# ---------------------------------------------------------------------------

def mahalanobis_distance(X: pd.DataFrame, mean: np.ndarray, cov_inv: np.ndarray):
    dists = np.empty(len(X))
    for i, row in enumerate(X.values):
        try:
            dists[i] = mahalanobis(row, mean, cov_inv)
        except Exception:
            dists[i] = np.nan
    pvals = 1 - chi2.cdf(dists ** 2, df=X.shape[1])
    return dists, pvals


def confidence_tier(row) -> str:
    obs_ratio = float(row.get('valid_obs_ratio', 0))
    pval      = float(row.get('spectral_pval', 0))
    if obs_ratio >= 0.5 and pval > OOD_PVAL_HIGH:
        return 'High'
    if obs_ratio >= 0.25 or pval > OOD_PVAL_LOW:
        return 'Medium'
    return 'Low'


def assign_risk_tier(prob: float) -> str:
    for lo, hi, name in BED_RISK_BINS:
        if lo <= prob < hi:
            return name
    return 'Unknown'


# ---------------------------------------------------------------------------
# Pixel-level inference
# ---------------------------------------------------------------------------

def run_pixel_inference(s2_df, gpi_m, decline_m, stress_m, stress_enc,
                        feature_cols, train_mean, train_cov_inv) -> pd.DataFrame:
    X = s2_df[feature_cols].fillna(0)

    pred_gpi    = gpi_m.predict(X)
    pred_decl   = np.clip(decline_m.predict(X), 0, 1)
    stress_prob = stress_m.predict_proba(X)
    stress_lab  = stress_enc.inverse_transform(stress_m.predict(X))

    out = s2_df[['pixel_id', 'SITE', 'latitude', 'longitude',
                  'window_center', 'valid_obs_count']].copy()
    out['predicted_GPI']         = pred_gpi.round(4)
    out['predicted_decline_prob'] = pred_decl.round(4)
    out['predicted_stress']      = stress_lab
    for i, cls in enumerate(stress_enc.classes_):
        out[f'prob_{cls}'] = stress_prob[:, i].round(4)

    out['risk_tier'] = out['predicted_decline_prob'].apply(assign_risk_tier)

    # OOD
    dists, pvals = mahalanobis_distance(X, train_mean, train_cov_inv)
    out['spectral_distance'] = np.round(dists, 4)
    out['spectral_pval']     = np.round(pvals, 4)

    # Data quality
    out['valid_obs_ratio'] = (out['valid_obs_count']
                               / EXPECTED_PASSES_PER_WINDOW).clip(0, 1).round(3)
    out['confidence']      = out.apply(confidence_tier, axis=1)
    return out


# ---------------------------------------------------------------------------
# Bed-level aggregation
# ---------------------------------------------------------------------------

def aggregate_to_bed(pixel_df: pd.DataFrame, region_id: str) -> pd.DataFrame:
    regions = json.loads(Path(REGIONS_FILE).read_text())['regions']
    bed_map = {}
    for r in regions:
        if r['id'] == region_id:
            for b in r['beds']:
                bed_map[b['site']] = b['id']

    rows = []
    for (site, wc), g in pixel_df.groupby(['SITE', 'window_center']):
        weights = g['valid_obs_count'].values
        w_total = weights.sum()
        if w_total == 0:
            continue

        w_decline = float(np.average(g['predicted_decline_prob'], weights=weights))
        w_gpi     = float(np.average(g['predicted_GPI'], weights=weights))
        high_frac = float((g['predicted_decline_prob'] > DECLINE_PIXEL_THRESHOLD).mean())
        dom_stress = g['predicted_stress'].mode().iloc[0]

        conf_score = g['confidence'].map({'High': 3, 'Medium': 2, 'Low': 1}).fillna(1).mean()
        bed_conf = ('High' if conf_score >= 2.5
                    else 'Medium' if conf_score >= 1.5
                    else 'Low')

        rows.append({
            'bed_id':                bed_map.get(site, f'{region_id}_{site}'),
            'region_id':             region_id,
            'site':                  site,
            'window_center':         wc,
            'n_pixels':              int(len(g)),
            'bed_GPI':               round(w_gpi, 4),
            'bed_decline_prob':      round(w_decline, 4),
            'high_risk_pixel_frac':  round(high_frac, 4),
            'dominant_stress':       dom_stress,
            'bed_risk_tier':         assign_risk_tier(w_decline),
            'bed_confidence':        bed_conf,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage: python -m pipeline.inference <features_csv> <region_id>')
        sys.exit(1)

    features_csv = sys.argv[1]
    region_id    = sys.argv[2]

    print(f'Loading artifacts from {MODELS_DIR}')
    (gpi_m, decline_m, stress_m, stress_enc,
     feature_cols, train_mean, train_cov_inv) = load_artifacts()

    print(f'Loading features: {features_csv}')
    s2 = pd.read_csv(features_csv)
    s2['window_center'] = pd.to_datetime(s2['window_center'])
    print(f'  rows={len(s2):,}  pixels={s2["pixel_id"].nunique():,}')

    # Ensure missing feature columns are present (pixel-window matrix from a
    # region may lack some pruned columns; fill with 0)
    for c in feature_cols:
        if c not in s2.columns:
            s2[c] = 0.0

    print('Pixel inference …')
    pixels = run_pixel_inference(s2, gpi_m, decline_m, stress_m, stress_enc,
                                  feature_cols, train_mean, train_cov_inv)

    print('Bed aggregation …')
    beds = aggregate_to_bed(pixels, region_id)

    pixels_out = PREDICTIONS_DIR / f'{region_id}_pixels.csv'
    beds_out   = PREDICTIONS_DIR / f'{region_id}_beds.csv'
    pixels.to_csv(pixels_out, index=False)
    beds.to_csv(beds_out, index=False)

    print(f'  → {pixels_out}  ({len(pixels):,} rows)')
    print(f'  → {beds_out}    ({len(beds):,} rows)')

    print('\nBed risk-tier counts:')
    print(beds['bed_risk_tier'].value_counts().to_string())
    print('\nDominant stress counts:')
    print(beds['dominant_stress'].value_counts().to_string())
