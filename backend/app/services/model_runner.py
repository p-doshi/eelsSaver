import json
import os
import joblib
import math
from app.config import settings

class DemoModel:
    feature_names = [
        "scene_count",
        "mean_cloud_cover",
        "bbox_area_proxy",
        "seasonal_phase",
        "nir_proxy",
        "red_proxy",
        "green_proxy",
        "swir_proxy",
        "ndvi_like",
        "ndwi_like",
        "turbidity_proxy",
        "texture_proxy",
        "temporal_stability_proxy",
    ]

    def predict_payload(self, feats: dict):
        ndvi = feats["ndvi_like"]
        turb = feats["turbidity_proxy"]
        stability = feats["temporal_stability_proxy"]
        scene_count = feats["scene_count"]

        eelgrass = max(5.0, min(92.0, 65 + 18 * ndvi - 4.2 * turb + 8 * stability + 1.5 * math.log1p(scene_count)))
        risk = 1 / (1 + math.exp(-(1.8 * turb - 2.4 * ndvi - 1.1 * stability + 0.4)))
        confidence = max(0.35, min(0.96, 0.58 + 0.04 * scene_count + 0.18 * stability))

        drivers = sorted([
            ("ndvi_like", abs(18 * ndvi)),
            ("turbidity_proxy", abs(-4.2 * turb)),
            ("temporal_stability_proxy", abs(8 * stability)),
            ("scene_count", abs(1.5 * math.log1p(scene_count)))
        ], key=lambda x: x[1], reverse=True)

        return {
            "eelgrass_pct_estimate": round(eelgrass, 2),
            "depletion_risk_90d": round(float(risk), 3),
            "confidence": round(float(confidence), 3),
            "top_drivers": [d[0] for d in drivers[:3]],
        }

class ModelService:
    def __init__(self):
        self.model = None
        self.meta = {}
        self.demo = DemoModel()
        if os.path.exists(settings.model_path):
            try:
                self.model = joblib.load(settings.model_path)
            except Exception:
                self.model = None
        if os.path.exists(settings.model_meta_path):
            with open(settings.model_meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)

    def predict(self, feats: dict):
        if self.model is None:
            return self.demo.predict_payload(feats)

        feature_names = self.meta.get("feature_names", list(feats.keys()))
        X = [[feats.get(name, 0.0) for name in feature_names]]

        eelgrass = float(self.model["cover"].predict(X)[0]) if isinstance(self.model, dict) and "cover" in self.model else 50.0
        risk = float(self.model["risk"].predict_proba(X)[0][1]) if isinstance(self.model, dict) and "risk" in self.model else 0.5

        conf = 0.8
        return {
            "eelgrass_pct_estimate": round(eelgrass, 2),
            "depletion_risk_90d": round(risk, 3),
            "confidence": round(conf, 3),
            "top_drivers": feature_names[:3],
        }

model_service = ModelService()