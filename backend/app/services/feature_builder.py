from typing import Dict, Any, List
import math

def build_features_from_stac(stac_response: Dict[str, Any], bbox: List[float]) -> Dict[str, float]:
    features = stac_response.get("features", [])
    scene_count = len(features)
    if scene_count == 0:
        raise ValueError("No Sentinel-2 scenes found for this area/date range.")

    cloud_vals = []
    months = []
    for f in features:
        props = f.get("properties", {})
        cc = props.get("eo:cloud_cover")
        dt = props.get("datetime", "")
        if cc is not None:
            cloud_vals.append(float(cc))
        if len(dt) >= 7:
            months.append(int(dt[5:7]))

    minx, miny, maxx, maxy = bbox
    area_proxy = abs((maxx - minx) * (maxy - miny))

    mean_cloud = sum(cloud_vals) / max(len(cloud_vals), 1)
    month_mean = sum(months) / max(len(months), 1)
    seasonal_phase = math.sin((month_mean / 12.0) * 2 * math.pi)

    return {
        "scene_count": float(scene_count),
        "mean_cloud_cover": mean_cloud,
        "bbox_area_proxy": area_proxy,
        "seasonal_phase": seasonal_phase,
        "nir_proxy": 0.61,
        "red_proxy": 0.18,
        "green_proxy": 0.12,
        "swir_proxy": 0.09,
        "ndvi_like": (0.61 - 0.18) / (0.61 + 0.18),
        "ndwi_like": (0.12 - 0.61) / (0.12 + 0.61),
        "turbidity_proxy": (0.18 + 0.12) / max(0.09, 1e-6),
        "texture_proxy": 0.33,
        "temporal_stability_proxy": max(0.1, 1.0 - mean_cloud / 100.0),
    }