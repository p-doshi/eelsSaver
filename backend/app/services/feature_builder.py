"""
Build the pixel × 90-day-window feature matrix expected by the trained
XGBoost students, sourcing pixel data from Google Earth Engine
(COPERNICUS/S2_SR_HARMONIZED).

This file mirrors backend-dev/gee/forillon_extraction.js so inference uses
the same preprocessing chain as training: same collection, same scaling
(DN / 10000), same 10 index formulas, same sample-at-10m semantics.

Flow:
    1. Build a PIXEL_GRID × PIXEL_GRID lat/lon point grid inside the bbox.
    2. Filter COPERNICUS/S2_SR_HARMONIZED by bbox, date, cloud.
    3. Server-side: divide-by-10000 + compute the 10 indices per image.
    4. sampleRegions our fixed point grid for each image, tag DATE+CLOUD_PCT.
    5. .getInfo() pulls the FeatureCollection to Python in one shot.
    6. Run pipeline.s2_features.compute_pixel_window_features (reused).

Auth:
    Run `earthengine authenticate --project=<GEE_PROJECT>` once on the host
    before starting the server. Credentials cache to
    ~/.config/earthengine/credentials and are reused across restarts.
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import ee
import pandas as pd

from app.config import settings

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# Allow `from pipeline.s2_features import ...` and `from config import ...`
_BACKEND_DEV = Path(__file__).resolve().parents[3] / "backend-dev"
if str(_BACKEND_DEV) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DEV))

from pipeline.s2_features import compute_pixel_window_features  # noqa: E402

_BANDS = ["B2", "B3", "B4", "B5", "B6", "B8", "B8A", "B11", "B12"]
_INDICES = [
    "NDAVI", "WAVI", "GB_ratio", "RG_ratio", "B3B2_diff",
    "NDWI", "turbidity", "red_edge_slope", "SABI", "depth_invariant",
]
_FEATURE_PROPS = (
    ["SITE", "DATE", "CLOUD_PCT", "longitude", "latitude"]
    + _BANDS + _INDICES
)

_ee_initialized = False


def _ensure_ee() -> None:
    """Lazy ee.Initialize so import-time failures don't crash the server."""
    global _ee_initialized
    if _ee_initialized:
        return
    if not settings.gee_project:
        raise RuntimeError(
            "GEE_PROJECT is not set. Add it to .env or export the variable, then restart."
        )
    try:
        ee.Initialize(project=settings.gee_project, opt_url="https://earthengine.googleapis.com")
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine initialization failed. Run "
            f"`earthengine authenticate --project={settings.gee_project}` "
            f"once on this host, then restart the server. Underlying error: {exc}"
        ) from exc
    _ee_initialized = True
    log.info("Earth Engine initialized for project=%s", settings.gee_project)


# ---------------------------------------------------------------------------
# Server-side image preprocessing — direct port of forillon_extraction.js
# ---------------------------------------------------------------------------

def _add_indices(image: "ee.Image") -> "ee.Image":
    s = 10_000
    refl = image.select(_BANDS).divide(s).rename(_BANDS)
    eps = ee.Image(1e-6)

    B2 = refl.select("B2")
    B3 = refl.select("B3")
    B4 = refl.select("B4")
    B5 = refl.select("B5")
    B6 = refl.select("B6")
    B8 = refl.select("B8")
    B11 = refl.select("B11")
    B12 = refl.select("B12")

    NDAVI = B8.subtract(B4).divide(B8.add(B4).add(eps)).rename("NDAVI")
    WAVI = (
        B8.subtract(B4).multiply(1.5)
        .divide(B8.add(B4).add(0.5))
        .rename("WAVI")
    )
    GB = B3.divide(B2.add(eps)).rename("GB_ratio")
    RG = B4.divide(B3.add(eps)).rename("RG_ratio")
    B3B2 = B3.subtract(B2).rename("B3B2_diff")
    NDWI = B3.subtract(B8).divide(B3.add(B8).add(eps)).rename("NDWI")
    TURB = B4.divide(B3.add(eps)).rename("turbidity")
    # forillon_extraction.js uses .divide(35) for the red-edge slope — kept identical.
    RES = B6.subtract(B5).divide(35).rename("red_edge_slope")
    SABI = B8.subtract(B4).divide(B3.add(B2).add(eps)).rename("SABI")
    DEPTH = B2.log().divide(B3.log().add(eps)).rename("depth_invariant")

    return (
        refl.addBands([NDAVI, WAVI, GB, RG, B3B2, NDWI, TURB, RES, SABI, DEPTH])
        .set("system:time_start", image.get("system:time_start"))
        .set("CLOUDY_PIXEL_PERCENTAGE", image.get("CLOUDY_PIXEL_PERCENTAGE"))
    )


# ---------------------------------------------------------------------------
# Sampling grid
# ---------------------------------------------------------------------------

def _build_point_grid(bbox: List[float], grid: int, site_name: str) -> "ee.FeatureCollection":
    """Fixed lat/lon grid inside bbox. Same grid for every scene so pipeline.s2_features
    can group by pixel_id across the time series."""
    lon_min, lat_min, lon_max, lat_max = bbox
    dx = (lon_max - lon_min) / grid
    dy = (lat_max - lat_min) / grid
    feats = []
    for i in range(grid):
        lat = lat_min + dy * (i + 0.5)
        for j in range(grid):
            lon = lon_min + dx * (j + 0.5)
            feats.append(
                ee.Feature(
                    ee.Geometry.Point([lon, lat]),
                    {"longitude": lon, "latitude": lat, "SITE": site_name},
                )
            )
    return ee.FeatureCollection(feats)


# ---------------------------------------------------------------------------
# Main entry — replaces the previous Planetary Computer implementation
# ---------------------------------------------------------------------------

def build_feature_matrix(
    bbox: List[float],
    datetime_start: str,
    datetime_end: str,
    max_cloud: float,
    site_name: str = "bbox_query",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """STAC search → indices → sample → 90-day rolling features, all via GEE."""
    _ensure_ee()

    region = ee.Geometry.Rectangle(bbox)
    grid = settings.pixel_grid
    points = _build_point_grid(bbox, grid, site_name)

    collection = (
        ee.ImageCollection(settings.gee_collection)
        .filterBounds(region)
        .filterDate(datetime_start, datetime_end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
        .sort("system:time_start")
        .limit(settings.max_scenes)
    )

    n_scenes = int(collection.size().getInfo())
    if n_scenes == 0:
        raise ValueError(
            f"No Sentinel-2 scenes found for bbox={bbox} "
            f"in [{datetime_start}, {datetime_end}] with cloud < {max_cloud}% "
            "(GEE COPERNICUS/S2_SR_HARMONIZED)."
        )

    scaled = collection.map(_add_indices)

    def _sample_image(image):
        date = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd")
        cloud = ee.Number(image.get("CLOUDY_PIXEL_PERCENTAGE"))
        sampled = image.sampleRegions(
            collection=points,
            scale=10,
            projection="EPSG:4326",
            geometries=False,
        )
        return sampled.map(
            lambda f: f.set({"DATE": date, "CLOUD_PCT": cloud})
        )

    # GEE's getInfo() refuses any FeatureCollection result > 5000 elements,
    # so we batch scenes such that (scenes_per_batch × grid²) stays under that.
    # Use 4500 as a safety margin.
    pixels_per_scene = grid * grid
    scenes_per_batch = max(1, 4500 // pixels_per_scene)

    all_rows: List[Dict[str, Any]] = []
    n_batches = (n_scenes + scenes_per_batch - 1) // scenes_per_batch
    for b in range(n_batches):
        offset = b * scenes_per_batch
        batch_list = scaled.toList(scenes_per_batch, offset)
        batch_ic = ee.ImageCollection(batch_list)
        batch_fc = batch_ic.map(_sample_image).flatten()
        try:
            result = batch_fc.select(_FEATURE_PROPS).getInfo()
        except ee.EEException as exc:
            raise RuntimeError(
                f"Earth Engine request failed on batch {b + 1}/{n_batches}: {exc}. "
                f"If this is an auth error, run "
                f"`earthengine authenticate --project={settings.gee_project}`."
            ) from exc
        batch_features = result.get("features", [])
        log.info(
            "GEE batch %d/%d (scenes %d–%d): %d pixel-scene rows",
            b + 1, n_batches,
            offset + 1, min(offset + scenes_per_batch, n_scenes),
            len(batch_features),
        )
        all_rows.extend(f["properties"] for f in batch_features)

    if not all_rows:
        raise ValueError(
            "Earth Engine returned 0 sampled pixels. The bbox may not "
            "intersect any cloud-free Sentinel-2 scenes in that date range."
        )

    pixel_df = pd.DataFrame(all_rows)

    # Drop sample failures (sampleRegions emits null for masked/missing bands)
    pixel_df = pixel_df.dropna(subset=["B2", "B3", "B4"], how="all")
    if pixel_df.empty:
        raise ValueError(
            "All sampled pixels were masked out by GEE (likely ocean nodata or cloud mask)."
        )

    pixel_df["DATE"] = pd.to_datetime(pixel_df["DATE"])

    # Match the schema pipeline.s2_features.load_gee_export() expects.
    pixel_df["pixel_lat"] = pixel_df["latitude"].astype(float).round(4)
    pixel_df["pixel_lon"] = pixel_df["longitude"].astype(float).round(4)
    pixel_df["pixel_id"] = (
        pixel_df["pixel_lat"].astype(str)
        + "_"
        + pixel_df["pixel_lon"].astype(str)
        + "_"
        + pixel_df["SITE"].astype(str)
    )

    features = compute_pixel_window_features(pixel_df)

    meta = {
        "scene_count": n_scenes,
        "mean_cloud_cover": float(pixel_df["CLOUD_PCT"].mean()),
        "date_min": pixel_df["DATE"].min().isoformat(),
        "date_max": pixel_df["DATE"].max().isoformat(),
        "pixel_count": int(pixel_df["pixel_id"].nunique()),
        "window_count": int(features["window_center"].nunique()) if not features.empty else 0,
        "bbox": list(bbox),
        "pixel_grid": grid,
        "data_source": f"earthengine:{settings.gee_collection}",
    }
    return features, meta
