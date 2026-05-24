"""
Build the ~140-column pixel × 90-day-window feature matrix expected by the
trained XGBoost students (backend-dev/pipeline/student.py).

Flow:
    1. STAC search for Sentinel-2 L2A scenes covering bbox in [start, end].
    2. For each scene, sample a PIXEL_GRID × PIXEL_GRID grid of reflectances
       for bands B2..B12 from the signed COGs (Microsoft Planetary Computer).
    3. Compute the 10 spectral indices used by the GEE extraction script.
    4. Run backend-dev's `pipeline.s2_features.compute_pixel_window_features`
       to produce the rolling-window feature matrix.

Reuses the pipeline directly via sys.path so the feature schema stays in
sync with what student.py was trained on.
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import planetary_computer
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.warp import transform as rio_transform
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

from app.config import settings

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# Allow `from pipeline.s2_features import ...` and `from config import ...`
_BACKEND_DEV = Path(__file__).resolve().parents[3] / "backend-dev"
if str(_BACKEND_DEV) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DEV))

from pipeline.s2_features import compute_pixel_window_features  # noqa: E402

# Planetary Computer Sentinel-2 asset keys → the column names the pipeline expects.
_BAND_TO_COL = {
    "B02": "B2",
    "B03": "B3",
    "B04": "B4",
    "B05": "B5",
    "B06": "B6",
    "B08": "B8",
    "B8A": "B8A",
    "B11": "B11",
    "B12": "B12",
}
_S2_REFLECTANCE_SCALE = 10_000.0  # S2 L2A surface reflectance is stored as int * 10_000


def search_scenes(
    bbox: List[float],
    datetime_start: str,
    datetime_end: str,
    max_cloud: float,
) -> List[Any]:
    """Return up to `settings.max_scenes` STAC items, oldest first."""
    catalog = Client.open(settings.stac_url, modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=[settings.stac_collection],
        bbox=bbox,
        datetime=f"{datetime_start}/{datetime_end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    items = list(search.items())
    items.sort(key=lambda it: it.datetime)
    return items[: settings.max_scenes]


def _read_band(href: str, bbox: List[float], grid: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a single COG band over `bbox` (lon/lat) and resample to `grid × grid`.

    Returns (reflectance_grid, lon_grid, lat_grid), each shaped (grid * grid,).
    """
    with rasterio.open(href) as src:
        bounds_native = transform_bounds("EPSG:4326", src.crs, *bbox)
        window = from_bounds(*bounds_native, transform=src.transform)
        arr = src.read(
            1,
            window=window,
            out_shape=(grid, grid),
            resampling=Resampling.average,
        ).astype("float32")

        # Compute the affine of the down-sampled output and pull pixel-center coords.
        win_transform = src.window_transform(window)
        out_transform = win_transform * win_transform.scale(
            window.width / grid, window.height / grid
        )
        rows, cols = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
        xs, ys = rasterio.transform.xy(out_transform, rows.ravel() + 0.5, cols.ravel() + 0.5)
        lon, lat = rio_transform(src.crs, "EPSG:4326", xs, ys)

    refl = arr.ravel() / _S2_REFLECTANCE_SCALE
    # Treat nodata (0) and absurd values as NaN.
    refl = np.where((refl > 0) & (refl < 2.0), refl, np.nan)
    return refl, np.asarray(lon), np.asarray(lat)


def _add_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 10 indices the pipeline was trained on."""
    eps = 1e-6
    B2, B3, B4, B5 = df["B2"], df["B3"], df["B4"], df["B5"]
    B6, B8, _B8A, B11, B12 = df["B6"], df["B8"], df["B8A"], df["B11"], df["B12"]

    df["NDAVI"] = (B8 - B2) / (B8 + B2 + eps)
    df["WAVI"] = 1.5 * (B8 - B2) / (B8 + B2 + 0.5)
    df["GB_ratio"] = B3 / (B2 + eps)
    df["RG_ratio"] = B4 / (B3 + eps)
    df["B3B2_diff"] = B3 - B2
    df["NDWI"] = (B3 - B8) / (B3 + B8 + eps)
    df["turbidity"] = (B4 + B3) / (B11 + B12 + eps)
    df["red_edge_slope"] = (B6 - B5) / (B6 + B5 + eps)
    df["SABI"] = (B8 - B4) / (B2 + B3 + eps)
    # Lyzenga depth-invariant index — log ratio, well-defined only for positive reflectance.
    safe_b2 = np.clip(B2.values, eps, None)
    safe_b3 = np.clip(B3.values, eps, None)
    df["depth_invariant"] = np.log(safe_b2) - np.log(safe_b3)
    return df


def build_scene_pixel_dataframe(
    items: List[Any],
    bbox: List[float],
    site_name: str,
    grid: int,
) -> pd.DataFrame:
    """Match the GEE-export schema that pipeline.s2_features.load_gee_export() expects."""
    scene_frames = []

    for item in items:
        per_band: Dict[str, np.ndarray] = {}
        lon = lat = None
        for asset_key, col in _BAND_TO_COL.items():
            asset = item.assets.get(asset_key)
            if asset is None:
                per_band[col] = np.full(grid * grid, np.nan, dtype="float32")
                continue
            try:
                refl, lon, lat = _read_band(asset.href, bbox, grid)
            except Exception as exc:
                log.warning("Failed to read %s for %s: %s", asset_key, item.id, exc)
                per_band[col] = np.full(grid * grid, np.nan, dtype="float32")
                continue
            per_band[col] = refl

        if lon is None or lat is None:
            continue

        scene_df = pd.DataFrame(per_band)
        scene_df["DATE"] = pd.Timestamp(item.datetime).tz_localize(None)
        scene_df["SITE"] = site_name
        scene_df["CLOUD_PCT"] = float(item.properties.get("eo:cloud_cover", 0.0))
        scene_df["latitude"] = lat
        scene_df["longitude"] = lon
        scene_frames.append(scene_df)

    if not scene_frames:
        return pd.DataFrame()

    df = pd.concat(scene_frames, ignore_index=True)
    df = df.dropna(subset=["B2", "B3", "B4"], how="all")
    df = _add_indices(df)
    return df


def build_feature_matrix(
    bbox: List[float],
    datetime_start: str,
    datetime_end: str,
    max_cloud: float,
    site_name: str = "bbox_query",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run the full STAC → pixel → 90-day-rolling-features pipeline.

    Returns (features_df, meta). `meta` carries summary stats the API surfaces
    to the frontend (scene_count, mean_cloud, etc.) even when the model run
    itself fails.
    """
    items = search_scenes(bbox, datetime_start, datetime_end, max_cloud)
    if not items:
        raise ValueError(
            f"No Sentinel-2 scenes found for bbox={bbox} "
            f"in [{datetime_start}, {datetime_end}] with cloud < {max_cloud}%."
        )

    pixel_df = build_scene_pixel_dataframe(items, bbox, site_name, settings.pixel_grid)
    if pixel_df.empty:
        raise ValueError("All STAC scenes failed to load — none yielded usable pixels.")

    pixel_df["pixel_lat"] = pixel_df["latitude"].round(4)
    pixel_df["pixel_lon"] = pixel_df["longitude"].round(4)
    pixel_df["pixel_id"] = (
        pixel_df["pixel_lat"].astype(str)
        + "_"
        + pixel_df["pixel_lon"].astype(str)
        + "_"
        + pixel_df["SITE"].astype(str)
    )

    features = compute_pixel_window_features(pixel_df)

    meta = {
        "scene_count": int(len(items)),
        "mean_cloud_cover": float(pixel_df["CLOUD_PCT"].mean()),
        "date_min": pixel_df["DATE"].min().isoformat(),
        "date_max": pixel_df["DATE"].max().isoformat(),
        "pixel_count": int(pixel_df["pixel_id"].nunique()),
        "window_count": int(features["window_center"].nunique()) if not features.empty else 0,
    }
    return features, meta
