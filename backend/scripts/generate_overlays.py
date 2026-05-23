import json
import math
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from rasterio.enums import Resampling

import planetary_computer
from pystac_client import Client

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "static" / "overlays"
CFG_PATH = ROOT / "scripts" / "hotspots.json"

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def percentile_stretch(arr, p2=2, p98=98):
    arr = arr.astype(np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo = np.percentile(finite, p2)
    hi = np.percentile(finite, p98)
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    arr[~np.isfinite(arr)] = 0
    return arr


def gamma(arr, g=1.0):
    arr = np.clip(arr, 0, 1)
    return np.power(arr, 1.0 / g)


def rgb_to_uint8(r, g, b):
    rgb = np.dstack([r, g, b])
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    return rgb


def ndvi(b8, b4):
    denom = b8 + b4
    denom = np.where(denom == 0, np.nan, denom)
    return (b8 - b4) / denom


def normalize01(x, lo=None, hi=None):
    x = x.astype(np.float32)
    if lo is None:
        lo = np.nanpercentile(x, 2)
    if hi is None:
        hi = np.nanpercentile(x, 98)
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((x - lo) / (hi - lo), 0, 1)


def colorize_stress_layer(stress01):
    h, w = stress01.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # teal -> yellow -> red ramp
    r = np.clip(255 * np.power(stress01, 1.2), 0, 255)
    g = np.clip(255 * (1 - np.abs(stress01 - 0.5) * 1.4), 0, 255)
    b = np.clip(255 * (1 - stress01) * 0.9, 0, 255)

    rgba[..., 0] = r.astype(np.uint8)
    rgba[..., 1] = g.astype(np.uint8)
    rgba[..., 2] = b.astype(np.uint8)
    rgba[..., 3] = (stress01 * 170).astype(np.uint8)
    return rgba


def colorize_risk_layer(risk01):
    h, w = risk01.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # transparent low-risk -> orange/red high-risk
    rgba[..., 0] = np.clip(255 * (0.9 * risk01 + 0.2), 0, 255).astype(np.uint8)
    rgba[..., 1] = np.clip(180 * (1 - risk01), 0, 255).astype(np.uint8)
    rgba[..., 2] = np.clip(80 * (1 - risk01), 0, 255).astype(np.uint8)
    rgba[..., 3] = (risk01 * 190).astype(np.uint8)
    return rgba


def save_rgb(path, arr):
    Image.fromarray(arr).save(path, quality=92)


def save_rgba(path, arr):
    Image.fromarray(arr, mode="RGBA").save(path)


def pick_best_item(items):
    def score(item):
        cc = item.properties.get("eo:cloud_cover", 100)
        dt = item.properties.get("datetime", "")
        return (cc, dt)
    return sorted(items, key=score)[0]



def read_band(item, asset_key, bbox_wgs84, max_dim=1024, out_shape=None):
    signed = planetary_computer.sign(item)
    href = signed.assets[asset_key].href

    with rasterio.open(href) as src:
        left, bottom, right, top = transform_bounds(
            "EPSG:4326",
            src.crs,
            bbox_wgs84[0], bbox_wgs84[1], bbox_wgs84[2], bbox_wgs84[3],
            densify_pts=21
        )

        left = max(left, src.bounds.left)
        right = min(right, src.bounds.right)
        bottom = max(bottom, src.bounds.bottom)
        top = min(top, src.bounds.top)

        if not (left < right and bottom < top):
            raise RuntimeError(f"Invalid transformed bounds for {asset_key}")

        window = from_bounds(left, bottom, right, top, transform=src.transform)
        window = window.round_offsets().round_lengths()

        h = max(1, int(window.height))
        w = max(1, int(window.width))

        if out_shape is None:
            scale = max(h / max_dim, w / max_dim, 1.0)
            out_h = max(1, int(h / scale))
            out_w = max(1, int(w / scale))
            out_shape = (out_h, out_w)

        arr = src.read(
            1,
            window=window,
            out_shape=out_shape,
            resampling=Resampling.bilinear,
            boundless=False
        )

    return arr.astype(np.float32)

def assert_valid(arr, name):
    if arr.size == 0:
        raise RuntimeError(f"{name} is empty")
    if np.all(~np.isfinite(arr)):
        raise RuntimeError(f"{name} contains no finite values")
    
def process_hotspot(h):
    bbox = h["bbox"]
    hotspot_id = h["id"]
    date_range = f'{h["date_start"]}/{h["date_end"]}'
    max_cloud = h.get("max_cloud", 20)

    catalog = Client.open(STAC_URL)
    search = catalog.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    items = list(search.items())
    if not items:
        raise RuntimeError(f"No scenes found for {hotspot_id}")

    item = pick_best_item(items)
    print(f"[{hotspot_id}] using item:", item.id)

    b4 = read_band(item, "B04", bbox, max_dim=1024)
    out_shape = b4.shape
    b3 = read_band(item, "B03", bbox, out_shape=out_shape)
    b2 = read_band(item, "B02", bbox, out_shape=out_shape)
    b8 = read_band(item, "B08", bbox, out_shape=out_shape)
    assert_valid(b4, "B04")
    assert_valid(b3, "B03")
    assert_valid(b2, "B02")
    assert_valid(b8, "B08")
    # Scale reflectance if needed
    scale = 10000.0
    b2 /= scale
    b3 /= scale
    b4 /= scale
    b8 /= scale

    # True color
    r = gamma(percentile_stretch(b4), 1.15)
    g = gamma(percentile_stretch(b3), 1.15)
    b = gamma(percentile_stretch(b2), 1.15)
    rgb = rgb_to_uint8(r, g, b)
    save_rgb(OUT_DIR / f"{hotspot_id}-truecolor.webp", rgb)

    # NDVI-based stress proxy
    ndvi_arr = ndvi(b8, b4)
    # lower NDVI => higher stress
    stress01 = 1.0 - normalize01(ndvi_arr, lo=0.1, hi=0.7)
    stress_rgba = colorize_stress_layer(stress01)
    save_rgba(OUT_DIR / f"{hotspot_id}-stress.png", stress_rgba)

    # Simple risk proxy: combines low NDVI + brightness anomaly
    brightness = normalize01((b2 + b3 + b4) / 3.0)
    risk01 = np.clip(0.7 * stress01 + 0.3 * brightness, 0, 1)
    risk_rgba = colorize_risk_layer(risk01)
    save_rgba(OUT_DIR / f"{hotspot_id}-risk.png", risk_rgba)

    meta = {
        "id": hotspot_id,
        "name": h["name"],
        "source_item_id": item.id,
        "datetime": item.properties.get("datetime"),
        "eo_cloud_cover": item.properties.get("eo:cloud_cover"),
        "bbox": bbox,
        "outputs": {
            "preview_url": f"/static/overlays/{hotspot_id}-truecolor.webp",
            "stress_overlay_url": f"/static/overlays/{hotspot_id}-stress.png",
            "risk_overlay_url": f"/static/overlays/{hotspot_id}-risk.png"
        }
    }

    with open(OUT_DIR / f"{hotspot_id}-meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main():
    hotspots = json.loads(CFG_PATH.read_text())
    for h in hotspots:
        try:
            process_hotspot(h)
        except Exception as e:
            print(f"Failed for {h['id']}: {e}")


if __name__ == "__main__":
    main()