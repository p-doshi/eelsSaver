import httpx
import hashlib
import json
import time
from app.config import settings
from app.state import stac_cache

def _cache_key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

async def search_sentinel2(bbox, datetime_start, datetime_end, max_cloud_cover=20.0, limit=8):
    payload = {
        "collections": [settings.stac_collection],
        "bbox": bbox,
        "datetime": f"{datetime_start}/{datetime_end}",
        "limit": limit,
        "query": {
            "eo:cloud_cover": {"lt": max_cloud_cover}
        }
    }
    key = _cache_key(payload)
    cached = stac_cache.get(key)
    now = time.time()
    if cached and now - cached["ts"] < settings.cache_ttl_seconds:
        return cached["data"]

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(settings.stac_url, json=payload)
        resp.raise_for_status()
        data = resp.json()

    stac_cache[key] = {"ts": now, "data": data}
    return data