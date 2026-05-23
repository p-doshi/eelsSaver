from fastapi import APIRouter, Query
import httpx

router = APIRouter(prefix="/api/photos", tags=["photos"])

@router.get("")
async def get_photos(
    lat: float = Query(...),
    lon: float = Query(...),
    radius: int = Query(10000),
    limit: int = Query(4)
):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "origin": "*",
        "generator": "geosearch",
        "ggscoord": f"{lat}|{lon}",
        "ggsradius": radius,
        "ggslimit": limit,
        "prop": "pageimages|coordinates",
        "piprop": "thumbnail",
        "pithumbsize": 600,
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

    pages = data.get("query", {}).get("pages", {})
    photos = []

    for page_id, page in pages.items():
        thumb = page.get("thumbnail", {})
        source = thumb.get("source")
        title = page.get("title", "Unknown place")

        if source:
            photos.append({
                "title": title,
                "url": source,
                "page_url": f"https://en.wikipedia.org/?curid={page_id}"
            })

    return {"photos": photos}