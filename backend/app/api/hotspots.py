from fastapi import APIRouter
from app.services.hotspot_repo import load_demo_hotspots

router = APIRouter(prefix="/api/hotspots", tags=["hotspots"])

@router.get("")
async def get_hotspots():
    print("ENTER get_hotspots", flush=True)
    data = load_demo_hotspots()
    print("LEAVE get_hotspots", flush=True)
    return data