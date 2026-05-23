from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.api.health import router as health_router
from app.api.hotspots import router as hotspots_router
from app.api.inference import router as inference_router
from app.api.photos import router as photos_router

BASE_DIR = Path(__file__).resolve().parent.parent   # backend/
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.frontend_origin == "*" else [settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(health_router)
app.include_router(hotspots_router)
app.include_router(inference_router)
app.include_router(photos_router)