from pathlib import Path
from pydantic import BaseModel
import os

# Repo root is two levels up from this file: backend/app/config.py → backend/ → repo/
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODELS_DIR = _REPO_ROOT / "backend-dev" / "data" / "models"

class Settings(BaseModel):
    app_name: str = os.getenv("APP_NAME", "Eelgrass Watch")
    app_env: str = os.getenv("APP_ENV", "dev")
    host: str = os.getenv("APP_HOST", "0.0.0.0")
    port: int = int(os.getenv("APP_PORT", "8000"))

    # Live data source: Google Earth Engine (COPERNICUS/S2_SR_HARMONIZED).
    # GEE_PROJECT must be set — without it ee.Initialize() raises.
    # Authenticate once on the host: `earthengine authenticate --project=<id>`
    gee_project: str = os.getenv("GEE_PROJECT", "northern-hope-497221-m7")
    gee_collection: str = os.getenv("GEE_COLLECTION", "COPERNICUS/S2_SR_HARMONIZED")

    # Trained XGBoost artifacts produced by backend-dev/pipeline/student.py.
    # MODELS_DIR should contain gpi_model.pkl, decline_model.pkl, stress_model.pkl,
    # feature_cols.pkl, train_X_mean.pkl, train_cov_inv.pkl.
    models_dir: str = os.getenv("MODELS_DIR", str(_DEFAULT_MODELS_DIR))

    # Pixel sampling grid edge length per S2 scene over the requested bbox.
    # PIXEL_GRID=16 → 256 sampled pixels/scene; bump for higher spatial detail.
    pixel_grid: int = int(os.getenv("PIXEL_GRID", "16"))
    max_scenes: int = int(os.getenv("MAX_SCENES", "40"))

    frontend_origin: str = os.getenv("FRONTEND_ORIGIN", "*")
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "900"))

settings = Settings()