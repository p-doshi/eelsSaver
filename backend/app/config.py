from pydantic import BaseModel
import os

class Settings(BaseModel):
    app_name: str = os.getenv("APP_NAME", "Eelgrass Watch")
    app_env: str = os.getenv("APP_ENV", "dev")
    host: str = os.getenv("APP_HOST", "0.0.0.0")
    port: int = int(os.getenv("APP_PORT", "8000"))
    stac_url: str = os.getenv("STAC_URL", "https://stac.dataspace.copernicus.eu/v1/search")
    stac_collection: str = os.getenv("STAC_COLLECTION", "sentinel-2-l2a")
    model_path: str = os.getenv("MODEL_PATH", "backend/models/gbm_model.pkl")
    model_meta_path: str = os.getenv("MODEL_META_PATH", "backend/models/model_meta.json")
    frontend_origin: str = os.getenv("FRONTEND_ORIGIN", "*")
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "900"))

settings = Settings()