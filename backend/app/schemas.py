from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional

class HotspotProperties(BaseModel):
    id: str
    name: str
    region: str
    country: str
    risk_level: Literal["low", "moderate", "high", "critical"]
    eelgrass_pct_estimate: float
    depletion_risk_90d: float
    confidence: float
    summary: str
    reports: List[Dict[str, str]] = []
    drivers: List[str] = []
    last_updated: str

class HotspotFeature(BaseModel):
    type: Literal["Feature"] = "Feature"
    geometry: Dict[str, Any]
    properties: HotspotProperties

class HotspotCollection(BaseModel):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: List[HotspotFeature]

class InferenceRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=120)
    bbox: List[float] = Field(..., min_length=4, max_length=4)
    datetime_start: str
    datetime_end: str
    max_cloud_cover: float = 20.0

class InferenceStartResponse(BaseModel):
    job_id: str
    status: str

class JobProgress(BaseModel):
    job_id: str
    stage: str
    progress: int
    message: str
    fact: Optional[str] = None

class InferenceResult(BaseModel):
    job_id: str
    area_name: str
    bbox: List[float]
    eelgrass_pct_estimate: float
    depletion_risk_90d: float
    confidence: float
    top_drivers: List[str]
    scene_count: int
    source: str
    status: str = "completed"