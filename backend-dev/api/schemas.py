"""Pydantic response models for the eelsSaver API."""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field


class StatusResponse(BaseModel):
    last_refresh:   Optional[datetime]
    model_version:  Optional[str]
    regions_loaded: list[str]
    bed_count:      int


class RegionSummary(BaseModel):
    id:    str
    name:  str
    role:  str
    bed_count: int
    high_risk_beds:   int = 0
    medium_risk_beds: int = 0
    low_risk_beds:    int = 0


class BedSummary(BaseModel):
    bed_id:          str
    region_id:       str
    site:            str
    latest_window:   Optional[datetime]
    bed_GPI:         Optional[float]
    bed_decline_prob: Optional[float]
    high_risk_pixel_frac: Optional[float]
    dominant_stress: Optional[str]
    bed_risk_tier:   Optional[str]
    bed_confidence:  Optional[str]
    n_pixels:        Optional[int]


class BedExplanation(BaseModel):
    feature:           str
    mean_abs_shap:     float
    ecological_phrase: str


class BedDetail(BedSummary):
    polygon:        list[list[float]] = Field(default_factory=list)
    stress_phrase:  Optional[str]
    explanations:   list[BedExplanation] = Field(default_factory=list)


class PixelScore(BaseModel):
    pixel_id:               str
    latitude:               float
    longitude:              float
    predicted_GPI:          float
    predicted_decline_prob: float
    predicted_stress:       str
    risk_tier:              str
    confidence:             str
    spectral_pval:          Optional[float]


class TimeseriesPoint(BaseModel):
    window_center:    datetime
    bed_GPI:          float
    bed_decline_prob: float
    bed_risk_tier:    str


class RefreshRequest(BaseModel):
    region_id:    str
    features_csv: Optional[str] = None
