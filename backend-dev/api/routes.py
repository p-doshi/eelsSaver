"""
API routes — read predictions from data/predictions/ and SHAP from data/models/
and surface them through versioned JSON endpoints.

All file reads are best-effort: missing files return empty lists / 404 rather
than raising. This lets the frontend render even when only some regions have
been processed.
"""

from __future__ import annotations
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from config import (
    PREDICTIONS_DIR, MODELS_DIR, REGIONS_FILE,
    STRESS_LANGUAGE, SHAP_LANGUAGE, BASE_DIR,
)
from api.schemas import (
    StatusResponse, RegionSummary, BedSummary, BedDetail,
    BedExplanation, PixelScore, TimeseriesPoint, RefreshRequest,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers — cached file reads
# ---------------------------------------------------------------------------

def _load_regions() -> list[dict]:
    if not REGIONS_FILE.exists():
        return []
    return json.loads(REGIONS_FILE.read_text())['regions']


def _bed_lookup() -> dict[str, dict]:
    """bed_id → {region_id, name, polygon, site}"""
    out = {}
    for r in _load_regions():
        for b in r['beds']:
            out[b['id']] = {
                'region_id': r['id'],
                'region_name': r['name'],
                'site':    b['site'],
                'polygon': b['polygon'],
            }
    return out


def _load_beds(region_id: Optional[str] = None) -> pd.DataFrame:
    frames = []
    for f in PREDICTIONS_DIR.glob('*_beds.csv'):
        if region_id and not f.name.startswith(f'{region_id}_'):
            continue
        try:
            df = pd.read_csv(f)
            df['window_center'] = pd.to_datetime(df['window_center'])
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_pixels(bed_id: str) -> pd.DataFrame:
    lookup = _bed_lookup().get(bed_id)
    if lookup is None:
        return pd.DataFrame()
    f = PREDICTIONS_DIR / f'{lookup["region_id"]}_pixels.csv'
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
    df['window_center'] = pd.to_datetime(df['window_center'])
    return df[df['SITE'] == lookup['site']]


def _shap_phrase(feature: str) -> str:
    """Translate an XGBoost feature name into ecological language."""
    for stem, phrase in SHAP_LANGUAGE.items():
        if stem in feature:
            stat = feature.split('_')[-1]
            return f'{phrase} ({stat})'
    return feature.replace('_', ' ')


def _load_shap(tag: str, top_n: int = 6) -> pd.DataFrame:
    f = MODELS_DIR / f'shap_{tag}.csv'
    if not f.exists():
        return pd.DataFrame()
    return pd.read_csv(f).head(top_n)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@router.get('/status', response_model=StatusResponse)
def status():
    files = list(PREDICTIONS_DIR.glob('*_beds.csv'))
    latest = max((f.stat().st_mtime for f in files), default=None)
    regions_loaded = sorted({f.name.replace('_beds.csv', '') for f in files})

    beds_df = _load_beds()
    return StatusResponse(
        last_refresh   = datetime.fromtimestamp(latest) if latest else None,
        model_version  = _model_version(),
        regions_loaded = regions_loaded,
        bed_count      = int(beds_df['bed_id'].nunique()) if len(beds_df) else 0,
    )


def _model_version() -> Optional[str]:
    f = MODELS_DIR / 'validation_report.json'
    if not f.exists():
        return None
    ts = datetime.fromtimestamp(f.stat().st_mtime)
    return ts.strftime('v%Y%m%d-%H%M')


# ---------------------------------------------------------------------------
# Regions & beds
# ---------------------------------------------------------------------------

@router.get('/regions', response_model=list[RegionSummary])
def list_regions():
    regions = _load_regions()
    beds_df = _load_beds()
    if len(beds_df):
        latest = (beds_df.sort_values('window_center')
                          .groupby('bed_id').tail(1))
    else:
        latest = pd.DataFrame(columns=['region_id', 'bed_risk_tier'])

    out = []
    for r in regions:
        sub = latest[latest['region_id'] == r['id']]
        out.append(RegionSummary(
            id   = r['id'],
            name = r['name'],
            role = r['role'],
            bed_count        = len(r['beds']),
            high_risk_beds   = int((sub['bed_risk_tier'] == 'High').sum()),
            medium_risk_beds = int((sub['bed_risk_tier'] == 'Medium').sum()),
            low_risk_beds    = int((sub['bed_risk_tier'] == 'Low').sum()),
        ))
    return out


@router.get('/beds', response_model=list[BedSummary])
def list_beds(
    region_id: Optional[str] = Query(None),
    risk_tier: Optional[str] = Query(None),
):
    beds_df = _load_beds(region_id)
    if not len(beds_df):
        return []

    latest = (beds_df.sort_values('window_center')
                      .groupby('bed_id').tail(1))
    if risk_tier:
        latest = latest[latest['bed_risk_tier'] == risk_tier]

    out = []
    for _, row in latest.iterrows():
        out.append(BedSummary(
            bed_id   = row['bed_id'],
            region_id = row['region_id'],
            site      = row['site'],
            latest_window         = row['window_center'],
            bed_GPI               = float(row.get('bed_GPI', 0)),
            bed_decline_prob      = float(row.get('bed_decline_prob', 0)),
            high_risk_pixel_frac  = float(row.get('high_risk_pixel_frac', 0)),
            dominant_stress       = row.get('dominant_stress'),
            bed_risk_tier         = row.get('bed_risk_tier'),
            bed_confidence        = row.get('bed_confidence'),
            n_pixels              = int(row.get('n_pixels', 0)),
        ))
    return out


@router.get('/beds/{bed_id}', response_model=BedDetail)
def get_bed(bed_id: str):
    lookup = _bed_lookup().get(bed_id)
    if not lookup:
        raise HTTPException(404, f'Unknown bed_id: {bed_id}')

    beds_df = _load_beds(lookup['region_id'])
    bed = beds_df[beds_df['bed_id'] == bed_id].sort_values('window_center')
    if not len(bed):
        raise HTTPException(404, f'No predictions for bed_id: {bed_id}')

    last = bed.iloc[-1]
    dom  = last.get('dominant_stress')

    shap_decline = _load_shap('decline')
    explanations = [
        BedExplanation(
            feature           = row['feature'],
            mean_abs_shap     = float(row['mean_abs_shap']),
            ecological_phrase = _shap_phrase(row['feature']),
        )
        for _, row in shap_decline.iterrows()
    ]

    return BedDetail(
        bed_id      = bed_id,
        region_id   = lookup['region_id'],
        site        = lookup['site'],
        latest_window         = last['window_center'],
        bed_GPI               = float(last['bed_GPI']),
        bed_decline_prob      = float(last['bed_decline_prob']),
        high_risk_pixel_frac  = float(last['high_risk_pixel_frac']),
        dominant_stress       = dom,
        bed_risk_tier         = last['bed_risk_tier'],
        bed_confidence        = last['bed_confidence'],
        n_pixels              = int(last['n_pixels']),
        polygon               = lookup['polygon'],
        stress_phrase         = STRESS_LANGUAGE.get(dom, ''),
        explanations          = explanations,
    )


@router.get('/beds/{bed_id}/pixels', response_model=list[PixelScore])
def get_bed_pixels(bed_id: str, window: Optional[str] = Query(None)):
    pixels = _load_pixels(bed_id)
    if not len(pixels):
        return []

    if window:
        try:
            target = pd.to_datetime(window)
            pixels = pixels[pixels['window_center'] == target]
        except Exception:
            raise HTTPException(400, f'Invalid window date: {window}')
    else:
        latest_wc = pixels['window_center'].max()
        pixels = pixels[pixels['window_center'] == latest_wc]

    return [
        PixelScore(
            pixel_id  = row['pixel_id'],
            latitude  = float(row['latitude']),
            longitude = float(row['longitude']),
            predicted_GPI          = float(row['predicted_GPI']),
            predicted_decline_prob = float(row['predicted_decline_prob']),
            predicted_stress       = row['predicted_stress'],
            risk_tier              = row['risk_tier'],
            confidence             = row['confidence'],
            spectral_pval          = float(row['spectral_pval']) if pd.notna(row['spectral_pval']) else None,
        )
        for _, row in pixels.iterrows()
    ]


@router.get('/beds/{bed_id}/timeseries', response_model=list[TimeseriesPoint])
def get_bed_timeseries(bed_id: str):
    lookup = _bed_lookup().get(bed_id)
    if not lookup:
        raise HTTPException(404, f'Unknown bed_id: {bed_id}')

    beds_df = _load_beds(lookup['region_id'])
    sub = beds_df[beds_df['bed_id'] == bed_id].sort_values('window_center')
    return [
        TimeseriesPoint(
            window_center    = row['window_center'],
            bed_GPI          = float(row['bed_GPI']),
            bed_decline_prob = float(row['bed_decline_prob']),
            bed_risk_tier    = row['bed_risk_tier'],
        )
        for _, row in sub.iterrows()
    ]


# ---------------------------------------------------------------------------
# Refresh trigger (admin)
# ---------------------------------------------------------------------------

@router.post('/refresh')
def trigger_refresh(req: RefreshRequest):
    """Re-run inference for a region from an existing features CSV."""
    if req.features_csv is None:
        raise HTTPException(400, 'features_csv is required (path to s2_pixel_features.csv)')

    features = Path(req.features_csv)
    if not features.exists():
        raise HTTPException(404, f'features_csv not found: {features}')

    cmd = ['python', '-m', 'pipeline.inference', str(features), req.region_id]
    proc = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True)
    if proc.returncode != 0:
        raise HTTPException(500, f'Inference failed:\n{proc.stderr}')

    return {'ok': True, 'region_id': req.region_id, 'log': proc.stdout[-2000:]}
