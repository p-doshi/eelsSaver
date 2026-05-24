"""
pipeline.s2_features
--------------------
Reads a GEE-exported per-pixel-per-image CSV and emits a tabular
feature matrix of pixel × 90-day-window rolling statistics
(~140 columns, the student model's training inputs).

Usage:
    python -m pipeline.s2_features <gee_csv> <output_csv>

If args are omitted, defaults from config are used (Forillon training set).
"""

from __future__ import annotations
import sys
import numpy as np
import pandas as pd

from config import (
    WINDOW_DAYS, STEP_DAYS, MIN_OBS, SPECTRAL_COLS,
    FORILLON_RAW_CSV, FORILLON_FEATURES,
)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_gee_export(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df['DATE'] = pd.to_datetime(df['DATE'])

    df['pixel_lat'] = df['latitude'].round(4)
    df['pixel_lon'] = df['longitude'].round(4)
    df['pixel_id'] = (df['pixel_lat'].astype(str) + '_'
                       + df['pixel_lon'].astype(str) + '_'
                       + df['SITE'].astype(str))
    return df


# ---------------------------------------------------------------------------
# Temporal feature engineering
# ---------------------------------------------------------------------------

def _trend(values: np.ndarray) -> float:
    if len(values) < 3:
        return 0.0
    t = np.arange(len(values), dtype=float)
    return float(np.polyfit(t, values, 1)[0])


def compute_pixel_window_features(
    df: pd.DataFrame,
    window_days: int = WINDOW_DAYS,
    step_days: int = STEP_DAYS,
    min_obs: int = MIN_OBS,
) -> pd.DataFrame:
    rows = []

    global_start = df['DATE'].min() + pd.Timedelta(days=window_days)
    global_end   = df['DATE'].max()
    window_centers = pd.date_range(global_start, global_end, freq=f'{step_days}D')

    grouped = df.groupby('pixel_id', sort=False)
    n_pixels = df['pixel_id'].nunique()

    for k, (pixel_id, pxdf) in enumerate(grouped):
        if k % 500 == 0:
            print(f'  pixel {k+1}/{n_pixels}', end='\r', flush=True)

        pxdf = pxdf.sort_values('DATE')
        meta = {
            'pixel_id':  pixel_id,
            'SITE':      pxdf['SITE'].iloc[0],
            'latitude':  pxdf['pixel_lat'].iloc[0],
            'longitude': pxdf['pixel_lon'].iloc[0],
        }

        for wc in window_centers:
            w0 = wc - pd.Timedelta(days=window_days)
            window_df = pxdf[(pxdf['DATE'] >= w0) & (pxdf['DATE'] <= wc)]
            n = len(window_df)
            if n < min_obs:
                continue

            doy = (w0 + pd.Timedelta(days=window_days // 2)).dayofyear
            row = {
                **meta,
                'window_center':   wc,
                'valid_obs_count': n,
                'sin_doy':         np.sin(2 * np.pi * doy / 365),
                'cos_doy':         np.cos(2 * np.pi * doy / 365),
            }

            for col in SPECTRAL_COLS:
                if col not in window_df.columns:
                    continue
                vals = window_df[col].dropna().values
                m = len(vals)
                if m == 0:
                    continue
                row[f'{col}_mean']    = float(np.mean(vals))
                row[f'{col}_median']  = float(np.median(vals))
                row[f'{col}_std']     = float(np.std(vals)) if m > 1 else 0.0
                row[f'{col}_min']     = float(np.min(vals))
                row[f'{col}_max']     = float(np.max(vals))
                row[f'{col}_trend']   = _trend(vals)
                row[f'{col}_n_valid'] = m

            rows.append(row)

    print()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    gee_csv  = sys.argv[1] if len(sys.argv) > 1 else str(FORILLON_RAW_CSV)
    out_csv  = sys.argv[2] if len(sys.argv) > 2 else str(FORILLON_FEATURES)

    print(f'Loading {gee_csv}')
    df = load_gee_export(gee_csv)
    print(f'  rows={len(df):,}  pixels={df["pixel_id"].nunique():,}  '
          f'dates={df["DATE"].min().date()}→{df["DATE"].max().date()}')

    print(f'Building {WINDOW_DAYS}-d rolling features (step={STEP_DAYS}d)')
    feats = compute_pixel_window_features(df)
    print(f'  output rows={len(feats):,}  columns={len(feats.columns)}')

    feats.to_csv(out_csv, index=False)
    print(f'  → {out_csv}')
