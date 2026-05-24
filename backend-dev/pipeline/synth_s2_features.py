"""
pipeline.synth_s2_features
--------------------------
Generates *synthetic* Sentinel-2 pixel-window features for Forillon as a
stand-in for the real GEE export, so the student can be trained end-to-end
without a working Earth Engine account.

Each (SITE, window) from teacher_labels.csv is expanded into ~N pixels at
randomized lat/lon. Spectral features are drawn from distributions whose
mean/trend depend on that window's GPI_90d, decline_prob_180d, and
dominant stress — giving the student a real (but noisy) signal to learn.

NOT for production. Replace with the real GEE export when available.

Usage:
    python -m pipeline.synth_s2_features
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from config import (
    TEACHER_LABELS_CSV, FORILLON_FEATURES,
    SPECTRAL_COLS, FORILLON_SITES,
)

# Pixels per site (kept small for runtime)
PIXELS_PER_SITE = 40
RNG = np.random.default_rng(2026)

# Site bounding boxes (lon_min, lon_max, lat_min, lat_max) — match regions.json
SITE_BBOX = {
    'Ouest':  (-64.458, -64.448, 48.792, 48.808),
    'Marais': (-64.438, -64.428, 48.798, 48.814),
    'Sud':    (-64.443, -64.433, 48.778, 48.794),
}

# Baseline spectral values for a healthy eelgrass pixel (rough literature priors)
BASE = {
    'B2': 0.06, 'B3': 0.10, 'B4': 0.04, 'B5': 0.05, 'B6': 0.07,
    'B8': 0.08, 'B8A': 0.08, 'B11': 0.02, 'B12': 0.01,
}


def _pixel_locations(site: str, n: int) -> list[tuple[float, float]]:
    lon_min, lon_max, lat_min, lat_max = SITE_BBOX[site]
    lons = RNG.uniform(lon_min, lon_max, n).round(4)
    lats = RNG.uniform(lat_min, lat_max, n).round(4)
    return list(zip(lons, lats))


def _per_window_spectrals(gpi: float, decline: float, stress: str) -> dict:
    """Generate one pixel's mean spectral values for a window."""
    health = gpi  # 0.05 - 1.0

    # Greenness scales with health
    b3 = BASE['B3'] * (0.6 + 0.8 * health)         # high health = greener
    b2 = BASE['B2'] * (1.0 + 0.3 * (1 - health))   # blue increases over bare sediment
    b4 = BASE['B4'] * (1.0 - 0.4 * health)         # red absorbed more when vegetation thick

    # Red edge increases with biomass
    b5 = BASE['B5'] * (0.8 + 0.4 * health)
    b6 = BASE['B6'] * (0.7 + 0.6 * health)
    b8 = BASE['B8'] * (0.7 + 0.6 * health)
    b8a = BASE['B8A'] * (0.7 + 0.6 * health)

    # SWIR not affected much underwater, but reflects nearby exposed sediment when bed thins
    b11 = BASE['B11'] * (1.0 + 0.5 * (1 - health))
    b12 = BASE['B12'] * (1.0 + 0.5 * (1 - health))

    # Stress modulations
    if stress == 'heat':
        b3 *= 0.85; b6 *= 0.85          # thinning canopy
    elif stress == 'cold':
        b11 *= 1.3; b12 *= 1.3          # exposed sediment
    elif stress == 'light':
        b4 /= max(0.5, 1 - 0.5 * decline)  # higher red = more turbidity
        b3 *= 0.95
    elif stress == 'senescence':
        b6 *= 0.9; b5 *= 0.9            # red-edge attenuation

    eps = 1e-6
    return {
        'B2': b2, 'B3': b3, 'B4': b4, 'B5': b5, 'B6': b6,
        'B8': b8, 'B8A': b8a, 'B11': b11, 'B12': b12,
        'NDAVI':           (b8 - b4) / (b8 + b4 + eps),
        'WAVI':            1.5 * (b8 - b4) / (b8 + b4 + 0.5),
        'GB_ratio':        b3 / (b2 + eps),
        'RG_ratio':        b4 / (b3 + eps),
        'B3B2_diff':       b3 - b2,
        'NDWI':            (b3 - b8) / (b3 + b8 + eps),
        'turbidity':       b4 / (b3 + eps),
        'red_edge_slope':  (b6 - b5) / 35,
        'SABI':            (b8 - b4) / (b3 + b2 + eps),
        'depth_invariant': np.log(max(b2, eps)) / max(np.log(max(b3, eps)), eps),
    }


def _temporal_stats(mu: float, decline: float, stress: str) -> dict:
    """Generate the 7 temporal statistics for one spectral feature."""
    # Std scales with decline (declining beds = unstable signal)
    std  = max(0.001, abs(mu) * (0.05 + 0.20 * decline))
    mean = mu + RNG.normal(0, std * 0.2)
    median = mean + RNG.normal(0, std * 0.1)
    lo = mean - 2 * std + RNG.normal(0, std * 0.1)
    hi = mean + 2 * std + RNG.normal(0, std * 0.1)
    # Trend: declining beds show downward trend in greenness features
    trend = -decline * abs(mu) * 0.01 + RNG.normal(0, 0.0005)
    if stress == 'heat':
        trend -= 0.0005
    n_valid = int(RNG.integers(3, 8))
    return dict(mean=mean, median=median, std=std, min=lo, max=hi,
                trend=trend, n_valid=n_valid)


def generate(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    stress_cols = ['frac_heat', 'frac_cold', 'frac_light', 'frac_senescence']
    stress_names = ['heat', 'cold', 'light', 'senescence']

    for site in FORILLON_SITES:
        sub = labels[labels['SITE'] == site].sort_values('window_center')
        if sub.empty:
            continue

        pixels = _pixel_locations(site, PIXELS_PER_SITE)
        print(f'  {site}: {len(sub)} windows × {len(pixels)} pixels = '
              f'{len(sub) * len(pixels):,} rows')

        for _, lbl in sub.iterrows():
            gpi   = float(lbl['GPI_90d'])
            decl  = float(lbl.get('decline_prob_180d', 0.0) or 0.0)
            frac  = lbl[stress_cols].astype(float).values
            stress = stress_names[int(np.argmax(frac))]

            for (lon, lat) in pixels:
                # Per-pixel jitter: each pixel sees the same window stats with noise
                pj = _per_window_spectrals(
                    gpi * (1 + RNG.normal(0, 0.05)),
                    decl * (1 + RNG.normal(0, 0.05)),
                    stress,
                )
                doy = pd.Timestamp(lbl['window_center']).dayofyear

                row = {
                    'pixel_id':        f'{lat}_{lon}_{site}',
                    'SITE':            site,
                    'latitude':        lat,
                    'longitude':       lon,
                    'window_center':   lbl['window_center'],
                    'valid_obs_count': int(RNG.integers(3, 8)),
                    'sin_doy':         np.sin(2 * np.pi * doy / 365),
                    'cos_doy':         np.cos(2 * np.pi * doy / 365),
                }
                for col in SPECTRAL_COLS:
                    stats = _temporal_stats(pj[col], decl, stress)
                    for stat_name, val in stats.items():
                        row[f'{col}_{stat_name}'] = float(val)
                    row[f'{col}_n_valid'] = stats['n_valid']
                rows.append(row)

    return pd.DataFrame(rows)


if __name__ == '__main__':
    print(f'Loading teacher labels: {TEACHER_LABELS_CSV}')
    lbl = pd.read_csv(TEACHER_LABELS_CSV)
    lbl['window_center'] = pd.to_datetime(lbl['window_center'])
    print(f'  rows={len(lbl)}  sites={sorted(lbl["SITE"].unique())}')

    print('\nGenerating synthetic Sentinel-2 features …')
    feats = generate(lbl)
    print(f'\n  total rows={len(feats):,}  columns={len(feats.columns)}')

    feats.to_csv(FORILLON_FEATURES, index=False)
    print(f'  → {FORILLON_FEATURES}')
