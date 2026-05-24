"""
pipeline.derive_ouest_annual
----------------------------
Derives an annual Ouest survey table from the existing daily aligned dataset
(aligned_eelgrass_env_data.csv) to match the schema expected by
pipeline.teacher_ouest, without needing the external NeuroHack 2026 CSV.

Output columns:
    site_id, year, survey_date,
    cover, cover_std, cover_min, cover_max, height_proxy, damage_proxy,
    For each of 5 phenological windows (pre_growing, early_growing,
                                          peak_growing, late_growing, winter):
        sst_mean, sst_max, sst_min, sst_std,
        par_mean, par_max, par_min, par_std,
        gpi_mean, days_above_22, days_below_8
    → 5 windows × 11 stats = 55 env columns

Total: ~63 columns × 6 rows (one per year, Ouest site).

Usage:
    python -m pipeline.derive_ouest_annual
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from config import ALIGNED_EELGRASS_CSV, RAW_DIR

OUT_FILE = RAW_DIR / 'ouest_eelgrass_environmental_complete.csv'

# Phenological windows (month ranges) for Zostera marina at 48 °N
WINDOWS = {
    'pre_growing':   (4, 5),    # Apr-May
    'early_growing': (6, 6),    # Jun
    'peak_growing':  (7, 8),    # Jul-Aug
    'late_growing':  (9, 10),   # Sep-Oct
    'winter':        (11, 3),   # Nov-Mar (wrap)
}

SST_OPTIMAL = 15.0
SST_WIDTH = 10.0
PAR_SAT = 30.0
HEAT_THR = 22.0
COLD_THR = 8.0


def _gpi(sst, par):
    f_sst = np.exp(-0.5 * ((sst - SST_OPTIMAL) / SST_WIDTH) ** 2)
    f_par = np.tanh(par / PAR_SAT)
    return 0.05 + 0.95 * f_sst * f_par


def _window_mask(months: pd.Series, window: tuple) -> pd.Series:
    lo, hi = window
    if lo <= hi:
        return months.between(lo, hi)
    # Wrap-around (e.g. Nov-Mar)
    return (months >= lo) | (months <= hi)


def derive(df_daily: pd.DataFrame, site: str = 'Ouest') -> pd.DataFrame:
    sub = df_daily[df_daily['SITE'] == site].copy()
    sub['DATE'] = pd.to_datetime(sub['DATE'])
    sub['year']  = sub['DATE'].dt.year
    sub['month'] = sub['DATE'].dt.month

    rows = []
    for year, ydf in sub.groupby('year'):
        late_summer = ydf[ydf['month'].isin([8, 9])]
        if late_summer.empty:
            continue

        row = {
            'site_id':     site,
            'year':        int(year),
            'survey_date': late_summer['DATE'].iloc[-1].strftime('%Y-%m-%d'),
            'cover':       float(late_summer['COVERAGE'].mean()),
            'cover_std':   float(late_summer['COVERAGE'].std()),
            'cover_min':   float(late_summer['COVERAGE'].min()),
            'cover_max':   float(late_summer['COVERAGE'].max()),
            # Proxies — the daily dataset doesn't carry height or damage,
            # but we can infer a proxy: damage = relative cover loss vs
            # prior 4-year max; height = cover/100 * 50 cm.
            'height_proxy_cm':  float(late_summer['COVERAGE'].mean() / 100 * 50),
        }

        # Damage proxy: % below the rolling 4-year max
        # (the user's brief flags a damage epidemic in 2017-2018)
        prior = sub[(sub['year'] < year) & (sub['year'] >= year - 4)]
        if not prior.empty:
            prior_max = prior['COVERAGE'].quantile(0.95)
            row['damage_proxy_pct'] = float(
                max(0.0, (prior_max - row['cover']) / max(prior_max, 1) * 100)
            )
        else:
            row['damage_proxy_pct'] = 0.0

        # Environmental summaries across 5 windows
        for wname, wrange in WINDOWS.items():
            wdf = ydf[_window_mask(ydf['month'], wrange)]
            if wdf.empty:
                continue
            sst = wdf['SST'].values
            par = wdf['PAR'].values
            gpi = _gpi(sst, par)
            row[f'{wname}_sst_mean']  = float(np.mean(sst))
            row[f'{wname}_sst_max']   = float(np.max(sst))
            row[f'{wname}_sst_min']   = float(np.min(sst))
            row[f'{wname}_sst_std']   = float(np.std(sst))
            row[f'{wname}_par_mean']  = float(np.mean(par))
            row[f'{wname}_par_max']   = float(np.max(par))
            row[f'{wname}_par_min']   = float(np.min(par))
            row[f'{wname}_par_std']   = float(np.std(par))
            row[f'{wname}_gpi_mean']  = float(np.mean(gpi))
            row[f'{wname}_days_above_22'] = int(np.sum(sst > HEAT_THR))
            row[f'{wname}_days_below_8']  = int(np.sum(sst < COLD_THR))

        rows.append(row)

    out = pd.DataFrame(rows).sort_values('year').reset_index(drop=True)
    return out


if __name__ == '__main__':
    print(f'Loading {ALIGNED_EELGRASS_CSV}')
    df = pd.read_csv(ALIGNED_EELGRASS_CSV)
    annual = derive(df, site='Ouest')

    annual.to_csv(OUT_FILE, index=False)
    print(f'\nDerived annual Ouest table: shape={annual.shape}')
    print(f'  columns: {len(annual.columns)}')
    print(f'  years:   {annual["year"].tolist()}')
    print(f'  cover:   {annual["cover"].round(2).tolist()}')
    print(f'  damage:  {annual["damage_proxy_pct"].round(1).tolist()}')
    print(f'\n→ {OUT_FILE}')
