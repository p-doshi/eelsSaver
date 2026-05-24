"""
pipeline.teacher_ouest
----------------------
Teacher labels for the annual Ouest dataset:
    ouest_eelgrass_environmental_complete.csv  (6 years × 65 variables)

This dataset is structured differently from aligned_eelgrass_env_data.csv:
  - One row per (site_id, year). Survey runs annually in late summer.
  - Eelgrass metrics already observed: cover, height, damage.
  - Environmental drivers summarised across 5 time windows per year.

Because we already have *observed* cover, the teacher's job becomes:
  1. GPI_annual          — Growth Potential Index from env summaries
                             (Gaussian SST × tanh PAR, peak-growing window).
  2. observed_cover      — direct from the survey (no modelling).
  3. observed_damage     — direct from the survey (used for 2017-2018 epidemic).
  4. decline_flag        — 1 if next-year cover falls below 50% of this year.
                             Computed retrospectively; NaN for the last year.
  5. dominant_stress     — classify by which env-summary window crossed
                             heat / cold / light thresholds most often.
  6. shap_sst, shap_par  — leave-one-out feature ablation on a GBM trained
                             to predict observed cover from env summaries
                             (very small N — interpret cautiously).

Output: data/labels/teacher_labels_ouest.csv with one row per (site_id, year).

Usage:
    python -m pipeline.teacher_ouest <ouest_eelgrass_environmental_complete.csv>
"""

from __future__ import annotations
import sys
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from config import (
    LABELS_DIR,
    SST_OPTIMAL, SST_WIDTH, PAR_SATURATION, GPI_FLOOR,
    SST_HEAT_THRESHOLD, SST_COLD_THRESHOLD,
)

OUT_FILE = LABELS_DIR / 'teacher_labels_ouest.csv'


# ---------------------------------------------------------------------------
# GPI from env summaries
# ---------------------------------------------------------------------------

def _gpi(sst: float, par: float) -> float:
    f_sst = float(np.exp(-0.5 * ((sst - SST_OPTIMAL) / SST_WIDTH) ** 2))
    f_par = float(np.tanh(par / PAR_SATURATION))
    return GPI_FLOOR + (1 - GPI_FLOOR) * f_sst * f_par


def _detect_env_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    The Ouest dataset documents 'environmental summaries across 5 time
    windows'. Column names aren't fully specified, so we sniff for any
    columns matching:
        sst_*, par_*, temp_*, mean_temp_*, light_*, etc.
    grouped by suffix (e.g., 'pre', 'early', 'peak', 'late', 'winter' or
    1..5). Returns {'sst': [...], 'par': [...]}.
    """
    cols = df.columns.tolist()
    sst_cols = [c for c in cols
                 if any(tok in c.lower()
                         for tok in ('sst', 'temp', 'temperature'))
                 and df[c].dtype.kind in 'fi']
    par_cols = [c for c in cols
                 if any(tok in c.lower()
                         for tok in ('par', 'light', 'radiation'))
                 and df[c].dtype.kind in 'fi']
    return {'sst': sst_cols, 'par': par_cols}


def _row_gpi(row: pd.Series, env_cols: dict[str, list[str]]) -> float:
    """Average GPI across all available SST × PAR pairings in the row."""
    sst_vals = row[env_cols['sst']].dropna().values
    par_vals = row[env_cols['par']].dropna().values
    if len(sst_vals) == 0 or len(par_vals) == 0:
        return float('nan')
    # Mean SST × mean PAR (coarse but robust to varying column counts)
    return _gpi(float(np.mean(sst_vals)), float(np.mean(par_vals)))


def _row_stress(row: pd.Series, env_cols: dict[str, list[str]]) -> str:
    """Classify by which threshold is most often crossed in env summaries."""
    sst_vals = row[env_cols['sst']].dropna().values
    par_vals = row[env_cols['par']].dropna().values
    if len(sst_vals) == 0:
        return 'unknown'
    heat = float(np.sum(sst_vals > SST_HEAT_THRESHOLD))
    cold = float(np.sum(sst_vals < SST_COLD_THRESHOLD))
    light = float(np.sum(par_vals < np.percentile(par_vals, 20))) if len(par_vals) else 0
    # Senescence is a calendar artifact for annual data — flag only if both
    # heat and cold are zero AND par is mid-range
    if max(heat, cold, light) == 0:
        return 'senescence'
    return ['heat', 'cold', 'light'][int(np.argmax([heat, cold, light]))]


# ---------------------------------------------------------------------------
# Decline flag & shap
# ---------------------------------------------------------------------------

def _decline_flag(df: pd.DataFrame) -> pd.Series:
    """1 if next year's cover < 0.5 × this year's cover."""
    df_sorted = df.sort_values(['site_id', 'year'])
    next_cover = df_sorted.groupby('site_id')['cover'].shift(-1)
    return (next_cover < 0.5 * df_sorted['cover']).astype('float').reindex(df.index)


def _gbm_shap(df: pd.DataFrame, env_cols: dict[str, list[str]]) -> tuple:
    """Train a tiny GBM cover ~ env_features. Return (shap_sst, shap_par)."""
    all_env = env_cols['sst'] + env_cols['par']
    X = df[all_env].fillna(df[all_env].median()).values
    y = df['cover'].fillna(df['cover'].median()).values
    if len(X) < 4:
        return (float('nan'), float('nan'))
    model = GradientBoostingRegressor(
        n_estimators=80, max_depth=2, learning_rate=0.05,
        random_state=42,
    )
    model.fit(X, y)
    # Cheap "ablation" SHAP — re-predict with each column zeroed and measure delta
    base_pred = model.predict(X).mean()
    deltas = []
    for j in range(X.shape[1]):
        Xp = X.copy()
        Xp[:, j] = 0
        deltas.append(abs(model.predict(Xp).mean() - base_pred))
    deltas = np.array(deltas)

    sst_idx = list(range(len(env_cols['sst'])))
    par_idx = list(range(len(env_cols['sst']), len(all_env)))
    shap_sst = float(deltas[sst_idx].sum()) if sst_idx else float('nan')
    shap_par = float(deltas[par_idx].sum()) if par_idx else float('nan')
    return round(shap_sst, 4), round(shap_par, 4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_labels(df: pd.DataFrame) -> pd.DataFrame:
    # Normalise column names — be permissive about case/whitespace
    df = df.rename(columns={c: c.strip() for c in df.columns})
    lower_map = {c.lower(): c for c in df.columns}

    # Required columns (case-insensitive lookup)
    def col(name, fallback=None):
        return lower_map.get(name.lower(), fallback)

    year_c  = col('year')
    site_c  = col('site_id', col('site'))
    cover_c = col('cover', col('eelgrass_cover'))
    if year_c is None or cover_c is None:
        raise ValueError(
            f'Could not find required columns. Saw: {list(df.columns)[:15]} …'
        )
    if site_c is None:
        df['site_id'] = 'Ouest'
        site_c = 'site_id'

    df['year']    = df[year_c].astype(int)
    df['cover']   = df[cover_c].astype(float)
    df['site_id'] = df[site_c].astype(str)

    damage_c = col('damage', col('eelgrass_damage'))
    height_c = col('height', col('eelgrass_height'))

    env_cols = _detect_env_columns(df)
    print(f'Detected env columns:')
    print(f'  SST-like: {env_cols["sst"]}')
    print(f'  PAR-like: {env_cols["par"]}')

    out = pd.DataFrame({
        'site_id': df['site_id'],
        'year':    df['year'],
        'cover':   df['cover'],
    })
    if damage_c: out['damage'] = df[damage_c]
    if height_c: out['height'] = df[height_c]

    out['GPI_annual']        = df.apply(lambda r: _row_gpi(r, env_cols), axis=1)
    out['dominant_stress']   = df.apply(lambda r: _row_stress(r, env_cols), axis=1)
    out['decline_flag']      = _decline_flag(out.rename(columns={'cover': 'cover'}))
    shap_sst, shap_par       = _gbm_shap(df, env_cols)
    out['shap_sst']          = shap_sst
    out['shap_par']          = shap_par

    return out.sort_values(['site_id', 'year']).reset_index(drop=True)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('usage: python -m pipeline.teacher_ouest '
                  '<ouest_eelgrass_environmental_complete.csv>')

    src = sys.argv[1]
    print(f'Loading {src}')
    df = pd.read_csv(src)
    print(f'  shape = {df.shape}, columns sample = {list(df.columns)[:10]}')

    labels = generate_labels(df)
    labels.to_csv(OUT_FILE, index=False)
    print(f'\n→ {OUT_FILE}  ({len(labels)} rows)')
    print(labels.to_string(index=False))
