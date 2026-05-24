"""
pipeline.teacher
----------------
Generates soft labels per (SITE, 90-day window) from Forillon in-situ +
environmental data. Outputs teacher_labels.csv consumed by student.py.

Columns: SITE, window_center, GPI_90d, decline_prob_180d,
         frac_heat, frac_cold, frac_light, frac_senescence,
         shap_sst, shap_par.
"""

from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingRegressor

from config import (
    ALIGNED_EELGRASS_CSV, TEACHER_LABELS_CSV,
    WINDOW_DAYS, STEP_DAYS,
    SST_OPTIMAL, SST_WIDTH, PAR_SATURATION, GPI_FLOOR,
    SST_HEAT_THRESHOLD, SST_COLD_THRESHOLD, PAR_LOW_PCTL, SENESCENCE_MONTHS,
    N_BOOTSTRAP, FORECAST_HORIZON, ML_WINDOW,
)


# ---------------------------------------------------------------------------
# Ecology
# ---------------------------------------------------------------------------

def calculate_gpi(sst: np.ndarray, par: np.ndarray) -> np.ndarray:
    f_sst = np.exp(-0.5 * ((sst - SST_OPTIMAL) / SST_WIDTH) ** 2)
    f_par = np.tanh(par / PAR_SATURATION)
    return GPI_FLOOR + (1 - GPI_FLOOR) * f_sst * f_par


def stress_regime(sst, par, dates) -> dict:
    n = len(sst)
    if n == 0:
        return dict(frac_heat=0.0, frac_cold=0.0, frac_light=0.0, frac_senescence=0.0)

    par_low = np.percentile(par, PAR_LOW_PCTL)
    heat  = float(np.sum(sst > SST_HEAT_THRESHOLD)) / n
    cold  = float(np.sum(sst < SST_COLD_THRESHOLD)) / n
    light = float(np.sum(par < par_low)) / n
    senes = float(np.sum(pd.DatetimeIndex(dates).month.isin(SENESCENCE_MONTHS))) / n

    total = heat + cold + light + senes
    if total > 0:
        heat, cold, light, senes = heat/total, cold/total, light/total, senes/total
    return dict(
        frac_heat=round(heat, 4), frac_cold=round(cold, 4),
        frac_light=round(light, 4), frac_senescence=round(senes, 4),
    )


# ---------------------------------------------------------------------------
# GBM-Multi feature engineering (mirrors modelling_3.py)
# ---------------------------------------------------------------------------

def build_features(data: pd.DataFrame, window: int = ML_WINDOW):
    cov = data['COVERAGE'].values
    sst = data['SST'].values
    par = data['PAR'].values
    dates = pd.to_datetime(data['DATE'].values)

    X, y = [], []
    for i in range(window, len(cov)):
        w_cov = cov[i - window:i]
        w_sst = sst[i - window:i]
        w_par = par[i - window:i]
        doy   = dates[i].dayofyear
        X.append([
            w_cov[-1], w_cov[-7] if window >= 7 else 0.0,
            w_cov[-30] if window >= 30 else 0.0,
            np.mean(w_cov[-7:]), np.std(w_cov[-7:]),
            np.mean(w_cov[-30:]),
            w_sst[-1], w_sst[-7] if window >= 7 else 0.0, np.mean(w_sst[-30:]),
            w_par[-1], w_par[-7] if window >= 7 else 0.0, np.mean(w_par[-30:]),
            np.sin(2 * np.pi * doy / 365), np.cos(2 * np.pi * doy / 365),
        ])
        y.append(cov[i])
    return np.array(X), np.array(y)


def train_gbm(site_df: pd.DataFrame):
    X, y = build_features(site_df)
    if len(X) < 60:
        return None, np.nan, np.nan

    split = int(0.8 * len(X))
    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.05,
        min_samples_leaf=5, subsample=0.8, random_state=42,
    )
    model.fit(X[:split], y[:split])

    sv = shap.TreeExplainer(model).shap_values(X[:split])
    mean_abs = np.abs(sv).mean(axis=0)
    shap_sst = float(mean_abs[6:9].sum())
    shap_par = float(mean_abs[9:12].sum())
    return model, round(shap_sst, 4), round(shap_par, 4)


# ---------------------------------------------------------------------------
# Decline probability via bootstrap recursive forecast
# ---------------------------------------------------------------------------

def decline_probability(model, cov_w, sst_w, par_w, current_cov,
                         horizon=FORECAST_HORIZON, n_bootstrap=N_BOOTSTRAP) -> float:
    if model is None:
        return float('nan')
    rng = np.random.default_rng(42)
    threshold = 0.5 * current_cov
    fallen = 0

    for _ in range(n_bootstrap):
        cw, sw, pw = cov_w.copy().astype(float), sst_w.copy().astype(float), par_w.copy().astype(float)
        below = False
        for step in range(min(horizon, 180)):
            doy = (step % 365) + 1
            feats = np.array([
                cw[-1], cw[-7] if len(cw) >= 7 else 0.0,
                cw[-30] if len(cw) >= 30 else 0.0,
                np.mean(cw[-7:]), np.std(cw[-7:]) if len(cw) > 1 else 0.0,
                np.mean(cw[-30:]),
                sw[-1], sw[-7] if len(sw) >= 7 else sw[-1], np.mean(sw[-30:]),
                pw[-1], pw[-7] if len(pw) >= 7 else pw[-1], np.mean(pw[-30:]),
                np.sin(2 * np.pi * doy / 365), np.cos(2 * np.pi * doy / 365),
            ])
            feats += rng.normal(0, 0.05 * np.abs(feats))
            pred = float(np.clip(model.predict([feats])[0], 0, 100))

            cw = np.append(cw[1:], pred)
            sw = np.append(sw[1:], sw[-1])
            pw = np.append(pw[1:], pw[-1])

            if pred < threshold:
                below = True
                break
        if below:
            fallen += 1

    return round(fallen / n_bootstrap, 4)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def generate_labels(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for site in df['SITE'].unique():
        print(f'\n  Site: {site}')
        sdf = df[df['SITE'] == site].sort_values('DATE').reset_index(drop=True)

        model, shap_sst, shap_par = train_gbm(sdf)
        if model is None:
            print(f'    insufficient data ({len(sdf)} rows) — skip')
            continue

        centers = pd.date_range(
            sdf['DATE'].min() + pd.Timedelta(days=WINDOW_DAYS),
            sdf['DATE'].max(),
            freq=f'{STEP_DAYS}D',
        )

        for wc in centers:
            w0 = wc - pd.Timedelta(days=WINDOW_DAYS // 2)
            w1 = wc + pd.Timedelta(days=WINDOW_DAYS // 2)
            wd = sdf[(sdf['DATE'] >= w0) & (sdf['DATE'] <= w1)]
            if len(wd) < 10:
                continue

            gpi_vals = calculate_gpi(wd['SST'].values, wd['PAR'].values)
            sr = stress_regime(wd['SST'].values, wd['PAR'].values, wd['DATE'].values)

            seed = sdf[sdf['DATE'] <= w1].tail(ML_WINDOW + 5)
            if len(seed) >= ML_WINDOW:
                dp = decline_probability(
                    model,
                    seed['COVERAGE'].values[-ML_WINDOW:],
                    seed['SST'].values[-ML_WINDOW:],
                    seed['PAR'].values[-ML_WINDOW:],
                    float(seed['COVERAGE'].iloc[-1]),
                )
            else:
                dp = float('nan')

            rows.append({
                'SITE':              site,
                'window_center':     wc,
                'GPI_90d':           round(float(gpi_vals.mean()), 4),
                'decline_prob_180d': dp,
                'shap_sst':          shap_sst,
                'shap_par':          shap_par,
                **sr,
            })
        print(f'    {sum(1 for r in rows if r["SITE"] == site)} windows')

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print(f'Loading {ALIGNED_EELGRASS_CSV}')
    df = pd.read_csv(ALIGNED_EELGRASS_CSV)
    df['DATE'] = pd.to_datetime(df['DATE'])
    print(f'  rows={len(df):,}  sites={sorted(df["SITE"].unique())}')

    labels = generate_labels(df)
    labels.to_csv(TEACHER_LABELS_CSV, index=False)

    print(f'\n→ {TEACHER_LABELS_CSV}  ({len(labels)} rows)')
    print('\nGPI_90d:'); print(labels['GPI_90d'].describe().round(4))
    print('\ndecline_prob_180d:'); print(labels['decline_prob_180d'].describe().round(4))
