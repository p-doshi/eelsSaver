"""
pipeline.visualize_advanced
---------------------------
Second batch of training visualizations. Covers diagnostics that the basic
plots in visualize.py don't show: temporal tracking, residual analysis,
reliability calibration, signed SHAP effects, partial dependence,
out-of-distribution detection, bed-level aggregation, and learning curves.

Output goes to data/figures/.

Plots produced:
    18_student_vs_teacher_timeseries.png — student tracking the teacher per site
    19_residuals.png                     — per-split residual boxplots + hist
    20_reliability_decline.png           — calibration / reliability diagram
    21_shap_signed_gpi.png               — directional SHAP beeswarm for GPI
    22_partial_dependence.png            — PD plots for top-4 GPI features
    23_mahalanobis_ood.png               — OOD distance distribution
    24_bed_aggregation_demo.png          — example pixel→bed roll-up
    25_learning_curve.png                — validation MAE vs ntree (GPI + Decline)
    26_monthly_error.png                 — error breakdown by month-of-year
    27_stress_probability_heatmap.png    — soft stress probabilities over time

Usage:
    python -m pipeline.visualize_advanced
"""

from __future__ import annotations
import json
import pickle
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.calibration import calibration_curve
from sklearn.inspection import partial_dependence
from scipy.spatial.distance import mahalanobis

from config import (
    BASE_DIR, MODELS_DIR,
    FORILLON_FEATURES, TEACHER_LABELS_CSV,
    FORILLON_SITES, STRESS_CLASSES,
    XGB_PARAMS,
)

FIG_DIR = BASE_DIR / 'data' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style('whitegrid')
plt.rcParams.update({'figure.dpi': 100, 'savefig.dpi': 140,
                      'axes.titlesize': 13, 'axes.labelsize': 11})

SITE_COLORS = {'Ouest': '#1f77b4', 'Marais': '#2ca02c', 'Sud': '#d62728'}
STRESS_COLORS = {'heat': '#e41a1c', 'cold': '#377eb8',
                  'light': '#ff7f00', 'senescence': '#984ea3'}
SPLIT_COLORS = {'train': '#94a3b8', 'val_spatial': '#1f77b4',
                 'test_temporal': '#d62728'}


def _save(fig, name):
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f'  → {path.name}')


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

print('Loading data + models …')
labels = pd.read_csv(TEACHER_LABELS_CSV, parse_dates=['window_center'])
feats  = pd.read_csv(FORILLON_FEATURES,   parse_dates=['window_center'])

with open(MODELS_DIR / 'gpi_model.pkl',     'rb') as f: gpi_m     = pickle.load(f)
with open(MODELS_DIR / 'decline_model.pkl', 'rb') as f: decline_m = pickle.load(f)
with open(MODELS_DIR / 'stress_model.pkl',  'rb') as f: stress_m, stress_enc = pickle.load(f)
with open(MODELS_DIR / 'feature_cols.pkl',  'rb') as f: feat_cols = pickle.load(f)
with open(MODELS_DIR / 'train_X_mean.pkl',  'rb') as f: train_mean = pickle.load(f)
with open(MODELS_DIR / 'train_cov_inv.pkl', 'rb') as f: train_cov_inv = pickle.load(f)

# Merge features + labels (mirrors student.py)
labels['wc_rounded'] = labels['window_center'].dt.round('14D')
feats['wc_rounded']  = feats['window_center'].dt.round('14D')
merged = pd.merge(feats, labels.drop(columns=['window_center'], errors='ignore'),
                  on=['SITE', 'wc_rounded'], how='inner')
stress_cols = ['frac_heat', 'frac_cold', 'frac_light', 'frac_senescence']
merged['dominant_stress'] = (merged[stress_cols]
                              .idxmax(axis=1)
                              .map(dict(zip(stress_cols, STRESS_CLASSES))))

# Recompute splits
test_mask  = merged['window_center'].dt.year == 2018
val_mask   = (merged['SITE'] == 'Sud') & ~test_mask
train_mask = ~merged['SITE'].isin(['Sud']) & ~test_mask
splits = {
    'train':         merged[train_mask].copy(),
    'val_spatial':   merged[val_mask].copy(),
    'test_temporal': merged[test_mask].copy(),
}

# Predict for all rows once
all_X = merged[feat_cols].fillna(0)
merged['pred_gpi']     = gpi_m.predict(all_X)
merged['pred_decline'] = np.clip(decline_m.predict(all_X), 0, 1)
merged['gpi_resid']    = merged['pred_gpi'] - merged['GPI_90d']
merged['dec_resid']    = merged['pred_decline'] - merged['decline_prob_180d']

# Rebuild splits AFTER predictions are attached
splits = {
    'train':         merged[train_mask].copy(),
    'val_spatial':   merged[val_mask].copy(),
    'test_temporal': merged[test_mask].copy(),
}
for k, s in splits.items():
    s['split'] = k


# ---------------------------------------------------------------------------
# 18  Student vs teacher tracking per site
# ---------------------------------------------------------------------------

print('\n18 — student vs teacher tracking …')
fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
for ax, site in zip(axes, FORILLON_SITES):
    sub = merged[merged['SITE'] == site].sort_values('window_center')
    # Aggregate pixel predictions to bed mean per window
    bed = sub.groupby('window_center').agg(
        teacher=('GPI_90d', 'mean'),
        student_mean=('pred_gpi', 'mean'),
        student_lo=('pred_gpi', lambda x: x.quantile(0.10)),
        student_hi=('pred_gpi', lambda x: x.quantile(0.90)),
    ).reset_index()
    ax.fill_between(bed['window_center'], bed['student_lo'], bed['student_hi'],
                    color=SITE_COLORS[site], alpha=0.18,
                    label='student 10–90 % pixel band')
    ax.plot(bed['window_center'], bed['student_mean'],
            color=SITE_COLORS[site], lw=1.8, label='student mean')
    ax.plot(bed['window_center'], bed['teacher'],
            color='black', ls='--', lw=1.2, label='teacher GPI_90d')
    ax.set_title(f'{site} — student tracking teacher')
    ax.set_ylabel('GPI')
    ax.set_ylim(0, 1)
    if site == 'Ouest':
        ax.legend(loc='upper right')
axes[-1].set_xlabel('Window center')
_save(fig, '18_student_vs_teacher_timeseries.png')


# ---------------------------------------------------------------------------
# 19  Residual analysis
# ---------------------------------------------------------------------------

print('19 — residual distributions …')
all_split = pd.concat([s.assign(split=k) for k, s in splits.items()],
                       ignore_index=True)
all_split['gpi_resid'] = all_split['pred_gpi'] - all_split['GPI_90d']
all_split['dec_resid'] = all_split['pred_decline'] - all_split['decline_prob_180d']

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, col, title in [
    (axes[0, 0], 'gpi_resid', 'GPI residuals (student − teacher)'),
    (axes[0, 1], 'dec_resid', 'Decline-prob residuals'),
]:
    for k, sub in all_split.groupby('split'):
        ax.hist(sub[col], bins=40, alpha=0.5, label=k, color=SPLIT_COLORS[k])
    ax.axvline(0, color='black', lw=1)
    ax.set_title(title)
    ax.set_xlabel('residual')
    ax.legend()

for ax, col, title in [
    (axes[1, 0], 'gpi_resid', 'GPI residual boxplot by split'),
    (axes[1, 1], 'dec_resid', 'Decline residual boxplot by split'),
]:
    sns.boxplot(data=all_split, x='split', y=col, ax=ax,
                palette=SPLIT_COLORS, order=['train', 'val_spatial', 'test_temporal'])
    ax.axhline(0, color='black', lw=1)
    ax.set_title(title)
_save(fig, '19_residuals.png')


# ---------------------------------------------------------------------------
# 20  Reliability / calibration curve for decline probability
# ---------------------------------------------------------------------------

print('20 — reliability diagram (decline) …')
fig, ax = plt.subplots(figsize=(7, 6))
for k in ['val_spatial', 'test_temporal']:
    sub = splits[k].dropna(subset=['decline_prob_180d'])
    if sub['decline_prob_180d'].nunique() < 2:
        continue
    y_bin = (sub['decline_prob_180d'].values > 0.5).astype(int)
    p     = np.clip(decline_m.predict(sub[feat_cols].fillna(0)), 0, 1)
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        continue
    frac, mean_pred = calibration_curve(y_bin, p, n_bins=10, strategy='uniform')
    ax.plot(mean_pred, frac, marker='o', lw=2, label=k, color=SPLIT_COLORS[k])
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
ax.set_xlabel('Mean predicted probability')
ax.set_ylabel('Observed fraction (decline > 0.5)')
ax.set_title('Reliability diagram — decline probability')
ax.legend()
_save(fig, '20_reliability_decline.png')


# ---------------------------------------------------------------------------
# 21  Signed SHAP "beeswarm" for GPI student
# ---------------------------------------------------------------------------

print('21 — signed SHAP effects (GPI) …')
sample_n = min(800, len(splits['train']))
X_sample = splits['train'][feat_cols].fillna(0).sample(sample_n, random_state=42)
explainer = shap.TreeExplainer(gpi_m)
shap_vals = explainer.shap_values(X_sample)

# Top-15 features by mean |SHAP|
order = np.argsort(np.abs(shap_vals).mean(axis=0))[-15:]

fig, ax = plt.subplots(figsize=(10, 7))
for rank, idx in enumerate(order):
    fv = X_sample.iloc[:, idx].values
    sv = shap_vals[:, idx]
    # Normalise feature value to 0-1 for color
    fv_norm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-12)
    ys = rank + (np.random.rand(len(sv)) - 0.5) * 0.5
    ax.scatter(sv, ys, c=fv_norm, cmap='coolwarm', s=12,
                alpha=0.7, edgecolors='none')
ax.set_yticks(range(len(order)))
ax.set_yticklabels([feat_cols[i] for i in order])
ax.axvline(0, color='black', lw=1)
ax.set_xlabel('SHAP value (impact on predicted GPI)')
ax.set_title('Signed SHAP effects — GPI student\n(red = high feature value, blue = low)')
sm = plt.cm.ScalarMappable(cmap='coolwarm')
sm.set_array([])
cb = plt.colorbar(sm, ax=ax, pad=0.01)
cb.set_label('feature value (normalised)')
_save(fig, '21_shap_signed_gpi.png')


# ---------------------------------------------------------------------------
# 22  Partial dependence plots for top-4 GPI features
# ---------------------------------------------------------------------------

print('22 — partial dependence …')
top_idx = list(order[-4:])[::-1]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
X_pd = splits['train'][feat_cols].fillna(0).sample(
    min(2000, len(splits['train'])), random_state=42)

for ax, idx in zip(axes.ravel(), top_idx):
    fname = feat_cols[idx]
    try:
        pd_out = partial_dependence(gpi_m, X_pd, features=[idx],
                                     grid_resolution=40)
        # API differences across sklearn versions
        if hasattr(pd_out, 'average'):  # PartialDependenceDisplay format
            avg = pd_out['average'][0] if isinstance(pd_out, dict) else pd_out.average[0]
            grid = pd_out['grid_values'][0] if isinstance(pd_out, dict) else pd_out.grid_values[0]
        else:
            avg = pd_out['average'][0] if 'average' in pd_out else pd_out[0][0]
            grid = pd_out['values'][0] if 'values' in pd_out else pd_out[1][0]
        ax.plot(grid, avg, color='#1d4ed8', lw=2)
        ax.set_xlabel(fname); ax.set_ylabel('Partial dependence (GPI)')
        ax.set_title(f'PD — {fname}')
    except Exception as e:
        ax.text(0.5, 0.5, f'PD failed:\n{e}', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title(fname)
_save(fig, '22_partial_dependence.png')


# ---------------------------------------------------------------------------
# 23  Mahalanobis OOD distance distribution
# ---------------------------------------------------------------------------

print('23 — Mahalanobis OOD distribution …')
def maha(X):
    return np.array([
        mahalanobis(row, train_mean, train_cov_inv) for row in X.values
    ])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for k, color in [('train', '#94a3b8'), ('val_spatial', '#1f77b4'),
                  ('test_temporal', '#d62728')]:
    X = splits[k][feat_cols].fillna(0)
    d = maha(X)
    axes[0].hist(d, bins=40, alpha=0.55, label=k, color=color, density=True)
axes[0].axvline(np.percentile(maha(splits['train'][feat_cols].fillna(0)), 95),
                color='black', ls='--', lw=1,
                label='train 95th pctl (OOD threshold)')
axes[0].set_title('Mahalanobis distance — distribution by split')
axes[0].set_xlabel('Distance')
axes[0].set_ylabel('Density')
axes[0].legend()

# Per-site distance in test set
test_X = splits['test_temporal'][feat_cols].fillna(0)
splits['test_temporal']['maha'] = maha(test_X)
sns.boxplot(data=splits['test_temporal'], x='SITE', y='maha', ax=axes[1],
            palette=SITE_COLORS, order=FORILLON_SITES)
axes[1].set_title('Mahalanobis distance — test set by site')
axes[1].set_xlabel('Site'); axes[1].set_ylabel('Distance')
_save(fig, '23_mahalanobis_ood.png')


# ---------------------------------------------------------------------------
# 24  Bed-level aggregation demo
# ---------------------------------------------------------------------------

print('24 — bed-level aggregation demo …')
# Demonstrate what inference.py produces: pixel → bed roll-up per window
bed_demo = (merged.groupby(['SITE', 'window_center'])
            .agg(bed_GPI=('pred_gpi', 'mean'),
                  bed_decline=('pred_decline', 'mean'),
                  high_risk_frac=('pred_decline',
                                  lambda x: float((x > 0.6).mean())),
                  n_pixels=('pixel_id', 'count'))
            .reset_index())

def risk_tier(p):
    return 'High' if p > 0.66 else ('Medium' if p > 0.33 else 'Low')
bed_demo['bed_risk_tier'] = bed_demo['bed_decline'].apply(risk_tier)

fig, axes = plt.subplots(2, 1, figsize=(13, 8))  # do NOT sharex (mixed axes)
for site in FORILLON_SITES:
    sub = bed_demo[bed_demo['SITE'] == site].sort_values('window_center')
    axes[0].plot(sub['window_center'].values, sub['bed_decline'].values,
                  color=SITE_COLORS[site], lw=1.6, marker='o', markersize=3,
                  label=site, alpha=0.85)
axes[0].axhline(0.33, ls='--', color='#d97706', alpha=0.6, label='Medium')
axes[0].axhline(0.66, ls='--', color='#b91c1c', alpha=0.6, label='High')
axes[0].set_ylabel('Bed-level decline probability')
axes[0].set_xlabel('Window center')
axes[0].set_title('Bed-level decline probability over time (mean across pixels)')
axes[0].set_ylim(-0.02, 1.02)
axes[0].legend(ncol=5, loc='upper right')

# Risk-tier composition stack
tier_count = (bed_demo.assign(year=bed_demo['window_center'].dt.year)
              .groupby(['year', 'bed_risk_tier']).size()
              .unstack(fill_value=0))
tier_count = tier_count[['Low', 'Medium', 'High']]
tier_count.plot.bar(stacked=True, ax=axes[1],
                     color=['#16a34a', '#d97706', '#b91c1c'])
axes[1].set_ylabel('# bed-window evaluations')
axes[1].set_xlabel('Year')
axes[1].set_title('Bed-window risk tier composition by year')
axes[1].legend(title='Risk tier')
plt.setp(axes[1].get_xticklabels(), rotation=0)
_save(fig, '24_bed_aggregation_demo.png')


# ---------------------------------------------------------------------------
# 25  Learning curve — validation MAE vs ntree
# ---------------------------------------------------------------------------

print('25 — learning curve …')

def _curve(target, splits, params, clip=None):
    Xtr = splits['train'][feat_cols].fillna(0)
    ytr = splits['train'][target].fillna(0)
    Xvl = splits['val_spatial'][feat_cols].fillna(0)
    yvl = splits['val_spatial'][target].fillna(0)
    Xte = splits['test_temporal'][feat_cols].fillna(0)
    yte = splits['test_temporal'][target].fillna(0)

    model = xgb.XGBRegressor(
        **{**params, 'n_estimators': 400},
        eval_metric='mae',
    )
    model.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xvl, yvl), (Xte, yte)],
              verbose=False)
    res = model.evals_result()
    return res

curves = {
    'GPI':     _curve('GPI_90d',           splits, XGB_PARAMS),
    'Decline': _curve('decline_prob_180d', splits, XGB_PARAMS),
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (name, res) in zip(axes, curves.items()):
    ax.plot(res['validation_0']['mae'], label='train',  color='#94a3b8')
    ax.plot(res['validation_1']['mae'], label='val_spatial', color='#1f77b4')
    ax.plot(res['validation_2']['mae'], label='test_temporal', color='#d62728')
    ax.set_xlabel('Boosting round')
    ax.set_ylabel('MAE')
    ax.set_title(f'{name} learning curve')
    ax.legend()
    ax.set_yscale('log')
_save(fig, '25_learning_curve.png')


# ---------------------------------------------------------------------------
# 26  Monthly error decomposition
# ---------------------------------------------------------------------------

print('26 — monthly error breakdown …')
merged['month'] = merged['window_center'].dt.month
monthly = (merged
            .groupby(['SITE', 'month'])
            .apply(lambda d: pd.Series({
                'gpi_mae': mean_absolute_error(d['GPI_90d'], d['pred_gpi']),
                'dec_mae': mean_absolute_error(d['decline_prob_180d'].fillna(0),
                                                d['pred_decline'].fillna(0)),
            }))
            .reset_index())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(data=monthly, x='month', y='gpi_mae', hue='SITE',
            ax=axes[0], palette=SITE_COLORS, hue_order=FORILLON_SITES)
axes[0].set_title('GPI MAE by month of year')
axes[0].set_xlabel('Month'); axes[0].set_ylabel('MAE')

sns.barplot(data=monthly, x='month', y='dec_mae', hue='SITE',
            ax=axes[1], palette=SITE_COLORS, hue_order=FORILLON_SITES)
axes[1].set_title('Decline MAE by month of year')
axes[1].set_xlabel('Month'); axes[1].set_ylabel('MAE')
_save(fig, '26_monthly_error.png')


# ---------------------------------------------------------------------------
# 27  Soft stress probability heatmap over time per site
# ---------------------------------------------------------------------------

print('27 — stress probability heatmap …')
# Use stress classifier probabilities aggregated per (site, window)
proba = stress_m.predict_proba(all_X)
for i, cls in enumerate(stress_enc.classes_):
    merged[f'pstress_{cls}'] = proba[:, i]

agg = (merged
        .groupby(['SITE', 'window_center'])
        [[f'pstress_{c}' for c in stress_enc.classes_]]
        .mean().reset_index())

fig, axes = plt.subplots(len(FORILLON_SITES), 1, figsize=(13, 8), sharex=True)
for ax, site in zip(axes, FORILLON_SITES):
    sub = agg[agg['SITE'] == site].sort_values('window_center')
    mat = sub[[f'pstress_{c}' for c in stress_enc.classes_]].values.T
    im = ax.imshow(mat, aspect='auto', cmap='magma_r',
                    vmin=0, vmax=1,
                    extent=[0, len(sub) - 1, len(stress_enc.classes_), 0])
    ax.set_yticks(np.arange(len(stress_enc.classes_)) + 0.5)
    ax.set_yticklabels(stress_enc.classes_)
    ax.set_title(f'{site} — soft stress probabilities')
    ax.set_xticks(np.linspace(0, len(sub) - 1, 6))
    ax.set_xticklabels(
        pd.to_datetime(np.linspace(
            sub['window_center'].iloc[0].value,
            sub['window_center'].iloc[-1].value, 6
        )).strftime('%Y-%m')
    )
cbar = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.02)
cbar.set_label('Soft probability')
_save(fig, '27_stress_probability_heatmap.png')


# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

count = len(list(FIG_DIR.glob('*.png')))
print(f'\nDone. {count} total figures in {FIG_DIR}/')
