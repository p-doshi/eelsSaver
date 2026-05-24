"""
pipeline.visualize
------------------
Generates training-time visualizations for the teacher labels, the feature
matrix, and the trained student models. Output goes to data/figures/.

Plots produced:
  Teacher:
    01_teacher_gpi_timeseries.png        — GPI_90d per site over time
    02_teacher_decline_timeseries.png    — decline_prob_180d per site over time
    03_teacher_stress_stacked.png        — stress-regime stacked area per site
    04_teacher_distributions.png         — violin + bar summary per site
  Features:
    05_feature_correlation.png           — Pearson heatmap of top 30 retained features
    06_spectral_by_stress.png            — band/index means by dominant stress class
    07_pixel_map.png                     — pixel scatter coloured by site
  Student:
    08_shap_gpi.png                      — top SHAP drivers for GPI
    09_shap_decline.png                  — top SHAP drivers for decline prob
    10_xgb_importance.png                — XGBoost gain importance per model
    11_calibration_gpi.png               — predicted vs actual GPI (val + test)
    12_calibration_decline.png           — predicted vs actual decline prob
    13_confusion_stress.png              — confusion matrix for stress classification
    14_roc_decline.png                   — ROC curve (decline prob, binarised)
    15_metrics_bars.png                  — val vs test metrics across all three models
  Predictions:
    16_pixel_predictions_map.png         — lat/lon scatter coloured by predicted decline prob
    17_teacher_vs_student.png            — scatter of teacher GPI vs student GPI per pixel

Usage:
    python -m pipeline.visualize
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
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, mean_absolute_error, r2_score,
)
from sklearn.preprocessing import LabelEncoder

from config import (
    BASE_DIR, MODELS_DIR, PREDICTIONS_DIR,
    FORILLON_FEATURES, TEACHER_LABELS_CSV,
    FORILLON_SITES, STRESS_CLASSES,
)

# Output dir
FIG_DIR = BASE_DIR / 'data' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style('whitegrid')
plt.rcParams.update({
    'figure.figsize':  (12, 6),
    'figure.dpi':      100,
    'savefig.dpi':     140,
    'axes.titlesize':  13,
    'axes.labelsize':  11,
    'legend.fontsize': 9,
})

SITE_COLORS   = {'Ouest': '#1f77b4', 'Marais': '#2ca02c', 'Sud': '#d62728'}
STRESS_COLORS = {'heat': '#e41a1c', 'cold': '#377eb8',
                 'light': '#ff7f00', 'senescence': '#984ea3'}
RISK_COLORS   = {'Low': '#16a34a', 'Medium': '#d97706', 'High': '#b91c1c'}


def _save(fig, name):
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f'  → {path.name}')


# ---------------------------------------------------------------------------
# Load everything
# ---------------------------------------------------------------------------

print('Loading data …')
labels = pd.read_csv(TEACHER_LABELS_CSV, parse_dates=['window_center'])
feats  = pd.read_csv(FORILLON_FEATURES,   parse_dates=['window_center'])

with open(MODELS_DIR / 'gpi_model.pkl',     'rb') as f: gpi_m     = pickle.load(f)
with open(MODELS_DIR / 'decline_model.pkl', 'rb') as f: decline_m = pickle.load(f)
with open(MODELS_DIR / 'stress_model.pkl',  'rb') as f: stress_m, stress_enc = pickle.load(f)
with open(MODELS_DIR / 'feature_cols.pkl',  'rb') as f: feat_cols = pickle.load(f)
with open(MODELS_DIR / 'validation_report.json') as f:  report = json.load(f)

# Re-merge so we can reproduce splits for prediction plots
labels['wc_rounded'] = labels['window_center'].dt.round('14D')
feats['wc_rounded']  = feats['window_center'].dt.round('14D')
merged = pd.merge(feats, labels.drop(columns=['window_center'], errors='ignore'),
                  on=['SITE', 'wc_rounded'], how='inner')
stress_cols = ['frac_heat', 'frac_cold', 'frac_light', 'frac_senescence']
merged['dominant_stress'] = (merged[stress_cols]
                              .idxmax(axis=1)
                              .map(dict(zip(stress_cols, STRESS_CLASSES))))

# Splits (mirror student.py)
test_mask  = merged['window_center'].dt.year == 2018
val_mask   = (merged['SITE'] == 'Sud') & ~test_mask
train_mask = ~merged['SITE'].isin(['Sud']) & ~test_mask
train, val, test = merged[train_mask], merged[val_mask], merged[test_mask]


# ---------------------------------------------------------------------------
# 01–04  Teacher diagnostics
# ---------------------------------------------------------------------------

print('\nTeacher diagnostics …')

# 01 — GPI time series
fig, ax = plt.subplots()
for site, sub in labels.groupby('SITE'):
    sub = sub.sort_values('window_center')
    ax.plot(sub['window_center'], sub['GPI_90d'],
            label=site, color=SITE_COLORS[site], linewidth=1.6)
ax.set_title('Teacher GPI_90d over time')
ax.set_ylabel('GPI_90d  (0 = no growth potential, 1 = optimal)')
ax.legend(title='Site')
_save(fig, '01_teacher_gpi_timeseries.png')

# 02 — Decline probability time series
fig, ax = plt.subplots()
for site, sub in labels.groupby('SITE'):
    sub = sub.sort_values('window_center')
    ax.plot(sub['window_center'], sub['decline_prob_180d'],
            label=site, color=SITE_COLORS[site], linewidth=1.6)
ax.axhline(0.5, ls='--', color='gray', alpha=0.5, label='0.5 alert')
ax.set_title('Teacher decline_prob_180d over time')
ax.set_ylabel('P(bed drops below 50% within 180 d)')
ax.legend(title='Site')
_save(fig, '02_teacher_decline_timeseries.png')

# 03 — Stress regime stacked area
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
for ax, site in zip(axes, FORILLON_SITES):
    sub = labels[labels['SITE'] == site].sort_values('window_center')
    fracs = sub[stress_cols].values.T
    ax.stackplot(sub['window_center'], fracs,
                  labels=STRESS_CLASSES,
                  colors=[STRESS_COLORS[c] for c in STRESS_CLASSES],
                  alpha=0.9)
    ax.set_ylim(0, 1)
    ax.set_ylabel(site)
    ax.set_title(f'Stress composition — {site}')
axes[0].legend(loc='upper right', ncol=4, frameon=True)
axes[-1].set_xlabel('Window center')
_save(fig, '03_teacher_stress_stacked.png')

# 04 — Distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
sns.violinplot(data=labels, x='SITE', y='GPI_90d', ax=axes[0],
                palette=SITE_COLORS, order=FORILLON_SITES)
axes[0].set_title('GPI_90d distribution')
sns.violinplot(data=labels, x='SITE', y='decline_prob_180d', ax=axes[1],
                palette=SITE_COLORS, order=FORILLON_SITES)
axes[1].set_title('Decline probability distribution')
axes[1].axhline(0.5, ls='--', color='gray', alpha=0.5)
dom = (labels[stress_cols].idxmax(axis=1)
        .map(dict(zip(stress_cols, STRESS_CLASSES))))
ct = pd.crosstab(labels['SITE'], dom).reindex(FORILLON_SITES)
ct.plot(kind='bar', stacked=True, ax=axes[2],
        color=[STRESS_COLORS[c] for c in ct.columns])
axes[2].set_title('Dominant-stress count per site')
axes[2].set_ylabel('Window count'); axes[2].set_xlabel('')
axes[2].tick_params(axis='x', rotation=0)
_save(fig, '04_teacher_distributions.png')


# ---------------------------------------------------------------------------
# 05–07  Feature diagnostics
# ---------------------------------------------------------------------------

print('\nFeature diagnostics …')

# 05 — correlation heatmap (top 30 retained features by SHAP importance)
shap_gpi = pd.read_csv(MODELS_DIR / 'shap_gpi.csv').head(30)
top_feat = shap_gpi['feature'].tolist()
corr = train[top_feat].corr()
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            xticklabels=True, yticklabels=True, ax=ax,
            cbar_kws={'label': 'Pearson r'})
ax.set_title('Feature correlation (top-30 SHAP-ranked, training set)')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
plt.setp(ax.get_yticklabels(), fontsize=8)
_save(fig, '05_feature_correlation.png')

# 06 — band means by dominant stress
band_cols = [f'{b}_mean' for b in
             ['B2', 'B3', 'B4', 'B5', 'B6', 'B8', 'GB_ratio', 'turbidity']]
band_cols = [c for c in band_cols if c in merged.columns]
melted = merged[band_cols + ['dominant_stress']].melt(
    id_vars='dominant_stress', var_name='band', value_name='value')
fig, ax = plt.subplots(figsize=(13, 5))
sns.boxplot(data=melted, x='band', y='value', hue='dominant_stress', ax=ax,
            palette=STRESS_COLORS, fliersize=1)
ax.set_title('Spectral feature means by dominant stress regime')
ax.legend(title='Stress', loc='upper right')
plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
_save(fig, '06_spectral_by_stress.png')

# 07 — pixel locations
fig, ax = plt.subplots(figsize=(8, 7))
for site, sub in feats.drop_duplicates('pixel_id').groupby('SITE'):
    ax.scatter(sub['longitude'], sub['latitude'],
               s=12, alpha=0.7, label=site, color=SITE_COLORS[site])
ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
ax.set_title(f'Synthetic Forillon pixel grid  ({feats["pixel_id"].nunique()} pixels)')
ax.legend(title='Site')
_save(fig, '07_pixel_map.png')


# ---------------------------------------------------------------------------
# 08–10  SHAP / importance
# ---------------------------------------------------------------------------

print('\nSHAP & importance …')

def _bar_top(df_csv, title, fname, n=20):
    df = pd.read_csv(df_csv).head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(df['feature'], df['mean_abs_shap'], color='#4f46e5')
    ax.set_title(title)
    ax.set_xlabel('mean(|SHAP|)')
    _save(fig, fname)

_bar_top(MODELS_DIR / 'shap_gpi.csv',
          'GPI student — top SHAP drivers',
          '08_shap_gpi.png')
_bar_top(MODELS_DIR / 'shap_decline.csv',
          'Decline-probability student — top SHAP drivers',
          '09_shap_decline.png')

# 10 — XGBoost gain importance for all three models side-by-side
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, model, name in [
    (axes[0], gpi_m,     'GPI'),
    (axes[1], decline_m, 'Decline'),
    (axes[2], stress_m,  'Stress'),
]:
    booster = model.get_booster()
    imp = booster.get_score(importance_type='gain')
    if not imp:
        ax.set_title(f'{name}: no importance'); continue
    # XGBoost may return either real feature names (DataFrame fit) or 'f0','f1'
    imp_named = {}
    for k, v in imp.items():
        if k in feat_cols:
            imp_named[k] = v
        elif k.startswith('f') and k[1:].isdigit() and int(k[1:]) < len(feat_cols):
            imp_named[feat_cols[int(k[1:])]] = v
    top = pd.Series(imp_named).sort_values(ascending=True).tail(15)
    ax.barh(top.index, top.values, color='#0ea5e9')
    ax.set_title(f'{name} — XGBoost gain (top 15)')
    ax.set_xlabel('gain')
_save(fig, '10_xgb_importance.png')


# ---------------------------------------------------------------------------
# 11–14  Calibration / confusion / ROC
# ---------------------------------------------------------------------------

print('\nCalibration / confusion / ROC …')

def _scatter_calibration(model, target, ylabel, fname, clip=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    for ax, split, tag in [(axes[0], val, 'Spatial val (Sud)'),
                            (axes[1], test, 'Temporal test (2018)')]:
        sub = split.dropna(subset=[target])
        X = sub[feat_cols].fillna(0)
        y = sub[target].values
        p = model.predict(X)
        if clip is not None:
            p = np.clip(p, *clip)
        mae = mean_absolute_error(y, p)
        r2  = r2_score(y, p) if y.std() > 0 else np.nan
        ax.scatter(y, p, s=8, alpha=0.4, color='#1f77b4')
        lims = [min(y.min(), p.min()), max(y.max(), p.max())]
        ax.plot(lims, lims, 'k--', lw=1)
        ax.set_xlabel(f'Teacher {target}')
        ax.set_ylabel(f'Student {target}')
        ax.set_title(f'{tag}\nMAE={mae:.4f}  R²={r2:.4f}')
    fig.suptitle(ylabel, fontsize=14)
    _save(fig, fname)

_scatter_calibration(gpi_m,     'GPI_90d',
                      'GPI calibration (predicted vs actual)',
                      '11_calibration_gpi.png')
_scatter_calibration(decline_m, 'decline_prob_180d',
                      'Decline-probability calibration',
                      '12_calibration_decline.png', clip=(0, 1))

# 13 — Stress confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, split, tag in [(axes[0], val, 'Spatial val (Sud)'),
                        (axes[1], test, 'Temporal test (2018)')]:
    sub = split.dropna(subset=['dominant_stress'])
    X = sub[feat_cols].fillna(0)
    y_true = stress_enc.transform(sub['dominant_stress'])
    y_pred = stress_m.predict(X)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(stress_enc.classes_)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=stress_enc.classes_, yticklabels=stress_enc.classes_,
                ax=ax)
    ax.set_title(tag); ax.set_xlabel('Predicted'); ax.set_ylabel('True')
fig.suptitle('Stress-regime confusion matrices', fontsize=14)
_save(fig, '13_confusion_stress.png')

# 14 — ROC curve for decline probability
fig, ax = plt.subplots(figsize=(7, 6))
for split, tag, color in [(val, 'Spatial val', '#1f77b4'),
                           (test, 'Temporal test', '#d62728')]:
    sub = split.dropna(subset=['decline_prob_180d'])
    if sub['decline_prob_180d'].gt(0.5).nunique() < 2:
        continue
    X = sub[feat_cols].fillna(0)
    p = np.clip(decline_m.predict(X), 0, 1)
    y = (sub['decline_prob_180d'].values > 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y, p)
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f'{tag}  AUC={auc(fpr, tpr):.3f}')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
ax.set_title('ROC — decline_prob > 0.5')
ax.legend(loc='lower right')
_save(fig, '14_roc_decline.png')


# ---------------------------------------------------------------------------
# 15  Metrics summary bars
# ---------------------------------------------------------------------------

print('\nMetrics summary …')

gpi_v = report['gpi']
dec_v = report['decline']
str_v = report['stress']

rows = []
for split in ['val_spatial', 'test_temporal']:
    rows.append({'model': 'GPI',     'split': split, 'metric': 'MAE',    'value': gpi_v[split]['mae']})
    rows.append({'model': 'GPI',     'split': split, 'metric': 'R²',     'value': gpi_v[split]['r2']})
    rows.append({'model': 'Decline', 'split': split, 'metric': 'MAE',    'value': dec_v[split]['mae']})
    rows.append({'model': 'Decline', 'split': split, 'metric': 'AUC',    'value': dec_v[split]['auc']})
    rows.append({'model': 'Stress',  'split': split, 'metric': 'Macro-F1','value': str_v[split]['macro_f1']})

mdf = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=mdf, x='metric', y='value', hue='split', ax=ax,
            palette={'val_spatial': '#1f77b4', 'test_temporal': '#d62728'})
ax.set_title('Student model metrics — spatial val (Sud) vs temporal test (2018)')
ax.set_ylim(0, 1.05)
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', fontsize=8, padding=2)
ax.legend(title='Split')
_save(fig, '15_metrics_bars.png')


# ---------------------------------------------------------------------------
# 16–17  Prediction / teacher-vs-student
# ---------------------------------------------------------------------------

print('\nPrediction maps …')

# Use the entire dataset (full predictions) for a spatial map of the latest window
all_X = merged[feat_cols].fillna(0)
merged['pred_gpi']     = gpi_m.predict(all_X)
merged['pred_decline'] = np.clip(decline_m.predict(all_X), 0, 1)

latest_wc = merged['window_center'].max()
latest = merged[merged['window_center'] == latest_wc]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sc1 = axes[0].scatter(latest['longitude'], latest['latitude'],
                       c=latest['pred_gpi'], cmap='RdYlGn',
                       vmin=0, vmax=1, s=18)
plt.colorbar(sc1, ax=axes[0], label='Predicted GPI')
axes[0].set_title(f'Pixel-level GPI — {latest_wc.date()}')
axes[0].set_xlabel('Longitude'); axes[0].set_ylabel('Latitude')

sc2 = axes[1].scatter(latest['longitude'], latest['latitude'],
                       c=latest['pred_decline'], cmap='RdYlGn_r',
                       vmin=0, vmax=1, s=18)
plt.colorbar(sc2, ax=axes[1], label='Predicted decline prob.')
axes[1].set_title(f'Pixel-level decline risk — {latest_wc.date()}')
axes[1].set_xlabel('Longitude'); axes[1].set_ylabel('Latitude')
_save(fig, '16_pixel_predictions_map.png')

# 17 — Teacher vs student scatter (full merged set)
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
axes[0].scatter(merged['GPI_90d'], merged['pred_gpi'],
                 s=4, alpha=0.3, color='#0ea5e9')
axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
axes[0].set_xlabel('Teacher GPI_90d'); axes[0].set_ylabel('Student GPI')
axes[0].set_title('Teacher vs Student — GPI')

axes[1].scatter(merged['decline_prob_180d'], merged['pred_decline'],
                 s=4, alpha=0.3, color='#a855f7')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
axes[1].set_xlabel('Teacher decline_prob_180d'); axes[1].set_ylabel('Student decline_prob')
axes[1].set_title('Teacher vs Student — Decline probability')
_save(fig, '17_teacher_vs_student.png')


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f'\nDone. {len(list(FIG_DIR.glob("*.png")))} figures in {FIG_DIR}/')
