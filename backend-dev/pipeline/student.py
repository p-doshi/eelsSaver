"""
pipeline.student
----------------
Trains three XGBoost students on Forillon pixel-window features labelled
by the teacher:
    1. GPI regression
    2. Decline probability regression
    3. Dominant stress classification

Validation uses spatial + temporal blocking:
    train = Ouest + Marais, year ≤ 2017
    val   = Sud (spatial hold-out)
    test  = 2018 windows (temporal hold-out)

Outputs to data/models/:
    gpi_model.pkl, decline_model.pkl, stress_model.pkl,
    feature_cols.pkl, train_cov_inv.pkl (LedoitWolf inverse for Mahalanobis OOD),
    train_X_mean.pkl, shap_gpi.csv, shap_decline.csv, validation_report.json
"""

from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')

import json, pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.metrics import (
    mean_absolute_error, r2_score, f1_score,
    roc_auc_score, brier_score_loss,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.covariance import LedoitWolf

from config import (
    FORILLON_FEATURES, TEACHER_LABELS_CSV, MODELS_DIR,
    SPATIAL_VAL_SITE, TEMPORAL_TEST_YR,
    XGB_PARAMS, CORR_DROP_THRESHOLD, STEP_DAYS,
)

TARGET_GPI     = 'GPI_90d'
TARGET_DECLINE = 'decline_prob_180d'
TARGET_STRESS  = 'dominant_stress'

_META_COLS = {
    'pixel_id', 'SITE', 'latitude', 'longitude',
    'window_center', 'valid_obs_count', 'wc_rounded',
    TARGET_GPI, TARGET_DECLINE, TARGET_STRESS,
    'frac_heat', 'frac_cold', 'frac_light', 'frac_senescence',
    'shap_sst', 'shap_par',
}


# ---------------------------------------------------------------------------
# Load & merge
# ---------------------------------------------------------------------------

def load_and_merge() -> pd.DataFrame:
    print(f'Loading features: {FORILLON_FEATURES}')
    s2 = pd.read_csv(FORILLON_FEATURES)
    s2['window_center'] = pd.to_datetime(s2['window_center'])

    print(f'Loading labels:   {TEACHER_LABELS_CSV}')
    lbl = pd.read_csv(TEACHER_LABELS_CSV)
    lbl['window_center'] = pd.to_datetime(lbl['window_center'])

    s2['wc_rounded']  = s2['window_center'].dt.round(f'{STEP_DAYS}D')
    lbl['wc_rounded'] = lbl['window_center'].dt.round(f'{STEP_DAYS}D')

    merged = pd.merge(
        s2, lbl.drop(columns=['window_center'], errors='ignore'),
        on=['SITE', 'wc_rounded'], how='inner',
    )
    stress_cols = ['frac_heat', 'frac_cold', 'frac_light', 'frac_senescence']
    stress_map  = dict(zip(stress_cols, ['heat', 'cold', 'light', 'senescence']))
    merged[TARGET_STRESS] = merged[stress_cols].idxmax(axis=1).map(stress_map)

    print(f'  merged rows = {len(merged):,}')
    return merged


# ---------------------------------------------------------------------------
# Features & splits
# ---------------------------------------------------------------------------

def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in _META_COLS
            and df[c].dtype in (np.float64, np.int64, float, int)]


def drop_correlated(X: pd.DataFrame, threshold: float = CORR_DROP_THRESHOLD):
    corr  = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    drop  = [c for c in upper.columns if (upper[c] > threshold).any()]
    return X.drop(columns=drop), drop


def block_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    test_mask  = df['window_center'].dt.year == TEMPORAL_TEST_YR
    val_mask   = (df['SITE'] == SPATIAL_VAL_SITE) & ~test_mask
    train_mask = ~df['SITE'].isin([SPATIAL_VAL_SITE]) & ~test_mask
    return {
        'train':          df[train_mask].reset_index(drop=True),
        'val_spatial':    df[val_mask].reset_index(drop=True),
        'test_temporal':  df[test_mask].reset_index(drop=True),
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def _prep(split: pd.DataFrame, target: str, feat: list[str]):
    sub = split.dropna(subset=[target])
    return sub[feat].fillna(0), sub[target]


def train_gpi(splits, feat) -> tuple[xgb.XGBRegressor, dict]:
    print('\n=== Model 1: GPI Regression ===')
    Xtr, ytr = _prep(splits['train'], TARGET_GPI, feat)
    Xvl, yvl = _prep(splits['val_spatial'], TARGET_GPI, feat)
    Xte, yte = _prep(splits['test_temporal'], TARGET_GPI, feat)

    model = xgb.XGBRegressor(
        **XGB_PARAMS, early_stopping_rounds=30, eval_metric='mae',
    )
    model.fit(Xtr, ytr, eval_set=[(Xvl, yvl)], verbose=False)

    metrics = {}
    for name, X, y in [('val_spatial', Xvl, yvl), ('test_temporal', Xte, yte)]:
        if not len(X): continue
        p = model.predict(X)
        metrics[name] = {'mae': float(mean_absolute_error(y, p)),
                          'r2':  float(r2_score(y, p))}
        print(f'  {name}: MAE={metrics[name]["mae"]:.4f}  R²={metrics[name]["r2"]:.4f}')
    return model, metrics


def train_decline(splits, feat) -> tuple[xgb.XGBRegressor, dict]:
    print('\n=== Model 2: Decline Probability Regression ===')
    Xtr, ytr = _prep(splits['train'], TARGET_DECLINE, feat)
    Xvl, yvl = _prep(splits['val_spatial'], TARGET_DECLINE, feat)
    Xte, yte = _prep(splits['test_temporal'], TARGET_DECLINE, feat)

    model = xgb.XGBRegressor(
        **XGB_PARAMS, early_stopping_rounds=30, eval_metric='mae',
    )
    model.fit(Xtr, ytr, eval_set=[(Xvl, yvl)], verbose=False)

    metrics = {}
    for name, X, y in [('val_spatial', Xvl, yvl), ('test_temporal', Xte, yte)]:
        if not len(X): continue
        p = np.clip(model.predict(X), 0, 1)
        y_bin = (y > 0.5).astype(int)
        m = {
            'mae': float(mean_absolute_error(y, p)),
            'brier': float(brier_score_loss(y_bin, p)) if y_bin.nunique() > 1 else None,
            'auc':   float(roc_auc_score(y_bin, p))    if y_bin.nunique() > 1 else None,
        }
        metrics[name] = m
        brier_str = f'{m["brier"]:.4f}' if m["brier"] is not None else 'n/a'
        auc_str   = f'{m["auc"]:.4f}'   if m["auc"]   is not None else 'n/a'
        print(f'  {name}: MAE={m["mae"]:.4f}  Brier={brier_str}  AUC={auc_str}')
    return model, metrics


def train_stress(splits, feat):
    print('\n=== Model 3: Stress Regime Classification ===')
    enc = LabelEncoder()
    all_labels = pd.concat([splits[k][TARGET_STRESS] for k in splits]).dropna()
    enc.fit(all_labels)

    def prep_cls(split):
        sub = split.dropna(subset=[TARGET_STRESS])
        return sub[feat].fillna(0), enc.transform(sub[TARGET_STRESS])

    Xtr, ytr = prep_cls(splits['train'])
    Xvl, yvl = prep_cls(splits['val_spatial'])
    Xte, yte = prep_cls(splits['test_temporal'])

    model = xgb.XGBClassifier(
        **XGB_PARAMS, num_class=len(enc.classes_),
        objective='multi:softprob',
        early_stopping_rounds=30, eval_metric='mlogloss',
    )
    model.fit(Xtr, ytr, eval_set=[(Xvl, yvl)], verbose=False)

    metrics = {'classes': enc.classes_.tolist()}
    for name, X, y in [('val_spatial', Xvl, yvl), ('test_temporal', Xte, yte)]:
        if not len(X): continue
        p = model.predict(X)
        macro = f1_score(y, p, average='macro', zero_division=0)
        per   = f1_score(y, p, average=None, zero_division=0)
        metrics[name] = {
            'macro_f1': float(macro),
            'per_class_f1': {cls: float(s) for cls, s in zip(enc.classes_, per)},
        }
        print(f'  {name}: Macro-F1={macro:.4f}')
        for cls, score in zip(enc.classes_, per):
            print(f'    {cls}: {score:.4f}')
    return model, enc, metrics


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def shap_table(model, X_sample, feat, tag: str) -> pd.DataFrame:
    print(f'SHAP: {tag}')
    sv = shap.TreeExplainer(model).shap_values(X_sample)
    if isinstance(sv, list):
        sv = np.stack(sv).mean(axis=0)
    imp = pd.DataFrame({
        'feature': feat,
        'mean_abs_shap': np.abs(sv).mean(axis=0),
    }).sort_values('mean_abs_shap', ascending=False)
    out = MODELS_DIR / f'shap_{tag}.csv'
    imp.to_csv(out, index=False)
    print(imp.head(10).to_string(index=False))
    return imp


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    merged = load_and_merge()
    feat   = feature_cols(merged)
    print(f'Features: {len(feat)}')

    X_red, dropped = drop_correlated(merged[feat].fillna(0))
    feat = list(X_red.columns)
    print(f'After |r|<{CORR_DROP_THRESHOLD} pruning: {len(feat)}  '
          f'(dropped {len(dropped)})')

    splits = block_split(merged)
    for k, v in splits.items():
        print(f'  {k}: {len(v)}')

    gpi_model,     gpi_metrics     = train_gpi(splits, feat)
    decline_model, decline_metrics = train_decline(splits, feat)
    stress_model, stress_enc, stress_metrics = train_stress(splits, feat)

    X_sample = splits['train'][feat].fillna(0).sample(
        min(1000, len(splits['train'])), random_state=42)
    shap_table(gpi_model,     X_sample, feat, 'gpi')
    shap_table(decline_model, X_sample, feat, 'decline')

    artifacts = {
        'gpi_model.pkl':     gpi_model,
        'decline_model.pkl': decline_model,
        'stress_model.pkl':  (stress_model, stress_enc),
        'feature_cols.pkl':  feat,
        'train_X_mean.pkl':  X_sample.mean().values,
    }
    for fn, obj in artifacts.items():
        with open(MODELS_DIR / fn, 'wb') as f:
            pickle.dump(obj, f)

    lw = LedoitWolf().fit(splits['train'][feat].fillna(0))
    with open(MODELS_DIR / 'train_cov_inv.pkl', 'wb') as f:
        pickle.dump(np.linalg.inv(lw.covariance_), f)

    with open(MODELS_DIR / 'validation_report.json', 'w') as f:
        json.dump({
            'features_kept': feat,
            'features_dropped': dropped,
            'gpi':     gpi_metrics,
            'decline': decline_metrics,
            'stress':  stress_metrics,
        }, f, indent=2, default=str)

    print(f'\n→ models + report saved to {MODELS_DIR}/')
