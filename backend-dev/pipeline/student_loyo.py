"""
pipeline.student_loyo
---------------------
Train the eelsSaver student with year-aware splits suited to the small,
strongly-seasonal Ouest dataset (6 annual surveys, 2013-2018).

Three strategies:
    loyo     — Leave-one-year-out CV  (6 folds; primary reporting metric)
    forward  — Forward-chaining expanding window (5 folds; deployment realism)
    tail     — Train 2013-2015 / val 2016 / test 2017-2018 (one fold; epidemic)

For each fold:
  • Trains all three students (GPI, decline flag, stress class).
  • Records per-fold metrics.
  • Aggregates with mean ± std across folds.

Output:
  data/models/student_<strategy>/
      fold_<k>_gpi.pkl, fold_<k>_decline.pkl, fold_<k>_stress.pkl
      cv_report.json    (per-fold + aggregate metrics)
      cv_summary.csv    (long-form metrics table)

Usage:
    python -m pipeline.student_loyo --strategy loyo
    python -m pipeline.student_loyo --strategy forward
    python -m pipeline.student_loyo --strategy tail
"""

from __future__ import annotations
import argparse
import json
import pickle
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error, r2_score, brier_score_loss,
    roc_auc_score, f1_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.covariance import LedoitWolf

from config import (
    FORILLON_FEATURES, MODELS_DIR, XGB_PARAMS, CORR_DROP_THRESHOLD, LABELS_DIR,
)
from pipeline.splits import build_splits, describe


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TEACHER_OUEST_CSV = LABELS_DIR / 'teacher_labels_ouest.csv'

TARGET_GPI     = 'GPI_annual'
TARGET_DECLINE = 'decline_flag'      # 0/1, NaN for last year
TARGET_STRESS  = 'dominant_stress'

# Meta columns that must not become features
_META_COLS = {
    'pixel_id', 'SITE', 'site_id', 'latitude', 'longitude',
    'window_center', 'valid_obs_count', 'wc_rounded', 'year',
    'cover', 'damage', 'height',
    TARGET_GPI, TARGET_DECLINE, TARGET_STRESS,
    'shap_sst', 'shap_par',
}


# ---------------------------------------------------------------------------
# Load + merge
# ---------------------------------------------------------------------------

def load() -> pd.DataFrame:
    if not TEACHER_OUEST_CSV.exists():
        raise SystemExit(
            f'Missing {TEACHER_OUEST_CSV}. Run:\n'
            f'  python -m pipeline.teacher_ouest <ouest_eelgrass_environmental_complete.csv>'
        )
    if not FORILLON_FEATURES.exists():
        raise SystemExit(
            f'Missing {FORILLON_FEATURES}. Run pipeline.s2_features '
            f'on the GEE export first.'
        )

    print(f'Loading {FORILLON_FEATURES.name}')
    feats = pd.read_csv(FORILLON_FEATURES, parse_dates=['window_center'])
    feats['year'] = feats['window_center'].dt.year

    print(f'Loading {TEACHER_OUEST_CSV.name}')
    lbl = pd.read_csv(TEACHER_OUEST_CSV)

    # Single-site merge — Ouest only (the new dataset is single-site)
    feats = feats[feats['SITE'].str.lower() == 'ouest'].copy()
    print(f'  pixel-window rows for Ouest: {len(feats):,}')

    merged = pd.merge(feats, lbl, on='year', how='inner')
    print(f'  merged rows: {len(merged):,}')

    if not len(merged):
        raise SystemExit(
            'Empty merge — check that years in S2 features overlap with '
            'years in teacher_labels_ouest.csv (need 2013-2018).'
        )
    return merged


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in _META_COLS
            and df[c].dtype in (np.float64, np.int64, float, int)]


def drop_correlated(X: pd.DataFrame, threshold: float = CORR_DROP_THRESHOLD):
    corr  = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    drop  = [c for c in upper.columns if (upper[c] > threshold).any()]
    return X.drop(columns=drop), drop


# ---------------------------------------------------------------------------
# Per-fold training
# ---------------------------------------------------------------------------

def _train_regressor(target: str, tr, vl, te, feat, df,
                      clip: tuple | None = None) -> dict:
    sub_tr = df.loc[tr].dropna(subset=[target])
    sub_vl = df.loc[vl].dropna(subset=[target])
    sub_te = df.loc[te].dropna(subset=[target])
    if len(sub_tr) < 10 or len(sub_te) == 0:
        return {'skipped': True}

    Xtr = sub_tr[feat].fillna(0); ytr = sub_tr[target].values
    Xvl = sub_vl[feat].fillna(0); yvl = sub_vl[target].values
    Xte = sub_te[feat].fillna(0); yte = sub_te[target].values

    model = xgb.XGBRegressor(
        **XGB_PARAMS, early_stopping_rounds=30, eval_metric='mae',
    )
    fit_kwargs = {'verbose': False}
    if len(Xvl) > 0:
        fit_kwargs['eval_set'] = [(Xvl, yvl)]
    model.fit(Xtr, ytr, **fit_kwargs)

    p_te = model.predict(Xte)
    if clip is not None:
        p_te = np.clip(p_te, *clip)
    out = {
        'mae':   float(mean_absolute_error(yte, p_te)),
        'r2':    float(r2_score(yte, p_te)) if yte.std() > 0 else None,
    }
    if clip == (0, 1) and len(np.unique(yte)) > 1:
        try:
            out['brier'] = float(brier_score_loss(yte.astype(int), p_te))
            out['auc']   = float(roc_auc_score(yte.astype(int), p_te))
        except Exception:
            pass
    return {'metrics': out, 'model': model}


def _train_stress(tr, vl, te, feat, df, enc: LabelEncoder) -> dict:
    sub_tr = df.loc[tr].dropna(subset=[TARGET_STRESS])
    sub_vl = df.loc[vl].dropna(subset=[TARGET_STRESS])
    sub_te = df.loc[te].dropna(subset=[TARGET_STRESS])
    if len(sub_tr) < 10 or len(sub_te) == 0:
        return {'skipped': True}

    Xtr = sub_tr[feat].fillna(0); ytr = enc.transform(sub_tr[TARGET_STRESS])
    Xvl = sub_vl[feat].fillna(0); yvl = enc.transform(sub_vl[TARGET_STRESS])
    Xte = sub_te[feat].fillna(0); yte = enc.transform(sub_te[TARGET_STRESS])

    model = xgb.XGBClassifier(
        **XGB_PARAMS, num_class=len(enc.classes_),
        objective='multi:softprob', early_stopping_rounds=30,
        eval_metric='mlogloss',
    )
    fit_kwargs = {'verbose': False}
    if len(Xvl) > 0:
        fit_kwargs['eval_set'] = [(Xvl, yvl)]
    model.fit(Xtr, ytr, **fit_kwargs)

    p_te = model.predict(Xte)
    return {
        'metrics': {
            'macro_f1': float(f1_score(yte, p_te, average='macro', zero_division=0)),
            'classes':  enc.classes_.tolist(),
        },
        'model': model,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(strategy: str):
    merged = load()

    feat = feature_cols(merged)
    print(f'\nFeature count before pruning: {len(feat)}')
    X_red, dropped = drop_correlated(merged[feat].fillna(0))
    feat = list(X_red.columns)
    print(f'After |r|<{CORR_DROP_THRESHOLD} pruning: {len(feat)} '
          f'(dropped {len(dropped)})')

    print(f'\nSplit strategy: {strategy.upper()}')
    print(describe(merged, strategy).to_string(index=False))

    out_dir = MODELS_DIR / f'student_{strategy}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fit one LabelEncoder across all labels so stress classes are consistent
    enc = LabelEncoder().fit(merged[TARGET_STRESS].dropna())

    folds = []
    for k, (tr, vl, te) in enumerate(build_splits(merged, strategy=strategy)):
        train_years = sorted(merged.loc[tr, 'year'].unique().tolist())
        test_years  = sorted(merged.loc[te, 'year'].unique().tolist())
        print(f'\n--- Fold {k}: train={train_years}  test={test_years} '
              f'(train_n={len(tr)}, val_n={len(vl)}, test_n={len(te)}) ---')

        gpi_res     = _train_regressor(TARGET_GPI,     tr, vl, te, feat, merged)
        decline_res = _train_regressor(TARGET_DECLINE, tr, vl, te, feat, merged,
                                        clip=(0, 1))
        stress_res  = _train_stress(tr, vl, te, feat, merged, enc)

        fold_record = {
            'fold': k,
            'train_years': train_years,
            'test_years':  test_years,
            'gpi':     gpi_res.get('metrics', {'skipped': True}),
            'decline': decline_res.get('metrics', {'skipped': True}),
            'stress':  stress_res.get('metrics', {'skipped': True}),
        }
        folds.append(fold_record)
        print('  GPI:    ', fold_record['gpi'])
        print('  Decline:', fold_record['decline'])
        print('  Stress: ', fold_record['stress'])

        # Persist per-fold artifacts
        for tag, res in [('gpi', gpi_res),
                          ('decline', decline_res),
                          ('stress', stress_res)]:
            if 'model' in res:
                with open(out_dir / f'fold_{k}_{tag}.pkl', 'wb') as f:
                    pickle.dump(res['model'], f)

    # Aggregate metrics
    def agg(metric_path):
        vals = []
        for f in folds:
            d = f
            for p in metric_path:
                d = d.get(p) if isinstance(d, dict) else None
                if d is None:
                    break
            if isinstance(d, (int, float)) and not np.isnan(d):
                vals.append(float(d))
        if not vals:
            return {'mean': None, 'std': None, 'n': 0}
        return {'mean': float(np.mean(vals)),
                'std':  float(np.std(vals)),
                'n':    len(vals)}

    report = {
        'strategy':       strategy,
        'feature_count':  len(feat),
        'features_kept':  feat,
        'features_dropped': dropped,
        'folds':          folds,
        'aggregate': {
            'gpi_mae':        agg(['gpi', 'mae']),
            'gpi_r2':         agg(['gpi', 'r2']),
            'decline_mae':    agg(['decline', 'mae']),
            'decline_brier':  agg(['decline', 'brier']),
            'decline_auc':    agg(['decline', 'auc']),
            'stress_macro_f1':agg(['stress', 'macro_f1']),
        },
    }

    with open(out_dir / 'cv_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    # Long-form metrics CSV
    rows = []
    for f in folds:
        for model_tag in ['gpi', 'decline', 'stress']:
            for metric_name, value in (f[model_tag] or {}).items():
                if isinstance(value, (int, float)):
                    rows.append({
                        'fold': f['fold'],
                        'model': model_tag,
                        'metric': metric_name,
                        'value': value,
                        'test_years': str(f['test_years']),
                    })
    pd.DataFrame(rows).to_csv(out_dir / 'cv_summary.csv', index=False)

    print(f'\n=== Aggregate ({strategy}, {len(folds)} folds) ===')
    for k, v in report['aggregate'].items():
        if v['mean'] is None:
            print(f'  {k}: n/a')
        else:
            print(f'  {k}: {v["mean"]:.4f} ± {v["std"]:.4f}  (n={v["n"]})')
    print(f'\nSaved to {out_dir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', '-s',
                         choices=['loyo', 'forward', 'tail'],
                         default='loyo',
                         help='Train/val/test split strategy')
    args = parser.parse_args()
    main(args.strategy)
