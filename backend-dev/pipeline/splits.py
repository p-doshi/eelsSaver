"""
pipeline.splits
---------------
Train/val/test split strategies for the eelsSaver student.

The Ouest dataset has only 6 annual surveys (2013-2018) with a documented
boom-bust-recovery-crash cycle and a damage epidemic in 2017-2018. That tiny
sample size + strong temporal structure makes the choice of split crucial.

Three strategies are provided:

  1. LOYO  — Leave-one-year-out CV. 6 folds, each year held out exactly once.
              Best for *robust* generalisation estimates. Default for
              hyperparameter tuning and reporting "average" performance.

  2. FORWARD — Forward-chaining time-series split (a.k.a. expanding window).
                Fold k: train [2013 … year_k], test [year_k+1].
                Best for "would this model have warned us next year?" — the
                actual deployment use-case.

  3. TAIL    — Single train/val/test split honouring the cycle:
                  train: 2013-2015   (boom / bust)
                  val:   2016        (recovery)
                  test:  2017-2018   (damage epidemic)
                Best for the headline "did we predict the epidemic?" number.

All three return an iterable of `(train_idx, val_idx, test_idx)` tuples so
the calling code is identical regardless of strategy.

Usage:
    from pipeline.splits import build_splits
    for fold, (tr, vl, te) in enumerate(build_splits(df, strategy='loyo')):
        ...

Each row in `df` must carry a `year` column (int). For pixel-level data,
that year is the calendar year the 90-day rolling window centres on.
"""

from __future__ import annotations
from typing import Iterator
import numpy as np
import pandas as pd


YEARS = [2013, 2014, 2015, 2016, 2017, 2018]


# ---------------------------------------------------------------------------
# 1. Leave-one-year-out CV
# ---------------------------------------------------------------------------

def loyo_splits(df: pd.DataFrame,
                 val_from_train_frac: float = 0.2) -> Iterator[tuple]:
    """
    6 folds. Each fold holds one calendar year as test.
    A random `val_from_train_frac` of the remaining 5 training years is
    used as the validation set for early stopping (sampled by *year*, not
    rows, to avoid leaking adjacent observations).

    Yields:
        (train_idx, val_idx, test_idx)
    """
    rng = np.random.default_rng(42)
    years = sorted(df['year'].unique().tolist())

    for held_year in years:
        train_years = [y for y in years if y != held_year]
        # Pick ~1 validation year from the training set (≈ 20%)
        n_val = max(1, int(round(len(train_years) * val_from_train_frac)))
        # Use the latest training year as val (chronological), so we don't
        # leak future info into training. Seed is for reproducibility if we
        # ever switch to random.
        _ = rng.integers(0, 1000)
        val_years = train_years[-n_val:]
        true_train_years = [y for y in train_years if y not in val_years]

        train_idx = df.index[df['year'].isin(true_train_years)].to_numpy()
        val_idx   = df.index[df['year'].isin(val_years)].to_numpy()
        test_idx  = df.index[df['year'] == held_year].to_numpy()
        yield train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# 2. Forward-chaining expanding window
# ---------------------------------------------------------------------------

def forward_chain_splits(df: pd.DataFrame) -> Iterator[tuple]:
    """
    Fold k: train [years 0..k-1], val [year k-1] (last training year),
    test [year k]. Years are sorted ascending. Fold 0 is skipped because
    it would have an empty training set.

    For the standard 2013-2018 Ouest dataset this yields 5 folds:
        train [2013],            val [2013], test 2014
        train [2013-2014],       val [2014], test 2015
        train [2013-2015],       val [2015], test 2016
        train [2013-2016],       val [2016], test 2017
        train [2013-2017],       val [2017], test 2018
    """
    years = sorted(df['year'].unique().tolist())
    for k in range(1, len(years)):
        train_years = years[:k]
        val_year    = years[k - 1]
        test_year   = years[k]

        train_idx = df.index[df['year'].isin(train_years)].to_numpy()
        val_idx   = df.index[df['year'] == val_year].to_numpy()
        test_idx  = df.index[df['year'] == test_year].to_numpy()
        yield train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# 3. Tail split — honours the documented ecological cycle
# ---------------------------------------------------------------------------

def tail_split(df: pd.DataFrame,
                train_years: tuple = (2013, 2014, 2015),
                val_years:   tuple = (2016,),
                test_years:  tuple = (2017, 2018)) -> Iterator[tuple]:
    """Single-fold train/val/test honouring the boom-bust-recovery-crash cycle.

    If the documented (2013-15, 2016, 2017-18) split has no overlap with the
    df's actual years (e.g. Sentinel-2 archive only covers 2016+), falls back
    to splitting whatever years ARE available: earliest year(s) → train,
    middle → val, latest → test. Prints a warning so the user knows.
    """
    available = sorted(df['year'].unique().tolist())
    train_overlap = set(available) & set(train_years)

    if not train_overlap:
        # Documented train years don't exist in the data — auto-fallback
        if len(available) < 3:
            print(f'[tail_split] WARNING: only {len(available)} year(s) available '
                  f'{available}; tail split needs ≥ 3. Yielding empty fold.')
            yield (np.array([], dtype=int),) * 3
            return
        # Earliest year(s) → train, next → val, latest year(s) → test
        # For 3 years: 1/1/1. For ≥4: half/1/half.
        n = len(available)
        n_test  = max(1, n // 3)
        n_val   = 1
        n_train = n - n_val - n_test
        train_years = tuple(available[:n_train])
        val_years   = tuple(available[n_train:n_train + n_val])
        test_years  = tuple(available[n_train + n_val:])
        print(f'[tail_split] Documented 2013-18 split unavailable. '
              f'Using fallback for years {available}:\n'
              f'  train={list(train_years)}  val={list(val_years)}  test={list(test_years)}')

    train_idx = df.index[df['year'].isin(train_years)].to_numpy()
    val_idx   = df.index[df['year'].isin(val_years)].to_numpy()
    test_idx  = df.index[df['year'].isin(test_years)].to_numpy()
    yield train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_splits(df: pd.DataFrame,
                  strategy: str = 'loyo',
                  **kwargs) -> Iterator[tuple]:
    """
    Dispatch by name.

    Parameters
    ----------
    df : DataFrame containing a `year` integer column.
    strategy : 'loyo' | 'forward' | 'tail'
    """
    if 'year' not in df.columns:
        raise ValueError("`df` must contain a `year` column (int).")

    s = strategy.lower()
    if s == 'loyo':
        return loyo_splits(df, **kwargs)
    if s == 'forward':
        return forward_chain_splits(df, **kwargs)
    if s == 'tail':
        return tail_split(df, **kwargs)
    raise ValueError(f"Unknown split strategy '{strategy}'. "
                      f"Choose loyo / forward / tail.")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def describe(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Tabular fold/size summary — useful for sanity-checking before training."""
    rows = []
    for fold, (tr, vl, te) in enumerate(build_splits(df, strategy=strategy)):
        rows.append({
            'fold':       fold,
            'train_n':    len(tr),
            'val_n':      len(vl),
            'test_n':     len(te),
            'train_years': sorted(df.loc[tr, 'year'].unique().tolist()),
            'val_years':   sorted(df.loc[vl, 'year'].unique().tolist()),
            'test_years':  sorted(df.loc[te, 'year'].unique().tolist()),
        })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    # Demo with a fake dataset
    import sys
    if len(sys.argv) > 1:
        df = pd.read_csv(sys.argv[1])
        if 'year' not in df.columns and 'window_center' in df.columns:
            df['year'] = pd.to_datetime(df['window_center']).dt.year
    else:
        df = pd.DataFrame({
            'year': np.repeat(YEARS, 36),
            'idx':  np.arange(6 * 36),
        })

    for strat in ['loyo', 'forward', 'tail']:
        print(f'\n=== {strat.upper()} ===')
        print(describe(df, strat).to_string(index=False))
