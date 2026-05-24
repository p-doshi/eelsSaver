"""
pipeline.calibration
--------------------
Optional isotonic-regression calibration on top of student predictions, fit
on a small ground-truth dataset from a target region (e.g. CERI restoration
monitoring or DFO PEI surveys).

Inputs:
    ground_truth.csv with columns:
        bed_id, window_center, observed_coverage  (0-100)
        observed_decline_flag  (0/1, optional)

The script joins observed coverage to the student's predicted_decline_prob
and fits an isotonic regression mapping. The calibrator pickle is consumed
by inference.py automatically when present.

Usage:
    python -m pipeline.calibration <ground_truth_csv> <region_id>
"""

from __future__ import annotations
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from config import MODELS_DIR, PREDICTIONS_DIR


# ---------------------------------------------------------------------------
# Calibrator class (saved + loaded by inference.py)
# ---------------------------------------------------------------------------

class DeclineCalibrator:
    """Isotonic remap of raw decline_prob → calibrated decline_prob."""

    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds='clip',
                                       y_min=0.0, y_max=1.0)
        self.n_samples = 0

    def fit(self, raw_probs: np.ndarray, observed_flag: np.ndarray):
        self.iso.fit(raw_probs, observed_flag)
        self.n_samples = len(raw_probs)
        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        return self.iso.transform(raw_probs)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage: python -m pipeline.calibration <ground_truth_csv> <region_id>')
        sys.exit(1)

    gt_csv    = sys.argv[1]
    region_id = sys.argv[2]

    print(f'Loading ground truth: {gt_csv}')
    gt = pd.read_csv(gt_csv)
    gt['window_center'] = pd.to_datetime(gt['window_center'])

    pred_csv = PREDICTIONS_DIR / f'{region_id}_beds.csv'
    print(f'Loading bed predictions: {pred_csv}')
    preds = pd.read_csv(pred_csv)
    preds['window_center'] = pd.to_datetime(preds['window_center'])

    merged = pd.merge(gt, preds, on=['bed_id', 'window_center'], how='inner')
    if len(merged) < 5:
        print(f'  Only {len(merged)} matched rows — too few to calibrate.')
        sys.exit(1)

    # Derive a decline flag if not present: observed coverage dropped to <50%
    # of region max.
    if 'observed_decline_flag' not in merged:
        cap = merged.groupby('bed_id')['observed_coverage'].transform('max')
        merged['observed_decline_flag'] = (merged['observed_coverage'] < 0.5 * cap).astype(int)

    raw  = merged['bed_decline_prob'].values
    obs  = merged['observed_decline_flag'].values

    cal = DeclineCalibrator().fit(raw, obs)
    out = MODELS_DIR / f'calibrator_{region_id}.pkl'
    with open(out, 'wb') as f:
        pickle.dump(cal, f)

    # Quick summary
    print(f'\nFit on {len(merged)} matched points.')
    print(f'Raw  decline mean = {raw.mean():.3f}')
    print(f'Calib mean        = {cal.transform(raw).mean():.3f}')
    print(f'→ {out}')
