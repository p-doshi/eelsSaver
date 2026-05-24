"""
pipeline.run_all
----------------
Orchestrates the training pipeline:
    1. s2_features.py  (build pixel × window matrix for Forillon)
    2. teacher.py      (generate soft labels)
    3. student.py      (train XGBoost models)

Inference and calibration are per-region and run separately.

Usage:
    python -m pipeline.run_all
"""

import subprocess
import sys
from pathlib import Path

from config import FORILLON_RAW_CSV, FORILLON_FEATURES, TEACHER_LABELS_CSV, MODELS_DIR


def step(name: str, cmd: list[str]):
    print(f'\n{"=" * 60}\n{name}\n{"=" * 60}')
    print('  $', ' '.join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        sys.exit(f'  ✗ {name} failed (exit {result.returncode}).')
    print(f'  ✓ {name} done.')


def main():
    if not Path(FORILLON_RAW_CSV).exists():
        sys.exit(f'Missing GEE export: {FORILLON_RAW_CSV}\n'
                 f'Run gee/forillon_extraction.js first and place the export there.')

    step('1/3 s2_features',
         ['python', '-m', 'pipeline.s2_features',
          str(FORILLON_RAW_CSV), str(FORILLON_FEATURES)])

    step('2/3 teacher',
         ['python', '-m', 'pipeline.teacher'])

    step('3/3 student',
         ['python', '-m', 'pipeline.student'])

    print('\nAll training stages complete.')
    print(f'Models in: {MODELS_DIR}')
    print('Next: run pipeline.inference per target region.')


if __name__ == '__main__':
    main()
