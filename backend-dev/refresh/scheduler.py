"""
refresh.scheduler
-----------------
Cron-friendly refresh loop. For every inference region registered in
regions.json:

  1. Look for a fresh raw GEE CSV (data/raw/<region>_s2_pixels.csv).
  2. Rebuild the rolling feature matrix.
  3. Run pixel + bed inference.

This script assumes the GEE-side scheduled export has already produced the
raw CSV in data/raw/ (mount your Google Drive 'EelgrassEEWS' folder or copy
via rclone — see README).

Cron example (every 10 days at 04:00):
    0 4 */10 * *  cd /path/to/eelsSaver_Backend \
        && python -m refresh.scheduler >> data/refresh.log 2>&1
"""

from __future__ import annotations
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from config import (
    BASE_DIR, RAW_DIR, FEATURES_DIR, PREDICTIONS_DIR,
    REGIONS_FILE, REFRESH_INTERVAL_DAYS,
)


def _log(msg: str):
    print(f'[{datetime.now().isoformat(timespec="seconds")}] {msg}', flush=True)


def _run(label: str, cmd: list[str]) -> bool:
    _log(f'$ {" ".join(cmd)}')
    proc = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True)
    if proc.returncode != 0:
        _log(f'✗ {label} failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}')
        return False
    _log(f'✓ {label} ok')
    return True


def refresh_region(region_id: str) -> bool:
    raw      = RAW_DIR / f'{region_id}_s2_pixels.csv'
    features = FEATURES_DIR / f'{region_id}_features.csv'

    if not raw.exists():
        _log(f'… skip {region_id} — no raw CSV at {raw}')
        return False

    if not _run(f'{region_id}: s2_features',
                ['python', '-m', 'pipeline.s2_features', str(raw), str(features)]):
        return False

    if not _run(f'{region_id}: inference',
                ['python', '-m', 'pipeline.inference', str(features), region_id]):
        return False

    return True


def main():
    if not REGIONS_FILE.exists():
        sys.exit(f'Missing {REGIONS_FILE}')

    regions = json.loads(REGIONS_FILE.read_text())['regions']
    inference_regions = [r['id'] for r in regions if r['role'] == 'inference']

    _log(f'Refresh cycle (interval = {REFRESH_INTERVAL_DAYS}d). '
         f'Regions: {inference_regions}')

    succeeded, failed = [], []
    for rid in inference_regions:
        if refresh_region(rid):
            succeeded.append(rid)
        else:
            failed.append(rid)

    _log(f'Done. ok={succeeded} failed={failed}')

    # Write a sentinel file so the API can surface "last refresh" time
    (PREDICTIONS_DIR / 'last_refresh.txt').write_text(
        datetime.now().isoformat(timespec='seconds')
    )


if __name__ == '__main__':
    main()
