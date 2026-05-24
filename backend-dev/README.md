# eelsSaver Backend

Pan-Atlantic eelgrass early-warning system. Trains a Sentinel-2 student model on Forillon
in-situ + teacher soft labels, then runs pixel-level inference on any Atlantic Canada
coastal polygon to score eelgrass beds for decline risk, GPI, and dominant stressor.

## Architecture

```
                           ┌─────────────────────────────┐
   Forillon in-situ  ─────▶│ teacher.py                  │
   (Eelgrass_Forillon,     │  • GPI_90d                  │
   SST, PAR)               │  • decline_prob_180d        │──┐
                           │  • stress_regime (4-vec)    │  │
                           │  • SHAP driver importance   │  │
                           └─────────────────────────────┘  │
                                                            │
   Forillon polygons       ┌─────────────────────────────┐  │  teacher_labels.csv
   ───────────────────────▶│ GEE: forillon_extraction.js │  │
   (Ouest, Marais, Sud)    │   9 bands + 10 indices      │  │
                           └──────────────┬──────────────┘  │
                                          │                 │
                                          ▼                 │
                           ┌─────────────────────────────┐  │
                           │ s2_features.py              │  │
                           │   90-day rolling stats      │  │
                           └──────────────┬──────────────┘  │
                                          │                 │
                                          ▼                 ▼
                           ┌──────────────────────────────────────────┐
                           │ student.py                               │
                           │   3 XGBoost models (GPI, decline, stress)│
                           │   spatial + temporal block validation    │
                           └──────────────┬───────────────────────────┘
                                          │
                                          ▼
   Pan-Atlantic polygons   ┌─────────────────────────────┐
   (Antigonish, Malpeque,  │ GEE: pan_atlantic_template  │
   Bras d'Or, Pictou ...) ▶│  same bands/indices         │
                           └──────────────┬──────────────┘
                                          │
                                          ▼
                           ┌─────────────────────────────┐
                           │ inference.py                │
                           │   pixel scoring             │
                           │   bed aggregation           │
                           │   Mahalanobis OOD detection │
                           │   confidence tiers          │
                           └──────────────┬──────────────┘
                                          │
                                          ▼
                           ┌─────────────────────────────┐
                           │ FastAPI (api/main.py)       │
                           │   /api/beds, /api/regions   │
                           │   /api/beds/{id}/pixels     │
                           │   /api/beds/{id}/timeseries │
                           └─────────────────────────────┘
                                          │
                                          ▼
                                  Leaflet web map
                                  (web/index.html)
```

## Quickstart

```bash
# 0. Install
pip install -r requirements.txt

# 1. (One-time) Run GEE extraction in https://code.earthengine.google.com
#    Paste gee/forillon_extraction.js → Run → wait for Drive export
#    Download exported CSV to data/raw/forillon_s2_pixels.csv

# 2. Build student feature matrix (90-day rolling stats per pixel)
python -m pipeline.s2_features data/raw/forillon_s2_pixels.csv data/features/s2_pixel_features.csv

# 3. Generate teacher soft labels from Forillon in-situ data
python -m pipeline.teacher

# 4. Train the 3 XGBoost students with spatial + temporal blocking
python -m pipeline.student

# 5. Run pan-Atlantic inference on a target region
#    First: run gee/pan_atlantic_template.js with your polygon, export CSV.
#    Then:
python -m pipeline.s2_features data/raw/antigonish_pixels.csv data/features/antigonish_features.csv
python -m pipeline.inference data/features/antigonish_features.csv antigonish

# 6. Serve predictions via API
uvicorn api.main:app --reload --port 8000

# 7. Open the map
open web/index.html   # points to http://localhost:8000/api
```

## Directory layout

| Path                      | Contents                                                            |
| ------------------------- | ------------------------------------------------------------------- |
| `config.py`               | Hyperparameters, paths, ecological thresholds, region registry.     |
| `regions.json`            | Target-region polygons (lat/lon arrays).                            |
| `gee/`                    | GEE JavaScript files to run in the Code Editor.                     |
| `pipeline/`               | Python pipeline modules (one per stage).                            |
| `api/`                    | FastAPI app exposing predictions to the web map.                    |
| `refresh/`                | Cron-style scheduler that re-runs inference every 10 days.          |
| `web/`                    | Leaflet map (static HTML/JS).                                       |
| `data/raw/`               | Raw GEE-exported CSVs.                                              |
| `data/features/`          | Output of `s2_features.py` — pixel × window feature matrices.       |
| `data/labels/`            | Output of `teacher.py` — soft labels per (site, window).            |
| `data/models/`            | Serialised XGBoost students + SHAP CSVs + OOD covariance.           |
| `data/predictions/`       | Output of `inference.py` — pixel and bed-level scores per region.   |

## Module reference

| Module                       | Inputs                                  | Outputs                                                            |
| ---------------------------- | --------------------------------------- | ------------------------------------------------------------------ |
| `pipeline.s2_features`       | GEE CSV (raw bands + indices per pixel) | Pixel × window feature matrix (~140 columns).                      |
| `pipeline.teacher`           | `aligned_eelgrass_env_data.csv`         | `teacher_labels.csv` (GPI_90d, decline_prob, stress, SHAP).        |
| `pipeline.student`           | Features + labels                       | 3 XGBoost models + SHAP + OOD stats.                               |
| `pipeline.inference`         | Target-region features + trained models | `<region>_pixels.csv` + `<region>_beds.csv`.                       |
| `pipeline.calibration`       | Ground-truth observations (optional)    | Isotonic/linear calibrator wrapping student predictions.           |
| `pipeline.run_all`           | —                                       | Runs steps 2–4 in sequence (training pipeline only).               |

## API endpoints

| Method | Path                                | Description                                                       |
| ------ | ----------------------------------- | ----------------------------------------------------------------- |
| GET    | `/api/status`                       | Last refresh time, available regions, model version.              |
| GET    | `/api/regions`                      | List all loaded regions with bed counts and risk summary.         |
| GET    | `/api/beds`                         | All beds (optionally filtered by region/risk_tier).               |
| GET    | `/api/beds/{bed_id}`                | Single bed: latest scores + dominant stress + SHAP explanation.   |
| GET    | `/api/beds/{bed_id}/pixels`         | Pixel-level predictions for heatmap rendering.                    |
| GET    | `/api/beds/{bed_id}/timeseries`     | Historical bed scores (90-day windows).                           |
| POST   | `/api/refresh`                      | Trigger refresh of a region (admin / scheduler).                  |

See `api/schemas.py` for response shapes.

## Refresh schedule

`refresh/scheduler.py` runs every 10 days (matching Sentinel-2 revisit cadence):

1. Triggers a GEE export for each registered region.
2. Waits for the Drive CSV, downloads it.
3. Runs `s2_features.py` → `inference.py`.
4. Replaces the latest predictions in `data/predictions/`.
5. The API serves the new data on next request (no restart needed).

Schedule with cron:
```cron
0 4 */10 * *  cd /path/to/eelsSaver_Backend && python -m refresh.scheduler >> data/refresh.log 2>&1
```

## Validation targets

| Model        | Spatial-val metric            | Temporal-test metric         |
| ------------ | ----------------------------- | ---------------------------- |
| GPI          | R² ≥ 0.65, MAE ≤ 0.10         | MAE ≤ 0.12                   |
| Decline prob | AUC ≥ 0.80, Brier ≤ 0.15      | AUC ≥ 0.75                   |
| Stress       | Macro-F1 ≥ 0.55 (4 classes)   | Macro-F1 ≥ 0.50              |

Below those, the student is not generalising — investigate spatial leakage, feature
redundancy, or label noise before extending to new regions.
