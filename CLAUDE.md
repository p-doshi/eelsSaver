# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout — two backends, one frontend

This repo contains **two parallel backend implementations** plus a static frontend. They are NOT integrated with each other — they are alternative implementations of the eelgrass early-warning system:

| Dir | Purpose | Stack |
| --- | --- | --- |
| [backend/](backend/) | **Demo/live-inference backend.** Synchronous FastAPI app that proxies STAC searches for Sentinel-2 scenes, runs a small GBM (or a hard-coded `DemoModel` fallback), and streams progress to the frontend over WebSocket. | FastAPI, httpx, scikit-learn, rasterio (overlays only) |
| [backend-dev/](backend-dev/) | **Full research pipeline.** Trains XGBoost students on Forillon Sentinel-2 + in-situ data, produces SHAP explanations, runs pixel-level inference on Atlantic-Canada polygons, serves results via a separate FastAPI app. | pandas, xgboost, shap, FastAPI, Google Earth Engine (JS, run externally) |
| [frontend/](frontend/) | Leaflet map + sidebar (vanilla JS, no build step). Talks to `backend/` on `http://localhost:8000` — **not** `backend-dev/`. | Leaflet, hand-rolled DOM |

`backend/` and `backend-dev/` have their own `requirements.txt`, their own FastAPI app, and overlap only conceptually. When asked to change "the API" or "the model," **first ask which backend** unless it is obvious from context.

The `frontend/` calls `backend/`'s endpoints (`/api/hotspots`, `/api/inference/start`, `/api/inference/ws/{id}`, `/api/photos`). It does NOT consume `backend-dev/`'s `/api/beds` / `/api/regions` shape — `backend-dev/web/index.html` is a separate Leaflet page that does.

## Commands

### backend/ (demo + live STAC inference)

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env                                # then edit
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Generate hotspot overlay images (optional, requires Planetary Computer access):
```bash
python scripts/generate_overlays.py
```

### backend-dev/ (training + research pipeline)

All `python -m pipeline.*` commands must be run **from `backend-dev/`** (the package uses top-level `config.py` as a module).

```bash
cd backend-dev
pip install -r requirements.txt

# Training pipeline (after dropping raw GEE CSV into data/raw/)
python -m pipeline.s2_features data/raw/forillon_s2_pixels.csv data/features/forillon_features.csv
python -m pipeline.teacher                       # uses in-situ env data
python -m pipeline.teacher_ouest <ouest.csv>     # alternative: annual Ouest survey
python -m pipeline.student                       # train on all data, single split
python -m pipeline.student_loyo --strategy loyo  # 6-fold leave-one-year-out CV
python -m pipeline.student_loyo --strategy forward  # expanding-window
python -m pipeline.student_loyo --strategy tail     # 2017–2018 epidemic test
python -m pipeline.run_all                       # convenience: s2_features → teacher → student

# Inference on a new region
python -m pipeline.s2_features data/raw/<region>_s2_pixels.csv data/features/<region>_features.csv
python -m pipeline.inference data/features/<region>_features.csv <region_id>

# Diagnostics / figures
python -m pipeline.visualize
python -m pipeline.visualize_advanced

# Inspect CV split assignments before training
python -m pipeline.splits data/features/forillon_features.csv

# Serve the API + Leaflet web map
uvicorn api.main:app --reload --port 8000

# Cron-style refresh of every inference region (expects new CSVs in data/raw/)
python -m refresh.scheduler
```

There is **no test suite, lint config, or formatter configured** in either backend. Do not invent one.

### frontend/

Static files — no build step.

```bash
open frontend/index.html              # or serve from any static server
```

`API_BASE` is hard-coded to `http://localhost:8000` in [frontend/app.js](frontend/app.js).

## Architecture — what's non-obvious

### `backend-dev/` pipeline data flow

The pipeline is **teacher → student → inference**, all driven by file artifacts (no DB, no in-memory state). Stages communicate only through CSVs/pickles under `data/`:

1. **Earth Engine extraction** (`gee/*.js`) — run manually in the GEE Code Editor, export to Drive, drop the CSV into `data/raw/`. The pipeline never calls GEE itself.
2. **`pipeline/s2_features`** — converts per-image pixel rows into pixel × 90-day-window feature matrices (~140 columns) with rolling stats. Stride = 14 days.
3. **`pipeline/teacher*`** — generates *soft* labels (GPI_90d, decline_prob_180d, dominant stress) from Forillon in-situ env data using an ODE-style model. There are **two teacher variants**: `teacher.py` (daily aligned data, original LSI dataset) and `teacher_ouest.py` (annual Ouest survey, 6 years).
4. **`pipeline/student*`** — three XGBoost models (GPI regression, decline regression, stress classification) trained on student features + teacher labels. Splits matter — see HANDOFF.md.
5. **`pipeline/inference`** — pixel scoring on new region features, then bed-level aggregation, Mahalanobis OOD detection (LedoitWolf covariance saved from training), confidence tiers.
6. **`api/`** — reads `data/predictions/*_beds.csv` and `*_pixels.csv` at request time. No caching; missing files return empty lists rather than 500s. `POST /api/refresh` shells out to `python -m pipeline.inference` synchronously.

### Split strategy choice (backend-dev)

Standard k-fold leaks years and inflates metrics because the Ouest dataset has only **6 annual rows**. The pipeline supports three split strategies via `student_loyo.py`:

- `loyo` — leave-one-year-out, primary reporting
- `forward` — expanding-window, deployment realism
- `tail` — held-out {2017, 2018} epidemic, the "did we catch it?" headline

All three should be reported together. See [backend-dev/HANDOFF.md](backend-dev/HANDOFF.md) for context on why and what diagnostics to watch.

### Single source of truth: `backend-dev/config.py`

All paths, hyperparameters, ecological thresholds, region IDs, stress class names, and SHAP→ecology phrase translations live in [backend-dev/config.py](backend-dev/config.py). The pipeline modules and API import from there — never duplicate constants in the modules themselves.

### `backend/` job pattern

[backend/app/services/jobs.py](backend/app/services/jobs.py) implements an in-memory job system: `create_job()` returns an ID and spawns an `asyncio.create_task`; progress is fan-out via per-job `asyncio.Queue` subscribers consumed by the WebSocket endpoint in [backend/app/api/inference.py](backend/app/api/inference.py). State lives in [backend/app/state.py](backend/app/state.py) (plain module-level dicts — single-process only, no Redis).

If `MODEL_PATH` doesn't resolve to a loadable joblib bundle, `ModelService.predict()` silently falls back to the hard-coded `DemoModel` in [backend/app/services/model_runner.py](backend/app/services/model_runner.py). The "live inference" demo is therefore deterministic even with no model present.

`backend/app/api/hotspots.py` returns hard-coded GeoJSON from [backend/app/services/hotspot_repo.py](backend/app/services/hotspot_repo.py) — it is NOT backed by `backend-dev/`'s predictions. Wiring the two together is not done.

### Regions are declarative

[backend-dev/regions.json](backend-dev/regions.json) is the registry of training + inference regions consumed by the API, refresh scheduler, and any new pipeline scripts. Each region has `id`, `role` (`training` | `inference`), and `beds` (each with `id`, `site`, `polygon`). Adding a new inference region = add an entry here + run GEE extraction + run `pipeline.s2_features` + `pipeline.inference`.
