# Handoff — Running on real data

State of the world when this doc was written:

- `data/raw/` is **empty** (no real GEE export yet).
- The annual Ouest dataset (`ouest_eelgrass_environmental_complete.csv`) is **not on disk**.
- Pipeline code is wired up to consume both as soon as they arrive.

This doc tells future-you exactly what to do when you have them.

---

## 0. Files you need to drop in

| File | Where it goes | What it is |
| --- | --- | --- |
| `forillon_s2_pixels.csv` (or `ouest_s2_pixels.csv`) | `data/raw/` | The CSV exported by `gee/forillon_extraction.js` from Earth Engine. One row per pixel per cloud-free image; columns: `SITE, DATE, CLOUD_PCT, longitude, latitude, B2…B12, NDAVI…depth_invariant`. |
| `ouest_eelgrass_environmental_complete.csv` | anywhere readable (e.g. `data/raw/`) | The 6 years × 65 variables dataset described in the brief. Must contain at minimum `year`, `cover`, plus any `sst*` / `par*` / `temp*` / `light*` columns. |

---

## 1. Run the pipeline

```bash
# 1. Convert GEE per-image rows → pixel × 90-day-window features
python -m pipeline.s2_features data/raw/forillon_s2_pixels.csv \
                                data/features/forillon_features.csv

# 2. Generate teacher labels from the annual Ouest survey + env summaries
python -m pipeline.teacher_ouest /path/to/ouest_eelgrass_environmental_complete.csv

# 3. Train students with each split strategy and compare
python -m pipeline.student_loyo --strategy loyo      # 6-fold LOYO-CV
python -m pipeline.student_loyo --strategy forward   # 5-fold expanding window
python -m pipeline.student_loyo --strategy tail      # 2017-2018 epidemic test

# 4. Regenerate diagnostics
python -m pipeline.visualize
python -m pipeline.visualize_advanced
```

Outputs land in:

- `data/models/student_loyo/`, `student_forward/`, `student_tail/`
  - `fold_<k>_{gpi,decline,stress}.pkl`
  - `cv_report.json`, `cv_summary.csv`
- `data/figures/` (overwrites the synthetic-data versions)

---

## 2. Why these three splits, in this order

The Ouest dataset has **6 annual rows**. Standard random k-fold leaks years
across folds and inflates metrics; classic GroupKFold by year is essentially
LOYO. The boom-bust-recovery-crash structure means *which* year you hold
out changes the answer dramatically.

| Strategy | Folds | Test composition | Use it for |
| --- | --- | --- | --- |
| **`loyo`**    | 6 | one calendar year per fold | **Primary reporting.** Average ± std across folds is the honest generalisation estimate. |
| **`forward`** | 5 | year k, trained on years 0…k-1 | **Deployment realism.** Mimics the operational question: "Given everything up to year *t*, would the model have caught year *t+1*?" |
| **`tail`**    | 1 | 2017 + 2018 (the epidemic) | **Headline number.** "Did the model see the 2017-2018 damage epidemic coming?" — the single test that matters for the demo. |

Report all three. They answer different questions; together they tell the
full story.

### Validation-year choice inside a fold

`pipeline/splits.py` uses the *last* year of the training set as
validation (for XGBoost early-stopping). This is intentional — using a
random year would leak future-state info into training.

Example for LOYO with held-year=2016:
- train = 2013, 2014, 2015 + 2017, 2018
- val (early-stopping) = 2018 (most recent training year)
- test = 2016 (held out)

### LOYO fold count vs sample size

With 6 years × ~hundreds of S2 pixels per year, each LOYO fold has
~5000 training rows and ~1000 test rows. That's enough for XGBoost,
but the underlying *label* space is still only 6 distinct annual values.
**Expect higher metric variance across folds than you'd see with daily
labels.** Report mean ± std, not mean alone.

---

## 3. What to look for in the results

| Concern | Diagnostic | Action if it fires |
| --- | --- | --- |
| GPI mean ± std variance across LOYO folds | `cv_summary.csv` filter `model=gpi, metric=r2` | High variance ( > 0.20 std) means features are too tightly coupled to specific years. Add more spectral data or add a neighbouring site. |
| 2017-2018 tail test catastrophically worse than LOYO mean | compare `student_tail/cv_report.json` to `student_loyo/cv_report.json` | The model can interpolate within the cycle but can't extrapolate into the epidemic. This is the EEWS use-case — flag the regime change but caveat predictions. |
| Forward-chaining test metric improving monotonically with fold | `student_forward/cv_summary.csv` plotted against fold index | Good — model is benefiting from more data. If it plateaus or drops, you're hitting label noise / concept drift. |
| Decline AUC ≈ 0.5 anywhere | `student_*/cv_report.json` `decline_auc` | The 0/1 decline_flag may be too sparse — only 1 positive label per site-year transition, so most LOYO folds will have very few positives in the test set. Switch to predicting `cover` directly and post-thresholding. |

---

## 4. Sanity checks before training

```bash
# Inspect the split assignments
python -m pipeline.splits data/features/forillon_features.csv
# → prints fold composition for loyo / forward / tail
```

You should see:
- Each LOYO fold has exactly 5 unique training years and 1 test year.
- Forward fold 4: train = 2013-2017, val = 2017, test = 2018.
- Tail: train = {2013, 2014, 2015}, val = {2016}, test = {2017, 2018}.

If `years detected != [2013, 2014, 2015, 2016, 2017, 2018]`, your S2
export probably has only partial year coverage — re-run the GEE script
with the right date range.

---

## 5. Quick model retraining cycle

While iterating:

```bash
# Single-fold debug (tail strategy is fastest — 1 fold)
python -m pipeline.student_loyo --strategy tail

# Time the LOYO sweep
time python -m pipeline.student_loyo --strategy loyo
```

Expected runtime on a laptop:
- s2_features: 1–3 min for Forillon-sized region
- teacher_ouest: < 5 s
- student_loyo --strategy loyo: 30 s – 2 min (6 folds × 3 models)

If LOYO takes more than 10 min something is wrong — likely too many
features survived correlation pruning or `n_estimators` was bumped.

---

## 6. Inference on a new region (post-training)

The API and refresh loop stay the same — they don't care which split
strategy was used to train. Pick *one* trained model per task (typical
choice: the LOYO model trained on **all 6 years** of Forillon, no
hold-out) and use it for pan-Atlantic inference:

```bash
# Build a "full training" model with no hold-out
python -m pipeline.student   # original full-data trainer

# Then for each new region
python -m pipeline.s2_features data/raw/antigonish_s2_pixels.csv \
                                 data/features/antigonish_features.csv
python -m pipeline.inference data/features/antigonish_features.csv antigonish
```

The LOYO/forward/tail runs are **diagnostics** — they tell you how much
to trust the deployed model, not which model to deploy.
