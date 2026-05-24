"""
config.py — single source of truth for paths, hyperparameters, and constants.

All pipeline modules import from here so paths/thresholds are not duplicated.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR        = Path(__file__).resolve().parent
DATA_DIR        = BASE_DIR / 'data'
RAW_DIR         = DATA_DIR / 'raw'
FEATURES_DIR    = DATA_DIR / 'features'
LABELS_DIR      = DATA_DIR / 'labels'
MODELS_DIR      = DATA_DIR / 'models'
PREDICTIONS_DIR = DATA_DIR / 'predictions'
REGIONS_FILE    = BASE_DIR / 'regions.json'

for d in (RAW_DIR, FEATURES_DIR, LABELS_DIR, MODELS_DIR, PREDICTIONS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Reference to the LSI directory (where the original aligned eelgrass data lives)
LSI_DIR = BASE_DIR.parent / 'LSI'
ALIGNED_EELGRASS_CSV = LSI_DIR / 'aligned_eelgrass_env_data.csv'

# Canonical output filenames
FORILLON_RAW_CSV     = RAW_DIR / 'forillon_s2_pixels.csv'
FORILLON_FEATURES    = FEATURES_DIR / 'forillon_features.csv'
TEACHER_LABELS_CSV   = LABELS_DIR / 'teacher_labels.csv'

# ---------------------------------------------------------------------------
# Sentinel-2 feature engineering
# ---------------------------------------------------------------------------
WINDOW_DAYS = 90       # rolling window matching teacher's GPI_90d
STEP_DAYS   = 14       # stride between window centers
MIN_OBS     = 3        # minimum cloud-free images per window

SPECTRAL_COLS = [
    'B2', 'B3', 'B4', 'B5', 'B6', 'B8', 'B8A', 'B11', 'B12',
    'NDAVI', 'WAVI', 'GB_ratio', 'RG_ratio', 'B3B2_diff',
    'NDWI', 'turbidity', 'red_edge_slope', 'SABI', 'depth_invariant',
]

# ---------------------------------------------------------------------------
# Teacher (ODE-based soft labels)
# ---------------------------------------------------------------------------
SST_OPTIMAL        = 15.0   # °C, Zostera marina optimal
SST_WIDTH          = 10.0   # Gaussian width
PAR_SATURATION     = 30.0   # Einstein m^-2 day^-1
GPI_FLOOR          = 0.05
SST_HEAT_THRESHOLD = 22.0
SST_COLD_THRESHOLD = 8.0
PAR_LOW_PCTL       = 20     # bottom 20% PAR = light limitation
SENESCENCE_MONTHS  = {9, 10, 11}

N_BOOTSTRAP        = 200    # decline-prob bootstrap trajectories
FORECAST_HORIZON   = 180    # days
ML_WINDOW          = 30     # GBM-Multi lag window

# ---------------------------------------------------------------------------
# Student (XGBoost)
# ---------------------------------------------------------------------------
SPATIAL_VAL_SITE   = 'Sud'  # held out for spatial validation
TEMPORAL_TEST_YR   = 2018   # held out for temporal test

XGB_PARAMS = dict(
    n_estimators=300, max_depth=6, learning_rate=0.03,
    min_child_weight=5, subsample=0.7, colsample_bytree=0.6,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, verbosity=0,
)

CORR_DROP_THRESHOLD = 0.95  # drop one of any pair with |r| above this

# ---------------------------------------------------------------------------
# Inference / bed aggregation
# ---------------------------------------------------------------------------
DECLINE_PIXEL_THRESHOLD = 0.6  # pixel flagged "high risk" above this
BED_RISK_BINS = [(0.0, 0.33, 'Low'), (0.33, 0.66, 'Medium'), (0.66, 1.01, 'High')]
EXPECTED_PASSES_PER_WINDOW = 6  # nominal S2 revisits per 90 days at Atlantic Canada latitudes

# Mahalanobis OOD detection
OOD_PVAL_HIGH  = 0.05
OOD_PVAL_LOW   = 0.01

# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------
REFRESH_INTERVAL_DAYS = 10

# ---------------------------------------------------------------------------
# Sites & friendly stress names
# ---------------------------------------------------------------------------
FORILLON_SITES = ['Ouest', 'Marais', 'Sud']
STRESS_CLASSES = ['heat', 'cold', 'light', 'senescence']

STRESS_LANGUAGE = {
    'heat':       'Heat stress — sea surface temperature above tolerance',
    'cold':       'Cold stress — sea surface temperature below tolerance',
    'light':      'Light limitation — reduced PAR (turbidity or shading)',
    'senescence': 'Seasonal senescence — natural autumn decline',
}

# SHAP → ecological language translations for the API
SHAP_LANGUAGE = {
    'GB_ratio':       'green-blue reflectance ratio (submerged vegetation density)',
    'B3':             'green reflectance (canopy biomass)',
    'red_edge_slope': 'red-edge slope (canopy structure)',
    'turbidity':      'turbidity proxy (water clarity)',
    'NDAVI':          'aquatic vegetation index',
    'WAVI':           'water-adjusted vegetation index',
    'NDWI':           'water/land discrimination',
    'depth_invariant': 'Lyzenga depth-invariant index',
    'SABI':           'surface algal bloom index',
}
