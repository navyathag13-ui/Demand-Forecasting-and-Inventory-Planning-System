"""
config.py — Global configuration for the Demand Forecasting and Inventory Planning System.

Edit constants here to change pipeline behaviour without touching source code.
"""

from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR    = Path(__file__).parent
DATA_DIR    = ROOT_DIR / "data"
OUTPUT_DIR  = ROOT_DIR / "outputs"
CHARTS_DIR  = OUTPUT_DIR / "charts"
FORECASTS_DIR = OUTPUT_DIR / "forecasts"
METRICS_DIR = OUTPUT_DIR / "metrics"
REPORTS_DIR = OUTPUT_DIR / "reports"

TRAIN_FILE  = DATA_DIR / "train.csv"
STORES_FILE = DATA_DIR / "stores.csv"

# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
RANDOM_SEED    = 42
HOLDOUT_WEEKS  = 12          # weeks held out for evaluation

LGBM_PARAMS = {
    "n_estimators":     500,
    "learning_rate":    0.05,
    "num_leaves":       31,
    "max_depth":        -1,
    "min_child_samples": 20,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "random_state":     RANDOM_SEED,
    "n_jobs":           -1,
    "verbose":          -1,
}

# ---------------------------------------------------------------------------
# Inventory planning
# ---------------------------------------------------------------------------
LEAD_TIME_WEEKS    = 2      # supplier lead time (weeks)
REVIEW_PERIOD_WEEKS = 4     # inventory review cycle (weeks)
SERVICE_LEVEL      = 0.95   # target in-stock probability → Z ≈ 1.645
Z_SCORE            = float(np.round(
    __import__("scipy.stats", fromlist=["norm"]).norm.ppf(SERVICE_LEVEL), 4
))

# Assumed unit economics (illustrative — not in raw dataset)
UNIT_HOLDING_COST_PCT = 0.20   # annual holding cost as % of unit value
UNIT_VALUE_USD        = 25.00  # assumed average item value ($)
ORDERING_COST_USD     = 150.0  # fixed cost per purchase order ($)

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
CHART_DPI    = 150
CHART_STYLE  = "seaborn-v0_8-whitegrid"
PALETTE      = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
