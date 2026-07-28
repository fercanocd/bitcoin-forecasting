"""
Central configuration for the Bitcoin forecasting experiment.

All methodological constants live here so that features, models and
evaluation modules share a single source of truth.
"""

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

DATA_START = "2018-01-01"
DATA_END   = "2026-06-01"

# ---------------------------------------------------------------------------
# Forecasting horizons
# ---------------------------------------------------------------------------

HORIZONS = [1, 7, 30]      # calendar days; BTC trades 24/7/365

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

TRAIN_START = "2018-01-01"
TRAIN_END   = "2023-12-31"
TEST_START  = "2024-01-01"
TEST_END    = "2026-06-01"  

# ---------------------------------------------------------------------------
# Time-series cross-validation (hyperparameter tuning on train set)
# ---------------------------------------------------------------------------

FOLD_UNIT       = "year"   # each fold expands by one year
MIN_TRAIN_YEARS = 2        # minimum years in the first fold's training window

# ---------------------------------------------------------------------------
# Walk-forward (test set evaluation)
# ---------------------------------------------------------------------------

REFIT_FREQ = "ME"          # pandas offset alias: month-end refit

# Rolling training window for daily coefficient refits, in calendar days.
# None => expanding window (all history since TRAIN_START) -- the default.
# An int => at each origin the model is fitted on the last TRAIN_WINDOW days
# only, so coefficients track the current regime instead of averaging in the
# thin, structurally different 2018-2019 market. Kept as a switch for the
# rolling-vs-expanding robustness analysis; expanding is the official setting.
TRAIN_WINDOW = None

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED = 42
