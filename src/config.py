"""
Central configuration for the Bitcoin forecasting experiment.

All methodological constants live here so that features, models and
evaluation modules share a single source of truth.
"""

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

DATA_START = "2018-01-01"
DATA_END   = "2026-12-31"

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
TEST_END    = "2026-05-20"  # experiment cutoff — provisional, adjust later

# ---------------------------------------------------------------------------
# Time-series cross-validation (hyperparameter tuning on train set)
# ---------------------------------------------------------------------------

FOLD_UNIT       = "year"   # each fold expands by one year
MIN_TRAIN_YEARS = 2        # minimum years in the first fold's training window

# ---------------------------------------------------------------------------
# Walk-forward (test set evaluation)
# ---------------------------------------------------------------------------

REFIT_FREQ = "ME"          # pandas offset alias: month-end refit

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED = 42
