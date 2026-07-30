"""
Central configuration for the Bitcoin forecasting experiment.

All methodological constants live here so that features, models and
evaluation modules share a single source of truth.
"""

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
# Time-series cross-validation (Round 1: hyperparameter selection on train)
# ---------------------------------------------------------------------------
# Expanding annual folds: the first fold trains on MIN_TRAIN_YEARS full
# calendar years starting at TRAIN_START, validates on the next year, then
# each subsequent fold extends the train window by one year. The list of
# validation years is derived in src/evaluation/cv.py from TRAIN_START,
# TEST_START and MIN_TRAIN_YEARS -- if any of those move, the folds move
# with them automatically.

MIN_TRAIN_YEARS = 2

# ---------------------------------------------------------------------------
# Walk-forward (Round 2: test evaluation)
# ---------------------------------------------------------------------------
# Refit cadence differs by model family and is set in each model file:
#   SARIMAX, Prophet  -> daily refit (cheap, and h-step forecasts compound)
#   XGBoost, LSTM     -> monthly refit (expensive, weights are stable)

# Rolling training window for daily coefficient refits, in calendar days.
# None => expanding window (all history since TRAIN_START) -- the default.
# An int => at each origin the model is fitted on the last TRAIN_WINDOW days
# only, so coefficients track the current regime instead of averaging in the
# thin, structurally different 2018-2019 market. Kept as a switch for the
# rolling-vs-expanding robustness analysis; expanding is the official setting.
TRAIN_WINDOW = None

# ---------------------------------------------------------------------------
# Statistical inference
# ---------------------------------------------------------------------------

# Two-sided significance level for the Diebold-Mariano test on the test set.
# 0.05 is the Fisher-standard convention; kept here as the single source of
# truth so the thesis can reference one number without hunting through code.
DM_ALPHA = 0.05

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED = 42
