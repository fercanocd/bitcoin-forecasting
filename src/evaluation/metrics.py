"""
Forecast evaluation metrics for one-step-ahead log-return predictions.

All metrics operate on aligned 1-D arrays of realised vs predicted
log-returns and are model-agnostic, so every model family (SARIMAX,
Prophet, XGBoost, LSTM) is scored identically.

  RMSE  -- root mean squared error   (penalises large misses)
  MAE   -- mean absolute error       (robust, same units as the target)
  DA    -- directional accuracy      (share of days the sign is right)

For zero-centred log-returns the natural reference points are:
  RMSE / MAE  vs a "predict zero" forecast (the unconditional mean)
  DA          vs 0.5 (a coin flip)
"""
from __future__ import annotations
import numpy as np


def _clean(y_true, y_pred):
    """Coerce to float arrays and drop positions where either is non-finite."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _clean(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def directional_accuracy(y_true, y_pred) -> float:
    """Share of observations where the predicted sign matches the realised
    sign. Days with a realised return of exactly zero are excluded (there is
    no direction to get right)."""
    y_true, y_pred = _clean(y_true, y_pred)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.sign(y_pred[mask]) == np.sign(y_true[mask])))


def always_up_da(y_true) -> float:
    """Directional accuracy of the trivial 'always predict up' rule: the share
    of non-zero observations that are positive. This is the honest reference
    for DA when a model carries an unconditional drift -- a DA that merely
    matches this number reflects the drift, not conditional skill."""
    y_true = np.asarray(y_true, dtype=float)
    y_true = y_true[np.isfinite(y_true)]
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(y_true[mask] > 0))


def summary(y_true, y_pred) -> dict:
    """All metrics in one dict, with baselines for context: predict-zero (for
    RMSE/MAE) and always-up (for DA). da_edge = da - da_up is the conditional
    directional skill over the drift."""
    y_true, y_pred = _clean(y_true, y_pred)
    zero = np.zeros_like(y_true)
    da    = directional_accuracy(y_true, y_pred)
    da_up = always_up_da(y_true)
    return {
        "n": int(y_true.size),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "da": da,
        "da_up": da_up,
        "da_edge": da - da_up,
        "rmse_zero": rmse(y_true, zero),
        "mae_zero": mae(y_true, zero),
    }
