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


def diebold_mariano(y_true, y_pred_1, y_pred_2, loss: str = "squared",
                    h: int = 1) -> dict:
    """Diebold--Mariano test of equal predictive accuracy of two forecasts.

    Tests H0: E[d_t] = 0 for the loss differential d_t = L(e_1t) - L(e_2t),
    where e_it is the error of forecast i. A negative statistic favours
    forecast 1 (lower loss); a positive one favours forecast 2.

    For an h-step-ahead forecast the loss differential is serially correlated
    up to lag h-1 (overlapping windows), so the variance of d-bar uses a
    Newey--West estimator with a truncation lag of h-1. The Harvey, Leybourne
    and Newbold (1997) small-sample correction is applied and the statistic is
    referred to a Student-t(N-1) distribution.

    Parameters
    ----------
    loss : "squared" (MSE loss, default) or "absolute" (MAE loss).
    h    : forecast horizon in steps; sets the Newey--West lag to h-1.

    Returns {dm, p_value, n, mean_diff, lag}.
    """
    from scipy import stats

    y_true = np.asarray(y_true, dtype=float)
    e1 = y_true - np.asarray(y_pred_1, dtype=float)
    e2 = y_true - np.asarray(y_pred_2, dtype=float)
    mask = np.isfinite(e1) & np.isfinite(e2)
    e1, e2 = e1[mask], e2[mask]

    if loss == "squared":
        d = e1 ** 2 - e2 ** 2
    elif loss == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'")

    n = d.size
    d_bar = float(d.mean())
    lag = max(0, h - 1)

    # Newey--West long-run variance of d-bar (autocovariances up to lag h-1)
    gamma0 = float(np.mean((d - d_bar) ** 2))
    var = gamma0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1)                       # Bartlett weight
        cov = float(np.mean((d[k:] - d_bar) * (d[:-k] - d_bar)))
        var += 2.0 * w * cov
    var_dbar = var / n

    dm = d_bar / np.sqrt(var_dbar) if var_dbar > 0 else np.nan
    # Harvey--Leybourne--Newbold small-sample correction
    if np.isfinite(dm):
        corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        dm *= corr
        p_value = 2 * (1 - stats.t.cdf(abs(dm), df=n - 1))
    else:
        p_value = np.nan

    return {"dm": float(dm), "p_value": float(p_value), "n": int(n),
            "mean_diff": d_bar, "lag": lag}


def summary(y_true, y_pred, drift: float | None = None, horizon: int = 1) -> dict:
    """All metrics in one dict, with baselines for context.

    Baselines for RMSE / MAE:
      predict-zero  -- always forecast 0 (random walk without drift).
      predict-drift -- always forecast horizon * drift (random walk with drift).
                       Only included when drift is provided. drift should be the
                       daily mean log-return estimated on the training set
                       (never on the test set, to avoid leakage).

    Baseline for DA:
      always-up     -- the share of positive realisations; da_edge = da - da_up
                       is the conditional directional skill over the drift.
    """
    y_true, y_pred = _clean(y_true, y_pred)
    zero = np.zeros_like(y_true)
    da    = directional_accuracy(y_true, y_pred)
    da_up = always_up_da(y_true)
    out = {
        "n": int(y_true.size),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "da": da,
        "da_up": da_up,
        "da_edge": da - da_up,
        "rmse_zero": rmse(y_true, zero),
        "mae_zero": mae(y_true, zero),
    }
    if drift is not None:
        drift_pred = np.full_like(y_true, horizon * drift)
        out["rmse_drift"] = rmse(y_true, drift_pred)
        out["mae_drift"]  = mae(y_true, drift_pred)
    return out
