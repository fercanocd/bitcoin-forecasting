"""
Expanding-window walk-forward engine for recursive multi-step forecasters.

Designed for state-space / recursive forecasters (SARIMAX, Prophet) that
keep an internal state advanced one observation at a time:

  * parameters are re-estimated periodically (e.g. monthly) on the full
    expanding history available up to that point;
  * between refits the realised observation of each day is fed back in
    (cheap filtering, no re-estimation) before forecasting the next day.

Supports any forecast horizon h. At each origin the model forecasts the
h-day-ahead *cumulative* log-return; this is compared against the realised
cumulative return sum(returns[t+1 .. t+h]), which equals target_log_return_{h}d
by construction. The loop stops early enough that the full h-day realisation
exists for every prediction (the last h origins cannot be scored).

This mirrors a realistic deployment: you retrain occasionally but keep
forecasting daily with the latest data. Direct models (XGBoost, LSTM) carry
no recursive state and will use a separate, simpler runner.

The engine is model-agnostic via the WalkForwardModel protocol:

  fit(y_hist, X_hist)    -- (re-)estimate parameters on the history
  forecast(h, X_next)    -- h-step-ahead cumulative point forecast
  observe(y_obs, X_obs)  -- advance internal state by one realised obs

Positions are integer indices into the aligned arrays; dates are carried
alongside purely for reporting. Working positionally keeps statsmodels free
of DatetimeIndex/frequency edge cases.
"""
from __future__ import annotations
from typing import Protocol
import numpy as np
import pandas as pd


class WalkForwardModel(Protocol):
    def fit(self, y_hist: np.ndarray, X_hist: np.ndarray | None) -> None: ...
    def forecast(self, horizon: int, X_next: np.ndarray | None) -> float: ...
    def observe(self, y_obs: np.ndarray, X_obs: np.ndarray | None) -> None: ...


def month_start_positions(dates: pd.DatetimeIndex, start: int, end: int) -> set[int]:
    """Refit schedule: positions in [start, end) that begin a new calendar
    month, plus the first position. One refit per month."""
    refit = {start}
    for i in range(start + 1, end):
        if (dates[i].year, dates[i].month) != (dates[i - 1].year, dates[i - 1].month):
            refit.add(i)
    return refit


def expanding_walk_forward_multi_horizon(
    y: np.ndarray,
    X: np.ndarray | None,
    dates: pd.DatetimeIndex,
    test_start: int,
    model: WalkForwardModel,
    refit_positions: set[int],
    horizons: list[int],
    test_end: int | None = None,
    verbose: bool = True,
) -> dict[int, pd.DataFrame]:
    """Walk-forward loop that fits once per origin and scores all horizons jointly.

    Correct for recursive models (SARIMAX, Prophet): the daily model is
    identical across horizons, so fitting separately per horizon is redundant
    and triples the compute cost for no benefit. At each origin the model is
    fit once, then forecast(h) is called for every h in horizons before
    observe() advances the state by one day.

    For each horizon h, only origins where y[i:i+h] is fully observed are
    scored (the last h-1 origins cannot be evaluated for that horizon).

    Returns a dict {h: DataFrame(date, y_true, y_pred)} -- one entry per horizon.
    """
    n = len(y)
    test_end = n if test_end is None else test_end
    rows = {h: [] for h in horizons}
    n_refits = 0

    for i in range(test_start, test_end):
        if i in refit_positions:
            model.fit(y[:i], None if X is None else X[:i])
            n_refits += 1
        X_next = None if X is None else X[i:i + 1]
        for h in horizons:
            if i + h > n:           # future window not fully observed yet
                continue
            y_pred = model.forecast(h, X_next)
            y_true = float(np.sum(y[i:i + h]))
            rows[h].append((dates[i], y_true, y_pred))
        model.observe(y[i:i + 1], None if X is None else X[i:i + 1])

    if verbose:
        counts = {h: len(rows[h]) for h in horizons}
        print(f"  Walk-forward (joint): {n_refits} refits | "
              + " | ".join(f"h={h}: {counts[h]} preds" for h in horizons))

    return {
        h: pd.DataFrame(rows[h], columns=["date", "y_true", "y_pred"]).set_index("date")
        for h in horizons
    }


def expanding_walk_forward(
    y: np.ndarray,
    X: np.ndarray | None,
    dates: pd.DatetimeIndex,
    test_start: int,
    model: WalkForwardModel,
    refit_positions: set[int],
    horizon: int = 1,
    test_end: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the walk-forward loop over positions [test_start, eval_end).

    Parameters
    ----------
    y, X            : full aligned arrays. X must already be lagged so that
                      row i is known when position i is forecast.
    dates           : DatetimeIndex aligned with y / X (reporting only).
    test_start      : first integer position whose return is forecast.
    horizon         : forecast horizon h (days). The cumulative return over
                      positions [i, i+h-1] is predicted at each step.
    refit_positions : positions at which to re-estimate parameters.
    test_end        : one past the last realised position to consider
                      (default len(y)); the effective end is capped so that
                      y[i:i+h] is always fully observed.

    Returns a DataFrame indexed by date with columns y_true, y_pred, where the
    date is the first day of the realised window.
    """
    n = len(y)
    test_end = n if test_end is None else test_end
    eval_end = min(test_end, n - horizon + 1)     # need y[i:i+h] fully observed
    rows, n_refits = [], 0
    for i in range(test_start, eval_end):
        if i in refit_positions:
            model.fit(y[:i], None if X is None else X[:i])
            n_refits += 1
        X_next = None if X is None else X[i:i + 1]
        y_pred = model.forecast(horizon, X_next)
        y_true = float(np.sum(y[i:i + horizon]))   # == target_log_return_{h}d
        rows.append((dates[i], y_true, y_pred))
        model.observe(y[i:i + 1], None if X is None else X[i:i + 1])
    if verbose:
        print(f"  Walk-forward (h={horizon}): {len(rows)} predictions, {n_refits} refits")
    return pd.DataFrame(rows, columns=["date", "y_true", "y_pred"]).set_index("date")
