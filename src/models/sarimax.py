"""
SARIMAX walk-forward forecasting for Bitcoin log-returns (multi-horizon).

Reads  : data/processed/arima_features_daily.csv
Outputs: reports/predictions/sarimax_{h}d.csv  (date, y_true, y_pred)

Model family  : ARIMA / SARIMAX (statsmodels state-space)
Endogenous    : log_return  -- daily log(P_t / P_{t-1}); stationary, so d = 0.
Exogenous     : log_volume_ratio, sp500/gold/dxy/eth log-returns.
Strategy      : recursive multi-step. The daily model is identical across
                horizons; an h-day forecast is the sum of h forecast daily
                log-returns (logs are additive: sum == ln(P_{t+h}/P_t)).

Exogenous handling -- LAGGED by one day (lag-1).
    The regressors are stochastic: their value on the forecast day is unknown
    at prediction time. We use the previous day's regressor row, so the FIRST
    future step is fed real, known values and the exogenous block actually
    contributes. For steps 2..h (h > 1) the regressors are genuinely unknown
    future returns and are set to 0 -- the honest assumption for a zero-mean
    series. We never read the lagged array beyond the first step, which would
    leak future macro that is not known at the origin.

Order search  : (p, d, q) by AIC on the initial training window (no pmdarima
                on this stack); the selected daily order is reused across the
                walk-forward and across horizons. No seasonal term in this
                baseline; it can be added later as SARIMA.
Trend         : a constant drift term (trend='c') is included so the model
                captures BTC's unconditional upward drift. It barely moves the
                1-day forecast but accumulates over multi-step horizons (the
                h-day forecast picks up ~h*c). Any directional-accuracy gain at
                long horizons reflects this drift, not conditional skill, so DA
                is read against an "always predict the drift sign" reference,
                not 0.5.
Validation    : expanding-window walk-forward, monthly refit. Between refits
                each realised return is filtered in (no re-estimation) before
                advancing. The last h origins are not scored (no realised
                h-day target yet).

Usage:
    python -m src.models.sarimax                      # all horizons (config.HORIZONS)
    python -m src.models.sarimax --horizon 7          # a single horizon
    python -m src.models.sarimax --horizon 30 --max-test-days 60   # smoke run
"""
from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from src.config import TEST_START, TEST_END, HORIZONS
from src.evaluation.metrics import summary
from src.evaluation.walk_forward import expanding_walk_forward, month_start_positions

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
RESULTS_DIR   = Path(__file__).resolve().parents[2] / "reports" / "predictions"

ENDOG_COL = "log_return"
EXOG_COLS = ["log_volume_ratio", "sp500_log_return", "gold_log_return",
             "dxy_log_return", "eth_log_return"]

warnings.simplefilter("ignore", ConvergenceWarning)


def load_data():
    """Load the ARIMA feature set and build the lag-1 exogenous design.

    Returns (dates, endog, exog) aligned, with the first row dropped (NaN from
    the shift). exog row i holds the regressor values of day i-1, i.e. the
    information known when forecasting day i.
    """
    df = pd.read_csv(PROCESSED_DIR / "arima_features_daily.csv",
                     index_col="Date", parse_dates=True)
    endog = df[ENDOG_COL]
    exog  = df[EXOG_COLS].shift(1)            # lag-1: known at forecast time
    valid = exog.dropna().index               # drop the leading NaN row (ascending)
    return valid, endog.loc[valid], exog.loc[valid]


def select_order(y, X, p_range=(0, 1, 2), q_range=(0, 1, 2), d=0, trend="c"):
    """Pick (p, d, q) minimising AIC on the initial training window.

    log_return is already a (log) first difference of price, so d = 0.
    The trend term is held fixed across candidates so the AIC comparison
    reflects the specification actually used at forecast time.
    """
    best_order, best_aic = None, np.inf
    for p in p_range:
        for q in q_range:
            try:
                res = SARIMAX(y, exog=X, order=(p, d, q), trend=trend,
                              enforce_stationarity=False,
                              enforce_invertibility=False).fit(disp=False)
            except Exception:
                continue
            if np.isfinite(res.aic) and res.aic < best_aic:
                best_order, best_aic = (p, d, q), res.aic
    if best_order is None:
        best_order = (1, 0, 1)
        print("  Order search failed; falling back to (1, 0, 1)")
    else:
        print(f"  Selected daily order {best_order}  (AIC={best_aic:.1f})")
    return best_order


class SarimaxModel:
    """Recursive multi-step SARIMAX wrapped for the walk-forward engine."""

    def __init__(self, order, trend="c", seasonal_order=(0, 0, 0, 0)):
        self.order = order
        self.trend = trend
        self.seasonal_order = seasonal_order
        self.res = None

    def fit(self, y_hist, X_hist):
        self.res = SARIMAX(
            y_hist, exog=X_hist, order=self.order, trend=self.trend,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)

    def forecast(self, horizon, X_next):
        """h-step cumulative forecast: sum of the h forecast daily returns.

        With lag-1 exogenous, only the first future step's regressors are known
        (X_next); steps 2..h use 0 (unknown future zero-mean returns).
        """
        if X_next is None:
            fc = self.res.forecast(steps=horizon)
        else:
            future_exog = np.zeros((horizon, X_next.shape[1]))
            future_exog[0] = X_next[0]
            fc = self.res.forecast(steps=horizon, exog=future_exog)
        return float(np.asarray(fc).ravel().sum())

    def observe(self, y_obs, X_obs):
        self.res = self.res.append(endog=y_obs, exog=X_obs, refit=False)


def run(horizon=1, order=None, trend="c", max_test_days=None, verbose=True, _data=None):
    print(f"\n[SARIMAX] {horizon}-day-ahead walk-forward (lag-1 exogenous)")
    dates, endog, exog = _data if _data is not None else load_data()
    y = endog.to_numpy(dtype=float)
    X = exog.to_numpy(dtype=float)

    test_start = int(dates.searchsorted(pd.Timestamp(TEST_START)))
    test_end   = len(y)
    if TEST_END is not None:
        test_end = int(dates.searchsorted(pd.Timestamp(TEST_END), side="right"))
    if max_test_days is not None:
        test_end = min(test_end, test_start + max_test_days)

    print(f"  Train: {dates[0].date()} .. {dates[test_start-1].date()}  ({test_start} obs)")
    print(f"  Test : {dates[test_start].date()} .. {dates[test_end-1].date()}  ({test_end-test_start} obs)")

    if order is None:
        order = select_order(y[:test_start], X[:test_start], trend=trend)

    model = SarimaxModel(order=order, trend=trend)
    refit = month_start_positions(dates, test_start, test_end)

    preds = expanding_walk_forward(
        y, X, dates, test_start, model, refit,
        horizon=horizon, test_end=test_end, verbose=verbose,
    )

    m = summary(preds["y_true"].to_numpy(), preds["y_pred"].to_numpy())
    print(f"  Results: n={m['n']}  "
          f"RMSE={m['rmse']:.5f} (zero {m['rmse_zero']:.5f})  "
          f"MAE={m['mae']:.5f} (zero {m['mae_zero']:.5f})  "
          f"DA={m['da']:.3f} (up {m['da_up']:.3f}, edge {m['da_edge']:+.3f})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"sarimax_{horizon}d.csv"
    preds.to_csv(out)
    print(f"  Saved -> {out}")
    return preds, m


def run_all(horizons=None, trend="c", max_test_days=None):
    """Run every horizon, selecting the daily order once and reusing it."""
    horizons = horizons or HORIZONS
    data = load_data()
    dates, endog, exog = data
    test_start = int(dates.searchsorted(pd.Timestamp(TEST_START)))
    order = select_order(endog.to_numpy(float)[:test_start],
                         exog.to_numpy(float)[:test_start], trend=trend)

    results = {h: run(horizon=h, order=order, trend=trend,
                      max_test_days=max_test_days, _data=data)[1] for h in horizons}

    print(f"\n{'='*64}\n  SARIMAX summary  (RMSE / MAE / DA  vs predict-zero RMSE)\n{'='*64}")
    for h, m in results.items():
        print(f"    h={h:>2}d   RMSE {m['rmse']:.5f}   MAE {m['mae']:.5f}   "
              f"DA {m['da']:.3f} (up {m['da_up']:.3f}, edge {m['da_edge']:+.3f})")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="SARIMAX multi-horizon walk-forward")
    ap.add_argument("--horizon", type=int, default=None,
                    help="single horizon in days (default: all in config.HORIZONS)")
    ap.add_argument("--max-test-days", type=int, default=None,
                    help="limit the test horizon for a quick smoke run")
    args = ap.parse_args()
    if args.horizon is None:
        run_all(max_test_days=args.max_test_days)
    else:
        run(horizon=args.horizon, max_test_days=args.max_test_days)
