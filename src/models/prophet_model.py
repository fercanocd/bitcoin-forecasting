"""
Prophet (Meta) walk-forward forecasting for Bitcoin log-returns (multi-horizon).

Reads  : data/processed/prophet_features_daily.csv
Outputs: reports/predictions/prophet_{h}d.csv  (date, y_true, y_pred)

Model family  : Prophet -- additive structural model (trend + seasonality +
                regressors). Paired with SARIMAX as the classical baseline.
Endogenous    : log_return  -- daily log(P_t / P_{t-1}); fed to Prophet as `y`.
Regressors    : log_volume_ratio, sp500/gold/dxy/eth log-returns, attached via
                add_regressor (additive mode). Same set as SARIMAX, so any gap
                reflects model architecture, not feature access.
Strategy      : recursive multi-step. The daily model is identical across
                horizons; an h-day forecast is the sum of h forecast daily
                log-returns (logs are additive: sum == ln(P_{t+h}/P_t)).

Trend         : growth='flat' -- a constant level, the Prophet analog of
                SARIMAX's trend='c' constant drift. log_return is ~zero-mean, so
                a piecewise-linear trend would chase noise; a flat level plus
                seasonality and regressors isolates whether Prophet's structural
                decomposition adds anything over the pure drift. Any directional
                gain still reflects that drift, so DA is read against an
                "always predict up" reference, not 0.5.

Exogenous handling -- LAGGED by one day (lag-1), identical to SARIMAX.
    The regressors are stochastic: their value on the forecast day is unknown
    at prediction time. We use the previous day's regressor row, so the FIRST
    future step is fed real, known values and the regressor block contributes.
    For steps 2..h (h > 1) the regressors are genuinely unknown future returns
    and are set to 0 -- the honest assumption for a zero-mean series. We never
    read the lagged array beyond the first step, which would leak future macro
    not known at the origin.

Validation    : expanding-window walk-forward, REFIT AT EVERY ORIGIN. Unlike
    SARIMAX -- whose state-space form filters each realised observation in
    cheaply (append, refit=False) between monthly refits -- Prophet has no
    cheap update: new data can only enter by re-estimating. Refitting every
    step is therefore what makes Prophet a fair, apples-to-apples comparison to
    SARIMAX (both always condition on the full realised history). A monthly
    schedule is available via --refit for fast dev runs, but between refits it
    ignores the most recent realised returns and is not directly comparable.
    The last h origins are not scored (no realised h-day target yet).

Usage:
    python -m src.models.prophet_model                      # all horizons
    python -m src.models.prophet_model --horizon 7          # a single horizon
    python -m src.models.prophet_model --refit monthly --max-test-days 60  # smoke
"""
from __future__ import annotations
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet

from src.config import TRAIN_START, TEST_START, TEST_END, HORIZONS, TRAIN_WINDOW
from src.evaluation.cv import DEFAULT_VAL_YEARS, aggregate_metrics, year_folds
from src.evaluation.metrics import summary
from src.evaluation.runtime import record
import time
from src.evaluation.walk_forward import (expanding_walk_forward,
                                         expanding_walk_forward_multi_horizon,
                                         month_start_positions)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
RESULTS_DIR   = Path(__file__).resolve().parents[2] / "reports" / "predictions"
MODELS_DIR    = Path(__file__).resolve().parents[2] / "models" / "saved"

ENDOG_COL      = "log_return"
REGRESSOR_COLS = ["log_volume_ratio", "sp500_log_return", "gold_log_return",
                  "dxy_log_return", "eth_log_return"]

# Prophet/cmdstanpy re-add handlers at runtime; disabled=True is the only
# reliable way to silence them completely.
for _name in ("prophet", "cmdstanpy", "stan"):
    logging.getLogger(_name).disabled = True
warnings.simplefilter("ignore", FutureWarning)


def load_data():
    """Load the Prophet feature set and build the lag-1 regressor design.

    Mirrors SARIMAX.load_data exactly: regressors are shifted by one day, so the
    row aligned to prediction position i holds day i-1's values -- the macro
    known when forecasting day i. The leading NaN row from the shift is dropped.
    Returns (dates, endog Series, exog DataFrame) all aligned.
    """
    df = pd.read_csv(PROCESSED_DIR / "prophet_features_daily.csv",
                     index_col="Date", parse_dates=True)
    endog = df[ENDOG_COL]
    exog  = df[REGRESSOR_COLS].shift(1)       # lag-1: known at forecast time
    valid = exog.dropna().index               # drop the leading NaN row
    return valid, endog.loc[valid], exog.loc[valid]


class ProphetModel:
    """Recursive multi-step Prophet wrapped for the walk-forward engine.

    Prophet needs a `ds` datetime column, but the engine works positionally and
    passes plain arrays. We therefore hand the model the full aligned date index
    at construction and slice it by training length: after fit(y[:i]) the model
    knows it was trained through position i-1 and forecasts the next h dates.

    observe() only advances a position counter -- there is no cheap state to
    filter, so the fair schedule refits every step (see run()). The counter also
    lets the optional monthly schedule forecast from the correct origin between
    refits (predicting farther ahead from the last fitted model).
    """

    def __init__(self, dates, regressors, growth="flat",
                 seasonality_mode="additive", yearly=True, weekly=True,
                 daily=False, changepoint_prior_scale=0.05):
        self._dates = pd.DatetimeIndex(dates)
        self._regressors = list(regressors)
        self._kw = dict(
            growth=growth,
            seasonality_mode=seasonality_mode,
            yearly_seasonality=yearly,
            weekly_seasonality=weekly,
            daily_seasonality=daily,
            changepoint_prior_scale=changepoint_prior_scale,
            uncertainty_samples=0,            # point forecast only -> much faster
        )
        self.m = None
        self._n_train = 0                     # absolute origin position at last fit
        self._n_seen = 0                      # observes since last fit (monthly)

    def _new_model(self) -> Prophet:
        m = Prophet(**self._kw)
        for col in self._regressors:
            m.add_regressor(col)              # additive, standardised (auto)
        return m

    def fit(self, y_hist, X_hist, t0=0):
        # t0 is the absolute position of the first row of y_hist in the full
        # series. With an expanding window t0=0 and len(y_hist)==origin; with a
        # rolling window t0>0, so we must slice the real calendar dates
        # [t0 : t0+n] (weekly/yearly seasonality is date-dependent) and record
        # the absolute origin position t0+n for forecast().
        n = len(y_hist)
        train = pd.DataFrame({"ds": self._dates[t0:t0 + n],
                              "y": np.asarray(y_hist, dtype=float)})
        if X_hist is not None:
            for j, col in enumerate(self._regressors):
                train[col] = np.asarray(X_hist, dtype=float)[:, j]
        self.m = self._new_model()
        self.m.fit(train)
        self._n_train = t0 + n                # absolute origin position
        self._n_seen = 0

    def forecast(self, horizon, X_next):
        """h-step cumulative forecast: sum of the h forecast daily returns.

        The origin is n_train + n_seen (advanced by observe when not refitting),
        so the future frame uses the real calendar dates of the realised window
        -- correct for the date-dependent weekly/yearly seasonality. With lag-1
        regressors only the first future step's values are known (X_next); steps
        2..h use 0 (unknown future zero-mean returns).
        """
        origin = self._n_train + self._n_seen
        fut = pd.DataFrame({"ds": self._dates[origin:origin + horizon]})
        reg = np.zeros((horizon, len(self._regressors)))
        if X_next is not None:
            reg[0] = np.asarray(X_next, dtype=float).ravel()
        for j, col in enumerate(self._regressors):
            fut[col] = reg[:, j]
        yhat = self.m.predict(fut)["yhat"].to_numpy()
        return float(yhat.sum())

    def observe(self, y_obs, X_obs):
        # No cheap state update in Prophet. Refit-every-step discards this; the
        # counter only matters for the optional monthly schedule.
        self._n_seen += 1


def run(horizon=1, growth="flat", refit="step", train_window=TRAIN_WINDOW,
        max_test_days=None, verbose=True, _data=None):
    print(f"\n[Prophet] {horizon}-day-ahead walk-forward "
          f"(growth={growth}, refit={refit}, lag-1 regressors)")
    dates, endog, exog = _data if _data is not None else load_data()
    y = endog.to_numpy(dtype=float)
    X = exog.to_numpy(dtype=float)

    test_start = int(dates.searchsorted(pd.Timestamp(TEST_START)))
    test_end   = len(y)
    if TEST_END is not None:
        test_end = int(dates.searchsorted(pd.Timestamp(TEST_END), side="right"))
    if max_test_days is not None:
        test_end = min(test_end, test_start + max_test_days)

    win = "expanding" if train_window is None else f"rolling {train_window}d"
    print(f"  Train: {dates[0].date()} .. {dates[test_start-1].date()}  ({test_start} obs, {win})")
    print(f"  Test : {dates[test_start].date()} .. {dates[test_end-1].date()}  ({test_end-test_start} obs)")

    model = ProphetModel(dates, REGRESSOR_COLS, growth=growth)
    if refit == "step":
        refit_positions = set(range(test_start, test_end))   # every origin
    else:
        refit_positions = month_start_positions(dates, test_start, test_end)

    preds = expanding_walk_forward(
        y, X, dates, test_start, model, refit_positions,
        horizon=horizon, test_end=test_end, train_window=train_window,
        verbose=verbose,
    )

    train_drift = float(y[:test_start].mean())
    m = summary(preds["y_true"].to_numpy(), preds["y_pred"].to_numpy(),
                drift=train_drift, horizon=horizon)
    print(f"  Results: n={m['n']}  "
          f"RMSE={m['rmse']:.5f} (zero {m['rmse_zero']:.5f}, drift {m['rmse_drift']:.5f})  "
          f"MAE={m['mae']:.5f} (zero {m['mae_zero']:.5f}, drift {m['mae_drift']:.5f})  "
          f"DA={m['da']:.3f} (up {m['da_up']:.3f}, edge {m['da_edge']:+.3f})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"prophet_{horizon}d.csv"
    preds.to_csv(out)
    print(f"  Saved -> {out}")
    return preds, m


def run_all(horizons=None, growth="flat", refit="step",
            train_window=TRAIN_WINDOW, max_test_days=None):
    """Run all horizons with a SINGLE walk-forward: one fit per origin, all horizons jointly.

    This is the conceptually correct approach for a recursive model: the daily
    Prophet model is identical regardless of the forecast horizon -- only the
    number of steps summed at the end differs. Fitting once per origin (instead
    of once per origin per horizon) cuts compute cost by len(horizons)x.

    train_window mirrors SARIMAX: None => expanding window; an int W => the
    model is refit on the last W days at each origin, so both baselines are
    evaluated under identical rolling-window conditions (see config.TRAIN_WINDOW).
    """
    horizons = horizons or HORIZONS
    win = "expanding" if train_window is None else f"rolling {train_window}d"
    print(f"\n[Prophet] multi-horizon walk-forward "
          f"(growth={growth}, refit={refit}, window={win}, horizons={horizons}, lag-1 regressors)")

    dates, endog, exog = load_data()
    y = endog.to_numpy(dtype=float)
    X = exog.to_numpy(dtype=float)

    test_start = int(dates.searchsorted(pd.Timestamp(TEST_START)))
    test_end   = len(y)
    if TEST_END is not None:
        test_end = int(dates.searchsorted(pd.Timestamp(TEST_END), side="right"))
    if max_test_days is not None:
        test_end = min(test_end, test_start + max_test_days)

    print(f"  Train: {dates[0].date()} .. {dates[test_start-1].date()}  ({test_start} obs, {win})")
    print(f"  Test : {dates[test_start].date()} .. {dates[test_end-1].date()}  ({test_end-test_start} obs)")

    model = ProphetModel(dates, REGRESSOR_COLS, growth=growth)
    refit_positions = (set(range(test_start, test_end)) if refit == "step"
                       else month_start_positions(dates, test_start, test_end))

    t0 = time.time()
    all_preds = expanding_walk_forward_multi_horizon(
        y, X, dates, test_start, model, refit_positions,
        horizons=horizons, test_end=test_end, train_window=train_window,
    )
    n_pred_total = sum(len(df) for df in all_preds.values())
    record("prophet", -1, "test", time.time() - t0,
           refit=refit, n_predictions=n_pred_total)

    train_drift = float(y[:test_start].mean())
    results = {}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for h, preds in all_preds.items():
        m = summary(preds["y_true"].to_numpy(), preds["y_pred"].to_numpy(),
                    drift=train_drift, horizon=h)
        results[h] = m
        out = RESULTS_DIR / f"prophet_{h}d.csv"
        preds.to_csv(out)
        print(f"  Saved -> {out}")

    print(f"\n{'='*64}\n  Prophet summary\n{'='*64}")
    for h, m in results.items():
        print(f"    h={h:>2}d   RMSE {m['rmse']:.5f} (zero {m['rmse_zero']:.5f}, drift {m['rmse_drift']:.5f})"
              f"   MAE {m['mae']:.5f} (zero {m['mae_zero']:.5f}, drift {m['mae_drift']:.5f})"
              f"   DA {m['da']:.3f} (up {m['da_up']:.3f}, edge {m['da_edge']:+.3f})")

    if max_test_days is None:
        save_final_model(growth=growth)

    return results


def run_cv(horizons=None, growth="flat", refit="step",
           train_window=TRAIN_WINDOW, val_years=None, max_val_days=None):
    """Round 1: expanding-window CV over calendar years within dev.

    Same fold definition as SARIMAX (see src.evaluation.cv) so the two
    baselines produce strictly comparable Round-1 validation metrics. Prophet
    has no structure selection step, so every fold uses the same specification
    -- only the (auto-selected) trend, seasonality and regressor coefficients
    change across folds as more data becomes available.
    """
    horizons = horizons or HORIZONS
    val_years = val_years or DEFAULT_VAL_YEARS
    win = "expanding" if train_window is None else f"rolling {train_window}d"
    print(f"\n[Prophet] CV ({len(val_years)} folds: val={val_years}) "
          f"(growth={growth}, refit={refit}, window={win}, horizons={horizons})")

    dates, endog, exog = load_data()
    y = endog.to_numpy(dtype=float)
    X = exog.to_numpy(dtype=float)
    folds = year_folds(dates, val_years=val_years, train_start_date=TRAIN_START)

    cv_dir = RESULTS_DIR / "cv"
    cv_dir.mkdir(parents=True, exist_ok=True)

    per_fold: dict[int, dict[int, dict]] = {h: {} for h in horizons}
    t0 = time.time()

    for f in folds:
        print(f"\n  --- Fold val={f.val_year}: "
              f"train {dates[f.train_start].date()}..{dates[f.train_end-1].date()} "
              f"({f.n_train} obs), val {dates[f.val_start].date()}..{dates[f.val_end-1].date()} "
              f"({f.n_val} obs) ---")

        val_end = (min(f.val_end, f.val_start + max_val_days)
                   if max_val_days is not None else f.val_end)
        model = ProphetModel(dates, REGRESSOR_COLS, growth=growth)
        refit_positions = (set(range(f.val_start, val_end)) if refit == "step"
                           else month_start_positions(dates, f.val_start, val_end))

        all_preds = expanding_walk_forward_multi_horizon(
            y, X, dates, f.val_start, model, refit_positions,
            horizons=horizons, test_end=val_end, train_window=train_window,
        )

        fold_drift = float(y[f.train_start:f.train_end].mean())
        for h, preds in all_preds.items():
            m = summary(preds["y_true"].to_numpy(), preds["y_pred"].to_numpy(),
                        drift=fold_drift, horizon=h)
            per_fold[h][f.val_year] = m
            preds.to_csv(cv_dir / f"prophet_fold{f.val_year}_{h}d.csv")
            print(f"    h={h:>2}d   n={m['n']}  "
                  f"RMSE={m['rmse']:.5f} (zero {m['rmse_zero']:.5f}, drift {m['rmse_drift']:.5f})  "
                  f"MAE={m['mae']:.5f}  DA={m['da']:.3f} (edge {m['da_edge']:+.3f})")

    dt = time.time() - t0
    print(f"\n{'='*72}\n  Prophet CV summary  (mean +/- std across {len(folds)} folds)"
          f"\n{'='*72}")
    agg = {}
    n_pred_total = 0
    for h in horizons:
        agg[h] = aggregate_metrics(per_fold[h])
        n_pred_total += int(agg[h]["n_total"])
    record("prophet", -1, "cv", dt, refit=refit, n_predictions=n_pred_total)
    for h in horizons:
        m = agg[h]
        print(f"    h={h:>2}d   "
              f"RMSE {m['rmse']:.5f} +/- {m['rmse_std']:.5f}   "
              f"MAE {m['mae']:.5f} +/- {m['mae_std']:.5f}   "
              f"DA {m['da']:.3f} +/- {m['da_std']:.3f}   "
              f"edge {m['da_edge']:+.3f} +/- {m['da_edge_std']:.3f}   "
              f"n_total={m['n_total']}")
    return agg, per_fold


def save_final_model(growth="flat"):
    """Fit Prophet on full history and pickle to models/saved/prophet_final.pkl.

    Prophet is horizon-agnostic: the same daily model is used for all horizons,
    with h steps summed at forecast time. One pickle covers all horizons.
    """
    import json, pickle
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dates, endog, exog = load_data()
    y = endog.to_numpy(dtype=float)
    X = exog.to_numpy(dtype=float)

    test_end = len(y)
    if TEST_END is not None:
        test_end = int(dates.searchsorted(pd.Timestamp(TEST_END), side="right"))

    model_obj = ProphetModel(dates, REGRESSOR_COLS, growth=growth)
    model_obj.fit(y[:test_end], X[:test_end])

    out_pkl = MODELS_DIR / "prophet_final.pkl"
    with open(out_pkl, "wb") as fh:
        pickle.dump(model_obj, fh)

    meta = {
        "growth": growth,
        "regressors": REGRESSOR_COLS,
        "train_end_date": str(dates[test_end - 1].date()),
        "horizons": "all (sum h daily forecasts at inference time)",
    }
    with open(MODELS_DIR / "prophet_final_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  [Prophet] saved -> {out_pkl}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Prophet multi-horizon walk-forward")
    ap.add_argument("--horizon", type=int, default=None,
                    help="single horizon in days (default: all in config.HORIZONS)")
    ap.add_argument("--growth", choices=["flat", "linear"], default="flat",
                    help="Prophet trend (default: flat, the SARIMAX trend='c' analog)")
    ap.add_argument("--refit", choices=["step", "monthly"], default="step",
                    help="refit every origin (default, comparable to SARIMAX) or monthly (fast dev)")
    ap.add_argument("--max-test-days", type=int, default=None,
                    help="limit the test horizon for a quick smoke run")
    ap.add_argument("--cv", action="store_true",
                    help="run Round-1 cross-validation instead of the test walk-forward")
    ap.add_argument("--max-val-days", type=int, default=None,
                    help="limit each CV fold's val length for a quick smoke run")
    args = ap.parse_args()
    if args.cv:
        run_cv(growth=args.growth, refit=args.refit, max_val_days=args.max_val_days)
    elif args.horizon is None:
        run_all(growth=args.growth, refit=args.refit, max_test_days=args.max_test_days)
    else:
        run(horizon=args.horizon, growth=args.growth, refit=args.refit,
            max_test_days=args.max_test_days)
