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
Regressor sel.: backward elimination by significance on the training window
                (drop exogenous regressors with p >= 0.05, re-fit on the rest).
                Done once on train only -> no look-ahead bias. Disable with
                --no-select to keep all five regressors.
Trend         : a constant drift term (trend='c') is included so the model
                captures BTC's unconditional upward drift. It barely moves the
                1-day forecast but accumulates over multi-step horizons (the
                h-day forecast picks up ~h*c). Any directional-accuracy gain at
                long horizons reflects this drift, not conditional skill, so DA
                is read against an "always predict the drift sign" reference,
                not 0.5.
Validation    : expanding-window walk-forward, daily refit (matches Prophet).
                Each day the model is re-estimated on the full expanding
                history before forecasting. The last h origins are not scored
                (no realised h-day target yet).

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

from src.config import TRAIN_START, TEST_START, TEST_END, HORIZONS, TRAIN_WINDOW
from src.evaluation.cv import DEFAULT_VAL_YEARS, aggregate_metrics, year_folds
from src.evaluation.metrics import summary
from src.evaluation.runtime import record
from src.evaluation.walk_forward import (expanding_walk_forward,
                                         expanding_walk_forward_multi_horizon,
                                         month_start_positions)
import time

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
RESULTS_DIR   = Path(__file__).resolve().parents[2] / "reports" / "predictions"
METRICS_DIR   = Path(__file__).resolve().parents[2] / "reports" / "metrics"
MODELS_DIR    = Path(__file__).resolve().parents[2] / "models" / "saved"

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


def select_regressors(y, X, order, exog_cols, alpha=0.05, trend="c",
                      return_pvalues=False):
    """Backward elimination of exogenous regressors by significance.

    Fits the full model once on the given (training) window and keeps only the
    regressors whose coefficient is significant at level alpha. Because the fit
    uses training data only, the selection introduces no look-ahead bias.

    Returns the list of kept column indices into exog_cols (in original order);
    an empty list means no regressor survived and the model reduces to drift
    plus noise. statsmodels names numpy exog columns x1..xk in column order, so
    we read each regressor's p-value by that name. With return_pvalues=True the
    function also returns {col: p-value} for every candidate (used to persist the
    per-fold selection spec).
    """
    res = SARIMAX(y, exog=X, order=order, trend=trend,
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    names = list(res.param_names)
    pvals = np.asarray(res.pvalues)
    kept, pval_by_col = [], {}
    for i, col in enumerate(exog_cols):
        p = float(pvals[names.index(f"x{i + 1}")])
        pval_by_col[col] = p
        print(f"    {col:22s} p={p:.4f}  {'KEEP' if p < alpha else 'drop'}")
        if p < alpha:
            kept.append(i)
    kept_names = [exog_cols[i] for i in kept] or ["(none)"]
    print(f"  Regressor selection (alpha={alpha}): kept {kept_names}")
    return (kept, pval_by_col) if return_pvalues else kept


# ---------------------------------------------------------------------------
# Model-parameter reporting  (structure + fitted coefficients)
# ---------------------------------------------------------------------------
# The pipeline saves predictions and metrics, but the *fitted model* was never
# persisted. SARIMAX has few, fully interpretable parameters, so we record both
# tiers of the two-round protocol:
#   * structure (order + kept regressors) -- selected once in Round 1;
#   * coefficients (const, regressor betas, sigma2) -- re-estimated every refit,
#     snapshotted at the two protocol reference points: end of development
#     (fit on 2018-2023) and end of test (fit on the full 2018-2026 span).

def _param_rows(res, phase, kept_cols):
    """Tidy [{phase, param, coef, std_err, pvalue}] rows from a fitted result.

    statsmodels labels exogenous coefficients x1..xk in the order they were
    passed; relabel them with the kept column names so the CSV is readable.
    """
    label = {f"x{i + 1}": col for i, col in enumerate(kept_cols)}
    names = list(res.param_names)
    coefs = np.asarray(res.params,  dtype=float)
    ses   = np.asarray(res.bse,     dtype=float)
    pvals = np.asarray(res.pvalues, dtype=float)
    return [{"phase": phase, "param": label.get(nm, nm),
             "coef": float(c), "std_err": float(se), "pvalue": float(p)}
            for nm, c, se, p in zip(names, coefs, ses, pvals)]


def _spec_row(val_year, order, trend, keep_idx, pval_by_col):
    """One CV-fold structure row: selected order, kept regressors, and the
    p-value of every candidate regressor on that fold's training window."""
    kept_cols = [EXOG_COLS[i] for i in keep_idx] if keep_idx else []
    return {"val_year": val_year, "order": str(order), "trend": trend,
            "regressors_kept": ",".join(kept_cols) or "(none)",
            **{f"pval_{c}": round(pval_by_col.get(c, float("nan")), 4)
               for c in EXOG_COLS}}


def _save_test_report(dates, y, X_full, test_start, test_end, order, trend,
                      keep_idx, train_window):
    """Persist frozen structure + fitted coefficients at the two protocol
    reference points: end of development (fit on 2018-2023) and end of test
    (fit on the full 2018-2026 span). The structure is identical across both
    (selected once on development); only the coefficients are re-estimated on
    the longer window, so the two rows show whether the drift / ETH beta shifted
    once the 2024-2026 data arrived."""
    kept_cols = [EXOG_COLS[i] for i in keep_idx] if keep_idx else []

    def _fit(hi):
        lo = 0 if train_window is None else max(0, hi - train_window)
        Xk = X_full[lo:hi][:, keep_idx] if keep_idx else None
        res = SARIMAX(y[lo:hi], exog=Xk, order=order, trend=trend,
                      enforce_stationarity=False,
                      enforce_invertibility=False).fit(disp=False)
        return res, lo

    res_dev, lo_dev = _fit(test_start)
    res_fin, lo_fin = _fit(test_end)

    spec = pd.DataFrame([
        {"phase": "development",
         "start": dates[lo_dev].date(), "end": dates[test_start - 1].date(),
         "n_obs": test_start - lo_dev, "order": str(order), "trend": trend,
         "regressors_kept": ",".join(kept_cols) or "(none)",
         "aic": round(float(res_dev.aic), 3)},
        {"phase": "test_final",
         "start": dates[lo_fin].date(), "end": dates[test_end - 1].date(),
         "n_obs": test_end - lo_fin, "order": str(order), "trend": trend,
         "regressors_kept": ",".join(kept_cols) or "(none)",
         "aic": round(float(res_fin.aic), 3)},
    ])
    coefs = pd.DataFrame(_param_rows(res_dev, "development", kept_cols)
                         + _param_rows(res_fin, "test_final", kept_cols))

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    spec.to_csv(METRICS_DIR / "sarimax_spec.csv", index=False)
    coefs.to_csv(METRICS_DIR / "sarimax_coefs.csv", index=False)
    print(f"  Model report -> {METRICS_DIR / 'sarimax_spec.csv'} | "
          f"{METRICS_DIR / 'sarimax_coefs.csv'}")


def _cv_spec_rows(dates, y, X, trend, select):
    """Per-fold structure-selection rows, re-run cheaply (no walk-forward)."""
    folds = year_folds(dates, val_years=DEFAULT_VAL_YEARS,
                       train_start_date=TRAIN_START)
    rows = []
    for f in folds:
        order = select_order(y[f.train_start:f.train_end],
                             X[f.train_start:f.train_end], trend=trend)
        keep_idx, pval_by_col = list(range(len(EXOG_COLS))), {}
        if select:
            keep_idx, pval_by_col = select_regressors(
                y[f.train_start:f.train_end], X[f.train_start:f.train_end],
                order, EXOG_COLS, trend=trend, return_pvalues=True)
        rows.append(_spec_row(f.val_year, order, trend, keep_idx, pval_by_col))
    return rows


class SarimaxModel:
    """Recursive multi-step SARIMAX wrapped for the walk-forward engine."""

    def __init__(self, order, trend="c", seasonal_order=(0, 0, 0, 0)):
        self.order = order
        self.trend = trend
        self.seasonal_order = seasonal_order
        self.res = None

    def fit(self, y_hist, X_hist, t0=0):
        # t0 (absolute start position) is unused: statsmodels works positionally
        # on the values, so a rolling window is just a shorter y_hist/X_hist.
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


def run(horizon=1, order=None, trend="c", keep_idx=None, select=True,
        train_window=TRAIN_WINDOW, max_test_days=None, verbose=True, _data=None):
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

    win = "expanding" if train_window is None else f"rolling {train_window}d"
    print(f"  Train: {dates[0].date()} .. {dates[test_start-1].date()}  ({test_start} obs, {win})")
    print(f"  Test : {dates[test_start].date()} .. {dates[test_end-1].date()}  ({test_end-test_start} obs)")

    if order is None:
        order = select_order(y[:test_start], X[:test_start], trend=trend)
    if select and keep_idx is None:
        keep_idx = select_regressors(y[:test_start], X[:test_start], order,
                                     EXOG_COLS, trend=trend)
    if keep_idx is not None:                    # apply the reduced regressor set
        X = X[:, keep_idx] if keep_idx else None

    model = SarimaxModel(order=order, trend=trend)
    refit = set(range(test_start, test_end))   # daily refit — matches Prophet

    preds = expanding_walk_forward(
        y, X, dates, test_start, model, refit,
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
    out = RESULTS_DIR / f"sarimax_{horizon}d.csv"
    preds.to_csv(out)
    print(f"  Saved -> {out}")
    return preds, m


def run_all(horizons=None, trend="c", select=True, refit="step",
            train_window=TRAIN_WINDOW, max_test_days=None):
    """Run all horizons with a SINGLE walk-forward: one fit per origin, all horizons jointly.

    This is the conceptually correct approach for a recursive model, and the
    same one Prophet uses: the daily SARIMAX model is identical regardless of
    the forecast horizon -- only the number of summed steps differs. Fitting
    once per origin (instead of once per origin per horizon) cuts compute cost
    by len(horizons)x with byte-identical results to running each horizon
    separately.

    Two time-scales of adaptation:
      * structure (order + significant regressor subset) is selected ONCE on
        the full initial training window and frozen across the test -- a
        low-frequency, structural decision reused across all horizons;
      * coefficient values are re-estimated at each refit on a rolling
        train_window of recent history (see config.TRAIN_WINDOW), so they
        track the current regime rather than averaging in the thin 2018-19
        market. train_window=None restores the expanding window.

    refit : "step" re-estimates every origin (matches the per-horizon default
            and Prophet); "monthly" refits at each month start and relies on
            SarimaxModel.observe's cheap Kalman filtering in between.
    """
    horizons = horizons or HORIZONS
    win = "expanding" if train_window is None else f"rolling {train_window}d"
    print(f"\n[SARIMAX] multi-horizon walk-forward "
          f"(trend={trend}, refit={refit}, window={win}, horizons={horizons}, lag-1 exogenous)")

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

    order = select_order(y[:test_start], X[:test_start], trend=trend)
    keep_idx = list(range(len(EXOG_COLS)))
    if select:
        keep_idx = select_regressors(y[:test_start], X[:test_start], order,
                                     EXOG_COLS, trend=trend)
    X_full = X                                   # keep the unreduced design for reporting
    X = X[:, keep_idx] if keep_idx else None

    model = SarimaxModel(order=order, trend=trend)
    refit_positions = (set(range(test_start, test_end)) if refit == "step"
                       else month_start_positions(dates, test_start, test_end))

    t0 = time.time()
    all_preds = expanding_walk_forward_multi_horizon(
        y, X, dates, test_start, model, refit_positions,
        horizons=horizons, test_end=test_end, train_window=train_window,
    )
    n_pred_total = sum(len(df) for df in all_preds.values())
    record("sarimax", -1, "test", time.time() - t0,
           refit=refit, n_predictions=n_pred_total)

    train_drift = float(y[:test_start].mean())
    results = {}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for h, preds in all_preds.items():
        m = summary(preds["y_true"].to_numpy(), preds["y_pred"].to_numpy(),
                    drift=train_drift, horizon=h)
        results[h] = m
        out = RESULTS_DIR / f"sarimax_{h}d.csv"
        preds.to_csv(out)
        print(f"  Saved -> {out}")

    print(f"\n{'='*64}\n  SARIMAX summary  (RMSE / MAE / DA  vs predict-zero RMSE)\n{'='*64}")
    for h, m in results.items():
        print(f"    h={h:>2}d   RMSE {m['rmse']:.5f} (zero {m['rmse_zero']:.5f}, drift {m['rmse_drift']:.5f})"
              f"   MAE {m['mae']:.5f} (zero {m['mae_zero']:.5f}, drift {m['mae_drift']:.5f})"
              f"   DA {m['da']:.3f} (up {m['da_up']:.3f}, edge {m['da_edge']:+.3f})")

    if max_test_days is None:                    # skip on smoke runs (partial test)
        _save_test_report(dates, y, X_full, test_start, test_end,
                          order, trend, keep_idx, train_window)
        save_final_model(dates, y, X_full, test_end,
                         order, trend, keep_idx, train_window)
    return results


def run_cv(horizons=None, trend="c", select=True, refit="step",
           train_window=TRAIN_WINDOW, val_years=None, max_val_days=None,
           verbose=False):
    """Round 1: expanding-window CV over calendar years within dev.

    For each fold (val = 2020, 2021, 2022, 2023 by default):
      1. Structure selection is re-done on that fold's TRAIN portion only
         (select_order + backward elimination of regressors); nothing from val
         ever enters the fit. This mirrors what would happen if the fold were
         the only data available at that moment.
      2. Walk-forward through the val year with daily refits, forecasting all
         horizons jointly (one fit per origin, len(horizons)x cheaper than per-h).
      3. Metrics per fold per horizon are recorded and predictions saved.

    Finally the four fold summaries are aggregated (mean +/- std across folds)
    and printed as the Round-1 validation metrics that will populate the
    Validation column of the final comparison table.
    """
    horizons = horizons or HORIZONS
    val_years = val_years or DEFAULT_VAL_YEARS
    win = "expanding" if train_window is None else f"rolling {train_window}d"
    print(f"\n[SARIMAX] CV ({len(val_years)} folds: val={val_years}) "
          f"(trend={trend}, refit={refit}, window={win}, horizons={horizons})")

    dates, endog, exog = load_data()
    y = endog.to_numpy(dtype=float)
    X_full = exog.to_numpy(dtype=float)
    folds = year_folds(dates, val_years=val_years, train_start_date=TRAIN_START)

    cv_dir = RESULTS_DIR / "cv"
    cv_dir.mkdir(parents=True, exist_ok=True)

    per_fold: dict[int, dict[int, dict]] = {h: {} for h in horizons}
    spec_rows: list[dict] = []
    t0 = time.time()

    for f in folds:
        print(f"\n  --- Fold val={f.val_year}: "
              f"train {dates[f.train_start].date()}..{dates[f.train_end-1].date()} "
              f"({f.n_train} obs), val {dates[f.val_start].date()}..{dates[f.val_end-1].date()} "
              f"({f.n_val} obs) ---")

        # Structure selection on the fold's TRAIN portion only (no leakage).
        order = select_order(y[f.train_start:f.train_end],
                             X_full[f.train_start:f.train_end], trend=trend)
        keep_idx, pval_by_col = list(range(len(EXOG_COLS))), {}
        if select:
            keep_idx, pval_by_col = select_regressors(
                y[f.train_start:f.train_end], X_full[f.train_start:f.train_end],
                order, EXOG_COLS, trend=trend, return_pvalues=True)
        X = X_full[:, keep_idx] if keep_idx else None
        spec_rows.append(_spec_row(f.val_year, order, trend, keep_idx, pval_by_col))

        val_end = (min(f.val_end, f.val_start + max_val_days)
                   if max_val_days is not None else f.val_end)
        model = SarimaxModel(order=order, trend=trend)
        refit_positions = (set(range(f.val_start, val_end)) if refit == "step"
                           else month_start_positions(dates, f.val_start, val_end))

        all_preds = expanding_walk_forward_multi_horizon(
            y, X, dates, f.val_start, model, refit_positions,
            horizons=horizons, test_end=val_end, train_window=train_window,
            verbose=verbose,
        )

        # Fold drift uses TRAIN mean only (no look-ahead into val).
        fold_drift = float(y[f.train_start:f.train_end].mean())
        for h, preds in all_preds.items():
            m = summary(preds["y_true"].to_numpy(), preds["y_pred"].to_numpy(),
                        drift=fold_drift, horizon=h)
            per_fold[h][f.val_year] = m
            preds.to_csv(cv_dir / f"sarimax_fold{f.val_year}_{h}d.csv")
            print(f"    h={h:>2}d   n={m['n']}  "
                  f"RMSE={m['rmse']:.5f} (zero {m['rmse_zero']:.5f}, drift {m['rmse_drift']:.5f})  "
                  f"MAE={m['mae']:.5f}  DA={m['da']:.3f} (edge {m['da_edge']:+.3f})")

    dt = time.time() - t0
    print(f"\n{'='*72}\n  SARIMAX CV summary  (mean +/- std across {len(folds)} folds)"
          f"\n{'='*72}")
    agg = {}
    n_pred_total = 0
    for h in horizons:
        agg[h] = aggregate_metrics(per_fold[h])
        n_pred_total += int(agg[h]["n_total"])
    record("sarimax", -1, "cv", dt, refit=refit, n_predictions=n_pred_total)
    for h in horizons:
        m = agg[h]
        print(f"    h={h:>2}d   "
              f"RMSE {m['rmse']:.5f} +/- {m['rmse_std']:.5f}   "
              f"MAE {m['mae']:.5f} +/- {m['mae_std']:.5f}   "
              f"DA {m['da']:.3f} +/- {m['da_std']:.3f}   "
              f"edge {m['da_edge']:+.3f} +/- {m['da_edge_std']:.3f}   "
              f"n_total={m['n_total']}")

    if max_val_days is None:                     # skip on smoke runs (partial val)
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(spec_rows).to_csv(METRICS_DIR / "sarimax_spec_cv.csv", index=False)
        print(f"  CV structure -> {METRICS_DIR / 'sarimax_spec_cv.csv'}")
    return agg, per_fold


def save_final_model(dates, y, X_full, test_end, order, trend,
                     keep_idx, train_window):
    """Fit SARIMAX on the full available history and pickle the result.

    This is the model you would use to predict tomorrow: fitted on every
    observation from TRAIN_START through the last test date, with the same
    structure (order, kept regressors) selected during Round 1."""
    import pickle, json
    kept_cols = [EXOG_COLS[i] for i in keep_idx] if keep_idx else []
    lo  = 0 if train_window is None else max(0, test_end - train_window)
    Xk  = X_full[lo:test_end][:, keep_idx] if keep_idx else None
    res = SARIMAX(y[lo:test_end], exog=Xk, order=order, trend=trend,
                  enforce_stationarity=False,
                  enforce_invertibility=False).fit(disp=False)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "sarimax_final.pkl", "wb") as f:
        pickle.dump(res, f)
    meta = {"order": list(order), "trend": trend,
            "regressors": kept_cols,
            "train_start": str(dates[lo].date()),
            "train_end":   str(dates[test_end - 1].date()),
            "n_obs": test_end - lo}
    (MODELS_DIR / "sarimax_final_meta.json").write_text(
        __import__("json").dumps(meta, indent=2), encoding="utf-8")
    print(f"  Final model -> {MODELS_DIR / 'sarimax_final.pkl'}")


def report_only(trend="c", select=True, train_window=TRAIN_WINDOW):
    """Regenerate the SARIMAX structure + coefficient reports without running the
    (expensive) walk-forward. Fits only the development, test-final, and per-fold
    models -- a handful of cheap fits -- and rewrites sarimax_spec.csv,
    sarimax_coefs.csv and sarimax_spec_cv.csv."""
    print("\n[SARIMAX] report-only: regenerating spec + coefficient CSVs")
    dates, endog, exog = load_data()
    y = endog.to_numpy(dtype=float)
    X = exog.to_numpy(dtype=float)

    test_start = int(dates.searchsorted(pd.Timestamp(TEST_START)))
    test_end   = len(y)
    if TEST_END is not None:
        test_end = int(dates.searchsorted(pd.Timestamp(TEST_END), side="right"))

    order = select_order(y[:test_start], X[:test_start], trend=trend)
    keep_idx = list(range(len(EXOG_COLS)))
    if select:
        keep_idx = select_regressors(y[:test_start], X[:test_start], order,
                                     EXOG_COLS, trend=trend)
    _save_test_report(dates, y, X, test_start, test_end, order, trend,
                      keep_idx, train_window)

    spec_rows = _cv_spec_rows(dates, y, X, trend, select)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(spec_rows).to_csv(METRICS_DIR / "sarimax_spec_cv.csv", index=False)
    print(f"  CV structure -> {METRICS_DIR / 'sarimax_spec_cv.csv'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="SARIMAX multi-horizon walk-forward")
    ap.add_argument("--horizon", type=int, default=None,
                    help="single horizon in days (default: all in config.HORIZONS)")
    ap.add_argument("--max-test-days", type=int, default=None,
                    help="limit the test horizon for a quick smoke run")
    ap.add_argument("--no-select", action="store_true",
                    help="keep all exogenous regressors (skip significance selection)")
    ap.add_argument("--refit", choices=["step", "monthly"], default="step",
                    help="refit every origin (step, default) or monthly (dev)")
    ap.add_argument("--cv", action="store_true",
                    help="run Round-1 cross-validation instead of the test walk-forward")
    ap.add_argument("--max-val-days", type=int, default=None,
                    help="limit each CV fold's val length for a quick smoke run")
    ap.add_argument("--report-only", action="store_true",
                    help="regenerate structure/coefficient CSVs without the walk-forward")
    args = ap.parse_args()
    select = not args.no_select
    if args.report_only:
        report_only(select=select)
    elif args.cv:
        run_cv(select=select, refit=args.refit, max_val_days=args.max_val_days)
    elif args.horizon is None:
        run_all(select=select, refit=args.refit, max_test_days=args.max_test_days)
    else:
        run(horizon=args.horizon, select=select, max_test_days=args.max_test_days)
