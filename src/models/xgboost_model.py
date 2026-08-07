"""
XGBoost direct multi-horizon walk-forward forecasting for Bitcoin log-returns.

Reads   : data/processed/xgboost_features_daily.csv  (26 features, 3 targets)
Outputs : reports/predictions/xgboost_{h}d.csv           (test walk-forward)
          reports/predictions/cv/xgboost_fold{year}_{h}d.csv  (CV predictions)
          reports/predictions/cv/xgboost_best_hparams.csv     (best combo per h)

Strategy:
  * DIRECT multi-horizon -- three independent models, one per horizon, each
    trained to predict target_log_return_{h}d = ln(P_{t+h} / P_t) directly.
    No recursive rollout: the target is the cumulative h-day return itself.
  * PURGE h rows at the end of every train window: row t's target uses
    P_{t+h}, so training rows in (i - h, i - 1] would leak future prices
    into a fit made at position i. Keep only rows with j <= i - h.
  * Round 1 (CV): STATIC train per fold -- fit once on the fold's train
    portion, predict all val rows in one batch. Cheap approximation to
    walk-forward inside val; the point of Round 1 is to rank hyperparameters,
    not to produce publishable metrics.
  * Round 2 (test): MONTHLY refit by default -- at the first origin of each
    calendar month, refit on all rows whose target is fully observable
    (positions [0, i - h + 1)) and reuse that model for every day of the
    month. Trees are stable to a one-day extension of the training set, so
    daily refitting adds compute cost without meaningful adaptation. LSTM
    already refits monthly for the same reason; matching XGBoost to that
    cadence keeps the two ML families comparable and slashes total fit count
    from ~883 to ~30 per horizon. Pass ``refit="step"`` to force the legacy
    daily-refit protocol for the SARIMAX/Prophet parity experiment.
  * Fixed base hyperparameters: reg:squarederror, tree_method="hist", uniform
    sample weights, 10% of the (chronological) train tail used as internal
    val for early stopping (auto-selects n_estimators).
  * Small manual grid (8 combos): max_depth x learning_rate x subsample.

Usage:
    python -m src.models.xgboost_model --cv                       # Round 1 CV grid
    python -m src.models.xgboost_model                             # Round 2 test (default hparams)
    python -m src.models.xgboost_model --horizon 7 --max-test-days 60   # smoke run
"""
from __future__ import annotations
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import (HORIZONS, SEED, TEST_END, TEST_START, TRAIN_START)
from src.evaluation.cv import (DEFAULT_VAL_YEARS, aggregate_metrics, year_folds)
from src.evaluation.metrics import summary
from src.evaluation.runtime import measure, record
from src.evaluation.walk_forward import month_start_positions
import time

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
RESULTS_DIR   = Path(__file__).resolve().parents[2] / "reports" / "predictions"
METRICS_DIR   = Path(__file__).resolve().parents[2] / "reports" / "metrics"
MODELS_DIR    = Path(__file__).resolve().parents[2] / "models" / "saved"

TARGET_TEMPLATE = "target_log_return_{h}d"

# Fixed base hyperparameters. Values in GRID override max_depth / learning_rate
# / subsample; everything else is held constant to keep the grid interpretable.
BASE_PARAMS = dict(
    n_estimators=1000,          # upper bound; early stopping picks the actual number
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=SEED,
    verbosity=0,
)

# 2 x 2 x 2 = 8 combinations. See BASE_PARAMS for values held fixed.
GRID = [
    dict(max_depth=d, learning_rate=lr, subsample=s)
    for d, lr, s in product([3, 5], [0.05, 0.1], [0.8, 1.0])
]

# Safe defaults used by run() when no hparams have been selected by CV yet.
DEFAULT_HPARAMS = dict(max_depth=3, learning_rate=0.05, subsample=0.8)

EARLY_STOPPING_ROUNDS = 20
VAL_FRACTION          = 0.10        # tail of train used as internal val for ES


def load_data(horizon: int):
    """Load the XGBoost feature CSV and return (dates, y, X) for one horizon.

    Rows without a fully observable h-day target are dropped (the tail of the
    series, where P_{t+h} is unknown). Feature columns are all non-target
    columns of the CSV, in file order.
    """
    df = pd.read_csv(PROCESSED_DIR / "xgboost_features_daily.csv",
                     index_col="Date", parse_dates=True)
    target_col   = TARGET_TEMPLATE.format(h=horizon)
    feature_cols = [c for c in df.columns if not c.startswith("target_")]
    df = df.dropna(subset=[target_col])          # drop last h rows (unobservable target)
    return df.index, df[target_col].to_numpy(dtype=float), df[feature_cols].to_numpy(dtype=float)


def _fit_es(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> xgb.XGBRegressor:
    """Fit XGBoost with a chronological tail-of-train val split for early stopping.

    A random split would leak future returns into the internal val. Splitting
    on time keeps the val strictly after the training portion, so the estimated
    stopping round reflects generalisation to a genuinely later period.
    """
    n = len(X_train)
    n_val = max(1, int(round(n * VAL_FRACTION)))
    X_tr, X_va = X_train[:-n_val], X_train[-n_val:]
    y_tr, y_va = y_train[:-n_val], y_train[-n_val:]
    model = xgb.XGBRegressor(**params, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model


# ---------------------------------------------------------------------------
# Round 1 -- CV grid search (static train per fold)
# ---------------------------------------------------------------------------

def _cv_one_combo(horizon: int, params: dict, dates, y, X, folds,
                  max_val_days: int | None, cv_dir: Path, tag: str,
                  save_preds: bool):
    """Run one hyperparameter combo across all CV folds and return per-fold metrics."""
    per_fold: dict[int, dict] = {}
    for f in folds:
        # Purge h rows at end of train so no training target uses prices in val.
        train_end = f.train_end - horizon
        if train_end <= f.train_start:
            raise ValueError(f"Fold val={f.val_year}: train too short after purge")
        X_tr, y_tr = X[f.train_start:train_end], y[f.train_start:train_end]

        val_end = (min(f.val_end, f.val_start + max_val_days)
                   if max_val_days is not None else f.val_end)
        X_va, y_va = X[f.val_start:val_end], y[f.val_start:val_end]

        model = _fit_es(X_tr, y_tr, params)
        y_pred = model.predict(X_va)

        # y_tr is the h-day cumulative target; divide by h to recover mu_daily
        # (see run() for the full explanation).
        fold_drift = float(y_tr.mean()) / horizon
        m = summary(y_va, y_pred, drift=fold_drift, horizon=horizon)
        per_fold[f.val_year] = m

        if save_preds:
            fold_dates = dates[f.val_start:val_end]
            df = pd.DataFrame({"y_true": y_va, "y_pred": y_pred}, index=fold_dates)
            df.index.name = "date"
            df.to_csv(cv_dir / f"xgboost_{tag}_fold{f.val_year}_{horizon}d.csv")
    return per_fold


def run_cv(horizons=None, val_years=None, max_val_days=None, save_preds=False,
           verbose=True):
    """Round 1 grid search: for each horizon try all GRID combos over the 4
    folds and select the combination with the lowest aggregate RMSE.

    save_preds=False by default: only the best combo's predictions per fold are
    written to disk (avoids 24 CSVs per horizon during a full search).
    """
    horizons  = horizons  or HORIZONS
    val_years = val_years or DEFAULT_VAL_YEARS
    print(f"\n[XGBoost] CV grid search: {len(GRID)} combos x {len(val_years)} folds "
          f"x {len(horizons)} horizons")

    cv_dir = RESULTS_DIR / "cv"
    cv_dir.mkdir(parents=True, exist_ok=True)

    best_by_h: dict[int, dict] = {}
    hparams_rows = []

    for h in horizons:
        t0 = time.time()
        print(f"\n  --- horizon = {h}d ---")
        dates, y, X = load_data(h)
        folds = year_folds(dates, val_years=val_years, train_start_date=TRAIN_START)

        combo_results = []           # list of (combo_dict, agg, per_fold)
        for combo in GRID:
            params = {**BASE_PARAMS, **combo}
            per_fold = _cv_one_combo(h, params, dates, y, X, folds,
                                     max_val_days=max_val_days, cv_dir=cv_dir,
                                     tag=f"d{combo['max_depth']}_lr{combo['learning_rate']}_s{combo['subsample']}",
                                     save_preds=False)
            agg = aggregate_metrics(per_fold)
            combo_results.append((combo, agg, per_fold))
            if verbose:
                print(f"    max_depth={combo['max_depth']}  lr={combo['learning_rate']}  "
                      f"sub={combo['subsample']}   "
                      f"RMSE {agg['rmse']:.5f} +/- {agg['rmse_std']:.5f}   "
                      f"DA {agg['da']:.3f}   edge {agg['da_edge']:+.3f}   "
                      f"n_total={agg['n_total']}")

        # Selection criterion: lowest aggregate RMSE (DA edge as tiebreaker).
        combo_results.sort(key=lambda r: (r[1]["rmse"], -r[1]["da_edge"]))
        best_combo, best_agg, best_per_fold = combo_results[0]
        best_by_h[h] = {"combo": best_combo, "agg": best_agg, "per_fold": best_per_fold}

        print(f"  Best  h={h}d: max_depth={best_combo['max_depth']}, "
              f"lr={best_combo['learning_rate']}, subsample={best_combo['subsample']}")
        print(f"    -> RMSE {best_agg['rmse']:.5f} +/- {best_agg['rmse_std']:.5f}   "
              f"MAE {best_agg['mae']:.5f}   DA {best_agg['da']:.3f}   "
              f"edge {best_agg['da_edge']:+.3f}   n_total={best_agg['n_total']}")

        # Save the best combo's per-fold predictions and aggregate row for the thesis.
        for fold_year, fold_df in _combo_preds_frames(h, {**BASE_PARAMS, **best_combo},
                                                     dates, y, X, folds, max_val_days).items():
            fold_df.to_csv(cv_dir / f"xgboost_fold{fold_year}_{h}d.csv")

        hparams_rows.append({
            "horizon": h,
            **best_combo,
            "rmse": best_agg["rmse"], "rmse_std": best_agg["rmse_std"],
            "mae":  best_agg["mae"],  "mae_std":  best_agg["mae_std"],
            "da":   best_agg["da"],   "da_std":   best_agg["da_std"],
            "da_edge": best_agg["da_edge"], "da_edge_std": best_agg["da_edge_std"],
            "n_total": best_agg["n_total"],
        })
        record("xgboost", h, "cv", time.time() - t0,
               refit="static-per-fold",
               n_predictions=int(best_agg["n_total"]))

    pd.DataFrame(hparams_rows).to_csv(cv_dir / "xgboost_best_hparams.csv", index=False)
    print(f"\n  Best hparams saved -> {cv_dir / 'xgboost_best_hparams.csv'}")
    return best_by_h


def _combo_preds_frames(horizon, params, dates, y, X, folds, max_val_days):
    """Re-run one combo across folds and return {val_year: DataFrame} of predictions."""
    out: dict[int, pd.DataFrame] = {}
    for f in folds:
        train_end = f.train_end - horizon
        X_tr, y_tr = X[f.train_start:train_end], y[f.train_start:train_end]
        val_end = (min(f.val_end, f.val_start + max_val_days)
                   if max_val_days is not None else f.val_end)
        X_va, y_va = X[f.val_start:val_end], y[f.val_start:val_end]
        model = _fit_es(X_tr, y_tr, params)
        y_pred = model.predict(X_va)
        fold_dates = dates[f.val_start:val_end]
        df = pd.DataFrame({"y_true": y_va, "y_pred": y_pred}, index=fold_dates)
        df.index.name = "date"
        out[f.val_year] = df
    return out


# ---------------------------------------------------------------------------
# Round 2 -- test walk-forward (daily refit, fixed hparams)
# ---------------------------------------------------------------------------

def _load_best_hparams(horizon: int) -> dict:
    """Read the best hparams saved by run_cv for this horizon, else defaults."""
    path = RESULTS_DIR / "cv" / "xgboost_best_hparams.csv"
    if not path.exists():
        print(f"  No CV hparams file at {path}; using DEFAULT_HPARAMS.")
        return DEFAULT_HPARAMS.copy()
    df = pd.read_csv(path)
    row = df.loc[df["horizon"] == horizon]
    if row.empty:
        print(f"  No CV row for h={horizon}d; using DEFAULT_HPARAMS.")
        return DEFAULT_HPARAMS.copy()
    row = row.iloc[0]
    return dict(max_depth=int(row["max_depth"]),
                learning_rate=float(row["learning_rate"]),
                subsample=float(row["subsample"]))


def run(horizon=1, hparams=None, refit="monthly", max_test_days=None,
        verbose=True, _data=None):
    """Test walk-forward for one horizon with fixed hparams.

    ``refit="monthly"`` (default): re-fit at each month start, reuse that
    model for every day of the month (matches LSTM's cadence).
    ``refit="step"``: refit at every origin (legacy behaviour, kept for the
    parity experiment against SARIMAX/Prophet)."""
    hparams = hparams or _load_best_hparams(horizon)
    params  = {**BASE_PARAMS, **hparams}

    dates, y, X = _data if _data is not None else load_data(horizon)
    n = len(y)

    test_start = int(dates.searchsorted(pd.Timestamp(TEST_START)))
    test_end   = n
    if TEST_END is not None:
        test_end = min(test_end, int(dates.searchsorted(pd.Timestamp(TEST_END), side="right")))
    if max_test_days is not None:
        test_end = min(test_end, test_start + max_test_days)

    refit_positions = (set(range(test_start, test_end)) if refit == "step"
                       else month_start_positions(dates, test_start, test_end))

    print(f"\n[XGBoost] {horizon}-day walk-forward (direct, {refit} refit)")
    print(f"  Hparams: {hparams}")
    print(f"  Train : {dates[0].date()} .. {dates[test_start-1].date()}  ({test_start} obs)")
    print(f"  Test  : {dates[test_start].date()} .. {dates[test_end-1].date()}  ({test_end-test_start} obs)")
    print(f"  Refits: {len(refit_positions)}")

    rows: list = []
    model: xgb.XGBRegressor | None = None
    with measure("xgboost", horizon, "test", refit=refit,
                 n_predictions_fn=lambda: len(rows)):
        for i in range(test_start, test_end):
            if i in refit_positions:
                train_end = i - horizon + 1
                if train_end <= 0:
                    continue
                X_tr, y_tr = X[:train_end], y[:train_end]
                model = _fit_es(X_tr, y_tr, params)
                if verbose:
                    print(f"    [refit {dates[i].date()}] train_end={train_end}")
            if model is None:
                continue
            y_pred = float(model.predict(X[i:i + 1])[0])
            rows.append((dates[i], float(y[i]), y_pred))

    preds = pd.DataFrame(rows, columns=["date", "y_true", "y_pred"]).set_index("date")
    # y is target_log_return_{h}d (cumulative h-day return), so y.mean() is
    # already h * mu_daily. summary() multiplies by horizon again, so divide
    # here to recover the daily drift and keep the predict-drift baseline honest.
    train_drift = float(y[:test_start].mean()) / horizon
    m = summary(preds["y_true"].to_numpy(), preds["y_pred"].to_numpy(),
                drift=train_drift, horizon=horizon)
    print(f"  Results: n={m['n']}  "
          f"RMSE={m['rmse']:.5f} (zero {m['rmse_zero']:.5f}, drift {m['rmse_drift']:.5f})  "
          f"MAE={m['mae']:.5f}  DA={m['da']:.3f} (up {m['da_up']:.3f}, edge {m['da_edge']:+.3f})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"xgboost_{horizon}d.csv"
    preds.to_csv(out)
    print(f"  Saved -> {out}")
    return preds, m


def _feature_names() -> list[str]:
    """Feature columns in the same order load_data() returns them (Date is the
    index and target_* columns are excluded)."""
    cols = pd.read_csv(PROCESSED_DIR / "xgboost_features_daily.csv", nrows=1).columns
    return [c for c in cols if not c.startswith("target_") and c != "Date"]


def save_final_model(horizon: int, hparams: dict | None = None):
    """Fit XGBoost on the full available history and save in native JSON format.

    Uses the same recipe as the walk-forward (best CV hparams, early stopping
    on a chronological tail). The saved model can be loaded with
    xgb.XGBRegressor() and model.load_model('xgboost_{h}d.json')."""
    hparams = hparams or _load_best_hparams(horizon)
    params  = {**BASE_PARAMS, **hparams, "importance_type": "gain"}
    dates, y, X = load_data(horizon)
    test_end = len(y)
    if TEST_END is not None:
        test_end = min(test_end,
                       int(dates.searchsorted(pd.Timestamp(TEST_END), side="right")))
    train_end = test_end - horizon          # purge h rows
    model = _fit_es(X[:train_end], y[:train_end], params)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / f"xgboost_{horizon}d.json"
    model.save_model(str(out))
    print(f"  Final model h={horizon}d -> {out}")


def save_feature_importance(horizons=None, hparams_by_h=None):
    """Persist XGBoost gain importance at the two protocol reference fits per
    horizon -- development (fit on 2018-2023, purged) and test_final (fit on the
    full 2018-2026 span, purged). This is the interpretable summary of the
    fitted trees, saved as numbers rather than only the figure, mirroring the
    SARIMAX coefficient report. Both fits reuse the same recipe as Figure 13
    (early-stopping on a chronological tail, best CV hparams, gain importance)."""
    horizons  = horizons or HORIZONS
    feat_names = _feature_names()
    rows = []
    for h in horizons:
        dates, y, X = load_data(h)
        hparams = (hparams_by_h or {}).get(h) or _load_best_hparams(h)
        params  = {**BASE_PARAMS, **hparams, "importance_type": "gain"}

        test_start = int(dates.searchsorted(pd.Timestamp(TEST_START)))
        test_end   = len(y)
        if TEST_END is not None:
            test_end = min(test_end,
                           int(dates.searchsorted(pd.Timestamp(TEST_END), side="right")))

        for phase, hi in (("development", test_start), ("test_final", test_end)):
            train_end = hi - h                   # purge h rows (no target leakage)
            if train_end <= 0:
                continue
            model = _fit_es(X[:train_end], y[:train_end], params)
            for feat, gain in zip(feat_names, model.feature_importances_):
                rows.append({"phase": phase, "horizon": h,
                             "feature": feat, "gain": float(gain)})

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = METRICS_DIR / "xgboost_importance.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  Feature importance -> {out}")


def run_all(horizons=None, hparams_by_h=None, refit="monthly",
            max_test_days=None, verbose=True):
    """Test walk-forward for every horizon. hparams_by_h may pin per-horizon
    combos; otherwise the CV file is consulted (falling back to defaults).
    ``verbose=False`` suppresses the per-refit progress lines."""
    horizons = horizons or HORIZONS
    print(f"\n[XGBoost] multi-horizon test walk-forward "
          f"(refit={refit}, horizons={horizons})")
    results = {}
    for h in horizons:
        hp = None if hparams_by_h is None else hparams_by_h.get(h)
        _, m = run(horizon=h, hparams=hp, refit=refit,
                   max_test_days=max_test_days, verbose=verbose)
        results[h] = m

    print(f"\n{'='*72}\n  XGBoost summary\n{'='*72}")
    for h, m in results.items():
        print(f"    h={h:>2}d   RMSE {m['rmse']:.5f} (zero {m['rmse_zero']:.5f}, drift {m['rmse_drift']:.5f})"
              f"   MAE {m['mae']:.5f}   DA {m['da']:.3f} (edge {m['da_edge']:+.3f})")

    if max_test_days is None:                    # skip on smoke runs (heavy extra fits)
        save_feature_importance(horizons, hparams_by_h)
        for h in horizons:
            hp = None if hparams_by_h is None else hparams_by_h.get(h)
            save_final_model(h, hp)
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="XGBoost direct multi-horizon walk-forward")
    ap.add_argument("--horizon", type=int, default=None,
                    help="single horizon in days (default: all in config.HORIZONS)")
    ap.add_argument("--max-test-days", type=int, default=None,
                    help="limit the test horizon for a quick smoke run")
    ap.add_argument("--cv", action="store_true",
                    help="run Round-1 grid search instead of the test walk-forward")
    ap.add_argument("--max-val-days", type=int, default=None,
                    help="limit each CV fold's val length for a quick smoke run")
    ap.add_argument("--refit", choices=["monthly", "step"], default="monthly",
                    help="test refit cadence: monthly (default) or step (every origin)")
    ap.add_argument("--importance-only", action="store_true",
                    help="regenerate the feature-importance CSV without the walk-forward")
    args = ap.parse_args()
    if args.importance_only:
        horizons = None if args.horizon is None else [args.horizon]
        save_feature_importance(horizons)
    elif args.cv:
        horizons = None if args.horizon is None else [args.horizon]
        run_cv(horizons=horizons, max_val_days=args.max_val_days)
    elif args.horizon is None:
        run_all(refit=args.refit, max_test_days=args.max_test_days)
    else:
        run(horizon=args.horizon, refit=args.refit, max_test_days=args.max_test_days)
