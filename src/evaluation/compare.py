"""
Four-model comparison engine: turns the saved walk-forward / CV predictions
into the reportable metric tables and Diebold-Mariano tests for the thesis.

Reads   : reports/predictions/{model}_{h}d.csv            (Round-2 test)
          reports/predictions/cv/{model}_fold{year}_{h}d.csv (Round-1 CV)
          data/processed/arima_features_daily.csv          (drift baseline)
Outputs : reports/metrics/comparison_test.csv / .md        (Test column)
          reports/metrics/comparison_validation.csv         (Validation column)
          reports/metrics/comparison_master.csv / .md        (Val vs Test side by side)
          reports/metrics/diebold_mariano.csv                (pairwise DM per horizon)

Why this module exists
----------------------
Each model file already prints its own metrics, but the four families are only
strictly comparable once their predictions are (a) re-keyed onto a common time
axis and (b) re-scored through a single code path with a single drift baseline.
This module is that single source of truth for the comparison section.

Date alignment (important)
--------------------------
The two families label a forecast by different dates:

  * Recursive (SARIMAX, Prophet): a row dated ``d`` holds the return REALISED on
    day ``d`` (forecast made from information up to ``d-1``).
  * Direct (XGBoost, LSTM): a row dated ``d`` holds ``target_log_return_h`` =
    ln(P_{d+h}/P_d), the return realised OVER (d, d+h], forecast from
    information up to and including ``d``.

Both solve the identical task -- "predict the next h-day return from everything
known at the current close" -- so each model's own aggregate metrics are valid
and comparable as-is. But a naive join on the label date would misalign the two
families by one day. We therefore re-key every series onto the **origin-close
date** (the date of the last observed close used to make the forecast):

  origin_date = label_date - 1 day   for recursive models
  origin_date = label_date           for direct models

After re-keying, the same origin_date + horizon carries an identical ``y_true``
across all four models, which is what the Diebold-Mariano pairing and the
overlay plots require.

Usage:
    python -m src.evaluation.compare            # build tables + write reports
    python -m src.evaluation.compare --quiet     # write files, minimal console
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DM_ALPHA, HORIZONS, TEST_START, TRAIN_START
from src.evaluation.cv import DEFAULT_VAL_YEARS, aggregate_metrics
from src.evaluation.metrics import diebold_mariano, summary
from src.evaluation import runtime

ROOT          = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
PRED_DIR      = ROOT / "reports" / "predictions"
CV_DIR        = PRED_DIR / "cv"
METRICS_DIR   = ROOT / "reports" / "metrics"

# Model registry -- display order, pretty names, and forecast family.
MODELS      = ["sarimax", "prophet", "xgboost", "lstm"]
DISPLAY     = {"sarimax": "SARIMAX", "prophet": "Prophet",
               "xgboost": "XGBoost", "lstm": "LSTM"}
RECURSIVE   = {"sarimax", "prophet"}          # label date = realised (target) date
DIRECT      = {"xgboost", "lstm"}             # label date = origin (close) date


# ---------------------------------------------------------------------------
# Drift baseline (train-only, no leakage) -- one market property shared by all
# ---------------------------------------------------------------------------

def _log_returns() -> pd.Series:
    """Daily BTC log-returns indexed by date (from the ARIMA feature set)."""
    df = pd.read_csv(PROCESSED_DIR / "arima_features_daily.csv",
                     index_col="Date", parse_dates=True)
    return df["log_return"]


def _drift_curve() -> pd.Series:
    """Expanding mean of daily BTC log-returns from TRAIN_START, indexed by date.

    The value at date ``t`` is the random-walk-with-drift rate estimated from
    every return up to and including ``t``. Read at a walk-forward origin it uses
    only data available at that origin's close, so it never leaks -- and, unlike
    a single train-window constant, it grows with the fit window exactly like the
    models it is benchmarked against (daily-refit SARIMAX/Prophet expand into the
    test and CV validation years; the drift baseline now expands with them)."""
    r = _log_returns().sort_index()
    r = r.loc[r.index >= pd.Timestamp(TRAIN_START)]
    return r.expanding().mean()


def drift_at(origin_dates) -> np.ndarray:
    """Per-origin predict-drift rate for the given origin-close dates.

    Aligns the expanding drift curve onto each origin (forward-filling so a
    non-trading origin inherits the last known rate). Multiply by the horizon to
    get the predict-drift forecast. This is the walk-forward ``predict-drift``
    baseline used for every reported table -- test and CV alike -- replacing the
    old frozen train-window constant."""
    curve = _drift_curve()
    idx = pd.DatetimeIndex(pd.to_datetime(list(origin_dates)))
    return curve.reindex(idx, method="ffill").to_numpy()


# ---------------------------------------------------------------------------
# Prediction loaders (re-keyed to origin-close date)
# ---------------------------------------------------------------------------

def _rekey(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Re-index a prediction frame onto the origin-close date (see module docstring)."""
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    if model in RECURSIVE:
        df.index = df.index - pd.Timedelta(days=1)
    df.index.name = "origin_date"
    return df


def load_test(model: str, horizon: int) -> pd.DataFrame | None:
    """Load one model's Round-2 test predictions, re-keyed to origin date.
    Returns None if the file does not exist yet (model/horizon not run)."""
    path = PRED_DIR / f"{model}_{horizon}d.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return _rekey(df, model)


def load_cv_folds(model: str, horizon: int) -> dict[int, pd.DataFrame]:
    """Load a model's per-fold CV predictions as {val_year: frame}, re-keyed.
    Missing folds are simply absent from the returned dict."""
    out: dict[int, pd.DataFrame] = {}
    for year in DEFAULT_VAL_YEARS:
        path = CV_DIR / f"{model}_fold{year}_{horizon}d.csv"
        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            out[year] = _rekey(df, model)
    return out


# ---------------------------------------------------------------------------
# Metric rows
# ---------------------------------------------------------------------------

def test_metrics(model: str, horizon: int) -> dict | None:
    """Full metric dict for one model/horizon on the test set, or None if absent.

    The predict-drift baseline is the per-origin walk-forward drift aligned to
    this model's own origin dates (see ``drift_at``)."""
    df = load_test(model, horizon)
    if df is None or df.empty:
        return None
    return summary(df["y_true"].to_numpy(), df["y_pred"].to_numpy(),
                   drift=drift_at(df.index), horizon=horizon)


def validation_metrics(model: str, horizon: int) -> dict | None:
    """CV metrics for one model/horizon: score each fold against the per-origin
    walk-forward drift, then aggregate mean +/- std across folds. None if no
    folds are present."""
    folds = load_cv_folds(model, horizon)
    if not folds:
        return None
    per_fold = {
        year: summary(df["y_true"].to_numpy(), df["y_pred"].to_numpy(),
                      drift=drift_at(df.index), horizon=horizon)
        for year, df in folds.items()
    }
    return aggregate_metrics(per_fold)


# ---------------------------------------------------------------------------
# Runtimes -- joined onto the tables
# ---------------------------------------------------------------------------

def _runtime_for(runtimes: pd.DataFrame, model: str, horizon: int,
                 kind: str) -> float | None:
    """Look up the training time for one (model, horizon, kind).

    Recursive models fit once per origin and forecast every horizon jointly,
    so their rows are stored with horizon = -1 and reported identically at
    every horizon. Direct models store one row per horizon."""
    if runtimes.empty:
        return None
    key = runtimes["model"].str.lower() == model.lower()
    hit = runtimes[key & (runtimes["kind"] == kind)
                   & (runtimes["horizon"].isin([horizon, -1]))]
    if hit.empty:
        return None
    # Prefer the horizon-specific row; fall back to the joint row.
    exact = hit[hit["horizon"] == horizon]
    hit = exact if not exact.empty else hit
    return float(hit.sort_values("timestamp").iloc[-1]["seconds"])


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

_TEST_COLS = ["model", "horizon", "n", "rmse", "rmse_zero", "rmse_drift",
              "mae", "da", "da_up", "da_edge", "train_seconds"]


def build_test_table(horizons=None) -> pd.DataFrame:
    """One row per available model/horizon on the test set."""
    horizons  = horizons or HORIZONS
    runtimes  = runtime.load()
    rows = []
    for model in MODELS:
        for h in horizons:
            m = test_metrics(model, h)
            if m is None:
                continue
            row = {"model": DISPLAY[model], "horizon": h,
                   **{k: m[k] for k in _TEST_COLS[2:-1] if k in m}}
            row["train_seconds"] = _runtime_for(runtimes, model, h, "test")
            rows.append(row)
    return pd.DataFrame(rows, columns=_TEST_COLS)


def build_validation_table(horizons=None) -> pd.DataFrame:
    """One row per available model/horizon on the CV (validation) set."""
    horizons = horizons or HORIZONS
    cols = ["model", "horizon", "n_total", "rmse", "rmse_std",
            "mae", "da", "da_edge", "da_edge_std", "train_seconds"]
    runtimes = runtime.load()
    rows = []
    for model in MODELS:
        for h in horizons:
            m = validation_metrics(model, h)
            if m is None:
                continue
            row = {"model": DISPLAY[model], "horizon": h,
                   **{k: m.get(k, np.nan) for k in cols[2:-1]}}
            row["train_seconds"] = _runtime_for(runtimes, model, h, "cv")
            rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def build_master_table(horizons=None) -> pd.DataFrame:
    """Validation vs Test side by side: the headline comparison table.

    A model generalises when its validation and test RMSE are close; a large
    test << val (or val << test) gap flags overfitting or regime shift."""
    val  = build_validation_table(horizons).rename(
        columns={"rmse": "rmse_val", "da": "da_val", "da_edge": "edge_val",
                 "train_seconds": "sec_val"})
    test = build_test_table(horizons).rename(
        columns={"rmse": "rmse_test", "rmse_zero": "zero_test",
                 "rmse_drift": "drift_test",
                 "da": "da_test", "da_up": "up_test", "da_edge": "edge_test",
                 "train_seconds": "sec_test"})
    keep_val  = ["model", "horizon", "rmse_val", "da_val", "edge_val", "sec_val"]
    keep_test = ["model", "horizon", "rmse_test", "zero_test", "drift_test",
                 "da_test", "up_test", "edge_test", "sec_test"]
    master = pd.merge(val[keep_val], test[keep_test],
                      on=["model", "horizon"], how="outer")
    # One yes/no column per baseline (symmetric across the three references).
    # None where the test result is missing, so it renders as "-" not "no".
    def _beats(a: str, b: str) -> pd.Series:
        m = master[a].notna() & master[b].notna()
        return np.where(m, master[a] < master[b], None)
    master["beats_zero"]  = _beats("rmse_test", "zero_test")
    master["beats_drift"] = _beats("rmse_test", "drift_test")
    # Directional: model beats "always up" when DA > DA_up (equivalently edge > 0).
    up_mask = master["da_test"].notna() & master["up_test"].notna()
    master["beats_up"] = np.where(
        up_mask, master["da_test"] > master["up_test"], None)
    order = {DISPLAY[m]: i for i, m in enumerate(MODELS)}
    master = master.sort_values(
        by=["horizon", "model"], key=lambda s: s.map(order) if s.name == "model" else s
    ).reset_index(drop=True)
    return master


# ---------------------------------------------------------------------------
# Diebold-Mariano: pairwise, correctly aligned on origin date
# ---------------------------------------------------------------------------

def dm_table(horizons=None, loss: str = "squared", tol: float = 1e-6) -> pd.DataFrame:
    """Pairwise Diebold-Mariano tests for every model pair, per horizon.

    The two series are inner-joined on the shared origin date so their y_true
    match by construction (asserted within ``tol``). A negative statistic
    favours the first model of the pair (lower loss); ``favoured`` names it."""
    horizons = horizons or HORIZONS
    rows = []
    for h in horizons:
        series = {m: load_test(m, h) for m in MODELS}
        series = {m: df for m, df in series.items() if df is not None and not df.empty}
        avail  = list(series)
        for i, a in enumerate(avail):
            for b in avail[i + 1:]:
                j = series[a].join(series[b], how="inner",
                                   lsuffix="_a", rsuffix="_b").dropna()
                if j.empty:
                    continue
                gap = float(np.max(np.abs(j["y_true_a"] - j["y_true_b"])))
                aligned = gap <= tol
                res = diebold_mariano(j["y_true_a"].to_numpy(),
                                      j["y_pred_a"].to_numpy(),
                                      j["y_pred_b"].to_numpy(), loss=loss, h=h)
                favoured = (DISPLAY[a] if res["dm"] < 0 else DISPLAY[b]) \
                    if np.isfinite(res["dm"]) else "-"
                rows.append({
                    "horizon": h, "model_a": DISPLAY[a], "model_b": DISPLAY[b],
                    "dm": res["dm"], "p_value": res["p_value"], "n": res["n"],
                    "favoured": favoured,
                    "significant": bool(res["p_value"] < DM_ALPHA),
                    "y_true_aligned": aligned,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Output: markdown + CSV
# ---------------------------------------------------------------------------

def to_markdown(df: pd.DataFrame, floatfmt: dict | None = None) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table (no deps)."""
    floatfmt = floatfmt or {}
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep    = "|" + "|".join("---" for _ in cols) + "|"
    lines  = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
                cells.append("-")
            elif isinstance(v, (bool, np.bool_)):
                cells.append("yes" if v else "no")
            elif isinstance(v, float):
                cells.append(format(v, floatfmt.get(c, ".5f")))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_reports(horizons=None, verbose=True) -> dict[str, pd.DataFrame]:
    """Build every table, write CSV + markdown to reports/metrics/, and return them."""
    horizons = horizons or HORIZONS
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    test   = build_test_table(horizons)
    val    = build_validation_table(horizons)
    master = build_master_table(horizons)
    dm     = dm_table(horizons)

    test.to_csv(METRICS_DIR / "comparison_test.csv", index=False)
    val.to_csv(METRICS_DIR / "comparison_validation.csv", index=False)
    master.to_csv(METRICS_DIR / "comparison_master.csv", index=False)
    dm.to_csv(METRICS_DIR / "diebold_mariano.csv", index=False)

    master_fmt = {"rmse_val": ".5f", "da_val": ".3f", "edge_val": "+.3f",
                  "rmse_test": ".5f", "zero_test": ".5f", "drift_test": ".5f",
                  "da_test": ".3f", "up_test": ".3f", "edge_test": "+.3f"}
    dm_fmt = {"dm": "+.3f", "p_value": ".4f"}
    master_md = master.copy()
    master_md["time_val"]  = master_md["sec_val"].map(runtime.fmt)
    master_md["time_test"] = master_md["sec_test"].map(runtime.fmt)
    master_md = master_md[["model", "horizon",
                           "rmse_val", "da_val", "edge_val", "time_val",
                           "rmse_test", "zero_test", "drift_test",
                           "da_test", "up_test", "edge_test", "time_test",
                           "beats_zero", "beats_drift", "beats_up"]]
    (METRICS_DIR / "comparison_master.md").write_text(
        "# Model comparison — Validation (CV) vs Test\n\n"
        + to_markdown(master_md, master_fmt) + "\n", encoding="utf-8")
    dm_md = dm.copy()
    if not dm_md.empty:
        dm_md = (dm_md.drop(columns=["y_true_aligned"])
                       .rename(columns={"significant": "signif"}))
    (METRICS_DIR / "diebold_mariano.md").write_text(
        f"# Diebold-Mariano pairwise tests (test set, squared loss; "
        f"signif = p<{DM_ALPHA})\n\n"
        + to_markdown(dm_md, dm_fmt) + "\n", encoding="utf-8")

    if verbose:
        print_summary(master, dm)
        print(f"\n  Wrote 4 CSV + 2 markdown tables -> {METRICS_DIR}")
    return {"test": test, "validation": val, "master": master, "dm": dm}


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(master: pd.DataFrame, dm: pd.DataFrame) -> None:
    """Clean, aligned console report of the comparison."""
    print(f"\n{'='*78}\n  MODEL COMPARISON  --  Validation (CV) vs Test\n{'='*78}")
    if master.empty:
        print("  (no predictions found -- run the models first)")
        return

    # Two-line header so the "beats?" section is clearly separated from the
    # numeric RMSE baselines (which happen to share the labels zero / drift).
    subhdr = (f"  {'':<9}{'':>4}   "
              f"{'':>9}{'':>10}{'':>9}{'':>9}   "
              f"{'':>8}{'':>7}   {'':>7}   "
              f"{'  beats?':>15}")
    hdr = (f"  {'model':<9}{'h':>4}   "
           f"{'RMSE_val':>9}{'RMSE_test':>10}{'zero':>9}{'drift':>9}   "
           f"{'DA_test':>8}{'edge':>7}   {'time':>7}   "
           f"{'zero':>5}{'drift':>6}{'up':>4}")
    def _yn(v):
        return "-" if v is None else ("yes" if v else "no")
    for h in sorted(master["horizon"].unique()):
        print(f"\n  --- horizon = {h}d ---")
        print(subhdr)
        print(hdr)
        sub = master[master["horizon"] == h]
        for _, r in sub.iterrows():
            rv = "-" if pd.isna(r["rmse_val"])  else f"{r['rmse_val']:.5f}"
            rt = "-" if pd.isna(r["rmse_test"]) else f"{r['rmse_test']:.5f}"
            zr = "-" if pd.isna(r["zero_test"]) else f"{r['zero_test']:.5f}"
            dr = "-" if pd.isna(r["drift_test"]) else f"{r['drift_test']:.5f}"
            da = "-" if pd.isna(r["da_test"])   else f"{r['da_test']:.3f}"
            ed = "-" if pd.isna(r["edge_test"]) else f"{r['edge_test']:+.3f}"
            tm = runtime.fmt(r.get("sec_test"))
            print(f"  {r['model']:<9}{r['horizon']:>4d}   "
                  f"{rv:>9}{rt:>10}{zr:>9}{dr:>9}   {da:>8}{ed:>7}   {tm:>7}   "
                  f"{_yn(r['beats_zero']):>5}{_yn(r['beats_drift']):>6}"
                  f"{_yn(r['beats_up']):>4}")

    if not dm.empty:
        print(f"\n{'='*78}\n  DIEBOLD-MARIANO  "
              f"(test, squared loss; signif = p<{DM_ALPHA})\n{'='*78}")
        for h in sorted(dm["horizon"].unique()):
            print(f"\n  --- horizon = {h}d ---")
            for _, r in dm[dm["horizon"] == h].iterrows():
                signif = "yes" if r["significant"] else "no"
                warn = "" if r["y_true_aligned"] else "  [!] y_true misaligned"
                print(f"  {r['model_a']:>8} vs {r['model_b']:<8}  "
                      f"DM={r['dm']:+.3f}  p={r['p_value']:.4f}   "
                      f"signif={signif:<3}  favours {r['favoured']}{warn}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Four-model comparison tables + DM tests")
    ap.add_argument("--quiet", action="store_true", help="write files, minimal console output")
    args = ap.parse_args()
    write_reports(verbose=not args.quiet)
