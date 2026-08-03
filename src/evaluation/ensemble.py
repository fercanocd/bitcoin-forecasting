"""
Forecast-combination (ensemble) engine for the four-model comparison.

Portfolio theory, applied to *forecasts* instead of assets. Each model is an
"asset"; its daily prediction error is the analogue of a return series. We ask
the classic question -- can combining them reduce the error variance below any
single model? -- and answer it with three weighting schemes of increasing
sophistication:

  (a) equal        w_i = 1/N                          (no estimation)
  (b) inv_rmse     w_i = (1/RMSE_i) / sum_j(1/RMSE_j)  (weight by accuracy)
  (c) min_var      argmin_w  w' Sigma w   s.t. sum(w)=1, w >= 0
                   (Bates-Granger minimum-variance combination; long-only,
                    the direct analogue of a minimum-variance portfolio)

Leakage discipline (the whole point of doing this properly)
-----------------------------------------------------------
Every weight is estimated on the **validation** errors only (the Round-1 CV
folds, 2020-2023). The **test** set (2024 onward) is never touched to choose a
weight -- it is used exclusively to score the frozen combination out of sample.
This mirrors calibrating a portfolio on history and measuring it the year after.

Why it usually does not help (the diagnostic to look at first)
--------------------------------------------------------------
Diversification only removes *idiosyncratic* error. If the models all collapse
toward the same near-drift forecast, their errors are dominated by the common,
irreducible market move (y_true) and are almost perfectly correlated -- nothing
to average away. The validation error-correlation matrix (reported here) is the
tell: correlations near 1 mean the ensemble can only tie the best single model.
The variance of an equal-weight combination is

    Var = (sigma^2 / N) * [1 + (N-1) * rho_bar],

so at rho_bar ~ 0.97 the RMSE reduction is ~1%; at rho_bar ~ 0 it would halve.

Reads   : reports/predictions/{model}_{h}d.csv              (test)
          reports/predictions/cv/{model}_fold{year}_{h}d.csv (validation)
Outputs : reports/metrics/ensemble_weights.csv
          reports/metrics/ensemble_test.csv / .md
          reports/metrics/ensemble_error_corr.csv / .md

Usage:
    python -m src.evaluation.ensemble           # build tables, print summary
    python -m src.evaluation.ensemble --quiet    # write files, minimal console
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import HORIZONS
from src.evaluation import compare
from src.evaluation.metrics import summary

METRICS_DIR = compare.METRICS_DIR

# The three weighting schemes, in the order they appear in every report.
METHODS = ["equal", "inv_rmse", "min_var"]
METHOD_LABEL = {"equal": "equal (1/N)", "inv_rmse": "inverse-RMSE",
                "min_var": "min-variance"}

# The predict-drift baseline can also *compete inside* the pool as a fifth
# "asset" (see write_reports augmented=True). predict-zero is deliberately not
# added: its error differs from drift's by a constant, and covariance is
# shift-invariant, so corr(zero, drift) = 1 exactly -- for the minimum-variance
# solve they are the same asset (and including both makes Sigma singular).
DRIFT = "drift"
DISPLAY = {**compare.DISPLAY, DRIFT: "Drift"}
_POOL_ORDER = compare.MODELS + [DRIFT]


def _pool(augment: bool) -> list[str]:
    """Model keys in the combination pool: the four models, plus the drift
    baseline when ``augment`` is set."""
    return compare.MODELS + ([DRIFT] if augment else [])


# ---------------------------------------------------------------------------
# Panels: y_true + one y_pred column per model, aligned on origin date
# ---------------------------------------------------------------------------

def _align(series: dict[str, pd.DataFrame], tol: float = 1e-6
           ) -> tuple[pd.DataFrame, list[str]]:
    """Inner-join per-model {y_true, y_pred} frames onto their common origin
    dates and return (panel, models). ``panel`` has one ``y_true`` column plus
    one column of predictions per model (named by the model key). y_true is
    taken from the first model and checked to match the others within ``tol``
    (they share an origin date and horizon, so they must)."""
    series = {m: df for m, df in series.items() if df is not None and not df.empty}
    models = [m for m in _POOL_ORDER if m in series]
    if len(models) < 2:
        return pd.DataFrame(), models

    common = None
    for m in models:
        idx = series[m].index
        common = idx if common is None else common.intersection(idx)
    common = common.sort_values()
    if len(common) == 0:
        return pd.DataFrame(), models

    panel = pd.DataFrame(index=common)
    panel["y_true"] = series[models[0]].loc[common, "y_true"]
    for m in models:
        yt = series[m].loc[common, "y_true"]
        if float(np.max(np.abs(yt.to_numpy() - panel["y_true"].to_numpy()))) > tol:
            print(f"  [!] y_true mismatch for {m} -- alignment check failed")
        panel[m] = series[m].loc[common, "y_pred"]
    return panel, models


def test_panel(horizon: int, augment: bool = False
               ) -> tuple[pd.DataFrame, list[str]]:
    """Aligned test panel for one horizon (origins where every model predicts).
    With ``augment`` the predict-drift baseline joins the pool as the per-origin
    walk-forward forecast h * drift_at(origin) sharing the models' index and
    y_true."""
    series = {m: compare.load_test(m, horizon) for m in compare.MODELS}
    if augment:
        template = next((series[m] for m in compare.MODELS
                         if series[m] is not None and not series[m].empty), None)
        if template is not None:
            d = template.copy()
            d["y_pred"] = horizon * compare.drift_at(d.index)
            series[DRIFT] = d
    return _align(series)


def val_panel(horizon: int, augment: bool = False
              ) -> tuple[pd.DataFrame, list[str]]:
    """Aligned validation panel for one horizon: each model's CV folds are
    concatenated across years (2020-2023) before aligning, so the weights see
    the full four-fold validation history. With ``augment`` the predict-drift
    baseline joins the pool as the per-origin walk-forward drift (drift_at),
    matching how the recursive models expand into each validation year."""
    series: dict[str, pd.DataFrame] = {}
    template_folds: dict[int, pd.DataFrame] | None = None
    for m in compare.MODELS:
        folds = compare.load_cv_folds(m, horizon)
        if folds:
            series[m] = pd.concat(folds.values()).sort_index()
            if template_folds is None:
                template_folds = folds
    if augment and template_folds is not None:
        parts = []
        for year, df in template_folds.items():
            d = df.copy()
            d["y_pred"] = horizon * compare.drift_at(d.index)
            parts.append(d)
        series[DRIFT] = pd.concat(parts).sort_index()
    return _align(series)


def _errors(panel: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Per-model error matrix e_i = y_true - y_pred_i (columns = models)."""
    return panel[models].rsub(panel["y_true"], axis=0)


# ---------------------------------------------------------------------------
# Weighting schemes (all fitted on the validation error matrix)
# ---------------------------------------------------------------------------

def _w_equal(models: list[str]) -> dict[str, float]:
    return {m: 1.0 / len(models) for m in models}


def _w_inv_rmse(err_val: pd.DataFrame) -> dict[str, float]:
    rmse = np.sqrt((err_val ** 2).mean())          # Series indexed by model
    inv  = 1.0 / rmse
    w    = inv / inv.sum()
    return w.to_dict()


def _w_min_var(err_val: pd.DataFrame) -> dict[str, float]:
    """Long-only minimum-variance weights: argmin w'Sigma w s.t. sum(w)=1,
    w>=0. Sigma is the covariance of the validation errors (variance
    minimisation in the Bates-Granger sense; with forecasts sitting on the
    near-zero drift, bias is negligible so this ~ minimising MSE). A tiny ridge
    keeps the solve well-posed when errors are near-collinear; on failure we
    fall back to equal weights."""
    from scipy.optimize import minimize

    models = list(err_val.columns)
    n = len(models)
    Sigma = err_val.cov().to_numpy()
    Sigma = Sigma + 1e-12 * np.eye(n)              # numerical floor
    res = minimize(lambda w: float(w @ Sigma @ w), x0=np.full(n, 1.0 / n),
                   method="SLSQP", bounds=[(0.0, 1.0)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}])
    w = res.x if res.success else np.full(n, 1.0 / n)
    w = np.clip(w, 0.0, None)
    w = w / w.sum()
    return dict(zip(models, w))


def weights_for(err_val: pd.DataFrame, models: list[str]) -> dict[str, dict[str, float]]:
    """All three weight vectors for one horizon, from its validation errors."""
    return {
        "equal":    _w_equal(models),
        "inv_rmse": _w_inv_rmse(err_val),
        "min_var":  _w_min_var(err_val),
    }


def _combine(panel: pd.DataFrame, w: dict[str, float]) -> np.ndarray:
    """Weighted ensemble forecast on a panel (weights keyed by model name)."""
    models = list(w)
    W = np.array([w[m] for m in models])
    return panel[models].to_numpy() @ W


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def build_weights_table(horizons=None, augment: bool = False) -> pd.DataFrame:
    """One row per (horizon, method) with the weight placed on each pool member.
    ``augment`` adds the drift baseline as a fifth column."""
    horizons = horizons or HORIZONS
    pool = _pool(augment)
    rows = []
    for h in horizons:
        vpanel, vmodels = val_panel(h, augment=augment)
        if vpanel.empty:
            continue
        err_val = _errors(vpanel, vmodels)
        for method, w in weights_for(err_val, vmodels).items():
            row = {"horizon": h, "method": method}
            row.update({DISPLAY[m]: w.get(m, np.nan) for m in pool})
            rows.append(row)
    cols = ["horizon", "method"] + [DISPLAY[m] for m in pool]
    return pd.DataFrame(rows, columns=cols)


def build_test_table(horizons=None, augment: bool = False) -> pd.DataFrame:
    """Out-of-sample scorecard: for every horizon, the three ensembles plus the
    best single member, with validation RMSE (in-sample for the weights), test
    RMSE, the honest baselines, directional accuracy and beats-flags. ``augment``
    lets the drift baseline compete inside the pool.

    The weights come from validation; the test columns never informed them."""
    horizons = horizons or HORIZONS
    rows = []
    for h in horizons:
        vpanel, vmodels = val_panel(h, augment=augment)
        tpanel, tmodels = test_panel(h, augment=augment)
        if vpanel.empty or tpanel.empty:
            continue
        models = [m for m in vmodels if m in tmodels]
        if len(models) < 2:
            continue
        err_val = _errors(vpanel[["y_true"] + models], models)

        yt_test = tpanel["y_true"].to_numpy()
        yt_val  = vpanel["y_true"].to_numpy()
        drift   = compare.drift_at(tpanel.index)   # per-origin walk-forward drift
        base = summary(yt_test, np.zeros_like(yt_test), drift=drift, horizon=h)
        zero_test, drift_test = base["rmse_zero"], base["rmse_drift"]

        def _row(name: str, w: dict[str, float]) -> dict:
            yp_test = _combine(tpanel, w)
            yp_val  = _combine(vpanel, w)
            m = summary(yt_test, yp_test, drift=drift, horizon=h)
            return {
                "horizon": h, "method": name,
                "rmse_val":  float(np.sqrt(np.mean((yt_val - yp_val) ** 2))),
                "rmse_test": m["rmse"], "rmse_zero": zero_test,
                "rmse_drift": drift_test,
                "da_test": m["da"], "edge_test": m["da_edge"],
                "beats_zero":  bool(m["rmse"] < zero_test),
                "beats_drift": bool(m["rmse"] < drift_test),
                "beats_up":    bool(m["da"] > m["da_up"]),
                "n_test": m["n"],
            }

        weights = weights_for(err_val, models)
        for method in METHODS:
            rows.append(_row(method, weights[method]))

        # Best single model on the *validation* RMSE (chosen without the test),
        # scored on test -- the honest "just pick the winner" reference.
        val_rmse = np.sqrt((err_val ** 2).mean())
        best = val_rmse.idxmin()
        rows.append(_row(f"best_single [{DISPLAY[best]}]",
                         {m: (1.0 if m == best else 0.0) for m in models}))

    cols = ["horizon", "method", "rmse_val", "rmse_test", "rmse_zero",
            "rmse_drift", "da_test", "edge_test",
            "beats_zero", "beats_drift", "beats_up", "n_test"]
    return pd.DataFrame(rows, columns=cols)


def _structure_table(kind: str, horizons=None, which: str = "val",
                     augment: bool = False) -> pd.DataFrame:
    """Long-format error covariance (``kind='cov'``) or correlation
    (``kind='corr'``) matrix per horizon -- the diversification diagnostic.

    Covariance is exactly what the min-variance optimiser minimises (w'Sigma w);
    correlation is its scale-free reading (near 1 => nothing to diversify).
    ``which='val'`` uses the validation errors that the weights were fit on;
    ``which='test'`` uses the held-out test errors. Rows: horizon, model_a,
    model_b, <kind>."""
    horizons = horizons or HORIZONS
    rows = []
    for h in horizons:
        panel, models = (val_panel(h, augment=augment) if which == "val"
                         else test_panel(h, augment=augment))
        if panel.empty:
            continue
        err = _errors(panel, models)
        mat = err.cov() if kind == "cov" else err.corr()
        for a in models:
            for b in models:
                rows.append({"horizon": h, "model_a": DISPLAY[a],
                             "model_b": DISPLAY[b],
                             kind: float(mat.loc[a, b])})
    return pd.DataFrame(rows, columns=["horizon", "model_a", "model_b", kind])


def build_corr_table(horizons=None, which: str = "val",
                     augment: bool = False) -> pd.DataFrame:
    """Long-format error-correlation matrix per horizon (see _structure_table)."""
    return _structure_table("corr", horizons, which, augment)


def build_cov_table(horizons=None, which: str = "val",
                    augment: bool = False) -> pd.DataFrame:
    """Long-format error variance-covariance matrix per horizon (what min-var
    minimises; diagonal = per-model error variance = RMSE^2)."""
    return _structure_table("cov", horizons, which, augment)


# ---------------------------------------------------------------------------
# Output: CSV + markdown
# ---------------------------------------------------------------------------

def _matrix_md(long: pd.DataFrame, horizon: int, value_col: str,
               fmt: str) -> str:
    """Render one horizon's long-format matrix as a square markdown table."""
    sub = long[long["horizon"] == horizon]
    models = [m for m in (DISPLAY[x] for x in _POOL_ORDER)
              if m in set(sub["model_a"])]
    wide = (sub.pivot(index="model_a", columns="model_b", values=value_col)
               .reindex(index=models, columns=models))
    header = "| | " + " | ".join(models) + " |"
    sep = "|" + "|".join("---" for _ in range(len(models) + 1)) + "|"
    lines = [header, sep]
    for a in models:
        cells = [format(wide.loc[a, b], fmt) for b in models]
        lines.append(f"| **{a}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_reports(horizons=None, verbose=True) -> dict[str, pd.DataFrame]:
    """Build the weights / test / correlation tables, write CSV + markdown,
    and return them. Reads only saved predictions, so it is safe after a
    partial run (horizons with fewer than two models are skipped)."""
    horizons = horizons or HORIZONS
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    weights = build_weights_table(horizons)
    test    = build_test_table(horizons)
    corr    = build_corr_table(horizons, which="val")
    cov     = build_cov_table(horizons, which="val")

    weights.to_csv(METRICS_DIR / "ensemble_weights.csv", index=False)
    test.to_csv(METRICS_DIR / "ensemble_test.csv", index=False)
    corr.to_csv(METRICS_DIR / "ensemble_error_corr.csv", index=False)
    cov.to_csv(METRICS_DIR / "ensemble_error_cov.csv", index=False)

    tfmt = {"rmse_val": ".5f", "rmse_test": ".5f", "rmse_zero": ".5f",
            "rmse_drift": ".5f", "da_test": ".3f", "edge_test": "+.3f"}
    (METRICS_DIR / "ensemble_test.md").write_text(
        "# Ensemble scorecard — weights fit on validation, scored on test\n\n"
        + compare.to_markdown(test, tfmt) + "\n", encoding="utf-8")

    # One markdown with, per horizon, the covariance Sigma (what min-var
    # minimises) followed by its scale-free correlation reading.
    blocks = ["# Validation error structure (why the ensemble can or cannot "
              "diversify)\n\n"
              "For each horizon: the variance-covariance matrix `Sigma` of the "
              "model errors (the quantity the minimum-variance combination "
              "minimises; diagonal = per-model error variance = RMSE^2), and its "
              "correlation reading (near 1 everywhere => the models miss on the "
              "same days, so combining them cannot cut the variance).\n"]
    for h in horizons:
        if (cov["horizon"] == h).any():
            blocks.append(f"\n## horizon = {h}d\n\n"
                          f"**Covariance (Sigma):**\n\n"
                          + _matrix_md(cov, h, "cov", ".2e") + "\n\n"
                          f"**Correlation:**\n\n"
                          + _matrix_md(corr, h, "corr", "+.3f") + "\n")
    (METRICS_DIR / "ensemble_error_structure.md").write_text(
        "\n".join(blocks), encoding="utf-8")

    # Augmented pool: let the predict-drift baseline compete inside the pool.
    # The headline question is where the min-variance solve puts its weight --
    # does it keep the four models, or defer to the naive constant?
    aweights = build_weights_table(horizons, augment=True)
    atest    = build_test_table(horizons, augment=True)
    aweights.to_csv(METRICS_DIR / "ensemble_augmented_weights.csv", index=False)
    atest.to_csv(METRICS_DIR / "ensemble_augmented_test.csv", index=False)
    wfmt = {DISPLAY[m]: ".3f" for m in _POOL_ORDER}
    (METRICS_DIR / "ensemble_augmented.md").write_text(
        "# Augmented pool — four models + predict-drift competing\n\n"
        "predict-zero is omitted: its error differs from drift's by a constant, "
        "so their covariances are identical (corr = 1) and the minimum-variance "
        "solve cannot tell them apart.\n\n"
        "## Weights\n\n" + compare.to_markdown(aweights, wfmt) + "\n\n"
        "## Scorecard\n\n" + compare.to_markdown(atest, tfmt) + "\n",
        encoding="utf-8")

    if verbose:
        print_summary(weights, test, corr, cov)
        print(f"\n{'='*78}\n  AUGMENTED POOL  --  four models + predict-drift "
              f"competing inside\n{'='*78}")
        print_summary(aweights, atest, None, None, header=False)
        print(f"\n  Wrote 6 CSV + 3 markdown tables -> {METRICS_DIR}")
    return {"weights": weights, "test": test, "corr": corr, "cov": cov,
            "aug_weights": aweights, "aug_test": atest}


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(weights: pd.DataFrame, test: pd.DataFrame,
                  corr: pd.DataFrame | None = None,
                  cov: pd.DataFrame | None = None, header: bool = True) -> None:
    """Aligned console report: error structure, weights, test scorecard.

    Pool-agnostic: the model columns are read off the weights table, so the
    same routine renders both the four-model pool and the augmented pool."""
    if header:
        print(f"\n{'='*78}\n  ENSEMBLE  --  forecast combination (weights from "
              f"validation, scored on test)\n{'='*78}")
    if test.empty:
        print("  (need >=2 models per horizon -- run more models first)")
        return

    def _yn(v):
        return "-" if v is None or v is pd.NA else ("yes" if v else "no")

    mcols = [c for c in weights.columns if c not in ("horizon", "method")]

    for h in sorted(test["horizon"].unique()):
        print(f"\n  --- horizon = {h}d ---")

        # Diagnostic first: average off-diagonal error correlation.
        if corr is not None:
            csub = corr[corr["horizon"] == h]
            off = csub[csub["model_a"] != csub["model_b"]]["corr"]
            if not off.empty:
                print(f"  mean error-correlation (validation) = {off.mean():+.3f}"
                      f"   [near 1 => little to diversify]")

        # Variance-covariance matrix Sigma (what min-var minimises).
        if cov is not None and (cov["horizon"] == h).any():
            vsub = cov[cov["horizon"] == h]
            models = [m for m in mcols if m in set(vsub["model_a"])]
            wide = (vsub.pivot(index="model_a", columns="model_b", values="cov")
                        .reindex(index=models, columns=models))
            print(f"  cov(errors) Sigma  " + "".join(f"{m:>11}" for m in models))
            for a in models:
                cells = "".join(f"{wide.loc[a, b]:>11.2e}" for b in models)
                print(f"    {a:<15}" + cells)

        # Weights per scheme.
        wsub = weights[weights["horizon"] == h]
        print(f"  {'weights':<16}" + "".join(f"{m:>9}" for m in mcols))
        for _, r in wsub.iterrows():
            cells = "".join(
                ("-".rjust(9) if pd.isna(r[m]) else f"{r[m]:>9.3f}") for m in mcols)
            print(f"    {METHOD_LABEL[r['method']]:<14}" + cells)

        # Test scorecard, with the beats-baseline flags spelled out in full.
        hdr = (f"  {'method':<22}{'RMSE_val':>9}{'RMSE_test':>10}{'zero':>9}"
               f"{'drift':>9}{'DA':>7}{'edge':>7}   "
               f"{'beats_zero':>11}{'beats_drift':>12}{'beats_up':>10}")
        print(hdr)
        for _, r in test[test["horizon"] == h].iterrows():
            print(f"  {r['method']:<22}"
                  f"{r['rmse_val']:>9.5f}{r['rmse_test']:>10.5f}"
                  f"{r['rmse_zero']:>9.5f}{r['rmse_drift']:>9.5f}"
                  f"{r['da_test']:>7.3f}{r['edge_test']:>+7.3f}   "
                  f"{_yn(r['beats_zero']):>11}{_yn(r['beats_drift']):>12}"
                  f"{_yn(r['beats_up']):>10}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Forecast-combination (ensemble) tables")
    ap.add_argument("--quiet", action="store_true",
                    help="write files, minimal console output")
    args = ap.parse_args()
    write_reports(verbose=not args.quiet)
