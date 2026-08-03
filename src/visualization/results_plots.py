"""
Model-results figures for the Bitcoin Forecasting TFM (thesis Results section).

Consumes the saved walk-forward / CV predictions via src.evaluation.compare and
renders the comparison figures. Every figure degrades gracefully: a model or
horizon whose predictions have not been generated yet is skipped with a note,
so the module can be smoke-tested on partial data before the full run.

Figures produced (saved to reports/figures/):
  07_rmse_vs_baseline.png      -- test RMSE per model/horizon vs predict-drift
  08_directional_accuracy.png  -- test DA per model/horizon vs always-up baseline
  09_predictions_timeseries.png-- forecast vs realised return over test (h=1, grid)
  10_cumulative_strategy.png   -- sign-of-forecast strategy equity vs buy-and-hold (h=1)
  11_pred_vs_actual_scatter.png-- predicted vs realised return, showing shrinkage (h=1)
  12_validation_vs_test.png    -- CV vs test RMSE per model/horizon (generalisation)
  13_xgboost_importance.png    -- XGBoost gain importance, top features per horizon
  14_diebold_mariano.png       -- pairwise DM significance heatmap per horizon
  15_ensemble_rmse.png         -- test RMSE by weighting scheme vs best-single / drift
  16_error_correlation.png     -- validation error-correlation heatmap per horizon

Usage:
  python -m src.visualization.results_plots            # all result figures
  python -m src.visualization.results_plots --only 07  # a single figure by number
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.config import HORIZONS, TEST_START
from src.evaluation import compare
# Shared figure style so EDA and results figures form one visual system.
from src.visualization.style import STYLE, FIG_DIR, _save

# ---------------------------------------------------------------------------
# Shared look-and-feel
# ---------------------------------------------------------------------------
# One fixed colour per model, used across every results figure.
MODEL_COLORS = {
    "SARIMAX": "#1f77b4",   # blue
    "Prophet": "#ff7f0e",   # orange
    "XGBoost": "#2ca02c",   # green
    "LSTM":    "#d62728",   # red
}
DRIFT_COLOR = "#7f7f7f"     # grey — the predict-drift baseline
ZERO_COLOR  = "#bcbd22"     # olive — the predict-zero baseline
MODEL_ORDER = [compare.DISPLAY[m] for m in compare.MODELS]


def _models_in(df: pd.DataFrame) -> list[str]:
    """Models present in a table, in canonical display order."""
    return [m for m in MODEL_ORDER if m in set(df["model"])]


# ---------------------------------------------------------------------------
# Figure 07 — Test RMSE vs the honest baselines
# ---------------------------------------------------------------------------

def plot_rmse_vs_baseline():
    print("\n[07] RMSE vs predict-drift baseline...")
    test = compare.build_test_table()
    if test.empty:
        print("  No test predictions found — skipping.")
        return

    horizons = sorted(test["horizon"].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(5 * len(horizons), 4.5),
                             squeeze=False)
    axes = axes[0]

    for ax, h in zip(axes, horizons):
        sub = test[test["horizon"] == h].set_index("model")
        models = _models_in(sub.reset_index())
        vals   = [sub.loc[m, "rmse"] for m in models]
        colors = [MODEL_COLORS[m] for m in models]
        x = np.arange(len(models))
        bars = ax.bar(x, vals, color=colors, width=0.6, zorder=3)

        # Honest reference lines (same for every model at this horizon).
        drift = float(sub["rmse_drift"].dropna().iloc[0]) if sub["rmse_drift"].notna().any() else None
        zero  = float(sub["rmse_zero"].dropna().iloc[0])  if sub["rmse_zero"].notna().any()  else None
        if drift is not None:
            ax.axhline(drift, color=DRIFT_COLOR, ls="--", lw=1.5, zorder=2,
                       label=f"predict-drift ({drift:.4f})")
        if zero is not None:
            ax.axhline(zero, color=ZERO_COLOR, ls=":", lw=1.5, zorder=2,
                       label=f"predict-zero ({zero:.4f})")

        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=8)

        ax.set_title(f"h = {h} day{'s' if h > 1 else ''}")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_ylabel("RMSE (log-return)")
        ax.set_ylim(0, max(vals + [drift or 0, zero or 0]) * 1.18)
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Test-set RMSE vs honest baselines — lower is better; "
                 "a model adds value only below predict-drift", y=1.02)
    fig.tight_layout()
    _save(fig, "07_rmse_vs_baseline.png")


# ---------------------------------------------------------------------------
# Figure 08 — Directional accuracy vs the always-up baseline
# ---------------------------------------------------------------------------

def plot_directional_accuracy():
    print("\n[08] Directional accuracy vs always-up...")
    test = compare.build_test_table()
    if test.empty:
        print("  No test predictions found — skipping.")
        return

    horizons = sorted(test["horizon"].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(5 * len(horizons), 4.5),
                             squeeze=False)
    axes = axes[0]

    for ax, h in zip(axes, horizons):
        sub = test[test["horizon"] == h].set_index("model")
        models = _models_in(sub.reset_index())
        da     = [sub.loc[m, "da"] for m in models]
        edge   = [sub.loc[m, "da_edge"] for m in models]
        colors = [MODEL_COLORS[m] for m in models]
        x = np.arange(len(models))
        bars = ax.bar(x, da, color=colors, width=0.6, zorder=3)

        da_up = float(sub["da_up"].dropna().iloc[0]) if sub["da_up"].notna().any() else None
        if da_up is not None:
            ax.axhline(da_up, color=DRIFT_COLOR, ls="--", lw=1.5, zorder=2,
                       label=f"always-up ({da_up:.3f})")
        ax.axhline(0.5, color="black", ls=":", lw=1, zorder=1, label="coin flip (0.5)")

        for b, v, e in zip(bars, da, edge):
            ax.text(b.get_x() + b.get_width() / 2, v,
                    f"{v:.3f}\n(edge {e:+.3f})", ha="center", va="bottom", fontsize=8)

        ax.set_title(f"h = {h} day{'s' if h > 1 else ''}")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_ylabel("Directional accuracy")
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Test-set directional accuracy — skill is the edge over "
                 "the always-up baseline, not the raw DA", y=1.02)
    fig.tight_layout()
    _save(fig, "08_directional_accuracy.png")


# ---------------------------------------------------------------------------
# Figure 09 — Forecast vs realised return over the test period (h=1)
# ---------------------------------------------------------------------------

def plot_predictions_timeseries(horizon: int = 1):
    print(f"\n[09] Forecast vs realised return over test (h={horizon})...")
    series = {compare.DISPLAY[m]: compare.load_test(m, horizon) for m in compare.MODELS}
    series = {m: df for m, df in series.items() if df is not None and not df.empty}
    if not series:
        print("  No test predictions found — skipping.")
        return

    models = [m for m in MODEL_ORDER if m in series]
    n = len(models)
    ncol = 2 if n > 1 else 1
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 3.2 * nrow),
                             squeeze=False, sharex=True)
    flat = axes.flatten()

    for ax, m in zip(flat, models):
        df = series[m]
        ax.plot(df.index, df["y_true"], color="0.6", lw=0.8, label="realised", zorder=1)
        ax.plot(df.index, df["y_pred"], color=MODEL_COLORS[m], lw=1.0,
                label="forecast", zorder=2)
        ax.axhline(0, color="black", lw=0.6, ls="--", zorder=0)
        ax.set_title(f"{m}  (n={len(df)})")
        ax.set_ylabel(f"{horizon}d log-return")
        ax.legend(fontsize=8, loc="upper right")

    for ax in flat[n:]:
        ax.set_visible(False)

    fig.suptitle(f"Forecast vs realised {horizon}-day log-return — forecasts "
                 "collapse toward the mean while realisations stay wide", y=1.01)
    fig.tight_layout()
    _save(fig, "09_predictions_timeseries.png")


# ---------------------------------------------------------------------------
# Figure 10 — Sign-of-forecast strategy equity vs buy-and-hold (h=1)
# ---------------------------------------------------------------------------

def plot_cumulative_strategy(horizon: int = 1):
    print(f"\n[10] Sign-of-forecast strategy equity (h={horizon})...")
    series = {compare.DISPLAY[m]: compare.load_test(m, horizon) for m in compare.MODELS}
    series = {m: df for m, df in series.items() if df is not None and not df.empty}
    if not series:
        print("  No test predictions found — skipping.")
        return

    # Compare strategies on the common origin dates so the curves are apples-to-apples.
    common = None
    for df in series.values():
        common = df.index if common is None else common.intersection(df.index)
    common = common.sort_values()
    if len(common) < 2:
        print("  Not enough overlapping dates across models — skipping.")
        return

    models = [m for m in MODEL_ORDER if m in series]
    fig, ax = plt.subplots(figsize=(12, 5))

    # Buy-and-hold: always long one unit; cumulative realised log-return.
    realised = series[models[0]].loc[common, "y_true"]
    ax.plot(common, realised.cumsum(), color="black", lw=1.8, ls="--",
            label="buy & hold", zorder=2)

    for m in models:
        df = series[m].loc[common]
        # Trade the sign of the forecast: +1 long, -1 short, 0 flat.
        position = np.sign(df["y_pred"].to_numpy())
        pnl = position * df["y_true"].to_numpy()
        ax.plot(common, np.cumsum(pnl), color=MODEL_COLORS[m], lw=1.4,
                label=f"{m} (sign strategy)", zorder=3)

    ax.axhline(0, color="0.5", lw=0.8, ls=":")
    ax.set_title(f"Cumulative log-return: sign-of-forecast strategy vs buy-and-hold "
                 f"(h={horizon}, n={len(common)} common origins)")
    ax.set_xlabel("Origin date")
    ax.set_ylabel("Cumulative log-return")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    _save(fig, "10_cumulative_strategy.png")


# ---------------------------------------------------------------------------
# Figure 11 — Predicted vs realised scatter (shrinkage) (h=1)
# ---------------------------------------------------------------------------

def plot_pred_vs_actual_scatter(horizon: int = 1):
    print(f"\n[11] Predicted vs realised scatter (h={horizon})...")
    series = {compare.DISPLAY[m]: compare.load_test(m, horizon) for m in compare.MODELS}
    series = {m: df for m, df in series.items() if df is not None and not df.empty}
    if not series:
        print("  No test predictions found — skipping.")
        return

    models = [m for m in MODEL_ORDER if m in series]
    n = len(models)
    ncol = 2 if n > 1 else 1
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.2 * nrow), squeeze=False)
    flat = axes.flatten()

    for ax, m in zip(flat, models):
        df = series[m]
        yt, yp = df["y_true"].to_numpy(), df["y_pred"].to_numpy()
        lim = float(np.nanmax(np.abs(yt))) * 1.05
        ax.axhline(0, color="0.7", lw=0.6); ax.axvline(0, color="0.7", lw=0.6)
        ax.plot([-lim, lim], [-lim, lim], color="black", lw=1, ls="--",
                label="perfect (y = x)")
        ax.scatter(yt, yp, s=8, alpha=0.4, color=MODEL_COLORS[m])
        # Correlation as a compact skill readout.
        if len(yt) > 2 and np.std(yp) > 0:
            corr = float(np.corrcoef(yt, yp)[0, 1])
            ax.text(0.04, 0.96, f"corr = {corr:+.3f}\npred sd / real sd = "
                    f"{np.std(yp) / np.std(yt):.2f}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8,
                    bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        ax.set_title(m)
        ax.set_xlabel("Realised log-return")
        ax.set_ylabel("Predicted log-return")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.legend(fontsize=8, loc="lower right")

    for ax in flat[n:]:
        ax.set_visible(False)

    fig.suptitle(f"Predicted vs realised {horizon}-day log-return — a narrow "
                 "vertical spread means forecasts barely move off the mean", y=1.01)
    fig.tight_layout()
    _save(fig, "11_pred_vs_actual_scatter.png")


# ---------------------------------------------------------------------------
# Figure 12 — Validation vs Test RMSE (generalisation)
# ---------------------------------------------------------------------------

def plot_validation_vs_test():
    print("\n[12] Validation vs test RMSE (generalisation)...")
    master = compare.build_master_table()
    if master.empty or master[["rmse_val", "rmse_test"]].isna().all().all():
        print("  Not enough metrics to compare — skipping.")
        return

    horizons = sorted(master["horizon"].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(5 * len(horizons), 4.5),
                             squeeze=False)
    axes = axes[0]

    for ax, h in zip(axes, horizons):
        sub = master[master["horizon"] == h]
        models = _models_in(sub)
        x = np.arange(len(models))
        w = 0.38
        val  = [float(sub[sub["model"] == m]["rmse_val"].iloc[0])  for m in models]
        test = [float(sub[sub["model"] == m]["rmse_test"].iloc[0]) for m in models]
        ax.bar(x - w / 2, val,  w, label="validation (CV)", color="0.6", zorder=3)
        ax.bar(x + w / 2, test, w, label="test", zorder=3,
               color=[MODEL_COLORS[m] for m in models])
        ax.set_title(f"h = {h} day{'s' if h > 1 else ''}")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_ylabel("RMSE (log-return)")
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Validation vs test RMSE — similar bars mean the model "
                 "generalises; a large gap flags overfitting or regime shift", y=1.02)
    fig.tight_layout()
    _save(fig, "12_validation_vs_test.png")


# ---------------------------------------------------------------------------
# Figure 13 — XGBoost gain importance (fits on the training window)
# ---------------------------------------------------------------------------

def plot_xgboost_importance(top_k: int = 12):
    print("\n[13] XGBoost feature importance (gain)...")
    try:
        from src.models import xgboost_model as xgb_mod
    except Exception as e:                       # xgboost not installed, etc.
        print(f"  Could not import xgboost_model ({e}) — skipping.")
        return

    feat_csv = xgb_mod.PROCESSED_DIR / "xgboost_features_daily.csv"
    if not feat_csv.exists():
        print("  xgboost_features_daily.csv not found — skipping.")
        return
    feature_names = [c for c in pd.read_csv(feat_csv, nrows=1).columns
                     if not c.startswith("target_") and c != "Date"]

    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(5.5 * len(HORIZONS), 5),
                             squeeze=False)
    axes = axes[0]
    drew_any = False

    for ax, h in zip(axes, HORIZONS):
        dates, y, X = xgb_mod.load_data(h)
        test_start = int(dates.searchsorted(pd.Timestamp(TEST_START)))
        train_end  = test_start - h            # purge h rows (no target leakage)
        if train_end <= 0:
            ax.set_visible(False)
            continue
        hparams = xgb_mod._load_best_hparams(h)
        params  = {**xgb_mod.BASE_PARAMS, **hparams, "importance_type": "gain"}
        model   = xgb_mod._fit_es(X[:train_end], y[:train_end], params)

        imp = pd.Series(model.feature_importances_, index=feature_names)
        imp = imp.sort_values(ascending=True).tail(top_k)
        ax.barh(imp.index, imp.values, color=MODEL_COLORS["XGBoost"], zorder=3)
        ax.set_title(f"h = {h} day{'s' if h > 1 else ''}")
        ax.set_xlabel("Gain importance (normalised)")
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        drew_any = True

    if not drew_any:
        plt.close(fig)
        print("  Not enough training data — skipping.")
        return

    fig.suptitle("XGBoost feature importance (gain) on the training window, "
                 f"top {top_k} per horizon", y=1.02)
    fig.tight_layout()
    _save(fig, "13_xgboost_importance.png")


# ---------------------------------------------------------------------------
# Figure 14 — Diebold-Mariano significance heatmap
# ---------------------------------------------------------------------------

def plot_diebold_mariano():
    print("\n[14] Diebold-Mariano significance heatmap...")
    dm = compare.dm_table()
    if dm.empty:
        print("  No overlapping model pairs to test — skipping.")
        return

    horizons = sorted(dm["horizon"].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(4.8 * len(horizons), 4.4),
                             squeeze=False)
    axes = axes[0]

    for ax, h in zip(axes, horizons):
        sub = dm[dm["horizon"] == h]
        models = [m for m in MODEL_ORDER
                  if m in set(sub["model_a"]) | set(sub["model_b"])]
        k = len(models)
        idx = {m: i for i, m in enumerate(models)}
        stat = np.full((k, k), np.nan)
        pmat = np.full((k, k), np.nan)
        for _, r in sub.iterrows():
            i, j = idx[r["model_a"]], idx[r["model_b"]]
            # Signed so a positive cell (row, col) means the ROW model has lower loss.
            stat[i, j], stat[j, i] = -r["dm"], r["dm"]
            pmat[i, j] = pmat[j, i] = r["p_value"]

        im = ax.imshow(stat, cmap="RdBu", vmin=-3, vmax=3)
        ax.set_xticks(range(k)); ax.set_yticks(range(k))
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_yticklabels(models)
        for i in range(k):
            for j in range(k):
                if i == j or np.isnan(stat[i, j]):
                    ax.text(j, i, "—", ha="center", va="center", color="0.5")
                    continue
                star = "*" if pmat[i, j] < 0.05 else ""
                ax.text(j, i, f"{stat[i, j]:+.2f}{star}", ha="center", va="center",
                        fontsize=8, color="black")
        ax.set_title(f"h = {h} day{'s' if h > 1 else ''}")

    cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label("DM statistic (row favoured if > 0)")
    fig.suptitle("Diebold-Mariano pairwise tests — '*' marks p < 0.05 "
                 "(statistically distinguishable accuracy)", y=1.03)
    _save(fig, "14_diebold_mariano.png")


# ---------------------------------------------------------------------------
# Figure 15 — Ensemble test RMSE by weighting scheme
# ---------------------------------------------------------------------------

def plot_ensemble_rmse():
    print("\n[15] Ensemble test RMSE by weighting scheme...")
    from src.evaluation import ensemble
    test = ensemble.build_test_table()
    if test.empty:
        print("  Need >=2 models per horizon — skipping.")
        return

    # Consistent bar order / colours: three schemes, then best-single, then drift.
    scheme_color = {"equal": "#9467bd", "inv_rmse": "#8c564b", "min_var": "#17becf"}
    horizons = sorted(test["horizon"].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(5 * len(horizons), 4.5),
                             squeeze=False)
    axes = axes[0]

    for ax, h in zip(axes, horizons):
        sub = test[test["horizon"] == h]
        labels, vals, colors = [], [], []
        for _, r in sub.iterrows():
            key = r["method"].split()[0]          # 'best_single [X]' -> 'best_single'
            labels.append(ensemble.METHOD_LABEL.get(r["method"], r["method"]))
            vals.append(r["rmse_test"])
            colors.append(scheme_color.get(key, "#2ca02c"))
        x = np.arange(len(labels))
        bars = ax.bar(x, vals, color=colors, width=0.6, zorder=3)

        drift = float(sub["rmse_drift"].iloc[0])
        zero  = float(sub["rmse_zero"].iloc[0])
        ax.axhline(drift, color=DRIFT_COLOR, ls="--", lw=1.5, zorder=2,
                   label=f"predict-drift ({drift:.4f})")
        ax.axhline(zero, color=ZERO_COLOR, ls=":", lw=1.5, zorder=2,
                   label=f"predict-zero ({zero:.4f})")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=8)

        ax.set_title(f"h = {h} day{'s' if h > 1 else ''}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Test RMSE (log-return)")
        ax.set_ylim(min(vals) * 0.985, max(vals + [drift, zero]) * 1.03)
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Ensemble test RMSE — weights fit on validation; a combination "
                 "adds value only if it drops below the best single model / drift",
                 y=1.02)
    fig.tight_layout()
    _save(fig, "15_ensemble_rmse.png")


# ---------------------------------------------------------------------------
# Figure 16 — Validation error-correlation heatmap (diversification diagnostic)
# ---------------------------------------------------------------------------

def plot_error_correlation():
    print("\n[16] Validation error-correlation heatmap...")
    from src.evaluation import ensemble
    corr = ensemble.build_corr_table(which="val")
    if corr.empty:
        print("  No validation panels found — skipping.")
        return

    horizons = sorted(corr["horizon"].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(4.8 * len(horizons), 4.4),
                             squeeze=False)
    axes = axes[0]
    im = None

    for ax, h in zip(axes, horizons):
        sub = corr[corr["horizon"] == h]
        models = [m for m in MODEL_ORDER if m in set(sub["model_a"])]
        wide = (sub.pivot(index="model_a", columns="model_b", values="corr")
                   .reindex(index=models, columns=models))
        im = ax.imshow(wide.to_numpy(), cmap="RdYlGn_r", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(models))); ax.set_yticks(range(len(models)))
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_yticklabels(models)
        for i in range(len(models)):
            for j in range(len(models)):
                v = wide.iloc[i, j]
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=8, color="black")
        ax.set_title(f"h = {h} day{'s' if h > 1 else ''}")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
        cbar.set_label("Error correlation")
    fig.suptitle("Validation error correlation — near 1 everywhere means the "
                 "models miss on the same days, so combining them barely helps",
                 y=1.03)
    _save(fig, "16_error_correlation.png")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

_FIGURES = {
    "07": plot_rmse_vs_baseline,
    "08": plot_directional_accuracy,
    "09": plot_predictions_timeseries,
    "10": plot_cumulative_strategy,
    "11": plot_pred_vs_actual_scatter,
    "12": plot_validation_vs_test,
    "13": plot_xgboost_importance,
    "14": plot_diebold_mariano,
    "15": plot_ensemble_rmse,
    "16": plot_error_correlation,
}


def plot_all_results():
    print("=" * 60)
    print("Generating results figures  ->  reports/figures/")
    print("=" * 60)
    with plt.style.context(STYLE):
        for fn in _FIGURES.values():
            fn()
    print(f"\nResults figures written to {FIG_DIR}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Model-results figures")
    ap.add_argument("--only", choices=sorted(_FIGURES), default=None,
                    help="render a single figure by its number (e.g. 07)")
    args = ap.parse_args()
    if args.only:
        with plt.style.context(STYLE):
            _FIGURES[args.only]()
    else:
        plot_all_results()
