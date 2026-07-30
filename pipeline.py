"""
Bitcoin Forecasting — Pipeline Orchestrator

Runs the full pipeline end to end:
  1. ingest   -- download raw data from Yahoo Finance
  2. features -- build ARIMA / Prophet / XGBoost / LSTM feature datasets
  3. models   -- walk-forward test forecasts for all four models
                 (SARIMAX, Prophet, XGBoost, LSTM)
  4. report   -- build the comparison tables + Diebold-Mariano tests
                 (reports/metrics/) and the result figures (reports/figures/)

Optional steps (opt-in):
  cv   -- Round-1 cross-validation (per-model hyperparameter selection)
  eda  -- exploratory data figures (input data, model-independent)

Usage:
  python pipeline.py                  # ingest + features + models + report
  python pipeline.py --skip-ingest    # reuse existing raw CSVs
  python pipeline.py --skip-models    # rebuild features + report only
  python pipeline.py --only-report    # just rebuild tables + figures from saved preds
  python pipeline.py --only-models    # just re-run the models
  python pipeline.py --cv             # also run Round-1 CV after the models
  python pipeline.py --eda            # also (re)generate the EDA figures
"""
import argparse
import time

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _step(label: str, fn) -> None:
    _header(label)
    t0 = time.time()
    fn()
    print(f"\n  Done in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def run_ingest():
    from src.data.ingest import download_all, save_raw
    btc, macro = download_all(start="2018-01-01")
    save_raw(btc, macro)


def run_features():
    from src.features.build_features import build_all
    build_all()


# Module-level knobs set by CLI flags; propagate to every model runner.
_SMOKE_DAYS: int | None = None
_QUIET: bool = False


def run_models():
    """Test walk-forward for all four models. XGBoost and LSTM read their best
    hparams from reports/predictions/cv/*_best_hparams.csv, falling back to
    sensible defaults if the Round-1 CV has not been run."""
    from src.models import sarimax, prophet_model, xgboost_model, lstm_model

    kw = {"max_test_days": _SMOKE_DAYS} if _SMOKE_DAYS else {}
    verbose = not _QUIET

    _header("SARIMAX (reduced, walk-forward)")
    sarimax.run_all(**kw)

    _header("Prophet (flat trend, walk-forward)")
    prophet_model.run_all(**kw)

    _header("XGBoost (direct, monthly refit)")
    xgboost_model.run_all(verbose=verbose, **kw)

    _header("LSTM (direct, monthly refit + warm start)")
    lstm_model.run_all(verbose=verbose, **kw)


def run_cv():
    """Round-1 cross-validation (4 expanding folds, val = 2020..2023) for all
    four models. Produces the Validation column of the final comparison table.
    Slower than the test walk-forward, so it is opt-in via --cv / --only-cv."""
    from src.models import sarimax, prophet_model, xgboost_model, lstm_model

    kw = {"max_val_days": _SMOKE_DAYS} if _SMOKE_DAYS else {}
    verbose = not _QUIET

    _header("SARIMAX (reduced) -- CV")
    sarimax.run_cv(**kw)

    _header("Prophet (flat trend) -- CV")
    prophet_model.run_cv(**kw)

    _header("XGBoost -- CV grid search")
    xgboost_model.run_cv(verbose=verbose, **kw)

    _header("LSTM -- CV grid search")
    lstm_model.run_cv(verbose=verbose, **kw)


def run_report():
    """Turn the saved predictions into the reportable artefacts: the
    Validation-vs-Test comparison tables and Diebold-Mariano tests
    (reports/metrics/) plus the result figures (reports/figures/). Reads only
    what is on disk, so it is safe to run after a partial model run."""
    from src.evaluation import compare, ensemble
    from src.visualization import results_plots

    _header("Metrics tables + Diebold-Mariano")
    compare.write_reports()

    _header("Forecast-combination (ensemble) tables")
    ensemble.write_reports()

    _header("Result figures")
    results_plots.plot_all_results()


def run_eda():
    """Exploratory data figures (input data only; independent of the models)."""
    from src.visualization.eda_plots import plot_all_eda
    plot_all_eda()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Bitcoin forecasting pipeline")
    skip = p.add_argument_group("skip steps")
    skip.add_argument("--skip-ingest",   action="store_true", help="skip data download")
    skip.add_argument("--skip-features", action="store_true", help="skip feature engineering")
    skip.add_argument("--skip-models",   action="store_true", help="skip model runs")
    skip.add_argument("--skip-report",   action="store_true", help="skip tables + figures")
    only = p.add_argument_group("run only one step")
    only.add_argument("--only-ingest",   action="store_true", help="only run data download")
    only.add_argument("--only-features", action="store_true", help="only run feature engineering")
    only.add_argument("--only-models",   action="store_true", help="only run the models")
    only.add_argument("--only-cv",       action="store_true", help="only run the Round-1 CV")
    only.add_argument("--only-report",   action="store_true", help="only build tables + figures")
    only.add_argument("--only-eda",      action="store_true", help="only build the EDA figures")
    p.add_argument("--cv",  action="store_true",
                   help="also run Round-1 cross-validation after the models step")
    p.add_argument("--eda", action="store_true",
                   help="also (re)generate the EDA figures after features")
    p.add_argument("--smoke", type=int, default=None, metavar="N",
                   help="fast end-to-end sanity check: cap every model's test "
                        "and CV walk-forward at N days (e.g. 30)")
    p.add_argument("--quiet", action="store_true",
                   help="suppress per-refit and per-combo progress lines "
                        "(keeps headers and final summaries)")
    return p.parse_args()


def main():
    args = parse_args()

    global _SMOKE_DAYS, _QUIET
    if args.smoke is not None:
        _SMOKE_DAYS = int(args.smoke)
        print(f"[SMOKE] capping every model's walk-forward at {_SMOKE_DAYS} days")
    if args.quiet:
        _QUIET = True
        print("[QUIET] per-refit and per-combo progress lines suppressed")

    # --only-* flags: run exactly that step and exit.
    only_map = [
        (args.only_ingest,   "Ingestion",          run_ingest),
        (args.only_features, "Feature engineering", run_features),
        (args.only_models,   "Models",             run_models),
        (args.only_cv,       "Round-1 CV",         run_cv),
        (args.only_report,   "Report",             run_report),
        (args.only_eda,      "EDA figures",        run_eda),
    ]
    for enabled, label, fn in only_map:
        if enabled:
            _step(f"Step 1/1 — {label}", fn)
            return

    # Full pipeline with optional skips / add-ons.
    t_start = time.time()
    candidates = [
        (not args.skip_ingest,   "Ingestion",           run_ingest),
        (not args.skip_features, "Feature engineering", run_features),
        (args.eda,               "EDA figures",         run_eda),
        (not args.skip_models,   "Models",              run_models),
        (args.cv,                "Round-1 CV",          run_cv),
        (not args.skip_report,   "Report",              run_report),
    ]
    active  = [(label, fn) for enabled, label, fn in candidates if enabled]
    skipped = [label for enabled, label, _ in candidates
               if not enabled and label not in ("EDA figures", "Round-1 CV")]

    if skipped:
        print(f"Skipping: {', '.join(skipped)}")

    n = len(active)
    for i, (label, fn) in enumerate(active, start=1):
        _step(f"Step {i}/{n} — {label}", fn)

    print(f"\n{'='*60}")
    print(f"  Pipeline complete  ({time.time() - t_start:.1f}s total)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
