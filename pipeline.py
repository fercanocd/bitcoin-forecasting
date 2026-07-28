"""
Bitcoin Forecasting — Pipeline Orchestrator

Runs the full pipeline end to end:
  1. ingest   -- download raw data from Yahoo Finance
  2. features -- build ARIMA / Prophet / XGBoost / LSTM feature datasets
  3. models   -- walk-forward forecasts for the implemented models
                 (SARIMAX, Prophet; XGBoost and LSTM are not yet implemented)

Usage:
  python pipeline.py                  # full pipeline (ingest + features + models)
  python pipeline.py --skip-ingest    # skip download (use existing raw CSVs)
  python pipeline.py --skip-features  # skip feature engineering
  python pipeline.py --skip-models    # skip model runs
  python pipeline.py --only-ingest    # only download raw data
  python pipeline.py --only-features  # only build features
  python pipeline.py --only-models    # only run the models
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


def _step(label: str, fn):
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


def run_models():
    """Test walk-forward for all four implemented models. XGBoost and LSTM read
    their best hparams from reports/predictions/cv/*_best_hparams.csv, falling
    back to sensible defaults if the CV has not been run."""
    from src.models import sarimax, prophet_model, xgboost_model, lstm_model

    _header("SARIMAX (reduced, walk-forward)")
    sarimax.run_all()

    _header("Prophet (flat trend, walk-forward)")
    prophet_model.run_all()

    _header("XGBoost (direct, daily refit)")
    xgboost_model.run_all()

    _header("LSTM (direct, monthly refit + warm start)")
    lstm_model.run_all()


def run_cv():
    """Round-1 cross-validation (4 expanding folds, val = 2020..2023) for all
    four models. Produces the Validation column of the final comparison table.
    Slower than test walk-forward, so it is opt-in via --cv / --only-cv."""
    from src.models import sarimax, prophet_model, xgboost_model, lstm_model

    _header("SARIMAX (reduced) -- CV")
    sarimax.run_cv()

    _header("Prophet (flat trend) -- CV")
    prophet_model.run_cv()

    _header("XGBoost -- CV grid search")
    xgboost_model.run_cv()

    _header("LSTM -- CV grid search")
    lstm_model.run_cv()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Bitcoin forecasting pipeline")
    skip = p.add_argument_group("skip steps")
    skip.add_argument("--skip-ingest",   action="store_true", help="skip data download")
    skip.add_argument("--skip-features", action="store_true", help="skip feature engineering")
    skip.add_argument("--skip-models",   action="store_true", help="skip model runs")
    only = p.add_argument_group("run only one step")
    only.add_argument("--only-ingest",   action="store_true", help="only run data download")
    only.add_argument("--only-features", action="store_true", help="only run feature engineering")
    only.add_argument("--only-models",   action="store_true", help="only run the models")
    only.add_argument("--only-cv",       action="store_true", help="only run the Round-1 CV")
    p.add_argument("--cv", action="store_true",
                   help="also run Round-1 cross-validation after the models step")
    return p.parse_args()


def main():
    args = parse_args()

    # --only-* flags: run exactly that step and exit
    if args.only_ingest:
        _step("Step 1/1 — Ingestion", run_ingest)
        return
    if args.only_features:
        _step("Step 1/1 — Feature engineering", run_features)
        return
    if args.only_models:
        _step("Step 1/1 — Models", run_models)
        return
    if args.only_cv:
        _step("Step 1/1 — Round-1 CV", run_cv)
        return

    # Full pipeline with optional skips
    t_start = time.time()
    steps = [
        (not args.skip_ingest,   "Step 1/3 — Ingestion",           run_ingest),
        (not args.skip_features, "Step 2/3 — Feature engineering",  run_features),
        (not args.skip_models,   "Step 3/3 — Models",              run_models),
    ]
    if args.cv:
        steps.append((True, "Step 4/4 — Round-1 CV", run_cv))

    active = [(label, fn) for enabled, label, fn in steps if enabled]
    skipped = [label for enabled, label, _ in steps if not enabled]

    if skipped:
        print(f"Skipping: {', '.join(s.split('—')[1].strip() for s in skipped)}")

    for label, fn in active:
        _step(label, fn)

    print(f"\n{'='*60}")
    print(f"  Pipeline complete  ({time.time() - t_start:.1f}s total)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
