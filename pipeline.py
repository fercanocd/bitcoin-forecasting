"""
Bitcoin Forecasting — Pipeline Orchestrator

Runs the full data preparation pipeline end to end:
  1. ingest   -- download raw data from Yahoo Finance
  2. features -- build ARIMA / Prophet / XGBoost / LSTM feature datasets

Usage:
  python pipeline.py                  # full pipeline
  python pipeline.py --skip-ingest    # skip download (use existing raw CSVs)
  python pipeline.py --skip-features  # skip feature engineering
  python pipeline.py --only-ingest    # only download raw data
  python pipeline.py --only-features  # only build features
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Bitcoin forecasting pipeline")
    skip = p.add_argument_group("skip steps")
    skip.add_argument("--skip-ingest",   action="store_true", help="skip data download")
    skip.add_argument("--skip-features", action="store_true", help="skip feature engineering")
    only = p.add_argument_group("run only one step")
    only.add_argument("--only-ingest",   action="store_true", help="only run data download")
    only.add_argument("--only-features", action="store_true", help="only run feature engineering")
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

    # Full pipeline with optional skips
    t_start = time.time()
    steps = [
        (not args.skip_ingest,   "Step 1/2 — Ingestion",           run_ingest),
        (not args.skip_features, "Step 2/2 — Feature engineering",  run_features),
    ]

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
