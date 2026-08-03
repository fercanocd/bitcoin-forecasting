"""
Persistent runtime log for the four model families.

Each model's ``run_all`` / ``run_cv`` wraps its work with :func:`measure` and
appends one row to ``reports/predictions/runtimes.csv``. ``compare.py`` reads
that file to enrich the comparison tables with a wall-clock training time,
so re-running just one model updates its column without touching the others.

Row schema:
  model         -- "sarimax" | "prophet" | "xgboost" | "lstm"
  horizon       -- integer horizon in days, or -1 when a single walk-forward
                    serves every horizon jointly (recursive models)
  kind          -- "test" | "cv"
  seconds       -- wall-clock elapsed time of the invocation
  refit         -- refit cadence used: "step" (every origin), "monthly", "weekly", ...
  n_predictions -- number of prediction rows produced (per horizon for direct
                    models, total across horizons for recursive)
  timestamp     -- UTC ISO-8601 of when the row was written
"""
from __future__ import annotations
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RUNTIMES_CSV = (Path(__file__).resolve().parents[2]
                / "reports" / "predictions" / "runtimes.csv")

COLUMNS = ["model", "horizon", "kind", "seconds", "refit",
           "n_predictions", "timestamp"]


def record(model: str, horizon: int, kind: str, seconds: float,
           refit: str = "step", n_predictions: int | None = None) -> None:
    """Append one runtime row to the persistent log.

    Overwrites any previous row for the same (model, horizon, kind, refit) so
    the file always reflects the latest execution. Horizon = -1 marks a joint
    run that covers every horizon at once (recursive models)."""
    RUNTIMES_CSV.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "model": model, "horizon": int(horizon), "kind": kind,
        "seconds": float(seconds), "refit": refit,
        "n_predictions": None if n_predictions is None else int(n_predictions),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if RUNTIMES_CSV.exists():
        df = pd.read_csv(RUNTIMES_CSV)
        mask = ~((df["model"] == model) & (df["horizon"] == horizon)
                 & (df["kind"] == kind) & (df["refit"] == refit))
        df = df.loc[mask]
        df = pd.concat([df, pd.DataFrame([row], columns=COLUMNS)], ignore_index=True)
    else:
        df = pd.DataFrame([row], columns=COLUMNS)
    df.to_csv(RUNTIMES_CSV, index=False)


@contextmanager
def measure(model: str, horizon: int, kind: str,
            refit: str = "step", n_predictions_fn=None):
    """Context manager: time the wrapped block and log the result.

    ``n_predictions_fn`` is called after the block with no arguments to
    resolve the number of predictions produced (kept as a callable so it can
    reference variables that only exist after the block runs)."""
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        n_pred = None
        if n_predictions_fn is not None:
            try:
                n_pred = n_predictions_fn()
            except Exception:
                n_pred = None
        record(model, horizon, kind, dt, refit=refit, n_predictions=n_pred)


def load() -> pd.DataFrame:
    """Return the runtimes log as a DataFrame, or an empty one if missing."""
    if not RUNTIMES_CSV.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(RUNTIMES_CSV)


def fmt(seconds: float) -> str:
    """Format a duration as ``12s`` / ``3m14s`` / ``1h05m`` for tables."""
    if seconds is None or pd.isna(seconds):
        return "-"
    s = int(round(float(seconds)))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"
