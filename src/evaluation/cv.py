"""
Cross-validation harness for Round 1 (hyperparameter / structure selection).

Expanding-window folds keyed on calendar years: the train portion always starts
at 2018-01-01 and grows one year at a time; the validation portion is one
calendar year. Every model (SARIMAX, Prophet, XGBoost, LSTM) shares the same
fold definition so their validation metrics are strictly comparable.

  Fold 1: train 2018-01-01 .. 2019-12-31  |  val = 2020
  Fold 2: train 2018-01-01 .. 2020-12-31  |  val = 2021
  Fold 3: train 2018-01-01 .. 2021-12-31  |  val = 2022
  Fold 4: train 2018-01-01 .. 2022-12-31  |  val = 2023

Per-fold logic that depends on the model family (structure selection, refit
schedule, static vs walk-forward within val) lives in each model file. This
module only provides fold boundaries and an aggregation helper.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_VAL_YEARS = [2020, 2021, 2022, 2023]


@dataclass
class Fold:
    """Positional boundaries of one CV fold on the aligned (dates, y, X) arrays."""
    val_year: int
    train_start: int    # inclusive
    train_end: int      # exclusive; equals val_start (train ends where val begins)
    val_start: int      # inclusive
    val_end: int        # exclusive

    @property
    def n_train(self) -> int:
        return self.train_end - self.train_start

    @property
    def n_val(self) -> int:
        return self.val_end - self.val_start


def year_folds(dates: pd.DatetimeIndex,
               val_years=DEFAULT_VAL_YEARS,
               train_start_date: str | pd.Timestamp = "2018-01-01") -> list[Fold]:
    """Build expanding-window folds from a list of validation years.

    The train portion is always [train_start_date, Jan 1 of val_year); the val
    portion is [Jan 1 of val_year, Jan 1 of val_year + 1). Positions are integer
    indices into `dates` (searchsorted, so dates that fall on holidays / weekends
    resolve to the next available trading day).
    """
    dates = pd.DatetimeIndex(dates)
    ts0 = int(dates.searchsorted(pd.Timestamp(train_start_date)))
    folds: list[Fold] = []
    for y in val_years:
        vs = int(dates.searchsorted(pd.Timestamp(f"{y}-01-01")))
        ve = int(dates.searchsorted(pd.Timestamp(f"{y + 1}-01-01")))
        if vs >= ve:
            raise ValueError(f"No dates fall inside validation year {y}")
        if vs <= ts0:
            raise ValueError(f"Validation year {y} does not follow train_start "
                             f"{train_start_date}")
        folds.append(Fold(val_year=y, train_start=ts0, train_end=vs,
                          val_start=vs, val_end=ve))
    return folds


def aggregate_metrics(per_fold: dict[int, dict]) -> dict:
    """Aggregate a dict {fold_id: metrics_dict} into mean +/- std across folds.

    Numeric metrics get their mean under the original key and their sample std
    under key + "_std". n is summed and reported as n_total (total val
    observations across folds). Non-numeric or missing values are skipped.
    """
    keys = set()
    for m in per_fold.values():
        keys.update(m.keys())
    out: dict = {}
    for k in keys:
        vals = [m[k] for m in per_fold.values() if k in m]
        if k == "n":
            out["n_total"] = int(sum(int(v) for v in vals))
            continue
        arr = np.asarray(
            [v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)],
            dtype=float,
        )
        if arr.size == 0:
            continue
        out[k] = float(arr.mean())
        out[k + "_std"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return out
