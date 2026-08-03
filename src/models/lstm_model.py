"""
LSTM direct multi-horizon walk-forward forecasting for Bitcoin log-returns.

Reads   : data/processed/lstm_features_daily.csv  (19 features, 3 targets)
Outputs : reports/predictions/lstm_{h}d.csv               (test walk-forward)
          reports/predictions/cv/lstm_fold{year}_{h}d.csv (CV predictions)
          reports/predictions/cv/lstm_best_hparams.csv    (best combo per h)

Strategy:
  * DIRECT multi-horizon -- three independent LSTMs (one per horizon), each
    trained to predict target_log_return_{h}d = ln(P_{t+h} / P_t) directly.
  * SEQUENCE input -- sliding window of T=60 days of features. The prediction
    at position i uses X[i - T + 1 : i + 1] under the current StandardScaler
    (fit on train only, no leakage).
  * PURGE h rows at the end of every train window: row t's target uses
    P_{t+h}, so training targets at rows [i - h + 1, i - 1] would leak future
    prices when fitting at position i. Keep only targets with j <= i - h.
  * Round 1 (CV): STATIC train per fold -- fit one LSTM per fold, predict all
    val sequences in one batch. Cheaper than walk-forward inside val and
    consistent with the XGBoost CV protocol.
  * Round 2 (test): MONTHLY refit with WARM START -- at each month start,
    retrain on all data up to that day, initialising the network with the
    previous month's weights so training converges in a few epochs. Between
    refits the frozen model is used for predictions. Justified simplification:
    daily backprop on ~2000 sequences x 3 horizons x 883 origins is
    computationally infeasible; monthly refits with warm start capture regime
    drift at a coarser but tractable cadence.
  * Fixed base training config: MSE loss, Adam (lr=1e-3, wd=1e-5), batch=32,
    max 100 epochs with early stopping on a 10% chronological val split of
    train, gradient clipping = 1.0.
  * Small manual grid (8 combos): hidden_size x num_layers x dropout.

Usage:
    python -m src.models.lstm_model --cv                              # Round 1 CV grid
    python -m src.models.lstm_model                                    # Round 2 test
    python -m src.models.lstm_model --horizon 7 --max-test-days 60     # smoke run
"""
from __future__ import annotations
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.config import (HORIZONS, SEED, TEST_END, TEST_START, TRAIN_START)
from src.evaluation.cv import (DEFAULT_VAL_YEARS, aggregate_metrics, year_folds)
from src.evaluation.metrics import summary
from src.evaluation.runtime import measure, record
from src.evaluation.walk_forward import month_start_positions
import time

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
RESULTS_DIR   = Path(__file__).resolve().parents[2] / "reports" / "predictions"

TARGET_TEMPLATE = "target_log_return_{h}d"
SEQ_LEN         = 60

# Training config held fixed across the grid.
BASE_TRAIN = dict(
    lr=1e-3,
    batch_size=32,
    weight_decay=1e-5,
    max_epochs=100,
    patience=10,
    grad_clip=1.0,
    val_fraction=0.10,          # chronological tail of train as internal val
)

# 2 x 2 x 2 = 8 combinations.
GRID = [
    dict(hidden_size=h, num_layers=n, dropout=d)
    for h, n, d in product([32, 64], [1, 2], [0.0, 0.2])
]

DEFAULT_HPARAMS = dict(hidden_size=32, num_layers=1, dropout=0.0)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(horizon: int):
    """Load the LSTM feature CSV and return (dates, y, X) for one horizon."""
    df = pd.read_csv(PROCESSED_DIR / "lstm_features_daily.csv",
                     index_col="Date", parse_dates=True)
    target_col   = TARGET_TEMPLATE.format(h=horizon)
    feature_cols = [c for c in df.columns if not c.startswith("target_")]
    df = df.dropna(subset=[target_col])
    return (df.index,
            df[target_col].to_numpy(dtype=np.float32),
            df[feature_cols].to_numpy(dtype=np.float32))


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int,
                   idx_start: int, idx_end: int):
    """Build (X_seq, y_seq) for target positions in [idx_start, idx_end).

    Each sequence is X[i - seq_len + 1 : i + 1]; the corresponding target
    is y[i]. idx_start is clamped upward so the first sequence is fully
    populated (i.e. i - seq_len + 1 >= 0).
    """
    idx_start = max(idx_start, seq_len - 1)
    n_seq     = max(0, idx_end - idx_start)
    n_feat    = X.shape[1]
    X_seq = np.zeros((n_seq, seq_len, n_feat), dtype=np.float32)
    y_seq = np.zeros(n_seq, dtype=np.float32)
    for k, i in enumerate(range(idx_start, idx_end)):
        X_seq[k] = X[i - seq_len + 1 : i + 1]
        y_seq[k] = y[i]
    positions = np.arange(idx_start, idx_end, dtype=np.int64)
    return X_seq, y_seq, positions


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LSTMForecaster(nn.Module):
    """Multi-layer LSTM taking the last hidden state through a linear head."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 dropout: float):
        super().__init__()
        # PyTorch requires dropout=0 when num_layers == 1 (else warns and is a no-op).
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)          # out: (batch, T, hidden)
        return self.head(out[:, -1, :]).squeeze(-1)   # last-step -> scalar per sample


def _train_one(X_seq: np.ndarray, y_seq: np.ndarray, hparams: dict,
               train_cfg: dict, prev_state: dict | None = None,
               seed: int = SEED, verbose: bool = False):
    """Train one LSTM on (X_seq, y_seq) with early stopping.

    Val split is the chronological TAIL of the training sequences (random
    splits would leak future returns into the internal val). Returns
    (model, best_val_loss, epoch_stopped, epochs_run).
    """
    torch.manual_seed(seed)
    n     = len(X_seq)
    n_val = max(1, int(round(n * train_cfg["val_fraction"])))
    if n_val >= n:
        n_val = 1
    X_tr, X_va = X_seq[:-n_val], X_seq[-n_val:]
    y_tr, y_va = y_seq[:-n_val], y_seq[-n_val:]

    X_tr_t = torch.from_numpy(X_tr)
    y_tr_t = torch.from_numpy(y_tr)
    X_va_t = torch.from_numpy(X_va)
    y_va_t = torch.from_numpy(y_va)

    model = LSTMForecaster(
        input_size=X_seq.shape[2],
        hidden_size=hparams["hidden_size"],
        num_layers=hparams["num_layers"],
        dropout=hparams["dropout"],
    )
    if prev_state is not None:
        try:
            model.load_state_dict(prev_state)
        except RuntimeError:
            pass   # architecture mismatch -> cold start silently
    opt     = torch.optim.Adam(model.parameters(),
                               lr=train_cfg["lr"],
                               weight_decay=train_cfg["weight_decay"])
    loss_fn = nn.MSELoss()

    best_val   = float("inf")
    best_state = None
    no_improve = 0
    epochs_run = 0

    batch_size = train_cfg["batch_size"]
    n_tr       = len(X_tr_t)

    for epoch in range(train_cfg["max_epochs"]):
        epochs_run = epoch + 1
        model.train()
        perm = torch.randperm(n_tr)
        for start in range(0, n_tr, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = X_tr_t[idx], y_tr_t[idx]
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           train_cfg["grad_clip"])
            opt.step()

        model.eval()
        with torch.no_grad():
            v_loss = loss_fn(model(X_va_t), y_va_t).item()
        if v_loss < best_val - 1e-7:
            best_val   = v_loss
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= train_cfg["patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val, epochs_run


def _predict(model: LSTMForecaster, X_seq: np.ndarray) -> np.ndarray:
    """Batched inference."""
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X_seq)).numpy()


# ---------------------------------------------------------------------------
# Round 1 -- CV grid search (static train per fold)
# ---------------------------------------------------------------------------

def _cv_one_combo(horizon: int, hparams: dict, dates, y, X, folds,
                  max_val_days: int | None, train_cfg: dict):
    """Fit + predict one hparams combo across all folds, static-per-fold."""
    per_fold: dict[int, dict] = {}
    per_fold_preds: dict[int, tuple] = {}

    for f in folds:
        train_end = f.train_end - horizon    # purge h rows
        if train_end - f.train_start < SEQ_LEN + 10:
            raise ValueError(f"Fold val={f.val_year}: not enough train rows after purge")

        # Scaler fit on train features only, then transformed over the whole series.
        scaler   = StandardScaler().fit(X[f.train_start:train_end])
        X_scaled = scaler.transform(X).astype(np.float32)

        # Train sequences: target positions in [SEQ_LEN - 1, train_end).
        X_tr, y_tr, _ = make_sequences(X_scaled, y, SEQ_LEN, SEQ_LEN - 1, train_end)
        model, _, _   = _train_one(X_tr, y_tr, hparams, train_cfg)

        # Val sequences.
        val_end       = (min(f.val_end, f.val_start + max_val_days)
                         if max_val_days is not None else f.val_end)
        X_va, y_va, positions = make_sequences(X_scaled, y, SEQ_LEN, f.val_start, val_end)
        y_pred        = _predict(model, X_va)

        # y is the h-day cumulative target; divide by h to recover mu_daily.
        fold_drift = float(y[f.train_start:train_end].mean()) / horizon
        m          = summary(y_va, y_pred, drift=fold_drift, horizon=horizon)
        per_fold[f.val_year]        = m
        per_fold_preds[f.val_year]  = (dates[positions], y_va, y_pred)

    return per_fold, per_fold_preds


def run_cv(horizons=None, val_years=None, max_val_days=None, verbose=True):
    """Round 1: grid search over GRID, static-per-fold, RMSE-then-edge selection."""
    horizons  = horizons  or HORIZONS
    val_years = val_years or DEFAULT_VAL_YEARS
    train_cfg = BASE_TRAIN.copy()
    print(f"\n[LSTM] CV grid search: {len(GRID)} combos x {len(val_years)} folds "
          f"x {len(horizons)} horizons  (T={SEQ_LEN})")

    cv_dir = RESULTS_DIR / "cv"
    cv_dir.mkdir(parents=True, exist_ok=True)

    best_by_h    = {}
    hparams_rows = []

    for h in horizons:
        t0 = time.time()
        print(f"\n  --- horizon = {h}d ---")
        dates, y, X = load_data(h)
        folds       = year_folds(dates, val_years=val_years, train_start_date=TRAIN_START)

        combo_results = []
        for combo in GRID:
            per_fold, per_fold_preds = _cv_one_combo(
                h, combo, dates, y, X, folds, max_val_days, train_cfg,
            )
            agg = aggregate_metrics(per_fold)
            combo_results.append((combo, agg, per_fold, per_fold_preds))
            if verbose:
                print(f"    hidden={combo['hidden_size']}  layers={combo['num_layers']}  "
                      f"drop={combo['dropout']}   "
                      f"RMSE {agg['rmse']:.5f} +/- {agg['rmse_std']:.5f}   "
                      f"DA {agg['da']:.3f}   edge {agg['da_edge']:+.3f}   "
                      f"n_total={agg['n_total']}")

        combo_results.sort(key=lambda r: (r[1]["rmse"], -r[1]["da_edge"]))
        best_combo, best_agg, best_per_fold, best_per_fold_preds = combo_results[0]
        best_by_h[h] = {"combo": best_combo, "agg": best_agg,
                        "per_fold": best_per_fold}

        print(f"  Best h={h}d: hidden={best_combo['hidden_size']}, "
              f"layers={best_combo['num_layers']}, dropout={best_combo['dropout']}")
        print(f"    -> RMSE {best_agg['rmse']:.5f} +/- {best_agg['rmse_std']:.5f}   "
              f"MAE {best_agg['mae']:.5f}   "
              f"DA {best_agg['da']:.3f}   edge {best_agg['da_edge']:+.3f}   "
              f"n_total={best_agg['n_total']}")

        for fold_year, (fold_dates, yt, yp) in best_per_fold_preds.items():
            pd.DataFrame({"y_true": yt, "y_pred": yp}, index=fold_dates) \
              .rename_axis("date") \
              .to_csv(cv_dir / f"lstm_fold{fold_year}_{h}d.csv")

        hparams_rows.append({
            "horizon": h, **best_combo,
            "rmse": best_agg["rmse"], "rmse_std": best_agg["rmse_std"],
            "mae":  best_agg["mae"],  "mae_std":  best_agg["mae_std"],
            "da":   best_agg["da"],   "da_std":   best_agg["da_std"],
            "da_edge": best_agg["da_edge"], "da_edge_std": best_agg["da_edge_std"],
            "n_total": best_agg["n_total"],
        })
        record("lstm", h, "cv", time.time() - t0,
               refit="static-per-fold",
               n_predictions=int(best_agg["n_total"]))

    pd.DataFrame(hparams_rows).to_csv(cv_dir / "lstm_best_hparams.csv", index=False)
    print(f"\n  Best hparams saved -> {cv_dir / 'lstm_best_hparams.csv'}")
    return best_by_h


# ---------------------------------------------------------------------------
# Round 2 -- test walk-forward (monthly refit, warm start)
# ---------------------------------------------------------------------------

def _load_best_hparams(horizon: int) -> dict:
    path = RESULTS_DIR / "cv" / "lstm_best_hparams.csv"
    if not path.exists():
        print(f"  No CV hparams at {path}; using DEFAULT_HPARAMS.")
        return DEFAULT_HPARAMS.copy()
    df  = pd.read_csv(path)
    row = df.loc[df["horizon"] == horizon]
    if row.empty:
        print(f"  No CV row for h={horizon}d; using DEFAULT_HPARAMS.")
        return DEFAULT_HPARAMS.copy()
    row = row.iloc[0]
    return dict(hidden_size=int(row["hidden_size"]),
                num_layers=int(row["num_layers"]),
                dropout=float(row["dropout"]))


def run(horizon=1, hparams=None, max_test_days=None, verbose=True, _data=None):
    """Test walk-forward: refit monthly with warm start, predict daily."""
    hparams   = hparams or _load_best_hparams(horizon)
    train_cfg = BASE_TRAIN.copy()

    dates, y, X = _data if _data is not None else load_data(horizon)
    n           = len(y)

    test_start = int(dates.searchsorted(pd.Timestamp(TEST_START)))
    test_end   = n
    if TEST_END is not None:
        test_end = min(test_end, int(dates.searchsorted(pd.Timestamp(TEST_END), side="right")))
    if max_test_days is not None:
        test_end = min(test_end, test_start + max_test_days)

    print(f"\n[LSTM] {horizon}-day walk-forward (direct, monthly refit + warm start)")
    print(f"  Hparams: {hparams}   |  T={SEQ_LEN}")
    print(f"  Train : {dates[0].date()} .. {dates[test_start-1].date()}  ({test_start} obs)")
    print(f"  Test  : {dates[test_start].date()} .. {dates[test_end-1].date()}  ({test_end-test_start} obs)")

    refit_positions = month_start_positions(dates, test_start, test_end)
    print(f"  Refits: {len(refit_positions)} monthly")

    model, prev_state, X_scaled = None, None, None
    rows: list = []

    t0 = time.time()
    for i in range(test_start, test_end):
        if i in refit_positions:
            train_end = i - horizon + 1
            if train_end - SEQ_LEN + 1 <= 0:
                continue                        # not enough history yet
            scaler   = StandardScaler().fit(X[:train_end])
            X_scaled = scaler.transform(X).astype(np.float32)
            X_tr, y_tr, _ = make_sequences(X_scaled, y, SEQ_LEN, SEQ_LEN - 1, train_end)
            model, best_val, epochs = _train_one(X_tr, y_tr, hparams, train_cfg,
                                                 prev_state=prev_state)
            prev_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            if verbose:
                print(f"    [refit {dates[i].date()}] train_end={train_end}  "
                      f"best_val={best_val:.6f}  epochs={epochs}")

        if X_scaled is None or model is None or i < SEQ_LEN - 1:
            continue
        # Predict target at position i using X[i-T+1 : i+1] under current scaler.
        x_pred = X_scaled[i - SEQ_LEN + 1 : i + 1][None, :, :]   # (1, T, F)
        model.eval()
        with torch.no_grad():
            y_pred = float(model(torch.from_numpy(x_pred)).item())
        rows.append((dates[i], float(y[i]), y_pred))
    record("lstm", horizon, "test", time.time() - t0,
           refit="monthly", n_predictions=len(rows))

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
    out = RESULTS_DIR / f"lstm_{horizon}d.csv"
    preds.to_csv(out)
    print(f"  Saved -> {out}")
    return preds, m


def run_all(horizons=None, hparams_by_h=None, max_test_days=None, verbose=True):
    """``verbose=False`` suppresses the per-refit progress lines."""
    horizons = horizons or HORIZONS
    print(f"\n[LSTM] multi-horizon test walk-forward (horizons={horizons})")
    results = {}
    for h in horizons:
        hp = None if hparams_by_h is None else hparams_by_h.get(h)
        _, m = run(horizon=h, hparams=hp, max_test_days=max_test_days,
                   verbose=verbose)
        results[h] = m

    print(f"\n{'='*72}\n  LSTM summary\n{'='*72}")
    for h, m in results.items():
        print(f"    h={h:>2}d   RMSE {m['rmse']:.5f} (zero {m['rmse_zero']:.5f}, drift {m['rmse_drift']:.5f})"
              f"   MAE {m['mae']:.5f}   DA {m['da']:.3f} (edge {m['da_edge']:+.3f})")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="LSTM direct multi-horizon walk-forward")
    ap.add_argument("--horizon", type=int, default=None,
                    help="single horizon in days (default: all in config.HORIZONS)")
    ap.add_argument("--max-test-days", type=int, default=None,
                    help="limit the test horizon for a quick smoke run")
    ap.add_argument("--cv", action="store_true",
                    help="run Round-1 grid search instead of the test walk-forward")
    ap.add_argument("--max-val-days", type=int, default=None,
                    help="limit each CV fold's val length for a quick smoke run")
    args = ap.parse_args()
    if args.cv:
        horizons = None if args.horizon is None else [args.horizon]
        run_cv(horizons=horizons, max_val_days=args.max_val_days)
    elif args.horizon is None:
        run_all(max_test_days=args.max_test_days)
    else:
        run(horizon=args.horizon, max_test_days=args.max_test_days)
