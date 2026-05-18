"""
LSTM walk-forward forecasting for Bitcoin log-returns (PyTorch).

Reads  : data/processed/lstm_features_daily.csv
Outputs: walk-forward one-step-ahead predictions (log-return)

Architecture  : multi-layer LSTM → Linear(hidden, 1)
Features (19) : log_return, log(P/SMA)×3, log(P/EMA)×2, MACD,
                RSI-14 [0,1], log_bb_width, log_atr_norm, log_volume_ratio,
                calendar×4, macro_return×4
Preprocessing : StandardScaler fitted on training window only (no leakage)
                Applied per walk-forward step
Training      : Adam optimiser, gradient clipping (clip_grad_norm_)
                to handle large return spikes
Sequence      : sliding window of T steps fed as (batch, T, n_features)
Validation    : expanding-window walk-forward

Usage:
    python -m src.models.lstm_model
"""
# TODO: implement walk-forward LSTM
