"""
XGBoost walk-forward forecasting for Bitcoin log-returns.

Reads  : data/processed/xgboost_features_daily.csv
Outputs: walk-forward one-step-ahead predictions (log-return)

Model         : xgboost.XGBRegressor
Features (26) : log_return, log(P/SMA)×3, log(P/EMA)×2, MACD,
                RSI-14 [0,1], log_bb_width, log_atr_norm, log_volume_ratio,
                log_return_lag×7, calendar×4, macro_return×4
Hyperparameters: tuned via time-series cross-validation on training window
Validation    : expanding-window walk-forward

Usage:
    python -m src.models.xgboost_model
"""
# TODO: implement walk-forward XGBoost
