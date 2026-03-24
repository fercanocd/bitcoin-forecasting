"""
XGBoost walk-forward forecasting for Bitcoin log-returns.

Reads  : data/processed/xgboost_features_daily.csv
Outputs: walk-forward one-step-ahead predictions (log-return)

Model         : xgboost.XGBRegressor
Features (31) : Close, log_return, SMA×4, EMA×2, RSI-14 [0,1],
                bb_width, atr_14, log_volume_ratio,
                return_lag×7, calendar×4, macro_close×4, macro_return×4
Hyperparameters: tuned via time-series cross-validation on training window
Validation    : expanding-window walk-forward

Usage:
    python -m src.models.xgboost_model
"""
# TODO: implement walk-forward XGBoost
