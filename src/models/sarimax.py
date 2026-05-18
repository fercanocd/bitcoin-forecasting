"""
SARIMAX walk-forward forecasting for Bitcoin log-returns.

Reads  : data/processed/arima_features_daily.csv
Outputs: walk-forward one-step-ahead predictions (log-return)

Model family  : ARIMA / SARIMA / SARIMAX (statsmodels)
Endogenous    : log_return
Exogenous     : log_volume_ratio, sp500_log_return, gold_log_return,
                dxy_log_return, eth_log_return
Order search  : auto_arima (pmdarima) on the initial training window,
                then the selected order is reused across the walk-forward loop
Validation    : expanding-window walk-forward with monthly refits

Usage:
    python -m src.models.sarimax
"""
# TODO: implement walk-forward SARIMAX
