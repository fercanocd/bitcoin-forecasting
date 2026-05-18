"""
Prophet (Meta) walk-forward forecasting for Bitcoin log-returns.

Reads  : data/processed/prophet_features_daily.csv
Outputs: walk-forward one-step-ahead predictions (log-return)

Model family  : Prophet (additive structural model: trend + seasonality +
                changepoints + regressors)
Target (`y`)  : log_return  (log(Pt / Pt-1) on day t)
Regressors    : log_volume_ratio, sp500_log_return, gold_log_return,
                dxy_log_return, eth_log_return  (attached via add_regressor)

Configuration :
    seasonality_mode       = "additive"     # log-returns are unbounded & ~zero-centered
    yearly_seasonality     = True           # macro / halving cycles
    weekly_seasonality     = True           # potential weekend effects in crypto
    daily_seasonality      = False          # data is already daily
    changepoint_prior_scale = 0.05          # Prophet default

Validation    : expanding-window walk-forward, refit at every step.
                At step t, train on rows up to and including t and forecast
                r_{t+1}. Because Prophet needs regressor values at the
                forecast horizon, the last observed regressor row (day t) is
                carried forward to the t+1 future frame -- consistent with
                the information set available at prediction time.

Usage:
    python -m src.models.prophet_model
"""
# TODO: implement walk-forward Prophet
