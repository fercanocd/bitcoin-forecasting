# Bitcoin Forecasting with Machine Learning Techniques

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Comparative study of four model families for **multi-horizon Bitcoin log-return forecasting** at horizons of 1, 7 and 30 days:

| Model | Type | Strategy |
|---|---|---|
| **SARIMAX** | Classical econometrics | Recursive (one daily model, projected h steps) |
| **Prophet** | Additive structural decomposition | Recursive (same as SARIMAX) |
| **XGBoost** | Gradient boosting | Direct (one model per horizon) |
| **LSTM** | Deep learning | Direct (one model per horizon) |

All models share the same prediction target (h-day cumulative log-return) and are evaluated under identical expanding walk-forward conditions. Macroeconomic context variables — S&P 500, Gold, US Dollar Index, Ethereum — are included as exogenous inputs based on documented co-movement with Bitcoin prices (Bouri et al., 2017).

---

## Results

Out-of-sample walk-forward results (test period 2024-01-01 → 2026-05-20, n=871 origins). `edge` = DA − DA_up (directional skill over the "always predict up" baseline).

| Model | h | RMSE | vs drift | DA | edge |
|---|---|---|---|---|---|
| SARIMAX | 1d | 0.02534 | +0.000 | 0.520 | +0.014 |
| SARIMAX | 7d | 0.06266 | +0.001 | 0.535 | +0.001 |
| SARIMAX | 30d | 0.14137 | +0.001 | 0.537 | +0.000 |
| Prophet | 1d | 0.02557 | +0.001 | 0.504 | −0.002 |
| Prophet | 7d | 0.06834 | +0.009 | 0.482 | −0.052 |
| Prophet | 30d | 0.16296 | +0.023 | 0.503 | −0.034 |
| XGBoost | — | — | — | — | — |
| LSTM | — | — | — | — | — |

**Key findings (baselines):** Neither SARIMAX nor Prophet beats the predict-drift baseline in RMSE. SARIMAX posts a small positive DA edge at h=1 (+0.014), attributable to ETH co-movement; all other edges are ≈ 0 or negative. Prophet is worse than SARIMAX at every horizon, increasingly so as h grows. These results set the bar for the ML models.

---

## Methodology

### Experimental design

The experiment uses an **expanding walk-forward** evaluation scheme. At each test origin `t` the model is fitted exclusively on data up to `t−1` and then forecasts the next `h` days. The training window grows by one observation each day — no future data ever enters the fit. This mirrors realistic deployment: you forecast with what you know today.

```
|←————————— train ——————————→|←———— test (walk-forward) ————→|
 2018-01-21              2023-12-31  2024-01-01          2026-05-20
                                     ↑ origin 1
                                          ↑ origin 2
                                               ↑ ...  (871 origins)
```

### Target variable

The prediction target at each origin is the **cumulative log-return over the next h days**:

```
target_h = ln(P_{t+h} / P_t) = Σ ln(P_{t+k} / P_{t+k−1})   for k = 1..h
```

Log-returns are additive and approximately stationary, which makes them suitable for both classical and ML models without differencing or scaling.

### Recursive vs direct strategy

- **Recursive (SARIMAX, Prophet):** a single daily model is fitted; the h-day forecast is obtained by iterating the model h times. The model is identical across horizons — only the number of projected steps differs. This means one walk-forward pass serves all three horizons simultaneously.
- **Direct (XGBoost, LSTM):** a separate model is trained per horizon, optimising directly against the h-day target. Models are genuinely different across horizons (the optimal features and weights for 1-day ahead differ from those for 30-day ahead), so three independent walk-forward passes are required.

### Train / test split

| Set | Period | Observations | Role |
|---|---|---|---|
| **Train** | 2018-01-21 → 2023-12-31 | 2,171 days | Fit models, select regressors and order |
| **Test** | 2024-01-01 → 2026-05-20 | 871 days | Out-of-sample honest evaluation |

There is no separate validation set. For the ML models (XGBoost, LSTM), hyperparameter selection is performed via time-series cross-validation entirely within the training window (expanding folds, no look-ahead).

### Evaluation metrics

**RMSE and MAE** measure forecast magnitude error. Both are reported against two honest baselines:

- `predict-zero` — always forecast zero return (random walk without drift).
- `predict-drift` — always forecast `h × drift_train`, where `drift_train` is the mean daily log-return on the training set (random walk with drift). This is the harder and more honest bar: a model that merely captures BTC's upward trend will match it but add no conditional information.

**DA (Directional Accuracy)** measures the share of origins where the predicted sign matches the realised sign. Reported alongside:

- `DA_up` — the share of positive realisations in the test set (≈ 50.6%); equivalent to "always predict up".
- `DA edge = DA − DA_up` — conditional directional skill net of drift. An edge of 0 means the model adds nothing beyond "always predict up".

**Diebold-Mariano (DM) test** assesses whether the RMSE difference between two models is statistically significant or within sampling noise. Implemented with Newey-West long-run variance (lag = h−1, required for h-step-ahead overlapping forecast errors) and the Harvey-Leybourne-Newbold (1997) small-sample correction, referred to a Student-t(N−1) distribution.

---

## Feature Engineering

Four feature sets are constructed, each tailored to its model family:

- **SARIMAX** (6 features): log-return, log volume ratio, macro log-returns (S&P 500, Gold, DXY, ETH). Backward elimination by p-value on the training window retains only ETH (p=0.002); all others dropped (p>0.15). The reduced model is the official baseline.
- **Prophet** (6 features): same regressor set as SARIMAX — trend, weekly and yearly seasonality are modelled internally by Prophet, not fed as features.
- **XGBoost** (26 features): log close-to-SMA ratios (×3), log close-to-EMA ratios (×2), MACD, RSI-14 [0,1], log Bollinger width, log normalised ATR, log volume ratio, lagged returns (×7), calendar sin/cos, macro log-returns.
- **LSTM** (19 features): same dimensionless trend/momentum/volatility block as XGBoost, with normalised ATR (ATR/P_t) and no lag block (the sequence window replaces explicit lags).

XGBoost and LSTM share an identical scale-invariant feature space: any performance gap between them reflects model architecture rather than feature access. The same is true between SARIMAX and Prophet at the baseline tier.

---

## Project Structure

```
bitcoin-forecasting/
├── data/                           # gitignored — regenerate with: python pipeline.py
│   ├── raw/                        # btc_ohlcv.csv, macro_raw.csv (Yahoo Finance)
│   └── processed/                  # arima / prophet / xgboost / lstm feature CSVs
├── reports/                        # gitignored — regenerate with eda_plots.py
│   ├── figures/                    # EDA plots + interactive candlestick chart
│   └── predictions/                # walk-forward predictions: sarimax/prophet_{h}d.csv
├── src/
│   ├── config.py                   # Central config: dates, horizons, seed
│   ├── data/
│   │   └── ingest.py               # Download via yfinance
│   ├── features/
│   │   └── build_features.py       # Feature engineering (SARIMAX / Prophet / XGBoost / LSTM)
│   ├── models/
│   │   ├── sarimax.py              # Walk-forward SARIMAX (recursive, multi-horizon)
│   │   ├── prophet_model.py        # Walk-forward Prophet (recursive, multi-horizon)
│   │   ├── xgboost_model.py        # Walk-forward XGBoost (direct, one model per horizon)
│   │   └── lstm_model.py           # Walk-forward LSTM — PyTorch (direct, one model per horizon)
│   ├── evaluation/
│   │   ├── metrics.py              # RMSE, MAE, DA, DA edge, Diebold-Mariano
│   │   └── walk_forward.py         # Expanding walk-forward engine (recursive + direct)
│   └── visualization/
│       ├── eda_plots.py            # Static EDA figures
│       └── candlestick_interactive.py  # Interactive HTML candlestick chart
├── pipeline.py                     # End-to-end pipeline orchestrator
└── requirements.txt
```

---

## Quickstart

```bash
git clone https://github.com/fercanocd/bitcoin-forecasting.git
cd bitcoin-forecasting
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

**Run the full pipeline** (ingest → features → all models → evaluation):

```bash
python pipeline.py
```

**Run individual models:**

```bash
python -m src.models.sarimax                        # all horizons
python -m src.models.sarimax --horizon 1            # single horizon
python -m src.models.sarimax --refit monthly        # fast dev run (monthly refit)
python -m src.models.prophet_model
python -m src.models.xgboost_model
python -m src.models.lstm_model
```

---

## Data

Daily OHLCV data retrieved from **Yahoo Finance** via `yfinance`, covering **1 January 2018 to present**.

| Series | Ticker | Role |
|---|---|---|
| Bitcoin | BTC-USD | Primary target |
| S&P 500 | ^GSPC | Global risk appetite proxy |
| Gold Futures | GC=F | Safe-haven asset |
| US Dollar Index | DX-Y.NYB | USD strength |
| Ethereum | ETH-USD | Within-crypto contagion |

Macro series are only available on exchange trading days; weekend/holiday gaps are forward-filled when merging with the Bitcoin calendar (365 days/year).

---

## Tech Stack

Python · statsmodels · prophet · xgboost · PyTorch · yfinance · pandas · ta · scipy · Plotly · matplotlib

---

## Author

**Fernando Cano Conde** — [LinkedIn](https://www.linkedin.com/in/fernando-cano-conde-b307052a2/) · [GitHub](https://github.com/fercanocd)

## License

MIT License
