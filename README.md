# Bitcoin Forecasting with Machine Learning Techniques



[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Comparative study of three model families for **one-step-ahead Bitcoin log-return forecasting**:

| Model | Type | Features |
|---|---|---|
| **SARIMAX** | Classical econometrics | Log-return, log volume ratio, macro log-returns |
| **XGBoost** | Gradient boosting | 31 features: technical indicators + macro |
| **LSTM** | Deep learning | 19 features: normalised ratios + macro |

All models share the same prediction target (next-day log-return r_{t+1}) and are evaluated under identical walk-forward validation conditions. Macroeconomic context variables — S&P 500, Gold, US Dollar Index, Ethereum — are included as exogenous inputs based on documented co-movement with Bitcoin prices (Bouri et al., 2017).

---

## Project Structure

```
bitcoin-forecasting/
├── data/                           # gitignored — regenerate with: python pipeline.py
│   ├── raw/                        # btc_ohlcv.csv, macro_raw.csv (Yahoo Finance)
│   └── processed/                  # arima / xgboost / lstm feature CSVs
├── reports/                        # gitignored — regenerate with eda_plots.py
│   └── figures/                    # EDA plots + interactive candlestick chart
├── src/
│   ├── data/
│   │   └── ingest.py               # Download via yfinance
│   ├── features/
│   │   └── build_features.py       # Feature engineering (ARIMA / XGBoost / LSTM)
│   ├── models/
│   │   ├── sarimax.py              # Walk-forward SARIMAX — statsmodels (WIP)
│   │   ├── xgboost_model.py        # Walk-forward XGBoost (WIP)
│   │   └── lstm_model.py           # Walk-forward LSTM — PyTorch (WIP)
│   ├── evaluation/                 # Walk-forward metrics: RMSE, MAE, DA (WIP)
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
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```



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

## Feature Engineering

Three feature sets are constructed, each tailored to its model family:

- **SARIMAX** (6 features): log-return, log volume ratio, macro log-returns
- **XGBoost** (31 features): price, SMAs, EMAs, RSI-14 [0,1], Bollinger width, ATR-14, log volume ratio, lagged returns, calendar sin/cos, macro
- **LSTM** (19 features): normalised price ratios (P/SMA, P/EMA), RSI-14 [0,1], Bollinger width, ATR/P, log volume ratio, calendar, macro log-returns

---

## Tech Stack

Python · statsmodels · pmdarima · xgboost · PyTorch · yfinance · pandas · ta · Plotly · matplotlib

---

## Author

**Fernando Cano Conde** — [LinkedIn](https://www.linkedin.com/in/fernando-cano-conde-b307052a2/) · [GitHub](https://github.com/fercanocd)

## License

MIT License
