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

The comparison is regenerated from the saved predictions every run and written to
disk, so the reported numbers never drift from the code:

- **`reports/metrics/comparison_master.md`** — Validation (CV) vs Test, per model
  and horizon, with the `predict-drift` baseline, wall-clock training time
  (`time_val` / `time_test`) and a `beats_drift` flag.
- **`reports/metrics/diebold_mariano.md`** — pairwise Diebold-Mariano tests on the
  test set (which RMSE differences are statistically real).
- **`reports/metrics/ensemble_*.md`** — forecast-combination results: the weights,
  the out-of-sample scorecard, and the validation error covariance/correlation
  that explains them (see *Forecast combination* below).
- **`reports/metrics/*.csv`** — the same tables in machine-readable form.
- **`reports/predictions/runtimes.csv`** — persistent log of every model's
  training time, written incrementally: re-running one model updates only its
  own row.
- **`reports/figures/07…16_*.png`** — the result figures (see below).

Rebuild them at any time without re-training:

```bash
python pipeline.py --only-report      # tables + figures from whatever predictions exist
```

**Baseline findings (SARIMAX, Prophet).** Neither classical baseline beats the
`predict-drift` reference in RMSE. SARIMAX posts only a marginal positive
directional edge at h=1 (attributable to ETH co-movement); all longer-horizon
edges are ≈ 0. Prophet degrades relative to SARIMAX as the horizon grows. These
set the bar the ML models (XGBoost, LSTM) must clear to add value.

### EDA figures (`reports/figures/`)

Input-data analysis, independent of the models (regenerate with `--eda`).

| # | Figure | Reads |
|---|---|---|
| 01 | BTC price history with bull/bear regimes | input data |
| 02 | Asset correlation heatmap of log-returns | input data |
| 03 | BTC log-return distribution vs Normal | input data |
| 04 | Price level vs log-returns (stationarity) | input data |
| 05 | Log volume ratio vs price movements | input data |
| 06 | ACF and PACF of BTC log-returns | input data |

### Result figures (`reports/figures/`)

| # | Figure | Reads |
|---|---|---|
| 07 | RMSE per model/horizon vs the `predict-drift` / `predict-zero` baselines | test |
| 08 | Directional accuracy vs the `always-up` baseline (edge) | test |
| 09 | Forecast vs realised return over the test period (h=1 grid) | test |
| 10 | Sign-of-forecast trading strategy equity vs buy-and-hold (h=1) | test |
| 11 | Predicted vs realised scatter — the shrinkage that explains RMSE ≈ baseline | test |
| 12 | Validation vs test RMSE — generalisation / overfitting check | CV + test |
| 13 | XGBoost gain importance, top features per horizon | train fit |
| 14 | Diebold-Mariano pairwise significance heatmap | test |
| 15 | Ensemble test RMSE by weighting scheme vs best-single / drift | CV + test |
| 16 | Validation error-correlation heatmap (diversification diagnostic) | CV |

### Forecast combination (ensemble)

As a final robustness check, the four model forecasts are combined — portfolio
theory applied to forecasts instead of assets — to ask whether *any* weighting
beats the best single model. Three schemes are fitted **on the validation errors
only** (the test set is never used to choose a weight) and scored out-of-sample:

- **equal** — `1/N`;
- **inverse-RMSE** — weight by individual accuracy;
- **minimum-variance** — long-only `argmin w'Σw`, the Bates-Granger combination
  (the direct analogue of a minimum-variance portfolio).

The finding reinforces the main result rather than overturning it: the models'
errors are correlated at **0.85–0.98**, so there is almost no idiosyncratic
error to diversify away. Minimum-variance collapses to equal weights at short
horizons (near-identical, near-perfectly-correlated forecasts) and concentrates
on SARIMAX at h=30. In an **augmented pool** where the `predict-drift` baseline
is allowed to compete, the optimum keeps drift as ~half the allocation
(54% SARIMAX / 46% drift at h=30) — the cleanest possible statement that the
trained models carry no exploitable signal beyond the naive constant. Written to
`reports/metrics/ensemble_*` and figures 15–16.

---

## Methodology

### Experimental design

The experiment uses an **expanding walk-forward** evaluation scheme. At each test origin `t` the model is fitted exclusively on data up to `t−1` and then forecasts the next `h` days. The fit window grows by one observation each day — no future data ever enters the fit. This mirrors realistic deployment: you forecast with what you know today.

**What "walk-forward on the test" means (and why it is not leakage).** The test is
not one train-then-evaluate split; it is 883 successive *refit-and-forecast* steps.
At each origin the parameters are re-estimated on **all data up to the previous day**
and used to forecast the next `h` days — so the fit window keeps **expanding into the
test period** as the walk proceeds. This is not cheating: to forecast day `t` only
data strictly before `t` is used, so every forecast is genuinely out-of-sample at the
moment it is made. The end of the 2018–2023 window is *not* where fitting stops — it
is only where **hyperparameter selection** stops (Level 1). During the walk (Level 2)
the hyperparameters stay frozen while the **parameters** refit on the growing window.

**Round 1 (before the test): expanding-window cross-validation.** The
hyperparameters are chosen entirely within the training era using expanding
folds. Each fold trains on the past and validates on the *single next year*;
the validation window slides forward one year per fold while the training
window expands behind it. The four folds tile 2020–2023 exactly once each
(equal weight, no overlap), and none ever crosses into the 2024–2026 test:

```
         2018  2019  2020  2021  2022  2023   | 2024  2025  26
overview TRAIN + VALIDATION (2018-2023)        | [==== test (fixed) ====]
fold 1   ============######                    | [==== test (fixed) ====]
fold 2   ==================######              | [==== test (fixed) ====]
fold 3   ========================######        | [==== test (fixed) ====]
fold 4   ==============================######   | [==== test (fixed) ====]

  =  train (expands each fold)     #  validation (1 year, slides right)
  gap between # and | = years not used in that fold
  test = fixed block 2024-2026, identical and aligned in every row
```

Validation reaches right up to the test wall but never crosses it: the last
fold validates 2023, the year immediately before the test. Validating 2024
would mean using test data to choose hyperparameters — the leakage this
split is designed to prevent. (2018–2019 are never a validation year: the
first fold needs at least two years of history to train on.) Why not let each
fold validate *everything* up to the test? Because that would weight the
later years far more heavily (2023 would appear in every fold, 2020 in only
one) and force early folds to forecast years ahead on a stale, tiny training
set — neither balanced nor realistic.

### Target variable

The prediction target at each origin is the **cumulative log-return over the next h days**:

```
target_h = ln(P_{t+h} / P_t) = Σ ln(P_{t+k} / P_{t+k−1})   for k = 1..h
```

Log-returns are additive and approximately stationary, which makes them suitable for both classical and ML models without differencing or scaling.

### Recursive vs direct strategy

- **Recursive (SARIMAX, Prophet):** a single daily model is fitted; the h-day forecast is obtained by iterating the model h times. The model is identical across horizons — only the number of projected steps differs. This means one walk-forward pass serves all three horizons simultaneously.
- **Direct (XGBoost, LSTM):** a separate model is trained per horizon, optimising directly against the h-day target. Models are genuinely different across horizons (the optimal features and weights for 1-day ahead differ from those for 30-day ahead), so three independent walk-forward passes are required.

**Date-label convention.** The two families label a forecast by different dates: a recursive-model row dated `d` holds the return realised *on* `d` (forecast from information up to `d−1`), whereas a direct-model row dated `d` holds `ln(P_{d+h}/P_d)`, the return realised *over* `(d, d+h]` (forecast from information up to `d`). Both solve the identical task — predict the next h-day return from everything known at the current close — so each model's own aggregate metrics are directly comparable. For the pairwise Diebold-Mariano tests and the overlay figures, `src/evaluation/compare.py` re-keys every series onto the common **origin-close date** (recursive: `d−1`; direct: `d`) so the paired `y_true` values match by construction.

### Two-level training scheme

Model training operates at two distinct levels that never mix:

**Level 1 — Hyperparameter selection (once, before the test period)**

Determines the model configuration using only training data. Never touches the test set.

| Model | What is decided | How |
|---|---|---|
| SARIMAX | Order `(p,d,q)`, which regressors to keep | AIC minimisation + p-value backward elimination on train window |
| Prophet | `growth='flat'`, seasonality mode | Design choice (log-returns are ~zero-mean, piecewise trend would chase noise) |
| XGBoost | `max_depth`, `learning_rate`, `n_estimators`, regularisation, ... | Time-series CV within train (expanding annual folds: train 2018–19 → val 2020, train 2018–20 → val 2021, ...) |
| LSTM | Hidden size, layers, dropout, learning rate, batch size, ... | Same time-series CV as XGBoost |

**Level 2 — Parameter refit during the test walk-forward**

With hyperparameters fixed, parameters (coefficients, tree weights, neural network weights) are re-estimated over the test period on an expanding window. The refit cadence differs by model family for one reason only — the marginal benefit of a one-day-newer fit relative to its compute cost:

| Family | Refit cadence | Rationale |
|---|---|---|
| SARIMAX, Prophet | **Daily** | Each fit is cheap (few coefficients, closed / near-closed form) and, since the h-day forecast is obtained by iterating the daily model h times, a stale fit propagates its error through every horizon. |
| XGBoost, LSTM | **Monthly** | Each fit is expensive (up to 1 000 boosted trees or many epochs of backprop over 60-step sequences). Trees and network weights are stable to a one-day extension of the training set, so daily refits would multiply compute cost 30× for a negligible reduction in error. |

Between refits, the frozen model is used for daily prediction (it is only the parameter *re-estimation* that pauses, not the forecasting). Both cadences fit an *expanding* window whose start is fixed at 2018-01-21.

```
Test walk-forward (hyperparameters frozen from Level 1)
├── Origin 2024-01-01  → fit + predict 1/7/30d                (all models refit here)
├── Origin 2024-01-02  → predict 1/7/30d                       (recursive: refit; ML: reuse)
├── ...
├── Origin 2024-02-01  → predict 1/7/30d                       (all models refit here)
├── ...
└── Origin 2026-06-01  → predict 1/7/30d
```

The two levels are strictly separated: hyperparameters never see the test set; parameter refits never change the hyperparameters.

### Train / test split

| Set | Period | Observations | Role |
|---|---|---|---|
| **Train** | 2018-01-21 → 2023-12-31 | 2,171 days | Level 1: hyperparameter selection |
| **Test** | 2024-01-01 → 2026-06-01 | 883 days | Level 2: parameter refit + out-of-sample evaluation |

The train/test cut is set at end-2023 so the test period includes the April 2024 halving — a structural market event the model has never seen, making the evaluation genuinely out-of-sample.

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
├── reports/                        # gitignored — regenerate with: python pipeline.py --only-report
│   ├── figures/                    # EDA plots (01-06) + result figures (07-16)
│   ├── metrics/                    # comparison + Diebold-Mariano + ensemble tables (CSV + markdown)
│   └── predictions/                # walk-forward predictions: {model}_{h}d.csv (+ cv/, runtimes.csv)
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
│   │   ├── walk_forward.py         # Expanding walk-forward engine (recursive + direct)
│   │   ├── cv.py                   # Round-1 CV folds + aggregation
│   │   ├── runtime.py              # Per-model training-time log (reports/predictions/runtimes.csv)
│   │   ├── compare.py              # Four-model comparison tables + DM (reports/metrics/)
│   │   └── ensemble.py             # Forecast combination: equal / inv-RMSE / min-variance
│   └── visualization/
│       ├── style.py                # Shared figure style (EDA + results)
│       ├── eda_plots.py            # Static EDA figures (01-06)
│       └── results_plots.py        # Result figures (07-16) from saved predictions
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

**Run the full pipeline** (ingest → features → all models → report):

```bash
python pipeline.py                    # everything
python pipeline.py --skip-ingest      # reuse existing raw CSVs
python pipeline.py --cv               # also run Round-1 hyperparameter CV
python pipeline.py --only-report      # rebuild tables + figures only (no training)
python pipeline.py --eda              # also (re)generate the EDA figures
python pipeline.py --quiet            # suppress per-refit / per-combo progress lines
```

**Fast sanity checks** — end-to-end but abbreviated (seconds instead of hours):

```bash
python pipeline.py --skip-ingest --skip-features --smoke 20        # ~1 min (all 4 models + report, 20 test days)
python pipeline.py --skip-ingest --skip-features --smoke 30 --cv   # ~5-10 min (also CV, 30 val days per fold)
```

**Run individual models** (all write `reports/predictions/{model}_{h}d.csv`):

```bash
python -m src.models.sarimax                        # all horizons
python -m src.models.sarimax --horizon 1            # single horizon
python -m src.models.sarimax --refit monthly        # fast dev run (monthly refit)
python -m src.models.prophet_model
python -m src.models.xgboost_model                  # monthly refit; --refit step for daily
python -m src.models.xgboost_model --cv             # Round-1 grid search
python -m src.models.lstm_model
```

**Build the comparison and figures** (from whatever predictions exist on disk):

```bash
python -m src.evaluation.compare                    # comparison tables + DM tests
python -m src.evaluation.ensemble                   # forecast-combination tables
python -m src.visualization.results_plots           # result figures 07-16
python -m src.visualization.results_plots --only 07 # a single figure
python -m src.visualization.eda_plots               # EDA figures 01-06
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

Python · statsmodels · prophet · xgboost · PyTorch · yfinance · pandas · ta · scipy · matplotlib · seaborn

---

## Author

**Fernando Cano Conde** — [LinkedIn](https://www.linkedin.com/in/fernando-cano-conde-b307052a2/) · [GitHub](https://github.com/fercanocd)

## License

MIT License
