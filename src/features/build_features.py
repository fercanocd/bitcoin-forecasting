"""
Feature engineering for Bitcoin price forecasting.

Generates four dataset families in data/processed/:

  ARIMA/SARIMA/SARIMAX  ->  arima_features_daily.csv
  XGBoost               ->  xgboost_features_daily.csv
  LSTM                  ->  lstm_features_daily.csv
  Prophet               ->  prophet_features_daily.csv

Source data:
  data/raw/btc_ohlcv.csv  -- BTC daily OHLCV
  data/raw/macro_raw.csv  -- SP500, Gold, DXY, ETH (daily)

Usage:
  python -m src.features.build_features
"""
import pandas as pd
import numpy as np
import ta
from pathlib import Path

from src.config import HORIZONS

RAW_DIR       = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_btc(path: Path = None) -> pd.DataFrame:
    path = path or RAW_DIR / "btc_ohlcv.csv"
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def load_macro(path: Path = None) -> pd.DataFrame:
    """Load macro raw CSV with prices forward-filled on weekends/holidays.

    SP500, Gold and DXY trade only on business days; ETH trades 24/7. The
    raw CSV uses the union of all indices, leaving NaN on weekends for the
    closed markets. Forward-filling the prices preserves the last known
    market state, so any derived feature (log-returns, ratios, ...) computed
    on top inherits the correct interpretation: weekends produce zero
    returns (no new information), and Mondays carry the real Fri-to-Mon
    price change.
    """
    path = path or RAW_DIR / "macro_raw.csv"
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.ffill()
    return df


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_macro_log_returns() -> pd.DataFrame:
    """Log-returns of macro close prices: log(Pt / Pt-1).
    Preferred over simple returns for linear models (ARIMA/SARIMAX) due to
    symmetry and consistency with the BTC log-return target variable.
    """
    raw = load_macro()
    macro = pd.DataFrame(index=raw.index)
    for name in ["sp500", "gold", "dxy", "eth"]:
        col = f"{name}_close"
        if col in raw.columns:
            macro[f"{name}_log_return"] = np.log(raw[col] / raw[col].shift(1))
    return macro


def _join_macro(df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Left-join macro onto BTC df.

    Macro prices are already forward-filled in load_macro(), so the ffill
    here is a defensive fallback for any residual gaps that could appear
    after the join (e.g. dates present in the BTC index but absent from
    the macro index).
    """
    df = df.join(macro, how="left")
    df[macro.columns] = df[macro.columns].ffill()
    return df


def _drop_warmup(df: pd.DataFrame) -> pd.DataFrame:
    """Drop NaN rows from indicator warm-up.

    All columns starting with 'target_' are treated as forecasting targets
    and excluded from the NaN check: their NaNs at the end of the sample
    (where the future is unknown) are expected and must be preserved.
    """
    target_cols  = [c for c in df.columns if c.startswith("target_")]
    feature_cols = [c for c in df.columns if c not in target_cols]
    before = len(df)
    df = df.dropna(subset=feature_cols)
    print(f"  Dropped {before - len(df)} warm-up rows -> {len(df)} rows remaining")
    return df


def _save(df: pd.DataFrame, filename: str) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / filename
    df.to_csv(path)
    print(f"  Saved -> {path}  ({df.shape[0]} rows x {df.shape[1]} cols)")
    return path


# ---------------------------------------------------------------------------
# Dataset 1: ARIMA / SARIMA / SARIMAX
# ---------------------------------------------------------------------------

def build_arima(save: bool = True) -> pd.DataFrame:
    """Build minimal feature set for ARIMA/SARIMA/SARIMAX models.

    Endogenous series : log_return  -- log(Pt / Pt-1), stationary by construction.
    Exogenous regressors: log_volume_ratio, macro log-returns (SARIMAX only).
    Targets           : target_log_return_{1d, 7d, 30d}  -- h-step-ahead
                        cumulative log-returns ln(P_{t+h} / P_t).

    No technical indicators: AR/MA terms capture autocorrelation internally.
    No price levels: log-returns are already stationary; no differencing needed.
    """
    print("\n[ARIMA] Building daily features...")
    df = load_btc()
    print(f"  Loaded {len(df)} rows")

    c, v = df["Close"], df["Volume"]

    out = pd.DataFrame(index=df.index)
    out["log_return"]       = np.log(c / c.shift(1))
    volume_sma_20           = ta.trend.sma_indicator(v, window=20)
    out["log_volume_ratio"] = np.log(v / volume_sma_20)

    print("  Joining macro log-returns...")
    macro = _get_macro_log_returns()
    out = _join_macro(out, macro)

    # Targets — h-step-ahead cumulative log-returns ln(P_{t+h}/P_t).
    for h in HORIZONS:
        out[f"target_log_return_{h}d"] = np.log(c.shift(-h) / c)
    out = _drop_warmup(out)

    print(f"  Shape: {out.shape}  |  Columns: {list(out.columns)}")
    if save:
        _save(out, "arima_features_daily.csv")
    return out


# ---------------------------------------------------------------------------
# Dataset 2: XGBoost
# ---------------------------------------------------------------------------

def build_xgboost(save: bool = True) -> pd.DataFrame:
    """Build feature set for XGBoost using scale-invariant transformations.

    Trees are invariant to monotonic transformations but sensitive to the
    absolute scale of inputs when train and test span very different price
    regimes. The feature set therefore uses log close-to-MA ratios and a
    normalised MACD instead of raw prices and absolute moving averages,
    and macro variables enter only as log-returns (not as price levels).
    """
    print("\n[XGBoost] Building daily features...")
    df = load_btc()
    print(f"  Loaded {len(df)} rows")

    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    out = pd.DataFrame(index=df.index)

    # Return (1)
    out["log_return"] = np.log(c / c.shift(1))

    # Trend — log close-to-SMA (3) + log close-to-EMA (2) + MACD (1)
    # EMAs computed once and reused for both log-ratios and MACD.
    ema_12 = ta.trend.ema_indicator(c, window=12)
    ema_26 = ta.trend.ema_indicator(c, window=26)
    for n in [21, 50, 200]:
        out[f"log_close_sma_{n}"] = np.log(c / ta.trend.sma_indicator(c, window=n))
    out["log_close_ema_12"] = np.log(c / ema_12)
    out["log_close_ema_26"] = np.log(c / ema_26)
    out["macd"]             = (ema_12 - ema_26) / c

    # Momentum (1) — RSI bounded in [0, 1]
    out["rsi_14"] = ta.momentum.rsi(c, window=14) / 100

    # Volatility (2) — log of dispersion ratios for symmetric, scale-invariant
    # distribution. bb_width = 4*sigma_20/SMA_20 and atr_norm = ATR_14/P_t are
    # kept as intermediate variables only; the log-versions are the features.
    eps                 = 1e-8
    sma_20              = ta.trend.sma_indicator(c, window=20)
    std_20              = c.rolling(20).std()
    bb_width            = (4 * std_20) / sma_20
    atr_norm            = ta.volatility.average_true_range(h, l, c, window=14) / c
    out["log_bb_width"] = np.log(np.maximum(bb_width, eps))
    out["log_atr_norm"] = np.log(np.maximum(atr_norm, eps))

    # Volume (1)
    volume_sma_20           = ta.trend.sma_indicator(v, window=20)
    out["log_volume_ratio"] = np.log(v / volume_sma_20)

    # Lags — log-return lagged, capturing return autocorrelation directly
    for lag in [1, 2, 3, 5, 7, 14, 21]:
        out[f"log_return_lag_{lag}"] = out["log_return"].shift(lag)

    # Calendar — cyclic sin/cos encoding to preserve wrap-around structure
    out["dow_sin"]   = np.sin(2 * np.pi * df.index.dayofweek / 7)
    out["dow_cos"]   = np.cos(2 * np.pi * df.index.dayofweek / 7)
    out["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

    # Macro — log-returns only (raw closes dropped: non-stationary across windows)
    print("  Joining macro log-returns...")
    macro = _get_macro_log_returns()
    out = _join_macro(out, macro)

    # Targets — h-step-ahead cumulative log-returns ln(P_{t+h}/P_t).
    for h in HORIZONS:
        out[f"target_log_return_{h}d"] = np.log(c.shift(-h) / c)
    out = _drop_warmup(out)

    print(f"  Shape: {out.shape}  |  {out.shape[1]} features")
    if save:
        _save(out, "xgboost_features_daily.csv")
    return out


# ---------------------------------------------------------------------------
# Dataset 3: LSTM
# ---------------------------------------------------------------------------

def build_lstm(save: bool = True) -> pd.DataFrame:
    """Build pre-normalised feature set for LSTM models.

    Design principles:
    - No lags (replaced by the LSTM sequence window)
    - Trend as scale-invariant log close-to-MA ratios + normalised MACD
    - One representative per indicator family
    - Macro returns only, no absolute prices

    Trend representation matches the XGBoost feature set, so any
    performance difference between the two reflects model architecture
    rather than feature scaling.
    """
    print("\n[LSTM] Building daily features...")
    df = load_btc()
    print(f"  Loaded {len(df)} rows")

    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    out = pd.DataFrame(index=df.index)

    # Return (1)
    out["log_return"] = np.log(c / c.shift(1))

    # Trend — log close-to-SMA (3) + log close-to-EMA (2) + MACD (1)
    # EMAs computed once and reused for both log-ratios and MACD.
    ema_12 = ta.trend.ema_indicator(c, window=12)
    ema_26 = ta.trend.ema_indicator(c, window=26)
    for n in [21, 50, 200]:
        out[f"log_close_sma_{n}"] = np.log(c / ta.trend.sma_indicator(c, window=n))
    out["log_close_ema_12"] = np.log(c / ema_12)
    out["log_close_ema_26"] = np.log(c / ema_26)
    out["macd"]             = (ema_12 - ema_26) / c

    # Momentum (1) — RSI bounded in [0, 1]
    out["rsi_14"] = ta.momentum.rsi(c, window=14) / 100

    # Volatility (2) — log of dispersion ratios; bb_width and atr_norm kept
    # as intermediate variables only.
    eps                 = 1e-8
    sma_20              = ta.trend.sma_indicator(c, window=20)
    std_20              = c.rolling(20).std()
    bb_width            = (4 * std_20) / sma_20
    atr_norm            = ta.volatility.average_true_range(h, l, c, window=14) / c
    out["log_bb_width"] = np.log(np.maximum(bb_width, eps))
    out["log_atr_norm"] = np.log(np.maximum(atr_norm, eps))

    # Volume (1)
    volume_sma_20           = ta.trend.sma_indicator(v, window=20)
    out["log_volume_ratio"] = np.log(v / volume_sma_20)

    # Calendar — cyclic sin/cos encoding to preserve wrap-around structure
    out["dow_sin"]   = np.sin(2 * np.pi * df.index.dayofweek / 7)
    out["dow_cos"]   = np.cos(2 * np.pi * df.index.dayofweek / 7)
    out["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

    # Macro — log-returns only
    print("  Joining macro log-returns...")
    macro = _get_macro_log_returns()
    out = _join_macro(out, macro)

    # Targets — h-step-ahead cumulative log-returns ln(P_{t+h}/P_t).
    for h in HORIZONS:
        out[f"target_log_return_{h}d"] = np.log(c.shift(-h) / c)
    out = _drop_warmup(out)

    print(f"  Shape: {out.shape}  |  {out.shape[1]} features")
    if save:
        _save(out, "lstm_features_daily.csv")
    return out


# ---------------------------------------------------------------------------
# Dataset 4: Prophet
# ---------------------------------------------------------------------------

def build_prophet(save: bool = True) -> pd.DataFrame:
    """Build minimal feature set for Prophet (Meta).

    Endogenous series : log_return  -- log(Pt / Pt-1); fed to Prophet as `y`
                        after renaming Date -> ds in the model layer.
    Regressors        : log_volume_ratio + macro log-returns, attached via
                        Prophet's add_regressor (additive).
    Targets           : target_log_return_{1d, 7d, 30d}  -- h-step-ahead
                        cumulative log-returns ln(P_{t+h} / P_t), same as
                        the other families for direct metric comparability.

    Design choices (mirrors SARIMAX feature set):
      - No technical indicators: Prophet decomposes trend, seasonality and
        changepoints internally; SMAs/EMAs would duplicate that signal.
      - No price level: Prophet operates directly on the log-return series.
      - Same regressors as SARIMAX so any performance gap reflects model
        architecture (structural decomposition + changepoints) rather than
        feature access.
    """
    print("\n[Prophet] Building daily features...")
    df = load_btc()
    print(f"  Loaded {len(df)} rows")

    c, v = df["Close"], df["Volume"]

    out = pd.DataFrame(index=df.index)
    out["log_return"]       = np.log(c / c.shift(1))
    volume_sma_20           = ta.trend.sma_indicator(v, window=20)
    out["log_volume_ratio"] = np.log(v / volume_sma_20)

    print("  Joining macro log-returns...")
    macro = _get_macro_log_returns()
    out = _join_macro(out, macro)

    # Targets — h-step-ahead cumulative log-returns ln(P_{t+h}/P_t).
    for h in HORIZONS:
        out[f"target_log_return_{h}d"] = np.log(c.shift(-h) / c)
    out = _drop_warmup(out)

    print(f"  Shape: {out.shape}  |  Columns: {list(out.columns)}")
    if save:
        _save(out, "prophet_features_daily.csv")
    return out


# ---------------------------------------------------------------------------
# build_all
# ---------------------------------------------------------------------------

def build_all(save: bool = True) -> dict:
    """Run all four build pipelines and print a summary."""
    print(f"{'='*60}")
    print("Building all feature sets  |  freq=daily")
    print(f"{'='*60}")
    results = {
        "arima":   build_arima(save=save),
        "xgboost": build_xgboost(save=save),
        "lstm":    build_lstm(save=save),
        "prophet": build_prophet(save=save),
    }
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for name, df in results.items():
        print(f"  {name:8s}  {df.shape[0]:>5} rows  x  {df.shape[1]:>3} cols  |  {list(df.columns)}")
    return results


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build_all()
