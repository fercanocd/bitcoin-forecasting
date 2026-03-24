"""
Feature engineering for Bitcoin price forecasting.

Generates three dataset families in data/processed/:

  ARIMA/SARIMA/SARIMAX  ->  arima_features_daily.csv
  XGBoost               ->  xgboost_features_daily.csv
  LSTM                  ->  lstm_features_daily.csv

Source data:
  data/raw/btc_ohlcv.csv  -- BTC daily OHLCV
  data/raw/macro_raw.csv  -- SP500, Gold, DXY, ETH (daily)

Usage:
  python -m src.features.build_features
"""
import pandas as pd
import numpy as np
import ta
import pathlib
from pathlib import Path

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


def load_macro(path: pathlib.Path = None) -> pd.DataFrame:
    path = path or RAW_DIR / "macro_raw.csv"
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
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


def _get_macro_full() -> pd.DataFrame:
    raw = load_macro()
    macro = pd.DataFrame(index=raw.index)
    for name in ["sp500", "gold", "dxy", "eth"]:
        col = f"{name}_close"
        if col in raw.columns:
            macro[col]                  = raw[col]
            macro[f"{name}_log_return"] = np.log(raw[col] / raw[col].shift(1))
    return macro


def _join_macro(df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Left-join macro onto BTC df, forward-filling weekends/holidays."""
    df = df.join(macro, how="left")
    df[macro.columns] = df[macro.columns].ffill()
    return df


def _drop_warmup(df: pd.DataFrame, target_col: str = "target_return") -> pd.DataFrame:
    """Drop NaN rows from indicator warm-up, preserving target_col NaN on last row."""
    feature_cols = [c for c in df.columns if c != target_col]
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
    Target            : target_return = log_return.shift(-1)  -- next-day log-return.

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

    out["target_return"] = out["log_return"].shift(-1)
    out = _drop_warmup(out)

    print(f"  Shape: {out.shape}  |  Columns: {list(out.columns)}")
    if save:
        _save(out, "arima_features_daily.csv")
    return out


# ---------------------------------------------------------------------------
# Dataset 2: XGBoost
# ---------------------------------------------------------------------------

def build_xgboost(save: bool = True) -> pd.DataFrame:
    """Build full feature set for XGBoost (no normalisation needed for trees).

    Includes price, trend, momentum, volatility, volume, lags, calendar,
    macro and target_return.
    """
    print("\n[XGBoost] Building daily features...")
    df = load_btc()
    print(f"  Loaded {len(df)} rows")

    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    out = pd.DataFrame(index=df.index)

    # Price
    out["Close"]      = c
    out["log_return"] = np.log(c / c.shift(1))

    # Trend (6)
    out["sma_7"]   = ta.trend.sma_indicator(c, window=7)
    out["sma_21"]  = ta.trend.sma_indicator(c, window=21)
    out["sma_50"]  = ta.trend.sma_indicator(c, window=50)
    out["sma_200"] = ta.trend.sma_indicator(c, window=200)
    out["ema_12"]  = ta.trend.ema_indicator(c, window=12)
    out["ema_26"]  = ta.trend.ema_indicator(c, window=26)

    # Momentum (1) — RSI normalised to [0, 1]
    out["rsi_14"] = ta.momentum.rsi(c, window=14) / 100

    # Volatility (2) — bb_width computed directly; band levels removed (co-move with SMAs)
    sma_20          = ta.trend.sma_indicator(c, window=20)
    std_20          = c.rolling(20).std()
    out["bb_width"] = (4 * std_20) / sma_20
    out["atr_14"]   = ta.volatility.average_true_range(h, l, c, window=14)

    # Volume (1) — OBV removed (numerically unstable ratio)
    volume_sma_20           = ta.trend.sma_indicator(v, window=20)
    out["log_volume_ratio"] = np.log(v / volume_sma_20)

    # Lags — log-return lagged, capturing return autocorrelation directly
    for lag in [1, 2, 3, 5, 7, 14, 21]:
        out[f"return_lag_{lag}"] = out["log_return"].shift(lag)

    # Calendar — cyclic sin/cos encoding to preserve wrap-around structure
    out["dow_sin"]   = np.sin(2 * np.pi * df.index.dayofweek / 7)
    out["dow_cos"]   = np.cos(2 * np.pi * df.index.dayofweek / 7)
    out["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

    # Macro — close prices + returns
    print("  Joining macro data...")
    macro = _get_macro_full()
    out = _join_macro(out, macro)

    out["target_return"] = out["log_return"].shift(-1)
    out = _drop_warmup(out)

    print(f"  Shape: {out.shape}  |  {out.shape[1]} features")
    if save:
        _save(out, "xgboost_features_daily.csv")
    return out


# ---------------------------------------------------------------------------
# Dataset 3: LSTM
# ---------------------------------------------------------------------------

def build_lstm(save: bool = True) -> pd.DataFrame:
    """Build curated, pre-normalised feature set for LSTM models.

    Design principles:
    - No lags (replaced by the LSTM sequence window)
    - Trend as scale-invariant ratios (Close / SMA)
    - Calendar normalised to [0, 1]
    - One representative per indicator family
    - Macro returns only, no absolute prices
    """
    print("\n[LSTM] Building daily features...")
    df = load_btc()
    print(f"  Loaded {len(df)} rows")

    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    out = pd.DataFrame(index=df.index)

    # Return (1)
    out["log_return"] = np.log(c / c.shift(1))

    # Trend — scale-invariant Close/SMA ratios (4) and Close/EMA ratios (2)
    for n in [7, 21, 50, 200]:
        out[f"close_to_sma{n}"] = c / ta.trend.sma_indicator(c, window=n)
    for n in [12, 26]:
        out[f"close_to_ema{n}"] = c / ta.trend.ema_indicator(c, window=n)

    # Momentum (1) — RSI normalised to [0, 1]
    out["rsi_14"] = ta.momentum.rsi(c, window=14) / 100

    # Volatility (2)
    sma_20          = ta.trend.sma_indicator(c, window=20)
    std_20          = c.rolling(20).std()
    out["bb_width"] = (4 * std_20) / sma_20
    out["atr_norm"] = ta.volatility.average_true_range(h, l, c, window=14) / c

    # Volume (1) — OBV removed (numerically unstable ratio)
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

    out["target_return"] = out["log_return"].shift(-1)
    out = _drop_warmup(out)

    print(f"  Shape: {out.shape}  |  {out.shape[1]} features")
    if save:
        _save(out, "lstm_features_daily.csv")
    return out


# ---------------------------------------------------------------------------
# build_all
# ---------------------------------------------------------------------------

def build_all(save: bool = True) -> dict:
    """Run all three build pipelines and print a summary."""
    print(f"{'='*60}")
    print("Building all feature sets  |  freq=daily")
    print(f"{'='*60}")
    results = {
        "arima":   build_arima(save=save),
        "xgboost": build_xgboost(save=save),
        "lstm":    build_lstm(save=save),
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
