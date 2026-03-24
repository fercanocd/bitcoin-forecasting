"""
Unified data ingestion: downloads BTC-USD OHLCV and macro data in one run.

Saves to data/raw/:
  btc_ohlcv.csv  -- BTC daily OHLCV (2018-01-01 to today)
  macro_raw.csv  -- SP500, Gold, DXY, ETH daily OHLCV

Usage:
  python -m src.data.ingest
"""
import yfinance as yf
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

MACRO_TICKERS = {
    "sp500": "^GSPC",
    "gold":  "GC=F",
    "dxy":   "DX-Y.NYB",
    "eth":   "ETH-USD",
}


def download_btc(start: str = "2018-01-01", end: str = None) -> pd.DataFrame:
    """Download daily BTC-USD OHLCV from Yahoo Finance."""
    ticker = yf.Ticker("BTC-USD")
    df = ticker.history(start=start, end=end, interval="1d", auto_adjust=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index.name = "Date"
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def download_macro(start: str, end: str = None) -> pd.DataFrame:
    """Download daily OHLCV for SP500, Gold, DXY and ETH-USD."""
    frames = {}
    for name, ticker_sym in MACRO_TICKERS.items():
        raw = yf.Ticker(ticker_sym).history(
            start=start, end=end, interval="1d", auto_adjust=True
        )
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in raw.columns:
                frames[f"{name}_{col.lower()}"] = raw[col]
    return pd.DataFrame(frames)


def download_all(start: str = "2018-01-01", end: str = None):
    """Download BTC daily and all macro assets. Returns (btc_df, macro_df)."""
    print(f"Downloading BTC-USD daily from {start}...")
    btc = download_btc(start=start, end=end)
    print(f"  -> {len(btc)} rows")

    print("Downloading macro data (SP500, Gold, DXY, ETH)...")
    macro = download_macro(start=start, end=end)
    print(f"  -> {len(macro)} rows, {len(macro.columns)} columns")

    return btc, macro


def save_raw(btc: pd.DataFrame, macro: pd.DataFrame):
    """Save all dataframes to data/raw/."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    btc_path = RAW_DIR / "btc_ohlcv.csv"
    btc.to_csv(btc_path)
    print(f"Saved BTC daily -> {btc_path}  ({len(btc)} rows)")

    macro_path = RAW_DIR / "macro_raw.csv"
    macro.to_csv(macro_path)
    print(f"Saved macro     -> {macro_path}  ({len(macro)} rows)")

    return btc_path, macro_path


if __name__ == "__main__":
    btc, macro = download_all(start="2018-01-01")
    save_raw(btc, macro)
    print("BTC daily tail:")
    print(btc.tail(3))
    print("Macro tail:")
    print(macro[["sp500_close", "gold_close", "dxy_close", "eth_close"]].tail(3))
