"""
Exploratory Data Analysis plots for the Bitcoin Forecasting TFM.

Generates figures for the Materials and Methods section of the thesis.

Plots produced (saved to reports/figures/):
  01_btc_price_history.png       -- BTC price with bull/bear regimes
  02_asset_correlation.png       -- Correlation heatmap of log-returns
  03_return_distribution.png     -- BTC log-return distribution vs Normal
  04_price_vs_returns.png        -- Price level vs log-returns (stationarity)
  05_volume_ratio_vs_price.png   -- Log volume ratio vs price movements
  06_acf_pacf.png                -- ACF and PACF of BTC log-returns

Plots requiring trained models (generated in results_plots.py):
  07_predictions_vs_actual.png   -- Model forecasts vs realised returns
  08_xgboost_feature_importance  -- SHAP / gain importance
  09_lstm_training_curve.png     -- Train vs validation loss

Usage:
  python -m src.visualization.eda_plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[2]
RAW_DIR     = ROOT / "data" / "raw"
PROC_DIR    = ROOT / "data" / "processed"
FIG_DIR     = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
STYLE   = "seaborn-v0_8-whitegrid"
PALETTE = sns.color_palette("tab10")
DPI     = 150
FIGSIZE = (12, 5)

plt.rcParams.update({
    "font.family":  "serif",
    "font.size":    11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
})

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_btc() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "btc_ohlcv.csv", index_col="Date", parse_dates=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def _load_arima() -> pd.DataFrame:
    df = pd.read_csv(PROC_DIR / "arima_features_daily.csv", index_col="Date", parse_dates=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def _load_macro() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "macro_raw.csv", index_col="Date", parse_dates=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def _save(fig: plt.Figure, name: str):
    path = FIG_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"  Saved -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 1 — BTC price history with bull/bear regimes
# ---------------------------------------------------------------------------

REGIMES = [
    # (start, end, label, color, alpha)
    ("2018-01-01", "2018-12-15", "Bear",  "#d62728", 0.15),
    ("2020-10-01", "2021-11-10", "Bull",  "#2ca02c", 0.15),
    ("2022-01-01", "2022-11-21", "Bear",  "#d62728", 0.15),
    ("2023-01-01", "2024-03-01", "Bull",  "#2ca02c", 0.15),
    ("2024-04-01", "2024-09-30", "Bear",  "#d62728", 0.15),
    ("2024-10-01", "2026-01-20", "Bull",  "#2ca02c", 0.15),
]


def plot_btc_price_history():
    print("\n[1] BTC price history...")
    btc = _load_btc()

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(btc.index, btc["Close"], color=PALETTE[0], linewidth=1.2, label="BTC-USD Close")

    bear_patch = mpatches.Patch(color="#d62728", alpha=0.3, label="Bear market")
    bull_patch = mpatches.Patch(color="#2ca02c", alpha=0.3, label="Bull market")

    for start, end, label, color, alpha in REGIMES:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   color=color, alpha=alpha)

    ax.set_title("Bitcoin Daily Closing Price (2018–present)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(handles=[ax.lines[0], bull_patch, bear_patch])
    fig.tight_layout()
    _save(fig, "01_btc_price_history.png")


# ---------------------------------------------------------------------------
# Plot 2 — Asset correlation heatmap (log-returns)
# ---------------------------------------------------------------------------

def plot_asset_correlation():
    print("\n[2] Asset correlation heatmap...")
    arima = _load_arima()

    # BTC log_return is called log_return in arima csv
    rename = {
        "log_return":       "BTC",
        "sp500_log_return": "S&P 500",
        "gold_log_return":  "Gold",
        "dxy_log_return":   "DXY",
        "eth_log_return":   "ETH",
    }
    sub = arima[[c for c in rename if c in arima.columns]].rename(columns=rename)
    corr = sub.corr()

    fig, ax = plt.subplots(figsize=(7, 5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
        vmin=-1, vmax=1, center=0,
        linewidths=0.5, ax=ax,
        annot_kws={"size": 11}
    )
    ax.set_title("Pearson Correlation of Daily Log-Returns (2018–present)")
    fig.tight_layout()
    _save(fig, "02_asset_correlation.png")


# ---------------------------------------------------------------------------
# Plot 3 — BTC log-return distribution vs Normal
# ---------------------------------------------------------------------------

def plot_return_distribution():
    print("\n[3] Return distribution...")
    arima = _load_arima()
    r = arima["log_return"].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left — histogram + KDE + Normal fit
    ax = axes[0]
    ax.hist(r, bins=120, density=True, color=PALETTE[0], alpha=0.5, label="Empirical")
    r_sorted = np.linspace(r.min(), r.max(), 300)
    mu, sigma = r.mean(), r.std()
    ax.plot(r_sorted, stats.norm.pdf(r_sorted, mu, sigma),
            color="#d62728", linewidth=2, label=f"Normal($\\mu$={mu:.4f}, $\\sigma$={sigma:.4f})")
    kde_x = np.linspace(r.min(), r.max(), 300)
    kde   = stats.gaussian_kde(r)
    ax.plot(kde_x, kde(kde_x), color=PALETTE[1], linewidth=2, linestyle="--", label="KDE")
    ax.set_title("Distribution of BTC Daily Log-Returns")
    ax.set_xlabel("Log-return")
    ax.set_ylabel("Density")
    ax.legend()

    # Right — Q-Q plot
    ax2 = axes[1]
    (osm, osr), (slope, intercept, _) = stats.probplot(r, dist="norm")
    ax2.scatter(osm, osr, s=4, alpha=0.4, color=PALETTE[0])
    line_x = np.array([osm[0], osm[-1]])
    ax2.plot(line_x, slope * line_x + intercept, color="#d62728", linewidth=2, label="Normal line")
    ax2.set_title("Q-Q Plot vs Normal Distribution")
    ax2.set_xlabel("Theoretical quantiles")
    ax2.set_ylabel("Sample quantiles")
    ax2.legend()

    # Annotations
    kurt  = r.kurtosis()
    skew  = r.skew()
    _, pval = stats.shapiro(r.sample(min(5000, len(r)), random_state=42))
    axes[0].text(0.97, 0.95,
                 f"Skewness: {skew:.3f}\nExcess kurtosis: {kurt:.3f}\nShapiro p: {pval:.2e}",
                 transform=axes[0].transAxes, ha="right", va="top",
                 fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    fig.tight_layout()
    _save(fig, "03_return_distribution.png")


# ---------------------------------------------------------------------------
# Plot 4 — Price level vs log-returns (stationarity)
# ---------------------------------------------------------------------------

def plot_price_vs_returns():
    print("\n[4] Price vs log-returns (stationarity)...")
    btc   = _load_btc()
    arima = _load_arima()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax1.plot(btc.index, btc["Close"], color=PALETTE[0], linewidth=1)
    ax1.set_title("Bitcoin Closing Price — Non-Stationary")
    ax1.set_ylabel("Price (USD)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    r = arima["log_return"]
    ax2.plot(r.index, r, color=PALETTE[1], linewidth=0.7, alpha=0.9)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_title("BTC Daily Log-Return — Approximately Stationary")
    ax2.set_ylabel("Log-return")
    ax2.set_xlabel("Date")
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    _save(fig, "04_price_vs_returns.png")


# ---------------------------------------------------------------------------
# Plot 5 — Log volume ratio vs price
# ---------------------------------------------------------------------------

def plot_volume_ratio_vs_price():
    print("\n[5] Log volume ratio vs price...")
    btc   = _load_btc()
    arima = _load_arima()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax1.plot(btc.index, btc["Close"], color=PALETTE[0], linewidth=1)
    ax1.set_title("Bitcoin Closing Price")
    ax1.set_ylabel("Price (USD)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    lvr = arima["log_volume_ratio"]
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in lvr]
    ax2.bar(lvr.index, lvr, color=colors, width=1, alpha=0.7)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")

    # Threshold lines
    threshold = 1.0
    ax2.axhline( threshold, color="#d62728", linewidth=1, linestyle=":", alpha=0.8,
                 label=f"+{threshold} (abnormally high volume)")
    ax2.axhline(-threshold, color="#d62728", linewidth=1, linestyle=":", alpha=0.8)

    ax2.set_title("Log Volume Ratio — Abnormal Trading Activity")
    ax2.set_ylabel(r"$\ln(V_t\,/\,\overline{V}_{t,20})$")
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    _save(fig, "05_volume_ratio_vs_price.png")


# ---------------------------------------------------------------------------
# Plot 6 — ACF and PACF of BTC log-returns
# ---------------------------------------------------------------------------

def plot_acf_pacf():
    print("\n[6] ACF / PACF...")
    arima = _load_arima()
    r = arima["log_return"].dropna()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    plot_acf(r,  lags=40, ax=ax1, color=PALETTE[0], vlines_kwargs={"colors": PALETTE[0]})
    plot_pacf(r, lags=40, ax=ax2, color=PALETTE[1], vlines_kwargs={"colors": PALETTE[1]},
              method="ywm")

    ax1.set_title("Autocorrelation Function (ACF) — BTC Log-Returns")
    ax1.set_xlabel("Lag (days)")
    ax1.set_ylabel("Autocorrelation")

    ax2.set_title("Partial Autocorrelation Function (PACF) — BTC Log-Returns")
    ax2.set_xlabel("Lag (days)")
    ax2.set_ylabel("Partial autocorrelation")

    fig.tight_layout()
    _save(fig, "06_acf_pacf.png")


# ---------------------------------------------------------------------------
# Run all EDA plots
# ---------------------------------------------------------------------------

def plot_all_eda():
    print("=" * 60)
    print("Generating EDA plots  ->  reports/figures/")
    print("=" * 60)
    with plt.style.context(STYLE):
        plot_btc_price_history()
        plot_asset_correlation()
        plot_return_distribution()
        plot_price_vs_returns()
        plot_volume_ratio_vs_price()
        plot_acf_pacf()
    print("\nAll EDA plots saved.")
    print("Plots 07-09 (model results) will be generated after training.")


if __name__ == "__main__":
    plot_all_eda()
