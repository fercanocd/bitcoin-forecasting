"""
Generate a fully self-contained interactive candlestick HTML chart.

Assets: BTC, ETH, SP500, Gold, DXY
Periods: 1W, 1M, 3M, 6M, YTD, 1Y, 3Y, All

Output: reports/figures/candlestick_interactive.html

Usage:
    python -m src.visualization.candlestick_interactive
"""
import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "reports" / "figures"


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

def load_assets() -> dict[str, pd.DataFrame]:
    btc = pd.read_csv(RAW_DIR / "btc_ohlcv.csv", parse_dates=["Date"], index_col="Date")
    btc.columns = [c.capitalize() for c in btc.columns]

    macro = pd.read_csv(RAW_DIR / "macro_raw.csv", parse_dates=["Date"], index_col="Date")

    assets = {"BTC": btc}
    prefix_map = {
        "ETH":   "eth",
        "SP500": "sp500",
        "Gold":  "gold",
        "DXY":   "dxy",
    }
    for name, prefix in prefix_map.items():
        cols = {
            f"{prefix}_open":   "Open",
            f"{prefix}_high":   "High",
            f"{prefix}_low":    "Low",
            f"{prefix}_close":  "Close",
            f"{prefix}_volume": "Volume",
        }
        available = {k: v for k, v in cols.items() if k in macro.columns}
        df = macro[[*available.keys()]].rename(columns=available)
        df = df.dropna(subset=["Open", "High", "Low", "Close"], how="all")
        assets[name] = df

    return assets


def df_to_json(df: pd.DataFrame) -> str:
    """Serialise OHLCV dataframe to a compact JSON object."""
    df = df.copy().sort_index()
    df.index = df.index.strftime("%Y-%m-%d")
    record = {
        "date":   df.index.tolist(),
        "open":   df["Open"].round(4).tolist(),
        "high":   df["High"].round(4).tolist(),
        "low":    df["Low"].round(4).tolist(),
        "close":  df["Close"].round(4).tolist(),
        "volume": df["Volume"].round(0).tolist() if "Volume" in df.columns else [],
    }
    return json.dumps(record)


# ---------------------------------------------------------------------------
# 2. HTML template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Candlestick Chart</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d1117; color: #e6edf3; font-family: 'Segoe UI', sans-serif; }}
  h1 {{ text-align: center; padding: 18px 0 6px; font-size: 1.3rem; color: #58a6ff; letter-spacing: .5px; }}

  #controls {{
    display: flex; flex-wrap: wrap; justify-content: center;
    gap: 10px; padding: 10px 16px 6px;
  }}

  .btn-group {{ display: flex; gap: 4px; align-items: center; }}
  .btn-group label {{ font-size: .75rem; color: #8b949e; margin-right: 4px; text-transform: uppercase; letter-spacing: .5px; }}

  button, select {{
    background: #161b22; color: #c9d1d9; border: 1px solid #30363d;
    border-radius: 6px; padding: 5px 12px; font-size: .82rem;
    cursor: pointer; transition: background .15s, border-color .15s;
  }}
  button:hover, select:hover {{ background: #21262d; border-color: #58a6ff; }}
  button.active {{ background: #1f6feb; border-color: #58a6ff; color: #fff; }}

  select {{ padding: 5px 8px; }}
  option {{ background: #161b22; }}

  #chart {{ width: 100%; height: calc(100vh - 120px); }}
</style>
</head>
<body>
<h1>Candlestick Chart — Bitcoin &amp; Macro Assets</h1>

<div id="controls">
  <div class="btn-group">
    <label>Asset</label>
    <select id="assetSelect">{asset_options}</select>
  </div>
  <div class="btn-group" id="periodBtns">
    <label>Period</label>
    <button data-p="7D">1W</button>
    <button data-p="1M">1M</button>
    <button data-p="3M">3M</button>
    <button data-p="6M">6M</button>
    <button data-p="YTD">YTD</button>
    <button data-p="1Y">1Y</button>
    <button data-p="3Y">3Y</button>
    <button data-p="ALL" class="active">All</button>
  </div>
</div>

<div id="chart"></div>

<script>
const DATA = {data_json};

// ---- helpers ---------------------------------------------------------------
function filterByPeriod(dates, period) {{
  if (period === "ALL") return [0, dates.length - 1];
  const last = new Date(dates[dates.length - 1]);
  let from = new Date(last);
  if      (period === "7D")  from.setDate(last.getDate() - 7);
  else if (period === "1M")  from.setMonth(last.getMonth() - 1);
  else if (period === "3M")  from.setMonth(last.getMonth() - 3);
  else if (period === "6M")  from.setMonth(last.getMonth() - 6);
  else if (period === "1Y")  from.setFullYear(last.getFullYear() - 1);
  else if (period === "3Y")  from.setFullYear(last.getFullYear() - 3);
  else if (period === "YTD") from = new Date(last.getFullYear(), 0, 1);
  const fromStr = from.toISOString().slice(0, 10);
  const start = dates.findIndex(d => d >= fromStr);
  return [start < 0 ? 0 : start, dates.length - 1];
}}

function colorCandles(opens, closes) {{
  return closes.map((c, i) => c >= opens[i] ? '#26a641' : '#f85149');
}}

// ---- render ----------------------------------------------------------------
let currentAsset  = Object.keys(DATA)[0];
let currentPeriod = "ALL";

function render() {{
  const d     = DATA[currentAsset];
  const [s, e] = filterByPeriod(d.date, currentPeriod);
  const dates  = d.date.slice(s, e + 1);
  const opens  = d.open.slice(s, e + 1);
  const highs  = d.high.slice(s, e + 1);
  const lows   = d.low.slice(s, e + 1);
  const closes = d.close.slice(s, e + 1);
  const vols   = d.volume.slice(s, e + 1);

  const colors = colorCandles(opens, closes);

  const candlestick = {{
    type: 'candlestick',
    x: dates, open: opens, high: highs, low: lows, close: closes,
    name: currentAsset,
    increasing: {{ line: {{ color: '#26a641' }}, fillcolor: '#26a641' }},
    decreasing: {{ line: {{ color: '#f85149' }}, fillcolor: '#f85149' }},
    xaxis: 'x', yaxis: 'y',
    hoverinfo: 'x+y',
  }};

  const traces = [candlestick];

  const layout = {{
    paper_bgcolor: '#0d1117',
    plot_bgcolor:  '#0d1117',
    font: {{ color: '#8b949e', size: 11 }},
    margin: {{ l: 60, r: 20, t: 20, b: 40 }},
    xaxis: {{
      rangeslider: {{ visible: false }},
      gridcolor: '#21262d', linecolor: '#30363d',
      type: 'date',
    }},
    yaxis: {{
      gridcolor: '#21262d', linecolor: '#30363d',
      title: currentAsset,
      titlefont: {{ color: '#58a6ff' }},
      side: 'right',
    }},
    hovermode: 'x unified',
    hoverlabel: {{ bgcolor: '#161b22', bordercolor: '#30363d', font: {{ color: '#e6edf3' }} }},
    showlegend: false,
  }};

  // Add volume subplot if available
  if (vols.length > 0 && vols.some(v => v > 0)) {{
    traces.push({{
      type: 'bar',
      x: dates, y: vols,
      marker: {{ color: colors, opacity: 0.55 }},
      name: 'Volume',
      xaxis: 'x', yaxis: 'y2',
      hovertemplate: 'Vol: %{{y:.3s}}<extra></extra>',
    }});
    layout.yaxis  = {{ ...layout.yaxis,  domain: [0.25, 1] }};
    layout.yaxis2 = {{
      domain: [0, 0.22],
      gridcolor: '#21262d', linecolor: '#30363d',
      title: 'Volume', titlefont: {{ color: '#8b949e', size: 10 }},
      side: 'right',
    }};
  }}

  Plotly.react('chart', traces, layout, {{
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['autoScale2d', 'lasso2d', 'select2d'],
    displaylogo: false,
  }});
}}

// ---- event wiring ----------------------------------------------------------
document.getElementById('assetSelect').addEventListener('change', e => {{
  currentAsset = e.target.value;
  render();
}});

document.querySelectorAll('#periodBtns button').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('#periodBtns button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentPeriod = btn.dataset.p;
    render();
  }});
}});

// initial draw
render();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# 3. Build & save
# ---------------------------------------------------------------------------

def build_html() -> Path:
    assets = load_assets()

    data_dict = {name: json.loads(df_to_json(df)) for name, df in assets.items()}
    data_json = json.dumps(data_dict, separators=(",", ":"))

    asset_options = "\n".join(
        f'    <option value="{name}">{name}</option>' for name in assets
    )

    html = HTML_TEMPLATE.format(
        data_json=data_json,
        asset_options=asset_options,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "candlestick_interactive.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Saved -> {out_path}  ({out_path.stat().st_size // 1024} KB)")
    return out_path


if __name__ == "__main__":
    build_html()
