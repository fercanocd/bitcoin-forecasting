"""
Shared figure style for the Bitcoin Forecasting TFM.

Both the EDA figures (eda_plots.py) and the model-results figures
(results_plots.py) import their look-and-feel from here so the whole thesis
renders as a single, consistent visual system. Kept dependency-light (only
matplotlib + seaborn) so importing the results plots never drags in the
modelling stack.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

ROOT    = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

STYLE   = "seaborn-v0_8-whitegrid"
PALETTE = sns.color_palette("tab10")
DPI     = 150
FIGSIZE = (12, 5)

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.labelsize":  11,
    "legend.fontsize": 10,
})


def _save(fig: plt.Figure, name: str) -> Path:
    """Save a figure to reports/figures/ at the shared DPI and close it."""
    path = FIG_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"  Saved -> {path}")
    plt.close(fig)
    return path
