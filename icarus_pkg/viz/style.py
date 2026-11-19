"""
style.py

Global Matplotlib rcParams and color palette definitions for ICARUS v3.4.
Enforces crash-free native STIX sans-serif mathtext (`usetex=False`) for
100% portability across headless HPC clusters and local workstations.
"""

__project__ = "ICARUS"
__author__ = "Joseph Havens"
__version__ = "3.4.0"

from typing import Dict

import matplotlib as mpl
import matplotlib.pyplot as plt

# Official ICARUS v3.4 Hex Color Palette
COLOR_DATA_RAW = "#34495E"  # Muted Slate background step
COLOR_ERR_BAND = "#8D99AE"  # Soft Cool Gray 1-sigma uncertainty envelope
COLOR_TOTAL_FIT = "#E63946"  # High-contrast Crimson total model
COLOR_CONT_PL = "#118AB2"  # Ocean Blue dashed power-law / local continuum
COLOR_CONT_FEII = "#F4A261"  # Warm Gold shaded Fe II pseudo-continuum
COLOR_NARROW = "#06D6A0"  # Emerald Green narrow core
COLOR_BROAD = "#FF9F1C"  # Amber Orange dash-dot broad BLR wing
COLOR_OUTFLOW = "#7209B7"  # Royal Violet dotted outflow wing
COLOR_LIMIT = "#6C757D"  # Neutral Gray 3-sigma upper limit marker
COLOR_WINDOW_BG = "#E8F0FE"  # Subtle Ice Blue complex integration window span

ICARUS_PALETTE: Dict[str, str] = {
    "data_raw": COLOR_DATA_RAW,
    "err_band": COLOR_ERR_BAND,
    "total_fit": COLOR_TOTAL_FIT,
    "cont_pl": COLOR_CONT_PL,
    "cont_feii": COLOR_CONT_FEII,
    "narrow": COLOR_NARROW,
    "broad": COLOR_BROAD,
    "outflow": COLOR_OUTFLOW,
    "limit": COLOR_LIMIT,
    "window_bg": COLOR_WINDOW_BG,
}


def apply_icarus_style() -> None:
    """
    Applies publication-grade Matplotlib rcParams globally without requiring
    an external system TeX installation (`text.usetex = False`).
    """
    plt.rcParams.update(
        {
            "text.usetex": False,
            "mathtext.fontset": "stixsans",
            "font.family": "sans-serif",
            "font.sans-serif": [
                "DejaVu Sans",
                "Arial",
                "Helvetica",
                "Liberation Sans",
            ],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.linewidth": 1.15,
            "axes.edgecolor": "#1D2D44",
            "axes.labelcolor": "#0D1B2A",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 5.0,
            "ytick.major.size": 5.0,
            "xtick.minor.size": 2.8,
            "ytick.minor.size": 2.8,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#CBD5E1",
            "legend.fontsize": 9.5,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    mpl.rcParams["axes.formatter.use_mathtext"] = True
