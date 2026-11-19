"""
bpt.py

Renders `[PANEL 4] BPT / High-z Excitation Diagnostic` for ICARUS v3.4.
Maps targets onto the canonical Baldwin-Phillips-Terlevich (1981) plane against
the Kewley et al. (2001), Kauffmann et al. (2003), and Schawinski et al. (2007)
demarcation curves, with an automatic high-redshift fallback to the
[Ne III] 3869 / [O II] 3727 vs. [O III] 5007 / H_beta (OHNO) diagram.
"""

__project__ = "ICARUS"
__author__ = "Joseph Havens"
__version__ = "3.4.0"

from typing import Any, Dict

import numpy as np
from matplotlib.axes import Axes

from .style import COLOR_CONT_PL, COLOR_DATA_RAW, COLOR_TOTAL_FIT


class BPTPlotter:
    """Renders the canonical BPT diagram or high-z OHNO excitation fallback."""

    @staticmethod
    def plot_bpt_panel(ax: Axes, metrics: Dict[str, Any]) -> None:
        log_n2_ha = float(metrics.get("log_N2_Ha", np.nan))
        log_o3_hb = float(metrics.get("log_O3_Hb", np.nan))
        log_ne3_o2 = float(metrics.get("log_Ne3_O2", np.nan))

        n2_lim = str(metrics.get("N2_Ha_limit", "none"))
        o3_lim = str(metrics.get("O3_Hb_limit", "none"))

        # High-z fallback: if [N II]/H_alpha is unavailable (e.g., z > 4) but [Ne III]/[O II] is measured
        if (
            not np.isfinite(log_n2_ha)
            and np.isfinite(log_ne3_o2)
            and np.isfinite(log_o3_hb)
        ):
            BPTPlotter._plot_ohno_fallback(ax, log_ne3_o2, log_o3_hb)
            return

        x_min, x_max = -2.0, 0.75
        y_min, y_max = -1.2, 1.5

        # 1. Kewley et al. (2001) theoretical extreme starburst boundary
        x_kew = np.linspace(x_min, 0.42, 300)
        y_kew = 0.61 / (x_kew - 0.47) + 1.19

        # 2. Kauffmann et al. (2003) empirical pure star-forming sequence
        x_kauf = np.linspace(x_min, -0.01, 300)
        y_kauf = 0.61 / (x_kauf - 0.05) + 1.30

        # 3. Schawinski et al. (2007) empirical Seyfert / LINER division
        x_schaw = np.linspace(-0.18, x_max, 200)
        y_schaw = 1.05 * x_schaw + 0.45

        # Subtle background contextual regime shading
        ax.fill_between(
            x_kauf,
            y_min,
            np.clip(y_kauf, y_min, y_max),
            color="#06D6A0",
            alpha=0.07,
            zorder=0,
        )
        ax.fill_between(
            x_kew,
            np.clip(y_kew, y_min, y_max),
            y_max,
            color=COLOR_TOTAL_FIT,
            alpha=0.06,
            zorder=0,
        )

        ax.plot(
            x_kew,
            y_kew,
            color=COLOR_TOTAL_FIT,
            lw=1.7,
            linestyle="-",
            label="Kewley+01 (Max SB)",
            zorder=2,
        )
        ax.plot(
            x_kauf,
            y_kauf,
            color=COLOR_CONT_PL,
            lw=1.5,
            linestyle="--",
            label="Kauffmann+03 (SF)",
            zorder=2,
        )
        ax.plot(
            x_schaw,
            y_schaw,
            color=COLOR_DATA_RAW,
            lw=1.3,
            linestyle=":",
            label="Schawinski+07",
            zorder=2,
        )

        # Regime contextual watermark labels
        ax.text(
            -1.45,
            -0.75,
            "H II / Star-Forming",
            fontsize=9.5,
            fontweight="bold",
            color="#058C68",
            alpha=0.85,
        )
        ax.text(
            -0.30,
            -0.55,
            "Composite",
            fontsize=9.0,
            fontweight="bold",
            color="#118AB2",
            alpha=0.85,
        )
        ax.text(
            -0.85,
            1.15,
            "AGN / Seyfert",
            fontsize=10.0,
            fontweight="bold",
            color=COLOR_TOTAL_FIT,
            alpha=0.90,
        )
        ax.text(
            0.22,
            -0.15,
            "LINER",
            fontsize=9.5,
            fontweight="bold",
            color=COLOR_DATA_RAW,
            alpha=0.85,
        )

        # Plot Target Jewel Marker + Directional 3-sigma Limit Arrows
        if np.isfinite(log_n2_ha) and np.isfinite(log_o3_hb):
            px = float(np.clip(log_n2_ha, x_min + 0.08, x_max - 0.08))
            py = float(np.clip(log_o3_hb, y_min + 0.08, y_max - 0.08))

            dx = -0.22 if n2_lim == "left" else (0.22 if n2_lim == "right" else 0.0)
            dy = -0.22 if o3_lim == "down" else (0.22 if o3_lim == "up" else 0.0)

            if dx != 0.0 or dy != 0.0:
                ax.annotate(
                    "",
                    xy=(px + dx, py + dy),
                    xytext=(px, py),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="#1D2D44",
                        lw=1.8,
                        mutation_scale=13,
                    ),
                    zorder=5,
                )

            ax.scatter(
                [px],
                [py],
                marker="D",
                s=95,
                facecolor=COLOR_TOTAL_FIT,
                edgecolor="black",
                linewidth=1.4,
                label="Target Galaxy",
                zorder=6,
            )
        else:
            ax.text(
                0.50,
                0.50,
                "Optical BPT Lines Off-Detector\n(or Unconstrained at Target Redshift)",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9.5,
                color="#6C757D",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="#F8FAFC",
                    edgecolor="#CBD5E1",
                    alpha=0.92,
                ),
                zorder=4,
            )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(r"$\log_{10}([\mathrm{N\,II}]\,\lambda 6583 / \mathrm{H}\alpha)$")
        ax.set_ylabel(r"$\log_{10}([\mathrm{O\,III}]\,\lambda 5007 / \mathrm{H}\beta)$")
        ax.set_title(
            "BPT Nebular Excitation Diagnostic", fontsize=11, fontweight="bold"
        )
        ax.legend(loc="lower left", fontsize=8.5, framealpha=0.90)

    @staticmethod
    def _plot_ohno_fallback(ax: Axes, log_ne3_o2: float, log_o3_hb: float) -> None:
        """
        High-redshift (z > 4) excitation diagnostic comparing
        log10([Ne III] 3869 / [O II] 3727) vs. log10([O III] 5007 / H_beta).
        """
        x_min, x_max = -1.6, 0.6
        y_min, y_max = -0.8, 1.5

        x_div = np.linspace(x_min, x_max, 200)
        y_div = -0.65 * x_div + 0.35

        ax.fill_between(
            x_div, y_div, y_max, color=COLOR_TOTAL_FIT, alpha=0.06, zorder=0
        )
        ax.fill_between(x_div, y_min, y_div, color="#06D6A0", alpha=0.07, zorder=0)
        ax.plot(
            x_div,
            y_div,
            color=COLOR_TOTAL_FIT,
            lw=1.6,
            linestyle="--",
            label="AGN / High-Ionization Division",
            zorder=2,
        )

        px = float(np.clip(log_ne3_o2, x_min + 0.08, x_max - 0.08))
        py = float(np.clip(log_o3_hb, y_min + 0.08, y_max - 0.08))

        ax.scatter(
            [px],
            [py],
            marker="D",
            s=95,
            facecolor=COLOR_TOTAL_FIT,
            edgecolor="black",
            linewidth=1.4,
            label="Target Galaxy",
            zorder=6,
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(
            r"$\log_{10}([\mathrm{Ne\,III}]\,\lambda 3869 / [\mathrm{O\,II}]\,\lambda 3727)$"
        )
        ax.set_ylabel(r"$\log_{10}([\mathrm{O\,III}]\,\lambda 5007 / \mathrm{H}\beta)$")
        ax.set_title(
            "High-z OHNO Ionization Diagnostic", fontsize=11, fontweight="bold"
        )
        ax.legend(loc="lower left", fontsize=8.5, framealpha=0.90)
