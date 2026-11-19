"""
plotter.py

Master 5-panel diagnostic figure builder (`DiagnosticPlotter`) for ICARUS v3.4.
Implements:
  - Panel 1: Continuous Logarithmic Rest-Frame Spectrum & Two-Stage Continuum
  - Panels 2 & 3: Featured Velocity-Space Complex Decompositions (+ Top AA Axis)
  - Panels 2R & 3R: Shared-Axis Normalized Residual Pull Strips ([-5, +5] sigma)
  - Panel 4: BPT / High-z OHNO Nebular Excitation Plane
  - Panel 5: Executive Virial & Accretion Telemetry Card
"""

__project__ = "ICARUS"
__author__ = "Joseph Havens"
__version__ = "3.4.0"

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from lmfit.model import ModelResult
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import NullFormatter, ScalarFormatter

from ..core.containers import LineComplex, Spectrum1D
from ..core.dev_tools import DevState, dev_hook
from ..core.logger import get_icarus_logger
from .bpt import BPTPlotter
from .style import (
    COLOR_BROAD,
    COLOR_CONT_FEII,
    COLOR_CONT_PL,
    COLOR_DATA_RAW,
    COLOR_ERR_BAND,
    COLOR_LIMIT,
    COLOR_NARROW,
    COLOR_OUTFLOW,
    COLOR_TOTAL_FIT,
    COLOR_WINDOW_BG,
    apply_icarus_style,
)
from .typography import decrowd_annotations, format_latex_ion

log = get_icarus_logger()
C_KMS = const.c.to("km/s").value


class DiagnosticPlotter:
    """Renders the publication-grade 5-panel diagnostic summary sheet."""

    @staticmethod
    def _select_featured_panels(
        complex_data: List[Tuple[LineComplex, Spectrum1D, ModelResult]],
        global_metrics: Dict[str, Any],
        priority_lines: Optional[List[str]] = None,
        max_panels: int = 2,
    ) -> List[Tuple[LineComplex, Spectrum1D, ModelResult]]:
        """
        Programmatically selects the top 2 featured complexes by ranking:
          1) Complex is within detector bounds (`comp.in_bounds == True`)
          2) Contains >= 1 detected transition (`not is_upper_limit` and SNR >= 3)
          3) Priority line hierarchy (`O_III_5007` + `H_beta` and `H_alpha` + `N_II_6583` first)
          4) Peak narrow SNR
        """
        if not priority_lines:
            priority_lines = [
                "O_III_5007",
                "H_beta",
                "H_alpha",
                "N_II_6583",
                "Pa_beta",
                "Pa_alpha",
                "He_I_10830",
                "S_III_9531",
                "Mg_II_2796",
                "C_IV_1548",
                "C_III_1909",
                "O_II_3729",
                "Ly_alpha",
            ]

        scored_candidates = []
        for idx, data in enumerate(complex_data):
            comp, sliced_spec, fit_result = data
            if fit_result is None:
                continue
            in_bounds = getattr(comp, "in_bounds", True)

            detected_lines = 0
            max_snr = 0.0
            best_priority_rank = 999

            for line in comp.lines:
                is_ul = bool(
                    global_metrics.get(f"{line.name}_narrow_is_upper_limit", True)
                )
                snr = float(global_metrics.get(f"{line.name}_narrow_snr", 0.0))
                if not is_ul and snr >= 3.0:
                    detected_lines += 1
                    max_snr = max(max_snr, snr)
                    if line.name in priority_lines:
                        best_priority_rank = min(
                            best_priority_rank, priority_lines.index(line.name)
                        )

            redchi = (
                float(fit_result.redchi)
                if (fit_result.redchi is not None and np.isfinite(fit_result.redchi))
                else 999.0
            )
            is_well_fit = (detected_lines > 0) and (redchi < 500.0)
            has_prio = best_priority_rank < 999

            score = (
                1 if in_bounds else 0,
                1 if is_well_fit else 0,
                1 if has_prio else 0,
                -best_priority_rank,
                max_snr,
                -idx,
            )
            scored_candidates.append((score, data))

        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        selected = [item[1] for item in scored_candidates[:max_panels]]
        selected.sort(key=lambda d: d[0].window_start.value)
        return selected

    @staticmethod
    @dev_hook(step_name="plot_target_summary")
    def plot_target_summary(
        result_or_spectrum: Any,
        output_path: Optional[Union[str, Path]] = None,
        show: bool = False,
        complex_data: Optional[
            List[Tuple[LineComplex, Spectrum1D, ModelResult]]
        ] = None,
        global_metrics: Optional[Dict[str, Any]] = None,
        priority_lines: Optional[List[str]] = None,
        global_continuum: Optional[np.ndarray] = None,
        feii_continuum: Optional[np.ndarray] = None,
    ) -> Optional[Path]:
        """
        Renders the 5-panel diagnostic PDF. Accepts either an `IcarusResult`
        instance directly (`result.plot(save_path)`) or explicit v3.3 arguments.
        """
        apply_icarus_style()
        DevState.emit("pre_plot_target_summary", locals())

        if hasattr(result_or_spectrum, "spectrum") and hasattr(
            result_or_spectrum, "complex_fits"
        ):
            res = result_or_spectrum
            full_spectrum: Spectrum1D = res.spectrum
            complex_fits = res.complex_fits
            metrics = res.raw_params if res.raw_params else res.metrics
            cont_eval = res.continuum_eval
            feii_eval = res.feii_eval
            conf_flag = int(res.confidence_flag)
            prio_lines = (
                res.config.globals.fitting.priority_lines
                if hasattr(res, "config") and res.config is not None
                else priority_lines
            )
        else:
            full_spectrum = result_or_spectrum
            complex_fits = complex_data or []
            metrics = global_metrics or {}
            cont_eval = global_continuum
            feii_eval = feii_continuum
            conf_flag = int(metrics.get("CONF_FLG", 0))
            prio_lines = priority_lines

        fig = plt.figure(figsize=(18, 12.5))
        gs_master = GridSpec(
            3,
            2,
            figure=fig,
            height_ratios=[1.05, 1.35, 1.0],
            hspace=0.34,
            wspace=0.18,
        )

        # Panel 1: Top full-width logarithmic overview
        ax_p1 = fig.add_subplot(gs_master[0, :])

        # Panels 2/2R (middle-left) and Panels 3/3R (middle-right)
        gs_p2 = gs_master[1, 0].subgridspec(
            2, 1, height_ratios=[1.0, 0.32], hspace=0.06
        )
        ax_p2 = fig.add_subplot(gs_p2[0])
        ax_p2r = fig.add_subplot(gs_p2[1], sharex=ax_p2)

        gs_p3 = gs_master[1, 1].subgridspec(
            2, 1, height_ratios=[1.0, 0.32], hspace=0.06
        )
        ax_p3 = fig.add_subplot(gs_p3[0])
        ax_p3r = fig.add_subplot(gs_p3[1], sharex=ax_p3)

        # Panel 4 (bottom-left BPT) & Panel 5 (bottom-right Telemetry Card)
        ax_p4 = fig.add_subplot(gs_master[2, 0])
        ax_p5 = fig.add_subplot(gs_master[2, 1])

        # --- Render Panel 1: Global Logarithmic Spectrum ---
        DiagnosticPlotter._plot_panel1_overview(
            ax_p1,
            full_spectrum,
            complex_fits,
            metrics,
            cont_eval,
            feii_eval,
        )

        # --- Render Panels 2/2R & 3/3R: Featured Velocity-Space Complexes ---
        featured = DiagnosticPlotter._select_featured_panels(
            complex_fits, metrics, priority_lines=prio_lines, max_panels=2
        )

        if len(featured) >= 1:
            DiagnosticPlotter._plot_velocity_complex(
                ax_p2,
                ax_p2r,
                featured[0][0],
                featured[0][1],
                featured[0][2],
                metrics,
                full_spectrum,
                feii_eval,
            )
        else:
            ax_p2.text(
                0.5,
                0.5,
                "No In-Bounds Featured Complex Detected",
                transform=ax_p2.transAxes,
                ha="center",
                va="center",
                color=COLOR_LIMIT,
            )
            ax_p2r.set_visible(False)

        if len(featured) >= 2:
            DiagnosticPlotter._plot_velocity_complex(
                ax_p3,
                ax_p3r,
                featured[1][0],
                featured[1][1],
                featured[1][2],
                metrics,
                full_spectrum,
                feii_eval,
            )
        else:
            ax_p3.text(
                0.5,
                0.5,
                "No Secondary Featured Complex Detected",
                transform=ax_p3.transAxes,
                ha="center",
                va="center",
                color=COLOR_LIMIT,
            )
            ax_p3r.set_visible(False)

        # --- Render Panel 4: BPT / High-z Excitation Diagnostic ---
        BPTPlotter.plot_bpt_panel(ax_p4, metrics)

        # --- Render Panel 5: Executive Virial & Accretion Telemetry Card ---
        DiagnosticPlotter._plot_panel5_telemetry_card(
            ax_p5, full_spectrum, complex_fits, metrics, conf_flag
        )

        fig.suptitle(
            rf"ICARUS v3.4 Spectroscopic Decomposition — Target: {full_spectrum.target.target_id} ($z = {full_spectrum.target.redshift:.4f}$)",
            fontsize=15,
            fontweight="bold",
            y=0.965,
        )

        final_path: Optional[Path] = None
        if output_path is not None:
            out_p = Path(output_path)
            if out_p.is_dir():
                final_path = out_p / f"summary_{full_spectrum.target.target_id}.pdf"
            else:
                final_path = out_p
            final_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(final_path, bbox_inches="tight", dpi=300)

        if show:
            plt.show()
        else:
            plt.close(fig)

        DevState.emit("post_plot_target_summary", locals())
        return final_path

    @staticmethod
    def _plot_panel1_overview(
        ax: Axes,
        full_spectrum: Spectrum1D,
        complex_fits: List[Tuple[LineComplex, Spectrum1D, ModelResult]],
        metrics: Dict[str, Any],
        cont_eval: Optional[np.ndarray],
        feii_eval: Optional[np.ndarray],
    ) -> None:
        x_full, y_full, err_full = full_spectrum.to_ndarrays()
        valid = (
            (~full_spectrum.mask)
            & np.isfinite(x_full)
            & np.isfinite(y_full)
            & (x_full > 0)
        )

        if not np.any(valid):
            return

        xw = x_full[valid]
        yw = y_full[valid]
        ew = err_full[valid]

        # 1. Background 1-sigma error envelope and observed spectrum step
        ax.fill_between(
            xw,
            yw - ew,
            yw + ew,
            color=COLOR_ERR_BAND,
            alpha=0.22,
            step="mid",
            zorder=0,
        )
        ax.step(
            xw,
            yw,
            where="mid",
            color=COLOR_DATA_RAW,
            alpha=0.50,
            lw=0.95,
            zorder=1,
            label="Observed Spectrum",
        )

        # 2. Stage 1 Global Power-Law + Host + Balmer Continuum
        has_cont = cont_eval is not None and len(cont_eval) == len(x_full)
        if has_cont:
            ax.plot(
                xw,
                cont_eval[valid],
                color=COLOR_CONT_PL,
                linestyle="--",
                lw=1.4,
                alpha=0.88,
                zorder=2,
                label="Global SED Continuum",
            )

        # 3. Stage 1 Convolved UV/Optical Fe II Pseudo-Continuum
        has_feii = (
            feii_eval is not None
            and len(feii_eval) == len(x_full)
            and np.nanmax(feii_eval) > 0
        )
        if has_feii:
            base_c = cont_eval[valid] if has_cont else np.zeros_like(xw)
            ax.fill_between(
                xw,
                base_c,
                base_c + feii_eval[valid],
                color=COLOR_CONT_FEII,
                alpha=0.38,
                zorder=2,
                label=r"Fe II Template",
            )

        # 4. Complex integration windows, total fits, and detected line annotations
        x_min, x_max = float(np.min(xw)), float(np.max(xw))
        ax.set_xscale("log")
        ax.set_xlim(x_min, x_max)
        candidate_ticks = [
            1000,
            1500,
            2000,
            2500,
            3000,
            4000,
            5000,
            6000,
            7500,
            10000,
            12500,
            15000,
            20000,
            30000,
        ]
        active_ticks = [t for t in candidate_ticks if x_min * 1.02 <= t <= x_max * 0.98]
        if active_ticks:
            ax.set_xticks(active_ticks)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())

        raw_annotations: List[Tuple[float, float, str]] = []
        added_fit_label = False

        for comp, sliced_spec, fit_result in complex_fits:
            if fit_result is None:
                continue
            x_c, _, _ = sliced_spec.to_ndarrays()
            ax.axvspan(
                float(np.min(x_c)),
                float(np.max(x_c)),
                color=COLOR_WINDOW_BG,
                alpha=0.35,
                zorder=0,
            )

            fe_slice = (
                np.interp(x_c, x_full, feii_eval, left=0.0, right=0.0)
                if has_feii
                else 0.0
            )
            ax.plot(
                x_c,
                fit_result.best_fit + fe_slice,
                color=COLOR_TOTAL_FIT,
                lw=1.6,
                zorder=4,
                label="Complex Fit" if not added_fit_label else None,
            )
            added_fit_label = True

            for line in comp.lines:
                is_limit = bool(
                    metrics.get(f"{line.name}_narrow_is_upper_limit", False)
                )
                if is_limit:
                    continue
                amp_key = f"{line.name}_narrow_amplitude"
                if (
                    amp_key in fit_result.params
                    and fit_result.params[amp_key].value < 1e-4
                ):
                    continue
                cen_key = f"{line.name}_narrow_center"
                cen = (
                    float(fit_result.params[cen_key].value)
                    if cen_key in fit_result.params
                    else float(line.rest_wavelength)
                )
                snr = float(metrics.get(f"{line.name}_narrow_snr", 0.0))
                if x_min <= cen <= x_max:
                    raw_annotations.append((cen, snr, line.name))

        decrowded = decrowd_annotations(
            raw_annotations, x_min, x_max, log_scale=True, min_frac_sep=0.020
        )
        trans_top = ax.get_xaxis_transform()
        bbox_props = dict(
            boxstyle="round,pad=0.18",
            facecolor="white",
            edgecolor="#1D2D44",
            lw=0.5,
            alpha=0.92,
        )

        for cen, _, l_name, show_label in decrowded:
            ax.axvline(
                cen, color="#1D2D44", linestyle=":", lw=0.95, alpha=0.50, zorder=2
            )
            if show_label:
                ax.text(
                    cen,
                    0.86,
                    format_latex_ion(l_name, is_limit=False),
                    transform=trans_top,
                    rotation=90,
                    ha="center",
                    va="top",
                    fontsize=9.5,
                    bbox=bbox_props,
                    zorder=5,
                )

        p99 = float(np.nanpercentile(yw, 99.2))
        max_y = max(float(np.nanmax(yw)), 1e-3)
        top_ymax = min(max_y, p99 * 1.75) if p99 > 0 else max_y
        ax.set_ylim(-0.08 * top_ymax, top_ymax * 1.28)

        scale_val = full_spectrum.flux_scale_factor
        exp_int = int(np.round(np.log10(scale_val))) if scale_val > 0 else 0
        scale_str = rf"10^{{{exp_int}}}\," if exp_int != 0 else ""
        ax.set_ylabel(
            rf"$f_\lambda\ [{scale_str}\mathrm{{erg\,s^{{-1}}\,cm^{{-2}}\,\AA^{{-1}}}}]$"
        )
        ax.set_xlabel(
            r"Rest-Frame Wavelength $\lambda_{\mathrm{rest}}\ [\mathrm{\AA}]$ (Log Scale)"
        )
        ax.legend(loc="upper left", ncol=4, fontsize=9.0)

    @staticmethod
    @dev_hook(step_name="plot_velocity_complex")
    def _plot_velocity_complex(
        ax_main: Axes,
        ax_pull: Axes,
        comp: LineComplex,
        sliced_spec: Spectrum1D,
        fit_result: ModelResult,
        metrics: Dict[str, Any],
        full_spectrum: Spectrum1D,
        feii_eval: Optional[np.ndarray],
    ) -> None:
        x_aa, y_raw, err = sliced_spec.to_ndarrays()
        valid = (~sliced_spec.mask) & np.isfinite(x_aa) & np.isfinite(y_raw)
        if not np.any(valid):
            return

        anchor_line = comp.primary_line
        lam0 = float(anchor_line.rest_wavelength)

        # Transform wavelength axis to relative velocity space Delta v [km/s]
        v_kms = C_KMS * (x_aa - lam0) / lam0
        v_val = v_kms[valid]
        y_val = y_raw[valid]
        e_val = err[valid]

        x_full, _, _ = full_spectrum.to_ndarrays()
        has_feii = (
            feii_eval is not None
            and len(feii_eval) == len(x_full)
            and np.nanmax(feii_eval) > 0
        )
        fe_slice = (
            np.interp(x_aa, x_full, feii_eval, left=0.0, right=0.0)
            if has_feii
            else np.zeros_like(x_aa)
        )

        # 1. Background 1-sigma ribbon and observed flux step
        ax_main.fill_between(
            v_val,
            y_val - e_val,
            y_val + e_val,
            color=COLOR_ERR_BAND,
            alpha=0.24,
            step="mid",
            zorder=0,
        )
        ax_main.step(
            v_val,
            y_val,
            where="mid",
            color=COLOR_DATA_RAW,
            alpha=0.55,
            lw=1.1,
            zorder=1,
        )

        components = fit_result.eval_components(x=x_aa)
        c_loc = components.get("cont_", np.zeros_like(x_aa))
        base_cont = c_loc + fe_slice
        total_model = fit_result.best_fit + fe_slice

        # 2. Local + Fe II Continuum
        ax_main.plot(
            v_kms,
            base_cont,
            color=COLOR_CONT_PL,
            linestyle="--",
            lw=1.4,
            label="Local Continuum",
            zorder=2,
        )
        if np.any(fe_slice > 1e-5):
            ax_main.fill_between(
                v_kms,
                c_loc,
                base_cont,
                color=COLOR_CONT_FEII,
                alpha=0.35,
                label="Fe II Shelf",
                zorder=2,
            )

        # 3. Decomposed Narrow, Broad, and Outflow Gaussian Profiles
        handled_labels = {"Local Continuum", "Fe II Shelf", "Total Fit"}
        for cname, cdata in components.items():
            if "cont_" in cname:
                continue
            elif "broad" in cname:
                if np.nanmax(cdata) < 1e-4:
                    continue
                lbl = "Broad BLR" if "Broad BLR" not in handled_labels else None
                ax_main.fill_between(
                    v_kms,
                    base_cont,
                    base_cont + cdata,
                    color=COLOR_BROAD,
                    alpha=0.20,
                    zorder=2,
                )
                ax_main.plot(
                    v_kms,
                    base_cont + cdata,
                    color=COLOR_BROAD,
                    linestyle="-.",
                    lw=1.9,
                    label=lbl,
                    zorder=3,
                )
                handled_labels.add("Broad BLR")
            elif "outflow" in cname:
                if np.nanmax(cdata) < 1e-4:
                    continue
                lbl = "Outflow Wing" if "Outflow Wing" not in handled_labels else None
                ax_main.fill_between(
                    v_kms,
                    base_cont,
                    base_cont + cdata,
                    color=COLOR_OUTFLOW,
                    alpha=0.18,
                    zorder=2,
                )
                ax_main.plot(
                    v_kms,
                    base_cont + cdata,
                    color=COLOR_OUTFLOW,
                    linestyle=":",
                    lw=1.8,
                    label=lbl,
                    zorder=3,
                )
                handled_labels.add("Outflow Wing")
            elif "narrow" in cname:
                base_ion = cname.rstrip("_").replace("_narrow", "")
                if bool(metrics.get(f"{base_ion}_narrow_is_upper_limit", False)):
                    continue
                lbl = "Narrow Core" if "Narrow Core" not in handled_labels else None
                ax_main.plot(
                    v_kms,
                    base_cont + cdata,
                    color=COLOR_NARROW,
                    linestyle="-",
                    lw=1.85,
                    label=lbl,
                    zorder=4,
                )
                handled_labels.add("Narrow Core")

        # 4. High-Contrast Crimson Total Fit
        ax_main.plot(
            v_kms,
            total_model,
            color=COLOR_TOTAL_FIT,
            lw=2.2,
            label="Total Fit",
            zorder=5,
        )

        v_min, v_max = float(np.min(v_val)), float(np.max(v_val))
        ax_main.set_xlim(v_min, v_max)

        # Secondary Top Rest-Frame Wavelength Axis [AA]
        sec_ax = ax_main.secondary_xaxis(
            "top",
            functions=(
                lambda v: lam0 * (1.0 + v / C_KMS),
                lambda w: C_KMS * (w - lam0) / lam0,
            ),
        )
        sec_ax.set_xlabel(
            rf"Rest-Frame $\lambda\ [\mathrm{{\AA}}]$ (Anchor: {format_latex_ion(anchor_line.name)})",
            fontsize=9.5,
        )
        sec_ax.tick_params(labelsize=8.5)

        # Annotate detected lines in velocity space
        sub_anns: List[Tuple[float, float, str]] = []
        for line in comp.lines:
            if bool(metrics.get(f"{line.name}_narrow_is_upper_limit", False)):
                continue
            cen_key = f"{line.name}_narrow_center"
            cen_aa = (
                float(fit_result.params[cen_key].value)
                if cen_key in fit_result.params
                else float(line.rest_wavelength)
            )
            v_cen = C_KMS * (cen_aa - lam0) / lam0
            snr = float(metrics.get(f"{line.name}_narrow_snr", 0.0))
            if v_min <= v_cen <= v_max:
                sub_anns.append((v_cen, snr, line.name))

        decrowded_sub = decrowd_annotations(
            sub_anns, v_min, v_max, log_scale=False, min_frac_sep=0.065
        )
        trans_sub = ax_main.get_xaxis_transform()
        bbox_sub = dict(
            boxstyle="round,pad=0.18",
            facecolor="white",
            edgecolor="#1D2D44",
            lw=0.5,
            alpha=0.92,
        )
        for v_cen, _, l_name, show_lbl in decrowded_sub:
            ax_main.axvline(
                v_cen, color="#1D2D44", linestyle="--", lw=1.0, alpha=0.55, zorder=2
            )
            if show_lbl:
                ax_main.text(
                    v_cen,
                    0.93,
                    format_latex_ion(l_name, is_limit=False),
                    transform=trans_sub,
                    rotation=90,
                    ha="center",
                    va="top",
                    fontsize=9.5,
                    bbox=bbox_sub,
                    zorder=6,
                )

        max_y = max(float(np.nanmax(y_val)), float(np.nanmax(total_model)), 1e-4)
        min_y = min(0.0, float(np.nanmin(y_val)))
        ax_main.set_ylim(min_y - 0.08 * abs(max_y), max_y * 1.32)
        ax_main.set_ylabel(r"Scaled $f_\lambda$")
        ax_main.tick_params(labelbottom=False)
        ax_main.legend(loc="upper left", fontsize=8.5)

        # --- Render Normalized Residual Pull Strip (Panels 2R & 3R) ---
        safe_e = np.where(e_val > 0, e_val, np.nan)
        pulls = (y_val - total_model[valid]) / safe_e

        ax_pull.axhspan(-1.0, 1.0, color="#CBD5E1", alpha=0.45, zorder=0)
        ax_pull.axhspan(-3.0, 3.0, color="#E2E8F0", alpha=0.30, zorder=0)
        ax_pull.axhline(0.0, color=COLOR_TOTAL_FIT, linestyle="--", lw=1.0, zorder=1)
        ax_pull.step(
            v_val,
            np.clip(pulls, -4.9, 4.9),
            where="mid",
            color=COLOR_DATA_RAW,
            lw=0.95,
            alpha=0.85,
            zorder=2,
        )
        ax_pull.set_ylim(-5.0, 5.0)
        ax_pull.set_yticks([-3, 0, 3])
        ax_pull.set_ylabel(r"Pull ($\sigma$)", fontsize=9.5)
        ax_pull.set_xlabel(r"Relative Velocity $\Delta v\ [\mathrm{km\,s^{-1}}]$")

    @staticmethod
    def _plot_panel5_telemetry_card(
        ax: Axes,
        spectrum: Spectrum1D,
        complex_fits: List[Tuple[LineComplex, Spectrum1D, ModelResult]],
        metrics: Dict[str, Any],
        conf_flag: int,
    ) -> None:
        ax.axis("off")

        mbh = float(metrics.get("M_BH_Msun", np.nan))
        mbh_cal = str(metrics.get("M_BH_Calibrator", "None"))
        lbol = float(metrics.get("L_bol_erg_s", np.nan))
        lamedd = float(metrics.get("lambda_Edd", np.nan))
        fwhm_b = float(metrics.get("FWHM_broad_kms", np.nan))
        v_out = float(metrics.get("v_outflow_OIII_kms", np.nan))
        r_feii = float(metrics.get("R_FeII", np.nan))
        sfr = float(metrics.get("SFR_uncorrected_Msun_yr", np.nan))
        sfr_ul = bool(metrics.get("SFR_is_upper_limit", False))

        redchis = [
            float(r.redchi)
            for _, _, r in complex_fits
            if r is not None and r.redchi is not None and np.isfinite(r.redchi)
        ]
        med_chi2 = float(np.median(redchis)) if redchis else np.nan

        mbh_str = (
            rf"$\log(M_{{\mathrm{{BH}}}}/M_\odot) = {np.log10(mbh):.2f}\ \pm 0.35$ [{mbh_cal}]"
            if (np.isfinite(mbh) and mbh > 0)
            else "Unconstrained (No Confirmed BLR)"
        )
        lbol_str = (
            rf"$\log(L_{{\mathrm{{bol}}}}/\mathrm{{erg\,s^{{-1}}}}) = {np.log10(lbol):.2f}$"
            if (np.isfinite(lbol) and lbol > 0)
            else "N/A"
        )
        edd_str = f"{lamedd:.3f}" if np.isfinite(lamedd) else "N/A"
        fwhm_str = (
            rf"${fwhm_b:.0f}\ \mathrm{{km\,s^{{-1}}}}$ (Deconvolved)"
            if np.isfinite(fwhm_b)
            else "N/A"
        )
        vout_str = (
            rf"${v_out:.0f}\ \mathrm{{km\,s^{{-1}}}}$"
            if np.isfinite(v_out)
            else "None Detected"
        )
        rfe_str = f"{r_feii:.2f}" if np.isfinite(r_feii) else "N/A"

        if np.isfinite(sfr) and sfr > 0:
            op = "<" if sfr_ul else "="
            sfr_str = rf"${op}\ {sfr:.2f}\ M_\odot\,\mathrm{{yr^{{-1}}}}$"
        else:
            sfr_str = "N/A"

        chi_str = f"{med_chi2:.2f}" if np.isfinite(med_chi2) else "N/A"

        # Decode 6-bit additive confidence flag
        flag_badges = []
        if conf_flag == 0:
            flag_badges.append("0 (Clean Global Fit)")
        else:
            if conf_flag & 1:
                flag_badges.append("Bit0:LocalCont")
            if conf_flag & 2:
                flag_badges.append("Bit1:FeII")
            if conf_flag & 4:
                flag_badges.append("Bit2:BroadBLR")
            if conf_flag & 8:
                flag_badges.append("Bit3:Outflow")
            if conf_flag & 16:
                flag_badges.append("Bit4:BoundHit")
            if conf_flag & 32:
                flag_badges.append("Bit5:AllUL")

        rows = [
            (
                "Target ID / Redshift",
                rf"{spectrum.target.target_id} ($z = {spectrum.target.redshift:.4f}$)",
            ),
            ("Virial BH Mass", mbh_str),
            ("Bolometric Luminosity", lbol_str),
            ("Eddington Ratio", edd_str),
            ("Intrinsic Broad FWHM", fwhm_str),
            ("[O III] Outflow Shift", vout_str),
            ("Optical Fe II Ratio", rfe_str),
            ("Narrow-Line SFR", sfr_str),
            (
                "Complexes Fit / Med. $\\chi_\\nu^2$",
                f"{len(complex_fits)} Windows / {chi_str}",
            ),
            ("Confidence Bitmask", f"{conf_flag} ({', '.join(flag_badges)})"),
        ]

        # Draw structured telemetry card
        card_box = plt.Rectangle(
            (0.02, 0.02),
            0.96,
            0.94,
            transform=ax.transAxes,
            facecolor="#F8FAFC",
            edgecolor="#1D2D44",
            linewidth=1.3,
            zorder=1,
        )
        ax.add_patch(card_box)

        header_box = plt.Rectangle(
            (0.02, 0.86),
            0.96,
            0.10,
            transform=ax.transAxes,
            facecolor="#1D2D44",
            edgecolor="#1D2D44",
            zorder=2,
        )
        ax.add_patch(header_box)
        ax.text(
            0.50,
            0.91,
            "EXECUTIVE VIRIAL & ACCRETION TELEMETRY CARD",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color="white",
            zorder=3,
        )

        y_cursor = 0.80
        y_step = 0.076
        for idx, (k_label, v_text) in enumerate(rows):
            if idx % 2 == 1:
                row_bg = plt.Rectangle(
                    (0.03, y_cursor - 0.032),
                    0.94,
                    y_step,
                    transform=ax.transAxes,
                    facecolor="#EDF2F7",
                    edgecolor="none",
                    zorder=1.5,
                )
                ax.add_patch(row_bg)

            ax.text(
                0.05,
                y_cursor,
                f"{k_label}:",
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=9.8,
                fontweight="bold",
                color="#1E293B",
                zorder=3,
            )
            ax.text(
                0.95,
                y_cursor,
                v_text,
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=9.8,
                color="#0F172A",
                zorder=3,
            )
            y_cursor -= y_step
