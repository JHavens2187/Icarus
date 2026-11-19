"""
api.py

Object-oriented Python interface for ICARUS v3.4.
Replaces the procedural loop of v3.3 main.py with a scriptable, stateless
orchestrator (AGNFitter) and an immutable result container (IcarusResult).
Supports both pre-loaded Spectrum1D objects and direct single-file FITS paths.
"""

from __future__ import annotations

__project__ = "ICARUS"
__author__ = "Joseph Havens"
__version__ = "3.4.0"

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.table import QTable

from .core.config import IcarusConfig
from .core.containers import EmissionLineDef, LineComplex, Spectrum1D
from .core.dev_tools import DevState, dev_hook
from .core.exceptions import IcarusConvergenceError, IcarusDataDimensionError
from .core.io import ResultWriter
from .core.logger import get_icarus_logger
from .physics.continuum import ContinuumFitter
from .physics.demographics import DemographicsCalculator
from .physics.fitting_engine import CoreFitter, SpectralPreparer
from .viz.plotter import DiagnosticPlotter
from .viz.typography import sanitize_line_name

log = get_icarus_logger()
SQRT_2_PI = np.sqrt(2.0 * np.pi)


@dataclass(frozen=True, kw_only=True)
class IcarusResult:
    """
    Immutable result container returned by AGNFitter.fit().
    Holds the input spectrum, converged models, HDU tables, demographics,
    and confidence bitmask.
    """

    spectrum: Spectrum1D
    config: IcarusConfig
    continuum_eval: Optional[np.ndarray] = None
    feii_eval: Optional[np.ndarray] = None
    complex_fits: List[Tuple[LineComplex, Spectrum1D, Any]] = field(
        default_factory=list
    )
    raw_params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    line_table: Optional[QTable] = None
    synth_table: Optional[QTable] = None
    confidence_flag: int = 0

    def plot(
        self, save_path: Optional[Union[str, Path]] = None, show: bool = False
    ) -> Optional[Path]:
        """Renders the 5-panel diagnostic PDF via DiagnosticPlotter."""
        out_path = Path(save_path) if save_path is not None else None
        return DiagnosticPlotter.plot_target_summary(self, out_path, show=show)

    def to_fits(self, save_path: Union[str, Path]) -> Path:
        """Writes the 3-HDU Multi-Extension FITS file via ResultWriter."""
        out_path = Path(save_path)
        return ResultWriter.write_mef_fits(self, out_path)

    def to_ecsv(self, save_path: Union[str, Path]) -> Path:
        """Writes the flat 1-row target catalog (.ecsv) for survey aggregation."""
        out_path = Path(save_path)
        return ResultWriter.write_flat_ecsv(self, out_path)

    def summary(self) -> str:
        """Returns an executive ASCII summary for terminal/Jupyter inspection."""
        t_id = self.spectrum.target.target_id
        z = self.spectrum.target.redshift
        mbh = self.metrics.get("M_BH_Msun", np.nan)
        mbh_cal = self.metrics.get("M_BH_Calibrator", "None")
        lbol = self.metrics.get("L_bol_erg_s", np.nan)
        edd = self.metrics.get("lambda_Edd", np.nan)
        sfr = self.metrics.get("SFR_uncorrected_Msun_yr", np.nan)

        lines = [
            f"================ ICARUS v{__version__} FIT SUMMARY ================",
            f"Target ID       : {t_id} (z = {z:.4f})",
            f"Complexes Fit   : {len(self.complex_fits)}",
            (
                f"log10(M_BH/Msun): {np.log10(mbh):.2f} [{mbh_cal}]"
                if (np.isfinite(mbh) and mbh > 0)
                else "log10(M_BH/Msun): N/A"
            ),
            (
                f"log10(L_bol)    : {np.log10(lbol):.2f} erg/s"
                if (np.isfinite(lbol) and lbol > 0)
                else "log10(L_bol)    : N/A"
            ),
            (
                f"Eddington Ratio : {edd:.3f}"
                if np.isfinite(edd)
                else "Eddington Ratio : N/A"
            ),
            (
                f"SFR (Msun/yr)   : {sfr:.2f}"
                if np.isfinite(sfr)
                else "SFR (Msun/yr)   : N/A"
            ),
            f"Confidence Flag : {self.confidence_flag} (0b{self.confidence_flag:06b})",
            "==============================================================",
        ]
        return "\n".join(lines)


class AGNFitter:
    """
    Stateless, picklable orchestrator class for single-target AGN spectral fitting.
    Accepts either an instantiated `Spectrum1D` or a direct FITS filepath.
    """

    BROAD_BASES = (
        "Ha",
        "H_alpha",
        "Hb",
        "H_beta",
        "Hg",
        "H_gamma",
        "Hd",
        "H_delta",
        "Heps",
        "H_epsilon",
        "Hzeta",
        "H_zeta",
        "Paa",
        "Pa_alpha",
        "Pab",
        "Pa_beta",
        "Pag",
        "Pa_gamma",
        "Pad",
        "Pa_delta",
        "Pa_epsilon",
        "Bra",
        "Br_alpha",
        "Brb",
        "Br_beta",
        "Brg",
        "Br_gamma",
        "Brd",
        "Br_delta",
        "Pfa",
        "Pf_alpha",
        "Pfb",
        "Pf_beta",
        "Pfg",
        "Pf_gamma",
        "Hua",
        "Hu_alpha",
        "Hub",
        "Hu_beta",
        "Hug",
        "Hu_gamma",
        "Lya",
        "Ly_alpha",
        "Lyb",
        "Ly_beta",
        "C_IV",
        "Mg_II",
        "He_II",
        "He_I",
        "C_III",
    )

    def __init__(
        self,
        spectrum: Union[Spectrum1D, str, Path],
        preset: Union[str, Path, IcarusConfig] = "jwst_nirspec_medium",
        redshift: Optional[float] = None,
        target_id: Optional[str] = None,
        setup: Optional[str] = None,
        column_mapping: Optional[Dict[str, str]] = None,
        **overrides: Any,
    ) -> None:
        if isinstance(preset, IcarusConfig):
            self.master_config = preset.apply_overrides(
                column_mapping=column_mapping, **overrides
            )
        else:
            self.master_config = IcarusConfig.from_preset(
                preset, column_mapping=column_mapping, **overrides
            )

        if isinstance(spectrum, (str, Path)):
            self.spectrum = Spectrum1D.read(
                filepath=spectrum,
                redshift=float(redshift or 0.0),
                target_id=target_id,
                preset=self.master_config,
                setup=setup,
                column_mapping=column_mapping,
            )
        else:
            self.spectrum = spectrum

        self.local_config = self.master_config.get_target_config(self.spectrum.target)

    @dev_hook(step_name="agn_fitter_fit")
    def fit(self) -> IcarusResult:
        """Executes the two-stage continuum and kinematic decomposition pipeline."""
        line_defs = self._load_line_defs()
        complexes = SpectralPreparer.group_into_complexes(
            line_defs, self.local_config.fitting
        )

        # Stage 1: Global Power-Law + Fe II Continuum Cascade
        global_cont_comp, global_cont_eval, feii_eval = self._run_stage1_continuum()

        # Stage 2: Local Complex Kinematic Decomposition
        complex_fits, merged_params = self._run_stage2_complexes(
            complexes, global_cont_comp, feii_eval
        )

        # Stage 3: Virial Demographics, BPT, SFR, and Confidence Bitmask
        metrics, conf_flag = DemographicsCalculator.compute_all(
            merged_params,
            self.spectrum,
            self.master_config,
            global_cont_eval=global_cont_eval,
            feii_eval=feii_eval,
        )
        merged_params.update(metrics)

        # Build HDU 1 (Line Catalog) & HDU 2 (Synthetic Spectrum) QTables
        line_table = ResultWriter.build_line_catalog_table(
            self.spectrum, complex_fits, merged_params
        )
        synth_table = ResultWriter.build_synthetic_spectrum_table(
            self.spectrum, global_cont_eval, feii_eval, complex_fits
        )

        res = IcarusResult(
            spectrum=self.spectrum,
            config=self.master_config,
            continuum_eval=global_cont_eval,
            feii_eval=feii_eval,
            complex_fits=complex_fits,
            raw_params=merged_params,
            metrics=metrics,
            line_table=line_table,
            synth_table=synth_table,
            confidence_flag=conf_flag,
        )
        DevState.flush_telemetry()
        return res

    def _load_line_defs(self) -> List[EmissionLineDef]:
        """Loads the vacuum master line table from icarus_pkg/data/."""
        line_table_path = Path(self.local_config.line_table)
        if not line_table_path.exists():
            raise FileNotFoundError(f"Master line table not found: {line_table_path}")

        line_catalog = QTable.read(line_table_path)
        str_cols = [
            c
            for c in line_catalog.colnames
            if line_catalog[c].dtype.kind in ["U", "S", "O"]
        ]
        name_col = str_cols[0]
        wave_col = [
            c for c in line_catalog.colnames if line_catalog[c].dtype.kind in ["f", "i"]
        ][0]

        profile_col = None
        for sc in str_cols[1:]:
            sample_vals = " ".join(str(v).lower() for v in line_catalog[sc][:5])
            if "narrow" in sample_vals or "broad" in sample_vals:
                profile_col = sc
                break

        target_lines = self.local_config.lines
        valid_lines = (
            line_catalog[name_col].tolist() if target_lines == "all" else target_lines
        )

        line_defs: List[EmissionLineDef] = []
        for raw_name in valid_lines:
            row_mask = line_catalog[name_col] == raw_name
            if not np.any(row_mask):
                continue

            raw_wave = line_catalog[wave_col][row_mask][0]
            rest_wave = float(
                raw_wave.value if hasattr(raw_wave, "value") else raw_wave
            )
            safe_name = sanitize_line_name(raw_name)

            if profile_col is not None:
                prof_str = str(line_catalog[profile_col][row_mask][0]).lower()
                is_broad = "broad" in prof_str and "O_III" not in safe_name
            else:
                is_broad = any(safe_name.startswith(b) for b in self.BROAD_BASES) and (
                    "O_III" not in safe_name
                )

            is_outflow = safe_name in ("O_III_4959", "O_III_5007")
            line_defs.append(
                EmissionLineDef(
                    name=safe_name,
                    rest_wavelength=rest_wave,
                    test_broad=is_broad,
                    outflow_candidate=is_outflow,
                )
            )

        return line_defs

    def _run_stage1_continuum(
        self,
    ) -> Tuple[Optional[Any], Optional[np.ndarray], Optional[np.ndarray]]:
        """Runs Stage 1 Two-Pass Global Continuum + Fe II Template Cascade."""
        return ContinuumFitter.fit_global_sed(self.spectrum, self.local_config.fitting)

    def _run_stage2_complexes(
        self,
        complexes: List[LineComplex],
        global_cont_comp: Optional[Any],
        feii_eval: Optional[np.ndarray],
    ) -> Tuple[List[Tuple[LineComplex, Spectrum1D, Any]], Dict[str, Any]]:
        """Runs Stage 2 Local Kinematic decomposition across all in-bounds complexes."""
        target = self.spectrum.target
        wave_full, _, _ = self.spectrum.to_ndarrays()
        has_feii = (
            feii_eval is not None
            and len(feii_eval) == len(wave_full)
            and np.nanmax(feii_eval) > 0
        )

        complex_fits: List[Tuple[LineComplex, Spectrum1D, Any]] = []
        merged_params: Dict[str, Any] = {
            "TargetID": target.target_id,
            "Redshift": target.redshift,
        }

        for comp in complexes:
            try:
                sliced_spec = SpectralPreparer.slice_spectrum(self.spectrum, comp)
                x_slice, y_data, _ = sliced_spec.to_ndarrays()

                fe_slice = (
                    np.interp(x_slice, wave_full, feii_eval, left=0.0, right=0.0)
                    if has_feii
                    else np.zeros_like(x_slice)
                )

                DevState.emit("pre_evaluate_broad_line", locals())

                winning_fit = CoreFitter.evaluate_broad_line(
                    comp,
                    sliced_spec,
                    self.local_config.fitting,
                    continuum=global_cont_comp,
                    feii_eval=fe_slice,
                )

                errors = CoreFitter.estimate_errors_boxcar(
                    sliced_spec, winning_fit, boxcar_width=5
                )
                merged_params[f"{comp.name}_Reduced_ChiSq"] = winning_fit.redchi

                for param_name, best_val in winning_fit.params.valuesdict().items():
                    err_16, err_50, err_84 = errors.get(
                        param_name, (np.nan, np.nan, np.nan)
                    )
                    merged_params[param_name] = best_val
                    merged_params[f"{param_name}_err16"] = err_16
                    merged_params[f"{param_name}_err50"] = err_50
                    merged_params[f"{param_name}_err84"] = err_84

                residuals = (y_data - fe_slice) - winning_fit.best_fit
                local_rms = float(
                    np.percentile(np.abs(residuals - np.median(residuals)), 68)
                )

                for line in comp.lines:
                    cont_at_line_dict = winning_fit.eval_components(
                        x=np.array([line.rest_wavelength])
                    )
                    fe_at_line = (
                        float(np.interp(line.rest_wavelength, wave_full, feii_eval))
                        if has_feii
                        else 0.0
                    )
                    cont_at_line = (
                        float(cont_at_line_dict.get("cont_", [0.0])[0]) + fe_at_line
                    )
                    merged_params[f"{line.name}_continuum_flux"] = cont_at_line

                    amp_key = f"{line.name}_narrow_amplitude"
                    if amp_key in merged_params:
                        best_amp = merged_params[amp_key]
                        inst_sig_kms = self.spectrum.get_inst_sigma_kms(
                            line.rest_wavelength
                        )
                        sigma_inst_aa = line.rest_wavelength * (
                            inst_sig_kms / 299792.458
                        )
                        f_err_approx = local_rms * SQRT_2_PI * max(sigma_inst_aa, 0.5)

                        snr = best_amp / f_err_approx if f_err_approx > 0 else 0.0
                        merged_params[f"{line.name}_narrow_snr"] = float(snr)

                        if (
                            snr < self.local_config.fitting.snr_threshold
                            or np.isnan(snr)
                            or best_amp <= 0
                        ):
                            merged_params[amp_key] = 3.0 * f_err_approx
                            merged_params[f"{line.name}_narrow_is_upper_limit"] = True
                        else:
                            merged_params[f"{line.name}_narrow_is_upper_limit"] = False

                complex_fits.append((comp, sliced_spec, winning_fit))
                DevState.emit("post_complex_success", locals())

            except IcarusDataDimensionError:
                continue
            except IcarusConvergenceError as e:
                log.warning(
                    f"[{target.target_id}] Fit failed for {comp.name}: {e}. Computing 3-sigma limits."
                )
                _, y_raw, _ = sliced_spec.to_ndarrays()
                _, _, std_dev = sigma_clipped_stats(y_raw, sigma=3.0, maxiters=5)
                for line in comp.lines:
                    inst_sig_kms = self.spectrum.get_inst_sigma_kms(
                        line.rest_wavelength
                    )
                    sigma_inst_aa = line.rest_wavelength * (inst_sig_kms / 299792.458)
                    f_err_approx = std_dev * SQRT_2_PI * max(sigma_inst_aa, 0.5)
                    merged_params[f"{line.name}_narrow_amplitude"] = 3.0 * f_err_approx
                    merged_params[f"{line.name}_narrow_is_upper_limit"] = True
                    merged_params[f"{line.name}_narrow_snr"] = 0.0
                continue
            except Exception as e:
                log.warning(
                    f"[{target.target_id}] Unexpected error in {comp.name}: {e}"
                )
                continue

        return complex_fits, merged_params
