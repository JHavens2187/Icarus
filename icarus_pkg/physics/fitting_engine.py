"""
fitting_engine.py

Core complex grouping, spectral slicing, quantum doublet tying, and Bayesian
Information Criterion (BIC) optimization engine for ICARUS v3.4.
"""

__project__ = "ICARUS"
__author__ = "Joseph Havens"
__version__ = "3.4.0"

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from astropy import units as u
from astropy.stats import sigma_clipped_stats
from lmfit.model import ModelResult

from ..core.config import FittingConfig
from ..core.containers import EmissionLineDef, LineComplex, Spectrum1D
from ..core.dev_tools import DevState, dev_hook
from ..core.exceptions import IcarusConvergenceError, IcarusDataDimensionError
from ..core.logger import get_icarus_logger
from .continuum import ContinuumFitter
from .line_models import LineComponentFactory

log = get_icarus_logger()
SQRT_2_PI = np.sqrt(2.0 * np.pi)
C_KMS = 299792.458


class SpectralPreparer:
    """Handles dynamic wavelength proximity grouping and boundary-safe complex slicing."""

    @staticmethod
    @dev_hook(step_name="group_into_complexes")
    def group_into_complexes(
        lines: List[EmissionLineDef],
        config: FittingConfig,
        narrow_merge_radius: float = 25.0,
        broad_merge_radius: float = 90.0,
    ) -> List[LineComplex]:
        if not lines:
            return []

        sorted_lines = sorted(lines, key=lambda x: x.rest_wavelength)
        complexes: List[LineComplex] = []
        current_group = [sorted_lines[0]]

        current_end = sorted_lines[0].rest_wavelength + (
            broad_merge_radius if sorted_lines[0].test_broad else narrow_merge_radius
        )

        # Cap single-side window padding to 150 A so adjacent complexes (e.g. [S II] 6716)
        # do not sit inside the H_alpha + [N II] fitting slice
        raw_pad = (
            (config.cont_width.to(u.AA).value / 2.0)
            if config.cont_width is not None
            else 120.0
        )
        pad = min(float(raw_pad), 150.0)

        for line in sorted_lines[1:]:
            line_start = line.rest_wavelength - (
                broad_merge_radius if line.test_broad else narrow_merge_radius
            )
            line_end = line.rest_wavelength + (
                broad_merge_radius if line.test_broad else narrow_merge_radius
            )

            if line_start <= current_end:
                current_group.append(line)
                current_end = max(current_end, line_end)
            else:
                complexes.append(
                    LineComplex(
                        lines=current_group,
                        window_start=(current_group[0].rest_wavelength - pad) * u.AA,
                        window_end=(current_group[-1].rest_wavelength + pad) * u.AA,
                    )
                )
                current_group = [line]
                current_end = line_end

        complexes.append(
            LineComplex(
                lines=current_group,
                window_start=(current_group[0].rest_wavelength - pad) * u.AA,
                window_end=(current_group[-1].rest_wavelength + pad) * u.AA,
            )
        )
        return complexes

    @staticmethod
    @dev_hook(step_name="slice_spectrum")
    def slice_spectrum(spectrum: Spectrum1D, comp: LineComplex) -> Spectrum1D:
        start = comp.window_start.to(spectrum.wavelength.unit).value
        end = comp.window_end.to(spectrum.wavelength.unit).value

        wave_val = spectrum.wavelength.value
        valid_mask = (
            (~spectrum.mask) & np.isfinite(wave_val) & np.isfinite(spectrum.flux.value)
        )
        window_mask = (wave_val >= start) & (wave_val <= end) & valid_mask

        if np.sum(window_mask) < 8:
            raise IcarusDataDimensionError(
                f"Complex {comp.name} falls outside valid data range or has insufficient unmasked pixels."
            )

        sliced_wave = wave_val[window_mask]

        # Prune individual lines whose core (+/- 12 A) sits on NaNs or off the detector
        active_lines = [
            line
            for line in comp.lines
            if np.sum(np.abs(sliced_wave - line.rest_wavelength) <= 12.0) >= 2
        ]

        if not active_lines:
            raise IcarusDataDimensionError(
                f"All emission line cores in {comp.name} fall on masked/NaN regions or off detector."
            )

        comp.lines = active_lines
        min_line_w = min(l.rest_wavelength for l in active_lines)
        max_line_w = max(l.rest_wavelength for l in active_lines)

        has_left_cont = np.min(sliced_wave) <= (min_line_w - 25.0)
        has_right_cont = np.max(sliced_wave) >= (max_line_w + 25.0)
        span_ratio = (np.max(sliced_wave) - np.min(sliced_wave)) / max(end - start, 1.0)
        comp.in_bounds = bool(has_left_cont and has_right_cont and (span_ratio >= 0.60))

        sliced_res = (
            spectrum.resolution_kms[window_mask]
            if spectrum.resolution_kms is not None
            else None
        )

        DevState.emit("post_slice_spectrum", locals())

        return Spectrum1D(
            wavelength=spectrum.wavelength[window_mask],
            flux=spectrum.flux[window_mask],
            uncertainty=spectrum.uncertainty[window_mask],
            mask=np.zeros(np.sum(window_mask), dtype=bool),
            target=spectrum.target,
            spectral_unit=spectrum.spectral_unit,
            flux_unit=spectrum.flux_unit,
            flux_scale_factor=spectrum.flux_scale_factor,
            resolution_kms=sliced_res,
            meta=spectrum.meta,
            history=spectrum.history + [f"Sliced to complex {comp.name}"],
        )

    @staticmethod
    @dev_hook(step_name="estimate_initial_amplitude")
    def estimate_initial_amplitude(
        sliced_spec: Spectrum1D,
        rest_wavelength: float,
        search_window_aa: float = 12.0,
        cont_eval: Optional[np.ndarray] = None,
    ) -> float:
        wave_vals, flux_vals, _ = sliced_spec.to_ndarrays()

        if cont_eval is not None and len(cont_eval) == len(flux_vals):
            sub_flux = flux_vals - cont_eval
        else:
            _, median_cont, _ = sigma_clipped_stats(flux_vals, sigma=2.5, maxiters=5)
            sub_flux = flux_vals - median_cont

        # Search within +/- 600 km/s (or at least search_window_aa) for the line peak
        vel_win_aa = max(search_window_aa, rest_wavelength * (600.0 / C_KMS))
        core_mask = (wave_vals >= rest_wavelength - vel_win_aa) & (
            wave_vals <= rest_wavelength + vel_win_aa
        )

        if not np.any(core_mask):
            return 1e-15

        peak_height = max(float(np.nanmax(sub_flux[core_mask])), 1e-5)
        inst_sig_kms = sliced_spec.get_inst_sigma_kms(rest_wavelength)
        rough_sigma_guess = max(rest_wavelength * (inst_sig_kms / C_KMS), 1.5)
        amplitude_guess = peak_height * rough_sigma_guess * SQRT_2_PI

        DevState.emit("post_amplitude_estimation", locals())
        return float(amplitude_guess)

    @staticmethod
    def estimate_peak_v_offset_kms(
        sliced_spec: Spectrum1D,
        rest_wavelength: float,
        cont_eval: Optional[np.ndarray] = None,
        max_search_kms: float = 550.0,
    ) -> float:
        """Finds the empirical velocity offset [km/s] of the highest pixel near `rest_wavelength`."""
        wave_vals, flux_vals, _ = sliced_spec.to_ndarrays()
        sub_flux = (
            flux_vals - cont_eval
            if (cont_eval is not None and len(cont_eval) == len(flux_vals))
            else flux_vals - np.nanmedian(flux_vals)
        )
        win_aa = rest_wavelength * (max_search_kms / C_KMS)
        mask = np.abs(wave_vals - rest_wavelength) <= win_aa
        if np.sum(mask) < 2:
            return 0.0
        sub_w = wave_vals[mask]
        sub_f = sub_flux[mask]
        if not np.any(np.isfinite(sub_f)):
            return 0.0
        peak_w = float(sub_w[int(np.nanargmax(sub_f))])
        return float(C_KMS * (peak_w - rest_wavelength) / rest_wavelength)


class CoreFitter:
    """Builds constrained complex models and executes BIC broad-line selection."""

    @staticmethod
    @dev_hook(step_name="build_complex_model")
    def build_complex_model(
        comp: LineComplex,
        sliced_spec: Spectrum1D,
        config: FittingConfig,
        include_broad: bool = False,
        include_outflow: bool = False,
        continuum: Any = None,
        seed_result: Optional[ModelResult] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Any]:
        x_vals, _, _ = sliced_spec.to_ndarrays()

        # 1. Stage 2 Local Boundary-Condition Continuum Pinning
        cont_comp = ContinuumFitter.build_local_boundary_continuum(
            sliced_spec=sliced_spec, lines=comp.lines, prefix="cont_"
        )
        model = cont_comp.model
        params = cont_comp.params.copy()
        cont_init_eval = model.eval(params, x=x_vals)

        # Estimate empirical systemic velocity shift from the Complex Anchor Line
        anchor_line_def = comp.primary_line
        v_anchor_init = SpectralPreparer.estimate_peak_v_offset_kms(
            sliced_spec, anchor_line_def.rest_wavelength, cont_eval=cont_init_eval
        )
        if seed_result is not None:
            seed_v_par = seed_result.params.get(
                f"{anchor_line_def.name}_narrow_v_offset"
            )
            if seed_v_par is not None:
                v_anchor_init = float(seed_v_par.value)

        # 2. Iteratively construct narrow, broad, and outflow components
        built_lines: List[EmissionLineDef] = []
        for line in comp.lines:
            if len(sliced_spec.wavelength) < len(model.param_names) + 4:
                continue

            amp_guess = SpectralPreparer.estimate_initial_amplitude(
                sliced_spec, line.rest_wavelength, cont_eval=cont_init_eval
            )
            inst_sig = sliced_spec.get_inst_sigma_kms(line.rest_wavelength)

            n_comp = LineComponentFactory.build_emission_line(
                name=f"{line.name}_narrow",
                rest_wavelength=line.rest_wavelength,
                is_broad=False,
                amplitude_guess=amp_guess,
                config=config,
                sigma_init_kms=max(inst_sig, 80.0),
                v_init_kms=v_anchor_init,
            )
            model += n_comp.model
            params.update(n_comp.params)
            built_lines.append(line)

            # If seeding from narrow-only fit (`seed_result`), anchor the narrow core
            # so a broad component cannot swallow the narrow core peak!
            if seed_result is not None and include_broad:
                s_amp = seed_result.params.get(f"{line.name}_narrow_amplitude")
                s_sig = seed_result.params.get(f"{line.name}_narrow_sigma_v")
                s_voff = seed_result.params.get(f"{line.name}_narrow_v_offset")
                if s_amp is not None and s_amp.value > 0:
                    params[f"{line.name}_narrow_amplitude"].set(
                        value=float(s_amp.value) * 0.85,
                        min=float(s_amp.value) * 0.45,
                        max=float(s_amp.value) * 1.35,
                    )
                if s_sig is not None and s_sig.value > 0:
                    params[f"{line.name}_narrow_sigma_v"].set(
                        value=float(s_sig.value),
                        min=max(30.0, float(s_sig.value) * 0.7),
                        max=min(450.0, float(s_sig.value) * 1.3),
                    )
                if s_voff is not None:
                    params[f"{line.name}_narrow_v_offset"].set(
                        value=float(s_voff.value),
                        min=max(-600.0, float(s_voff.value) - 100.0),
                        max=min(600.0, float(s_voff.value) + 100.0),
                    )

            if include_broad and line.test_broad:
                b_comp = LineComponentFactory.build_emission_line(
                    name=f"{line.name}_broad",
                    rest_wavelength=line.rest_wavelength,
                    is_broad=True,
                    amplitude_guess=amp_guess * 0.35,
                    config=config,
                    v_init_kms=v_anchor_init,
                )
                model += b_comp.model
                params.update(b_comp.params)

            if include_outflow and (
                line.outflow_candidate or line.name in ("O_III_4959", "O_III_5007")
            ):
                outflow_comp = LineComponentFactory.build_outflow_line(
                    name=f"{line.name}_outflow",
                    rest_wavelength=line.rest_wavelength,
                    amplitude_guess=amp_guess * 0.2,
                    v_sys_kms=v_anchor_init,
                )
                model += outflow_comp.model
                params.update(outflow_comp.params)

        # 3. Quantum Mechanical Doublet Tying Matrix
        doublet_slaves = set()
        if config.doublet_tying:
            for _, definition in config.doublet_tying.items():
                target_lines = definition.get("lines", [])
                if len(target_lines) != 2:
                    continue

                line_left, line_right = target_lines[0], target_lines[1]
                l_amp = f"{line_left}_narrow_amplitude"
                r_amp = f"{line_right}_narrow_amplitude"

                if l_amp in params and r_amp in params:
                    doublet_slaves.add(line_left)
                    flux_ratio = definition.get("flux_ratio")
                    if flux_ratio is not None:
                        params[l_amp].set(expr=f"{r_amp} * {flux_ratio}")

                    if definition.get("tie_kinematics", False):
                        params[f"{line_left}_narrow_sigma_v"].set(
                            expr=f"{line_right}_narrow_sigma_v"
                        )
                        params[f"{line_left}_narrow_v_offset"].set(
                            expr=f"{line_right}_narrow_v_offset"
                        )

                    l_b_amp = f"{line_left}_broad_amplitude"
                    r_b_amp = f"{line_right}_broad_amplitude"
                    if include_broad and l_b_amp in params and r_b_amp in params:
                        if flux_ratio is not None:
                            params[l_b_amp].set(expr=f"{r_b_amp} * {flux_ratio}")
                        params[f"{line_left}_broad_sigma_v"].set(
                            expr=f"{line_right}_broad_sigma_v"
                        )
                        params[f"{line_left}_broad_v_offset"].set(
                            expr=f"{line_right}_broad_v_offset"
                        )

                    l_out_amp = f"{line_left}_outflow_amplitude"
                    r_out_amp = f"{line_right}_outflow_amplitude"
                    if include_outflow and l_out_amp in params and r_out_amp in params:
                        if flux_ratio is not None:
                            params[l_out_amp].set(expr=f"{r_out_amp} * {flux_ratio}")
                        params[f"{line_left}_outflow_sigma_v"].set(
                            expr=f"{line_right}_outflow_sigma_v"
                        )
                        params[f"{line_left}_outflow_v_offset"].set(
                            expr=f"{line_right}_outflow_v_offset"
                        )

        # 4. Systemic Narrow Kinematic Coupling within a Multi-Line Complex
        if getattr(config, "couple_narrow_v_offset", True) and len(built_lines) > 1:
            anchor_line = comp.primary_line
            if anchor_line.name in doublet_slaves:
                for candidate in built_lines:
                    if candidate.name not in doublet_slaves:
                        anchor_line = candidate
                        break

            anchor_v = f"{anchor_line.name}_narrow_v_offset"
            if anchor_v in params and not params[anchor_v].expr:
                for line in built_lines:
                    if (
                        line.name != anchor_line.name
                        and line.name not in doublet_slaves
                    ):
                        v_key = f"{line.name}_narrow_v_offset"
                        if v_key in params and not params[v_key].expr:
                            params[v_key].set(expr=anchor_v)

        DevState.emit("post_model_build", locals())
        return model, params

    @staticmethod
    @dev_hook(step_name="evaluate_broad_line")
    def evaluate_broad_line(
        comp: LineComplex,
        sliced_spec: Spectrum1D,
        config: FittingConfig,
        continuum: Any = None,
        feii_eval: Optional[np.ndarray] = None,
        full_wave: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> ModelResult:
        x_data, y_raw, err_data = sliced_spec.to_ndarrays()
        weights = 1.0 / np.maximum(err_data, 1e-5)

        # Subtract Stage 1 Fe II pseudo-continuum if active over this complex window
        fe_slice = np.zeros_like(y_raw)
        if feii_eval is not None and np.nanmax(feii_eval) > 0:
            if len(feii_eval) == len(y_raw):
                fe_slice = np.asarray(feii_eval, dtype=np.float64)
            elif full_wave is not None and len(full_wave) == len(feii_eval):
                fe_slice = np.interp(x_data, full_wave, feii_eval, left=0.0, right=0.0)

        y_data = y_raw - fe_slice
        fit_spec = (
            Spectrum1D(
                wavelength=sliced_spec.wavelength,
                flux=y_data * sliced_spec.flux_unit,
                uncertainty=sliced_spec.uncertainty,
                mask=sliced_spec.mask,
                target=sliced_spec.target,
                spectral_unit=sliced_spec.spectral_unit,
                flux_unit=sliced_spec.flux_unit,
                flux_scale_factor=sliced_spec.flux_scale_factor,
                resolution_kms=sliced_spec.resolution_kms,
                meta=sliced_spec.meta,
                history=sliced_spec.history,
            )
            if np.any(fe_slice > 0)
            else sliced_spec
        )

        mod_a, pars_a = CoreFitter.build_complex_model(
            comp,
            fit_spec,
            config,
            include_broad=False,
            include_outflow=False,
            continuum=continuum,
        )

        DevState.emit("pre_narrow_fit", locals())
        result_a = mod_a.fit(y_data, pars_a, x=x_data, weights=weights, max_nfev=800)
        DevState.emit("post_narrow_fit", locals())

        res_a = y_data - result_a.best_fit
        local_rms = max(
            float(np.percentile(np.abs(res_a - np.median(res_a)), 68)), 1e-5
        )

        # Optional [O III] Outflow Wing Check (only if narrow [O III] 5007 clears 5x RMS)
        has_oiii = any(
            l.outflow_candidate or l.name in ("O_III_4959", "O_III_5007")
            for l in comp.lines
        )
        if has_oiii:
            o3_amp = result_a.params.get("O_III_5007_narrow_amplitude")
            o3_sig = result_a.params.get("O_III_5007_narrow_sigma")
            o3_v_nar = result_a.params.get("O_III_5007_narrow_v_offset")
            if (
                o3_amp
                and o3_sig
                and o3_sig.value > 0
                and (o3_amp.value / (o3_sig.value * SQRT_2_PI)) >= 5.0 * local_rms
            ):
                mod_out, pars_out = CoreFitter.build_complex_model(
                    comp,
                    fit_spec,
                    config,
                    include_broad=False,
                    include_outflow=True,
                    continuum=continuum,
                    seed_result=result_a,
                )
                res_out = mod_out.fit(
                    y_data, pars_out, x=x_data, weights=weights, max_nfev=800
                )
                o3_out_amp = res_out.params.get("O_III_5007_outflow_amplitude")
                o3_out_sig = res_out.params.get("O_III_5007_outflow_sigma")
                o3_out_v = res_out.params.get("O_III_5007_outflow_v_offset")
                v_nar_val = float(o3_v_nar.value) if o3_v_nar else 0.0
                if (
                    o3_out_amp
                    and o3_out_sig
                    and o3_out_v
                    and o3_out_sig.value > 0
                    and (o3_out_amp.value / (o3_out_sig.value * SQRT_2_PI))
                    >= 2.5 * local_rms
                    and (v_nar_val - float(o3_out_v.value)) >= 160.0
                    and res_out.bic < (result_a.bic - 10.0)
                ):
                    result_a = res_out

        if not any(line.test_broad for line in comp.lines):
            if result_a.best_fit is None or not np.all(np.isfinite(result_a.best_fit)):
                raise IcarusConvergenceError(
                    f"Narrow-only fit failed to converge for complex: {comp.name}"
                )
            return result_a

        # Speed Gatekeeper: Skip broad optimization pass if narrow core has < 2.0x RMS signal
        if getattr(config, "enable_speed_gatekeeper", True):
            has_core_signal = False
            for line in comp.lines:
                if line.test_broad:
                    n_amp = result_a.params.get(f"{line.name}_narrow_amplitude")
                    n_sig = result_a.params.get(f"{line.name}_narrow_sigma")
                    if n_amp and n_sig and n_sig.value > 0:
                        if (n_amp.value / (n_sig.value * SQRT_2_PI)) >= 2.0 * local_rms:
                            has_core_signal = True
                            break

            if not has_core_signal:
                return result_a

        has_confirmed_outflow = "O_III_5007_outflow_amplitude" in result_a.params
        mod_b, pars_b = CoreFitter.build_complex_model(
            comp,
            fit_spec,
            config,
            include_broad=True,
            include_outflow=has_confirmed_outflow,
            continuum=continuum,
            seed_result=result_a,
        )

        DevState.emit("pre_broad_fit", locals())
        result_b = mod_b.fit(y_data, pars_b, x=x_data, weights=weights, max_nfev=800)
        DevState.emit("post_broad_fit", locals())

        a_valid = result_a.best_fit is not None and np.all(
            np.isfinite(result_a.best_fit)
        )
        b_valid = result_b.best_fit is not None and np.all(
            np.isfinite(result_b.best_fit)
        )

        if not a_valid and not b_valid:
            raise IcarusConvergenceError(
                f"Both structural models failed to converge for complex: {comp.name}"
            )
        if not b_valid:
            return result_a
        if not a_valid:
            return result_b

        DevState.emit("pre_bic_evaluation", locals())

        # Physical Broad Veto Checks
        window_span = float(np.max(x_data) - np.min(x_data))
        any_valid_broad = False
        veto_broad = False

        for line in comp.lines:
            if not line.test_broad:
                continue

            b_amp = result_b.params.get(f"{line.name}_broad_amplitude")
            n_amp = result_b.params.get(f"{line.name}_narrow_amplitude")
            b_sig = result_b.params.get(f"{line.name}_broad_sigma")
            n_sig = result_b.params.get(f"{line.name}_narrow_sigma")
            b_sig_v = result_b.params.get(f"{line.name}_broad_sigma_v")
            n_sig_v = result_b.params.get(f"{line.name}_narrow_sigma_v")

            if (
                b_amp
                and n_amp
                and b_sig
                and n_sig
                and b_sig.value > 0
                and n_sig.value > 0
            ):
                b_peak = b_amp.value / (b_sig.value * SQRT_2_PI)
                n_peak = n_amp.value / (n_sig.value * SQRT_2_PI)
                b_fwhm_aa = b_sig.value * 2.3548

                # 1. Continuum-soak veto: broad FWHM > 70% of window span
                if b_fwhm_aa > 0.70 * window_span:
                    veto_broad = True
                    break

                # 2. Amplitude floor veto: broad peak < 5% of narrow peak
                if n_peak > 0 and b_peak < 0.05 * n_peak:
                    veto_broad = True
                    break

                # 3. Core-swallowing veto: broad dispersion must be >= 2.2x narrow dispersion
                if (
                    b_sig_v is not None
                    and n_sig_v is not None
                    and float(b_sig_v.value) < 2.2 * float(n_sig_v.value)
                ):
                    veto_broad = True
                    break

                # 4. Minimum SNR veto: broad peak must clear 2.5x local RMS
                if b_peak >= 2.5 * local_rms:
                    any_valid_broad = True

        if veto_broad or not any_valid_broad:
            DevState.emit("post_bic_evaluation", locals())
            return result_a

        delta_bic = result_b.bic - result_a.bic
        if delta_bic <= -10.0:
            log.info(
                f"Broad component confirmed for {comp.name} (Delta BIC: {delta_bic:.1f})"
            )
            DevState.emit("post_bic_evaluation", locals())
            return result_b

        DevState.emit("post_bic_evaluation", locals())
        return result_a

    @staticmethod
    @dev_hook(step_name="estimate_errors_boxcar")
    def estimate_errors_boxcar(
        sliced_spec: Spectrum1D, winning_result: ModelResult, boxcar_width: int = 5
    ) -> Dict[str, Tuple[float, float, float]]:
        _, base_flux, err_flux = sliced_spec.to_ndarrays()

        kernel = np.ones(boxcar_width) / float(boxcar_width)
        smoothed_flux = np.convolve(base_flux, kernel, mode="same")
        residuals = base_flux - smoothed_flux

        local_rms = float(np.sqrt(np.mean(residuals**2)))
        median_err = float(np.median(err_flux))
        scale_factor = local_rms / median_err if median_err > 0 else 1.0

        DevState.emit("mid_boxcar_calculation", locals())

        final_errors: Dict[str, Tuple[float, float, float]] = {}
        for name, param in winning_result.params.items():
            best_val = float(param.value)
            if param.stderr is not None and np.isfinite(param.stderr):
                scaled_err = float(param.stderr) * scale_factor
                final_errors[name] = (
                    best_val - scaled_err,
                    best_val,
                    best_val + scaled_err,
                )
            else:
                final_errors[name] = (np.nan, best_val, np.nan)

        DevState.emit("post_boxcar_calculation", locals())
        return final_errors
