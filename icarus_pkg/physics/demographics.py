"""
demographics.py

Single-Epoch Virial Black Hole Mass Calibrations, Coatman et al. (2017) C IV
Blueshift Corrections, Accretion Telemetry (L_bol, lambda_Edd, R_FeII),
BPT Diagnostics, Star Formation Rates, and 6-Bit Confidence Flag Evaluation
for ICARUS v3.4.
"""

__project__ = "ICARUS"
__author__ = "Joseph Havens"
__version__ = "3.4.0"

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from astropy import constants as const
from astropy import units as u

from ..core.config import IcarusConfig
from ..core.containers import Spectrum1D
from ..core.dev_tools import DevState, dev_hook
from .line_models import UV_WIND_IONS

C_KMS = const.c.to("km/s").value
MPC_TO_CM = 3.08567758e24
FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))
_TRAPZ = getattr(np, "trapezoid", getattr(np, "trapz", None))


class DemographicsCalculator:
    """Computes deconvolved AGN virial demographics, BPT ratios, and confidence bitmasks."""

    @staticmethod
    def _get_luminosity_distance_cm(redshift: float, config: IcarusConfig) -> float:
        if redshift <= 0.0:
            return np.nan
        dl_mpc = config.globals.cosmology.luminosity_distance(redshift).value
        return float(dl_mpc) * MPC_TO_CM

    @staticmethod
    def _flux_to_luminosity(flux: float, dl_cm: float, scale_factor: float) -> float:
        if np.isnan(flux) or flux <= 0.0 or np.isnan(dl_cm):
            return np.nan
        flux_cgs = float(flux) * float(scale_factor)
        return float(flux_cgs * 4.0 * np.pi * (dl_cm**2))

    @staticmethod
    def _deconvolve_fwhm_kms(sigma_obs_kms: float, inst_sigma_kms: float) -> float:
        """Subtracts local instrumental velocity dispersion in quadrature."""
        if not np.isfinite(sigma_obs_kms) or sigma_obs_kms <= 0.0:
            return np.nan
        sigma_int = np.sqrt(
            max(float(sigma_obs_kms) ** 2 - float(inst_sigma_kms) ** 2, 0.0)
        )
        return float(sigma_int * FWHM_FACTOR)

    @staticmethod
    def _eval_monochromatic_lum(
        rest_wave_aa: float,
        wave_arr: np.ndarray,
        cont_eval: Optional[np.ndarray],
        dl_cm: float,
        scale_factor: float,
    ) -> float:
        """
        Computes monochromatic continuum luminosity lambda * L_lambda(rest_wave_aa) [erg/s]
        from the Stage 1 AGN continuum array.
        """
        if (
            cont_eval is None
            or len(cont_eval) != len(wave_arr)
            or np.isnan(dl_cm)
            or rest_wave_aa < np.min(wave_arr)
            or rest_wave_aa > np.max(wave_arr)
        ):
            return np.nan

        f_lambda_norm = float(np.interp(rest_wave_aa, wave_arr, cont_eval))
        if not np.isfinite(f_lambda_norm) or f_lambda_norm <= 0.0:
            return np.nan

        f_lambda_cgs = f_lambda_norm * float(scale_factor)
        return float(4.0 * np.pi * (dl_cm**2) * rest_wave_aa * f_lambda_cgs)

    @staticmethod
    def _resolve_line(
        clean: Dict[str, Any], raw: Dict[str, Any], aliases: List[str], suffix: str
    ) -> Tuple[float, bool]:
        for alias in aliases:
            key = f"{alias}_{suffix}"
            if key in clean and clean[key] != 0.0:
                is_ul = bool(raw.get(f"{alias}_narrow_is_upper_limit", False))
                return float(clean[key]), is_ul
        return 0.0, False

    @staticmethod
    def _check_kinematic_bound_hit(
        clean: Dict[str, Any], raw: Dict[str, Any], config: IcarusConfig
    ) -> bool:
        """
        Evaluates Bit 4 (16): Checks whether any detected narrow, broad, or outflow
        component pegged within 1% of its configured velocity dispersion (sigma_v)
        or velocity offset (v_offset) boundary wall.
        """
        fit_cfg = config.globals.fitting
        n_min_sig = (
            float(u.Quantity(fit_cfg.min_sigma).to(u.km / u.s).value)
            if fit_cfg.min_sigma is not None
            else 30.0
        )
        n_max_sig = (
            float(u.Quantity(fit_cfg.max_sigma).to(u.km / u.s).value)
            if fit_cfg.max_sigma is not None
            else 450.0
        )

        bk = fit_cfg.broad_kinematics or {}
        b_min_sig = (
            float(u.Quantity(bk["min_sigma"]).to(u.km / u.s).value)
            if bk.get("min_sigma") is not None
            else 500.0
        )
        b_max_sig = (
            float(u.Quantity(bk["max_sigma"]).to(u.km / u.s).value)
            if bk.get("max_sigma") is not None
            else 8000.0
        )

        def _near_wall(
            val: float, low: float, high: float, tol_frac: float = 0.01
        ) -> bool:
            span = max(abs(high - low), 1.0)
            return (val <= low + tol_frac * span) or (val >= high - tol_frac * span)

        for key, val in clean.items():
            if not isinstance(val, (int, float)) or not np.isfinite(val):
                continue

            # 1. Narrow components (only check non-upper-limit detections)
            if key.endswith("_narrow_sigma_v"):
                ion = key.removesuffix("_narrow_sigma_v")
                is_ul = bool(raw.get(f"{ion}_narrow_is_upper_limit", False))
                amp = float(clean.get(f"{ion}_narrow_amplitude", 0.0))
                if not is_ul and amp > 0.0:
                    if _near_wall(float(val), n_min_sig, n_max_sig):
                        return True
                    v_off = float(clean.get(f"{ion}_narrow_v_offset", 0.0))
                    if _near_wall(v_off, -350.0, 350.0):
                        return True

            # 2. Broad BLR components
            elif key.endswith("_broad_sigma_v"):
                ion = key.removesuffix("_broad_sigma_v")
                amp = float(clean.get(f"{ion}_broad_amplitude", 0.0))
                if amp > 0.0:
                    if _near_wall(float(val), b_min_sig, b_max_sig):
                        return True
                    v_off = float(clean.get(f"{ion}_broad_v_offset", 0.0))
                    is_uv = any(uv in ion for uv in UV_WIND_IONS)
                    v_low = -4000.0 if is_uv else -600.0
                    v_high = 1000.0 if is_uv else 600.0
                    if _near_wall(v_off, v_low, v_high):
                        return True

            # 3. Asymmetric Outflow components
            elif key.endswith("_outflow_sigma_v"):
                ion = key.removesuffix("_outflow_sigma_v")
                amp = float(clean.get(f"{ion}_outflow_amplitude", 0.0))
                if amp > 1e-4:
                    if _near_wall(float(val), 250.0, 765.0):
                        return True
                    v_off = float(clean.get(f"{ion}_outflow_v_offset", -350.0))
                    if _near_wall(v_off, -2000.0, -50.0):
                        return True

        return False

    @classmethod
    @dev_hook(step_name="compute_all_demographics")
    def compute_all(
        cls,
        row: Dict[str, Any],
        spectrum: Spectrum1D,
        config: IcarusConfig,
        global_cont_eval: Optional[np.ndarray] = None,
        feii_eval: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[str, Any], int]:
        """
        Computes all derived AGN and host galaxy physical metrics and compiles
        the 6-bit additive `confidence_flag`:
          Bit 0 (1) : Local Continuum Fallback (Stage 1 global continuum unavailable)
          Bit 1 (2) : Fe II Template Active
          Bit 2 (4) : Broad BLR Component Confirmed
          Bit 3 (8) : [O III] Outflow Wing Confirmed
          Bit 4 (16): Parameter Hit Kinematic Boundary
          Bit 5 (32): All Lines Upper Limits (Zero 3-sigma detections)
        """
        DevState.emit("pre_astrophysics_calc", locals())

        target = spectrum.target
        scale_factor = spectrum.flux_scale_factor
        wave_arr, _, _ = spectrum.to_ndarrays()

        clean: Dict[str, Any] = {}
        for k, v in row.items():
            if k == "TargetID":
                clean[k] = str(v)
            else:
                try:
                    clean[k] = (
                        float(v) if v is not None and not np.isnan(float(v)) else 0.0
                    )
                except (ValueError, TypeError):
                    clean[k] = v

        metrics: Dict[str, Any] = {}
        dl_cm = cls._get_luminosity_distance_cm(target.redshift, config)

        # 1. Rest-Frame Equivalent Widths
        for key, val in clean.items():
            if key.endswith("_amplitude") and isinstance(val, (int, float)) and val > 0:
                line_prefix = key.replace("_amplitude", "")
                base_ion = (
                    line_prefix.replace("_narrow", "")
                    .replace("_broad", "")
                    .replace("_outflow", "")
                )
                cont_val = clean.get(
                    f"{base_ion}_continuum_flux",
                    clean.get("cont_intercept", clean.get("cont_c", 0.0)),
                )
                if isinstance(cont_val, (int, float)) and cont_val > 1e-6:
                    metrics[f"{line_prefix}_EW"] = float(val / cont_val)

        # 2. Monochromatic Continuum Luminosities (L_1350, L_3000, L_5100)
        l_1350 = cls._eval_monochromatic_lum(
            1350.0, wave_arr, global_cont_eval, dl_cm, scale_factor
        )
        l_3000 = cls._eval_monochromatic_lum(
            3000.0, wave_arr, global_cont_eval, dl_cm, scale_factor
        )
        l_5100 = cls._eval_monochromatic_lum(
            5100.0, wave_arr, global_cont_eval, dl_cm, scale_factor
        )
        metrics["L_1350_erg_s"] = l_1350
        metrics["L_3000_erg_s"] = l_3000
        metrics["L_5100_erg_s"] = l_5100

        # 3. Narrow Line Luminosities & BPT Diagnostics
        ha_n, ha_ul = cls._resolve_line(
            clean, row, ["H_alpha", "Ha"], "narrow_amplitude"
        )
        hb_n, hb_ul = cls._resolve_line(
            clean, row, ["H_beta", "Hb"], "narrow_amplitude"
        )
        pab_n, pab_ul = cls._resolve_line(
            clean, row, ["Pa_beta", "Pab"], "narrow_amplitude"
        )
        oiii_f, oiii_ul = cls._resolve_line(
            clean, row, ["O_III_5007"], "narrow_amplitude"
        )
        nii_f, nii_ul = cls._resolve_line(clean, row, ["N_II_6583"], "narrow_amplitude")
        neiii_f, neiii_ul = cls._resolve_line(
            clean, row, ["Ne_III_3869"], "narrow_amplitude"
        )
        oii_f, oii_ul = cls._resolve_line(
            clean, row, ["O_II_3729", "O_II_3726"], "narrow_amplitude"
        )

        if ha_n > 0:
            metrics["L_Ha_narrow_erg_s"] = cls._flux_to_luminosity(
                ha_n, dl_cm, scale_factor
            )
        if pab_n > 0:
            metrics["L_Pab_narrow_erg_s"] = cls._flux_to_luminosity(
                pab_n, dl_cm, scale_factor
            )

        # BPT Ratios + Upper-Limit Arrow Metadata for Panel 4
        if oiii_f > 0 and hb_n > 0 and not (oiii_ul and hb_ul):
            metrics["log_O3_Hb"] = float(np.log10(oiii_f / hb_n))
            metrics["O3_Hb_limit"] = "down" if oiii_ul else ("up" if hb_ul else "none")
        else:
            metrics["log_O3_Hb"] = np.nan
            metrics["O3_Hb_limit"] = "none"

        if nii_f > 0 and ha_n > 0 and not (nii_ul and ha_ul):
            metrics["log_N2_Ha"] = float(np.log10(nii_f / ha_n))
            metrics["N2_Ha_limit"] = (
                "left" if nii_ul else ("right" if ha_ul else "none")
            )
        else:
            metrics["log_N2_Ha"] = np.nan
            metrics["N2_Ha_limit"] = "none"

        if neiii_f > 0 and oii_f > 0 and not (neiii_ul and oii_ul):
            metrics["log_Ne3_O2"] = float(np.log10(neiii_f / oii_f))
        else:
            metrics["log_Ne3_O2"] = np.nan

        # Systemic narrow velocity anchor (for C IV blueshift calculation)
        v_sys_kms = 0.0
        for sys_ion in ("O_III_5007", "O_II_3729", "Ne_III_3869", "N_II_6583"):
            v_val, is_ul = cls._resolve_line(clean, row, [sys_ion], "narrow_v_offset")
            if not is_ul and f"{sys_ion}_narrow_v_offset" in row:
                v_sys_kms = v_val
                break

        # 4. Single-Epoch Virial Black Hole Mass Waterfall
        # Hierarchy: H_beta (VP06) -> H_alpha (GH05) -> Mg_II (VO09) -> C_IV (Coatman17) -> Pa_beta
        hb_b_f, _ = cls._resolve_line(clean, row, ["H_beta", "Hb"], "broad_amplitude")
        hb_b_sv, _ = cls._resolve_line(clean, row, ["H_beta", "Hb"], "broad_sigma_v")

        ha_b_f, _ = cls._resolve_line(clean, row, ["H_alpha", "Ha"], "broad_amplitude")
        ha_b_sv, _ = cls._resolve_line(clean, row, ["H_alpha", "Ha"], "broad_sigma_v")

        mg_b_f, _ = cls._resolve_line(clean, row, ["Mg_II_2796"], "broad_amplitude")
        mg_b_sv, _ = cls._resolve_line(clean, row, ["Mg_II_2796"], "broad_sigma_v")

        civ_b_f, _ = cls._resolve_line(clean, row, ["C_IV_1548"], "broad_amplitude")
        civ_b_sv, _ = cls._resolve_line(clean, row, ["C_IV_1548"], "broad_sigma_v")
        civ_b_voff, _ = cls._resolve_line(clean, row, ["C_IV_1548"], "broad_v_offset")

        pab_b_f, _ = cls._resolve_line(
            clean, row, ["Pa_beta", "Pab"], "broad_amplitude"
        )
        pab_b_sv, _ = cls._resolve_line(clean, row, ["Pa_beta", "Pab"], "broad_sigma_v")

        mbh_msun = np.nan
        lbol_erg_s = np.nan
        fwhm_broad_kms = np.nan
        mbh_cal = "None"

        if hb_b_f > 0 and hb_b_sv > 0:
            l_hb_b = cls._flux_to_luminosity(hb_b_f, dl_cm, scale_factor)
            metrics["L_Hb_broad_erg_s"] = l_hb_b
            fwhm_hb = cls._deconvolve_fwhm_kms(
                hb_b_sv, spectrum.get_inst_sigma_kms(4862.68)
            )
            fwhm_broad_kms = fwhm_hb
            l5100_eff = (
                l_5100 if (np.isfinite(l_5100) and l_5100 > 0) else (l_hb_b * 113.0)
            )
            lbol_erg_s = l5100_eff * 9.26
            mbh_msun = (
                (10.0**6.91)
                * ((fwhm_hb / 1000.0) ** 2.0)
                * ((l5100_eff / 1e44) ** 0.50)
            )
            mbh_cal = "Hbeta_VP06"

        elif ha_b_f > 0 and ha_b_sv > 0:
            l_ha_b = cls._flux_to_luminosity(ha_b_f, dl_cm, scale_factor)
            metrics["L_Ha_broad_erg_s"] = l_ha_b
            fwhm_ha = cls._deconvolve_fwhm_kms(
                ha_b_sv, spectrum.get_inst_sigma_kms(6564.61)
            )
            fwhm_broad_kms = fwhm_ha
            lbol_erg_s = (
                l_5100 * 9.26
                if (np.isfinite(l_5100) and l_5100 > 0)
                else l_ha_b * 130.0
            )
            mbh_msun = 2.0e6 * ((l_ha_b / 1e42) ** 0.55) * ((fwhm_ha / 1000.0) ** 2.06)
            mbh_cal = "Halpha_GH05"

        elif mg_b_f > 0 and mg_b_sv > 0:
            l_mg_b = cls._flux_to_luminosity(mg_b_f, dl_cm, scale_factor)
            fwhm_mg = cls._deconvolve_fwhm_kms(
                mg_b_sv, spectrum.get_inst_sigma_kms(2796.35)
            )
            fwhm_broad_kms = fwhm_mg
            l3000_eff = (
                l_3000 if (np.isfinite(l_3000) and l_3000 > 0) else (l_mg_b * 55.0)
            )
            lbol_erg_s = l3000_eff * 5.15
            mbh_msun = (
                (10.0**6.86)
                * ((fwhm_mg / 1000.0) ** 2.0)
                * ((l3000_eff / 1e44) ** 0.50)
            )
            mbh_cal = "MgII_VO09"

        elif civ_b_f > 0 and civ_b_sv > 0:
            l_civ_b = cls._flux_to_luminosity(civ_b_f, dl_cm, scale_factor)
            fwhm_civ_meas = cls._deconvolve_fwhm_kms(
                civ_b_sv, spectrum.get_inst_sigma_kms(1548.19)
            )
            # Coatman et al. (2017) non-virial blueshift correction
            dv_blue_kms = max(float(v_sys_kms - civ_b_voff), 0.0)
            metrics["CIV_blueshift_kms"] = dv_blue_kms
            if dv_blue_kms > 0.0:
                fwhm_civ_corr = fwhm_civ_meas / (0.36 * (dv_blue_kms / 1000.0) + 0.61)
            else:
                fwhm_civ_corr = fwhm_civ_meas

            fwhm_broad_kms = fwhm_civ_corr
            l1350_eff = (
                l_1350 if (np.isfinite(l_1350) and l_1350 > 0) else (l_civ_b * 35.0)
            )
            lbol_erg_s = l1350_eff * 3.81
            mbh_msun = (
                (10.0**6.66)
                * ((fwhm_civ_corr / 1000.0) ** 2.0)
                * ((l1350_eff / 1e44) ** 0.53)
            )
            mbh_cal = "CIV_Coatman17"

        elif pab_b_f > 0 and pab_b_sv > 0:
            l_pab_b = cls._flux_to_luminosity(pab_b_f, dl_cm, scale_factor)
            metrics["L_Pab_broad_erg_s"] = l_pab_b
            l_ha_equiv = l_pab_b * 17.5
            fwhm_pab = cls._deconvolve_fwhm_kms(
                pab_b_sv, spectrum.get_inst_sigma_kms(12821.59)
            )
            fwhm_broad_kms = fwhm_pab
            lbol_erg_s = l_ha_equiv * 130.0
            mbh_msun = (
                2.0e6 * ((l_ha_equiv / 1e42) ** 0.55) * ((fwhm_pab / 1000.0) ** 2.06)
            )
            mbh_cal = "PaBeta_CaseB"

        metrics["M_BH_Msun"] = mbh_msun
        metrics["M_BH_Calibrator"] = mbh_cal
        metrics["M_BH_sys_err_dex"] = 0.35 if np.isfinite(mbh_msun) else np.nan
        metrics["L_bol_erg_s"] = lbol_erg_s
        metrics["FWHM_broad_kms"] = fwhm_broad_kms

        if np.isfinite(mbh_msun) and mbh_msun > 0 and np.isfinite(lbol_erg_s):
            l_edd = 1.26e38 * mbh_msun
            metrics["lambda_Edd"] = float(lbol_erg_s / l_edd)
        else:
            metrics["lambda_Edd"] = np.nan

        # 5. Optical Fe II Relative Strength (R_FeII = F_FeII(4434-4684) / F_Hb_broad)
        if (
            feii_eval is not None
            and len(feii_eval) == len(wave_arr)
            and hb_b_f > 0
            and _TRAPZ is not None
        ):
            opt_fe_mask = (wave_arr >= 4434.0) & (wave_arr <= 4684.0) & (~spectrum.mask)
            if np.sum(opt_fe_mask) >= 3:
                fe_int_flux = float(
                    _TRAPZ(feii_eval[opt_fe_mask], wave_arr[opt_fe_mask])
                )
                metrics["R_FeII"] = (
                    float(fe_int_flux / hb_b_f) if fe_int_flux > 0 else 0.0
                )
            else:
                metrics["R_FeII"] = np.nan
        else:
            metrics["R_FeII"] = np.nan

        # 6. [O III] Outflow Velocity Readout
        oiii_out_amp, _ = cls._resolve_line(
            clean, row, ["O_III_5007"], "outflow_amplitude"
        )
        oiii_out_voff, _ = cls._resolve_line(
            clean, row, ["O_III_5007"], "outflow_v_offset"
        )
        metrics["v_outflow_OIII_kms"] = (
            float(oiii_out_voff) if oiii_out_amp > 1e-4 else np.nan
        )

        # 7. Narrow-Line Star Formation Rate (Kennicutt & Evans 2012)
        sfr_ha = 5.37e-42 * metrics.get("L_Ha_narrow_erg_s", 0.0)
        sfr_pab = (10.0**-40.02) * metrics.get("L_Pab_narrow_erg_s", 0.0)

        if ha_n > 0 and not ha_ul and ha_b_f == 0:
            metrics["SFR_uncorrected_Msun_yr"] = float(sfr_ha)
            metrics["SFR_Indicator"] = "Halpha"
            metrics["SFR_is_upper_limit"] = False
        elif pab_n > 0 and not pab_ul and pab_b_f == 0:
            metrics["SFR_uncorrected_Msun_yr"] = float(sfr_pab)
            metrics["SFR_Indicator"] = "PaBeta"
            metrics["SFR_is_upper_limit"] = False
        elif ha_n > 0:
            metrics["SFR_uncorrected_Msun_yr"] = float(sfr_ha)
            metrics["SFR_Indicator"] = "Halpha"
            metrics["SFR_is_upper_limit"] = True if ha_b_f > 0 else ha_ul
        elif pab_n > 0:
            metrics["SFR_uncorrected_Msun_yr"] = float(sfr_pab)
            metrics["SFR_Indicator"] = "PaBeta"
            metrics["SFR_is_upper_limit"] = True if pab_b_f > 0 else pab_ul
        else:
            metrics["SFR_uncorrected_Msun_yr"] = np.nan
            metrics["SFR_Indicator"] = "None"
            metrics["SFR_is_upper_limit"] = False

        # 8. Additive Confidence Bitmask Evaluation
        conf_flag = 0
        if global_cont_eval is None:
            conf_flag |= 1  # Bit 0: Local continuum fallback
        if feii_eval is not None and np.nanmax(feii_eval) > 0:
            conf_flag |= 2  # Bit 1: Fe II template active
        if np.isfinite(mbh_msun) and mbh_msun > 0:
            conf_flag |= 4  # Bit 2: Broad BLR component confirmed
        if oiii_out_amp > 1e-4:
            conf_flag |= 8  # Bit 3: [O III] outflow wing confirmed
        if cls._check_kinematic_bound_hit(clean, row, config):
            conf_flag |= 16  # Bit 4: Detected component pegged against kinematic bound

        ul_flags = [
            bool(v) for k, v in row.items() if k.endswith("_narrow_is_upper_limit")
        ]
        if ul_flags and all(ul_flags):
            conf_flag |= 32  # Bit 5: All lines are 3-sigma upper limits

        DevState.emit("post_astrophysics_calc", locals())
        return metrics, conf_flag
