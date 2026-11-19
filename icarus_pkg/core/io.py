"""
io.py

Multi-Extension FITS (MEF) and Extended CSV (ECSV) output writer for ICARUS v3.4.
Implements the 3-HDU standard (Primary Header, Row-Per-Line Parameters Table,
and 1D Synthetic Model Spectrum Table) for seamless TOPCAT/Astropy interoperability.
"""

__project__ = "ICARUS"
__author__ = "Joseph Havens"
__version__ = "3.4.0"

import datetime
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.table import QTable
from astropy.utils.exceptions import AstropyWarning

from .containers import Spectrum1D
from .logger import get_icarus_logger

log = get_icarus_logger()


class ResultWriter:
    """Handles standardized 3-HDU FITS and flat ECSV serialization."""

    @staticmethod
    def build_line_catalog_table(
        spectrum: Spectrum1D,
        complex_fits: List[Any],
        global_params: Dict[str, Any],
    ) -> QTable:
        """
        Builds the HDU 1 ('LINE_CATALOG') row-per-line QTable (~50 rows x 23 columns).
        Eliminates the 999-column FITS BINTABLE limit while preserving all TUNIT keys.
        """
        scale = spectrum.flux_scale_factor
        rows = []

        for comp, sliced_spec, fit_result in complex_fits:
            redchi = (
                float(fit_result.redchi)
                if (fit_result is not None and fit_result.redchi is not None)
                else np.nan
            )
            for line in comp.lines:
                name = line.name
                rest_w = float(line.rest_wavelength)
                inst_sig = spectrum.get_inst_sigma_kms(rest_w)

                # Narrow component metrics
                n_amp = (
                    float(global_params.get(f"{name}_narrow_amplitude", np.nan)) * scale
                )
                n_e16 = (
                    float(global_params.get(f"{name}_narrow_amplitude_err16", np.nan))
                    * scale
                )
                n_e84 = (
                    float(global_params.get(f"{name}_narrow_amplitude_err84", np.nan))
                    * scale
                )
                n_voff = float(global_params.get(f"{name}_narrow_v_offset", 0.0))
                n_sig_obs = float(global_params.get(f"{name}_narrow_sigma_v", np.nan))
                n_sig_int = (
                    float(np.sqrt(max(n_sig_obs**2 - inst_sig**2, 0.0)))
                    if np.isfinite(n_sig_obs)
                    else np.nan
                )

                # Broad component metrics
                b_amp = float(global_params.get(f"{name}_broad_amplitude", 0.0)) * scale
                b_e16 = (
                    float(global_params.get(f"{name}_broad_amplitude_err16", np.nan))
                    * scale
                )
                b_e84 = (
                    float(global_params.get(f"{name}_broad_amplitude_err84", np.nan))
                    * scale
                )
                b_voff = float(global_params.get(f"{name}_broad_v_offset", np.nan))
                b_sig_obs = float(global_params.get(f"{name}_broad_sigma_v", np.nan))
                b_sig_int = (
                    float(np.sqrt(max(b_sig_obs**2 - inst_sig**2, 0.0)))
                    if np.isfinite(b_sig_obs)
                    else np.nan
                )

                # Outflow component metrics
                o_amp = (
                    float(global_params.get(f"{name}_outflow_amplitude", 0.0)) * scale
                )
                o_voff = float(global_params.get(f"{name}_outflow_v_offset", np.nan))
                o_sig_obs = float(global_params.get(f"{name}_outflow_sigma_v", np.nan))

                cont_flux = (
                    float(global_params.get(f"{name}_continuum_flux", 0.0)) * scale
                )
                ew_narrow = float(global_params.get(f"{name}_narrow_EW", np.nan))
                ew_broad = float(global_params.get(f"{name}_broad_EW", np.nan))
                snr = float(global_params.get(f"{name}_narrow_snr", 0.0))
                is_ul = bool(global_params.get(f"{name}_narrow_is_upper_limit", True))

                rows.append(
                    {
                        "LINE_NAME": name,
                        "REST_WAVE": rest_w,
                        "COMPLEX": comp.name,
                        "AMP_NARROW": n_amp,
                        "AMP_NARROW_ERR16": n_e16,
                        "AMP_NARROW_ERR84": n_e84,
                        "V_OFFSET_NARROW": n_voff,
                        "SIGMA_V_NARROW_OBS": n_sig_obs,
                        "SIGMA_V_NARROW_INT": n_sig_int,
                        "AMP_BROAD": b_amp,
                        "AMP_BROAD_ERR16": b_e16,
                        "AMP_BROAD_ERR84": b_e84,
                        "V_OFFSET_BROAD": b_voff,
                        "SIGMA_V_BROAD_INT": b_sig_int,
                        "AMP_OUTFLOW": o_amp,
                        "V_OFFSET_OUTFLOW": o_voff,
                        "SIGMA_V_OUTFLOW": o_sig_obs,
                        "CONT_FLUX_DENS": cont_flux,
                        "EW_NARROW": ew_narrow,
                        "EW_BROAD": ew_broad,
                        "SNR_NARROW": snr,
                        "IS_UPPER_LIMIT": is_ul,
                        "COMPLEX_REDCHI": redchi,
                    }
                )

        if not rows:
            tbl = QTable(
                names=(
                    "LINE_NAME",
                    "REST_WAVE",
                    "COMPLEX",
                    "AMP_NARROW",
                    "SNR_NARROW",
                    "IS_UPPER_LIMIT",
                ),
                dtype=("U32", "f8", "U64", "f8", "f8", "bool"),
            )
            tbl.meta["EXTNAME"] = "LINE_CATALOG"
            return tbl

        tbl = QTable(rows=rows)
        int_flux_unit = u.Unit("erg / (cm2 s)")
        flux_dens_unit = u.Unit("erg / (cm2 s AA)")
        vel_unit = u.km / u.s

        tbl["REST_WAVE"].unit = u.AA
        for col in (
            "AMP_NARROW",
            "AMP_NARROW_ERR16",
            "AMP_NARROW_ERR84",
            "AMP_BROAD",
            "AMP_BROAD_ERR16",
            "AMP_BROAD_ERR84",
            "AMP_OUTFLOW",
        ):
            tbl[col].unit = int_flux_unit

        for col in (
            "V_OFFSET_NARROW",
            "SIGMA_V_NARROW_OBS",
            "SIGMA_V_NARROW_INT",
            "V_OFFSET_BROAD",
            "SIGMA_V_BROAD_INT",
            "V_OFFSET_OUTFLOW",
            "SIGMA_V_OUTFLOW",
        ):
            tbl[col].unit = vel_unit

        tbl["CONT_FLUX_DENS"].unit = flux_dens_unit
        tbl["EW_NARROW"].unit = u.AA
        tbl["EW_BROAD"].unit = u.AA
        tbl.meta["EXTNAME"] = "LINE_CATALOG"
        return tbl

    @staticmethod
    def build_synthetic_spectrum_table(
        spectrum: Spectrum1D,
        continuum_eval: Optional[np.ndarray],
        feii_eval: Optional[np.ndarray],
        complex_fits: List[Any],
    ) -> QTable:
        """
        Builds the HDU 2 ('SYNTH_SPEC') 1D table preserving rest-frame wavelengths,
        observed data, decomposed physical components, and normalized pulls in CGS.
        """
        wave, flux, err = spectrum.to_ndarrays()
        scale = spectrum.flux_scale_factor
        n_pix = len(wave)

        cont_pl = (
            np.asarray(continuum_eval, dtype=np.float64)
            if (continuum_eval is not None and len(continuum_eval) == n_pix)
            else np.zeros(n_pix, dtype=np.float64)
        )
        cont_feii = (
            np.asarray(feii_eval, dtype=np.float64)
            if (feii_eval is not None and len(feii_eval) == n_pix)
            else np.zeros(n_pix, dtype=np.float64)
        )

        comp_local_cont = np.copy(cont_pl)
        comp_narrow = np.zeros(n_pix, dtype=np.float64)
        comp_broad = np.zeros(n_pix, dtype=np.float64)
        comp_outflow = np.zeros(n_pix, dtype=np.float64)
        model_total = np.copy(cont_pl + cont_feii)

        for comp, sliced_spec, fit_result in complex_fits:
            if fit_result is None:
                continue
            x_c, _, _ = sliced_spec.to_ndarrays()
            idx_mask = (wave >= np.min(x_c)) & (wave <= np.max(x_c)) & (~spectrum.mask)
            if not np.any(idx_mask):
                continue

            x_sub = wave[idx_mask]
            eval_comps = fit_result.eval_components(x=x_sub)
            c_loc = eval_comps.get("cont_", np.zeros_like(x_sub))
            comp_local_cont[idx_mask] = c_loc

            n_sum = np.zeros_like(x_sub)
            b_sum = np.zeros_like(x_sub)
            o_sum = np.zeros_like(x_sub)

            for cname, cdata in eval_comps.items():
                if "narrow" in cname:
                    n_sum += cdata
                elif "broad" in cname:
                    b_sum += cdata
                elif "outflow" in cname:
                    o_sum += cdata

            comp_narrow[idx_mask] = n_sum
            comp_broad[idx_mask] = b_sum
            comp_outflow[idx_mask] = o_sum
            model_total[idx_mask] = fit_result.eval(x=x_sub) + cont_feii[idx_mask]

        safe_err = np.where(err > 0, err, np.nan)
        norm_pull = np.where(~spectrum.mask, (flux - model_total) / safe_err, np.nan)

        flux_dens_unit = u.Unit("erg / (cm2 s AA)")
        synth_tbl = QTable()
        synth_tbl["WAVE_REST"] = wave * u.AA
        synth_tbl["FLUX_OBS"] = (flux * scale) * flux_dens_unit
        synth_tbl["FLUX_ERR"] = (err * scale) * flux_dens_unit
        synth_tbl["MASK"] = np.asarray(spectrum.mask, dtype=bool)
        synth_tbl["CONT_POWERLAW"] = (cont_pl * scale) * flux_dens_unit
        synth_tbl["CONT_FEII"] = (cont_feii * scale) * flux_dens_unit
        synth_tbl["CONT_LOCAL"] = (comp_local_cont * scale) * flux_dens_unit
        synth_tbl["COMP_NARROW"] = (comp_narrow * scale) * flux_dens_unit
        synth_tbl["COMP_BROAD"] = (comp_broad * scale) * flux_dens_unit
        synth_tbl["COMP_OUTFLOW"] = (comp_outflow * scale) * flux_dens_unit
        synth_tbl["MODEL_TOTAL"] = (model_total * scale) * flux_dens_unit
        synth_tbl["NORM_PULL"] = norm_pull
        synth_tbl.meta["EXTNAME"] = "SYNTH_SPEC"
        return synth_tbl

    @staticmethod
    def write_mef_fits(result: Any, output_path: Union[str, Path]) -> Path:
        """
        Writes the 3-HDU Multi-Extension FITS file:
          HDU 0: Primary Header (provenance, cosmology, demographics, CONF_FLG bitmask)
          HDU 1: LINE_CATALOG (row-per-line parameters QTable)
          HDU 2: SYNTH_SPEC (1D synthetic model arrays QTable)
        """
        out_path = Path(output_path)
        primary_hdu = fits.PrimaryHDU()
        hdr = primary_hdu.header

        m = result.metrics or {}
        hdr["PIPELINE"] = ("ICARUS v3.4.0", "Spectroscopic minimization engine")
        hdr["AUTHOR"] = ("Joseph Havens", "University of Kansas")
        hdr["DATE_RUN"] = (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "UTC execution timestamp",
        )
        hdr["TARGETID"] = (str(result.spectrum.target.target_id), "Target identifier")
        hdr["Z_SYS"] = (float(result.spectrum.target.redshift), "Systemic redshift")
        hdr["SCL_FAC"] = (
            float(result.spectrum.flux_scale_factor),
            "Internal O(1) flux normalization factor",
        )
        hdr["CONF_FLG"] = (
            int(result.confidence_flag),
            "Bitmask confidence flag (0=Clean)",
        )

        def _set_safe_float(key: str, val: Any, comment: str):
            try:
                if val is not None and np.isfinite(float(val)) and float(val) > 0:
                    hdr[key] = (round(float(val), 4), comment)
            except (TypeError, ValueError):
                pass

        if (
            "M_BH_Msun" in m
            and np.isfinite(m.get("M_BH_Msun", np.nan))
            and m["M_BH_Msun"] > 0
        ):
            hdr["LOG_MBH"] = (
                round(float(np.log10(m["M_BH_Msun"])), 3),
                "log10(M_BH / M_sun)",
            )
            hdr["MBH_CAL"] = (
                str(m.get("M_BH_Calibrator", "None"))[:16],
                "Virial mass calibrator",
            )

        if (
            "L_bol_erg_s" in m
            and np.isfinite(m.get("L_bol_erg_s", np.nan))
            and m["L_bol_erg_s"] > 0
        ):
            hdr["LOG_LBOL"] = (
                round(float(np.log10(m["L_bol_erg_s"])), 3),
                "log10(L_bol / erg s^-1)",
            )

        _set_safe_float("LAM_EDD", m.get("lambda_Edd"), "Eddington ratio L_bol / L_Edd")
        _set_safe_float("R_FEII", m.get("R_FeII"), "Optical Fe II / H_beta_broad ratio")
        _set_safe_float(
            "SFR_VAL",
            m.get("SFR_uncorrected_Msun_yr"),
            "Star formation rate [M_sun/yr]",
        )

        line_tbl = (
            result.line_table
            if result.line_table is not None
            else ResultWriter.build_line_catalog_table(
                result.spectrum, result.complex_fits, result.raw_params
            )
        )
        synth_tbl = (
            result.synth_table
            if result.synth_table is not None
            else ResultWriter.build_synthetic_spectrum_table(
                result.spectrum,
                result.continuum_eval,
                result.feii_eval,
                result.complex_fits,
            )
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=AstropyWarning)
            hdu_lines = fits.table_to_hdu(line_tbl)
            hdu_lines.name = "LINE_CATALOG"

            hdu_synth = fits.table_to_hdu(synth_tbl)
            hdu_synth.name = "SYNTH_SPEC"

            hdul = fits.HDUList([primary_hdu, hdu_lines, hdu_synth])
            hdul.writeto(out_path, overwrite=True)

        log.info(
            f"[{result.spectrum.target.target_id}] Wrote 3-HDU MEF FITS to {out_path.name}"
        )
        return out_path

    @staticmethod
    def write_flat_ecsv(result: Any, output_path: Union[str, Path]) -> Path:
        """
        Writes a 1-row Extended CSV (ECSV) catalog containing all merged line
        parameters and derived demographics for survey-wide cross-matching.
        """
        out_path = Path(output_path)
        out_table = QTable()

        scale = result.spectrum.flux_scale_factor
        integrated_flux_unit = u.Unit("erg / (cm2 s)")
        source_dict = result.raw_params if result.raw_params else result.metrics

        out_table.meta["EXTNAME"] = "ICARUS_CATALOG"
        out_table.meta["TARGET"] = result.spectrum.target.target_id
        out_table.meta["REDSHIFT"] = result.spectrum.target.redshift
        out_table.meta["PIPELINE"] = "ICARUS v3.4.0"
        out_table.meta["CONF_FLG"] = int(result.confidence_flag)

        for key, value in source_dict.items():
            if (
                "TargetID" in key
                or "Redshift" in key
                or "Indicator" in key
                or "Calibrator" in key
                or isinstance(value, (bool, str))
            ):
                out_table[key] = [value]
            elif "amplitude" in key or "erg_s" in key or "Msun" in key or "yr" in key:
                true_val = value * scale if "amplitude" in key else value
                if "amplitude" in key:
                    out_table[key] = [true_val] * integrated_flux_unit
                elif "erg_s" in key:
                    out_table[key] = [true_val] * (u.erg / u.s)
                elif "yr" in key:
                    out_table[key] = [true_val] * (u.M_sun / u.yr)
                elif "Msun" in key:
                    out_table[key] = [true_val] * u.M_sun
            else:
                out_table[key] = [value]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=AstropyWarning)
            out_table.write(out_path, format="ascii.ecsv", overwrite=True)

        return out_path
