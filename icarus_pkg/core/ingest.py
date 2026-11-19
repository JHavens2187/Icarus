"""
ingest.py

The universal translator and file crawler for the ICARUS pipeline.
Handles the ingestion of metadata catalogs and the strict unpacking, dynamic
FITS schema resolution (including SDSS loglam/ivar, custom column mappings, and
1D WCS ImageHDUs), unit conversion, instrumental resolution mapping, and O(1) scaling.
"""

__project__ = "ICARUS"
__author__ = "Joseph Havens"
__version__ = "3.4.0"

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import astropy.units as u
import numpy as np
import pandas as pd
from astropy import constants as const
from astropy.io import fits
from astropy.table import QTable
from astropy.utils.exceptions import AstropyWarning

from .containers import Spectrum1D, Target
from .dev_tools import DevState, dev_hook
from .exceptions import IcarusIngestionError
from .logger import get_icarus_logger

log = get_icarus_logger()
STANDARD_FLUX_UNIT = u.Unit("erg / (cm2 s AA)")
C_KMS = const.c.to("km/s").value
FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


class CatalogRegistry:
    def __init__(self, catalog_path: Union[str, Path]):
        self.catalog_path = Path(catalog_path)
        if not self.catalog_path.exists():
            log.critical(f"Master catalog not found at {self.catalog_path}")
            raise IcarusIngestionError(f"Master catalog missing: {self.catalog_path}")

        self.table = self._load_catalog()
        self._normalize_columns()
        log.info(
            f"Successfully loaded {len(self.table)} targets from {self.catalog_path.name}"
        )

    def _load_catalog(self) -> QTable:
        suffix = self.catalog_path.suffix.lower()
        if suffix == ".csv":
            return QTable.from_pandas(pd.read_csv(self.catalog_path))
        elif suffix in [".fits", ".fit"]:
            return QTable.read(self.catalog_path)
        return QTable.read(self.catalog_path, format="ascii")

    def _normalize_columns(self) -> None:
        col_map = {
            "target_id": [
                "id",
                "gid",
                "sourceid",
                "source_number",
                "galaxy_id",
                "target_id",
            ],
            "redshift": ["z", "redshift", "z_spec", "z_used", "z_phot"],
            "sample": ["sample", "survey"],
            "setup": ["setup", "instrument", "grating"],
        }
        existing_cols = [c.lower() for c in self.table.colnames]

        for standard_name, aliases in col_map.items():
            for alias in aliases:
                if alias in existing_cols and standard_name not in self.table.colnames:
                    original_col = self.table.colnames[existing_cols.index(alias)]
                    self.table.rename_column(original_col, standard_name)
                    break

        if (
            "target_id" not in self.table.colnames
            or "redshift" not in self.table.colnames
        ):
            raise IcarusIngestionError(
                "Catalog must contain at minimum a target ID and redshift."
            )

    def generate_targets(self) -> List[Target]:
        targets = []
        for row in self.table:
            t_id = row["target_id"]
            z = row["redshift"]

            t = Target(
                target_id=str(t_id.value if hasattr(t_id, "value") else t_id),
                redshift=float(z.value if hasattr(z, "value") else z),
                sample=str(row["sample"]) if "sample" in row.colnames else None,
                setup=str(row["setup"]) if "setup" in row.colnames else None,
            )
            targets.append(t)
        return targets


class ObservationBuilder:
    DEFAULT_ALIASES: Dict[str, List[str]] = {
        "wave": ["WAVELENGTH", "WAVE", "wave", "LAMBDA", "lambda", "lam"],
        "loglam": ["LOGLAM", "loglam", "LOG_WAVE", "log_wave"],
        "flux": ["FLUX", "flux", "SPEC", "spec", "F_LAMBDA", "f_lambda", "DATA"],
        "err": [
            "ERROR",
            "ERR",
            "err",
            "FLUX_ERROR",
            "FLUX_ERR",
            "ERR_FLUX",
            "SIGMA",
            "sigma",
            "UNCERTAINTY",
        ],
        "ivar": ["IVAR", "ivar", "INVVAR", "invvar", "IVAR_FLUX"],
        "var": ["VAR", "var", "VARIANCE", "variance"],
        "mask": ["MASK", "mask", "AND_MASK", "DQ", "dq", "FLAGS"],
    }

    def __init__(
        self,
        data_directory: Optional[Union[str, Path]] = None,
        column_mapping: Optional[Dict[str, str]] = None,
    ):
        self.data_dir = Path(data_directory) if data_directory else Path(".")
        self.column_mapping = column_mapping or {}

    def locate_file(self, target: Target) -> Optional[Path]:
        gid_str = str(target.target_id)
        if Path(gid_str).exists():
            return Path(gid_str)
        matches = list(self.data_dir.rglob(f"*{gid_str}*.fits"))

        if not matches:
            log.warning(f"No FITS file found for Target ID: {gid_str}")
            return None
        return matches[0]

    @staticmethod
    def _read_fits_table_or_wcs(filepath: Path) -> QTable:
        """
        Reads a FITS file as a QTable if a table HDU is present, or falls back
        to reconstructing WAVE, FLUX, and ERR from a 1D WCS ImageHDU.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyWarning)
            try:
                return QTable.read(filepath)
            except Exception:
                pass

        with fits.open(filepath) as hdul:
            for hdu in hdul:
                if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                    return QTable.read(hdu)

            # Fallback to 1D WCS Primary/ImageHDU
            primary = hdul[0]
            if primary.data is None:
                raise IcarusIngestionError(
                    f"No table or image data found in {filepath.name}"
                )

            data_arr = np.asarray(primary.data, dtype=np.float64)
            flux_1d = data_arr[0].flatten() if data_arr.ndim > 1 else data_arr.flatten()
            n_pix = len(flux_1d)

            hdr = primary.header
            crval = float(hdr.get("CRVAL1", 1.0))
            cdelt = float(hdr.get("CDELT1", hdr.get("CD1_1", 1.0)))
            crpix = float(hdr.get("CRPIX1", 1.0))
            pix_grid = np.arange(n_pix, dtype=np.float64) + 1.0 - crpix
            wave_1d = crval + pix_grid * cdelt

            if len(hdul) > 1 and hdul[1].data is not None:
                err_1d = np.asarray(hdul[1].data, dtype=np.float64).flatten()
            elif data_arr.ndim > 1 and data_arr.shape[0] > 1:
                err_1d = data_arr[1].flatten()
            else:
                err_1d = np.full_like(flux_1d, np.nanstd(flux_1d) * 0.1)

            tbl = QTable()
            tbl["WAVE"] = wave_1d
            tbl["FLUX"] = flux_1d
            tbl["ERR"] = err_1d
            return tbl

    @classmethod
    def _resolve_column_name(
        cls,
        data_table: QTable,
        role: str,
        active_map: Dict[str, str],
    ) -> Optional[str]:
        """
        Resolves the actual column name in `data_table` for a given `role`
        ('wave', 'loglam', 'flux', 'err', 'ivar', 'var', 'mask') using
        case-insensitive matching against user overrides first, then standard aliases.
        """
        col_lookup = {c.lower(): c for c in data_table.colnames}

        if role in active_map and active_map[role]:
            requested = active_map[role]
            if requested in data_table.colnames:
                return requested
            if requested.lower() in col_lookup:
                return col_lookup[requested.lower()]

        for candidate in cls.DEFAULT_ALIASES.get(role, []):
            if candidate in data_table.colnames:
                return candidate
            if candidate.lower() in col_lookup:
                return col_lookup[candidate.lower()]

        return None

    @staticmethod
    def _extract_and_flatten(
        data_table: QTable, col_name: str
    ) -> Tuple[np.ndarray, Optional[u.Unit]]:
        raw_col = data_table[col_name]
        if len(raw_col) == 1 and hasattr(raw_col, "shape") and len(raw_col.shape) > 1:
            data = raw_col[0]
        else:
            data = raw_col

        unit = data.unit if hasattr(data, "unit") else None
        val = np.asarray(
            data.value if hasattr(data, "value") else data, dtype=np.float64
        ).flatten()
        return val, unit

    @staticmethod
    def _compute_instrumental_sigma_kms(
        wave_obs_aa: np.ndarray, config: Any
    ) -> np.ndarray:
        """
        Evaluates instrumental velocity dispersion sigma_inst(lambda_obs) [km/s]
        at every observed wavelength pixel prior to de-redshifting.
        """
        disp_curve_path = getattr(config, "dispersion_curve", None)
        if disp_curve_path is not None:
            path_obj = Path(disp_curve_path)
            if not path_obj.exists():
                pkg_disp = (
                    Path(__file__).resolve().parent.parent
                    / "data"
                    / "dispersion"
                    / path_obj.name
                )
                if pkg_disp.exists():
                    path_obj = pkg_disp

            if path_obj.exists():
                try:
                    disp_table = QTable.read(path_obj)
                    w_col = disp_table.colnames[0]
                    r_col = (
                        "R" if "R" in disp_table.colnames else disp_table.colnames[-1]
                    )
                    w_disp = disp_table[w_col]
                    w_disp_aa = (
                        w_disp.to(u.AA).value
                        if hasattr(w_disp, "unit") and w_disp.unit is not None
                        else np.asarray(w_disp, dtype=float) * 1e4
                    )
                    r_vals = np.asarray(
                        (
                            disp_table[r_col].value
                            if hasattr(disp_table[r_col], "value")
                            else disp_table[r_col]
                        ),
                        dtype=float,
                    )
                    r_interp = np.interp(
                        wave_obs_aa, w_disp_aa, r_vals, left=r_vals[0], right=r_vals[-1]
                    )
                    return C_KMS / (FWHM_TO_SIGMA * np.maximum(r_interp, 10.0))
                except Exception as e:
                    log.warning(
                        f"Dispersion curve load failed ({path_obj.name}): {e}. Using scalar fallback."
                    )

        res_qty = getattr(config, "resolution", None)
        if res_qty is not None:
            try:
                q = u.Quantity(res_qty)
                if q.unit.is_equivalent(u.km / u.s):
                    fwhm_kms = float(q.to(u.km / u.s).value)
                    return np.full_like(
                        wave_obs_aa, fwhm_kms / FWHM_TO_SIGMA, dtype=np.float64
                    )
                elif q.unit == u.dimensionless_unscaled:
                    r_val = max(float(q.value), 10.0)
                    return np.full_like(
                        wave_obs_aa, C_KMS / (FWHM_TO_SIGMA * r_val), dtype=np.float64
                    )
            except Exception:
                pass

        return np.full_like(wave_obs_aa, 300.0 / FWHM_TO_SIGMA, dtype=np.float64)

    @dev_hook(step_name="build_spectrum")
    def build_spectrum(
        self,
        target: Target,
        filepath: Union[str, Path],
        config: Any = None,
        dry_run: bool = False,
        column_mapping: Optional[Dict[str, str]] = None,
    ) -> Optional[Spectrum1D]:
        DevState.emit("pre_build_spectrum", locals())

        filepath = Path(filepath)
        if not filepath.exists():
            raise IcarusIngestionError(f"Spectral file does not exist: {filepath}")

        if dry_run:
            log.info(f"[DRY RUN] Validated access to {filepath.name}")
            return None

        history = [f"Ingested from {filepath.name}"]

        try:
            with fits.open(filepath) as hdul:
                meta = dict(hdul[0].header)
                z_hdr = None
                for z_key in ("Z", "REDSHIFT", "Z_SPEC", "Z_VI"):
                    if z_key in meta and isinstance(meta[z_key], (int, float)):
                        z_hdr = float(meta[z_key])
                        break

            if target.redshift <= 0.0 and z_hdr is not None and z_hdr > 0.0:
                target = Target(
                    target_id=target.target_id,
                    redshift=z_hdr,
                    sample=target.sample,
                    setup=target.setup,
                    pointing=target.pointing,
                    ra=target.ra,
                    dec=target.dec,
                )
                history.append(f"Adopted header redshift z={z_hdr:.4f}")

            data_table = self._read_fits_table_or_wcs(filepath)

            cfg_map = getattr(config, "column_mapping", {}) if config else {}
            active_map = {**cfg_map, **self.column_mapping, **(column_mapping or {})}

            # 1. Resolve Wavelength or Log-Wavelength Column
            wave_col = self._resolve_column_name(data_table, "wave", active_map)
            loglam_col = self._resolve_column_name(data_table, "loglam", active_map)

            if wave_col is not None and wave_col in data_table.colnames:
                wave_vals, wave_unit = self._extract_and_flatten(data_table, wave_col)
            elif loglam_col is not None and loglam_col in data_table.colnames:
                loglam_vals, _ = self._extract_and_flatten(data_table, loglam_col)
                wave_vals = 10.0**loglam_vals
                wave_unit = u.AA
                history.append(
                    f"Converted log10 wavelength column '{loglam_col}' to AA"
                )
            else:
                raise IcarusIngestionError(
                    f"Could not locate wavelength column in {filepath.name} (Available: {data_table.colnames})"
                )

            # 2. Resolve Flux Column
            flux_col = self._resolve_column_name(data_table, "flux", active_map)
            if flux_col is None or flux_col not in data_table.colnames:
                raise IcarusIngestionError(
                    f"Could not locate flux column in {filepath.name} (Available: {data_table.colnames})"
                )
            flux_vals, flux_unit = self._extract_and_flatten(data_table, flux_col)

            # 3. Resolve Uncertainty, Inverse Variance (ivar), or Variance (var) Column
            err_col = self._resolve_column_name(data_table, "err", active_map)
            ivar_col = self._resolve_column_name(data_table, "ivar", active_map)
            var_col = self._resolve_column_name(data_table, "var", active_map)

            ivar_zero_mask = np.zeros_like(flux_vals, dtype=bool)
            if err_col is not None and err_col in data_table.colnames:
                err_vals, err_unit = self._extract_and_flatten(data_table, err_col)
            elif ivar_col is not None and ivar_col in data_table.colnames:
                ivar_vals, _ = self._extract_and_flatten(data_table, ivar_col)
                ivar_zero_mask = (~np.isfinite(ivar_vals)) | (ivar_vals <= 0)
                err_vals = np.where(
                    ~ivar_zero_mask,
                    1.0 / np.sqrt(np.maximum(ivar_vals, 1e-30)),
                    np.nan,
                )
                err_unit = flux_unit
                history.append(
                    f"Converted inverse variance column '{ivar_col}' to 1-sigma uncertainty"
                )
            elif var_col is not None and var_col in data_table.colnames:
                var_vals, _ = self._extract_and_flatten(data_table, var_col)
                ivar_zero_mask = (~np.isfinite(var_vals)) | (var_vals <= 0)
                err_vals = np.where(
                    ~ivar_zero_mask, np.sqrt(np.maximum(var_vals, 0.0)), np.nan
                )
                err_unit = flux_unit
            else:
                raise IcarusIngestionError(
                    f"Could not locate uncertainty/ivar column in {filepath.name} (Available: {data_table.colnames})"
                )

            # 4. Optional Bitmask / DQ Column
            mask_col = self._resolve_column_name(data_table, "mask", active_map)
            file_dq_mask = np.zeros_like(flux_vals, dtype=bool)
            if mask_col is not None and mask_col in data_table.colnames:
                raw_m, _ = self._extract_and_flatten(data_table, mask_col)
                file_dq_mask = raw_m != 0

            # --- EXPLICIT CONFIGURATION & PRE-SCALED FITS HANDLING ---
            try:
                fb_wave = u.Unit(getattr(config, "wave_unit", "um"))
                fb_flux = u.Unit(getattr(config, "flux_unit", "1e-20 erg / (cm2 s AA)"))

                orig_wave_unit = (
                    wave_unit
                    if (wave_unit is not None and wave_unit != u.dimensionless_unscaled)
                    else fb_wave
                )

                if (
                    flux_unit is None
                    or flux_unit == u.dimensionless_unscaled
                    or (
                        getattr(flux_unit, "scale", 1.0) == 1.0
                        and getattr(fb_flux, "scale", 1.0) != 1.0
                    )
                ):
                    orig_flux_unit = fb_flux
                    orig_err_unit = fb_flux
                else:
                    orig_flux_unit = flux_unit
                    orig_err_unit = err_unit if err_unit else flux_unit

            except Exception as e:
                log.warning(
                    f"[{target.target_id}] Fallback unit parsing failed: {e}. Coercing manually."
                )
                orig_wave_unit = wave_unit if wave_unit else u.um
                orig_flux_unit = (
                    flux_unit if flux_unit else u.Unit("1e-20 erg / (cm2 s AA)")
                )
                orig_err_unit = err_unit if err_unit else orig_flux_unit

            # --- Robust Spectral Axis Transformation (Vacuum Angstroms) ---
            wave_qty_obs = (wave_vals * orig_wave_unit).to(
                u.AA, equivalencies=u.spectral()
            )

            # --- Evaluate Instrumental Dispersion Prior to De-Redshifting ---
            inst_sigma_kms = self._compute_instrumental_sigma_kms(
                wave_qty_obs.value, config
            )

            # --- Robust Flux Axis Density Standardization ---
            flux_qty_obs = (flux_vals * orig_flux_unit).to(
                STANDARD_FLUX_UNIT,
                equivalencies=u.spectral_density(wave_qty_obs),
            )
            err_qty_obs = (err_vals * orig_err_unit).to(
                STANDARD_FLUX_UNIT,
                equivalencies=u.spectral_density(wave_qty_obs),
            )

            finite_obs = np.abs(
                flux_qty_obs.value[
                    np.isfinite(flux_qty_obs.value) & (flux_qty_obs.value != 0)
                ]
            )
            if len(finite_obs) > 0 and np.nanmedian(finite_obs) > 1e-8:
                log.info(
                    f"[{target.target_id}] Detected unscaled O(1) FITS flux array; applying 1e-20 cgs normalization."
                )
                flux_qty_obs = flux_qty_obs * 1e-20
                err_qty_obs = err_qty_obs * 1e-20

            # --- Cosmological De-Redshifting ---
            z = target.redshift
            if z > 0:
                wave_qty = wave_qty_obs / (1.0 + z)
                flux_qty = flux_qty_obs * (1.0 + z)
                err_qty = err_qty_obs * (1.0 + z)
                history.append(f"De-redshifted arrays from z={z:.4f} to rest-frame")
            else:
                wave_qty = wave_qty_obs
                flux_qty = flux_qty_obs
                err_qty = err_qty_obs

            # --- Dynamic Scale Normalization ---
            unmasked_flux = flux_qty.value[np.isfinite(flux_qty.value)]
            non_zero_flux = np.abs(unmasked_flux[unmasked_flux != 0])

            if len(non_zero_flux) == 0:
                scale_factor = 1.0
            else:
                median_flux = float(np.nanmedian(non_zero_flux))
                scale_factor = float(10.0 ** np.floor(np.log10(median_flux)))
                if not np.isfinite(scale_factor) or scale_factor == 0:
                    scale_factor = 1.0

            norm_flux_qty = flux_qty / scale_factor
            norm_err_qty = err_qty / scale_factor
            history.append(
                f"Dynamically scaled flux parameters by 10^{int(np.log10(scale_factor))} to stabilize optimizer"
            )

            mask_arr = (
                ~np.isfinite(norm_flux_qty.value)
                | ~np.isfinite(norm_err_qty.value)
                | (norm_err_qty.value <= 0)
                | ivar_zero_mask
                | file_dq_mask
            )

            if not np.any(~mask_arr):
                raise IcarusIngestionError("Data arrays are completely masked or NaN.")

        except Exception as e:
            log.error(f"FITS extraction failed for {target.target_id}: {e}")
            raise IcarusIngestionError(f"Failed to extract or coerce data arrays: {e}")

        spec_obj = Spectrum1D(
            wavelength=wave_qty,
            flux=norm_flux_qty,
            uncertainty=norm_err_qty,
            mask=mask_arr,
            target=target,
            spectral_unit=u.AA,
            flux_unit=STANDARD_FLUX_UNIT,
            flux_scale_factor=scale_factor,
            resolution_kms=inst_sigma_kms,
            meta=meta,
            history=history,
        )

        DevState.emit(
            "post_build_spectrum",
            {"target_id": target.target_id, "scale_factor": scale_factor},
        )
        return spec_obj
