"""
feii_model.py

Empirical UV and Optical Fe II pseudo-continuum FFT convolution engine for ICARUS v3.4.
Convolves Vestergaard & Wilkes (2001) / Tsuzuki et al. (2006) UV and Boroson & Green (1992)
I Zw 1 optical templates on a uniform logarithmic wavelength grid (dv = 10.0 km/s)
with intrinsic template width subtraction.
"""

__project__ = "ICARUS"
__author__ = "Joseph Havens"
__version__ = "3.4.0"

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from astropy import constants as const
from astropy.table import QTable
from lmfit import Model
from scipy.signal import fftconvolve

from ..core.containers import SpectralComponent
from ..core.dev_tools import DevState, dev_hook
from ..core.logger import get_icarus_logger

log = get_icarus_logger()
C_KMS = const.c.to("km/s").value
INTRINSIC_IZW1_SIGMA_KMS = 900.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # 382.19 km/s
DV_GRID_KMS = 10.0
PACKAGE_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "templates"


class FeIITemplateConvolver:
    """
    Manages uniform log-wavelength grids and fast FFT Gaussian broadening
    for UV (1200-3500 A) and Optical (3500-7500 A) Fe II pseudo-continua.
    """

    def __init__(
        self,
        uv_template_path: Optional[Union[str, Path]] = None,
        opt_template_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.uv_path = (
            Path(uv_template_path)
            if uv_template_path
            else PACKAGE_DATA_DIR / "feii_uv_vw01_tz06.fits"
        )
        self.opt_path = (
            Path(opt_template_path)
            if opt_template_path
            else PACKAGE_DATA_DIR / "feii_opt_bg92.fits"
        )

        self.ln_wave_grid, self.wave_grid, self.uv_flux, self.opt_flux = (
            self._load_or_synthesize_templates()
        )

    def _load_or_synthesize_templates(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads the log-wavelength UV and Optical Fe II templates from disk,
        or synthesizes an analytic multiplet basis at intrinsic I Zw 1 resolution
        (FWHM = 900 km/s) if the FITS templates have not been generated yet.
        """
        d_ln_lam = DV_GRID_KMS / C_KMS
        ln_min = np.log(1000.0)
        ln_max = np.log(10000.0)
        ln_grid = np.arange(ln_min, ln_max, d_ln_lam, dtype=np.float64)
        wave_grid = np.exp(ln_grid)

        uv_arr = np.zeros_like(wave_grid)
        opt_arr = np.zeros_like(wave_grid)

        loaded_uv = False
        if self.uv_path.exists():
            try:
                tbl_uv = QTable.read(self.uv_path)
                w_u = np.asarray(tbl_uv[tbl_uv.colnames[0]], dtype=np.float64)
                f_u = np.asarray(tbl_uv[tbl_uv.colnames[1]], dtype=np.float64)
                uv_arr = np.interp(wave_grid, w_u, f_u, left=0.0, right=0.0)
                loaded_uv = True
            except Exception as e:
                log.warning(
                    f"Failed to read UV Fe II template ({self.uv_path.name}): {e}"
                )

        loaded_opt = False
        if self.opt_path.exists():
            try:
                tbl_opt = QTable.read(self.opt_path)
                w_o = np.asarray(tbl_opt[tbl_opt.colnames[0]], dtype=np.float64)
                f_o = np.asarray(tbl_opt[tbl_opt.colnames[1]], dtype=np.float64)
                opt_arr = np.interp(wave_grid, w_o, f_o, left=0.0, right=0.0)
                loaded_opt = True
            except Exception as e:
                log.warning(
                    f"Failed to read Optical Fe II template ({self.opt_path.name}): {e}"
                )

        if not loaded_uv:
            # Analytic UV Fe II multiplet complexes (UV 1, 2, 3, 34, 60, 62, 63, 78, Balmer wings)
            uv_multiplets = [
                (1610.0, 45.0, 0.35),
                (1715.0, 40.0, 0.30),
                (1860.0, 55.0, 0.45),
                (2100.0, 50.0, 0.35),
                (2260.0, 45.0, 0.50),
                (2340.0, 55.0, 0.85),
                (2385.0, 45.0, 0.95),
                (2445.0, 50.0, 0.90),
                (2505.0, 55.0, 1.00),
                (2565.0, 45.0, 0.85),
                (2615.0, 40.0, 0.90),
                (2745.0, 35.0, 0.75),
                (2850.0, 40.0, 0.55),
                (2950.0, 55.0, 0.65),
                (3210.0, 65.0, 0.45),
            ]
            for cen, width_aa, rel_amp in uv_multiplets:
                sig_tot = np.hypot(width_aa, cen * (INTRINSIC_IZW1_SIGMA_KMS / C_KMS))
                uv_arr += rel_amp * np.exp(-0.5 * ((wave_grid - cen) / sig_tot) ** 2)

        if not loaded_opt:
            # Analytic Optical Fe II multiplet complexes (m37, m38, m42, m48, m49)
            opt_multiplets = [
                # Blue shelf (4434 - 4684 A, m37 & m38)
                (4491.4, 0.45),
                (4508.3, 0.55),
                (4515.3, 0.65),
                (4520.2, 0.60),
                (4522.6, 0.70),
                (4549.5, 0.95),
                (4555.9, 0.85),
                (4576.3, 0.50),
                (4583.8, 1.00),
                (4620.5, 0.40),
                (4629.3, 0.75),
                (4666.8, 0.35),
                # m42 multiplet (4924, 5018, 5169 A)
                (4923.9, 0.65),
                (5018.4, 0.75),
                (5169.0, 0.85),
                # Red shelf (5150 - 5400 A, m48 & m49)
                (5197.6, 0.70),
                (5234.6, 0.80),
                (5264.8, 0.45),
                (5276.0, 0.85),
                (5284.1, 0.40),
                (5316.6, 0.95),
                (5325.6, 0.45),
                (5362.9, 0.65),
            ]
            for cen, rel_amp in opt_multiplets:
                sig_aa = cen * (INTRINSIC_IZW1_SIGMA_KMS / C_KMS)
                opt_arr += rel_amp * np.exp(-0.5 * ((wave_grid - cen) / sig_aa) ** 2)

        # Normalize peak of each template to 1.0 so amplitude parameters represent peak flux density
        if np.max(uv_arr) > 0:
            uv_arr = uv_arr / np.max(uv_arr)
        if np.max(opt_arr) > 0:
            opt_arr = opt_arr / np.max(opt_arr)

        return ln_grid, wave_grid, uv_arr, opt_arr

    @dev_hook(step_name="convolve_feii_template")
    def convolve_and_project(
        self,
        x: np.ndarray,
        feii_uv_amp: float = 0.0,
        feii_opt_amp: float = 0.0,
        feii_v_offset: float = 0.0,
        feii_sigma_v: float = 1200.0,
    ) -> np.ndarray:
        """
        Broadens the combined UV + Optical Fe II template on the uniform log-wavelength
        grid via FFT Gaussian convolution (accounting for the 382.19 km/s intrinsic
        I Zw 1 width) and projects onto the target rest-frame wavelength array `x`.
        """
        if feii_uv_amp <= 1e-12 and feii_opt_amp <= 1e-12:
            return np.zeros_like(x, dtype=np.float64)

        base_template = (
            max(float(feii_uv_amp), 0.0) * self.uv_flux
            + max(float(feii_opt_amp), 0.0) * self.opt_flux
        )

        # Subtract intrinsic I Zw 1 dispersion in quadrature
        clamped_sigma = max(float(feii_sigma_v), 400.0)
        sigma_kern_kms = np.sqrt(
            max(clamped_sigma**2 - INTRINSIC_IZW1_SIGMA_KMS**2, DV_GRID_KMS**2)
        )
        sigma_pix = sigma_kern_kms / DV_GRID_KMS

        half_width = max(int(np.ceil(5.0 * sigma_pix)), 3)
        kx = np.arange(-half_width, half_width + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (kx / sigma_pix) ** 2)
        kernel /= np.sum(kernel)

        broadened = fftconvolve(base_template, kernel, mode="same")

        # Apply velocity shift in log-wavelength space: ln(lambda_shifted) = ln(lambda) + v / c
        shifted_wave = self.wave_grid * (1.0 + float(feii_v_offset) / C_KMS)
        projected = np.interp(x, shifted_wave, broadened, left=0.0, right=0.0)
        return np.asarray(projected, dtype=np.float64)

    def build_lmfit_model(
        self,
        prefix: str = "",
        uv_amp_guess: float = 0.05,
        opt_amp_guess: float = 0.05,
        max_amp: float = 100.0,
    ) -> SpectralComponent:
        """Wraps `convolve_and_project` as a constrainable LMFIT Model."""

        def _feii_func(
            x: np.ndarray,
            feii_uv_amp: float = 0.05,
            feii_opt_amp: float = 0.05,
            feii_v_offset: float = 0.0,
            feii_sigma_v: float = 1200.0,
        ) -> np.ndarray:
            return self.convolve_and_project(
                x,
                feii_uv_amp=feii_uv_amp,
                feii_opt_amp=feii_opt_amp,
                feii_v_offset=feii_v_offset,
                feii_sigma_v=feii_sigma_v,
            )

        model = Model(_feii_func, prefix=prefix)
        params = model.make_params()

        safe_max = max(float(max_amp), 1.0)
        params[f"{prefix}feii_uv_amp"].set(
            value=max(float(uv_amp_guess), 0.0), min=0.0, max=safe_max
        )
        params[f"{prefix}feii_opt_amp"].set(
            value=max(float(opt_amp_guess), 0.0), min=0.0, max=safe_max
        )
        params[f"{prefix}feii_v_offset"].set(value=0.0, min=-500.0, max=500.0)
        params[f"{prefix}feii_sigma_v"].set(value=1200.0, min=400.0, max=5000.0)

        return SpectralComponent(
            name=prefix.strip("_") or "feii",
            model=model,
            params=params,
            component_type="feii_template",
        )
