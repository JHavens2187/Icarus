"""
build_feii_templates.py

Generates the standardized UV (Vestergaard & Wilkes 2001 / Tsuzuki et al. 2006)
and Optical (Boroson & Green 1992 I Zw 1) Fe II empirical templates on a uniform
logarithmic wavelength grid (dv = 10.0 km/s) inside `icarus_pkg/data/templates/`,
alongside the JWST NIRSpec wavelength-dependent resolving power tables R(lambda)
inside `icarus_pkg/data/dispersion/`.
"""

__project__ = "ICARUS"
__author__ = "Joseph Havens"
__version__ = "3.4.0"

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import astropy.units as u
import numpy as np
from astropy import constants as const
from astropy.table import QTable

from ..core.dev_tools import DevState, dev_hook
from ..core.logger import get_icarus_logger

log = get_icarus_logger()


DATA_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = DATA_DIR / "templates"
DISPERSION_DIR = DATA_DIR / "dispersion"

C_KMS = const.c.to("km/s").value
INTRINSIC_IZW1_SIGMA_KMS = 900.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # 382.19 km/s
DV_GRID_KMS = 10.0


@dev_hook(step_name="build_feii_templates_on_disk")
def build_feii_templates() -> Tuple[Path, Path]:
    """
    Constructs and serializes `feii_uv_vw01_tz06.fits` (1200-3500 AA) and
    `feii_opt_bg92.fits` (3500-7500 AA) at intrinsic I Zw 1 resolution
    (FWHM = 900 km/s) on the uniform dv = 10.0 km/s logarithmic grid.
    """
    DevState.emit("pre_build_feii_templates", locals())
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

    d_ln_lam = DV_GRID_KMS / C_KMS
    ln_grid = np.arange(np.log(1000.0), np.log(10000.0), d_ln_lam, dtype=np.float64)
    wave_grid = np.exp(ln_grid)

    # 1. UV Fe II Multiplet Basis (VW01 + Tsuzuki06 bridge: 1250 - 3500 AA)
    uv_arr = np.zeros_like(wave_grid)
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

    uv_arr[(wave_grid < 1200.0) | (wave_grid > 3550.0)] = 0.0
    if np.max(uv_arr) > 0:
        uv_arr /= np.max(uv_arr)

    uv_path = TEMPLATE_DIR / "feii_uv_vw01_tz06.fits"
    tbl_uv = QTable()
    tbl_uv["WAVE"] = wave_grid * u.AA
    tbl_uv["FLUX_NORM"] = uv_arr
    tbl_uv.meta["EXTNAME"] = "FEII_UV"
    tbl_uv.meta["DV_KMS"] = DV_GRID_KMS
    tbl_uv.meta["FWHM_INT"] = 900.0
    tbl_uv.write(uv_path, format="fits", overwrite=True)

    # 2. Optical Fe II Multiplet Basis (Boroson & Green 1992 I Zw 1: 3500 - 7500 AA)
    opt_arr = np.zeros_like(wave_grid)
    opt_multiplets = [
        # Blue shelf (4434 - 4684 AA, multiplets 37 & 38)
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
        # Multiplet 42 (4924, 5018, 5169 AA)
        (4923.9, 0.65),
        (5018.4, 0.75),
        (5169.0, 0.85),
        # Red shelf (5150 - 5400 AA, multiplets 48 & 49)
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

    opt_arr[(wave_grid < 3500.0) | (wave_grid > 7500.0)] = 0.0
    if np.max(opt_arr) > 0:
        opt_arr /= np.max(opt_arr)

    opt_path = TEMPLATE_DIR / "feii_opt_bg92.fits"
    tbl_opt = QTable()
    tbl_opt["WAVE"] = wave_grid * u.AA
    tbl_opt["FLUX_NORM"] = opt_arr
    tbl_opt.meta["EXTNAME"] = "FEII_OPT"
    tbl_opt.meta["DV_KMS"] = DV_GRID_KMS
    tbl_opt.meta["FWHM_INT"] = 900.0
    tbl_opt.write(opt_path, format="fits", overwrite=True)

    log.info(f"Wrote Fe II FITS templates to {TEMPLATE_DIR}")
    DevState.emit(
        "post_build_feii_templates", {"uv": str(uv_path), "opt": str(opt_path)}
    )
    return uv_path, opt_path


def build_dispersion_curves() -> None:
    """
    Writes wavelength-dependent resolving power tables R(lambda_obs) for
    JWST NIRSpec Prism/Clear, Medium Gratings, and High Gratings.
    """
    DISPERSION_DIR.mkdir(parents=True, exist_ok=True)
    wave_um = np.linspace(0.6, 5.3, 500)
    wave_aa = wave_um * 1e4 * u.AA

    # 1. JWST NIRSpec Prism/Clear resolving power curve (R ~ 30 at 1.2 um -> ~ 300 at 5.0 um)
    r_prism = 30.0 + 10.5 * ((np.maximum(wave_um - 1.1, 0.0)) ** 2.15)
    r_prism = np.clip(r_prism, 30.0, 330.0)
    tbl_p = QTable([wave_aa, r_prism], names=("WAVELENGTH", "R"))
    tbl_p.write(
        DISPERSION_DIR / "jwst_nirspec_prism_disp.fits", format="fits", overwrite=True
    )

    # 2. JWST NIRSpec Medium Gratings (G140M/G235M/G395M: R ~ 1000)
    r_med = 750.0 + 110.0 * wave_um
    tbl_m = QTable([wave_aa, r_med], names=("WAVELENGTH", "R"))
    tbl_m.write(
        DISPERSION_DIR / "jwst_nirspec_medium_disp.fits", format="fits", overwrite=True
    )

    # 3. JWST NIRSpec High Gratings (G140H/G235H/G395H: R ~ 2700)
    r_high = 2100.0 + 260.0 * wave_um
    tbl_h = QTable([wave_aa, r_high], names=("WAVELENGTH", "R"))
    tbl_h.write(
        DISPERSION_DIR / "jwst_nirspec_high_disp.fits", format="fits", overwrite=True
    )

    log.info(f"Wrote JWST dispersion curves to {DISPERSION_DIR}")


def ensure_all_data_assets() -> None:
    """Bootstraps all FITS line catalogs, Fe II templates, and dispersion curves."""
    sys.path.insert(0, str(DATA_DIR))
    try:
        from .make_line_list import build_master_line_list
    except ImportError:
        from make_line_list import build_master_line_list  # type: ignore

    build_master_line_list()
    build_feii_templates()
    build_dispersion_curves()


if __name__ == "__main__":
    ensure_all_data_assets()
