"""
make_line_list.py

Generates the standardized vacuum rest-frame atomic transition catalog
(`icarus_master_lines.fits`) for ICARUS v3.4.
All wavelengths are strictly in vacuum Angstroms (AA) and all transition names
match the exact identifiers expected by `core/config.py`, `physics/line_models.py`,
and `physics/demographics.py`.
"""

__project__ = "ICARUS"
__author__ = "Joseph Havens"
__version__ = "3.4.0"

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import astropy.units as u
from astropy.table import QTable

from ..core.dev_tools import DevState, dev_hook
from ..core.logger import get_icarus_logger

log = get_icarus_logger()


DATA_DIR = Path(__file__).resolve().parent

# (LINE_NAME, Vacuum Rest Wavelength [AA], PROFILE ["broad" | "narrow"], OUTFLOW_CANDIDATE)
MASTER_VACUUM_LINES: List[Tuple[str, float, str, bool]] = [
    # --- Far-UV & Lyman Series ---
    ("Ly_gamma", 972.537, "broad", False),
    ("C_III_977", 977.020, "broad", False),
    ("Ly_beta", 1025.722, "broad", False),
    ("O_VI_1032", 1031.926, "broad", False),
    ("O_VI_1038", 1037.617, "broad", False),
    ("Ly_alpha", 1215.670, "broad", False),
    ("N_V_1240", 1240.810, "broad", False),
    ("O_I_1304", 1304.858, "narrow", False),
    ("C_II_1335", 1335.310, "narrow", False),
    ("Si_IV_1397", 1396.760, "broad", False),
    ("O_IV_1402", 1402.770, "broad", False),
    ("N_IV_1486", 1486.496, "narrow", False),
    ("C_IV_1548", 1548.187, "broad", False),
    ("C_IV_1551", 1550.772, "broad", False),
    ("He_II_1640", 1640.420, "broad", False),
    ("O_III_1661", 1660.809, "narrow", False),
    ("O_III_1666", 1666.150, "narrow", False),
    ("N_III_1750", 1750.000, "narrow", False),
    ("Si_III_1883", 1882.707, "narrow", False),
    ("Si_III_1892", 1892.030, "narrow", False),
    ("C_III_1907", 1906.683, "narrow", False),
    ("C_III_1909", 1908.734, "broad", False),
    # --- Near-UV ---
    ("C_II_2326", 2326.113, "narrow", False),
    ("Ne_IV_2424", 2423.830, "narrow", False),
    ("Mg_II_2796", 2796.352, "broad", False),
    ("Mg_II_2803", 2803.531, "broad", False),
    ("Ne_V_3347", 3346.790, "narrow", False),
    ("Ne_V_3427", 3426.850, "narrow", False),
    # --- Rest-Frame Optical ---
    ("O_II_3726", 3727.092, "narrow", False),
    ("O_II_3729", 3729.875, "narrow", False),
    ("Ne_III_3869", 3869.860, "narrow", False),
    ("He_I_3889", 3889.750, "broad", False),
    ("Ne_III_3968", 3968.590, "narrow", False),
    ("H_epsilon", 3971.198, "broad", False),
    ("H_delta", 4102.892, "broad", False),
    ("H_gamma", 4341.684, "broad", False),
    ("O_III_4363", 4364.436, "narrow", False),
    ("He_II_4686", 4687.020, "broad", False),
    ("H_beta", 4862.683, "broad", False),
    ("O_III_4959", 4960.295, "narrow", True),
    ("O_III_5007", 5008.240, "narrow", True),
    ("N_I_5200", 5200.257, "narrow", False),
    ("He_I_5876", 5877.249, "broad", False),
    ("O_I_6300", 6302.046, "narrow", False),
    ("O_I_6363", 6365.536, "narrow", False),
    ("N_II_6548", 6549.860, "narrow", False),
    ("H_alpha", 6564.614, "broad", False),
    ("N_II_6583", 6585.270, "narrow", False),
    ("S_II_6716", 6718.295, "narrow", False),
    ("S_II_6731", 6732.674, "narrow", False),
    ("Ar_III_7136", 7137.770, "narrow", False),
    # --- Near-Infrared (JWST NIRSpec / MIRI) ---
    ("S_III_9069", 9071.100, "narrow", False),
    ("S_III_9531", 9533.200, "narrow", False),
    ("Pa_delta", 10052.128, "broad", False),
    ("He_I_10830", 10833.306, "broad", False),
    ("Pa_gamma", 10941.090, "broad", False),
    ("Fe_II_12570", 12570.206, "narrow", False),
    ("Pa_beta", 12821.588, "broad", False),
    ("Fe_II_16440", 16440.018, "narrow", False),
    ("Pa_alpha", 18756.096, "broad", False),
    ("Br_delta", 19450.870, "broad", False),
    ("Br_gamma", 21661.210, "broad", False),
    ("Br_beta", 26258.670, "broad", False),
    ("Pf_gamma", 37405.560, "broad", False),
    ("Br_alpha", 40522.620, "broad", False),
]


@dev_hook(step_name="build_master_line_list")
def build_master_line_list(output_path: Optional[Path] = None) -> Path:
    """Writes `icarus_master_lines.fits` to `icarus_pkg/data/`."""
    DevState.emit("pre_build_master_line_list", locals())
    out_file = output_path or (DATA_DIR / "icarus_master_lines.fits")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    names = [row[0] for row in MASTER_VACUUM_LINES]
    waves = [row[1] for row in MASTER_VACUUM_LINES] * u.AA
    profiles = [row[2] for row in MASTER_VACUUM_LINES]
    outflows = [row[3] for row in MASTER_VACUUM_LINES]

    tbl = QTable()
    tbl["LINE_NAME"] = names
    tbl["REST_WAVE"] = waves
    tbl["PROFILE"] = profiles
    tbl["OUTFLOW"] = outflows
    tbl.meta["EXTNAME"] = "MASTER_LINES"
    tbl.meta["VERSION"] = __version__

    tbl.write(out_file, format="fits", overwrite=True)
    log.info(f"Wrote {len(tbl)} vacuum transitions to {out_file.name}")
    DevState.emit("post_build_master_line_list", {"out_file": str(out_file)})
    return out_file


if __name__ == "__main__":
    build_master_line_list()
