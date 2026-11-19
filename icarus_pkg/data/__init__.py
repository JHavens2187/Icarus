"""
icarus_pkg.data

Master vacuum atomic line list, UV/Optical Fe II empirical template generators,
and instrumental resolving power dispersion curves for ICARUS v3.4.
"""

__project__ = "ICARUS"
__author__ = "Joseph Havens"
__version__ = "3.4.0"

from .build_feii_templates import (
    build_dispersion_curves,
    build_feii_templates,
    ensure_all_data_assets,
)
from .make_line_list import MASTER_VACUUM_LINES, build_master_line_list

__all__ = [
    "MASTER_VACUUM_LINES",
    "build_master_line_list",
    "build_feii_templates",
    "build_dispersion_curves",
    "ensure_all_data_assets",
]
