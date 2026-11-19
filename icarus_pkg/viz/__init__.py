"""
icarus_pkg.viz

Publication-grade 5-panel diagnostic visualization engine, BPT/OHNO excitation
diagnostics, LaTeX mathtext typography, and global rcParams styling for ICARUS v3.4.
"""

__project__ = "ICARUS"
__author__ = "Joseph Havens"
__version__ = "3.4.0"

from .bpt import BPTPlotter
from .plotter import DiagnosticPlotter
from .style import ICARUS_PALETTE, apply_icarus_style
from .typography import decrowd_annotations, format_latex_ion, sanitize_line_name

__all__ = [
    "BPTPlotter",
    "DiagnosticPlotter",
    "ICARUS_PALETTE",
    "apply_icarus_style",
    "decrowd_annotations",
    "format_latex_ion",
    "sanitize_line_name",
]
