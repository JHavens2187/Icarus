"""
ICARUS: An AGN-Specialized Emission Line & Kinematic Fitting Engine.
"""

__project__ = "ICARUS"
__author__ = "Joseph Havens"
__version__ = "3.4.0"

from .api import AGNFitter, IcarusResult
from .core.config import IcarusConfig
from .core.containers import Spectrum1D, Target
from .core.logger import get_icarus_logger
from .viz.style import apply_icarus_style

# Initialize global headless/mathtext rcParams upon package import
apply_icarus_style()

__all__ = [
    "AGNFitter",
    "IcarusResult",
    "IcarusConfig",
    "Spectrum1D",
    "Target",
    "get_icarus_logger",
    "apply_icarus_style",
]
