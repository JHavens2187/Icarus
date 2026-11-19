"""
icarus_pkg.physics

Spectroscopic physical modeling, FFT Fe II template convolution, relative-velocity
line factories, Bayesian complex optimization, and single-epoch virial demographics.
"""

__project__ = "ICARUS"
__author__ = "Joseph Havens"
__version__ = "3.4.0"

from .continuum import ContinuumFitter
from .demographics import DemographicsCalculator
from .feii_model import FeIITemplateConvolver
from .fitting_engine import CoreFitter, SpectralPreparer
from .line_models import LineComponentFactory

__all__ = [
    "ContinuumFitter",
    "DemographicsCalculator",
    "FeIITemplateConvolver",
    "CoreFitter",
    "SpectralPreparer",
    "LineComponentFactory",
]
