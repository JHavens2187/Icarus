"""
icarus_pkg.core

Core infrastructure, immutable data containers, Pydantic configuration schemas,
unit-safe FITS ingestion, Multi-Extension FITS I/O, and runtime debugging hooks.
"""

__project__ = "ICARUS"
__author__ = "Joseph Havens"
__version__ = "3.4.0"

from .config import FittingConfig, GlobalConfig, IcarusConfig, SetupConfig
from .containers import (
    EmissionLineDef,
    LineComplex,
    SpectralComponent,
    Spectrum1D,
    Target,
)
from .dev_tools import DevState, dev_hook
from .exceptions import (
    IcarusConfigurationError,
    IcarusConvergenceError,
    IcarusDataDimensionError,
    IcarusError,
    IcarusIngestionError,
    IcarusKinematicError,
    IcarusTemplateError,
)
from .ingest import CatalogRegistry, ObservationBuilder
from .io import ResultWriter
from .logger import get_icarus_logger

__all__ = [
    "FittingConfig",
    "GlobalConfig",
    "IcarusConfig",
    "SetupConfig",
    "EmissionLineDef",
    "LineComplex",
    "SpectralComponent",
    "Spectrum1D",
    "Target",
    "DevState",
    "dev_hook",
    "IcarusError",
    "IcarusDataDimensionError",
    "IcarusIngestionError",
    "IcarusConfigurationError",
    "IcarusKinematicError",
    "IcarusTemplateError",
    "IcarusConvergenceError",
    "CatalogRegistry",
    "ObservationBuilder",
    "ResultWriter",
    "get_icarus_logger",
]
