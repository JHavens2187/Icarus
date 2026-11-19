"""
exceptions.py

Custom exception hierarchy for the ICARUS pipeline.
Ensures precise error tracing and isolates specific domain failures.
"""

__project__ = "ICARUS"
__author__ = "Joseph Havens"
__version__ = "3.4.0"


class IcarusError(Exception):
    """Base exception for all ICARUS pipeline errors."""

    pass


class IcarusDataDimensionError(IcarusError):
    """Raised when spectral arrays (flux, wavelength, error) have mismatched shapes or invalid bounds."""

    pass


class IcarusIngestionError(IcarusError):
    """Raised when FITS files or catalogs cannot be located, parsed, or dynamically scaled."""

    pass


class IcarusConfigurationError(IcarusError):
    """Raised when the YAML configuration contains invalid parameters or unphysical units."""

    pass


class IcarusKinematicError(IcarusError):
    """Raised when physical constraints (e.g., velocity bounds) are mathematically invalid."""

    pass


class IcarusTemplateError(IcarusError):
    """Raised when empirical Fe II templates or dispersion curves cannot be loaded or convolved."""

    pass


class IcarusConvergenceError(IcarusError):
    """Raised when the MCMC or LMFIT optimizer entirely fails to find a mathematically valid solution."""

    pass
