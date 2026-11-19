"""
containers.py

Immutable data structures and spectral containers for the ICARUS v3.4 pipeline.
Provides catalog-free single-file ingestion via `Spectrum1D.read()` and in-memory
array initialization via `Spectrum1D.from_arrays()`.
"""

__project__ = "ICARUS"
__author__ = "Joseph Havens"
__version__ = "3.4.0"

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from astropy import units as u
from lmfit import Parameters

from .dev_tools import DevState, dev_hook
from .exceptions import IcarusDataDimensionError
from .logger import get_icarus_logger

log = get_icarus_logger()
STANDARD_FLUX_UNIT = u.Unit("erg / (cm2 s AA)")


@dataclass(frozen=True, kw_only=True)
class Target:
    """Immutable metadata for a single astrophysical source."""

    target_id: str
    redshift: float = 0.0
    sample: Optional[str] = None
    setup: Optional[str] = None
    pointing: Optional[str] = None
    ra: Optional[float] = None
    dec: Optional[float] = None


@dataclass(frozen=True, kw_only=True)
class Spectrum1D:
    """
    Immutable container for 1D spectral arrays.
    Tracks structural dimensions, target context, wavelength-dependent instrumental
    dispersion (sigma_inst in km/s), and O(1) dynamic scaling parameters.
    """

    wavelength: u.Quantity
    flux: u.Quantity
    uncertainty: u.Quantity
    mask: np.ndarray
    target: Target
    spectral_unit: u.Unit = u.AA
    flux_unit: u.Unit = STANDARD_FLUX_UNIT
    flux_scale_factor: float = 1.0
    resolution_kms: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    history: List[str] = field(default_factory=list)

    def __post_init__(self):
        shapes = {
            "wavelength": self.wavelength.shape,
            "flux": self.flux.shape,
            "uncertainty": self.uncertainty.shape,
            "mask": self.mask.shape,
        }
        if self.resolution_kms is not None:
            shapes["resolution_kms"] = np.asarray(self.resolution_kms).shape

        if len(set(shapes.values())) > 1:
            raise IcarusDataDimensionError(
                f"Array shape mismatch in Spectrum1D initialization for {self.target.target_id}: {shapes}"
            )

        if not (len(self.wavelength) == len(self.flux) == len(self.uncertainty)):
            raise IcarusDataDimensionError(
                "Spectral arrays must have identical lengths."
            )

        if not self.wavelength.unit.is_equivalent(u.AA):
            log.warning(
                f"[{self.target.target_id}] Internal spectral axis must be equivalent to Angstroms."
            )

    @dev_hook(step_name="spectrum_to_ndarrays")
    def to_ndarrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (wavelength, flux, uncertainty) float64 arrays in O(1) scaled space."""
        DevState.emit("pre_to_ndarrays", {"target_id": self.target.target_id})
        arrays = (
            np.asarray(self.wavelength.value, dtype=np.float64),
            np.asarray(self.flux.value, dtype=np.float64),
            np.asarray(self.uncertainty.value, dtype=np.float64),
        )
        DevState.emit("post_to_ndarrays", {"target_id": self.target.target_id})
        return arrays

    def get_inst_sigma_kms(self, rest_wavelength: float) -> float:
        """Returns the instrumental velocity dispersion sigma_inst [km/s] at `rest_wavelength`."""
        if self.resolution_kms is None or len(self.resolution_kms) == 0:
            return 127.4  # Default FWHM = 300 km/s
        w_vals = np.asarray(self.wavelength.value, dtype=np.float64)
        idx = int(np.argmin(np.abs(w_vals - rest_wavelength)))
        return float(self.resolution_kms[idx])

    @classmethod
    def from_arrays(
        cls,
        wavelength: Union[np.ndarray, u.Quantity],
        flux: Union[np.ndarray, u.Quantity],
        uncertainty: Union[np.ndarray, u.Quantity],
        redshift: float = 0.0,
        target_id: str = "synthetic_target",
        wave_unit: Union[str, u.Unit] = u.AA,
        flux_unit: Union[str, u.Unit] = STANDARD_FLUX_UNIT,
        resolution_fwhm_kms: float = 300.0,
        mask: Optional[np.ndarray] = None,
        is_rest_frame: bool = False,
    ) -> "Spectrum1D":
        """
        Constructs a normalized, rest-frame Spectrum1D directly from in-memory arrays.
        Ideal for interactive notebook experimentation and synthetic CLOUDY tests.
        """
        w_unit = u.Unit(wave_unit)
        f_unit = u.Unit(flux_unit)

        w_qty_obs = (
            wavelength.to(u.AA, equivalencies=u.spectral())
            if isinstance(wavelength, u.Quantity)
            else (np.asarray(wavelength, dtype=np.float64) * w_unit).to(
                u.AA, equivalencies=u.spectral()
            )
        )
        f_qty_obs = (
            flux.to(STANDARD_FLUX_UNIT, equivalencies=u.spectral_density(w_qty_obs))
            if isinstance(flux, u.Quantity)
            else (np.asarray(flux, dtype=np.float64) * f_unit).to(
                STANDARD_FLUX_UNIT, equivalencies=u.spectral_density(w_qty_obs)
            )
        )
        e_qty_obs = (
            uncertainty.to(
                STANDARD_FLUX_UNIT, equivalencies=u.spectral_density(w_qty_obs)
            )
            if isinstance(uncertainty, u.Quantity)
            else (np.asarray(uncertainty, dtype=np.float64) * f_unit).to(
                STANDARD_FLUX_UNIT, equivalencies=u.spectral_density(w_qty_obs)
            )
        )

        if redshift > 0.0 and not is_rest_frame:
            w_qty = w_qty_obs / (1.0 + redshift)
            f_qty = f_qty_obs * (1.0 + redshift)
            e_qty = e_qty_obs * (1.0 + redshift)
        else:
            w_qty, f_qty, e_qty = w_qty_obs, f_qty_obs, e_qty_obs

        finite_f = np.abs(f_qty.value[np.isfinite(f_qty.value) & (f_qty.value != 0)])
        if len(finite_f) > 0:
            scale_factor = float(10.0 ** np.floor(np.log10(np.nanmedian(finite_f))))
            if not np.isfinite(scale_factor) or scale_factor == 0:
                scale_factor = 1.0
        else:
            scale_factor = 1.0

        norm_f = f_qty / scale_factor
        norm_e = e_qty / scale_factor

        auto_mask = (
            ~np.isfinite(norm_f.value)
            | ~np.isfinite(norm_e.value)
            | (norm_e.value <= 0)
        )
        if mask is not None:
            auto_mask = auto_mask | np.asarray(mask, dtype=bool)

        sigma_inst = float(resolution_fwhm_kms) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        res_arr = np.full(len(w_qty), sigma_inst, dtype=np.float64)

        return cls(
            wavelength=w_qty,
            flux=norm_f,
            uncertainty=norm_e,
            mask=auto_mask,
            target=Target(target_id=str(target_id), redshift=float(redshift)),
            spectral_unit=u.AA,
            flux_unit=STANDARD_FLUX_UNIT,
            flux_scale_factor=scale_factor,
            resolution_kms=res_arr,
            history=["Initialized in-memory via Spectrum1D.from_arrays()"],
        )

    @classmethod
    def read(
        cls,
        filepath: Union[str, Path],
        redshift: float = 0.0,
        target_id: Optional[str] = None,
        preset: Optional[Union[str, Path, Any]] = "jwst_nirspec_medium",
        setup: Optional[str] = None,
        column_mapping: Optional[Dict[str, str]] = None,
        **overrides: Any,
    ) -> "Spectrum1D":
        """
        Catalog-free single-file loader for interactive notebooks and tutorials:
        `spec = ic.Spectrum1D.read("jades_4096.fits", redshift=2.46)`
        """
        from .config import IcarusConfig
        from .ingest import ObservationBuilder

        fpath = Path(filepath)
        tid = target_id or fpath.stem
        target = Target(target_id=str(tid), redshift=float(redshift), setup=setup)

        if isinstance(preset, IcarusConfig):
            ic_cfg = preset.apply_overrides(column_mapping=column_mapping, **overrides)
            local_cfg = ic_cfg.get_target_config(target)
        elif preset is not None:
            ic_cfg = IcarusConfig.from_preset(
                preset, column_mapping=column_mapping, **overrides
            )
            local_cfg = ic_cfg.get_target_config(target)
        else:
            local_cfg = None

        builder = ObservationBuilder(fpath.parent, column_mapping=column_mapping)
        spec = builder.build_spectrum(
            target,
            fpath,
            config=local_cfg,
            dry_run=False,
            column_mapping=column_mapping,
        )
        if spec is None:
            raise IcarusDataDimensionError(f"Failed to read Spectrum1D from {fpath}")
        return spec


@dataclass
class EmissionLineDef:
    name: str
    rest_wavelength: float
    test_broad: bool = False
    outflow_candidate: bool = False


@dataclass
class LineComplex:
    lines: List[EmissionLineDef]
    window_start: u.Quantity
    window_end: u.Quantity
    in_bounds: bool = True

    @property
    def name(self) -> str:
        return "_".join([line.name for line in self.lines])

    @property
    def primary_line(self) -> EmissionLineDef:
        """Selects the primary anchor transition for velocity-space centering."""
        priority_anchors = (
            "O_III_5007",
            "H_alpha",
            "H_beta",
            "Pa_beta",
            "Pa_alpha",
            "He_I_10830",
            "S_III_9531",
            "Mg_II_2796",
            "C_IV_1548",
            "C_III_1909",
            "O_II_3729",
            "Ly_alpha",
        )
        for anchor in priority_anchors:
            for line in self.lines:
                if line.name == anchor:
                    return line
        return self.lines[0]


@dataclass
class SpectralComponent:
    name: str
    model: Any
    params: Parameters
    component_type: str
    rest_wavelength: Optional[float] = None
