"""
config.py

Configuration schemas for the ICARUS v3.4 pipeline.
Handles YAML preset resolution, type enforcement, flexible cosmology models,
custom FITS column mappings, and dynamic compilation of overrides via Pydantic.
"""

__project__ = "ICARUS"
__author__ = "Joseph Havens"
__version__ = "3.4.0"

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import astropy.cosmology as astro_cosmo
import yaml
from astropy import units as u
from astropy.cosmology import Cosmology, FlatLambdaCDM
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .containers import Target
from .exceptions import IcarusConfigurationError

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def parse_astropy_quantity(value: Union[str, float, u.Quantity]) -> u.Quantity:
    """Helper to parse configuration strings into Astropy units."""
    if isinstance(value, u.Quantity):
        return value
    if isinstance(value, (int, float)):
        return u.Quantity(value)
    if isinstance(value, str):
        try:
            return u.Quantity(value)
        except Exception as e:
            raise IcarusConfigurationError(
                f"Cannot parse '{value}' into an astropy.units.Quantity: {e}"
            )
    raise IcarusConfigurationError(f"Invalid type for quantity parsing: {type(value)}")


class FittingConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    cont_type: Literal["powerlaw_feii", "polynomial", "powerlaw"] = "powerlaw_feii"
    continuum_degree: int = Field(
        default=2, ge=0, le=3, description="Order of polynomial continuum fallback"
    )

    continuum_windows: List[List[float]] = Field(
        default_factory=lambda: [
            [1330.0, 1370.0],
            [1445.0, 1485.0],
            [1700.0, 1800.0],
            [2100.0, 2200.0],
            [3010.0, 3070.0],
            [4180.0, 4230.0],
            [4430.0, 4770.0],
            [5080.0, 5450.0],
            [7400.0, 7700.0],
            [8400.0, 8800.0],
            [10100.0, 10500.0],
            [11800.0, 12200.0],
            [15000.0, 15500.0],
            [20000.0, 21000.0],
        ],
        description="Rest-frame windows (Angstroms) for anchoring the global continuum.",
    )

    feii_windows: List[List[float]] = Field(
        default_factory=lambda: [
            [2200.0, 2650.0],
            [4434.0, 4684.0],
            [5150.0, 5400.0],
        ],
        description="Rest-frame UV and Optical Fe II pseudo-continuum fitting windows.",
    )

    enable_feii: bool = Field(
        default=True,
        description="Enable empirical UV/Optical Fe II template FFT deconvolution.",
    )

    couple_narrow_v_offset: bool = Field(
        default=True,
        description="Couple secondary narrow line v_offset parameters to the primary anchor ion in multi-line complexes.",
    )

    enable_speed_gatekeeper: bool = Field(
        default=True,
        description="Skip broad-line optimization pass if no narrow core clears 2.0x local RMS noise.",
    )

    priority_lines: List[str] = Field(
        default_factory=lambda: [
            "O_III_5007",
            "H_beta",
            "H_alpha",
            "N_II_6583",
            "Pa_beta",
            "Pa_alpha",
            "He_I_10830",
            "S_III_9531",
            "Mg_II_2796",
            "C_IV_1548",
            "C_III_1909",
            "O_II_3729",
            "Ly_alpha",
        ],
        description="Programmatic priority ranking for selecting featured diagnostic subpanels.",
    )

    cont_width: Optional[u.Quantity] = u.Quantity("500.0 Angstrom")
    center: Literal["free", "constrained", "fixed"] = "constrained"
    snr_threshold: float = Field(default=3.0, description="Minimum S/N for detection")

    min_sigma: Optional[u.Quantity] = u.Quantity("30.0 km / s")
    max_sigma: Optional[u.Quantity] = u.Quantity("450.0 km / s")

    doublet_tying: Dict[str, Any] = Field(
        default_factory=lambda: {
            "O_III_doublet": {
                "lines": ["O_III_4959", "O_III_5007"],
                "flux_ratio": 0.3333333333,
                "tie_kinematics": True,
            },
            "N_II_doublet": {
                "lines": ["N_II_6548", "N_II_6583"],
                "flux_ratio": 0.339,
                "tie_kinematics": True,
            },
            "S_III_doublet": {
                "lines": ["S_III_9069", "S_III_9531"],
                "flux_ratio": 0.4048,
                "tie_kinematics": True,
            },
            "O_II_doublet": {
                "lines": ["O_II_3726", "O_II_3729"],
                "tie_kinematics": True,
            },
            "S_II_doublet": {
                "lines": ["S_II_6716", "S_II_6731"],
                "tie_kinematics": True,
            },
            "C_IV_doublet": {
                "lines": ["C_IV_1548", "C_IV_1551"],
                "tie_kinematics": True,
            },
            "Mg_II_doublet": {
                "lines": ["Mg_II_2796", "Mg_II_2803"],
                "tie_kinematics": True,
            },
        }
    )

    broad_kinematics: Dict[str, Any] = Field(
        default_factory=lambda: {
            "min_sigma": "500.0 km / s",
            "max_sigma": "8000.0 km / s",
        }
    )

    @field_validator("cont_width", "min_sigma", "max_sigma", mode="before")
    @classmethod
    def validate_quantities(cls, v):
        return parse_astropy_quantity(v) if v is not None else v


class GlobalConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    line_table: str = str(PACKAGE_ROOT / "data" / "icarus_master_lines.fits")
    lines: Union[List[str], Literal["all"]] = "all"

    wave_unit: str = "um"
    flux_unit: str = "1e-20 erg / (cm2 s AA)"
    resolution: Optional[u.Quantity] = u.Quantity("300.0 km / s")
    dispersion_curve: Optional[str] = None

    # Dynamic FITS schema mapping (supports wave/loglam, flux, err/ivar/var, mask)
    column_mapping: Dict[str, str] = Field(
        default_factory=lambda: {
            "wave": "WAVE",
            "flux": "FLUX",
            "err": "ERR",
        },
        description="Custom mapping of standard keys ('wave', 'flux', 'err', 'ivar', 'var', 'loglam', 'mask') to FITS column names.",
    )

    sky: Optional[str] = None
    mask_sky: bool = False
    fitting: FittingConfig = Field(default_factory=FittingConfig)

    H0: float = 70.0
    Om0: float = 0.3
    cosmology_model: Optional[Union[str, Cosmology]] = Field(
        default=None,
        description="Optional named Astropy cosmology (e.g., 'Planck18', 'WMAP9') or Cosmology instance.",
    )

    @field_validator("resolution", mode="before")
    @classmethod
    def validate_res(cls, v):
        return parse_astropy_quantity(v) if v is not None else v

    @property
    def cosmology(self) -> Cosmology:
        if isinstance(self.cosmology_model, Cosmology):
            return self.cosmology_model
        if isinstance(self.cosmology_model, str) and self.cosmology_model.strip():
            name = self.cosmology_model.strip()
            if hasattr(astro_cosmo, name):
                return getattr(astro_cosmo, name)
            raise IcarusConfigurationError(
                f"Unknown Astropy cosmology model '{name}'. Valid options include 'Planck18', 'Planck15', 'WMAP9'."
            )
        return FlatLambdaCDM(H0=self.H0, Om0=self.Om0)


class SetupConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    resolution: Optional[u.Quantity] = None
    dispersion_curve: Optional[str] = None
    column_mapping: Optional[Dict[str, str]] = None
    fitting: Optional[FittingConfig] = None

    @field_validator("resolution", mode="before")
    @classmethod
    def validate_quantity(cls, v):
        return parse_astropy_quantity(v) if v is not None else v


class IcarusConfig(BaseModel):
    globals: GlobalConfig = Field(default_factory=GlobalConfig)
    setups: Dict[str, SetupConfig] = Field(default_factory=dict)
    sources: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    @classmethod
    def from_preset(
        cls,
        preset: Union[str, Path, Dict[str, Any], None] = "jwst_nirspec_medium",
        **overrides: Any,
    ) -> "IcarusConfig":
        """
        Loads an IcarusConfig from a built-in preset name (e.g., 'jwst_nirspec_medium'),
        an explicit YAML file path, or a raw dictionary, and applies any keyword
        overrides directly to `globals` or `globals.fitting`.
        """
        if preset is None:
            raw_yaml: Dict[str, Any] = {}
        elif isinstance(preset, dict):
            raw_yaml = dict(preset)
        else:
            path_candidate = Path(preset)
            if not path_candidate.exists():
                preset_name = str(preset).removesuffix(".yaml").removesuffix(".yml")
                pkg_preset = PACKAGE_ROOT / "presets" / f"{preset_name}.yaml"
                if pkg_preset.exists():
                    path_candidate = pkg_preset
                else:
                    raise IcarusConfigurationError(
                        f"Could not locate preset or configuration file: '{preset}'"
                    )

            with open(path_candidate, "r") as f:
                raw_yaml = yaml.safe_load(f) or {}

        if "globals" not in raw_yaml:
            raw_yaml["globals"] = {}

        if "line_table" in raw_yaml["globals"]:
            lt_path = Path(raw_yaml["globals"]["line_table"])
            if not lt_path.exists():
                pkg_lt = PACKAGE_ROOT / "data" / lt_path.name
                if pkg_lt.exists():
                    raw_yaml["globals"]["line_table"] = str(pkg_lt)

        cfg = cls(**raw_yaml)
        if overrides:
            cfg = cfg.apply_overrides(**overrides)
        return cfg

    def apply_overrides(self, **overrides: Any) -> "IcarusConfig":
        """
        Applies runtime keyword overrides (e.g., `cosmology=Planck18`,
        `column_mapping={...}`, `snr_threshold=2.5`, `enable_feii=False`,
        `couple_narrow_v_offset=False`, `enable_speed_gatekeeper=False`)
        directly to a deep copy of this configuration.
        """
        updated = self.model_copy(deep=True)
        global_fields = set(GlobalConfig.model_fields.keys())
        fitting_fields = set(FittingConfig.model_fields.keys())

        for key, val in overrides.items():
            if val is None:
                continue
            if key == "cosmology":
                updated.globals.cosmology_model = val
            elif key == "column_mapping" and isinstance(val, dict):
                updated.globals.column_mapping = {
                    **updated.globals.column_mapping,
                    **val,
                }
            elif key in global_fields:
                if key == "resolution":
                    val = parse_astropy_quantity(val)
                setattr(updated.globals, key, val)
            elif key in fitting_fields:
                if key in ("cont_width", "min_sigma", "max_sigma"):
                    val = parse_astropy_quantity(val)
                setattr(updated.globals.fitting, key, val)

        return updated

    def get_target_config(self, target: Target) -> GlobalConfig:
        """Layers Globals -> Setup -> Source to yield a unified configuration."""
        final_config = self.globals.model_copy(deep=True)

        if target.setup and target.setup in self.setups:
            setup_conf = self.setups[target.setup]
            if setup_conf.resolution is not None:
                final_config.resolution = setup_conf.resolution
            if setup_conf.dispersion_curve is not None:
                final_config.dispersion_curve = setup_conf.dispersion_curve
            if setup_conf.column_mapping is not None:
                final_config.column_mapping = {
                    **final_config.column_mapping,
                    **setup_conf.column_mapping,
                }
            if setup_conf.fitting:
                merged_fitting = {
                    **final_config.fitting.model_dump(exclude_none=True),
                    **setup_conf.fitting.model_dump(exclude_none=True),
                }
                final_config.fitting = FittingConfig(**merged_fitting)

        if target.target_id in self.sources:
            source_overrides = self.sources[target.target_id]

            if "lines" in source_overrides:
                final_config.lines = source_overrides["lines"]

            if "column_mapping" in source_overrides:
                final_config.column_mapping = {
                    **final_config.column_mapping,
                    **source_overrides["column_mapping"],
                }

            if "fitting" in source_overrides:
                merged_fitting = {
                    **final_config.fitting.model_dump(exclude_none=True),
                    **source_overrides["fitting"],
                }
                final_config.fitting = FittingConfig(**merged_fitting)

        return final_config
