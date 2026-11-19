"""
line_models.py

Relative velocity-space Gaussian component factories for ICARUS v3.4.
Parameterizes all emission transitions exclusively in relative velocity space
(v_offset and sigma_v in km/s) and enforces strict roadmap kinematic boundaries.
"""

__project__ = "ICARUS"
__author__ = "Joseph Havens"
__version__ = "3.4.0"

from typing import Any, Optional

from astropy import constants as const
from astropy import units as u
from lmfit.models import GaussianModel

from ..core.containers import SpectralComponent
from ..core.dev_tools import DevState, dev_hook
from ..core.logger import get_icarus_logger

log = get_icarus_logger()
C_KMS = const.c.to("km/s").value

# High-ionization UV lines subject to strong radiation-driven winds
UV_WIND_IONS = (
    "C_IV_1548",
    "C_IV_1551",
    "He_II_1640",
    "Si_IV_1397",
    "O_IV_1402",
    "N_V_1240",
)


class LineComponentFactory:
    """Constructs velocity-parameterized Gaussian LMFIT components."""

    @staticmethod
    @dev_hook(step_name="build_emission_line")
    def build_emission_line(
        name: str,
        rest_wavelength: float,
        is_broad: bool = False,
        amplitude_guess: float = 1e-15,
        config: Optional[Any] = None,
        velocity_bound_kms: float = 600.0,
        sigma_init_kms: float = 120.0,
        v_init_kms: float = 0.0,
    ) -> SpectralComponent:
        """
        Builds a narrow or broad emission-line Gaussian model locked in velocity space:
          center = rest_wavelength * (1.0 + v_offset / C_KMS)
          sigma  = rest_wavelength * (sigma_v  / C_KMS)
        """
        DevState.emit("pre_build_emission_line", locals())

        prefix = f"{name}_"
        model = GaussianModel(prefix=prefix)
        params = model.make_params()

        if is_broad:
            # Expand velocity offset window for high-ionization UV wind transitions
            is_uv_wind = any(uv in name for uv in UV_WIND_IONS)
            v_min = -4000.0 if is_uv_wind else -800.0
            v_max = 1000.0 if is_uv_wind else 800.0

            b_min_sig = 700.0
            b_max_sig = 8000.0
            if config is not None and getattr(config, "broad_kinematics", None):
                bk = config.broad_kinematics
                if "min_sigma" in bk and bk["min_sigma"] is not None:
                    b_min_sig = float(u.Quantity(bk["min_sigma"]).to(u.km / u.s).value)
                if "max_sigma" in bk and bk["max_sigma"] is not None:
                    b_max_sig = float(u.Quantity(bk["max_sigma"]).to(u.km / u.s).value)

            v0 = float(min(max(v_init_kms, v_min * 0.8), v_max * 0.8))
            params.add(f"{prefix}v_offset", value=v0, min=v_min, max=v_max)
            params.add(
                f"{prefix}sigma_v",
                value=max(b_min_sig * 1.5, 1300.0),
                min=b_min_sig,
                max=b_max_sig,
            )
            comp_type = "emission_broad"
        else:
            n_min_sig = 30.0
            n_max_sig = 450.0
            if config is not None:
                if getattr(config, "min_sigma", None) is not None:
                    n_min_sig = float(u.Quantity(config.min_sigma).to(u.km / u.s).value)
                if getattr(config, "max_sigma", None) is not None:
                    n_max_sig = float(u.Quantity(config.max_sigma).to(u.km / u.s).value)

            v0 = float(
                min(
                    max(v_init_kms, -velocity_bound_kms * 0.85),
                    velocity_bound_kms * 0.85,
                )
            )
            params.add(
                f"{prefix}v_offset",
                value=v0,
                min=-velocity_bound_kms,
                max=velocity_bound_kms,
            )
            params.add(
                f"{prefix}sigma_v",
                value=min(max(sigma_init_kms, n_min_sig * 1.2), n_max_sig * 0.8),
                min=n_min_sig,
                max=n_max_sig,
            )
            comp_type = "emission_narrow"

        params[f"{prefix}center"].set(
            expr=f"{rest_wavelength:.6f} * (1.0 + {prefix}v_offset / {C_KMS:.6f})"
        )
        params[f"{prefix}sigma"].set(
            expr=f"{rest_wavelength:.6f} * ({prefix}sigma_v / {C_KMS:.6f})"
        )
        params[f"{prefix}amplitude"].set(
            value=max(float(amplitude_guess), 1e-15), min=0.0
        )

        comp = SpectralComponent(
            name=name,
            model=model,
            params=params,
            component_type=comp_type,
            rest_wavelength=rest_wavelength,
        )
        DevState.emit("post_build_emission_line", locals())
        return comp

    @staticmethod
    @dev_hook(step_name="build_outflow_line")
    def build_outflow_line(
        name: str,
        rest_wavelength: float,
        amplitude_guess: float = 1e-15,
        v_sys_kms: float = 0.0,
    ) -> SpectralComponent:
        """
        Builds an asymmetric, blue-shifted ionized gas wind component for primary
        forbidden transitions ([O III] 4959/5007).
        Strictly enforces blueshift relative to the narrow systemic core:
        v_offset in [v_sys - 2000, v_sys - 150] km/s and sigma_v in [250, 765] km/s.
        """
        DevState.emit("pre_build_outflow_line", locals())

        prefix = f"{name}_"
        model = GaussianModel(prefix=prefix)
        params = model.make_params()

        v_high = float(v_sys_kms - 150.0)
        v_low = float(v_sys_kms - 2000.0)
        v_init = float(v_sys_kms - 400.0)

        params.add(f"{prefix}v_offset", value=v_init, min=v_low, max=v_high)
        params.add(f"{prefix}sigma_v", value=420.0, min=250.0, max=765.0)

        params[f"{prefix}center"].set(
            expr=f"{rest_wavelength:.6f} * (1.0 + {prefix}v_offset / {C_KMS:.6f})"
        )
        params[f"{prefix}sigma"].set(
            expr=f"{rest_wavelength:.6f} * ({prefix}sigma_v / {C_KMS:.6f})"
        )
        params[f"{prefix}amplitude"].set(
            value=max(float(amplitude_guess), 1e-15), min=0.0
        )

        comp = SpectralComponent(
            name=name,
            model=model,
            params=params,
            component_type="emission_outflow",
            rest_wavelength=rest_wavelength,
        )
        DevState.emit("post_build_outflow_line", locals())
        return comp
