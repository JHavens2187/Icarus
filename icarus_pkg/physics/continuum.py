"""
continuum.py

Stage 1 Two-Pass Global Pseudo-Continuum Cascade (AGN Power-Law + Host Starlight +
Balmer Edge + FFT Fe II Deconvolution) and Stage 2 Local Boundary-Condition Pinning.
"""

__project__ = "ICARUS"
__author__ = "Joseph Havens"
__version__ = "3.4.0"

from typing import List, Optional, Tuple

import numpy as np
from astropy.stats import sigma_clipped_stats
from lmfit import Model
from lmfit.models import ConstantModel, LinearModel, PolynomialModel, QuadraticModel

from ..core.config import FittingConfig
from ..core.containers import EmissionLineDef, SpectralComponent, Spectrum1D
from ..core.dev_tools import DevState, dev_hook
from ..core.exceptions import IcarusKinematicError
from ..core.logger import get_icarus_logger
from .feii_model import FeIITemplateConvolver

log = get_icarus_logger()
PIVOT_WAVE_AA = 5100.0
BALMER_EDGE_AA = 3646.0


def _agn_powerlaw_host_balmer(
    x: np.ndarray,
    pl_amp: float = 1.0,
    pl_alpha: float = -1.5,
    host_amp: float = 0.0,
    host_slope: float = 1.2,
    balmer_amp: float = 0.0,
) -> np.ndarray:
    """
    AGN non-thermal power-law diluted by flexible host-galaxy starlight plus an
    optional optically thick Balmer continuum edge below 3646 A ("Small Blue Bump").
    """
    w_safe = np.maximum(np.asarray(x, dtype=np.float64), 100.0)
    pl_flux = pl_amp * ((w_safe / PIVOT_WAVE_AA) ** pl_alpha)

    # Smooth host galaxy starlight envelope (peaking in rest-frame optical/NIR ~ 1.6 um)
    host_flux = (
        host_amp * ((w_safe / PIVOT_WAVE_AA) ** host_slope) * np.exp(-w_safe / 22000.0)
    )

    # Optically thick Balmer continuum below 3646 A (Grandi 1982 approximation)
    balmer_flux = np.zeros_like(w_safe)
    if balmer_amp > 0.0:
        uv_mask = w_safe <= BALMER_EDGE_AA
        if np.any(uv_mask):
            w_uv = w_safe[uv_mask]
            ratio = w_uv / BALMER_EDGE_AA
            exponent = -(1.4388e8 / 15000.0) * (1.0 / w_uv - 1.0 / BALMER_EDGE_AA)
            balmer_flux[uv_mask] = (
                balmer_amp * (ratio**3) * np.exp(np.clip(exponent, -50.0, 0.0))
            )

    return pl_flux + host_flux + balmer_flux


class ContinuumFitter:
    """Executes Stage 1 global SED + Fe II fitting and Stage 2 local boundary pinning."""

    _FEII_CONVOLVER: Optional[FeIITemplateConvolver] = None

    @classmethod
    def _get_feii_convolver(cls) -> FeIITemplateConvolver:
        if cls._FEII_CONVOLVER is None:
            cls._FEII_CONVOLVER = FeIITemplateConvolver()
        return cls._FEII_CONVOLVER

    @staticmethod
    @dev_hook(step_name="build_polynomial_continuum")
    def build_polynomial_continuum(
        degree: int = 2, prefix: str = "cont_", flux_median: float = 1.0
    ) -> SpectralComponent:
        """Constructs a bounded polynomial continuum component (degrees 0-3)."""
        safe_median = abs(float(flux_median)) if abs(flux_median) > 1e-5 else 1e-5

        if degree == 0:
            model = ConstantModel(prefix=prefix)
            params = model.make_params(
                c=dict(value=flux_median, min=-safe_median * 5, max=safe_median * 5)
            )
        elif degree == 1:
            model = LinearModel(prefix=prefix)
            params = model.make_params(
                slope=0.0,
                intercept=dict(
                    value=flux_median, min=-safe_median * 5, max=safe_median * 5
                ),
            )
        elif degree == 2:
            model = QuadraticModel(prefix=prefix)
            params = model.make_params(
                a=0.0,
                b=0.0,
                c=dict(value=flux_median, min=-safe_median * 5, max=safe_median * 5),
            )
        elif degree == 3:
            model = PolynomialModel(degree=3, prefix=prefix)
            params = model.make_params(
                c0=dict(value=flux_median, min=-safe_median * 5, max=safe_median * 5),
                c1=0.0,
                c2=0.0,
                c3=0.0,
            )
        else:
            raise IcarusKinematicError("Continuum degree must be between 0 and 3.")

        return SpectralComponent(
            name=prefix.strip("_"),
            model=model,
            params=params,
            component_type="continuum",
        )

    @classmethod
    @dev_hook(step_name="fit_global_sed")
    def fit_global_sed(
        cls, spectrum: Spectrum1D, config: FittingConfig
    ) -> Tuple[Optional[SpectralComponent], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Executes the Stage 1 Two-Pass Global Pseudo-Continuum Cascade:
          Pass 1: Fits AGN power-law + host starlight (and compares against smooth polynomial)
                  in line-free and Fe-free anchor windows.
          Pass 2: Tightly bounds Pass 1 parameters and co-fits UV/Optical Fe II templates
                  only if statistically significant positive iron shelf flux is present.
        """
        DevState.emit("pre_stage1_continuum", locals())

        wave_full, flux_full, err_full = spectrum.to_ndarrays()
        valid_base = (~spectrum.mask) & np.isfinite(wave_full) & np.isfinite(flux_full)

        # Exclude Fe II shelf windows from Pass 1 pure-continuum anchor mask
        feii_shelf_mask = np.zeros_like(wave_full, dtype=bool)
        for fw in config.feii_windows:
            feii_shelf_mask |= valid_base & (wave_full >= fw[0]) & (wave_full <= fw[1])

        cont_mask = np.zeros_like(wave_full, dtype=bool)
        populated_windows = 0
        for window in config.continuum_windows:
            w_in = (
                valid_base
                & (~feii_shelf_mask)
                & (wave_full >= window[0])
                & (wave_full <= window[1])
            )
            if np.sum(w_in) >= 3:
                populated_windows += 1
                cont_mask |= w_in

        feii_eval = np.zeros_like(wave_full, dtype=np.float64)

        if np.sum(cont_mask) < 10 or populated_windows < 2:
            log.info(
                f"[{spectrum.target.target_id}] Sparse global anchor coverage ({populated_windows} windows). Using local boundary-condition continuum."
            )
            return None, None, feii_eval

        x_cont = wave_full[cont_mask]
        y_cont = flux_full[cont_mask]
        err_cont = err_full[cont_mask]
        weights_cont = 1.0 / np.maximum(err_cont, 1e-5)
        median_flux = max(float(np.nanmedian(np.abs(y_cont))), 1e-4)

        # Estimate empirical log-log slope across blue and red ends of anchor pixels
        n_q = max(3, len(x_cont) // 5)
        w_blue = float(np.nanmedian(x_cont[:n_q]))
        w_red = float(np.nanmedian(x_cont[-n_q:]))
        f_blue = max(float(np.nanmedian(y_cont[:n_q])), 1e-4)
        f_red = max(float(np.nanmedian(y_cont[-n_q:])), 1e-4)

        if w_red > w_blue * 1.1:
            alpha_guess = float(
                np.clip(np.log(f_red / f_blue) / np.log(w_red / w_blue), -3.0, 2.5)
            )
        else:
            alpha_guess = -1.2

        # Interpolate initial flux density at pivot wavelength (5100 A)
        amp_5100_guess = max(
            float(np.interp(PIVOT_WAVE_AA, [w_blue, w_red], [f_blue, f_red])),
            1e-4,
        )

        # --- PASS 1: Pure Continuum Anchor Fit ---
        global_cont_eval: Optional[np.ndarray] = None
        global_cont_comp: Optional[SpectralComponent] = None
        used_powerlaw = False
        best_redchi = np.inf

        if config.cont_type in ("powerlaw_feii", "powerlaw"):
            pl_model = Model(_agn_powerlaw_host_balmer, prefix="sed_")
            pl_params = pl_model.make_params()

            pl_params["sed_pl_amp"].set(
                value=amp_5100_guess * 0.7,
                min=0.0,
                max=max(median_flux * 100.0, 20.0),
            )
            pl_params["sed_pl_alpha"].set(value=alpha_guess, min=-3.5, max=3.0)
            pl_params["sed_host_amp"].set(
                value=amp_5100_guess * 0.3,
                min=0.0,
                max=max(median_flux * 100.0, 20.0),
            )
            pl_params["sed_host_slope"].set(value=1.0, min=-0.5, max=2.5)

            has_balmer_edge = np.min(x_cont) < 3600.0 and np.max(x_cont) > 3700.0
            if has_balmer_edge:
                pl_params["sed_balmer_amp"].set(
                    value=median_flux * 0.05, min=0.0, max=median_flux * 3.0
                )
            else:
                pl_params["sed_balmer_amp"].set(value=0.0, vary=False)

            try:
                res_p1 = pl_model.fit(
                    y_cont,
                    pl_params,
                    x=x_cont,
                    weights=weights_cont,
                    max_nfev=600,
                )
                if res_p1.redchi is not None and np.isfinite(res_p1.redchi):
                    global_cont_eval = res_p1.eval(x=wave_full)
                    global_cont_comp = SpectralComponent(
                        name="sed",
                        model=pl_model,
                        params=res_p1.params.copy(),
                        component_type="continuum",
                    )
                    used_powerlaw = True
                    best_redchi = float(res_p1.redchi)
            except Exception as e:
                log.debug(
                    f"[{spectrum.target.target_id}] Power-law fit fallback triggered: {e}"
                )

        # Always evaluate polynomial continuum and adopt it if power-law failed or fit poorly
        poly_comp = cls.build_polynomial_continuum(
            degree=config.continuum_degree, flux_median=median_flux
        )
        try:
            res_poly = poly_comp.model.fit(
                y_cont,
                poly_comp.params,
                x=x_cont,
                weights=weights_cont,
                max_nfev=400,
            )
            poly_redchi = (
                float(res_poly.redchi)
                if (res_poly.redchi is not None and np.isfinite(res_poly.redchi))
                else np.inf
            )
            if global_cont_eval is None or (poly_redchi < 0.75 * best_redchi):
                global_cont_eval = res_poly.eval(x=wave_full)
                global_cont_comp = SpectralComponent(
                    name="cont",
                    model=poly_comp.model,
                    params=res_poly.params.copy(),
                    component_type="continuum",
                )
                used_powerlaw = False
                best_redchi = poly_redchi
        except Exception as e:
            if global_cont_eval is None:
                log.warning(
                    f"[{spectrum.target.target_id}] Stage 1 global continuum warning: {e}"
                )
                return None, None, feii_eval

        # --- PASS 2: Empirical Fe II Pseudo-Continuum Co-Fit ---
        if (
            config.enable_feii
            and config.cont_type == "powerlaw_feii"
            and global_cont_comp is not None
            and global_cont_eval is not None
        ):
            if np.sum(feii_shelf_mask) >= 12:
                y_fe_res = (
                    flux_full[feii_shelf_mask] - global_cont_eval[feii_shelf_mask]
                )
                e_fe = err_full[feii_shelf_mask]
                med_res = float(np.nanmedian(y_fe_res))
                med_err = max(float(np.nanmedian(e_fe)), 1e-5)

                # Only trigger Fe II template convolution if median residual in iron shelf clears 1.5 sigma
                if med_res > 1.5 * med_err:
                    fe_guess = float(np.nanpercentile(y_fe_res[y_fe_res > 0], 80))
                    convolver = cls._get_feii_convolver()
                    fe_comp = convolver.build_lmfit_model(
                        uv_amp_guess=fe_guess,
                        opt_amp_guess=fe_guess,
                        max_amp=max(median_flux * 3.0, 1.0),
                    )

                    x_fe_only = wave_full[feii_shelf_mask]
                    if np.min(x_fe_only) > 3500.0:
                        fe_comp.params["feii_uv_amp"].set(value=0.0, vary=False)
                    if np.max(x_fe_only) < 3500.0:
                        fe_comp.params["feii_opt_amp"].set(value=0.0, vary=False)

                    try:
                        w_fe = 1.0 / np.maximum(e_fe, 1e-5)
                        fe_res = fe_comp.model.fit(
                            y_fe_res,
                            fe_comp.params,
                            x=x_fe_only,
                            weights=w_fe,
                            max_nfev=300,
                        )
                        # Require BIC improvement before keeping Fe II template
                        null_chisq = float(np.sum((y_fe_res * w_fe) ** 2))
                        if fe_res.chisqr < (null_chisq - 15.0):
                            feii_eval = fe_res.eval(x=wave_full)
                    except Exception as e:
                        log.debug(
                            f"[{spectrum.target.target_id}] Fe II template co-fit skipped: {e}"
                        )

        DevState.emit("post_stage1_continuum", locals())
        return global_cont_comp, global_cont_eval, feii_eval

    @staticmethod
    @dev_hook(step_name="build_local_continuum")
    def build_local_boundary_continuum(
        sliced_spec: Spectrum1D,
        lines: List[EmissionLineDef],
        prefix: str = "cont_",
    ) -> SpectralComponent:
        """
        Stage 2 Local Boundary-Condition Pinning:
        Constructs a local linear continuum anchored directly to the left and right
        line-free boundary edges (25% margins) of the sliced complex window so
        Gaussian wings strictly obey local data boundary conditions.
        """
        x, y, err = sliced_spec.to_ndarrays()
        valid = (~sliced_spec.mask) & np.isfinite(x) & np.isfinite(y)
        x_val, y_val, err_val = x[valid], y[valid], err[valid]

        line_free_mask = np.ones_like(x_val, dtype=bool)
        for line in lines:
            line_free_mask &= np.abs(x_val - line.rest_wavelength) > 25.0

        if np.sum(line_free_mask) >= 6:
            x_lf = x_val[line_free_mask]
            y_lf = y_val[line_free_mask]
            e_lf = err_val[line_free_mask]
        else:
            x_lf, y_lf, e_lf = x_val, y_val, err_val

        n_pts = len(x_lf)
        edge_n = max(2, int(n_pts * 0.25))

        x_left = float(np.nanmedian(x_lf[:edge_n]))
        _, y_left, std_left = sigma_clipped_stats(y_lf[:edge_n], sigma=2.5, maxiters=3)
        x_right = float(np.nanmedian(x_lf[-edge_n:]))
        _, y_right, std_right = sigma_clipped_stats(
            y_lf[-edge_n:], sigma=2.5, maxiters=3
        )

        local_rms = max(float(np.nanmedian([std_left, std_right])), 1e-4)

        model = LinearModel(prefix=prefix)
        if abs(x_right - x_left) > 5.0:
            slope_init = (y_right - y_left) / (x_right - x_left)
            intercept_init = y_left - slope_init * x_left

            params = model.make_params(slope=slope_init, intercept=intercept_init)
            try:
                weights = 1.0 / np.maximum(e_lf, 1e-5)
                pre_res = model.fit(y_lf, params, x=x_lf, weights=weights, max_nfev=200)
                slope_init = float(pre_res.params[f"{prefix}slope"].value)
                intercept_init = float(pre_res.params[f"{prefix}intercept"].value)
            except Exception:
                pass

            slope_tol = max(
                abs(slope_init) * 0.25, (2.0 * local_rms) / abs(x_right - x_left)
            )
            int_tol = max(abs(intercept_init) * 0.15, 2.0 * local_rms)

            params[f"{prefix}slope"].set(
                value=slope_init,
                min=slope_init - slope_tol,
                max=slope_init + slope_tol,
            )
            params[f"{prefix}intercept"].set(
                value=intercept_init,
                min=intercept_init - int_tol,
                max=intercept_init + int_tol,
            )
        else:
            _, med_y, _ = sigma_clipped_stats(y_lf, sigma=2.5, maxiters=3)
            safe_med = max(abs(float(med_y)), 1e-4)
            params = model.make_params(
                slope=dict(value=0.0, vary=False),
                intercept=dict(
                    value=float(med_y),
                    min=float(med_y) - 2.0 * safe_med,
                    max=float(med_y) + 2.0 * safe_med,
                ),
            )

        return SpectralComponent(
            name=prefix.strip("_"),
            model=model,
            params=params,
            component_type="continuum",
        )
