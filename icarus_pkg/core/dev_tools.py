"""
dev_tools.py

Middleware debugging, stress-testing, and benchmarking engine for ICARUS v3.4.
Facilitates non-destructive pipeline instrumentation via the @dev_hook
decorator and the DevState event emitter.

Supported Directives:
  - TRACK_OPTIMIZER     : Records hardware-agnostic LMFIT metrics (nfev, nvarys, ndata, redchi, bic, covar_cond).
  - ABERRATE_INSTRUMENT : Applies wavelength/velocity shifts, LSF scaling, or continuum ripples to Spectrum1D.
  - INJECT_ARTIFACT     : Injects cosmic-ray spikes or detector chip-gap NaN troughs into Spectrum1D.
  - DUMP_TELEMETRY      : Serializes all collected runtime, memory, optimizer, and watch telemetry to JSON/CSV.
  - SYNTH_INJECT        : Injects synthetic Gaussian emission lines at specified rest wavelengths.
  - PERTURB             : Executes Monte Carlo Gaussian/Poisson noise realizations over a pipeline step.
  - FORCE_FAIL          : Raises IcarusConvergenceError at a designated step to test fallback pathways.
  - WATCH               : Tracks state mutations of designated local variables across inline emit events.
  - TRACE_MEMORY        : Profiles heap memory high-water marks (MB) per step via tracemalloc.
  - PROFILE             : Profiles wall-clock execution time (ms) per step.
"""

__project__ = "ICARUS"
__author__ = "Joseph Havens"
__version__ = "3.4.0"

import ast
import atexit
import functools
import json
import os
import pprint
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# Attempt to load IPython for interactive pauses, fallback to native pdb
try:
    from IPython import embed

    HAS_IPYTHON = True
except ImportError:
    import pdb

    HAS_IPYTHON = False

from .exceptions import IcarusConvergenceError
from .logger import get_icarus_logger

log = get_icarus_logger("DEV_ENGINE")
C_KMS = 299792.458


def _find_spectrum(args: tuple, kwargs: dict, result: Any = None) -> Optional[Any]:
    """
    Locates a `Spectrum1D` instance across positional args, keyword args,
    `AGNFitter.spectrum` (`self.spectrum`), or hook return values.
    """
    for item in (
        list(args) + list(kwargs.values()) + ([result] if result is not None else [])
    ):
        if item is None:
            continue
        if (
            hasattr(item, "flux")
            and hasattr(item, "wavelength")
            and hasattr(item, "uncertainty")
        ):
            return item
        if hasattr(item, "spectrum"):
            sub = getattr(item, "spectrum")
            if hasattr(sub, "flux") and hasattr(sub, "wavelength"):
                return sub
    return None


def _extract_optimizer_metrics(
    fit_res: Any, label: str = ""
) -> Optional[Dict[str, Any]]:
    """Extracts hardware-agnostic optimization metrics from an lmfit ModelResult."""
    if (
        fit_res is None
        or not hasattr(fit_res, "nfev")
        or not hasattr(fit_res, "params")
    ):
        return None

    covar_cond = float("nan")
    covar = getattr(fit_res, "covar", None)
    if covar is not None:
        try:
            cov_arr = np.asarray(covar, dtype=np.float64)
            if cov_arr.ndim == 2 and cov_arr.size > 0 and np.all(np.isfinite(cov_arr)):
                covar_cond = float(np.linalg.cond(cov_arr))
        except Exception:
            pass

    return {
        "label": label,
        "success": bool(getattr(fit_res, "success", False)),
        "nfev": int(getattr(fit_res, "nfev", 0)),
        "nvarys": int(getattr(fit_res, "nvarys", 0)),
        "ndata": int(getattr(fit_res, "ndata", 0)),
        "chisqr": float(getattr(fit_res, "chisqr", np.nan)),
        "redchi": float(getattr(fit_res, "redchi", np.nan)),
        "aic": float(getattr(fit_res, "aic", np.nan)),
        "bic": float(getattr(fit_res, "bic", np.nan)),
        "covar_cond": covar_cond,
    }


class _DevState:
    """Singleton managing global developer state, active directives, and telemetry."""

    def __init__(self):
        self.dev_file: Optional[Path] = None
        self.is_active: bool = False
        self.step_mode: bool = False
        self.dag_registry: Dict[str, str] = {}

        self.directives: Dict[str, Dict[str, Dict[str, Any]]] = {
            "DEBUG": {},
            "PAUSE": {},
            "BYPASS": {},
            "ISOLATE": {},
            "INJECT": {},
            "SYNTH_ISOLATE": {},
            "SYNTH_INJECT": {},
            "WATCH": {},
            "VISUALIZE": {},
            "PERTURB": {},
            "FORCE_FAIL": {},
            "DUMP_DAG": {},
            "PROFILE": {},
            "TRACE_MEMORY": {},
            "TRACK_OPTIMIZER": {},
            "ABERRATE_INSTRUMENT": {},
            "INJECT_ARTIFACT": {},
            "DUMP_TELEMETRY": {},
        }

        self.telemetry: Dict[str, Any] = {
            "time": {},
            "memory": {},
            "optimizer": [],
            "watch": [],
            "perturb": {},
        }
        self._shadow_watch: Dict[str, Any] = {}
        atexit.register(self.flush_telemetry)

    def initialize(self, path: Path) -> None:
        """Explicit activation trigger called by cli.py or interactive sessions."""
        self.dev_file = Path(path)
        if self.dev_file.exists() and self.dev_file.stat().st_size > 0:
            self._parse_dev_file()
            self.is_active = True

    def _parse_dev_file(self) -> None:
        """Ingests the .dev file and populates the directives routing table."""
        if not self.dev_file:
            return

        with open(self.dev_file, "r") as f:
            lines = f.readlines()

        parsed_command_count = 0

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if ":" not in line:
                log.warning(
                    f"DEV_TOOL: Skipping malformed line (missing colon): {line}"
                )
                continue

            try:
                cmd_part, rest = line.split(":", 1)
                cmd = cmd_part.strip().upper()

                if cmd not in self.directives:
                    continue

                args_dict: Dict[str, Any] = {}
                step_name = "GLOBAL"

                if "@" in rest:
                    args_part, step_name = rest.split("@", 1)
                    step_name = step_name.strip()
                else:
                    args_part = rest

                for arg in args_part.split():
                    if "=" in arg:
                        k, v = arg.split("=", 1)
                        try:
                            args_dict[k] = ast.literal_eval(v)
                        except (ValueError, SyntaxError):
                            args_dict[k] = v
                    else:
                        step_name = arg.strip()

                if step_name not in self.directives[cmd]:
                    self.directives[cmd][step_name] = {}

                self.directives[cmd][step_name].update(args_dict)
                parsed_command_count += 1

            except Exception as e:
                log.error(f"Failed to parse .dev instruction: '{line}' | Error: {e}")

        if parsed_command_count > 0:
            self.is_active = True
            log.warning(
                f"DEVELOPMENT MODE ACTIVE ({parsed_command_count} directives). Multiprocessing bypass engaged."
            )

    def _is_enabled(self, directive: str, name: str) -> Optional[Dict[str, Any]]:
        """Returns directive arguments if active for `name` or `GLOBAL`."""
        d_map = self.directives.get(directive, {})
        if name in d_map:
            return d_map[name]
        if "GLOBAL" in d_map:
            return d_map["GLOBAL"]
        return None

    def run_preflight(self) -> None:
        """Handles non-runtime directives like DUMP_DAG."""
        if not self.is_active:
            return

        if "DUMP_DAG" in self.directives and self.directives["DUMP_DAG"]:
            out_file = self.directives["DUMP_DAG"].get("GLOBAL", {}).get("OUT", None)

            dag_str = "=" * 50 + "\n"
            dag_str += "ICARUS v3.4 - DIRECTED ACYCLIC GRAPH (HOOKS)\n"
            dag_str += "=" * 50 + "\n"
            for step, mod in sorted(self.dag_registry.items()):
                dag_str += f"[{mod}] -> {step}\n"

            if out_file:
                with open(out_file, "w") as f:
                    f.write(dag_str)
                print(f"DAG map successfully dumped to {out_file}")
            else:
                print(dag_str)

            sys.exit(0)

    def flush_telemetry(self) -> Optional[Path]:
        """Writes collected telemetry to disk if DUMP_TELEMETRY is active."""
        if not self.is_active or not self.directives.get("DUMP_TELEMETRY"):
            return None

        cfg = self._is_enabled("DUMP_TELEMETRY", "GLOBAL") or next(
            iter(self.directives["DUMP_TELEMETRY"].values()), {}
        )
        out_str = cfg.get("OUT", cfg.get("out", "icarus_dev_telemetry.json"))
        out_path = Path(str(out_str))
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(out_path, "w") as f:
                json.dump(self.telemetry, f, indent=2, default=str)
            log.info(f"[DUMP_TELEMETRY] Wrote telemetry payload to {out_path}")
            return out_path
        except Exception as e:
            log.error(f"[DUMP_TELEMETRY] Failed to serialize telemetry: {e}")
            return None

    def trigger_pause(self, msg: str, local_vars: dict) -> None:
        """Drops into an interactive shell with a 'next_step' hook injected."""
        print(f"\n{'='*50}\n{msg}\n{'='*50}")
        if HAS_IPYTHON:

            def next_step():
                self.step_mode = True
                print("Step mode activated. Proceeding to the next @dev_hook...")
                from IPython import get_ipython

                if get_ipython() is not None:
                    get_ipython().ask_exit()

            local_vars["next_step"] = next_step
            local_vars["DevState"] = self
            embed(
                header="Type 'exit' to resume normal execution.\nType 'next_step()' to pause at the very next @dev_hook.",
                user_ns=local_vars,
            )
        else:
            print("Type 'c' to continue, or 'n' to step lines in pdb.")
            pdb.set_trace()

    def _apply_aberrations(
        self, spec: Any, args_dict: Dict[str, Any], step_name: str
    ) -> None:
        """Applies instrumental wavelength/velocity shifts, LSF scaling, or continuum ripples."""
        w = spec.wavelength.value
        f = spec.flux.value

        v_shift = float(args_dict.get("v_shift_kms", 0.0))
        w_shift = float(args_dict.get("wave_shift_aa", 0.0))
        if v_shift != 0.0:
            w[:] = w * (1.0 + v_shift / C_KMS)
        if w_shift != 0.0:
            w[:] = w + w_shift

        lsf_scale = float(args_dict.get("lsf_scale", 1.0))
        if lsf_scale != 1.0 and spec.resolution_kms is not None:
            spec.resolution_kms[:] = spec.resolution_kms * lsf_scale

        rip_amp = float(args_dict.get("ripple_amp", 0.0))
        rip_period = float(args_dict.get("ripple_period_aa", 250.0))
        if rip_amp != 0.0 and rip_period > 0:
            med_f = (
                float(np.nanmedian(np.abs(f[np.isfinite(f)])))
                if np.any(np.isfinite(f))
                else 1.0
            )
            f[:] = f + rip_amp * med_f * np.sin(2.0 * np.pi * w / rip_period)

        print(
            f"[ABERRATE_INSTRUMENT @ {step_name}] Applied v_shift={v_shift} km/s, "
            f"wave_shift={w_shift} A, lsf_scale={lsf_scale}, ripple_amp={rip_amp}"
        )

    def _apply_artifacts(
        self, spec: Any, args_dict: Dict[str, Any], step_name: str
    ) -> None:
        """Injects cosmic-ray spikes or detector chip-gap NaN troughs into `spec`."""
        w = spec.wavelength.value
        f = spec.flux.value
        e = spec.uncertainty.value
        mode = str(args_dict.get("mode", args_dict.get("MODE", "cosmic_ray"))).lower()

        if mode == "cosmic_ray":
            snr_spike = float(args_dict.get("snr", 100.0))
            target_wave = args_dict.get("wave", None)
            count = int(args_dict.get("count", 1))
            width_pix = int(args_dict.get("width_pix", 1))

            if target_wave is not None:
                waves = (
                    [target_wave] if not isinstance(target_wave, list) else target_wave
                )
                indices = [int(np.argmin(np.abs(w - float(tw)))) for tw in waves]
            else:
                valid_idx = np.where(~spec.mask & np.isfinite(f))[0]
                indices = (
                    np.random.choice(
                        valid_idx, size=min(count, len(valid_idx)), replace=False
                    ).tolist()
                    if len(valid_idx) > 0
                    else []
                )

            for idx in indices:
                lo = max(0, idx - width_pix // 2)
                hi = min(len(f), lo + max(1, width_pix))
                local_err = max(float(np.nanmedian(e[lo:hi])), 1e-4)
                f[lo:hi] += snr_spike * local_err

            print(
                f"[INJECT_ARTIFACT @ {step_name}] Injected {len(indices)} cosmic-ray spike(s) ({snr_spike} sigma)."
            )

        elif mode in ("chip_gap", "nan_trough"):
            center_w = float(args_dict.get("wave", np.nanmedian(w)))
            width_aa = float(args_dict.get("width_aa", 30.0))
            gap_mask = np.abs(w - center_w) <= (width_aa / 2.0)
            f[gap_mask] = np.nan
            spec.mask[gap_mask] = True
            print(
                f"[INJECT_ARTIFACT @ {step_name}] Masked chip gap at {center_w:.1f} +/- {width_aa/2:.1f} A ({int(np.sum(gap_mask))} pixels)."
            )

    def emit(self, event_name: str, context_locals: Optional[Dict] = None) -> None:
        """Inline event listener for mid-logic breakpoints, optimizer tracking, and variable watches."""
        if not self.is_active:
            return

        context = context_locals or {}

        # 1. Handle Inline Optimizer Tracking (e.g. post_narrow_fit, post_broad_fit)
        if self._is_enabled("TRACK_OPTIMIZER", event_name) is not None:
            comp_obj = context.get("comp")
            comp_name = getattr(comp_obj, "name", event_name)
            for res_key in (
                "result_a",
                "result_b",
                "winning_fit",
                "res_p1",
                "co_res",
                "fe_res",
            ):
                if res_key in context:
                    opt_meta = _extract_optimizer_metrics(
                        context[res_key], label=f"{event_name}:{comp_name}:{res_key}"
                    )
                    if opt_meta is not None:
                        self.telemetry["optimizer"].append(opt_meta)

        # 2. Handle Inline Pauses
        if event_name in self.directives["PAUSE"]:
            args = self.directives["PAUSE"][event_name]
            condition = args.get("CONDITION", "True")
            try:
                if eval(condition, globals(), context):
                    self.trigger_pause(f"DEV PAUSE INLINE EVENT: {event_name}", context)
            except Exception as e:
                log.error(f"Pause condition '{condition}' failed evaluation: {e}")

        # 3. Handle Inline Visualizations
        if event_name in self.directives["VISUALIZE"]:
            args = self.directives["VISUALIZE"][event_name]
            save_path = args.get("SAVE", None)
            import matplotlib.pyplot as plt

            fig = context.get("fig") or plt.gcf()
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                full_path = os.path.join(save_path, f"vis_{event_name}.png")
                fig.savefig(full_path, bbox_inches="tight")
                print(f"[VISUALIZE] Saved trace to {full_path}")
            else:
                plt.show(block=True)

        # 4. Handle Variable Watches
        for var_name, args in self.directives["WATCH"].items():
            if var_name in context:
                current_val = context[var_name]
                prev_val = self._shadow_watch.get(var_name, None)

                changed = False
                if isinstance(current_val, np.ndarray) and isinstance(
                    prev_val, np.ndarray
                ):
                    if current_val.shape == prev_val.shape:
                        changed = not np.array_equal(current_val, prev_val)
                    else:
                        changed = True
                else:
                    changed = current_val != prev_val

                if changed:
                    msg = f"[WATCH] '{var_name}' altered at '{event_name}':\n  OLD: {prev_val}\n  NEW: {current_val}"
                    self.telemetry["watch"].append(
                        {
                            "event": event_name,
                            "variable": var_name,
                            "new": str(current_val),
                        }
                    )
                    log_file = args.get("LOG", None)
                    if log_file:
                        with open(log_file, "a") as f:
                            f.write(msg + "\n")
                    else:
                        print(msg)
                    self._shadow_watch[var_name] = (
                        np.copy(current_val)
                        if isinstance(current_val, np.ndarray)
                        else current_val
                    )

    def execute_hook(
        self, step_name: str, func: Callable, args: tuple, kwargs: dict
    ) -> Any:
        """Master routing switchboard for the @dev_hook decorator."""
        if step_name in self.directives["FORCE_FAIL"]:
            log.critical(f"DEV_TOOL: Forcing artificial failure at {step_name}")
            raise IcarusConvergenceError(
                f"Artificial dev forced failure at {step_name}"
            )

        if step_name in self.directives["BYPASS"]:
            ret_str = self.directives["BYPASS"][step_name].get("RETURN", "None")
            log.warning(
                f"DEV_TOOL: Bypassing {step_name}, returning injected structure."
            )
            return ast.literal_eval(ret_str)

        if step_name in self.directives["INJECT"]:
            inject_args = self.directives["INJECT"][step_name]
            log.info(f"DEV_TOOL: Overwriting kwargs for {step_name}: {inject_args}")
            kwargs.update(inject_args)

        spec_in = _find_spectrum(args, kwargs)

        # Pre-execution Instrument Aberration
        aberrate_cfg = self.directives["ABERRATE_INSTRUMENT"].get(step_name)
        if aberrate_cfg is not None and spec_in is not None:
            self._apply_aberrations(spec_in, aberrate_cfg, step_name)

        # Pre-execution Detector Artifact Injection
        artifact_cfg = self.directives["INJECT_ARTIFACT"].get(step_name)
        if artifact_cfg is not None and spec_in is not None:
            self._apply_artifacts(spec_in, artifact_cfg, step_name)

        # Pre-execution Synthetic Emission Line Injection
        for cmd in ["SYNTH_INJECT", "SYNTH_ISOLATE"]:
            if step_name in self.directives[cmd] and spec_in is not None:
                args_dict = self.directives[cmd][step_name]
                flux_val = args_dict.get("flux", 0.0)
                wave_val = args_dict.get("wave", 0.0)
                sigma_val = args_dict.get("sigma", 2.0)

                if not isinstance(flux_val, list):
                    flux_val = [flux_val]
                if not isinstance(wave_val, list):
                    wave_val = [wave_val]
                if not isinstance(sigma_val, list):
                    sigma_val = [sigma_val]

                w = spec_in.wavelength.value
                f = spec_in.flux.value
                for fw, ww, sw in zip(flux_val, wave_val, sigma_val):
                    spike = float(fw) * np.exp(
                        -0.5 * ((w - float(ww)) / float(sw)) ** 2
                    )
                    f[:] = f + spike
                print(
                    f"[{cmd} @ {step_name}] Injected synthetic emission features at {wave_val} A"
                )

                if cmd == "SYNTH_ISOLATE":
                    self.directives["ISOLATE"]["GLOBAL"] = {"wave": wave_val}

        profile_active = self._is_enabled("PROFILE", step_name) is not None
        if profile_active:
            t_start = time.perf_counter()

        mem_active = self._is_enabled("TRACE_MEMORY", step_name) is not None
        if mem_active:
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            m_start, _ = tracemalloc.get_traced_memory()

        pause_args = self.directives["PAUSE"].get(step_name, {})
        should_pause_before = self.step_mode or "-b" in pause_args

        if should_pause_before:
            condition = pause_args.get("CONDITION", "True")
            try:
                if eval(condition, globals(), locals()):
                    self.step_mode = False
                    self.trigger_pause(f"DEV PAUSE BEFORE: {step_name}", locals())
            except Exception as e:
                log.error(f"Pause condition '{condition}' failed: {e}")

        if step_name in self.directives["PERTURB"]:
            p_cfg = self.directives["PERTURB"][step_name]
            iterations = int(p_cfg.get("iterations", 5))
            mode = str(p_cfg.get("MODE", p_cfg.get("mode", "gaussian"))).lower()
            snr_scale = float(p_cfg.get("noise_scale", 1.0))
            print(
                f"[PERTURB] Running {step_name} {iterations} times with {mode} noise (scale={snr_scale}x)..."
            )

            original_flux = np.copy(spec_in.flux.value) if spec_in is not None else None
            results_pool: List[Any] = []

            for _ in range(iterations):
                if spec_in is not None and original_flux is not None:
                    err = np.maximum(spec_in.uncertainty.value, 1e-5) * snr_scale
                    if mode == "gaussian":
                        noise = np.random.normal(loc=0.0, scale=err)
                    elif mode == "poisson":
                        lam = np.maximum(np.abs(original_flux), 1e-4)
                        noise = (np.random.poisson(lam=lam) - lam) * snr_scale
                    else:
                        noise = 0.0
                    spec_in.flux.value[:] = original_flux + noise

                res = func(*args, **kwargs)
                results_pool.append(res)

            if spec_in is not None and original_flux is not None:
                spec_in.flux.value[:] = original_flux

            self.telemetry["perturb"][step_name] = {
                "iterations": iterations,
                "mode": mode,
                "noise_scale": snr_scale,
            }
            result = results_pool[-1]
        else:
            result = func(*args, **kwargs)

        # Post-execution hooks when `build_spectrum` returns a newly created Spectrum1D
        if spec_in is None:
            spec_out = _find_spectrum((), {}, result=result)
            if spec_out is not None:
                if aberrate_cfg is not None:
                    self._apply_aberrations(spec_out, aberrate_cfg, step_name)
                if artifact_cfg is not None:
                    self._apply_artifacts(spec_out, artifact_cfg, step_name)

        # Record Optimizer Telemetry if function returned a ModelResult
        if self._is_enabled("TRACK_OPTIMIZER", step_name) is not None:
            comp_arg = next(
                (a for a in args if hasattr(a, "lines") and hasattr(a, "name")), None
            )
            lbl = f"{step_name}:{getattr(comp_arg, 'name', '')}".rstrip(":")
            opt_meta = _extract_optimizer_metrics(result, label=lbl)
            if opt_meta is not None:
                self.telemetry["optimizer"].append(opt_meta)

        if "-a" in pause_args:
            condition = pause_args.get("CONDITION", "True")
            try:
                if eval(condition, globals(), locals()):
                    self.trigger_pause(f"DEV PAUSE AFTER: {step_name}", locals())
            except Exception as e:
                log.error(f"Pause condition '{condition}' failed: {e}")

        if profile_active:
            elapsed = (time.perf_counter() - t_start) * 1000.0
            self.telemetry["time"].setdefault(step_name, []).append(elapsed)
            print(f"[PROFILE] {step_name}: {elapsed:.2f} ms")

        if mem_active:
            m_end, m_peak = tracemalloc.get_traced_memory()
            delta_mb = (m_end - m_start) / (1024 * 1024)
            peak_mb = m_peak / (1024 * 1024)
            self.telemetry["memory"][step_name] = {
                "delta_mb": delta_mb,
                "peak_mb": peak_mb,
            }
            print(f"[MEMORY] {step_name}: {delta_mb:+.3f} MB (Peak: {peak_mb:.3f} MB)")

        if step_name in self.directives["DEBUG"]:
            depth = int(self.directives["DEBUG"][step_name].get("DEPTH", 2))
            print(f"\n{'='*50}\nDEBUG TRACE: {step_name}\n{'='*50}")
            print("--- KWARGS ---")
            pprint.pprint(kwargs, depth=depth)
            print("--- RETURN ---")
            pprint.pprint(result, depth=depth)
            print("==================================================\n")

        return result


DevState = _DevState()


def dev_hook(step_name: Optional[str] = None):
    """
    Function wrapper that registers the step in the DAG and routes execution
    through the DevState engine if a .dev file is active.
    """

    def decorator(func: Callable):
        name = step_name or func.__name__
        DevState.dag_registry[name] = func.__module__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not DevState.is_active:
                return func(*args, **kwargs)
            return DevState.execute_hook(name, func, args, kwargs)

        return wrapper

    return decorator
