"""
cli.py

Command-Line Interface and ProcessPoolExecutor Cluster Orchestrator for ICARUS v3.4.
Supports multi-target survey catalogs (--catalog & --data), catalog-free
single-file execution (--file & --redshift), and an interactive terminal wizard
when invoked with zero arguments.
"""

__project__ = "ICARUS"
__author__ = "Joseph Havens"
__version__ = "3.4.0"

import logging
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Enforce headless Agg backend strictly inside the CLI entry point so interactive
# Jupyter Notebook sessions importing `icarus_pkg` retain inline display capabilities.
import matplotlib

matplotlib.use("Agg")

import click
import pandas as pd
from astropy.table import QTable, vstack
from astropy.utils.exceptions import AstropyWarning
from colorama import Fore, Style, init

from .api import AGNFitter
from .core.config import IcarusConfig
from .core.containers import Target
from .core.dev_tools import DevState
from .core.exceptions import IcarusConfigurationError
from .core.ingest import CatalogRegistry, ObservationBuilder
from .core.logger import get_icarus_logger

warnings.filterwarnings("ignore", category=FutureWarning)
init(autoreset=True)
log = get_icarus_logger()

BUILTIN_PRESETS: Dict[str, Tuple[str, str]] = {
    "1": ("jwst_nirspec_medium", "JWST NIRSpec Medium Gratings (G140M/G235M/G395M)"),
    "2": ("jwst_nirspec_prism", "JWST NIRSpec Prism/Clear (Wide-band Low-Res)"),
    "3": ("jwst_nirspec_high", "JWST NIRSpec High-Res Gratings (G140H/G235H/G395H)"),
    "4": ("roman_wfi", "Nancy Grace Roman WFI Grism"),
    "5": ("sdss_boss", "SDSS / BOSS Optical Benchmark"),
}


def print_banner() -> None:
    banner = f"""
{Fore.CYAN}{Style.BRIGHT}    ██╗ ██████╗ █████╗ ██████╗ ██╗   ██╗███████╗
    ██║██╔════╝██╔══██╗██╔══██╗██║   ██║██╔════╝
    ██║██║     ███████║██████╔╝██║   ██║███████╗
    ██║██║     ██╔══██║██╔══██╗██║   ██║╚════██║
    ██║╚██████╗██║  ██║██║  ██║╚██████╔╝███████║
    ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝{Style.RESET_ALL}
    {Fore.WHITE}{Style.DIM}An AGN-Specialized Emission Line & Kinematic Fitting Engine.{Style.RESET_ALL}
    {Fore.YELLOW}Version: {__version__} | Author: {__author__}{Style.RESET_ALL}
    """
    print(banner)


def _prompt_existing_path(prompt_text: str, is_dir: bool = False) -> Path:
    """Prompts the user for a filesystem path and validates its existence in real time."""
    while True:
        raw_val = click.prompt(
            f"  {Fore.CYAN}➤{Style.RESET_ALL} {prompt_text}", type=str
        ).strip()
        # Strip accidental surrounding quotes from drag-and-drop terminal paths
        cleaned = raw_val.strip("'").strip('"')
        candidate = Path(cleaned).expanduser()

        if not candidate.exists():
            print(f"    {Fore.RED}✖ Path does not exist: {candidate}{Style.RESET_ALL}")
            continue
        if is_dir and not candidate.is_dir():
            print(
                f"    {Fore.RED}✖ Expected a directory, got a file: {candidate}{Style.RESET_ALL}"
            )
            continue
        if not is_dir and candidate.is_dir():
            print(
                f"    {Fore.RED}✖ Expected a file, got a directory: {candidate}{Style.RESET_ALL}"
            )
            continue
        return candidate.resolve()


def _run_interactive_wizard() -> Dict[str, Any]:
    """
    Interactive CLI configuration wizard triggered when `icarus` is invoked
    with zero command-line arguments.
    """
    print(
        f"  {Fore.MAGENTA}{Style.BRIGHT}✦ INTERACTIVE LAUNCH WIZARD ENGAGED ✦{Style.RESET_ALL}"
    )
    print(
        f"  {Fore.WHITE}{Style.DIM}No CLI flags detected. Answer the prompts below (or press Ctrl+C to abort).{Style.RESET_ALL}\n"
    )

    # 1. Execution Mode Selection
    print(f"  {Style.BRIGHT}1. Select Execution Mode:{Style.NORMAL}")
    print(
        f"     {Fore.CYAN}[1]{Style.RESET_ALL} Single FITS Spectrum {Style.DIM}(Quick Analysis / Tutorial){Style.RESET_ALL}"
    )
    print(
        f"     {Fore.CYAN}[2]{Style.RESET_ALL} Batch Survey Catalog {Style.DIM}(Multi-Target / Cluster Run){Style.RESET_ALL}"
    )
    mode_choice = click.prompt(
        f"  {Fore.CYAN}➤{Style.RESET_ALL} Mode [1/2]",
        type=click.Choice(["1", "2"]),
        default="1",
        show_choices=False,
    )

    catalog_path: Optional[Path] = None
    data_dir: Optional[Path] = None
    single_file: Optional[Path] = None
    redshift: float = 0.0
    target_id: Optional[str] = None

    print()
    if mode_choice == "1":
        print(f"  {Style.BRIGHT}2. Target Spectrum Configuration:{Style.NORMAL}")
        single_file = _prompt_existing_path("Path to spectral FITS file", is_dir=False)
        default_tid = single_file.stem
        target_id = click.prompt(
            f"  {Fore.CYAN}➤{Style.RESET_ALL} Target ID",
            default=default_tid,
            type=str,
        ).strip()
        redshift = click.prompt(
            f"  {Fore.CYAN}➤{Style.RESET_ALL} Systemic Redshift (z) [0.0 = try FITS header]",
            default=0.0,
            type=float,
        )
    else:
        print(f"  {Style.BRIGHT}2. Survey Catalog & Data Paths:{Style.NORMAL}")
        catalog_path = _prompt_existing_path(
            "Path to master catalog (.csv / .fits)", is_dir=False
        )
        data_dir = _prompt_existing_path(
            "Root directory containing spectral FITS files", is_dir=True
        )

    # 3. Instrument Preset Selection
    print(f"\n  {Style.BRIGHT}3. Instrument & Survey Preset:{Style.NORMAL}")
    for key, (p_name, p_desc) in BUILTIN_PRESETS.items():
        print(
            f"     {Fore.CYAN}[{key}]{Style.RESET_ALL} {p_name:<22} {Style.DIM}— {p_desc}{Style.RESET_ALL}"
        )
    print(f"     {Fore.CYAN}[6]{Style.RESET_ALL} Custom YAML Path")
    preset_choice = click.prompt(
        f"  {Fore.CYAN}➤{Style.RESET_ALL} Select Preset [1-6]",
        type=click.Choice(["1", "2", "3", "4", "5", "6"]),
        default="1",
        show_choices=False,
    )

    if preset_choice == "6":
        custom_yaml = _prompt_existing_path(
            "Path to custom YAML configuration file", is_dir=False
        )
        preset_str = str(custom_yaml)
    else:
        preset_str = BUILTIN_PRESETS[preset_choice][0]

    # 4. Output & Runtime Options
    print(f"\n  {Style.BRIGHT}4. Output & Runtime Options:{Style.NORMAL}")
    out_raw = click.prompt(
        f"  {Fore.CYAN}➤{Style.RESET_ALL} Output directory",
        default="./ICARUS_RESULTS",
        type=str,
    ).strip()
    output_path = Path(out_raw.strip("'").strip('"')).expanduser()

    plot_flag = click.confirm(
        f"  {Fore.CYAN}➤{Style.RESET_ALL} Generate 5-panel PDF diagnostic plots?",
        default=True,
    )

    default_cores = max(1, (os.cpu_count() or 2) - 1)
    if mode_choice == "2":
        cores = click.prompt(
            f"  {Fore.CYAN}➤{Style.RESET_ALL} CPU threads to allocate",
            default=default_cores,
            type=int,
        )
    else:
        cores = 1

    verbose_flag = click.confirm(
        f"  {Fore.CYAN}➤{Style.RESET_ALL} Enable verbose debug logging?",
        default=False,
    )

    # Print reproducible CLI command for the user's notes
    plot_arg = " --plot" if plot_flag else ""
    verb_arg = " --verbose" if verbose_flag else ""
    if mode_choice == "1":
        equiv_cmd = (
            f'icarus --file "{single_file}" --redshift {redshift} '
            f'--target-id "{target_id}" --preset "{preset_str}" '
            f'--output "{output_path}"{plot_arg}{verb_arg}'
        )
    else:
        equiv_cmd = (
            f'icarus --catalog "{catalog_path}" --data "{data_dir}" '
            f'--preset "{preset_str}" --output "{output_path}" '
            f"--cores {cores}{plot_arg}{verb_arg}"
        )

    print(
        f"\n  {Fore.YELLOW}{Style.DIM}💡 Equivalent CLI Command for future scripts:\n     {equiv_cmd}{Style.RESET_ALL}\n"
    )

    return {
        "catalog": str(catalog_path) if catalog_path else None,
        "data": str(data_dir) if data_dir else None,
        "single_file": str(single_file) if single_file else None,
        "redshift": redshift,
        "target_id": target_id,
        "preset": preset_str,
        "output": str(output_path),
        "cores": max(1, cores),
        "plot": plot_flag,
        "dry_run": False,
        "verbose": verbose_flag,
        "dev": None,
    }


def _process_single_target_worker(
    target: Target,
    master_config: IcarusConfig,
    data_dir: Path,
    out_dir: Path,
    dry_run: bool = False,
    generate_plots: bool = False,
    explicit_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """Stateless top-level worker function for ProcessPoolExecutor pickling."""
    status: Dict[str, Any] = {
        "target_id": target.target_id,
        "status": "FAILED",
        "reason": "",
        "complexes_fit": 0,
        "ecsv_path": None,
        "summary_text": None,
    }
    try:
        local_config = master_config.get_target_config(target)
        builder = ObservationBuilder(data_dir)
        filepath = explicit_file if explicit_file else builder.locate_file(target)
        if not filepath or not filepath.exists():
            raise FileNotFoundError(
                f"Could not locate FITS file for {target.target_id}"
            )

        spectrum = builder.build_spectrum(
            target, filepath, local_config, dry_run=dry_run
        )
        if dry_run:
            status["status"] = "SUCCESS (DRY RUN)"
            return status

        fitter = AGNFitter(spectrum, preset=master_config)
        result = fitter.fit()

        if len(result.complex_fits) > 0:
            fits_path = out_dir / f"icarus_fit_{target.target_id}.fits"
            ecsv_path = out_dir / f"icarus_fit_{target.target_id}.ecsv"
            result.to_fits(fits_path)
            result.to_ecsv(ecsv_path)
            if generate_plots:
                result.plot(out_dir / f"summary_{target.target_id}.pdf")

            status["status"] = "SUCCESS"
            status["complexes_fit"] = len(result.complex_fits)
            status["ecsv_path"] = str(ecsv_path)
            status["summary_text"] = result.summary()
        else:
            status["reason"] = "Zero complexes converged."
    except Exception as e:
        log.error(f"[{target.target_id}] Worker failure: {e}")
        status["reason"] = str(e)

    return status


def aggregate_survey_catalog(
    results: List[Dict[str, Any]], out_dir: Path
) -> Optional[Path]:
    """Combines individual 1-row target ECSV files into a master survey catalog."""
    ecsv_files = [
        Path(r["ecsv_path"])
        for r in results
        if r.get("ecsv_path") and Path(r["ecsv_path"]).exists()
    ]
    if not ecsv_files:
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=AstropyWarning)
            tables = [QTable.read(p, format="ascii.ecsv") for p in ecsv_files]
            master_table = vstack(tables, metadata_conflicts="silent")
            master_path = out_dir / "ICARUS_Master_Survey_Catalog.ecsv"
            master_table.write(master_path, format="ascii.ecsv", overwrite=True)
        return master_path
    except Exception as e:
        log.warning(f"Survey catalog aggregation warning: {e}")
        return None


@click.command()
@click.option(
    "--catalog",
    default=None,
    type=click.Path(exists=True),
    help="Path to master CSV/FITS catalog (for batch survey mode).",
)
@click.option(
    "--data",
    default=None,
    type=click.Path(exists=True),
    help="Root directory of spectral FITS files (for batch survey mode).",
)
@click.option(
    "--file",
    "single_file",
    default=None,
    type=click.Path(exists=True),
    help="Path to a single FITS file (bypasses --catalog requirement).",
)
@click.option(
    "--redshift",
    default=0.0,
    type=float,
    help="Target redshift when running in single-file mode (--file).",
)
@click.option(
    "--target-id",
    default=None,
    type=str,
    help="Optional Target ID override when running in single-file mode.",
)
@click.option(
    "--preset",
    "--config",
    "preset",
    default="jwst_nirspec_medium",
    help="Built-in preset name or path to custom YAML config.",
)
@click.option(
    "--output", default="./ICARUS_RESULTS", type=click.Path(), help="Output directory."
)
@click.option(
    "--cores",
    default=max(1, (os.cpu_count() or 2) - 1),
    type=int,
    help="CPU threads to utilize.",
)
@click.option("--plot", is_flag=True, help="Generate 5-panel PDF diagnostic plots.")
@click.option("--dry-run", is_flag=True, help="Validate inputs without executing fits.")
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
@click.option(
    "--dev", type=click.Path(exists=True), help="Path to .dev breakpoints file."
)
def pipeline(
    catalog,
    data,
    single_file,
    redshift,
    target_id,
    preset,
    output,
    cores,
    plot,
    dry_run,
    verbose,
    dev,
):
    """ICARUS v3.4: AGN Kinematic Minimization & Demographics Pipeline."""
    print_banner()

    # Launch interactive wizard if invoked bare (`icarus` with no flags)
    if len(sys.argv) == 1:
        try:
            wiz = _run_interactive_wizard()
        except (KeyboardInterrupt, click.Abort):
            print(
                f"\n{Fore.YELLOW}⚠ Interactive wizard aborted by user.{Style.RESET_ALL}"
            )
            sys.exit(0)

        catalog = wiz["catalog"]
        data = wiz["data"]
        single_file = wiz["single_file"]
        redshift = wiz["redshift"]
        target_id = wiz["target_id"]
        preset = wiz["preset"]
        output = wiz["output"]
        cores = wiz["cores"]
        plot = wiz["plot"]
        dry_run = wiz["dry_run"]
        verbose = wiz["verbose"]
        dev = wiz["dev"]

    start_time = time.time()
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        log.setLevel(logging.DEBUG)
        for handler in log.handlers:
            handler.setLevel(logging.DEBUG)

    if dev:
        print(
            f"{Fore.RED}{Style.BRIGHT}⚠ DEV MODE ENGAGED: Loading breakpoints from {dev}{Style.RESET_ALL}"
        )
        print(
            f"{Fore.RED}{Style.BRIGHT}⚠ OVERRIDING MULTIPROCESSING: Forcing synchronous execution for terminal access.{Style.RESET_ALL}\n"
        )
        DevState.initialize(Path(dev))
        DevState.run_preflight()
        cores = 1

    print(f"{Fore.BLUE}➤{Style.RESET_ALL} Initializing Research Environment...")
    try:
        master_config = IcarusConfig.from_preset(preset)
        log.info(f"Loaded configuration preset: {preset}")
    except IcarusConfigurationError as e:
        print(f"{Fore.RED}✖ CONFIG ERROR: {e}")
        sys.exit(1)

    explicit_file_path: Optional[Path] = None
    if single_file:
        explicit_file_path = Path(single_file)
        data_dir = explicit_file_path.parent
        tid = target_id or explicit_file_path.stem
        targets = [Target(target_id=str(tid), redshift=float(redshift))]
        cores = 1
    elif catalog and data:
        print(f"{Fore.BLUE}➤{Style.RESET_ALL} Registering Survey Catalog...")
        data_dir = Path(data)
        try:
            registry = CatalogRegistry(catalog)
            targets = registry.generate_targets()
        except Exception as e:
            print(f"{Fore.RED}✖ CATALOG ERROR: {e}")
            sys.exit(1)
    else:
        print(
            f"{Fore.RED}✖ USAGE ERROR: Provide either (--catalog and --data) for batch mode, (--file [--redshift]) for single-target mode, or run bare with zero flags to launch the interactive wizard."
        )
        sys.exit(1)

    num_targets = len(targets)
    mode_str = (
        f"{Fore.YELLOW}DRY RUN (Validation Mode)"
        if dry_run
        else (
            f"{Fore.GREEN}SINGLE-TARGET FIT"
            if single_file
            else f"{Fore.GREEN}BATCH SURVEY PRODUCTION"
        )
    )
    manifest = f"""
    {Style.BRIGHT}--- PIPELINE MANIFEST ---{Style.NORMAL}
    {Fore.WHITE}Mode       : {mode_str}
    {Fore.WHITE}Preset     : {Fore.CYAN}{preset}
    {Fore.WHITE}Threads    : {Fore.CYAN}{cores} Core{'s' if cores > 1 else ''} {Fore.RED + '(DEV OVERRIDE)' if dev else ''}
    {Fore.WHITE}Sources    : {Fore.CYAN}{num_targets} Target{'s' if num_targets != 1 else ''}
    {Fore.WHITE}Output     : {Fore.CYAN}{out_dir.resolve()}
    {Fore.WHITE}Plots      : {Fore.CYAN}{'Enabled (5-Panel PDFs)' if plot else 'Disabled'}
    """
    print(manifest)

    results: List[Dict[str, Any]] = []
    success_count = 0
    total_complexes_fit = 0

    if cores == 1:
        print(f"{Fore.MAGENTA}{Style.BRIGHT}🚀 ENGAGING SYNCHRONOUS EXECUTION LOOP\n")
        for i, t in enumerate(targets):
            res = _process_single_target_worker(
                t,
                master_config,
                data_dir,
                out_dir,
                dry_run,
                plot,
                explicit_file=explicit_file_path,
            )
            results.append(res)
            progress = (i + 1) / num_targets * 100
            if "SUCCESS" in res["status"]:
                success_count += 1
                status_ind = f"{Fore.GREEN}✔"
                total_complexes_fit += res.get("complexes_fit", 0)
            else:
                status_ind = f"{Fore.RED}✘"

            short_reason = (
                (res["reason"][:40] + "...")
                if len(res.get("reason", "")) > 40
                else res.get("reason", "")
            )
            print(
                f" {status_ind} [{progress:4.1f}%] {Fore.WHITE}{t.target_id:<15} {Style.DIM}| {res['status']:<10} {short_reason}",
                end="\r",
            )
            sys.stdout.flush()
    else:
        print(f"{Fore.MAGENTA}{Style.BRIGHT}🚀 ENGAGING MULTIPROCESSING POOL\n")
        with ProcessPoolExecutor(max_workers=cores) as executor:
            futures = {
                executor.submit(
                    _process_single_target_worker,
                    t,
                    master_config,
                    data_dir,
                    out_dir,
                    dry_run,
                    plot,
                ): t
                for t in targets
            }
            for i, future in enumerate(as_completed(futures)):
                t_obj = futures[future]
                res = future.result()
                results.append(res)
                progress = (i + 1) / num_targets * 100
                if "SUCCESS" in res["status"]:
                    success_count += 1
                    status_ind = f"{Fore.GREEN}✔"
                    total_complexes_fit += res.get("complexes_fit", 0)
                else:
                    status_ind = f"{Fore.RED}✘"

                short_reason = (
                    (res["reason"][:40] + "...")
                    if len(res.get("reason", "")) > 40
                    else res.get("reason", "")
                )
                print(
                    f" {status_ind} [{progress:4.1f}%] {Fore.WHITE}{t_obj.target_id:<15} {Style.DIM}| {res['status']:<10} {short_reason}",
                    end="\r",
                )
                sys.stdout.flush()

    duration = time.time() - start_time
    df_report = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "summary_text"} for r in results]
    )
    report_path = out_dir / "ICARUS_Execution_Report.csv"
    df_report.to_csv(report_path, index=False)

    master_cat_path = None
    if num_targets > 1 and not dry_run:
        master_cat_path = aggregate_survey_catalog(results, out_dir)

    print(
        f"\n\n{Fore.CYAN}{Style.BRIGHT}=================================================="
    )
    print(
        f"{Fore.CYAN}{Style.BRIGHT}               SURVEY DEBRIEFING                  "
    )
    print(
        f"{Fore.CYAN}{Style.BRIGHT}=================================================="
    )

    if single_file and results and results[0].get("summary_text"):
        print(f"\n{results[0]['summary_text']}")

    summary_stats = f"""
    {Fore.WHITE}Total Runtime   : {Fore.YELLOW}{duration:.2f} seconds
    {Fore.WHITE}Average Pace    : {Fore.YELLOW}{duration / max(num_targets, 1):.2f} s/target
    {Fore.WHITE}Execution Log   : {Fore.CYAN}{report_path.name}
    {Fore.WHITE}Survey Catalog  : {Fore.CYAN}{master_cat_path.name if master_cat_path else 'N/A (Single Target)'}

    {Style.BRIGHT}--- SCIENCE YIELD ---{Style.NORMAL}
    {Fore.WHITE}Target Success  : {Fore.GREEN}{success_count} {Fore.WHITE}/ {num_targets}
    {Fore.WHITE}Complexes Fit   : {Fore.GREEN}{total_complexes_fit}
    """
    print(summary_stats)

    if success_count == num_targets:
        print(
            f"{Fore.GREEN}{Style.BRIGHT}✔ PIPELINE EXECUTION COMPLETE. 100% TARGET CONVERGENCE.{Style.RESET_ALL}"
        )
    else:
        print(
            f"{Fore.YELLOW}{Style.BRIGHT}⚠ PIPELINE EXECUTION COMPLETE WITH WARNINGS. SEE REPORT FOR DETAILS.{Style.RESET_ALL}"
        )


if __name__ == "__main__":
    pipeline()
