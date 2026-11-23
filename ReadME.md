# ICARUS

**An AGN-Specialized, Multi-Core Spectroscopic Decomposition & Virial Black Hole Mass Estimation Pipeline for UV/Optical/NIR Spectroscopy**

[![Version](https://img.shields.io/badge/version-3.4.0-blue.svg)](https://github.com/JHavens2187/Icarus)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-green.svg)](LICENSE)
[![Astropy](https://img.shields.io/badge/powered%20by-Astropy-orange.svg)](https://www.astropy.org/)

---

## Overview

**ICARUS** (`icarus_pkg`) is a high-throughput Python spectroscopic fitting engine built to decompose blended Active Galactic Nuclei (AGN) and host-galaxy spectra from modern surveys such as **JWST/NIRSpec** (JADES, MEGA, COSMOS-3D), **Roman/WFI**, and **SDSS/BOSS**.

Legacy spectral fitters often force a trade-off between slow Bayesian MCMC sampling that struggles to scale across large survey catalogs and uncoupled Gaussian fits that suffer from broad-vs-narrow line degeneracy. ICARUS bridges this gap with a **two-stage physics-coupled cascade**, automated **$\Delta\mathrm{BIC}$ model arbitration**, and **shared-memory multiprocessing** optimized for both local workstations and SLURM High-Performance Computing (HPC) clusters.

> **Interactive Tutorials Coming Soon:** A dedicated walkthrough guide and Jupyter notebook tutorial suite (`notebooks/`) are currently in preparation.

---

## Key Capabilities

* **Two-Pass Global Continuum Cascade (Stage 1):**
  * Dynamically scales flux arrays to $\mathcal{O}(1)$ space via $10^{\lfloor \log_{10}(\mathrm{median\ flux}) \rfloor}$ to prevent floating-point underflow on faint high- $z$ spectra ($\sim 10^{-20}\ \mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$).
  * **Pass 1** models the global AGN accretion-disk power-law, host starlight envelope, and optically thick Balmer recombination continuum ($3646\ \mathrm{\AA}$ edge).
  * **Pass 2** fits FFT-convolved UV and optical $\mathrm{Fe II}$ pseudo-continua on a uniform logarithmic velocity grid, anchored locally by outer-25% window medians.
* **Physics-Coupled Emission-Line Engine (Stage 2):**
  * Simultaneously fits up to **22 rest-frame UV/Optical/NIR spectral complexes** ($1500\text{--}13000\ \mathrm{\AA}$) in relative velocity space ($\Delta v$) using Levenberg-Marquardt (`lmfit`) optimization.
  * Enforces **quantum-mechanical doublet flux ratios** ($[\mathrm{O\,III}]\,\lambda\lambda 5007,4959 = 2.98$, $[\mathrm{N\,II}]\,\lambda\lambda 6583,6548 = 2.95$) and couples systemic narrow-line velocity offsets ($\Delta v_{\mathrm{narrow}}$).
* **Automated Model Arbitration & Speed Gatekeeper:**
  * Evaluates nested kinematic models (Narrow-Only vs. Narrow + $[\mathrm{O III}]$ Biconical Outflow Wing vs. Broad BLR) using a strict **$\Delta\mathrm{BIC} \leq -10$** threshold paired with physical peak/width vetoes to eliminate false-positive broad lines.
  * A **$2\sigma$ Speed Gatekeeper** pre-screens weak or absent complexes before launching multi-component optimization, deriving $3\sigma$ analytical upper limits from local boxcar RMS for non-detections.
* **Automated Virial Waterfall & Nebular Diagnostics:**
  * Deconvolves wavelength-dependent instrumental LSF dispersion $R(\lambda_{\mathrm{obs}})$ to compute single-epoch **Virial Black Hole Masses** ($\log_{10}(M_{\mathrm{BH}}/M_\odot)$ via $\mathrm{H}\beta$, $\mathrm{H}\alpha$, $\mathrm{Mg\,II}$, blueshift-corrected $\mathrm{C\,IV}$, or $\mathrm{Pa}\beta$), **Bolometric Luminosities** ($\log L_{\mathrm{bol}}$), **Eddington Ratios** ($\lambda_{\mathrm{Edd}}$), narrow-line **SFR**, and **BPT** excitation classifications.
* **Survey-Scale Parallelism & Middleware Telemetry:**
  * Scales across multi-core workstations and SLURM HPC compute nodes via `concurrent.futures.ProcessPoolExecutor`.
  * Includes a non-destructive `@dev_hook` middleware engine (`core/dev_tools.py`) for interactive debugging, telemetry tracing, and synthetic injection without altering production code.

---

## Installation

### Standard Install (Direct from GitHub)
```bash
pip install git+https://github.com/JHavens2187/Icarus.git
```

### Editable / Developer Install (Recommended for HPC & Customization)
```bash
git clone https://github.com/JHavens2187/Icarus.git
cd Icarus
pip install -r requirements.txt
pip install -e .
```

**Requirements:** Python 3.10+, `numpy`, `scipy`, `astropy`, `lmfit`, `pydantic`, `pyyaml`, `pandas`, `matplotlib`, `numba`.

---

## Quickstart

### 1. Interactive Python API (Catalog-Free)
Ideal for Jupyter notebooks and single-target inspection using built-in instrument presets (`jwst_nirspec_medium`, `jwst_nirspec_high`, `jwst_nirspec_prism`, `roman_wfi`, `sdss_boss`):

```python
from icarus_pkg.api import AGNFitter
from icarus_pkg.core.containers import Spectrum1D

# 1. Load 1D spectrum into an immutable, unit-safe container
spec = Spectrum1D.read(
    "path/to/target.spec.fits",
    redshift=2.5798,
    target_id="TARGET_0001",
    preset="jwst_nirspec_medium",
)

# 2. Run the 2-stage continuum + coupled emission-line cascade
result = AGNFitter(spec, preset="jwst_nirspec_medium").fit()

# 3. Print executive telemetry summary & render 5-panel diagnostic PDF
print(result.summary())
result.plot("demo_summary_TARGET_0001.pdf")
```

### 2. Multi-Core CLI Batch Processing
For survey-scale catalogs (`catalog.csv` + `spectra/*.fits`):

```bash
python -m icarus_pkg.cli \
    --catalog path/to/catalog.csv \
    --spectra-dir path/to/spectra \
    --out-dir path/to/results \
    --preset jwst_nirspec_medium \
    --cores 40
```

> **HPC Tip:** When running multi-core jobs on Linux SLURM compute nodes, export `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1` before launching to prevent BLAS/OpenMP thread oversubscription across `ProcessPoolExecutor` workers.

---

## Output Data Products

For every survey run, ICARUS generates standardized, TOPCAT- and Astropy-ready outputs that bypass the 999-column FITS binary table limit:

```text
RESULTS_DIR/
├── icarus_fit_<TARGET_ID>.fits      # 3-HDU Multi-Extension FITS (MEF) per target
│   ├── HDU 0 (PRIMARY)              # Target provenance, Virial M_BH, L_bol, Eddington ratio, CONF_FLG bitmask
│   ├── HDU 1 (LINE_CATALOG)         # Row-per-line table: fluxes, deconvolved FWHM, offsets, EWs, & 3σ upper limits
│   └── HDU 2 (SYNTH_SPEC)           # 1D rest-frame arrays: observed flux, continua, component models, & normalized pulls
├── master_survey_catalog.ecsv       # Aggregated Astropy QTable across all targets in the batch
└── demo_summary_<TARGET_ID>.pdf     # Optional 5-panel publication diagnostic figure
```

---

## Validation & HPC Benchmarks

To avoid circular ($A = A$) validation, ICARUS v3.4 was benchmarked against an independent **Cloudy C23** radiative-transfer synthetic suite ($N = 64$ targets spanning Type 1 BLAGN, Type 2 NLAGN, and Star-Forming $\mathrm{H\,II}$ galaxies) featuring broken accretion disks, host $\mathrm{D}_n4000$ breaks with stellar Balmer absorption troughs, Osterbrock bound-free Balmer/Paschen edges, and **non-Gaussian Voigt BLR profiles**.

| Benchmark Metric | ICARUS v3.4 Result | Notes |
| :--- | :---: | :--- |
| **Type 1 BLAGN Recall / Precision** | **100% / 100%** | $35/35$ TP, $0/29$ FP ($\Delta\mathrm{BIC} \leq -10$) |
| **$[\mathrm{O\,III}]$ Outflow Recall** | **100%** | $37/37$ outflows recovered; $\langle \Delta v_{\mathrm{out}} \rangle = +17.9 \pm 40.3\ \mathrm{km\,s^{-1}}$ |
| **Single-Epoch Virial $M_{\mathrm{BH}}$ Scatter** | **$\sigma_{\mathrm{MAD}} = 0.06\ \mathrm{dex}$** | Median offset $-0.06\ \mathrm{dex}$, Spearman $r_s = 0.98$ |
| **Deconvolved Broad BLR $\mathrm{FWHM}$** | **$+0.8\% \pm 6.4\%$** | Tested under Voigt-to-Gaussian wing mismatch |
| **Deconvolved Narrow Core $\sigma_{\mathrm{core}}$** | **$\mathrm{RMSE} = 4.5\ \mathrm{km\,s^{-1}}$** | After $[\mathrm{O\,III}]$ outflow wing removal & LSF deconvolution |
| **40-Core KU HPC Strong Scaling** | **$27.25\times$ Speedup** | Amdahl serial fraction $f_s = 0.012$; throughput $>4.5\ \mathrm{spectra\,s^{-1}}$ |

---

## Package Structure

```text
Icarus/
├── icarus_pkg/
│   ├── __init__.py         # Public namespace & styling defaults
│   ├── api.py              # High-level AGNFitter object-oriented interface
│   ├── cli.py              # Multi-core ProcessPoolExecutor batch runner & interactive wizard
│   ├── core/               # Spectrum1D containers, Pydantic config, FITS ingest, 3-HDU MEF I/O, @dev_hook
│   ├── data/               # Vacuum atomic line list, instrumental R(λ) dispersion curves, & Fe II templates
│   ├── physics/            # 2-pass continuum cascade, FFT Fe II convolver, coupled LMFIT engine, virial & BPT calculators
│   ├── presets/            # Built-in YAML instrument presets (JWST NIRSpec, Roman WFI, SDSS/BOSS)
│   └── viz/                # 5-panel diagnostic plotter, BPT plane, MathText ion typography, & rcParams styling
├── pyproject.toml          # Build metadata & package entry points
├── requirements.txt        # Core dependencies
└── LICENSE                 # BSD 3-Clause License
```

---

## Feedback & Contributing

Feedback, bug reports, and feature requests are very appreciated!

* **Feedback Form:** [Feedback form here!](https://forms.gle/f68dbRkJmVhE2g1d8)
* **Issues & Pull Requests:** Open an issue or PR directly on [GitHub](https://github.com/JHavens2187/Icarus/issues).

---

## Acknowledgements & Citation

Developed by **Joseph Havens** in collaboration with **Dr. Allison Kirkpatrick** and the **University of Kansas Department of Physics & Astronomy** (AGNerds Research Group). Computational benchmarks were performed using the **University of Kansas Center for Research Computing (CRC)** HPC cluster.

If you use ICARUS in your research, please cite this repository:

```bibtex
@software{havens_2026_icarus,
  author       = {Havens, Joseph and Kirkpatrick, Allison},
  title        = {{ICARUS: An AGN-Specialized Emission Line Fitting Pipeline for UV/Optical/NIR Spectroscopy}},
  year         = 2026,
  version      = {3.4.0},
  url          = {https://github.com/JHavens2187/Icarus}
}
```
