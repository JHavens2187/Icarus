"""
typography.py

Ion string sanitization, mathtext LaTeX formatting, and spatial label de-crowding engine for ICARUS v3.4.
"""

__project__ = "ICARUS"
__author__ = "Joseph Havens"
__version__ = "3.4.0"

import re
from typing import List, Tuple

import numpy as np


def sanitize_line_name(raw_name: str) -> str:
    """
    Normalizes raw catalog transition names into valid Python/LMFIT parameter
    identifiers (e.g., '[O III] 5007' -> 'O_III_5007', 'H-alpha' -> 'H_alpha').
    """
    cleaned = str(raw_name).strip()
    cleaned = cleaned.replace("[", "").replace("]", "")
    cleaned = re.sub(r"[\s\-/]+", "_", cleaned)
    cleaned = re.sub(r"[^A-Za-z0-9_]", "", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def format_latex_ion(raw_name: str, is_limit: bool = False) -> str:
    """
    Parses sanitized catalog transition names into formatted LaTeX strings
    compatible with Matplotlib's native STIX mathtext engine.
    """
    prefix = r"\downarrow " if is_limit else ""

    # 1. Match Hydrogen recombination series (Lyman, Balmer, Paschen, Brackett, Pfund, Humphreys)
    m_hydro = re.match(r"^(H|Pa|Br|Pf|Hu|Ly)_?([a-z]+)$", raw_name)
    if m_hydro:
        series, letter = m_hydro.groups()
        greek_map = {
            "a": r"\alpha",
            "b": r"\beta",
            "g": r"\gamma",
            "d": r"\delta",
            "eps": r"\epsilon",
            "zeta": r"\zeta",
            "alpha": r"\alpha",
            "beta": r"\beta",
            "gamma": r"\gamma",
            "delta": r"\delta",
            "epsilon": r"\epsilon",
        }
        greek = greek_map.get(letter, letter)
        return rf"${prefix}\mathrm{{{series}}}{greek}$"

    # 2. Match atomic ions with wavelength suffix (e.g., O_III_5007, Mg_II_2796, C_III_1909)
    m_ion = re.match(r"^([A-Za-z]+)_([IVX]+)_(\d+)$", raw_name)
    if m_ion:
        elem, ion, wave = m_ion.groups()
        # Semi-forbidden C III] 1909 notation
        if elem == "C" and ion == "III" and wave in ("1907", "1909"):
            return rf"${prefix}\mathrm{{C\,III}}]\,\lambda {wave}$"
        forbidden_elems = {"O", "N", "S", "Ne", "Ar"}
        is_forbidden = (elem in forbidden_elems) and not (
            (elem == "O" and ion == "VI") or (elem == "N" and ion == "V")
        )
        if is_forbidden:
            return rf"${prefix}[\mathrm{{{elem}\,{ion}}}]\,\lambda {wave}$"
        return rf"${prefix}\mathrm{{{elem}\,{ion}}}\,\lambda {wave}$"

    # 3. Match atomic ions without wavelength suffix
    m_ion_nowave = re.match(r"^([A-Za-z]+)_([IVX]+)$", raw_name)
    if m_ion_nowave:
        elem, ion = m_ion_nowave.groups()
        forbidden_elems = {"O", "N", "S", "Ne", "Ar"}
        if elem in forbidden_elems:
            return rf"${prefix}[\mathrm{{{elem}\,{ion}}}]$"
        return rf"${prefix}\mathrm{{{elem}\,{ion}}}$"

    safe_str = raw_name.replace("_", r"\_")
    return rf"${prefix}\mathrm{{{safe_str}}}$"


def decrowd_annotations(
    annotations: List[Tuple[float, float, str]],
    x_min: float,
    x_max: float,
    log_scale: bool = True,
    min_frac_sep: float = 0.022,
) -> List[Tuple[float, float, str, bool]]:
    """
    Sorts detected emission lines by SNR descending and determines which text
    labels can be placed without spatial overlap in either logarithmic wavelength
    space (`log_scale=True`) or linear velocity space (`log_scale=False`).

    Input:  List of `(x_coord, snr, line_name)`
    Output: List of `(x_coord, snr, line_name, show_text_label)`
    """
    if not annotations:
        return []

    sorted_anns = sorted(annotations, key=lambda item: item[1], reverse=True)
    placed_coords: List[float] = []
    results: List[Tuple[float, float, str, bool]] = []

    if log_scale and x_min > 0 and x_max > x_min:
        span = np.log10(x_max) - np.log10(x_min)
        min_sep = min_frac_sep * max(span, 1e-4)

        for x_val, snr, name in sorted_anns:
            if x_val <= 0:
                continue
            lx = float(np.log10(x_val))
            can_place = all(abs(lx - px) >= min_sep for px in placed_coords)
            if can_place:
                placed_coords.append(lx)
            results.append((x_val, snr, name, can_place))
    else:
        span = max(abs(x_max - x_min), 1.0)
        min_sep = min_frac_sep * span

        for x_val, snr, name in sorted_anns:
            can_place = all(abs(x_val - px) >= min_sep for px in placed_coords)
            if can_place:
                placed_coords.append(float(x_val))
            results.append((x_val, snr, name, can_place))

    return results
