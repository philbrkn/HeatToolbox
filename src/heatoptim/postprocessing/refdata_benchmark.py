# refdata_utils.py
# Helper functions for loading Tur-Prats reference data and computing error norms.
# © 2025 Imperial College London – released under MIT licence.

import numpy as np
from pathlib import Path
from scipy import interpolate


def load_reference(field: str, kn: float, axis: str, folder="data/turprats"):
    """
    field ∈ {"T", "qx", "qy"}
    kn    : 0.1, 1, 2, ...
    axis  ∈ {"horizontal", "vertical"}
    Returns: (s, f_ref)  with s already non-dimensionalised by ℓ
    """
    fn = Path(folder) / f"Kn{kn:g}_{field}_{axis}.csv"
    if not fn.exists():
        raise FileNotFoundError(fn)
    s, f = np.loadtxt(fn, delimiter=",", skiprows=1, unpack=True)
    return s, f


def l2_linf(sim_x, sim_f, ref_x, ref_f):
    """
    Interpolates the simulated profile onto the reference abscissa,
    then returns (L²-relative, L∞-relative).
    """
    interp = interpolate.interp1d(sim_x, sim_f, kind="linear", fill_value="extrapolate")
    diff = interp(ref_x) - ref_f
    eps2 = np.linalg.norm(diff) / np.linalg.norm(ref_f)
    eps_inf = np.max(np.abs(diff)) / np.max(np.abs(ref_f))
    return eps2, eps_inf
