# benchmark_profiles.py
"""
Standalone script to overlay FEniCSx Guyer–Krumhansl centre‑line profiles
with the reference data of Tur‑Prats et al. (2024) and to quantify the error.

Usage (serial run recommended):
    python benchmark_profiles.py --config path/to/sim.json --kn 1.0 \
                                 --refdir data/turprats --outdir figures/05_2_Results_Verify

The script runs a single‑material simulation with the parameters defined in
`sim.json`, samples the temperature and heat‑flux fields along the same two
centre‑lines used in the reference paper, and produces:

* PDF figures with solver lines + reference markers
* a CSV file accumulating the \varepsilon_2 and \varepsilon_\infty metrics
* a small LaTeX block (printed to stdout) for direct inclusion in the report.

Dependencies
------------
* FEniCSx 0.8+
* heatoptim (local package)
* numpy, scipy, matplotlib, pandas
* The reference CSV files obtained with WebPlotDigitizer: see README in
  `data/turprats` for naming convention.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpi4py import MPI

from heatoptim.config.sim_config import SimulationConfig
from heatoptim.solvers.solver_gke_module import GKESolver
from heatoptim.solvers.mesh_generator import MeshGenerator
from heatoptim.postprocessing.post_processing_gke import PostProcessingGKE

# --------------------------------------------------------------------------------------
# Helper utilities ---------------------------------------------------------------------
# --------------------------------------------------------------------------------------


def load_reference(field: str, kn: float, line: str, folder: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return abscissa ``s`` and reference values ``f``.

    Naming convention is ``Kn{kn}_{field}_{line}.csv`` where
    ``field`` \in {"T", "qx", "qy"} and ``line`` \in {"horizontal", "vertical"}.
    The CSV must contain two header‑less columns: ``s``, ``f_ref``.
    """
    fn = folder / f"Kn{kn:g}_{field}_{line}.csv"
    if not fn.is_file():
        raise FileNotFoundError(f"Reference trace not found: {fn}")
    s, f = np.loadtxt(fn, delimiter=",", unpack=True)
    return s, f


def l2_linf(sim: np.ndarray, ref: np.ndarray) -> Tuple[float, float]:
    """Return (L2_error, Linf_error) normalised by the reference norms."""
    eps2 = np.linalg.norm(sim - ref) / np.linalg.norm(ref)
    eps_inf = np.max(np.abs(sim - ref)) / np.max(np.abs(ref))
    return eps2, eps_inf


# --------------------------------------------------------------------------------------
# Sampling utilities -------------------------------------------------------------------
# --------------------------------------------------------------------------------------


def sample_centerline(post: PostProcessingGKE, f, msh, orientation: str, cfg) -> Tuple[np.ndarray, np.ndarray]:
    """Return (s/ell, values) along the chosen centre‑line.

    * ``orientation`` = "horizontal" or "vertical"
    * Re‑uses PostProcessingGKE.get_temperature_line so the sampling is identical
      to the existing plots.
    """

    # Select the same physical lines as in PostProcessingGKE.plot_profiles
    if orientation == "horizontal":
        x_char = cfg.L_X if cfg.symmetry else cfg.L_X / 2
        s_end = x_char
        y_val = cfg.L_Y - 4 * cfg.LENGTH / 8
        s, vals = post.get_temperature_line(f, msh, "horizontal", start=0, end=s_end, value=y_val)
    elif orientation == "vertical":
        y_end = cfg.L_Y + cfg.SOURCE_HEIGHT
        x_val = (cfg.L_X if cfg.symmetry else cfg.L_X / 2) - cfg.LENGTH / 8
        s, vals = post.get_temperature_line(f, msh, "vertical", start=0, end=y_end, value=x_val)
    else:
        raise ValueError("orientation must be 'horizontal' or 'vertical'")

    # Convert to non‑dim abscissa s/ell (Tur‑Prats' convention)
    s_nd = s / cfg.ELL_SI
    return s_nd, vals


# --------------------------------------------------------------------------------------
# Main benchmarking routine ------------------------------------------------------------
# --------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to simulation JSON")
    parser.add_argument("--kn", type=float, required=True, help="Knudsen number to compare against")
    parser.add_argument("--refdir", default="data/turprats", help="Folder with reference CSV files")
    parser.add_argument("--outdir", default="figures/05_2_Results_Verify", help="Output folder for figs + csv")
    parser.add_argument("--mesh_res", type=float, default=12.0, help="Characteristic mesh resolution")
    args = parser.parse_args()

    refdir = Path(args.refdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Build config and mesh (serial run only)
    # ---------------------------------------------------------------------
    cfg = SimulationConfig(args.config, arg_res=args.mesh_res)
    if MPI.COMM_WORLD.size > 1:
        raise RuntimeError("Run this script in serial; error metrics need all fields on rank 0.")

    # Override the single‑material Knudsen number for clarity (optional)
    cfg.KNUDSEN_SI = args.kn

    mesh_gen = MeshGenerator(cfg)
    mesh_gen.create_mesh() if not cfg.symmetry else mesh_gen.sym_create_mesh()
    msh, cell_markers, facet_markers = mesh_gen.msh, mesh_gen.cell_markers, mesh_gen.facet_markers

    solver = GKESolver(msh, facet_markers, cfg)

    # No random micro‑structure: use default image (solid material)
    img_list = [np.zeros((cfg.n_px, cfg.n_px))]  # dummy placeholder for single material
    solver.solve_image(img_list)

    q_fun, T_fun = solver.U.sub(0).collapse(), solver.U.sub(1).collapse()
    V_T, _ = solver.U.function_space.sub(1).collapse()

    post = PostProcessingGKE(rank=0, config=cfg, logger=None)

    # ---------------------------------------------------------------------
    # Sample FEM solution on the two centre‑lines
    # ---------------------------------------------------------------------
    sx_h, T_h = sample_centerline(post, T_fun, msh, "horizontal", cfg)
    sy_v, T_v = sample_centerline(post, T_fun, msh, "vertical", cfg)

    qx_fun, qy_fun = q_fun.split()
    sx_qx, qx_h = sample_centerline(post, qx_fun, msh, "horizontal", cfg)
    sy_qy, qy_v = sample_centerline(post, qy_fun, msh, "vertical", cfg)

    # ---------------------------------------------------------------------
    # Load reference data
    # ---------------------------------------------------------------------
    s_T_h_ref, T_h_ref = load_reference("T", args.kn, "horizontal", refdir)
    s_T_v_ref, T_v_ref = load_reference("T", args.kn, "vertical",   refdir)

    s_qx_h_ref, qx_h_ref = load_reference("qx", args.kn, "horizontal", refdir)
    s_qy_v_ref, qy_v_ref = load_reference("qy", args.kn, "vertical",   refdir)

    # ---------------------------------------------------------------------
    # Compute errors (interpolate FEM onto reference abscissa)
    # ---------------------------------------------------------------------
    eps_T2,  eps_Tinf  = l2_linf(np.interp(s_T_h_ref, sx_h,  T_h),  T_h_ref)
    eps_qx2, eps_qxinf = l2_linf(np.interp(s_qx_h_ref, sx_qx, qx_h), qx_h_ref)
    eps_qy2, eps_qyinf = l2_linf(np.interp(s_qy_v_ref, sy_qy, qy_v), qy_v_ref)

    # ---------------------------------------------------------------------
    # Accumulate into CSV for all runs
    # ---------------------------------------------------------------------
    csv_path = outdir / "benchmark_errors.csv"
    header = not csv_path.is_file()
    with csv_path.open("a") as fh:
        if header:
            fh.write("Kn,eps2_T,epsinf_T,eps2_qx,epsinf_qx,eps2_qy,epsinf_qy\n")
        fh.write(f"{args.kn},{eps_T2:.4e},{eps_Tinf:.4e},{eps_qx2:.4e},{eps_qxinf:.4e},{eps_qy2:.4e},{eps_qyinf:.4e}\n")

    # ---------------------------------------------------------------------
    # Overlay plots --------------------------------------------------------
    # ---------------------------------------------------------------------
    def overlay(ax, s_sim, f_sim, s_ref, f_ref, label_sim, label_ref):
        ax.plot(s_sim, f_sim, lw=2, label=label_sim)
        ax.scatter(s_ref, f_ref, facecolors="none", edgecolors="k", label=label_ref, zorder=3)

    # Temperature ---------------------------------------------------------
    fig_T, ax_T = plt.subplots(figsize=(6, 4))
    overlay(ax_T, sx_h, T_h, s_T_h_ref, T_h_ref, "FEM horiz.", "Tur‑Prats horiz.")
    overlay(ax_T, sy_v, T_v, s_T_v_ref, T_v_ref, "FEM vert.",  "Tur‑Prats vert.")
    ax_T.set_xlabel(r"$s/\ell$")
    ax_T.set_ylabel(r"$T/T_{\max}$")
    ax_T.set_title(f"Temperature, Kn = {args.kn:g}")
    ax_T.legend(frameon=False)
    fig_T.tight_layout()
    fig_T.savefig(outdir / f"T_profiles_Kn{args.kn:g}.pdf")

    # Heat flux -----------------------------------------------------------
    fig_q, (ax_qx, ax_qy) = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    overlay(ax_qx, sx_qx, qx_h, s_qx_h_ref, qx_h_ref, "FEM horiz.", "Ref horiz.")
    ax_qx.set_xlabel(r"$s/\ell$")
    ax_qx.set_ylabel(r"$q_x/q_0$")
    ax_qx.set_title("Horizontal centre‑line")

    overlay(ax_qy, sy_qy, qy_v, s_qy_v_ref, qy_v_ref, "FEM vert.", "Ref vert.")
    ax_qy.set_xlabel(r"$s/\ell$")
    ax_qy.set_ylabel(r"$q_y/q_0$")
    ax_qy.set_title("Vertical centre‑line")

    for ax in (ax_qx, ax_qy):
        ax.legend(frameon=False)
    fig_q.tight_layout()
    fig_q.savefig(outdir / f"q_profiles_Kn{args.kn:g}.pdf")

    # ---------------------------------------------------------------------
    # Echo LaTeX table row to stdout for convenience ----------------------
    # ---------------------------------------------------------------------
    print("\nLaTeX row (copy into tabular):")
    print(f"{args.kn:g} & {eps_T2:.2e} & {eps_Tinf:.2e} & {eps_qx2:.2e} & {eps_qxinf:.2e} & {eps_qy2:.2e} & {eps_qyinf:.2e} \\")


if __name__ == "__main__":
    main()
