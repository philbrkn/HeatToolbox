#!/usr/bin/env python3
"""
benchmark_gke.py

Standalone script to reproduce Tur-Prats et al. benchmarks for the single-material G–K solver,
overlay FEM vs. reference curves, compute L2/L∞ errors, and emit a LaTeX-ready error table.

Instructions:
- Edit CONFIG_PATH to point at your JSON config for a simple single-material run.
- Adjust RESOLUTION, KN_LIST, and OUTPUT_DIR as needed.
- Run with a single MPI rank (e.g., `mpiexec -n 1 python3 benchmark_gke.py`).
- This script will:
    1. For each Knudsen number in KN_LIST:
         • Create mesh, instantiate GKESolver, solve steady G–K problem.
         • Sample T and q along horizontal/vertical center-lines.
         • Load digitized reference CSVs from data/turprats/ (must exist).
         • Plot overlay FEM vs. Tur-Prats profiles (saved as PDF in OUTPUT_DIR).
         • Compute normalized L2 and L∞ errors, appending to benchmark_errors.csv.
    2. After all Kn, read benchmark_errors.csv and print a LaTeX tabular to stdout.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dolfinx

from mpi4py import MPI

# --- Import your project's modules ---
from heatoptim.config.sim_config import SimulationConfig
from heatoptim.solvers.mesh_generator import MeshGenerator
from heatoptim.solvers.solver_gke_module import GKESolver
from heatoptim.postprocessing.post_processing_gke import PostProcessingGKE

# ----------------------------------------
#  USER-DEFINED PARAMETERS (edit here)
# ----------------------------------------

# Path to a JSON configuration that defines a single-material domain.
# It must specify e.g. LENGTH, L_X, L_Y, SOURCE_HEIGHT, symmetry=True, etc.
CONFIG_PATH = "configs/kn01benchmarkconfig.json"

# Mesh resolution argument passed to SimulationConfig
RESOLUTION = 8

# List of Knudsen numbers to benchmark (must match reference CSV filenames)
# KN_LIST = [0.1, 1.0, 2.0]
KN_LIST = [0.1]

# Folder where Tur-Prats reference CSVs live:
# e.g. data/turprats/Kn1_T_horizontal.csv, etc.
REF_FOLDER = "data/turprats"

# Output directory for figures and CSV (will be created if missing)
OUTPUT_DIR = "benchmark_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CSV file to accumulate error metrics
ERROR_CSV = os.path.join(OUTPUT_DIR, "benchmark_errors.csv")
# If the file exists, remove it so we start fresh
if os.path.exists(ERROR_CSV):
    os.remove(ERROR_CSV)

# Number of sampling points along each centre-line
NUM_SAMPLES = 500

# ----------------------------------------
#  HELPER FUNCTIONS
# ----------------------------------------

def load_reference(field: str, kn: float, line: str, folder=REF_FOLDER):
    """
    Load Tur-Prats reference data from CSV.
    Expects CSV with header and two columns: s, f_ref
      - s      : abscissa (already normalized by ℓ)
      - f_ref  : field value (T/Tmax or q/ q0)

    field : 'T', 'qx', or 'qy'
    kn    : e.g. 0.1, 1.0, 2.0
    line  : 'horizontal' or 'vertical'
    """
    filename = f"Kn{kn:g}_{field}_{line}.csv"
    path = os.path.join(folder, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Reference file not found: {path}")
    s, f = np.loadtxt(path, delimiter=",", skiprows=1, unpack=True)
    return s, f

def l2_linf(sim_values: np.ndarray, ref_values: np.ndarray):
    """
    Compute normalized L2 and L∞ errors between sim_values and ref_values
    on identical abscissa arrays.
    """
    diff = sim_values - ref_values
    eps2 = np.linalg.norm(diff) / np.linalg.norm(ref_values)
    eps_inf = np.max(np.abs(diff)) / np.max(np.abs(ref_values))
    return eps2, eps_inf

def sample_profiles_from_solver(solver, postproc, kn_value):
    """
    Given a solved GKESolver and its PostProcessingGKE, sample T and q along
    horizontal and vertical center-lines. Returns:
      x_vals, T_x,    horizontal: abscissa s, and T(s)
      y_vals, T_y,    vertical:   abscissa s, and T(s)
      x_vals, qx_h,   horizontal: abscissa s, and q_x(s)
      y_vals, qy_v    vertical:   abscissa s, and q_y(s)
    Normalization: distances already returned by postproc are in s/ℓ form.
    """
    # Extract solution functions
    q_field, T_field = solver.U.sub(0).collapse(), solver.U.sub(1).collapse()
    V_T, _ = solver.U.function_space.sub(1).collapse()
    # Directly call plot_profiles logic but intercept data before plotting:
    # We'll temporarily monkey-patch plot_profiles to return data and skip plotting.
    # Instead, call the get_* methods manually:
    # Reproduce logic from PostProcessingGKE.plot_profiles:

    # 1. Split q into components
    qx, qy = q_field.split()

    # 2. Build curl if needed (omitted here, since only T and q are needed)
    # curl_q = postproc.calculate_curl(q_field, solver.msh, plot_curl=False)

    # 3. Effective conductivity (not needed for benchmarking)
    # eff_cond = postproc.calculate_eff_thermal_cond(q_field, T_field, solver.msh)

    # 4. Define centre-line positions in physical coords
    x_char = solver.config.L_X if solver.config.symmetry else solver.config.L_X / 2
    # Horizontal line: y = L_Y - 4*(LENGTH)/8 = L_Y - LENGTH/2
    y_val = solver.config.L_Y - solver.config.LENGTH / 8
    # Vertical line: x = x_char - LENGTH/8
    x_val = x_char - 3*solver.config.LENGTH / 8

    # 5. Sample T on horizontal (start=0, end=x_char, y=y_val)
    x_pts, T_x = postproc.get_temperature_line(T_field, solver.msh,
                                               "horizontal",
                                               start=0.0, end=x_char,
                                               value=y_val,
                                               num_points=NUM_SAMPLES)
    _,   qx_h = postproc.get_temperature_line(qx, solver.msh,
                                               "horizontal",
                                               start=0.0, end=x_char,
                                               value=y_val,
                                               num_points=NUM_SAMPLES)
    # 6. Sample T on vertical (start=0, end=L_Y+SOURCE_HEIGHT, x=x_val)
    y_end = solver.config.L_Y + solver.config.SOURCE_HEIGHT
    y_pts, T_y = postproc.get_temperature_line(T_field, solver.msh,
                                               "vertical",
                                               start=0.0, end=y_end,
                                               value=x_val,
                                               num_points=NUM_SAMPLES)
    _,   qy_v = postproc.get_temperature_line(qy, solver.msh,
                                               "vertical",
                                               start=0.0, end=y_end,
                                               value=x_val,
                                               num_points=NUM_SAMPLES)
    # 7. Normalize abscissa: postproc returns raw x, y that are (distance from end)/ℓ.
    # Actually, in your code: 
    #   x_vals = (x_vals[-1] - x_vals)/ELL_SI, so if postproc returned raw model coords,
    #   we need to replicate that. But get_temperature_line does return raw coords along axis.
    # So let's reapply normalization exactly as plot_profiles did:
    x_vals_norm = (x_pts[-1] - x_pts) / solver.config.ELL_SI
    y_vals_norm = (y_pts[-1] - y_pts) / solver.config.ELL_SI

    # 8. Flip signs as in plot_profiles: 
    #    qx_horiz was flipped: qx_h *= -1
    #    qy_vert is not flipped in your code.
    qx_h = -1.0 * qx_h
    # leave qy_v as is

    return x_vals_norm, T_x, x_vals_norm, qx_h, y_vals_norm, T_y, y_vals_norm, qy_v

# ----------------------------------------
#  MAIN BENCHMARK LOOP
# ----------------------------------------

def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # Only rank 0 writes output files / prints tables
    for kn in KN_LIST:
        if rank == 0:
            print(f"=== Running benchmark for Kn = {kn} ===")

        # --- 1. Build SimulationConfig and override Knudsen ---
        config = SimulationConfig(CONFIG_PATH, arg_res=RESOLUTION)
        config.KNUDSEN_SI = kn
        config.KNUDSEN_DI = kn  # if needed
        # Force single-material: ensure gamma is uniform inside your config

        # --- 2. Generate mesh (rank 0) ---
        mesh_generator = MeshGenerator(config)
        if rank == 0:
            if config.symmetry:
                mesh_generator.sym_create_mesh()
            else:
                mesh_generator.create_mesh()
        comm.barrier()

        # --- 3. Read mesh on all ranks ---
        # Assumes mesh file is "domain_with_extrusions.msh" in working directory
        msh, cell_markers, facet_markers = dolfinx.io.gmshio.read_from_msh(
            "domain_with_extrusions.msh", MPI.COMM_SELF, gdim=2
        )

        # --- 4. Solve G–K problem ---
        solver = GKESolver(msh, facet_markers, config)
        img = [np.zeros((128, 128))]  # blank image
        solver.solve_image(img)  # or appropriate call to assemble+solve steady-state

        # --- 5. Post-process: sample profiles & plot overlays + compute errors ---
        postproc = PostProcessingGKE(rank, config, logger=None)
        x_h, T_h, x_h2, qx_h, y_v, T_v, y_v2, qy_v = sample_profiles_from_solver(solver, postproc, kn)

        # --- 5a. Load Tur-Prats reference data ---
        s_T_h_ref,  T_h_ref  = load_reference("T",  kn, "horizontal")
        s_T_v_ref,  T_v_ref  = load_reference("T",  kn, "vertical")
        s_qx_h_ref, qx_h_ref = load_reference("qx", kn, "horizontal")
        s_qy_v_ref, qy_v_ref = load_reference("qy", kn, "vertical")

        # --- 5b. Plot overlays: Temperature ---
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x_h, T_h,   label="FEM (horiz)",  lw=1.5)
        ax.plot(y_v, T_v,   label="FEM (vert)",   lw=1.5)
        ax.scatter(s_T_h_ref, T_h_ref, facecolors="none", edgecolors="k",
                    label="Tur-Prats (horiz)", s=15, zorder=3)
        ax.scatter(s_T_v_ref, T_v_ref, facecolors="none", edgecolors="gray",
                    label="Tur-Prats (vert)", s=15, zorder=3)
        ax.set_xlabel(r"$s/\ell$")
        ax.set_ylabel(r"$T/T_{\max}$")
        ax.set_title(f"Temperature profiles (Kn={kn:g})")
        ax.legend(frameon=False, fontsize=8, loc="best")
        ax.grid(alpha=0.15)
        fig.tight_layout()
        fname = os.path.join(OUTPUT_DIR, f"Kn{kn:g}_temperature_profiles.pdf")
        fig.savefig(fname, dpi=300)
        plt.close(fig)

        # --- 5c. Plot overlays: Heat-flux ---
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        # vertical flux vs vertical abscissa
        axs[0].plot(y_v, qy_v, label="FEM $q_y$", lw=1.5)
        axs[0].scatter(s_qy_v_ref, qy_v_ref, facecolors="none", edgecolors="k",
                        label="Tur-Prats $q_y$", s=15, zorder=3)
        axs[0].set_xlabel(r"$s/\ell$")
        axs[0].set_ylabel(r"$q_y/q_0$")
        axs[0].set_title(f"$q_y$ (vert), Kn={kn:g}")
        axs[0].legend(frameon=False, fontsize=8, loc="best")
        axs[0].grid(alpha=0.15)
        # horizontal flux vs horizontal abscissa
        axs[1].plot(x_h, qx_h, label="FEM $q_x$", lw=1.5, color="tab:red")
        axs[1].scatter(s_qx_h_ref, qx_h_ref, facecolors="none", edgecolors="gray",
                        label="Tur-Prats $q_x$", s=15, zorder=3)
        axs[1].set_xlabel(r"$s/\ell$")
        axs[1].set_ylabel(r"$q_x/q_0$")
        axs[1].set_title(f"$q_x$ (horiz), Kn={kn:g}")
        axs[1].legend(frameon=False, fontsize=8, loc="best")
        axs[1].grid(alpha=0.15)
        fig.tight_layout()
        fname = os.path.join(OUTPUT_DIR, f"Kn{kn:g}_flux_profiles.pdf")
        fig.savefig(fname, dpi=300)
        plt.close(fig)

        # --- 5d. Compute error metrics (with explicit 1-D flattening + sorting) ---
        # Ensure everything is a 1-D numpy array:
        x_h_arr = np.asarray(x_h).ravel()
        T_h_arr = np.asarray(T_h).ravel()
        qx_h_arr = np.asarray(qx_h).ravel()

        y_v_arr = np.asarray(y_v).ravel()
        T_v_arr = np.asarray(T_v).ravel()
        qy_v_arr = np.asarray(qy_v).ravel()

        # Sort horizontal (so xp is strictly increasing)
        idx_h = np.argsort(x_h_arr)
        x_h_sorted  = x_h_arr[idx_h]
        T_h_sorted  = T_h_arr[idx_h]
        qx_h_sorted = qx_h_arr[idx_h]

        # Sort vertical
        idx_v = np.argsort(y_v_arr)
        y_v_sorted = y_v_arr[idx_v]
        T_v_sorted = T_v_arr[idx_v]
        qy_v_sorted = qy_v_arr[idx_v]

        # Now interpolate onto the Tur-Prats abscissa:
        T_h_interp  = np.interp(s_T_h_ref,  x_h_sorted,  T_h_sorted)
        T_v_interp  = np.interp(s_T_v_ref,  y_v_sorted,  T_v_sorted)
        qx_h_interp = np.interp(s_qx_h_ref, x_h_sorted,  qx_h_sorted)
        qy_v_interp = np.interp(s_qy_v_ref, y_v_sorted,  qy_v_sorted)

        # Compute normalized L2 and L∞ errors:
        eps2_Th,  epsinf_Th  = l2_linf(T_h_interp,  T_h_ref)
        eps2_Tv,  epsinf_Tv  = l2_linf(T_v_interp,  T_v_ref)
        eps2_qx,  epsinf_qx  = l2_linf(qx_h_interp, qx_h_ref)
        eps2_qy,  epsinf_qy  = l2_linf(qy_v_interp, qy_v_ref)

        # Append one row to CSV
        with open(ERROR_CSV, "a") as fout:
            fout.write(f"{kn:g},"
                        f"{eps2_Th:.4e},{epsinf_Th:.4e},"
                        f"{eps2_Tv:.4e},{epsinf_Tv:.4e},"
                        f"{eps2_qx:.4e},{epsinf_qx:.4e},"
                        f"{eps2_qy:.4e},{epsinf_qy:.4e}\n")

        print(f"Done Kn={kn:g}: "
                f"eps2_T_h={eps2_Th:.2e}, eps∞_T_h={epsinf_Th:.2e}, "
                f"eps2_qx={eps2_qx:.2e}, eps∞_qx={epsinf_qx:.2e}")
        print("  CSV row appended.")

    # ----------------------------------------
    #  6. Generate LaTeX error table (rank 0 only)
    # ----------------------------------------
    if rank == 0:
        # Define column names
        cols = ["Kn",
                "ε₂(Tₕ)", "ε∞(Tₕ)",
                "ε₂(Tᵥ)", "ε∞(Tᵥ)",
                "ε₂(qₓ)", "ε∞(qₓ)",
                "ε₂(qᵧ)", "ε∞(qᵧ)"]
        df = pd.read_csv(ERROR_CSV, header=None, names=cols)
        # Format floats in scientific notation with 2 decimal places
        def fmt(x): return f"{x:.2e}"
        df_str = df.copy()
        for c in cols[1:]:
            df_str[c] = df[c].map(fmt)

        # Print LaTeX tabular
        latex_table = df_str.to_latex(index=False, escape=False,
                                      column_format="c" * len(cols))
        print("\n====== LaTeX TABLE ======\n")
        print(latex_table)
        print("\n=========================\n")


if __name__ == "__main__":
    main()
