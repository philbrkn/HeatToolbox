#!/usr/bin/env python3
"""
mesh_refinement_study2.py

Performs a mesh refinement study by running simulations at various mesh resolutions,
extracting the average temperature from log files, computing convergence rates,
and plotting both the error versus element size (log–log) and the raw average
temperature versus element size.

Assumptions:
- The simulation prints a line containing "Average temperature: <value>" in the log.
- The command to run the solver is:
    python3 -m heatoptim.main --config <config_path> --res <resolution>
- `res` corresponds to the number of elements per characteristic length, so that
  the physical element size is h = LENGTH / res.
"""

import os
import re
import subprocess

import numpy as np
import matplotlib.pyplot as plt


def run_simulations(resolutions, config_template, output_folder, length):
    """
    Run the heatoptim solver for each mesh resolution and parse the average temperature.

    Parameters
    ----------
    resolutions : list of float
        Number of elements per LENGTH (from coarse to fine).
    config_template : str
        Path to the template config file (assumed to be JSON). Each run will overwrite
        or copy this to the run-specific config if needed.
    output_folder : str
        Directory in which to save log files and plots.
    length : float
        Characteristic length (m) used to compute element size h = length / res.

    Returns
    -------
    collected : list of tuples (h, T_avg)
        Filtered list of (element size, average temperature) for successful runs.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    raw_results = []  # Will hold tuples (res, T_avg or None)
    pattern = re.compile(r"Average\s+temperature\s*:\s*([\d\.]+)")

    for res in resolutions:
        # Compute physical element size
        h = length / res

        # Construct command
        log_file = os.path.join(output_folder, f"sim_res_{res:.2e}.log")
        command = [
            "python3",
            "-m",
            "heatoptim.main",
            "--config", config_template,
            "--res", str(res)
        ]

        print(f"Running simulation at res = {res:.2e}  (h = {h:.2e} m)")
        try:
            with open(log_file, "w") as logf:
                subprocess.run(command,
                               stdout=logf,
                               stderr=subprocess.STDOUT,
                               text=True,
                               check=True)
        except subprocess.CalledProcessError:
            print(f"  → Simulation failed for res = {res:.2e}. See {log_file}.")
            raw_results.append((res, None))
            continue

        # Parse average temperature from log
        T_avg = None
        with open(log_file, "r") as logf:
            for line in logf:
                match = pattern.search(line)
                if match:
                    try:
                        T_avg = float(match.group(1))
                    except ValueError:
                        T_avg = None
                    break

        if T_avg is None:
            print(f"  → 'Average temperature' not found or could not parse in {log_file}.")
        else:
            print(f"  → Extracted T_avg = {T_avg:.6f} K")

        raw_results.append((res, T_avg))

        # (Optional) remove log file to save space:
        # os.remove(log_file)

    # Filter out failed runs (where T_avg is None)
    filtered = [(res, T) for (res, T) in raw_results if T is not None]

    # Convert to (h, T_avg) and sort by h ascending
    sorted_data = sorted(
        [(length / res, T) for (res, T) in filtered],
        key=lambda pair: pair[0]
    )

    return sorted_data


def compute_error_and_rate(sorted_data):
    """
    Given sorted (h, T_avg) data, compute the error versus the finest mesh and
    estimate the convergence rate via linear regression on log-log scale.

    Parameters
    ----------
    sorted_data : list of tuples (h, T_avg)
        Must be sorted by h in ascending order.

    Returns
    -------
    h_list : numpy.ndarray
        Array of element sizes (ascending).
    T_list : numpy.ndarray
        Array of corresponding average temperatures.
    err_list : numpy.ndarray
        Absolute difference |T(h) - T(h_min)|, where h_min is the smallest h.
    p : float
        Estimated convergence rate (slope on log-log plot).
    C : float
        Estimated constant in error ≈ C * h^p.
    """
    h_list = np.array([pair[0] for pair in sorted_data])
    T_list = np.array([pair[1] for pair in sorted_data])

    # Reference: T at the finest mesh (smallest h)
    T_ref = T_list[-1]
    err_list = np.abs(T_list - T_ref)

    # Exclude the finest mesh from regression if err = 0 exactly
    nonzero_indices = np.where(err_list > 0)[0]
    if len(nonzero_indices) < 2:
        # Not enough points to fit a slope
        return h_list, T_list, err_list, np.nan, np.nan

    h_fit = h_list[nonzero_indices]
    err_fit = err_list[nonzero_indices]

    logh = np.log(h_fit)
    loge = np.log(err_fit)
    coeff = np.polyfit(logh, loge, 1)
    p = coeff[0]
    C = np.exp(coeff[1])

    return h_list, T_list, err_list, p, C


def plot_results(h_list, T_list, err_list, p, C, output_folder):
    """
    Create and save two plots:
      1. Error vs. h on a log–log scale, with a fitted line of slope p.
      2. Raw average temperature vs. h (semilog or linear).

    Parameters
    ----------
    h_list : numpy.ndarray
        Array of element sizes.
    T_list : numpy.ndarray
        Array of average temperatures.
    err_list : numpy.ndarray
        Array of errors relative to the finest mesh.
    p : float
        Convergence slope from regression.
    C : float
        Constant in C * h^p.
    output_folder : str
        Directory in which to save the plots.
    """
    # 1. Plot error vs. h on log–log scale
    plt.figure(figsize=(6, 5))
    plt.loglog(h_list, err_list, 'o-', label="Computed error")
    if not np.isnan(p):
        h_fit = np.linspace(h_list.min(), h_list.max(), 50)
        err_fit = C * h_fit**p
        plt.loglog(h_fit, err_fit, '--',
                   label=f"Fit: slope = {p:.2f}")
    plt.xlabel("Element size $h$ (m)")
    plt.ylabel("Error $|T_{\\mathrm{avg}}(h) - T_{\\mathrm{avg}}(h_{\\min})|$ (K)")
    plt.title("Mesh Convergence: Error vs. Element Size")
    plt.grid(True, which="both", ls="--")
    plt.legend(loc="lower right")
    plt.gca().invert_xaxis()  # Optional: finer meshes on the right
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "error_vs_h_loglog.png"))
    plt.close()

    # 2. Plot raw average temperature vs. h
    plt.figure(figsize=(6, 5))
    plt.semilogx(h_list, T_list, 'o-')
    plt.xlabel("Element size $h$ (m)")
    plt.ylabel("Average Temperature $T_{\\mathrm{avg}}$ (K)")
    plt.title("Mesh Convergence: $T_{\\mathrm{avg}}$ vs. Element Size")
    plt.grid(True, which="both", ls="--")
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "Tavg_vs_h_semilog.png"))
    plt.close()


def main():
    # Characteristic length of the domain (m)
    LENGTH = 5e-07

    # Folder to store logs and plots
    OUTPUT_NAME = "bimaterial"
    OUTPUT_FOLDER = os.path.join("mesh_refinement_study", OUTPUT_NAME)
    CONFIG_TEMPLATE = os.path.join(OUTPUT_FOLDER, "config.json")  # Replace with actual config path

    # Mesh resolutions to test (number of elements per LENGTH)
    resolutions = [
        10, 15, 20, 30, 5, 8, 12, 2, 3, 1, 0.5, 0.1
    ]
    # Sort resolutions so that h (LENGTH / res) is ascending
    resolutions = sorted(resolutions, key=lambda r: LENGTH / r)

    # Run simulations and collect (h, T_avg)
    sorted_data = run_simulations(
        resolutions=resolutions,
        config_template=CONFIG_TEMPLATE,
        output_folder=OUTPUT_FOLDER,
        length=LENGTH
    )

    if len(sorted_data) < 2:
        print("Not enough valid data points for convergence analysis.")
        return

    # Compute error and convergence rate
    h_list, T_list, err_list, p, C = compute_error_and_rate(sorted_data)

    # Print convergence rate
    if not np.isnan(p):
        print(f"\nEstimated convergence rate: p = {p:.4f}")
    else:
        print("\nUnable to estimate convergence rate (insufficient non-zero errors).")

    # Print mesh sizes and errors
    print("\nMesh Size (h)   Average Temperature (K)   Error (K)")
    for h, T, e in zip(h_list, T_list, err_list):
        print(f"{h:.2e}        {T:.6f}                {e:.6f}")

    # Plot and save figures
    plot_results(h_list, T_list, err_list, p, C, OUTPUT_FOLDER)

    print(f"\nPlots saved in '{OUTPUT_FOLDER}'.")


if __name__ == "__main__":
    main()
