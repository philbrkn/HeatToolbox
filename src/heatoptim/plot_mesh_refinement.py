#!/usr/bin/env python3
"""
mesh_refinement_plot.py

Reads all mesh-refinement log files (sim_res_*.log), extracts average-temperature data,
computes convergence errors against the finest mesh, fits a log-log convergence rate,
and then plots:

  1) Error vs. element size (h) on a log-log scale, showing the fitted slope.
  2) Raw average temperature vs. element size on a semilog-x scale.

Also prints out which element size h* meets the specified tolerance.

Instructions:
  - Place this script in the same folder as your log files (or adjust LOG_FOLDER).
  - Ensure each log file is named exactly "sim_res_{res:.2e}.log".
  - Make sure each log file contains a line like: "Average temperature: 345.6789".
"""

import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

# ─────────────── USER‐DEFINED PARAMETERS ────────────────────

# 1. Path to folder containing all "sim_res_*.log" files
LOG_NAME = "blank"
LOG_FOLDER = os.path.join("mesh_refinement_study", LOG_NAME)

# 2. Characteristic physical length (in meters) used in h = LENGTH / res
LENGTH = 5e-7

# 3. Absolute tolerance (in Kelvin) for |T_avg(h) - T_avg(h_finest)|
DELTA_T_TOL = 0.0005  # e.g. 0.5 K

# 4. Regex pattern to match "sim_res_<res>.log" and extract the numeric res
FILENAME_PATTERN = r"sim_res_([0-9eE\+\-\.]+)\.log"

# 5. Regex to find the "Average temperature:" line inside each log file
TMP_PATTERN = re.compile(r"Average\s+temperature\s*:\s*([\d\.]+)")

# ─────────────── END USER PARAMETERS ────────────────────────


def parse_all_logs(log_folder):
    """
    Scan `log_folder` for files matching sim_res_*.log, extract:
      - res (float)
      - T_avg (float, from inside the file)
    Returns a list of tuples: [(res, T_avg), ...], unsorted.
    """
    out = []
    # Build the full glob pattern
    search_path = os.path.join(log_folder, "sim_res_*.log")
    for filepath in glob.glob(search_path):
        fname = os.path.basename(filepath)
        m = re.match(FILENAME_PATTERN, fname)
        if not m:
            continue
        try:
            res_val = float(m.group(1))
        except ValueError:
            print(f"  [Warning] Could not parse resolution from filename: {fname}")
            continue

        T_avg = None
        with open(filepath, "r") as f:
            for line in f:
                mm = TMP_PATTERN.search(line)
                if mm:
                    try:
                        T_avg = float(mm.group(1))
                    except ValueError:
                        T_avg = None
                    break

        if T_avg is None:
            print(f"  [Warning] 'Average temperature' not found or parse failed in {fname}")
        else:
            out.append((res_val, T_avg))

    return out


def build_convergence_data(parsed_list, length):
    """
    Given [(res, T_avg), ...], compute and return:
      - h_list: sorted array of element sizes (ascending)
      - T_list: corresponding average temperatures (same order)
      - err_list: |T(h) - T(h_min)|
    """
    # Filter out any entries where T_avg is None (just in case)
    filtered = [(res, T) for (res, T) in parsed_list if T is not None]

    # Compute h = length / res for each entry
    data = [(length / res, T) for (res, T) in filtered]

    # Sort by h ascending
    data.sort(key=lambda pair: pair[0])


    h_list = np.array([pair[0] for pair in data])
    T_list = np.array([pair[1] for pair in data])

    # The finest mesh is the one with smallest h (last element)
    T_ref = T_list[0]
    err_list = np.abs(T_list - T_ref)

    return h_list, T_list, err_list


def fit_convergence_rate(h_list, err_list):
    """
    Perform linear regression on log(h) vs log(err) for points where err > 0.
    Returns (p, C) such that err ≈ C * h^p.
    If too few nonzero points, returns (np.nan, np.nan).
    """
    nonzero_mask = err_list > 0
    if np.count_nonzero(nonzero_mask) < 2:
        return np.nan, np.nan

    h_fit = h_list[nonzero_mask]
    err_fit = err_list[nonzero_mask]
    logh = np.log(h_fit)
    loge = np.log(err_fit)
    coeff = np.polyfit(logh, loge, 1)
    p = coeff[0]
    C = np.exp(coeff[1])
    return p, C


def plot_convergence(h_list, T_list, err_list, p, C, tol, output_folder):
    """
    Create and save two plots:
      (1) log-log plot of err_list vs h_list, with a fitted line
      (2) semilog-x plot of T_list vs h_list
    Also draws a horizontal line at err = tol and annotates the chosen h*.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Remove the point where err == 0 (the reference mesh)
    nonzero_mask = err_list > 0         # already used for regression
    h_plot = h_list[nonzero_mask]
    err_plot = err_list[nonzero_mask]

    # Find the smallest h where err < tol
    below_tol_indices = np.where(err_list < tol)[0]
    h_star = None
    if below_tol_indices.size > 0:
        idx = below_tol_indices[0]
        h_star = h_list[idx]
        err_star = err_list[idx]
    else:
        idx = None

    # 1) Error vs. h (log-log)
    plt.figure(figsize=(6, 5))
    plt.loglog(h_plot, err_plot, 'o-', label="Computed error")
    if not np.isnan(p):
        h_fit = np.logspace(np.log10(h_list.min()), np.log10(h_list.max()), 50)
        err_fit = C * h_fit**p
        plt.loglog(h_fit, err_fit, '--', label=f"Fit: slope = {p:.2f}")
    # Draw horizontal tolerance line
    plt.axhline(y=tol, color='gray', linestyle=':', label=f"tol = {tol:.2f} K")
    # Mark chosen point
    if h_star is not None:
        plt.plot(h_star, err_star, 'rD', label=f"h* = {h_star:.2e} m")
        plt.vlines(h_star, ymin=err_star, ymax=tol, colors='r', linestyles=':')
        plt.hlines(err_star, xmin=h_list.min(), xmax=h_star, colors='r', linestyles=':')
    plt.xlabel("Element size $h$ (m)")
    plt.ylabel("Error $|T_{\\mathrm{avg}}(h) - T_{\\mathrm{avg}}(h_{\\min})|$ (K)")
    plt.title("Mesh Convergence: Error vs. Element Size")
    plt.grid(True, which="both", ls="--")
    plt.legend(loc="lower right")
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "error_vs_h_loglog.png"))
    plt.close()

    # 2) Raw T_avg vs. h (semilog-x)
    plt.figure(figsize=(6, 5))
    plt.semilogx(h_list, T_list, 'o-')
    if h_star is not None:
        plt.plot(h_star, T_list[idx], 'rD')
    plt.xlabel("Element size $h$ (m)")
    plt.ylabel("Average Temperature $T_{\\mathrm{avg}}$ (K)")
    plt.title("Mesh Convergence: $T_{\\mathrm{avg}}$ vs. Element Size")
    plt.grid(True, which="both", ls="--")
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "Tavg_vs_h_semilog.png"))
    plt.close()

    return h_star


def main():
    # 1) Parse all logs
    parsed = parse_all_logs(LOG_FOLDER)
    if len(parsed) == 0:
        print("No valid log files found in:", LOG_FOLDER)
        return

    # 2) Build (h, T_avg, err) arrays
    h_list, T_list, err_list = build_convergence_data(parsed, LENGTH)

    # 3) Fit convergence rate
    p, C = fit_convergence_rate(h_list, err_list)
    if np.isnan(p):
        print("Not enough data points to fit a convergence rate.")
    else:
        print(f"Estimated convergence rate: p = {p:.4f}")

    # 4) Identify h* that meets the tolerance
    h_star = plot_convergence(
        h_list=h_list,
        T_list=T_list,
        err_list=err_list,
        p=p,
        C=C,
        tol=DELTA_T_TOL,
        output_folder=LOG_FOLDER
    )

    if h_star is not None:
        print(f"\nTolerance of {DELTA_T_TOL:.2f} K is first met at h* = {h_star:.2e} m.")
    else:
        print(f"\nNo mesh size satisfies err < {DELTA_T_TOL:.2f} K.")


if __name__ == "__main__":
    main()
