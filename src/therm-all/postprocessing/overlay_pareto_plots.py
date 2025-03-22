#!/usr/bin/env python
"""
overlay_pareto_fronts.py

This script overlays Pareto fronts from different NSGA runs.
For example, in the base directory logs/_ONE_SOURCE_NSGA2,
it looks for subfolders like:
  test_nsga_10mpi_z2
  test_nsga_10mpi_z4
  test_nsga_10mpi_z8
  test_nsga_10mpi_z16

Each subfolder must contain an NSGA_Result.pk1 file.
The script extracts the Pareto front (objective values) from each file
and plots them on a single figure, saving the overlay plot.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse


def load_pareto_front(iter_path):
    """
    Load NSGA_Result.pk1 from the given folder and return its Pareto front (F).
    """
    nsga_file = os.path.join(iter_path, "NSGA_Result.pk1")
    with open(nsga_file, "rb") as f:
        res = pickle.load(f)
    # Assume that the Pareto front is stored in res.F
    return res.F


def overlay_pareto_fronts(base_dir, output_file):
    """
    Loop over subdirectories in base_dir that contain NSGA_Result.pk1,
    extract their Pareto fronts, and overlay them on a single 2D plot.
    """
    # Find all subdirectories with NSGA_Result.pk1
    subdirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
        and os.path.exists(os.path.join(base_dir, d, "NSGA_Result.pk1"))
    ]
    subdirs.sort()  # sort alphabetically (or numerically if names permit)

    if not subdirs:
        print("No subdirectories with NSGA_Result.pk1 found in", base_dir)
        return

    plt.figure(figsize=(10, 8))

    # Define markers and colors for different runs.
    markers = ["o", "s", "^", "D", "v"]
    colors = ["r", "b", "g", "m", "c", "y", "k"]

    for i, subdir in enumerate(subdirs):
        try:
            F = load_pareto_front(subdir)
            # Check if the Pareto front is 2-dimensional
            if F.shape[1] == 2:
                # Use the subfolder name (e.g., "test_nsga_10mpi_z2") as label
                label = os.path.basename(subdir)
                plt.scatter(
                    F[:, 0],
                    F[:, 1],
                    marker=markers[i % len(markers)],
                    color=colors[i % len(colors)],
                    label=label,
                    alpha=0.7,
                )
            else:
                print(
                    f"Skipping {subdir}: 3D Pareto front encountered (only 2D is supported)."
                )
        except Exception as e:
            print(f"Error processing {subdir}: {e}")

    plt.xlabel("Average temperature")
    plt.ylabel("Temperature std dev")
    plt.title("Overlay of Pareto Fronts for Different z Values")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Overlayed Pareto front saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Overlay Pareto fronts from different NSGA runs."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory containing subfolders (e.g., logs/_ONE_SOURCE_NSGA2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="overlay_pareto_fronts.png",
        help="Output file for the overlay plot (PNG format)",
    )
    args = parser.parse_args()
    overlay_pareto_fronts(args.base_dir, args.output)


if __name__ == "__main__":
    # main()
    ITER_PATH = "logs/_ONE_SOURCE_NSGA3"
    output_file = os.path.join(ITER_PATH, "overlay_pareto_fronts.png")
    overlay_pareto_fronts(ITER_PATH, output_file)
