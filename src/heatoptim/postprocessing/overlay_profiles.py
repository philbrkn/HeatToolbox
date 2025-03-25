"""
overlay_profiles.py

This script iterates over all subfolders inside a given base directory (e.g. "logs/_ONE_SOURCE_CMAES"),
loads each subfolder’s config.json, forces load_cma_result, and re‑runs the heat simulation to extract a
temperature profile. Finally, it overlays the extracted profiles into one plot.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

# Import your modules – adjust these imports if your folder structure differs
from heatoptim.config.sim_config import SimulationConfig
from heatoptim.solvers.mesh_generator import MeshGenerator
from heatoptim.solvers.solver_gke_module import GKESolver
from heatoptim.utilities.vae_module import load_vae_model
from heatoptim.utilities.image_processing import generate_images
from heatoptim.postprocessing.post_processing_gke import PostProcessingGKE
from heatoptim.utilities.log_utils import read_last_latent_vector
import dolfinx.io


def get_temperature_profile_from_simulation(config):
    """
    Run a single simulation (using the load_cma_result branch) and return
    the horizontal temperature profile.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # --- Mesh generation ---
    mesh_gen = MeshGenerator(config)
    if rank == 0:
        if config.symmetry:
            mesh_gen.sym_create_mesh()
        else:
            mesh_gen.create_mesh()
    comm.Barrier()
    # The mesh file must be available at this path
    msh, cell_markers, facet_markers = dolfinx.io.gmshio.read_from_msh(
        "domain_with_extrusions.msh", MPI.COMM_SELF, gdim=2
    )

    # --- Create solver instance ---
    if config.solver_type == "gke":
        solver = GKESolver(msh, facet_markers, config)
    elif config.solver_type == "fourier":
        raise NotImplementedError(
            "Only GKE solver is implemented in this overlay example."
        )
    else:
        raise ValueError(f"Unknown solver type: {config.solver_type}")

    # --- Load VAE model ---
    model = load_vae_model(rank, config.latent_size)

    # --- Load latent vectors from CMA log ---
    cma_log_file = os.path.join(config.log_dir, "cma_logs", "outcma_xrecentbest.dat")
    latent_vectors = read_last_latent_vector(
        cma_log_file, config.latent_size, len(config.source_positions)
    )

    # --- Generate image list and solve the simulation ---
    img_list = generate_images(config, latent_vectors, model)
    # This call solves the heat problem and stores the solution in solver.U
    _ = solver.solve_image(img_list)

    # --- Extract the temperature profile ---
    if config.solver_type == "gke":
        T = solver.U.sub(1).collapse()
        q = solver.U.sub(0).collapse()
        q_x, q_y = q.split()  # extract components
        # HORIZONTAL FIRST
        x_char = config.L_X if config.symmetry else config.L_X / 2
        x_end = x_char
        y_val = config.L_Y - (4 * config.LENGTH / 8)
        # Use a temporary PostProcessingModule instance to extract the profile
        ppm = PostProcessingGKE(rank, config)
        (x_vals, T_x) = ppm.get_temperature_line(T, msh, "horizontal", start=0, end=x_end, value=y_val)
        (_, q_x_vals_horiz) = ppm.get_temperature_line(q_x, msh, "horizontal", start=0, end=x_end, value=y_val)

        # vertical line:
        y_end = config.L_Y + config.SOURCE_HEIGHT
        x_val = x_char - config.LENGTH / 8
        (y_vals, T_y) = ppm.get_temperature_line(T, msh, "vertical", start=0, end=y_end, value=x_val)
        (_, q_y_vals_vert) = ppm.get_temperature_line(q_y, msh, "vertical", start=0, end=y_end, value=x_val)

    # normalize x and y vals by config.ell_si
    x_vals = (x_vals[-1] - x_vals) / config.ELL_SI
    y_vals = (y_vals[-1] - y_vals) / config.ELL_SI

    return x_vals, y_vals, T_x, T_y, q_x_vals_horiz, q_y_vals_vert


def overlay_temperature_profiles(base_dir):
    """
    For each subfolder (each simulation run) in base_dir, load its config,
    force load_cma_result, run the simulation, and extract the temperature profile.
    Then overlay all the profiles into one plot.
    """
    Tx_overlay_data = []
    Ty_overlay_data = []
    q_x_overlay_data = []
    q_y_overlay_data = []
    # List subfolders (assumes each subfolder contains a config.json)
    subfolders = [
        f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))
    ]
    subfolders.sort()  # sort for consistent order

    for subfolder in subfolders:
        subfolder_path = os.path.join(base_dir, subfolder)
        config_path = os.path.join(subfolder_path, "config.json")
        if not os.path.exists(config_path):
            print(f"No config.json in {subfolder_path}, skipping.")
            continue

        # Load the configuration from the subfolder
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        # --- Update configuration for overlay run ---
        # Force loading CMA-ES result and disable interactive plotting
        config_dict["load_cma_result"] = True
        config_dict["visualize"] = {
            "gamma": False,
            "temperature": False,
            "flux": False,
            "profiles": False,
            "pregamma": False,
            "effective_conductivity": False,
        }
        # config["res"] = 20.0  # Increase resolution for better visualization
        config_dict["plot_mode"] = "screenshot"

        # Make sure the log_dir is set to the current subfolder (so that CMA logs are found)
        # remove the logs/ if its there:
        if config_dict["log_name"].startswith("logs/"):
            config_dict["log_name"] = config_dict["log_name"][5:]
        else:
            config_dict["log_name"] = config_dict["log_name"]

        config = SimulationConfig(config_dict)
        print(f"Running simulation for subfolder: {subfolder} ...")
        # try:
        x_vals, y_vals, T_x, T_y, q_x_vals_horiz, q_y_vals_vert = get_temperature_profile_from_simulation(config)
        Tx_overlay_data.append((subfolder, x_vals, T_x))
        Ty_overlay_data.append((subfolder, y_vals, T_y))
        q_x_overlay_data.append((subfolder, x_vals, q_x_vals_horiz))
        q_y_overlay_data.append((subfolder, y_vals, q_y_vals_vert))
        # except Exception as e:
        #     print(f"Simulation failed for {subfolder} with error: {e}")

    # --- Create overlay plot ---
    if len(Tx_overlay_data) == 0:
        print("No valid simulation data found; exiting.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.hsv(np.linspace(0, 1, len(Tx_overlay_data)))
    for i, (label, x_vals, T_profile) in enumerate(Tx_overlay_data):
        ax.plot(x_vals, T_profile, label=f"z={2**(i+1)}") #, color=colors[i])
    ax.set_xlabel("Position (normalized)")
    ax.set_ylabel("Temperature (T)")
    ax.set_title("Overlay of Horizontal Temperature Profiles")
    ax.legend()
    ax.grid()
    # Save the overlay plot in the base directory
    output_path = os.path.join(base_dir, "horiz_overlay_temperature_profiles.png")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Overlay plot saved to {output_path}")

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.hsv(np.linspace(0, 1, len(Ty_overlay_data)))
    for i, (label, y_vals, T_profile) in enumerate(Ty_overlay_data):
        ax.plot(y_vals, T_profile, label=f"z={2**(i+1)}") #, color=colors[i])
    ax.set_xlabel("Position (normalized)")
    ax.set_ylabel("Temperature (T)")
    ax.set_title("Overlay of Vertical Temperature Profiles")
    ax.legend()
    ax.grid()
    # Save the overlay plot in the base directory
    output_path = os.path.join(base_dir, "vert_overlay_temperature_profiles.png")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Overlay plot saved to {output_path}")

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.hsv(np.linspace(0, 1, len(q_x_overlay_data)))
    for i, (label, x_vals, q_x) in enumerate(q_x_overlay_data):
        ax.plot(x_vals, q_x, label=f"z={2**(i+1)}") #, color=colors[i])
    ax.set_xlabel("Position (normalized)")
    ax.set_ylabel("Heat flux")
    ax.set_title("Overlay of Horizontal Flux Profiles")
    ax.legend()
    ax.grid()
    # Save the overlay plot in the base directory
    output_path = os.path.join(base_dir, "horiz_overlay_q_profiles.png")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Overlay plot saved to {output_path}")

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.hsv(np.linspace(0, 1, len(q_y_overlay_data)))
    for i, (label, y_vals, q_y) in enumerate(q_y_overlay_data):
        ax.plot(y_vals, q_y, label=f"z={2**(i+1)}")
    ax.set_xlabel("Position (normalized)")
    ax.set_ylabel("Heat flux")
    ax.set_title("Overlay of Vertical Flux Profiles")
    ax.legend()
    ax.grid()
    # Save the overlay plot in the base directory
    output_path = os.path.join(base_dir, "vert_overlay_q_profiles.png")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Overlay plot saved to {output_path}")



if __name__ == "__main__":
    # Adjust the base directory to match your folder structure.
    # In your case, you might run:
    base_directory = os.path.join("logs", "_ONE_SOURCE_CMAES")
    # base_directory = os.path.join("logs", "_THREE_SOURCES_CMAES")
    overlay_temperature_profiles(base_directory)
