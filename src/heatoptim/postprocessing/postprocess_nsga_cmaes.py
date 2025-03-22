"""
compare_nsga_cmaes.py

This script loads a configuration file, sets up the simulation (mesh, solver, VAE, etc.),
loads the latent vectors from the CMA-ES log, generates the image list, runs the heat simulation
(using GKESolver.solve_image and get_std_dev), and then uses the NSGA post-processing module
to plot the hypervolume convergence, Pareto front, and overlay the CMA-ES solution on the Pareto plot.
"""

import os
import json
import numpy as np
import dolfinx.io
from mpi4py import MPI


def main(config_file, ITER_PATH_NSGA):
    # Load configuration from JSON file.
    with open(config_file, "r") as f:
        config_dict = json.load(f)

    # Create a SimulationConfig instance (assumes sim_config.SimulationConfig exists).
    from heatoptim.config.sim_config import SimulationConfig

    config = SimulationConfig(config_dict)

    # Set up MPI.
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # Load the mesh (ensure that 'domain_with_extrusions.msh' is available in the working directory).
    msh, cell_markers, facet_markers = dolfinx.io.gmshio.read_from_msh(
        "domain_with_extrusions.msh", comm, gdim=2
    )

    # Create the solver instance (here using the GKE solver).
    from solver_gke_module import GKESolver

    solver = GKESolver(msh, facet_markers, config)

    # Load the VAE model.
    from heatoptim.utilities.vae_module import load_vae_model

    model = load_vae_model(rank, config.latent_size)

    # Load latent vectors from the CMA-ES log.
    from log_utils import read_last_latent_vector

    cma_log_file = os.path.join(config.log_dir, "cma_logs", "outcma_xrecentbest.dat")
    latent_vectors = read_last_latent_vector(
        cma_log_file, config.latent_size, len(config.source_positions)
    )

    # Generate the list of images from the latent vectors.
    from heatoptim.utilities.image_processing import generate_images

    img_list = generate_images(config, latent_vectors, model)

    # Run the simulation: solve for the temperature field and compute average temperature.
    avg_temp = solver.solve_image(img_list)  # Average temperature from simulation
    std_dev = solver.get_std_dev()  # Standard deviation of temperature field
    print(f"CMAES Average temperature: {avg_temp}, Standard deviation: {std_dev}")
    # Form the CMA-ES objective vector; here assumed to be two objectives.
    cmaes_obj = np.array([avg_temp, std_dev])

    # Run NSGA-related post-processing and overlay the CMA-ES point on the Pareto plot.
    from post_process_pymoo import PymooPostprocess

    PymooPost = PymooPostprocess(ITER_PATH_NSGA)
    print("Saving pymoo post-processing...")

    # Plot hypervolume convergence.
    # PymooPost.get_hyperbolic_convergence(path=ITER_PATH)

    # Plot the NSGA Pareto front.
    # PymooPost.get_pareto_plot(
    #     path=ITER_PATH_NSGA)
    #     plot_compromise=False,
    #     filter_outliers=False
    # )

    # Overlay the CMA-ES solution on the Pareto plot.
    PymooPost.get_pareto_plot_with_cmaes(ITER_PATH_NSGA, cmaes_obj)


if __name__ == "__main__":
    # Define the iteration (output) folder and configuration file.
    ITER_PATH_NSGA = "logs/_ONE_SOURCE_NSGA2/test_nsga_10mpi_z8"
    ITER_PATH_CMAES = "logs/_ONE_SOURCE_CMAES/cmaes_sym_1source_z8"
    
    config_file = os.path.join(ITER_PATH_CMAES, "config.json")
    main(config_file, ITER_PATH_NSGA)
