#!/usr/bin/env python3

import argparse
import os
import numpy as np
from mpi4py import MPI
import dolfinx.io
from heatoptim.solvers.mesh_generator import MeshGenerator
from heatoptim.utilities.vae_module import load_vae_model
from heatoptim.utilities.image_processing import generate_images
from heatoptim.solvers.solver_gke_module import GKESolver
from heatoptim.solvers.solver_fourier_module import FourierSolver
from heatoptim.postprocessing.post_processing_gke import PostProcessingGKE
from heatoptim.config.sim_config import SimulationConfig


def main():

    solutions = {}
    labels = ["silicon", "diamond", "bimaterial"]
    for label in labels:
        
        if label == "bimaterial":
            CONFIG_PATH = "configs/bimaterialconfig.json"
            config = SimulationConfig(CONFIG_PATH)
            if config.latent_method == "manual":
                z = np.array(config.latent)
            else:
                z = np.random.randn(config.latent_size)
            model = load_vae_model(rank, config.latent_size)
            img = generate_images(config, [z], model)[0]

        elif label == "diamond":
            CONFIG_PATH = "configs/bimaterialconfig.json"
            config = SimulationConfig(CONFIG_PATH)
            img = np.ones((128, 128))
        else:
            CONFIG_PATH = "configs/blankconfig.json"
            config = SimulationConfig(CONFIG_PATH)
            img = np.zeros((128, 128))

        comm = MPI.COMM_WORLD
        rank = comm.rank
        mesh_generator = MeshGenerator(config)
        if rank == 0:
            if config.symmetry:
                mesh_generator.sym_create_mesh()
            else:
                mesh_generator.create_mesh()
        comm.barrier()

        msh, cell_markers, facet_markers = dolfinx.io.gmshio.read_from_msh(
            "domain_with_extrusions.msh", MPI.COMM_SELF, gdim=2
        )

        solver = GKESolver(msh, facet_markers, config)
        img_list = [img]
        avg_temp = solver.solve_image(img_list)
        if rank == 0:
            print(
                f"{label.capitalize()}: "
                f"avg T = {avg_temp:.6e} K, "
                f"avg flux = {solver.get_avg_flux():.6e} W/m^2, "
                f"σ_T = {solver.get_std_dev():.6e} K"
            )
        if config.solver_type == "gke":
            q_fun, T_fun = solver.U.sub(0).collapse(), solver.U.sub(1).collapse()
        else:
            q_fun = None
            T_fun = solver.T
        solutions[label] = (q_fun, T_fun)

    if rank == 0:
        post = PostProcessingGKE(rank, config)
        post.plot_overlaid_profiles(msh, solutions)


if __name__ == "__main__":
    main()
