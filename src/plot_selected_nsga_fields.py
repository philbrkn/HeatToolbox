import os
import json
import numpy as np
import pickle
from mpi4py import MPI
import dolfinx.io
from sim_config import SimulationConfig
from solver_gke_module import GKESolver
from vae_module import load_vae_model
from image_processing import generate_images
from post_processing_fenicsx import PostProcessingModule
import matplotlib.pyplot as plt


class NSGAVisualizer:
    def __init__(self, iter_path, config_path):
        self.iter_path = iter_path
        with open(config_path, "r") as f:
            self.config_dict = json.load(f)
        self.config_dict["visualize"]["gamma"] = True
        self.config = SimulationConfig(self.config_dict)
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.model = load_vae_model(self.rank, self.config.latent_size)

        with open(iter_path + "/NSGA_Result.pk1", "rb") as f:
            self.res = pickle.load(f)

    def visualize_fields(self, positions=[0, 1, 2]):
        """Visualize gamma fields for selected positions on Pareto front."""
        for pos in positions:
            latent_vector = self.res.X[pos]

            # Generate image
            img_list = generate_images(self.config, [latent_vector], self.model)

            # Solve the heat problem with Fenicsx
            solver = self.run_fenics_solver(img_list=img_list)

            # Post-process results
            post_processor = PostProcessingModule(self.rank, self.config)
            q, T = solver.U.sub(0).collapse(), solver.U.sub(1).collapse()
            V1, _ = solver.U.function_space.sub(1).collapse()

            # Plot gamma field
            # join with self.iter_oath
            gamma_field_filename = os.path.join(self.iter_path, f"pos{positions.index(pos)}_")
            post_processor.postprocess_results(q, T, V=solver.W, msh=solver.msh, gamma=solver.gamma, fileadd=gamma_field_filename)

    def run_fenics_solver(self, img_list):
        mesh_file = "domain_with_extrusions.msh"
        msh, cell_markers, facet_markers = dolfinx.io.gmshio.read_from_msh(
            mesh_file, MPI.COMM_WORLD, gdim=2
        )
        solver = GKESolver(msh, facet_markers, self.config)
        solver.solve_image(img_list)
        return solver


if __name__ == "__main__":
    ITER_PATH = "logs/_ONE_SOURCE_NSGA/test_nsga_10mpi_z16"
    CONFIG_PATH = ITER_PATH + "/config.json"

    visualizer = NSGAVisualizer(ITER_PATH, CONFIG_PATH)
    # Select the points from Pareto front: 0 - min avg temp, 1 - compromise, 2 - min std dev
    visualizer.visualize_fields(positions=[0, 1, 2])
