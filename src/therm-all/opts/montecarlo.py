import numpy as np
import torch
import json
import os
from mpi4py import MPI
import dolfinx.io
from solver_gke_module import GKESolver
from sim_config import SimulationConfig
from image_processing import generate_images
from vae_module import load_vae_model
from mesh_generator import MeshGenerator
import matplotlib.pyplot as plt


class MonteCarloOptimizer:
    def __init__(self, config_path, num_samples=100):
        with open(config_path, "r") as f:
            self.config_dict = json.load(f)

        self.config = SimulationConfig(self.config_dict)
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank

        # Load VAE model
        self.model = load_vae_model(self.rank, self.config.latent_size)

        # Solver
        self.solver = None
        self.init_solver()

        # Monte Carlo parameters
        self.num_samples = num_samples
        self.results = []

    def run_solver(self, latent_vector):
        img_list = generate_images(self.config, [latent_vector], self.model)
        avg_temp = self.solver.solve_image(img_list)
        std_dev = self.solver.get_std_dev()
        return avg_temp, std_dev

    def init_solver(self):
        mesh_gen = MeshGenerator(self.config)
        if self.rank == 0:
            if self.config.symmetry:
                mesh_gen.sym_create_mesh()
            else:
                mesh_gen.create_mesh()

        self.comm.barrier()
        msh, _, facet_markers = dolfinx.io.gmshio.read_from_msh(
            "domain_with_extrusions.msh", MPI.COMM_SELF, gdim=2
        )

        # Solver
        self.solver = GKESolver(msh, facet_markers, self.config)

    def run_monte_carlo(self, n_samples=100):
        best_obj = np.inf
        best_z = None
        best_results = None
        all_results = []
        pareto_front = []

        # run once first to add to the pareto front
        z = np.random.randn(self.config.latent_size)
        avg_temp, std_dev = self.run_solver(z)
        pareto_front.append[[avg_temp, std_dev]]
        # Loop through
        for i in range(n_samples):
            z = np.random.randn(self.config.latent_size)
            avg_temp, std_dev = self.run_solver(z)

            # Store results
            all_results.append([avg_temp, std_dev])
        
            # current_obj = avg_temp + std_dev
            # if current_obj < best_obj:
            #     best_obj = current_obj
            #     best_z = z
            #     best_results = (avg_temp, std_dev)

            # check with pareto front:


            if self.rank == 0:
                print(f"Iteration {i+1}/{n_samples} -> Avg Temp: {avg_temp}, Std Dev: {std_dev}")

        if self.rank == 0:
            np.save("monte_carlo_results.npy", np.array(all_results))
            np.save("best_z_vector.npy", best_z)
            print(f"Best found solution: Avg Temp={best_results[0]}, Std Dev={best_results[1]}")
            self.save_best_results(best_z, best_results)

        return best_z, best_results

    def save_best_results(self, best_z, best_results):
        os.makedirs("results", exist_ok=True)
        np.save("results/best_z.npy", best_z)
        with open("results/best_results.json", "w") as f:
            json.dump({"avg_temp": best_results[0], "std_dev": best_results[1]}, f)


if __name__ == "__main__":
    config_path = "logs/_ONE_SOURCE_CMAES/cmaes_sym_1source_z4/config.json"
    mc_optimizer = MonteCarloOptimizer(config_path)
    mc_optimizer.init_solver()
    best_z, best_results = mc_optimizer.run_monte_carlo(n_samples=50)
