# main.py

# import libraries
import torch
from mpi4py import MPI
import argparse
import numpy as np
import dolfinx.io
import os

# import modules
from heatoptim.solvers.mesh_generator import MeshGenerator
from heatoptim.utilities.vae_module import load_vae_model, VAE, Flatten, UnFlatten
from heatoptim.utilities.image_processing import z_to_img, generate_images
from heatoptim.opts.cmaes import CMAESModule
from heatoptim.opts.bayes import BayesianModule
from heatoptim.postprocessing.post_processing_gke import PostProcessingGKE
from heatoptim.postprocessing.post_processing_fourier import PostProcessingFourier
from heatoptim.solvers.solver_gke_module import GKESolver
from heatoptim.solvers.solver_fourier_module import FourierSolver
from heatoptim.utilities.logging_module import LoggingModule
from heatoptim.config.sim_config import SimulationConfig
from heatoptim.utilities.log_utils import read_last_latent_vector


class SimulationController:
    def __init__(self, config):
        self.config = config
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.logger = LoggingModule(config) if config.logging_enabled and self.rank == 0 else None
        self.model = load_vae_model(self.rank, self.config.latent_size)

    def run_simulation(self):
        # Mesh generation
        mesh_generator = MeshGenerator(self.config)
        time1 = MPI.Wtime()

        # Create solver instance
        if self.rank == 0:
            if self.config.symmetry:
                mesh_generator.sym_create_mesh()
            else:
                mesh_generator.create_mesh()
        else:
            msh = None
            cell_markers = None
            facet_markers = None

        # Broadcast mesh data to all processes
        self.comm.barrier()
        msh, cell_markers, facet_markers = dolfinx.io.gmshio.read_from_msh(
            "domain_with_extrusions.msh", MPI.COMM_SELF, gdim=2
        )

        # create solver instance
        if self.config.solver_type == "gke":
            solver = GKESolver(msh, facet_markers, self.config)
        elif self.config.solver_type == "fourier":
            solver = FourierSolver(msh, facet_markers, self.config)
        else:
            raise ValueError(f"Unknown solver type: {self.config.solver_type}")

        if self.config.optim:
            # Run optimization
            optimizer = None
            if self.config.optimizer == "cmaes":
                optimizer = CMAESModule(
                    solver,
                    self.model,
                    torch.device("cpu"),
                    self.rank,
                    self.config,
                    logger=self.logger,
                )
                best_z_list = optimizer.optimize(
                    n_iter=100
                )  # Adjust iterations as needed
            elif self.config.optimizer == "nsga2":
                if self.config.parallelize:
                    from heatoptim.opts.nsga_mpi import optimize_nsga
                    best_z_list = optimize_nsga(solver, self.model, self.config, logger=self.logger)
                else:
                    from heatoptim.opts.nsga import optimize_nsga
                    best_z_list = optimize_nsga(solver, self.model, self.config, logger=self.logger)
            elif self.config.optimizer == "bayesian":
                optimizer = BayesianModule(
                    solver,
                    self.model,
                    torch.device("cpu"),
                    self.rank,
                    self.config,
                    logger=self.logger,
                )
                best_z_list = optimizer.optimize(init_points=10, n_iter=100)

            latent_vectors = best_z_list
            # Optional: Save the best_z to a file for future solving
            # if self.rank == 0:
            #     self.logger.save_optimized_latent_vectors(latent_vectors)
        elif self.config.load_cma_result:
            # Load latent vectors from CMA-ES log file
            latent_vectors = self.load_latent_vectors_from_cma_log()
            print(latent_vectors)
            # Proceed to generate images and solve
            img_list = generate_images(self.config, latent_vectors, self.model)

            if self.config.visualize['pregamma']:
                self.plot_image_list(img_list, self.config, logger=self.logger)

            avg_temp_global = solver.solve_image(img_list)
            time2 = MPI.Wtime()
            if self.rank == 0:
                print(f"Average temperature: {avg_temp_global} K")
                print(f"Time taken to solve: {time2 - time1:.3f} seconds")

            # Check if visualize list is not empty
            if self.config.visualize:
                import cma
                cma_log_dir = os.path.join(self.config.log_dir, "cma_logs")
                cma.plot(os.path.join(cma_log_dir, "outcma_"))
                cma.s.figsave(os.path.join(cma_log_dir, 'convergence_plots.png'))

                if self.config.solver_type == "gke":
                    post_processor = PostProcessingGKE(
                        self.rank, self.config, logger=self.logger
                    )
                    q, T = solver.U.sub(0).collapse(), solver.U.sub(1).collapse()
                    V1, _ = solver.U.function_space.sub(1).collapse()
                    post_processor.postprocess_results(q, T, V1, solver.msh, solver.gamma)
                elif self.config.solver_type == "fourier":
                    post_processor = PostProcessingFourier(
                        self.rank, self.config, logger=self.logger
                    )
                    q, T = None, solver.T
                    V1 = solver.V
                    post_processor.postprocess_results(T, V1, solver.msh, solver.gamma)

        else:
            # print the config dict:
            latent_vectors = self.get_latent_vectors()

            # Generate image from latent vector
            img_list = generate_images(self.config, latent_vectors, self.model)
            # check if pregamma key is true in the visualize dictinoary
            if self.config.visualize['pregamma']:
                self.plot_image_list(img_list, self.config, logger=self.logger)

            avg_temp_global = solver.solve_image(img_list)
            time2 = MPI.Wtime()
            if self.rank == 0:
                print(f"Average temperature: {avg_temp_global} K")
                print(f"Time taken to solve: {time2 - time1:.3f} seconds")

            # Check if visualize list is not empty
            if self.config.solver_type == "gke":
                post_processor = PostProcessingGKE(
                    self.rank, self.config, logger=self.logger
                )
                q, T = solver.U.sub(0).collapse(), solver.U.sub(1).collapse()
                V1, _ = solver.U.function_space.sub(1).collapse()
                post_processor.postprocess_results(q, T, V1, solver.msh, solver.gamma)
            elif self.config.solver_type == "fourier":
                post_processor = PostProcessingFourier(
                    self.rank, self.config, logger=self.logger
                )
                q, T = None, solver.T
                V1 = solver.V
                post_processor.postprocess_results(T, V1, solver.msh, solver.gamma)

    def get_latent_vectors(self):
        # Handle latent vector based on the selected method
        latent_vectors = []
        if self.config.latent_method == "manual":
            # Use the latent vector provided in args.latent
            z = np.array(self.config.latent)
            if len(z) != self.config.latent_size:
                raise ValueError(
                    f"Expected latent vector of size {self.config.latent_size}, got {len(z)}."
                )
            latent_vectors = [z] * len(self.config.source_positions)
        elif self.config.latent_method == "random":
            # Generate random latent vectors
            for _ in range(len(self.config.source_positions)):
                z = np.random.randn(self.config.latent_size)
                latent_vectors.append(z)
        elif self.config.latent_method == "preloaded":
            # Load latent vectors from file
            try:
                best_z_list = np.load("best_latent_vector.npy", allow_pickle=True)
                print("Opening best vector from file")
                latent_vectors = best_z_list
            except FileNotFoundError:
                raise FileNotFoundError(
                    "No saved latent vectors found. Please provide a valid file."
                )
        return latent_vectors

    def plot_image_list(self, img_list, config, logger=None):
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, len(img_list), figsize=(15, 5))
        if len(img_list) == 1:
            axs.imshow(img_list[0], cmap="gray")
            axs.axis("off")
        else:
            for i, img in enumerate(img_list):
                axs[i].imshow(img, cmap="gray")
                axs[i].axis("off")
        if config.plot_mode == "screenshot":
            if logger:
                logger.save_image(fig, "image_list.png")
            else:
                plt.savefig("image_list.png")
        else:
            plt.show()

    def load_latent_vectors_from_cma_log(self):
        """Load the most recent CMA-ES result and update latent vectors."""
        cma_log_file = os.path.join(self.config.log_dir, "cma_logs", "outcma_xrecentbest.dat")
        z_dim = self.config.latent_size
        num_sources = len(self.config.source_positions)
        print(f"Zdim and num sources: {z_dim}, {num_sources}")
        latent_vectors = read_last_latent_vector(cma_log_file, z_dim, num_sources)
        return latent_vectors


def parse_arguments():
    # Command-line arguments to determine the modes
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration JSON file."
    )
    parser.add_argument(
        "--res", type=float, default=12.0, help="Mesh resolution (default: 12.0)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    config = SimulationConfig(args.config, arg_res=args.res)
    controller = SimulationController(config)
    controller.run_simulation()
