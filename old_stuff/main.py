# main.py

# import libraries
import torch
from mpi4py import MPI
import argparse
import numpy as np
import dolfinx.io

# import modules
from mesh_generator import MeshGenerator
from vae_module import load_vae_model, VAE, Flatten, UnFlatten
from image_processing import z_to_img
from optimization_module import BayesianModule, CMAESModule
from post_processing import PostProcessingModule
from solver_module import Solver
from logging_module import LoggingModule
from sim_config import SimulationConfig


class SimulationController:
    def __init__(self, config):
        self.config = config
        self.logger = LoggingModule(config) if config.logging_enabled else None
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.model = load_vae_model(self.rank)

    def run_simulation(self):
        # Mesh generation
        mesh_generator = MeshGenerator(self.config)
        time1 = MPI.Wtime()

        # Create solver instance
        if self.rank == 0:
            if self.config.symmetry:
                msh, cell_markers, facet_markers = mesh_generator.sym_create_mesh()
            else:
                msh, cell_markers, facet_markers = mesh_generator.create_mesh()
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
        solver = Solver(msh, facet_markers, self.config)

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
            if self.rank == 0:
                np.save("best_latent_vector.npy", best_z_list)
        else:
            latent_vectors = self.get_latent_vectors()

        # Generate image from latent vector
        img_list = self.generate_images(latent_vectors)

        if "pregamma" in self.config.visualize:
            self.plot_image_list(img_list, self.config, logger=self.logger)

        avg_temp_global = solver.solve_image(img_list)
        time2 = MPI.Wtime()
        if self.rank == 0:
            print(f"Average temperature: {avg_temp_global} K")
            print(f"Time taken to solve: {time2 - time1:.3f} seconds")

        # Check if visualize list is not empty
        if self.config.visualize:
            post_processor = PostProcessingModule(
                self.rank, self.config, logger=self.logger
            )
            post_processor.postprocess_results(solver.U, solver.msh, solver.gamma)

        self.log_final_results(avg_temp_global, time2 - time1, latent_vectors)

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

    def generate_images(self, latent_vectors):
        # Generate image from latent vector
        img_list = []
        for z in latent_vectors:
            if self.config.blank:
                img = np.zeros((128, 128))
            else:
                # Ensure z is reshaped correctly if needed
                img = z_to_img(z.reshape(1, -1), self.model, self.config.vol_fraction)
            img_list.append(img)

        # Apply symmetry to each image if enabled
        if self.config.symmetry:
            img_list = [img[:, : img.shape[1] // 2] for img in img_list]

        return img_list

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

    def log_final_results(self, avg_temp_global, runtime, latent_vectors):
        if self.config.optim:
            final_results = {
                "average_temperature": avg_temp_global,
                "best_latent_vector": latent_vectors,
            }
            if self.logger:
                self.logger.log_results(final_results)
        else:
            results = {"average_temperature": avg_temp_global, "runtime": runtime}
            if self.logger:
                self.logger.log_results(results)


def parse_arguments():
    # Command-line arguments to determine the modes
    parser = argparse.ArgumentParser()
    parser.add_argument("--optim", action="store_true", help="Run optimization.")
    parser.add_argument(
        "--optimizer",
        choices=["cmaes", "bayesian"],
        default="bayesian",
        help="Choose between 'cmaes' or 'bayesian' optimization (default: bayesian).",
    )
    parser.add_argument(
        "--latent",
        nargs=4,
        type=float,
        default=None,
        help="Specify z values (z1, z2, z3, z4) for 'solve' mode.",
    )
    parser.add_argument(
        "--symmetry",
        action="store_true",
        help="Enable left-right symmetry in the domain.",
    )
    parser.add_argument("--blank", action="store_true", help="Run with a blank image.")
    parser.add_argument(
        "--sources",
        nargs="*",
        type=float,
        default=None,
        help="List of source positions and heat source values as pairs, e.g., --sources 0.5 80 0.75 40",
    )
    parser.add_argument(
        "--res",
        type=float,
        default=None,
        help="Set the mesh resolution (default: LENGTH / 12).",
    )
    parser.add_argument(
        "--visualize",
        nargs="*",
        type=str,
        choices=["gamma", "temperature", "flux", "profiles", "pregamma"],
        help=(
            "Specify what to visualize. Options: "
            "'none' (no visualization), 'gamma', 'temperature', 'flux', 'profiles', 'all' "
            "(default: all). Multiple options can be specified."
        ),
    )
    parser.add_argument(
        "--vf",
        type=float,
        default=0.2,
        help=(
            "Set the desired volume fraction (default: 0.2)."
            "Negative means no volume fraction control."
        ),
    )
    parser.add_argument(
        "--plot-mode",
        choices=["screenshot", "interactive"],
        default="screenshot",
        help="Choose between 'screenshot' (save plots) or 'interactive' (display plots).",
    )
    parser.add_argument(
        "--latent-size",
        type=int,
        choices=[2, 4, 8, 16],
        default=4,
        help="Size of the latent vector (2, 4, 8, or 16). Default is 4.",
    )
    parser.add_argument(
        "--latent-method",
        choices=["manual", "random", "preloaded"],
        default="manual",
        help="How to obtain the latent vector: 'manual', 'random', or 'preloaded'. Default is 'manual'.",
    )
    parser.add_argument(
        "--no-logging",
        action="store_true",
        help="Disable logging.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    config = SimulationConfig(args)
    controller = SimulationController(config)
    controller.run_simulation()
