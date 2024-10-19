# main.py

# import libraries
import torch
from mpi4py import MPI
import argparse
import numpy as np
from petsc4py import PETSc

# import modules
# from logging_module import LoggingModule
from mesh_generator import MeshGenerator
from vae_module import load_vae_model, z_to_img, VAE, Flatten, UnFlatten
from optimization_module import OptimizationModule
from post_processing import PostProcessingModule
from solver_module import Solver


class SimulationConfig:
    def __init__(self, args):
        # Physical properties
        self.C = PETSc.ScalarType(1.0)  # Slip parameter for fully diffusive boundaries
        self.T_ISO = PETSc.ScalarType(0.0)  # Isothermal temperature, K
        self.Q_L = PETSc.ScalarType(80)  # Line heat source, W/m

        self.MEAN_FREE_PATH = 0.439e-6  # Characteristic length, adjust as necessary
        self.KNUDSEN = 1  # Knudsen number, adjust as necessary

        # Geometric properties
        self.LENGTH = self.MEAN_FREE_PATH / self.KNUDSEN  # Characteristic length, L
        # Base rectangle:
        self.L_X = 25 * self.LENGTH
        self.L_Y = 12.5 * self.LENGTH
        # Source rectangle:
        self.SOURCE_WIDTH = self.LENGTH
        self.SOURCE_HEIGHT = self.LENGTH * 0.25
        self.mask_extrusion = True

        # Set the resolution of the mesh
        # self.RESOLUTION = self.LENGTH / 30  # to get REALLY good profiles
        # self.RESOLUTION = self.LENGTH / 20  # to get good profiles
        self.RESOLUTION = self.LENGTH / 5  # fast but ish profiles

        # material properties
        self.ELL_SI = PETSc.ScalarType(
            self.MEAN_FREE_PATH / np.sqrt(5)
        )  # Non-local length, m
        self.ELL_DI = PETSc.ScalarType(196e-8)
        self.KAPPA_SI = PETSc.ScalarType(141.0)  # W/mK, thermal conductivity
        self.KAPPA_DI = PETSc.ScalarType(600.0)

        if args.sources is not None:
            if len(args.sources) % 2 != 0:
                raise ValueError(
                    "Each source must have a position and a heat value. Please provide pairs of values."
                )
            # Group the list into pairs
            sources_pairs = [
                (args.sources[i], args.sources[i + 1]) for i in range(0, len(args.sources), 2)
            ]
            self.source_positions = []
            self.Q_sources = []
            for pos, Q in sources_pairs:
                if pos < 0 or pos > 1:
                    raise ValueError("Source positions must be between 0 and 1 (normalized).")
                self.source_positions.append(pos)
                self.Q_sources.append(PETSc.ScalarType(Q))
        else:
            # Default to a single source at the center
            self.source_positions = [0.5]
            self.Q_sources = [PETSc.ScalarType(self.Q_L)]

        # Cannot be symmetry and two sources:
        if len(self.source_positions) > 1 and args.symmetry:
            raise ValueError("Cannot have both symmetry and two sources.")

        # symmetry in geometry
        if args.symmetry:
            self.L_X = self.L_X / 2
            self.SOURCE_WIDTH = self.SOURCE_WIDTH / 2
            self.symmetry = True  # Enable or disable symmetry
        else:
            self.symmetry = False

        if args.blank:
            self.mask_extrusion = False


def main():
    # Command-line arguments to determine the mode
    parser = argparse.ArgumentParser()
    parser.add_argument("--optim", action="store_true", help="Run optimization.")
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
    # parser.add_argument("--gamma_only", action="store_true", help="Show gamma only.")
    args = parser.parse_args()

    # Initialize configuration
    config = SimulationConfig(args)

    # Load VAE model
    rank = MPI.COMM_WORLD.rank
    model = load_vae_model(rank)

    # Mesh generation
    mesh_generator = MeshGenerator(config)
    msh, cell_markers, facet_markers = mesh_generator.create_mesh()

    time1 = MPI.Wtime()

    # Create solver instance
    solver = Solver(msh, facet_markers, config)

    if args.optim:
        # Run optimization
        optimizer = OptimizationModule(solver, model, torch.device("cpu"), rank)
        best_z = optimizer.optimize(init_points=10, n_iter=60)
        latent_vector = best_z
        # Optional: Save the best_z to a file for future solving
        if rank == 0:
            np.save("best_latent_vector.npy", best_z)

    # Run solving based on provided latent vector
    else:
        if args.latent is None:
            if rank == 0:
                # latent_vector = np.load("best_latent_vector.npy")
                # latent_vector = np.random.randn(4)
                # latent_vector =
                # np.array([[0.91578013,  0.06633388,  0.3837567,  -0.36896428]])
                latent_vector = np.array([0.8019, 1.0, -0.5918, 0.4979])
            else:
                latent_vector = None
            latent_vector = MPI.COMM_WORLD.bcast(latent_vector, root=0)
        else:
            latent_vector = np.array(args.latent)

    # Generate image from latent vector
    if rank == 0:
        img_list = []
        for idx in range(len(config.source_positions)):
            if args.blank:
                img = np.zeros((128, 128))
            else:
                # for now same latent vector for all sources
                img = z_to_img(latent_vector, model)

                # PLOT IMAGE TO TEST #
                # import matplotlib.pyplot as plt
                # plt.imshow(img)
                # plt.show()

            img_list.append(img)
            if config.symmetry:
                # Take the left half of the image
                img_list[0] = img_list[0][:, : img_list[0].shape[1] // 2]
    else:
        img_list = None

    img_list = MPI.COMM_WORLD.bcast(img_list, root=0)

    # Solve the image using the solver
    avg_temp_global = solver.solve_image(img_list)
    time2 = MPI.Wtime()
    if rank == 0:
        print(f"Average temperature: {avg_temp_global:.4f} K")
        print(f"Time taken to solve: {time2 - time1:.3f} seconds")

    # Optional Post-processing
    post_processor = PostProcessingModule(rank, config)
    post_processor.postprocess_results(solver.U, solver.msh, solver.gamma)


if __name__ == "__main__":
    main()
