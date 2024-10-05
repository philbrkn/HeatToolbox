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
        self.C = PETSc.ScalarType(1.0)  # Slip parameter for fully diffusive boundaries
        self.T_ISO = PETSc.ScalarType(0.0)  # Isothermal temperature, K
        self.Q_L = 80
        self.Q = PETSc.ScalarType(self.Q_L)

        self.MEAN_FREE_PATH = 0.439e-6  # Characteristic length, adjust as necessary
        self.KNUDSEN = 1  # Knudsen number, adjust as necessary

        # geometric properties
        self.LENGTH = self.MEAN_FREE_PATH / self.KNUDSEN  # Characteristic length, L
        self.L_X = 25 * self.LENGTH
        self.L_Y = 12.5 * self.LENGTH
        self.SOURCE_WIDTH = self.LENGTH
        self.SOURCE_HEIGHT = (self.LENGTH * 0.25)
        # self.RESOLUTION = self.LENGTH / 20  # to get good profiles
        self.RESOLUTION = self.LENGTH / 10  # fast but ish profiles
        self.mask_extrusion = True

        # material properties
        self.ELL_SI = PETSc.ScalarType(self.MEAN_FREE_PATH / np.sqrt(5))  # Non-local length, m
        self.ELL_DI = PETSc.ScalarType(196e-8)
        self.KAPPA_SI = PETSc.ScalarType(141.0)  # W/mK, thermal conductivity
        self.KAPPA_DI = PETSc.ScalarType(600.0)

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
    parser.add_argument("--latent", nargs=4, type=float, default=None,
                        help="Specify latent vector values (z1, z2, z3, z4) for 'solve' mode.")
    parser.add_argument("--symmetry", action="store_true", help="Enable symmetry in the domain.")
    parser.add_argument("--blank", action="store_true", help="Run with a blank image.")
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
        optimizer = OptimizationModule(solver, model, torch.device('cpu'), rank)
        best_z = optimizer.optimize(init_points=10, n_iter=60)
        latent_vector = best_z
        # Optional: Save the best_z to a file for future solving
        if rank == 0:
            np.save("best_latent_vector.npy", best_z)

    else:
        # Run solving based on provided latent vector
        if args.latent is None:
            # If latent vector is not provided, load the best saved latent vector
            if rank == 0:
                latent_vector = np.load("best_latent_vector.npy")
                # latent_vector =
                # np.array([[0.91578013,  0.06633388,  0.3837567,  -0.36896428]])
            else:
                latent_vector = None
            latent_vector = MPI.COMM_WORLD.bcast(latent_vector, root=0)
        else:
            latent_vector = np.array(args.latent)

    # Generate image from latent vector
    if rank == 0:
        if args.blank:
            sample = np.zeros((128, 128))
        else:
            sample = z_to_img(latent_vector, model)
        sample = sample[:, :sample.shape[1] // 2]  # Take the left half of the image
        # resymmetrize the image if symmetry is not enabled
        if not config.symmetry:
            sample = np.concatenate((sample, sample[:, ::-1]), axis=1)
    else:
        sample = None

    sample = MPI.COMM_WORLD.bcast(sample, root=0)

    # Solve the image using the solver
    avg_temp_global = solver.solve_image(sample)
    time2 = MPI.Wtime()
    print(f"Average temperature: {avg_temp_global:.4f} K")
    print(f"Time taken to solve: {time2 - time1:.3f} seconds")

    # Optional Post-processing
    post_processor = PostProcessingModule(rank, config)
    post_processor.postprocess_results(solver.U, solver.msh, sample, solver.gamma)


if __name__ == "__main__":
    main()
