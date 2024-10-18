from bayes_opt import BayesianOptimization
import PIL
from PIL import Image
from solver_module import Solver
import numpy as np
from vae_module import z_to_img


class OptimizationModule:
    def __init__(self, solver, model, device, rank):
        self.model = model
        self.device = device
        self.rank = rank
        # Initialize the solver
        self.solver = solver

    def evaluate(self, z1, z2, z3, z4):
        z = np.array([z1, z2, z3, z4]).reshape(1, 4)
        if self.rank == 0:
            sample = z_to_img(z, self.model, self.device)
            sample = sample[:, : sample.shape[1] // 2]  # Symmetrize
        else:
            sample = None
        Nx, Ny = 128, 128
        im = Image.fromarray(sample)
        new_image = np.array(im.resize((Nx, Ny), PIL.Image.BICUBIC))
        # Solve the image (assuming solve_image is defined elsewhere)
        obj = self.solver.solve_image(new_image)
        return 1 / obj

    def optimize(self, init_points=10, n_iter=60):
        pbounds = {"z1": (-1, 1), "z2": (-1, 1), "z3": (-1, 1), "z4": (-1, 1)}
        optimizer = BayesianOptimization(
            f=self.evaluate, pbounds=pbounds, random_state=1
        )
        optimizer.maximize(init_points=init_points, n_iter=n_iter)
        best_params = optimizer.max["params"]
        best_z = np.array(
            [
                best_params["z1"],
                best_params["z2"],
                best_params["z3"],
                best_params["z4"],
            ]
        )
        return best_z
