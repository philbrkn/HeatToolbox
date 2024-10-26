from bayes_opt import BayesianOptimization
import PIL
from PIL import Image
from solver_module import Solver
import numpy as np
from image_processing import z_to_img


class OptimizationModule:
    def __init__(self, solver, model, device, rank, config):
        self.model = model
        self.device = device
        self.rank = rank
        # Initialize the solver
        self.solver = solver
        self.config = config
        self.N_sources = len(config.source_positions)

    def evaluate(self, **kwargs):
        # Extract latent variables for each source
        latent_vectors = []
        for i in range(self.N_sources):
            z = np.array([
                kwargs[f'z{i}_1'],
                kwargs[f'z{i}_2'],
                kwargs[f'z{i}_3'],
                kwargs[f'z{i}_4']
            ]).reshape(1, 4)
            latent_vectors.append(z)

        if self.rank == 0:
            img_list = []
            for z in latent_vectors:
                sample = z_to_img(z, self.model, self.config.vol_fraction)
                # If symmetry is enabled, take left half of the image
                if self.config.symmetry:
                    sample = sample[:, : sample.shape[1] // 2]
                img_list.append(sample)
        else:
            img_list = None

        # Solve the images
        obj = self.solver.solve_image(img_list)

        # Return the objective function value
        # Assuming you want to maximize 1 / obj
        return 1 / obj

    def optimize(self, init_points=10, n_iter=60):
        # Prepare bounds for all latent variables
        pbounds = {}
        for i in range(self.N_sources):
            pbounds.update({
                f'z{i}_1': (-2.5, 2.5),
                f'z{i}_2': (-2.5, 2.5),
                f'z{i}_3': (-2.5, 2.5),
                f'z{i}_4': (-2.5, 2.5)
            })

        optimizer = BayesianOptimization(
            f=self.evaluate, pbounds=pbounds, random_state=1
        )
        optimizer.maximize(init_points=init_points, n_iter=n_iter)

        # Extract best latent vectors
        best_params = optimizer.max["params"]
        best_z_list = []
        for i in range(self.N_sources):
            best_z = np.array([
                best_params[f'z{i}_1'],
                best_params[f'z{i}_2'],
                best_params[f'z{i}_3'],
                best_params[f'z{i}_4']
            ])
            best_z_list.append(best_z)

        return best_z_list
