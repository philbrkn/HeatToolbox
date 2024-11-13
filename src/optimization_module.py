# optimization_module.py

# from bayes_opt import BayesianOptimization
# import PIL
# from PIL import Image
# from solver_module import Solver
import numpy as np
from image_processing import z_to_img

# external optimizers:
# implement: only import if you want to use them
try:
    from bayes_opt import BayesianOptimization
except ImportError:
    BayesianOptimization = None
import cma


class CMAESModule:
    def __init__(self, solver, model, device, config, logger=None):
        """
        Initialize the CMAESModule.

        Parameters:
        - solver: An instance of your solver class that can evaluate a solution.
        - model: The VAE model used for decoding latent vectors to images.
        - device: The device (CPU or GPU) to run computations on.
        - config: The simulation configuration containing settings and parameters.
        - logger: Optional logger instance for logging optimization progress.
        """
        self.solver = solver
        self.model = model
        self.device = device
        self.config = config
        self.logger = logger

        # Initialize CMA-ES parameters
        self.init_z = np.zeros(self.config.latent_size)  # Initial latent vector
        self.sigma0 = 0.5  # Initial standard deviation

    def evaluate_candidate(self, z):
        """
        Evaluate a candidate solution.

        Parameters:
        - z: The latent vector representing the candidate solution.

        Returns:
        - fitness: The fitness value of the candidate (e.g., average temperature).
        """
        # Decode the latent vector to an image
        if self.config.blank:
            img = np.zeros((128, 128))
        else:
            img = self.model.decode(torch.tensor(z, dtype=torch.float32).unsqueeze(0)).cpu().detach().numpy()[0, 0]

        # Solve the problem using the solver with the generated image
        fitness = self.solver.solve_image([img])

        return fitness

    def optimize(self, n_iter=100):
        """
        Run the CMA-ES optimization.

        Parameters:
        - n_iter: Number of iterations to run the optimization.

        Returns:
        - best_z: The best latent vector found during optimization.
        """
        # Initialize CMA-ES optimizer
        es = cma.CMAEvolutionStrategy(self.init_z, self.sigma0)

        for generation in range(n_iter):
            # Ask for candidate solutions
            candidate_solutions = es.ask()

            # Evaluate each candidate solution
            fitnesses = []
            for z in candidate_solutions:
                fitness = self.evaluate_candidate(z)
                fitnesses.append(fitness)

            # Tell CMA-ES the fitnesses of the candidates
            es.tell(candidate_solutions, fitnesses)

            # Logging and displaying progress
            es.disp()
            if self.logger:
                self.logger.log_generation_data(generation, {'fitnesses': fitnesses})

        # Get the best solution found
        result = es.result
        best_z = result.xbest  # Best latent vector
        best_fitness = result.fbest  # Best fitness value

        if self.logger:
            self.logger.log_results({
                'best_z': best_z.tolist(),
                'best_fitness': best_fitness
            })

        return best_z


class BayesianModule:
    def __init__(self, solver, model, device, rank, config, logger=None):
        self.logger = logger
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
            z = np.array(
                [
                    kwargs[f"z{i}_1"],
                    kwargs[f"z{i}_2"],
                    kwargs[f"z{i}_3"],
                    kwargs[f"z{i}_4"],
                ]
            ).reshape(1, 4)
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
            pbounds.update(
                {
                    f"z{i}_1": (-2.5, 2.5),
                    f"z{i}_2": (-2.5, 2.5),
                    f"z{i}_3": (-2.5, 2.5),
                    f"z{i}_4": (-2.5, 2.5),
                }
            )

        optimizer = BayesianOptimization(
            f=self.evaluate, pbounds=pbounds, random_state=1
        )
        optimizer.maximize(init_points=init_points, n_iter=n_iter)

        # Log each iteration's results

        for i, res in enumerate(optimizer.res):
            generation_data = {
                "iteration": i,
                "target": res["target"],
                "params": res["params"],
            }
            if self.logger:
                self.logger.log_generation_data(i, generation_data)

        # Extract best latent vectors
        best_params = optimizer.max["params"]
        best_z_list = []
        for i in range(self.N_sources):
            best_z = np.array(
                [
                    best_params[f"z{i}_1"],
                    best_params[f"z{i}_2"],
                    best_params[f"z{i}_3"],
                    best_params[f"z{i}_4"],
                ]
            )
            best_z_list.append(best_z)

        return best_z_list
