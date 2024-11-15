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
from mpi4py import MPI


class CMAESModule:
    def __init__(self, solver, model, device, rank, config, logger=None):
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
        self.rank = rank

        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()  # Total number of MPI processes

        self.N_sources = len(config.source_positions)
        self.z_dim = config.latent_size
        # Initialize CMA-ES parameters
        self.init_z = np.zeros(self.z_dim * self.N_sources)  # Initial latent vector
        self.sigma0 = 0.5  # Initial standard deviation

    def evaluate_candidate(self, latent_vectors):
        """
        Evaluate a candidate solution.

        Parameters:
        - z: The latent vector representing the candidate solution.

        Returns:
        - fitness: The fitness value of the candidate (e.g., average temperature).
        """
        # Decode the latent vector to an image only in the root process
        img_list = []
        for z in latent_vectors:
            img = z_to_img(z.reshape(1, -1), self.model, self.config.vol_fraction)
            # Apply symmetry if enabled
            if self.config.symmetry:
                img = img[:, : img.shape[1] // 2]
            img_list.append(img)
        # Solve the problem using the solver with the generated image
        fitness = self.solver.solve_image(img_list)

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
        if self.rank == 0:
            es = cma.CMAEvolutionStrategy(self.init_z, self.sigma0)
        else:
            es = None
        
        for generation in range(n_iter):
            if self.rank == 0:
                # Ask for candidate solutions
                candidate_solutions = es.ask()
            else:
                candidate_solutions = None

            candidate_solutions = self.comm.bcast(candidate_solutions, root=0)

            # Determine the workload for each process
            num_candidates = len(candidate_solutions)
            counts = [num_candidates // self.size] * self.size
            for i in range(num_candidates % self.size):
                counts[i] += 1
            offsets = np.cumsum([0] + counts[:-1])

            # Each process selects its subset of candidate solutions
            start = offsets[self.rank]
            end = offsets[self.rank] + counts[self.rank]
            local_candidates = candidate_solutions[start:end]

            # Each process evaluates its local candidates
            local_fitnesses = []

            for x in local_candidates:
                # Split latent vectors per source
                latent_vectors = [
                    x[i * self.z_dim: (i + 1) * self.z_dim] for i in range(self.N_sources)
                ]  # Split latent vectors per source
                fitness = self.evaluate_candidate(latent_vectors)
                local_fitnesses.append(fitness)

            all_fitness = self.comm.gather(local_fitnesses, root=0)

            if self.rank == 0:
                fitnesses = [fitness for sublist in all_fitness for fitness in sublist]
                # Tell CMA-ES the fitnesses of the candidates
                es.tell(candidate_solutions, fitnesses)

                # Pair each candidate solution with its fitness
                results = list(zip(candidate_solutions, fitnesses))

                # Find the candidate with the minimum fitness
                best_solution, min_fitness = min(results, key=lambda s: s[1])

                # Prepare generation data for logging
                generation_data = {
                    "best_value": min_fitness,
                    "best_solution": best_solution.tolist(),  # Convert numpy array to list
                }

                # Logging and displaying progress
                es.disp()
                if self.logger:
                    self.logger.log_generation_data(generation, generation_data)

        # After optim, get the best solution found
        if self.rank == 0:
            result = es.result
            best_z = result.xbest  # Best latent vector
            best_fitness = result.fbest  # Best fitness value

            if self.logger:
                self.logger.log_results({
                    'best_z': best_z.tolist(),
                    'best_fitness': best_fitness
                })
        else:
            best_z = None

        best_z = self.comm.bcast(best_z, root=0)

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
