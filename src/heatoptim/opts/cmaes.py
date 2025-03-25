# optimization_module.py

# from bayes_opt import BayesianOptimization
# import PIL
# from PIL import Image
# from solver_module import Solver
import numpy as np
from heatoptim.utilities.image_processing import z_to_img, generate_images

# external optimizers:
import cma
from mpi4py import MPI
import os
from heatoptim.postprocessing.post_processing_gke import PostProcessingGKE


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

        # Directory for CMA logs
        self.cma_log_dir = os.path.join(self.logger.log_dir, "cma_logs") if self.logger else "cma_logs"
        if self.rank == 0:
            os.makedirs(self.cma_log_dir, exist_ok=True)

        timeout_parts = list(map(int, config.walltime.split(":")))
        self.timeout = (
            timeout_parts[0]*3600 + timeout_parts[1] * 60 + timeout_parts[2]
        )
        self.timeout *= 0.98  # Use 98% of walltime as timeout buffer

    def evaluate_candidate(self, latent_vectors):
        """
        Evaluate a candidate solution.

        Parameters:
        - z: The latent vector representing the candidate solution.

        Returns:
        - fitness: The fitness value of the candidate (e.g., average temperature).
        """
        # Decode the latent vector to an image only in the root process
        img_list = generate_images(self.config, latent_vectors, self.model)

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
            on_hpc = self.config.hpc_enabled
            cma_options = {
                'verb_filenameprefix': os.path.join(self.cma_log_dir, "outcma_"),
                'verb_disp': 0 if on_hpc else 1,  # 100 #v verbosity: display console output every verb_disp iteration
                'verb_log': 1 if on_hpc else 0,  # verbosity: write data to files every verb_log iteration
                'verb_append': 1,  # initial evaluation counter, if append, do not overwrite output files
                'timeout': self.timeout,  # Stop after timeout seconds
                'popsize': self.config.popsize,  # Population size
                'bounds': self.config.bounds,  # Bounds
            }
            es = cma.CMAEvolutionStrategy(self.init_z, self.sigma0, cma_options)
        else:
            es = None

        for generation in range(self.config.n_iter):
            start_time = MPI.Wtime()
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
                    x[i * self.z_dim : (i + 1) * self.z_dim]
                    for i in range(self.N_sources)
                ]  # Split latent vectors per source
                fitness = self.evaluate_candidate(latent_vectors)
                local_fitnesses.append(fitness)

            all_fitness = self.comm.gather(local_fitnesses, root=0)

            if self.rank == 0:
                fitnesses = [fitness for sublist in all_fitness for fitness in sublist]
                # Tell CMA-ES the fitnesses of the candidates
                es.tell(candidate_solutions, fitnesses)
                # Log results for this generation
                es.logger.add()  # Add generation data to the log files

                # Pair each candidate solution with its fitness
                results = list(zip(candidate_solutions, fitnesses))

                # Find the candidate with the minimum fitness
                best_solution, min_fitness = min(results, key=lambda s: s[1])
                generation_time = MPI.Wtime() - start_time  # Measure time for the generation

                # Logging and displaying progress
                es.disp()
                if self.logger:
                    self.logger.log_generation_data(
                        generation, {
                            "best_value": min(fitnesses),
                            "generation_time": generation_time,
                            "population_size": len(candidate_solutions),
                        }
                    )

        # After optim, get the best solution found
        if self.rank == 0:
            result = es.result
            best_z = result.xbest  # Best latent vector
            self.save_cma_plots()  # Save CMA-ES plots
        else:
            best_z = None

        best_z = self.comm.bcast(best_z, root=0)

        return best_z

    def save_cma_plots(self):
        """
        Generate and save plots from the CMA-ES optimization.

        Parameters:
        - es: The CMAEvolutionStrategy instance.
        """
        import matplotlib
        matplotlib.use('Agg')  # Ensure we're using a non-interactive backend
        import matplotlib.pyplot as plt

        # Plot the data using cma's logger
        cma.plot(os.path.join(self.cma_log_dir, "outcma_"))

        # Save the figure
        plot_path = os.path.join(self.cma_log_dir, "cmaes_convergence.png")
        cma.s.figsave(plot_path)  # Save all current figures to a file

        print(f"CMA-ES convergence plot saved to {plot_path}")

