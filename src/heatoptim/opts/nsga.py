# nsgamodule.py

from pymoo.core.problem import Problem
import numpy as np
from heatoptim.utilities.image_processing import generate_images  # using your generate_images function

# nsga_optimization.py

import os
import pickle
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination.max_time import TimeBasedTermination
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.core.termination import TerminateIfAny
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.core.callback import Callback


class MyCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.n_evals = []
        self.opt = []

    def notify(self, algorithm):
        # Append row of "F" with best sum of objectives
        self.n_evals.append(algorithm.evaluator.n_eval)
        self.opt.append(algorithm.opt.get("F"))


class NSGAProblem(Problem):
    """
    NSGA2 problem definition for optimizing the latent vectors.
    This problem class mimics the CMA-ES evaluation:
    each candidate (a flattened latent vector) is split into per-source latent vectors,
    decoded into images, and then evaluated with the solver.

    Note: This implementation assumes a single objective (e.g., minimizing average temperature).
    If you have multiple objectives, adjust n_obj accordingly.
    """

    def __init__(self, solver, model, config):
        self.solver = solver
        self.model = model
        self.config = config
        self.N_sources = len(config.source_positions)
        self.z_dim = config.latent_size

        # The decision variable is a flattened vector of all latent vectors (one per source)
        n_var = self.z_dim * self.N_sources
        # For a single objective optimization set n_obj=1.
        # Adjust n_obj if you decide to add additional objectives.
        super().__init__(
            n_var=n_var, n_obj=2, n_constr=0, xl=config.bounds[0], xu=config.bounds[1]
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate each candidate solution:
         - Split the candidate into individual latent vectors.
         - Generate images from these latent vectors.
         - Use the solver to evaluate the candidate (e.g., average temperature).
        """
        losses = []
        # x is of shape (n_samples, n_var)
        for gene in x:
            # Split gene into latent vectors for each source
            latent_vectors = [
                gene[i * self.z_dim: (i + 1) * self.z_dim]
                for i in range(self.N_sources)
            ]
            # Generate images using your provided function
            img_list = generate_images(self.config, latent_vectors, self.model)
            # Evaluate candidate using the solver (this is analogous to the CMA-ES evaluation)
            loss = [self.solver.solve_image(img_list), self.solver.get_std_dev()]
            losses.append(loss)
            print(f"Loss: {loss}")
        # Make sure F has shape (n_samples, n_obj)
        out["F"] = np.array(losses)

    # Modify pickling behavior
    def __reduce__(self):
        # Return a callable (usually a class or function) and a tuple of
        # arguments to pass to the callable. The callable is used to recreate
        # the object when deserializing. The '_recreate' method can be a
        # @staticmethod or another external function.
        return (self._recreate, (self.n_var, self.n_obj, self.n_constr,
                                 self.xl, self.xu))

    @staticmethod
    def _recreate(n_var, n_obj, n_constr, xl, xu):
        # This method will be called when deserializing.
        obj = NSGAProblem.__new__(NSGAProblem)  # Create a new instance
        # obj.n_var = n_var
        # obj.n_obj = n_obj
        # obj.n_constr = n_constr
        # obj.xl = xl
        # obj.xu = xu
        # NOTE: self.ass is not set here. If you need to reinitialize it after
        # deserialization, you should do it outside of the pickling process or
        # add additional logic.
        return obj


class CustomOutput(MultiObjectiveOutput):
    # Redo the initialization to include the logger
    def __init__(self, logger=None):
        super().__init__()
        self.logger = logger

    def update(self, algorithm):
        super().update(algorithm)
        if self.logger:
            # Log the data to a file
            log_entry = {
                "n_non_dom": len(algorithm.opt),
                "eps": self.eps.value,
                "indicator": self.indicator.value,
            }
            self.logger.log_generation_data(algorithm.n_gen, log_entry)


def optimize_nsga(solver, model, config, logger=None):
    """
    Run NSGA2 optimization using the NSGAProblem.
    """
    # Create the problem instance
    problem = NSGAProblem(solver, model, config)

    # Set up NSGA2 algorithm.
    # You can add your custom mutation/crossover/survival operators if needed.
    algorithm = NSGA2(
        pop_size=config.popsize,
        output=CustomOutput(logger)
        )

    # Termination criteria: use max evaluations and/or time-based termination
    time_term = TimeBasedTermination(config.maxtime)
    default_term = DefaultMultiObjectiveTermination(n_max_evals=config.n_iter)
    termination = TerminateIfAny(default_term, time_term)

    # Run the minimization (or maximization if you change the sign of your fitness)
    res = minimize(
        problem,
        algorithm,
        termination=termination,
        callback=MyCallback(),
        save_history=False,
        verbose=True,
    )

    print("Best solution found:")
    print("X =", res.X)
    print("F =", res.F)

    # Save results if desired
    result_path = os.path.join(config.log_dir, "NSGA_Result.pk1")
    with open(result_path, "wb") as f:
        pickle.dump(res, f)

    return res.X
