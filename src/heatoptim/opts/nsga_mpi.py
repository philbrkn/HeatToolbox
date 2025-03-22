# nsga.py

import os
import pickle

import numpy as np
from mpi4py import MPI

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination.max_time import TimeBasedTermination
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.core.termination import TerminateIfAny
from pymoo.core.problem import Problem
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.core.callback import Callback

from heatoptim.utilities.image_processing import generate_images


class MyCallback(Callback):
    def __init__(self):
        super().__init__()
        self.n_evals = []
        self.opt = []
        self.rank = MPI.COMM_WORLD.Get_rank()

    def notify(self, algorithm):
        if self.rank == 0:
            # only rank 0 tracks these
            self.n_evals.append(algorithm.evaluator.n_eval)
            self.opt.append(algorithm.opt.get("F"))


class NSGAProblem(Problem):
    """
    NSGA2 problem definition for optimizing the latent vectors.
    (Multi-objective: 2 objectives: [avg_temp, std_dev])
    This class is IDENTICAL in structure to your original NSGA code,
    but now does the exact same MPI partitioning logic as CMA-ES.
    """

    def __init__(self, solver, model, config):
        self.solver = solver
        self.model = model
        self.config = config

        # For direct MPI usage
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.N_sources = len(config.source_positions)
        self.z_dim = config.latent_size
        # Each solution is a flattened vector of length (z_dim * N_sources)
        n_var = self.z_dim * self.N_sources

        # Suppose we track 2 objectives: [avg_temp, get_std_dev()]
        super().__init__(
            n_var=n_var,
            n_obj=2,
            n_constr=0,
            xl=config.bounds[0],
            xu=config.bounds[1],
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """
        EXACT MPI logic from your CMA-ES code:

         1) rank 0 has the entire population x.
         2) broadcast x to all other ranks.
         3) each rank takes a slice of x (local_x), evaluates it,
         4) gather partial results to rank 0,
         5) rank 0 flattens them into out["F"],
         6) broadcast out["F"] if needed so all have consistent data.
        """
        # On rank!=0, set x = None so we can broadcast from rank=0
        if self.rank != 0:
            x = None

        # Broadcast the population to all ranks
        x = self.comm.bcast(x, root=0)

        # Partition the candidate set among ranks
        num_candidates = len(x)  # x.shape[0]
        counts = [num_candidates // self.size] * self.size
        for i in range(num_candidates % self.size):
            counts[i] += 1
        offsets = np.cumsum([0] + counts[:-1])

        start = offsets[self.rank]
        end = offsets[self.rank] + counts[self.rank]

        # Each rank evaluates its local portion
        local_x = x[start:end]
        local_losses = []
        for gene in local_x:
            # Split gene into latent vectors for each source
            latent_vectors = [
                gene[i * self.z_dim : (i + 1) * self.z_dim]
                for i in range(self.N_sources)
            ]
            img_list = generate_images(self.config, latent_vectors, self.model)

            # Evaluate with your solver
            avg_temp = self.solver.solve_image(img_list)
            std_dev = self.solver.get_std_dev()
            local_losses.append([avg_temp, std_dev])

        # Gather local results on rank=0
        all_losses = self.comm.gather(local_losses, root=0)

        # If rank=0, flatten everything into out["F"]
        if self.rank == 0:
            merged_losses = [loss for sublist in all_losses for loss in sublist]
            out["F"] = np.array(merged_losses)
        else:
            # Non-root must define out["F"] with matching shape, though it won’t be used
            out["F"] = np.zeros((num_candidates, self.n_obj))

        # Optionally broadcast out["F"] so all ranks see the final array
        out["F"] = self.comm.bcast(out["F"], root=0)

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
        # Not strictly necessary, but if you only want rank=0 logging:
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.last_time = None

    def update(self, algorithm):
        super().update(algorithm)
        if self.logger and self.rank == 0:
            gen_time = 0
            if self.last_time is None:
                self.last_time = MPI.Wtime()
            else:
                now = MPI.Wtime()
                gen_time = now - self.last_time
                self.last_time = now
                # print(f"Generation {algorithm.n_gen} time: {gen_time:.3f}s")
            log_entry = {
                "n_non_dom": len(algorithm.opt),
                "eps": self.eps.value,
                "indicator": self.indicator.value,
                "gen_time": gen_time,
            }
            self.logger.log_generation_data(algorithm.n_gen, log_entry)


def optimize_nsga(solver, model, config, logger=None):
    """
    EXACT same "structure" as before:
     1) create NSGAProblem
     2) create NSGA2
     3) run pymoo's minimize
     The difference is that _evaluate(...) now
     does the same MPI partition/gather logic as your CMA-ES.
    """
    problem = NSGAProblem(solver, model, config)

    algorithm = NSGA2(
        pop_size=config.popsize,
        output=CustomOutput(logger)
    )

    time_term = TimeBasedTermination(config.maxtime)
    default_term = DefaultMultiObjectiveTermination(n_max_evals=config.n_iter)
    termination = TerminateIfAny(default_term, time_term)

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

    # rank 0: save results
    if MPI.COMM_WORLD.Get_rank() == 0:
        result_path = os.path.join(config.log_dir, "NSGA_Result.pk1")
        with open(result_path, "wb") as f:
            pickle.dump(res, f)

    return res.X
