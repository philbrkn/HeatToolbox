
''' module docstring'''
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.termination import TerminateIfAny
from pymoo.optimize import minimize
from pymoo.termination.max_time import TimeBasedTermination
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.core.callback import Callback
from pymoo.core.population import Population
from pymoo.operators.mutation.pm import PolynomialMutation
# from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding


import torch
import numpy as np
import logging
import json
import os
import pickle
import copy
from heatoptim.utilities.image_processing import z_to_img
# from memory_profiler import profile
# from .utils.logging_utils import generate_log_name, initialize_logger_folder


class CloakProblem(Problem):
    '''setting up optimization problem for a bi-objective thermal cloak'''
    def __init__(self, settings):
        N_OBJ = len(self.z_dim * self.N_sources)
        # print(N_OBJ)
        super().__init__(n_var=self.ass.ngen,
                         n_obj=N_OBJ,  # how many objectives considering
                         n_constr=0,  # zero constraints being used
                         xl=-10,  # Lower bound for variables
                         xu=10)  # Upper bound for variables

    def _evaluate(self, x, out, *args, **kwargs):
        '''parameters: x is the population filled with genes
        '''
        losses = []
        for i, gene in enumerate(x):
            # Convert to torch and reshape
            gene_torch = torch.from_numpy(np.array(gene)).cuda()
            gene_torch = torch.reshape(gene_torch, (1, -1))

            # Calculate objective
            loss = self.ass.sample_eval(gene_torch)

            # Append loss to list
            losses.append(loss)

            log_entry = {
                "subject": i,
                "loss": loss
            }
            logging.info(json.dumps(log_entry))
            # Decode the latent vector to an image only in the root process
            img_list = []
            for z in gene_torch:
                img = z_to_img(z.reshape(1, -1), self.model, self.config.vol_fraction)
                # Apply symmetry if enabled
                if self.config.symmetry:
                    img = img[:, : img.shape[1] // 2]
                img_list.append(img)
            # Solve the problem using the solver with the generated image
            fitness = self.solver.solve_image(img_list)

        out["F"] = np.array(losses)

    def __deepcopy__(self, memo):
        # Create a shallow copy
        cls = self.__class__
        new_instance = cls.__new__(cls)
        memo[id(self)] = new_instance

        # Copy other attributes, but skip `self.ass`
        for k, v in self.__dict__.items():
            if k != "ass":
                setattr(new_instance, k, copy.deepcopy(v, memo))

        return new_instance

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
        obj = CloakProblem.__new__(CloakProblem)  # Create a new instance
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
    def update(self, algorithm):
        super().update(algorithm)

        # Log the data to a file
        log_entry = {
            "generation": algorithm.n_gen,
            "n_non_dom": len(algorithm.opt),
            "eps": self.eps.value,
            "indicator": self.indicator.value,
        }
        logging.info(json.dumps(log_entry))


class MyCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.n_evals = []
        self.opt = []

    def notify(self, algorithm):
        # Append row of "F" with best sum of objectives
        self.n_evals.append(algorithm.evaluator.n_eval)
        self.opt.append(algorithm.opt.get('F'))


class CustomNSGA2(NSGA2):

    def _advance(self, infills=None, **kwargs):
        '''calls this instead of _advance in GeneticAlgorithm class
        Normalize objective values before survival and setting pareto front
        '''
        # the current population
        pop = self.pop

        # merge the offsprings with the current population
        if infills is not None:
            pop = Population.merge(self.pop, infills)

        # This part is new
        # Get the ideal and nadir points from the Pareto front
        pf = self.opt.get('F')
        nadir = np.max(pf, axis=0)
        ideal = np.min(pf, axis=0)

        # Log ideal and nadir
        log_entry = {
            "ideal": list(ideal),
            "nadir": list(nadir)
        }
        logging.info(json.dumps(log_entry))

        # Normalize the objective values of the merged population
        F = pop.get('F')
        F_normalized = (F - ideal) / (nadir - ideal)
        pop.set('F', F_normalized)
        # self.pop = ?
        # end new part

        # execute the survival to find the fittest solutions
        self.pop = self.survival.do(self.problem, pop, n_survive=self.pop_size, algorithm=self, **kwargs)


def optim_main(config, log_fold_path=None):
    '''main function for optimization'''

    # Set up problem and algorithm
    problem = CloakProblem(config)

    algorithm = NSGA2(
        pop_size=config['genetic']['nsubjects'],
        n_offsprings=config['genetic']['noffsprings'],
        # sampling=FloatRandomSampling(),
        # crossover=SBX(),
        mutation=PolynomialMutation(prob=0.95, eta=10),
        survival=RankAndCrowding(crowding_func="ce"),
        eliminate_duplicates=True,
        output=CustomOutput()
    )

    # RSNGA2
    # from pymoo.algorithms.moo.rnsga2 import RNSGA2
    # # Define reference points
    # ref_points = np.array([[0.5, 0.5, 0.5]])

    # # Get Algorithm
    # algorithm = RNSGA2(
    #     ref_points=ref_points,
    #     pop_size=100, #config['genetic']['nsubjects'],
    #     epsilon=0.01,
    #     normalization='front',
    #     extreme_points_as_reference_points=False,
    #     weights=np.array([0.5, 0.5, 0.5]),
    #     output=CustomOutput()
    # )

    # Termination criterion
    TimeTerm = TimeBasedTermination(config['genetic']['maxtime'])
    DefaultTerm = DefaultMultiObjectiveTermination(n_max_evals=config['genetic']['niterations'])
    termination = TerminateIfAny(DefaultTerm, TimeTerm)

    print("Starting minimization...")
    res = minimize(problem,
                   algorithm,
                   termination=termination,
                   callback=MyCallback(),
                   save_history=False,
                   verbose=True)

    print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

    # For post-processing
    print("Copying to pickle object...")
    with open(os.path.join(log_fold_path, "ResultObject.pk1"), "wb") as f:
        pickle.dump(res, f)

    # End of main
    print("Optim over.")
