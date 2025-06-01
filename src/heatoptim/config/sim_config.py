import json
import os
from petsc4py import PETSc
import numpy as np
from mpi4py import MPI


class SimulationConfig:
    def __init__(self, config, arg_res=None):
        """
        Initialize the simulation configuration.

        Parameters:
        - config (str or dict): Path to a JSON configuration file or a dictionary containing configuration data.
        """
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()  # Total number of MPI processes

        if isinstance(config, str):  # If a file path is provided
            if self.comm.rank == 0:
                print("Loading config from file on rank 0")
                with open(config, "r") as f:
                    loaded_config = json.load(f)
            else:
                loaded_config = None
            # Broadcast the config to all ranks
            loaded_config = self.comm.bcast(loaded_config, root=0)
            self.config = loaded_config
        elif isinstance(config, dict):  # If a dictionary is provided
            # If a dictionary was directly passed in, just broadcast that
            if self.comm.rank == 0:
                loaded_config = config
            else:
                loaded_config = None

            loaded_config = self.comm.bcast(loaded_config, root=0)
            self.config = loaded_config
        else:
            raise TypeError("Config must be either a file path (str) or a dictionary.")

        # Physical properties
        self.C = PETSc.ScalarType(1.0)  # Slip parameter
        self.T_ISO = PETSc.ScalarType(0.0)
        self.Q_L = PETSc.ScalarType(80)

        self.solver_type = self.config.get("solver_type", "gke")

        # Volume fraction
        self.vol_fraction = (
            self.config.get("vf_value", 0.2)
            if self.config.get("vf_enabled", True)
            else None
        )

        # Geometric properties
        # if solver type is fourier, set length manually to 1
        if self.solver_type == "fourier":
            self.LENGTH = 5e-3
        elif self.solver_type == "gke":
            # self.LENGTH = self.MEAN_FREE_PATH / self.KNUDSEN
            # self.LENGTH = 0.439e-6
            self.LENGTH = 0.5e-6  # 0.5 microns or 500 nm
            # self.LENGTH = 5e-3
            # self.LENGTH = 1
        self.LENGTH = self.config.get("length", self.LENGTH)  # Default to 0.5 microns if not set

        # AT 300K
        self.MEAN_FREE_PATH_SI = 0.439e-6  # 439 nm @ 300 K
        self.ELL_SI = PETSc.ScalarType(self.MEAN_FREE_PATH_SI / np.sqrt(5))  # should be 196 nm
        self.ELL_DI = PETSc.ScalarType(600e-9)  # 600 nm @ 300 K
        # self.ELL_DI = PETSc.ScalarType(1960e-9)
        self.MEAN_FREE_PATH_DI = self.ELL_DI * np.sqrt(5)

        self.KNUDSEN_SI = self.MEAN_FREE_PATH_SI / self.LENGTH
        self.KNUDSEN_DI = self.MEAN_FREE_PATH_DI / self.LENGTH
        self.KAPPA_SI = PETSc.ScalarType(141.0)
        self.KAPPA_DI = PETSc.ScalarType(2000.0)  # Estimated value for diamond
        # self.KAPPA_DI = PETSc.ScalarType(600.0)  # Estimated value for diamond

        self.L_X = 25 * self.LENGTH
        self.L_Y = 12.5 * self.LENGTH
        self.SOURCE_WIDTH = self.LENGTH
        self.SOURCE_HEIGHT = self.LENGTH * 0.25
        self.mask_extrusion = True
        if self.config.get("blank", False):
            self.blank = True
            self.mask_extrusion = False
        else:
            self.blank = False

        # Mesh resolution
        if arg_res is not None:
            self.RESOLUTION = self.LENGTH / arg_res
        else:
            res = self.config.get("res", 12.0)
            self.RESOLUTION = self.LENGTH / res if res > 0 else self.LENGTH / 12

        # Sources
        self.sources = self.config.get("sources", [0.5, self.Q_L])
        self.process_sources()

        # Symmetry
        self.symmetry = self.config.get("symmetry", False)
        if self.symmetry:
            self.L_X /= 2

        # Visualization
        self.visualize = self.config.get("visualize", [])
        self.plot_mode = self.config.get("plot_mode", "screenshot")

        # Optimization parameters
        self.optim = self.config.get("optim", False)
        self.optimizer = self.config.get("optimizer", "cmaes")
        self.latent = self.config.get("latent", None)
        self.latent_size = self.config.get("latent_size", 4)
        self.latent_method = self.config.get("latent_method", "preloaded")
        self.walltime = self.config.get("walltime", "03:00:00")
        self.popsize = self.config.get("popsize", 8)
        self.bounds = self.config.get("bounds", [-2.5, 2.5])
        self.n_iter = self.config.get("n_iter", 100)

        # Logging
        self.logging_enabled = self.config.get("logging_enabled", True)
        self.log_name = self.config.get("log_name", None)  # New: user-defined log name
        self.log_dir = os.path.join("logs", self.log_name)
        # Ensure the log directory  exists
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

        self.hpc_enabled = self.config.get("hpc_enabled", True)
        self.load_cma_result = self.config.get("load_cma_result", False)
        # if hpc activated:
        if self.hpc_enabled:
            # set to wall time * 0.95
            self.maxtime = int(float(self.walltime.split(":")[0]) * 3600 * 0.95+float(self.walltime.split(":")[1]) * 60 * 0.95)
            # print(f" Max time is {self.maxtime} seconds")
        else:
            self.maxtime = self.config.get("maxtime", 3600)  # default to 1 hour if not set
        
        self.parallelize = self.config.get("parallelize", False)

    def process_sources(self):
        sources = self.sources
        if len(sources) % 2 != 0:
            raise ValueError("Each source must have a position and a heat value.")
        source_pairs = [(sources[i], sources[i + 1]) for i in range(0, len(sources), 2)]
        self.source_positions = []
        self.Q_sources = []
        for pos, Q in source_pairs:
            if pos < 0 or pos > 1:
                raise ValueError("Source positions must be between 0 and 1.")
            self.source_positions.append(pos)
            self.Q_sources.append(PETSc.ScalarType(Q))
        # Sort sources
        combined_sources = sorted(zip(self.source_positions, self.Q_sources))
        self.source_positions, self.Q_sources = list(zip(*combined_sources))
        # num sources:
        self.num_sources = len(self.source_positions)