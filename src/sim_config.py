import json
import os
from petsc4py import PETSc
import numpy as np


class SimulationConfig:
    def __init__(self, config):
        """
        Initialize the simulation configuration.

        Parameters:
        - config (str or dict): Path to a JSON configuration file or a dictionary containing configuration data.
        """
        if isinstance(config, str):  # If a file path is provided
            with open(config, "r") as f:
                print("Loading config from file")
                self.config = json.load(f)
        elif isinstance(config, dict):  # If a dictionary is provided
            self.config = config
        else:
            raise TypeError("Config must be either a file path (str) or a dictionary.")

        # Physical properties
        self.C = PETSc.ScalarType(1.0)  # Slip parameter
        self.T_ISO = PETSc.ScalarType(0.0)
        self.Q_L = PETSc.ScalarType(80)

        self.MEAN_FREE_PATH = 0.439e-6
        self.KNUDSEN = 1

        # Volume fraction
        self.vol_fraction = (
            self.config.get("vf_value", 0.2)
            if self.config.get("vf_enabled", True)
            else None
        )

        # Geometric properties
        self.LENGTH = self.MEAN_FREE_PATH / self.KNUDSEN
        self.L_X = 25 * self.LENGTH
        self.L_Y = 12.5 * self.LENGTH
        self.SOURCE_WIDTH = self.LENGTH
        self.SOURCE_HEIGHT = self.LENGTH * 0.25
        self.mask_extrusion = not self.config.get("blank", False)
        self.blank = self.config.get("blank", False)

        # Mesh resolution
        res = self.config.get("res", 12.0)
        self.RESOLUTION = self.LENGTH / res if res > 0 else self.LENGTH / 12

        # Material properties
        self.ELL_SI = PETSc.ScalarType(self.MEAN_FREE_PATH / np.sqrt(5))
        self.ELL_DI = PETSc.ScalarType(196e-8)
        self.KAPPA_SI = PETSc.ScalarType(141.0)
        self.KAPPA_DI = PETSc.ScalarType(600.0)

        # Sources
        self.sources = self.config.get("sources", [0.5, self.Q_L])
        self.process_sources()

        # Symmetry
        self.symmetry = self.config.get("symmetry", False)
        if self.symmetry:
            self.L_X /= 2
            self.SOURCE_WIDTH /= 2

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