import json
import os
from petsc4py import PETSc
import numpy as np


class SimulationConfig:
    def __init__(self, config_path):
        # Load configurations from JSON file
        with open(config_path, "r") as f:
            config = json.load(f)

        self.config = config  # Store the entire config for later use

        # Physical properties
        self.C = PETSc.ScalarType(1.0)  # Slip parameter
        self.T_ISO = PETSc.ScalarType(0.0)
        self.Q_L = PETSc.ScalarType(80)

        self.MEAN_FREE_PATH = 0.439e-6
        self.KNUDSEN = 1

        # Volume fraction
        self.vol_fraction = (
            config.get("vf_value", 0.2) if config.get("vf_enabled", True) else None
        )

        # Geometric properties
        self.LENGTH = self.MEAN_FREE_PATH / self.KNUDSEN
        self.L_X = 25 * self.LENGTH
        self.L_Y = 12.5 * self.LENGTH
        self.SOURCE_WIDTH = self.LENGTH
        self.SOURCE_HEIGHT = self.LENGTH * 0.25
        self.mask_extrusion = not config.get("blank", False)
        self.blank = config.get("blank", False)

        # Mesh resolution
        res = config.get("res", 12.0)
        self.RESOLUTION = self.LENGTH / res if res > 0 else self.LENGTH / 12

        # Material properties
        self.ELL_SI = PETSc.ScalarType(self.MEAN_FREE_PATH / np.sqrt(5))
        self.ELL_DI = PETSc.ScalarType(196e-8)
        self.KAPPA_SI = PETSc.ScalarType(141.0)
        self.KAPPA_DI = PETSc.ScalarType(600.0)

        # Sources
        self.sources = config.get("sources", [0.5, self.Q_L])
        self.process_sources()

        # Symmetry
        self.symmetry = config.get("symmetry", False)
        if self.symmetry:
            self.L_X /= 2
            self.SOURCE_WIDTH /= 2

        # Visualization
        self.visualize = config.get("visualize", [])
        self.plot_mode = config.get("plot_mode", "screenshot")

        # Optimization parameters
        self.optim = config.get("optim", False)
        self.optimizer = config.get("optimizer", "cmaes")
        self.latent = config.get("latent", None)
        self.latent_size = config.get("latent_size", 4)
        self.latent_method = config.get("latent_method", "preloaded")

        # Logging
        self.logging_enabled = config.get("logging_enabled", True)
        self.log_name = config.get("log_name", None)  # New: user-defined log name
        self.log_dir = os.path.join("logs", self.log_name)
        # Ensure the log directory  exists
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

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
