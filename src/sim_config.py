from petsc4py import PETSc


class SimulationConfig:
    def __init__(self, args):
        self.args = args

        # Physical properties
        self.C = PETSc.ScalarType(1.0)  # Slip parameter for fully diffusive boundaries
        self.T_ISO = PETSc.ScalarType(0.0)  # Isothermal temperature, K
        self.Q_L = PETSc.ScalarType(80)  # Line heat source, W/m

        self.MEAN_FREE_PATH = 0.439e-6  # Characteristic length, adjust as necessary
        self.KNUDSEN = 1  # Knudsen number, adjust as necessary

        # Volume fraction
        if args.vf is None:
            self.vol_fraction = None
        else:
            self.vol_fraction = args.vf
        # Geometric properties
        self.LENGTH = self.MEAN_FREE_PATH / self.KNUDSEN  # Characteristic length, L
        # Base rectangle:
        self.L_X = 25 * self.LENGTH
        self.L_Y = 12.5 * self.LENGTH
        # Source rectangle:
        self.SOURCE_WIDTH = self.LENGTH
        self.SOURCE_HEIGHT = self.LENGTH * 0.25
        self.mask_extrusion = True

        # Set the resolution of the mesh as a divider of LENGTH
        if args.res is not None and args.res > 0:
            self.RESOLUTION = self.LENGTH / args.res
        else:
            self.RESOLUTION = self.LENGTH / 12  # default value
            # self.RESOLUTION = self.LENGTH / 5  # to get quick profiles

        # material properties
        self.ELL_SI = PETSc.ScalarType(
            self.MEAN_FREE_PATH / np.sqrt(5)
        )  # Non-local length, m
        self.ELL_DI = PETSc.ScalarType(196e-8)
        self.KAPPA_SI = PETSc.ScalarType(141.0)  # W/mK, thermal conductivity
        self.KAPPA_DI = PETSc.ScalarType(600.0)

        if args.sources is not None:
            if len(args.sources) % 2 != 0:
                raise ValueError(
                    "Each source must have a position and a heat value. Please provide pairs of values."
                )
            # Group the list into pairs
            sources_pairs = [
                (args.sources[i], args.sources[i + 1])
                for i in range(0, len(args.sources), 2)
            ]
            self.source_positions = []
            self.Q_sources = []
            for pos, Q in sources_pairs:
                if pos < 0 or pos > 1:
                    raise ValueError(
                        "Source positions must be between 0 and 1 (normalized)."
                    )
                self.source_positions.append(pos)
                self.Q_sources.append(PETSc.ScalarType(Q))
            # Sort the sources by position
            combined_sources = sorted(zip(self.source_positions, self.Q_sources))
            self.source_positions, self.Q_sources = list(zip(*combined_sources))
        else:
            # Default to a single source at the center
            self.source_positions = [0.5]
            self.Q_sources = [PETSc.ScalarType(self.Q_L)]

        # Cannot be symmetry and two sources:
        if len(self.source_positions) > 1 and args.symmetry:
            raise ValueError("Cannot have both symmetry and two sources.")

        # symmetry in geometry
        if args.symmetry:
            self.L_X = self.L_X / 2
            self.SOURCE_WIDTH = self.SOURCE_WIDTH / 2
            self.symmetry = True  # Enable or disable symmetry
        else:
            self.symmetry = False

        if args.blank:
            self.mask_extrusion = False

        # Parse visualize argument
        if args.visualize is None:
            self.visualize = []
        else:
            self.visualize = args.visualize
        # set plot mode only if it exists
        self.plot_mode = args.plot_mode if hasattr(args, "plot_mode") else None
        self.optim = args.optim
        self.optimizer = args.optimizer
        self.latent = args.latent
        self.blank = args.blank

        self.latent_size = args.latent_size  # New: size of the latent vector
        self.latent_method = (
            args.latent_method
        )  # New: method to obtain the latent vector

        self.logging_enabled = (
            not args.no_logging
        )  # Logging is enabled unless --no-logging is used
