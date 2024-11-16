import tkinter as tk
import traceback
from tkinter import messagebox

from .optimization_frame import OptimizationFrame
from .material_frame import MaterialFrame
from .solving_frame import SolvingFrame
from .visualization_frame import VisualizationFrame
from .sources_frame import SourcesFrame
from .hpc_frame import HPCFrame

from main import SimulationConfig, main
from .helpers import (generate_command_line, parse_sources, get_visualize_options,
                      generate_hpc_script)


class SimulationConfigGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulation Configuration")
        self.options = self.initialize_options()

        # Initialize frames
        self.optimization_frame = OptimizationFrame(
            self.root, self.options
        )
        self.material_frame = MaterialFrame(self.root, self.options)
        self.solving_frame = SolvingFrame(self.root, self.options)
        self.visualization_frame = VisualizationFrame(
            self.root,
            self.options,
            visualize_options=["gamma", "temperature", "flux", "profiles", "pregamma"]
        )
        self.sources_frame = SourcesFrame(self.root, self.options, self.material_frame)
        self.hpc_frame = HPCFrame(self.root, self.options)

        # Add buttons
        self.add_buttons()

    def initialize_options(self):
        return {
            "optim": tk.BooleanVar(),
            "optimizer": tk.StringVar(value="cmaes"),
            "latent_size": tk.IntVar(value=4),
            "latent_method": tk.StringVar(value="preloaded"),
            "latent": [tk.DoubleVar() for _ in range(4)],
            "symmetry": tk.BooleanVar(),
            "blank": tk.BooleanVar(),
            "sources": [],
            "res": tk.DoubleVar(value=12),
            "visualize": {},
            "vf_enabled": tk.BooleanVar(value=True),
            "vf_value": tk.DoubleVar(value=0.2),
            "plot_mode": tk.StringVar(value="screenshot"),
            "logging_enabled": tk.BooleanVar(value=True),
            # New HPC script options
            "nodes": tk.IntVar(value=1),
            "ncpus": tk.IntVar(value=4),
            "mem": tk.IntVar(value=8),  # in GB
            "walltime": tk.StringVar(value="03:00:00"),
            "timeout": tk.StringVar(value="2:55:00"),
            "parallelize": tk.BooleanVar(value=False),
            "mpiprocs": tk.IntVar(value=4),
            "conda_env_path": tk.StringVar(value="~/miniforge3/bin/conda"),
            "conda_env_name": tk.StringVar(value="fenicsx_torch"),
        }

    def add_buttons(self):
        button_frame = tk.Frame(self.root)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)

        tk.Button(
            button_frame, text="Run Simulation", command=self.run_simulation
        ).pack(side="left", padx=5)
        tk.Button(
            button_frame, text="Generate Command", command=self.generate_command
        ).pack(side="left", padx=5)
        tk.Button(
            button_frame, text="Generate HPC Script", command=self.generate_hpc_script
        ).pack(side="left", padx=5)

    def run_simulation(self):
        try:
            # Extracting the options
            args = self.get_args()

            # Initialize SimulationConfig with the extracted args
            config = SimulationConfig(args)

            # Run the simulation
            main(config)

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", str(e))

    def generate_command(self):
        try:
            # Extracting the options
            args = self.get_args()

            # Generate the command-line command
            command = generate_command_line(args)

            # Print the command in the terminal
            print("Generated Command: ")
            print(command)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def generate_hpc_script(self):
        try:
            # Extracting the options
            args = self.get_args()

            # Generate the HPC script content
            script_content = generate_hpc_script(args)

            # Save the script to a file
            with open("hpc_run.sh", "w") as f:
                f.write(script_content)

            messagebox.showinfo("HPC Script Generated", "HPC script saved as hpc_run.sh")

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", str(e))

    def get_args(self):
        class Args:
            pass

        args = Args()
        args.optim = self.options["optim"].get()
        args.optimizer = self.options["optimizer"].get()
        args.symmetry = self.options["symmetry"].get()
        args.blank = self.options["blank"].get()
        args.sources = parse_sources(self.options["sources"])
        # Pass resolution as the divider input
        args.res = self.options["res"].get() if self.options["res"].get() > 0 else None
        args.visualize = get_visualize_options(self.options["visualize"])
        if self.options["vf_enabled"].get():
            args.vf = self.options["vf_value"].get()
        else:
            args.vf = None  # Disable volume fraction control
        if args.visualize:
            args.plot_mode = self.options["plot_mode"].get()
        args.latent_size = self.options["latent_size"].get()
        args.latent_method = self.options["latent_method"].get()
        if args.latent_method == "manual":
            args.latent = [val.get() for val in self.options["latent"]]
        else:
            args.latent = None  # Latent vector will be handled based on method
        args.no_logging = not self.options["logging_enabled"].get()

        # HPC script options
        args.nodes = self.options["nodes"].get()
        args.ncpus = self.options["ncpus"].get()
        args.mem = self.options["mem"].get()
        args.walltime = self.options["walltime"].get()
        args.timeout = self.options["timeout"].get()
        args.parallelize = self.options["parallelize"].get()
        args.mpiprocs = self.options["mpiprocs"].get()
        args.total_procs = args.mpiprocs * args.nodes if args.parallelize else None

        return args
