import tkinter as tk
import traceback
from tkinter import messagebox

from .optimization_frame import OptimizationFrame
from .material_frame import MaterialFrame
from .solving_frame import SolvingFrame
from .visualization_frame import VisualizationFrame
from .sources_frame import SourcesFrame
from .hpc_frame import HPCFrame

from sim_config import SimulationConfig
from main import SimulationController
from .helpers import (generate_command_line, parse_sources, get_visualize_options,
                      generate_hpc_script)
import subprocess


class SimulationConfigGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulation Configuration")
        # Set the GUI window to appear in front of all apps
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.update()  # Apply the change immediately
        self.root.attributes("-topmost", False)  # Allow other windows to take focus later

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
            button_frame, text="Transfer and submit to HPC", command=self.submit_to_hpc
        ).pack(side="left", padx=5)

    def run_simulation(self):
        try:
            # Extracting the options
            args = self.get_args()

            # Initialize SimulationConfig with the extracted args
            config = SimulationConfig(args)
            sim = SimulationController(config)

            # Run the simulation
            sim.run_simulation()

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
        args.conda_env_path = self.options["conda_env_path"].get()
        args.conda_env_name = self.options["conda_env_name"].get()
        return args

    def submit_to_hpc(self):
        try:
            # Extract options and generate the HPC script
            args = self.get_args()
            hpc_script = generate_hpc_script(args)

            # Save the HPC script locally
            with open("hpc_run.sh", "w") as f:
                f.write(hpc_script)

            # Transfer files to HPC
            self.transfer_files_to_hpc()

            # Submit the job on the HPC
            self.submit_job_on_hpc()

            messagebox.showinfo("HPC Job Submitted", "The job has been submitted to the HPC.")

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", str(e))

    def submit_job_on_hpc(self):
        hpc_user = "pt721"
        hpc_host = "login.cx3.hpc.ic.ac.uk"
        hpc_remote_path = "~/BTE-NO"
        password = self.get_password(prompt="Enter HPC password for file transfer")

        try:
            # SSH command with environment setup for qsub
            ssh_command = [
                "sshpass", "-p", password,
                "ssh", f"{hpc_user}@{hpc_host}",
                "bash -l -c 'cd ~/BTE-NO && qsub hpc_run.sh'"
            ]

            # Run the command
            result = subprocess.run(ssh_command, check=True, text=True, capture_output=True)

            # Extract the job ID from the result
            job_id = result.stdout.strip()
            messagebox.showinfo("Success", f"Job submitted successfully!\nJob ID: {job_id}")

        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Job submission failed.\n{e.stderr}")

        except Exception as e:
            messagebox.showerror("Unexpected Error", f"An unexpected error occurred: {str(e)}")

    def transfer_files_to_hpc(self):
        local_path = "."
        hpc_user = "pt721"
        hpc_host = "login.cx3.hpc.ic.ac.uk"
        hpc_remote_path = "~/BTE-NO"

        # Get password
        password = self.get_password(prompt="Enter HPC password for file transfer")

        # Use subprocess with `sshpass` for password-based file transfer
        import subprocess
        rsync_command = [
            "sshpass", "-p", password,
            "rsync", "-avz", "--progress",
            "--exclude-from", f"{local_path}/hpc_exclude.txt",
            f"{local_path}/src",
            f"{local_path}/hpc_run.sh",
            f"{hpc_user}@{hpc_host}:{hpc_remote_path}"
        ]
        subprocess.run(rsync_command, check=True)

    def get_password(self, prompt="Enter HPC Password"):
        """Prompt the user to enter their HPC password."""
        password_window = tk.Toplevel(self.root)
        password_window.title(prompt)

        tk.Label(password_window, text=prompt).pack(pady=5)

        password_var = tk.StringVar()
        password_entry = tk.Entry(password_window, show="*", textvariable=password_var)
        password_entry.pack(pady=5)
        password_entry.focus_set()

        def on_submit():
            password_window.destroy()

        tk.Button(password_window, text="Submit", command=on_submit).pack(pady=5)

        # Wait for the user to enter the password
        password_window.wait_window()

        return password_var.get()
