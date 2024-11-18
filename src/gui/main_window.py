import tkinter as tk
import traceback
from tkinter import messagebox, filedialog

from .optimization_frame import OptimizationFrame
from .material_frame import MaterialFrame
from .solving_frame import SolvingFrame
from .visualization_frame import VisualizationFrame
from .sources_frame import SourcesFrame
from .hpc_frame import HPCFrame

from sim_config import SimulationConfig
from main import SimulationController
from hpc.script_generator import generate_hpc_script
from .utils_config import initialize_options,  get_config_dict, save_config, load_config
from hpc.hpc_utils import submit_job, prompt_password
import os


class SimulationConfigGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulation Configuration")
        self.set_focus()

        # Initialize options
        self.options = initialize_options()

        # Initialize frames
        self.init_frames()

        # Add buttons
        self.add_buttons()

    def set_focus(self):
        """Ensure the GUI window appears in front of other apps."""
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.update()
        self.root.attributes("-topmost", False)

    def init_frames(self):
        """Initialize and add all frames to the GUI."""
        self.optimization_frame = OptimizationFrame(self.root, self.options)
        self.material_frame = MaterialFrame(self.root, self.options)
        self.solving_frame = SolvingFrame(self.root, self.options)
        self.visualization_frame = VisualizationFrame(
            self.root, self.options, visualize_options=["gamma", "temperature", "flux", "profiles", "pregamma"]
        )
        self.sources_frame = SourcesFrame(self.root, self.options, self.material_frame)
        self.hpc_frame = HPCFrame(self.root, self.options)

    def add_buttons(self):
        button_frame = tk.Frame(self.root)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)

        tk.Button(button_frame, text="Run Simulation", command=self.run_simulation).pack(side="left", padx=5)
        tk.Button(button_frame, text="Transfer and submit to HPC", command=self.submit_to_hpc).pack(side="left", padx=5)
        tk.Button(button_frame, text="Save Config", command=self.save_config).pack(side="left", padx=5)
        tk.Button(button_frame, text="Load Config", command=self.load_config).pack(side="left", padx=5)

    def run_simulation(self):
        try:
            # Extracting the options
            config_dict = get_config_dict(self.options)

            # Initialize SimulationConfig with the extracted args
            config = SimulationConfig(config_dict)
            sim = SimulationController(config)

            # Run the simulation
            sim.run_simulation()

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", str(e))

    def submit_to_hpc(self):
        try:
            # Generate HPC script
            config = get_config_dict(self.options)

            log_name = self.options["log_name"].get() or "default_log"
            log_dir = os.path.join("logs", log_name)
            os.makedirs(log_dir, exist_ok=True)
            save_config(self.options, log_dir)
            config_path = os.path.join(log_dir, "config.json")

            script_content = generate_hpc_script(config, config_path)
            self.save_hpc_script(script_content, log_dir)

            # Prompt for password and submit the job
            password = prompt_password()
            hpc_user = self.options["hpc_user"].get()
            hpc_host = self.options["hpc_host"].get()
            hpc_path = self.options["hpc_dir"].get()
            submit_job(hpc_user, hpc_host, hpc_path, log_dir, password)

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", str(e))

    def save_hpc_script(self, hpc_script, log_dir):
        """Save the HPC script in the log directory."""
        # Save HPC script
        hpc_path = os.path.join(log_dir, "hpc_run.sh")
        with open(hpc_path, "w") as f:
            f.write(hpc_script)

    def save_config(self):
        log_name = self.options["log_name"].get() or "default_log"
        log_dir = os.path.join("logs", log_name)
        os.makedirs(log_dir, exist_ok=True)
        save_config(self.options, log_dir)

    def load_config(self):
        file_path = filedialog.askopenfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            load_config(self.options, file_path)
