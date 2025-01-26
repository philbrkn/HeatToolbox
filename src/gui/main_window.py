# import tkinter as tk
import customtkinter as ctk
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
import json


class SimulationConfigGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Set Modern Appearance
        ctk.set_appearance_mode("Dark")  # Modes: "Light", "Dark", "System"
        ctk.set_default_color_theme("blue")  # Themes: "blue", "dark-blue", "green"

        # Ensure the main window stretches properly but doesn't leave extra space
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure(3, weight=1)
        self.grid_rowconfigure(4, weight=1)
        self.grid_rowconfigure(5, weight=1)  # Ensure the last row stretches
        self.grid_rowconfigure(99, weight=0)  # Ensure last row does NOT expand

        # self.root = root
        # self.root.title("Simulation Configuration")
        # self.set_focus()

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
        self.optimization_frame = OptimizationFrame(self, self.options)
        self.material_frame = MaterialFrame(self, self.options)
        self.solving_frame = SolvingFrame(self, self.options)
        self.visualization_frame = VisualizationFrame(
            self, self.options, visualize_options=["gamma", "temperature", "flux", "profiles", "pregamma"]
        )
        self.sources_frame = SourcesFrame(self, self.options, self.material_frame)
        self.hpc_frame = HPCFrame(self, self.options)

    def add_buttons(self):
        """Create modern buttons using CTkButton."""
        button_frame = ctk.CTkFrame(self)
        button_frame.grid(row=3, column=0, columnspan=3, pady=10, padx=10, sticky="ew")

        self.grid_columnconfigure(0, weight=1)  # Ensure buttons expand evenly

        buttons = [
            ("Run Simulation", self.run_simulation),
            ("Submit to HPC", self.submit_to_hpc),
            ("Visualize Best", self.visualize_best_result),
            ("Save Config", self.save_config),
            ("Load Config", self.load_config),
        ]

        for idx, (text, command) in enumerate(buttons):
            ctk.CTkButton(button_frame, text=text, command=command, width=150, height=40).grid(row=0, column=idx, padx=5, pady=5, sticky="ew")

        # REMOVE EMPTY SPACE BELOW
        self.grid_rowconfigure(3, weight=0)  # Fix the bottom row to not expand

    def run_simulation(self):
        try:
            # Extracting the options
            # config = get_config_dict(self.options)
            log_name = self.options["log_name"].get() or "default_log"
            log_dir = os.path.join("logs", log_name)
            os.makedirs(log_dir, exist_ok=True)
            save_config(self.options, log_dir)
            config_path = os.path.join(log_dir, "config.json")

            # Initialize SimulationConfig with the extracted args
            config = SimulationConfig(config_path)
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

    def visualize_best_result(self):
        """Load config, modify it, and run simulation on the best latent vector."""
        # Ask the user to select a config file
        file_path = filedialog.askopenfilename(
            title="Select Configuration File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            try:
                # Load the configuration from the selected file
                with open(file_path, "r") as f:
                    config = json.load(f)

                # Modify the configuration in memory
                config["optim"] = False
                config["hpc_enabled"] = False
                # Turn all visualization options on
                if "visualize" in config:
                    for key in config["visualize"]:
                        config["visualize"][key] = True
                else:
                    config["visualize"] = {
                        "gamma": True,
                        "temperature": True,
                        "flux": True,
                        "profiles": True,
                        "pregamma": True
                    }

                # # Ensure we have the log directory
                # if "log_name" in config:
                #     log_dir = os.path.join("logs", config["log_name"])
                # else:
                #     messagebox.showerror("Error", "The configuration file must contain 'log_name'.")
                #     return

                config["load_cma_result"] = True

                # Update options in the GUI (optional, if we want to reflect changes)
                # self.options = {}  # Reset options
                # set_options_from_config(self.options, config)

                # Create a SimulationConfig object
                simulation_config = SimulationConfig(config)

                # Create a SimulationController and run the simulation
                controller = SimulationController(simulation_config)
                controller.run_simulation()

                # Inform the user
                messagebox.showinfo("Success", "Simulation completed and visualizations saved.")

            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {e}")
