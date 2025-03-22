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
import gmsh
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import subprocess
import sys


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

        self.simulation_process = None  # To track the running subprocess


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
            self, self.options, visualize_options=["gamma", "temperature", "flux", "profiles", "pregamma","effective_conductivity"]
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
            ("Visualize Mesh", self.visualize_gmsh_mesh),  # <--- NEW
            ("Save Config", self.save_config),
            ("Load Config", self.load_config),
        ]

        for idx, (text, command) in enumerate(buttons):
            ctk.CTkButton(button_frame, text=text, command=command, width=150, height=40).grid(row=0, column=idx, padx=5, pady=5, sticky="ew")

        # REMOVE EMPTY SPACE BELOW
        self.grid_rowconfigure(3, weight=0)  # Fix the bottom row to not expand

    def run_simulation(self, config_path=None):
        # Check if a subprocess is already running
        if self.simulation_process and self.simulation_process.poll() is None:
            response = messagebox.askyesno(
                "Simulation Running",
                "A simulation is already running. Do you want to terminate it and start a new one?"
            )
            if response:  # IF YES, TERINATE AND RUN NEW SIMULATION
                self.simulation_process.terminate()
            else:  # IF NO, DO NOT RUN NEW SIMULATION
                return
        # IF NO SIMULATION RUNNING, OR TERMINATED, START NEW SIMULATION
        try:
            log_name = self.options["log_name"].get() or "default_log"
            if config_path is None:
                log_dir = os.path.join("logs", log_name)
                os.makedirs(log_dir, exist_ok=True)
                save_config(self.options, log_dir)
                config_path = os.path.join(log_dir, "config.json")
            command = [
                sys.executable,
                "src/main.py",
                "--config",
                config_path
            ]

            # Launch subprocess and keep reference
            self.simulation_process = subprocess.Popen(command)
            messagebox.showinfo("Simulation Started", f"Simulation started with config '{log_name}'.")

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
        # ISSUE: REMOVE THE DEFAULT LOG AND JUST DONT SAVE CONFIG IF EMPTY
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
                # check if any in visualize options are true
                # convert BooleanVars to actual values:
                config["visualize"] = {k: v.get() for k, v in self.options["visualize"].items()}
                # Turn all visualization options if none were clicked
                if not any(config["visualize"].values()):
                    config["visualize"] = {
                        "gamma": True,
                        "temperature": True,
                        "flux": True,
                        "profiles": True,
                        "pregamma": True,
                        "effective_conductivity": True
                    }

                config["res"] = 20.0  # Increase resolution for better visualization

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
                config_path = os.path.join("logs", config['log_name'], "viz_config.json")
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=4)
                # Create a SimulationController and run the simulation
                # controller = SimulationController(simulation_config)
                self.run_simulation(config_path=config_path)

                # Inform the user
                messagebox.showinfo("Success", "Simulation completed and visualizations saved.")

            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {e}")

    def visualize_gmsh_mesh(self):
            """
            Example method to build or load the Gmsh geometry,
            extract boundary lines and color them in a Matplotlib figure
            embedded in Tkinter.
            """
            try:
                # 1) Build or load your Gmsh model using your config
                #    (You could also call your 'sym_create_mesh' or 'create_mesh' 
                #    but *do not finalize gmsh yet*, because we want to query it.)
                
                gmsh.initialize()
                gmsh.model.add("visual_demo")

                # For illustration, let's do something simpler:
                # We'll just add a rectangle + a circle or so
                # Or call your existing geometry builder. For example:
                # my_mesh_gen = MeshGenerator(self.options)  # if that’s how you do it
                # my_mesh_gen.sym_create_mesh()              # but skip finalizing
                #
                # Instead, as a self-contained example:
                p0 = gmsh.model.geo.addPoint(0, 0, 0)
                p1 = gmsh.model.geo.addPoint(2, 0, 0)
                p2 = gmsh.model.geo.addPoint(2, 1, 0)
                p3 = gmsh.model.geo.addPoint(0, 1, 0)
                l0 = gmsh.model.geo.addLine(p0, p1)
                l1 = gmsh.model.geo.addLine(p1, p2)
                l2 = gmsh.model.geo.addLine(p2, p3)
                l3 = gmsh.model.geo.addLine(p3, p0)
                loop = gmsh.model.geo.addCurveLoop([l0,l1,l2,l3])
                s = gmsh.model.geo.addPlaneSurface([loop])
                # Physical tags
                gmsh.model.addPhysicalGroup(1, [l0], tag=1)
                gmsh.model.setPhysicalName(1, 1, "Bottom")
                gmsh.model.addPhysicalGroup(1, [l1], tag=2)
                gmsh.model.setPhysicalName(1, 2, "Right")
                gmsh.model.addPhysicalGroup(1, [l2,l3], tag=3)
                gmsh.model.setPhysicalName(1, 3, "OtherBoundary")
                gmsh.model.addPhysicalGroup(2, [s], tag=1)
                gmsh.model.setPhysicalName(2, 1, "Domain")

                gmsh.model.geo.synchronize()
                gmsh.model.mesh.generate(2)

                # 2) Extract boundary line data for each Physical Group
                dim = 1  # lines
                phys_groups = gmsh.model.getPhysicalGroups(dim) 
                # => [(1, tag1), (1, tag2), ...]

                # We'll store the lines (list of segments) and each segment's color / name
                boundary_segments = []
                color_map = {}
                color_cycle = plt.cm.tab10.colors  # or any color palette
                color_index = 0

                for (d, group_tag) in phys_groups:
                    # fetch the entity tags (line IDs) in this group
                    line_ids = gmsh.model.getEntitiesForPhysicalGroup(d, group_tag)
                    # get the group name
                    group_name = gmsh.model.getPhysicalName(d, group_tag)
                    # pick a color from the color cycle
                    this_color = color_cycle[color_index % len(color_cycle)]
                    color_index += 1
                    color_map[group_tag] = (group_name, this_color)

                    # For each line_id, get the node coordinates in 2D
                    for lid in line_ids:
                        # get mesh nodes for this line
                        node_tags, node_coords, _ = gmsh.model.mesh.getNodes(dim=d, tag=lid)
                        # node_coords is a flat list: [x1,y1,z1, x2,y2,z2, ...]
                        # The line connectivity
                        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=d, tag=lid)
                        # Typically there's 1 element type (e.g. 1D line segments):
                        # we can parse elem_node_tags[0] in pairs (or in order) to figure out each segment

                        # But for a simple line, it might be just 1 segment with 2 endpoints.
                        # We'll parse them thoroughly in case it's subdivided for the mesh.

                        # Build a mapping from nodeTag -> (x,y)
                        node_map = {}
                        for i, ntag in enumerate(node_tags):
                            x = node_coords[3*i]
                            y = node_coords[3*i+1]
                            node_map[ntag] = (x, y)

                        # Now get the connectivity
                        # e.g. for 2-node line segments, each element has 2 node tags
                        econn = elem_node_tags[0]  # the actual node tags in the element
                        # group them in pairs
                        pairs = [econn[i:i+2] for i in range(0, len(econn), 2)]
                        for (nd1, nd2) in pairs:
                            x1, y1 = node_map[nd1]
                            x2, y2 = node_map[nd2]
                            boundary_segments.append({
                                "group_tag": group_tag,
                                "x": [x1, x2],
                                "y": [y1, y2],
                            })

                # 3) Now gmsh.finalize() if you like
                gmsh.finalize()

                # 4) Create a Matplotlib Figure, plot each segment in its group color
                fig = Figure(figsize=(5, 4), dpi=100)
                ax = fig.add_subplot(111)
                ax.set_title("Gmsh Boundary Visualization")
                ax.set_aspect("equal", "box")

                for seg in boundary_segments:
                    grp = seg["group_tag"]
                    (grp_name, c) = color_map[grp]
                    ax.plot(seg["x"], seg["y"], color=c, lw=2, label=grp_name)

                # Because multiple segments from the same group can appear,
                # we don't want a repeating label in the legend. Let's do a trick:
                # get existing labels, only label the first segment of each group.
                handles, labels = ax.get_legend_handles_labels()
                # remove duplicates
                unique = dict()
                for h, l in zip(handles, labels):
                    if l not in unique:
                        unique[l] = h
                ax.legend(unique.values(), unique.keys(), loc='best')

                # 5) Embed this figure in a Tkinter Toplevel or inside your main window
                top = ctk.CTkToplevel(self)
                top.title("Mesh Visualization")

                canvas = FigureCanvasTkAgg(fig, master=top)  
                canvas.draw()
                canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

            except Exception as e:
                traceback.print_exc()
                messagebox.showerror("Error", str(e))