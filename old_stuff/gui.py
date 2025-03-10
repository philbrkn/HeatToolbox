# gui.py
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from vae_module import VAE, Flatten, UnFlatten
import traceback

from main import SimulationConfig, main


class SimulationConfigGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulation Configuration")

        # Command-line equivalent options
        self.options = {
            "optim": tk.BooleanVar(),
            "optimizer": tk.StringVar(value="cmaes"),  # Default is bayesian
            "latent_size": tk.IntVar(value=4),  # New: latent vector size
            "latent_method": tk.StringVar(value="preloaded"),  # New: latent vector method
            "latent": [],  # Update: will be initialized based on latent size
            "symmetry": tk.BooleanVar(),
            "blank": tk.BooleanVar(),
            "sources": [],  # List to hold source entries
            "res": tk.DoubleVar(value=12),
            "visualize": {},  # Dictionary to hold visualization options
            "vf_enabled": tk.BooleanVar(value=True),
            "vf_value": tk.DoubleVar(value=0.2),
            "plot_mode": tk.StringVar(value="screenshot"),  # 'screenshot' or 'interactive'
            "logging_enabled": tk.BooleanVar(value=True),  # Default is logging enabled
        }

        self.options["latent"] = [tk.DoubleVar() for _ in range(4)]

        # Visualization options
        self.visualize_options = ["gamma", "temperature", "flux", "profiles", "pregamma", "effective_conductivity"]

        # Create the GUI components
        self.create_widgets()

    def create_widgets(self):
        # Optimization Section
        optimization_frame = ttk.LabelFrame(self.root, text="Optimization Options")
        optimization_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")

        # Material Topology Section
        material_frame = ttk.LabelFrame(self.root, text="Material Topologies")
        material_frame.grid(row=0, column=1, padx=10, pady=5, sticky="nsew")

        # Solving Section
        solving_frame = ttk.LabelFrame(self.root, text="Solving Options")
        solving_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

        # Visualization Section
        visualization_frame = ttk.LabelFrame(self.root, text="Visualization Options")
        visualization_frame.grid(row=1, column=1, padx=10, pady=5, sticky="nsew")

        # Buttons Section
        button_frame = tk.Frame(self.root)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)

        # Optimization Section
        tk.Checkbutton(
            optimization_frame,
            text="Run Optimization",
            variable=self.options["optim"],
            command=self.toggle_optimizer,
        ).grid(row=0, column=0, sticky="w")

        self.optimizer_label = tk.Label(optimization_frame, text="Select Optimizer")
        self.optimizer_label.grid(row=1, column=0, sticky="w")
        self.optimizer_label.grid_remove()

        self.optimizer_menu = tk.OptionMenu(
            optimization_frame, self.options["optimizer"], "bayesian", "cmaes"
        )
        self.optimizer_menu.grid(row=1, column=1)
        self.optimizer_menu.grid_remove()

        # Material Topology Section
        # Add latent size option
        tk.Label(material_frame, text="Latent Size").grid(row=0, column=0, sticky="w")
        tk.OptionMenu(
            material_frame,
            self.options["latent_size"],
            2, 4, 8, 16,
            command=self.update_latent_size  # Callback to update latent variables
        ).grid(row=0, column=1, sticky="w")

        # Add latent method option
        tk.Label(material_frame, text="Latent Method").grid(row=1, column=0, sticky="w")
        tk.OptionMenu(
            material_frame,
            self.options["latent_method"],
            "manual", "random", "preloaded",
            command=self.update_latent_method  # Callback to update latent entry fields
        ).grid(row=1, column=1, sticky="w")

        # Latent Values Entry (will be updated based on latent size and method)
        self.latent_frame = tk.Frame(material_frame)
        self.latent_frame.grid(row=2, column=0, columnspan=2, sticky="w")
        self.update_latent_entries()  # Initialize latent entries

        # "Enable Symmetry" Checkbutton (moved to row 3)
        tk.Checkbutton(
            material_frame, text="Enable Symmetry", variable=self.options["symmetry"]
        ).grid(row=3, column=0, sticky="w")

        # "Run with Blank Image" Checkbutton (moved to row 4)
        tk.Checkbutton(
            material_frame, text="Run with Blank Image", variable=self.options["blank"]
        ).grid(row=4, column=0, sticky="w")

        # "Sources (Position, Heat)" Label (moved to row 5)
        tk.Label(material_frame, text="Sources (Position, Heat)").grid(
            row=5, column=0, sticky="w"
        )

        # Sources Frame (moved to row 6)
        self.sources_frame = tk.Frame(material_frame)
        self.sources_frame.grid(row=6, column=1, columnspan=4, sticky="w")

        # "Add Source" Button (moved to row 7)
        tk.Button(
            material_frame, text="Add Source", command=self.add_source_row
        ).grid(row=7, column=0, sticky="w")

        # Solving Section
        tk.Label(solving_frame, text="Mesh Resolution: Length / [] ").grid(
            row=0, column=0, sticky="w"
        )
        tk.Entry(solving_frame, textvariable=self.options["res"], width=10).grid(
            row=0, column=1
        )

        tk.Checkbutton(
            solving_frame,
            text="Enable Volume Fraction Control",
            variable=self.options["vf_enabled"],
            command=self.toggle_volume_fraction,
        ).grid(row=1, column=0, sticky="w")

        tk.Label(solving_frame, text="Volume Fraction").grid(
            row=2, column=0, sticky="w"
        )
        self.vf_entry = tk.Entry(solving_frame, textvariable=self.options["vf_value"], width=10)
        self.vf_entry.grid(row=2, column=1)
        self.toggle_volume_fraction()

        # Logging Toggle
        tk.Checkbutton(
            solving_frame,
            text="Enable Logging",
            variable=self.options["logging_enabled"],
        ).grid(row=3, column=0, sticky="w")

        # Visualization Section
        tk.Label(visualization_frame, text="Visualization Options").grid(
            row=0, column=0, sticky="nw"
        )
        self.visualize_frame = tk.Frame(visualization_frame)
        self.visualize_frame.grid(row=0, column=1, sticky="w")

        for option in self.visualize_options:
            self.options["visualize"][option] = tk.BooleanVar()
            tk.Checkbutton(
                self.visualize_frame,
                text=option,
                variable=self.options["visualize"][option],
                command=self.update_plot_mode_visibility  # Attach callback
            ).pack(anchor="w")

        tk.Label(visualization_frame, text="Plotting Mode").grid(row=1, column=0, sticky="w")

        # Plotting Mode Section (initially hidden)
        self.plot_mode_frame = tk.Frame(visualization_frame)
        self.plot_mode_frame.grid(row=1, column=1, sticky="w")
        self.plot_mode_frame.grid_remove()  # Hide initially

        tk.Radiobutton(
            self.plot_mode_frame,
            text="Save Screenshots",
            variable=self.options["plot_mode"],
            value="screenshot"
        ).grid(row=1, column=1, sticky="w")

        tk.Radiobutton(
            self.plot_mode_frame,
            text="Interactive Plotting",
            variable=self.options["plot_mode"],
            value="interactive"
        ).grid(row=2, column=1, sticky="w")

        # Buttons Section
        self.run_button = tk.Button(
            button_frame, text="Run Simulation", command=self.run_simulation
        )
        self.run_button.pack(side="left", padx=5)

        self.generate_button = tk.Button(
            button_frame, text="Generate Command", command=self.generate_command
        )
        self.generate_button.pack(side="left", padx=5)

    def add_source_row(self):
        source_row = {}
        row_frame = tk.Frame(self.sources_frame)
        row_frame.pack(anchor="w", pady=2)

        # Position Entry
        tk.Label(row_frame, text="Position:").pack(side="left")
        position_var = tk.DoubleVar()
        tk.Entry(row_frame, textvariable=position_var, width=5).pack(side="left")

        # Heat Value Entry
        tk.Label(row_frame, text="Heat:").pack(side="left")
        heat_var = tk.DoubleVar()
        tk.Entry(row_frame, textvariable=heat_var, width=5).pack(side="left")

        # Remove Button
        remove_button = tk.Button(row_frame, text="Remove", command=lambda: self.remove_source_row(source_row))
        remove_button.pack(side="left", padx=5)

        # Store references
        source_row["frame"] = row_frame
        source_row["position"] = position_var
        source_row["heat"] = heat_var
        source_row["remove_button"] = remove_button

        self.options["sources"].append(source_row)

    def remove_source_row(self, source_row):
        source_row["frame"].destroy()
        self.options["sources"].remove(source_row)

    def run_simulation(self):
        try:
            # Disable the "Generate Command" button while running the simulation
            self.generate_button.config(state="disabled")

            # Extracting the options
            args = self.get_args()

            # Initialize SimulationConfig with the extracted args
            config = SimulationConfig(args)

            # Run the simulation
            main(config)

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", str(e))

        finally:
            # Re-enable the "Generate Command" button after simulation
            self.generate_button.config(state="normal")

    def generate_command(self):
        try:
            # Disable the "Run Simulation" button while generating the command
            self.run_button.config(state="disabled")

            # Extracting the options
            args = self.get_args()

            # Generate the command-line command
            command = generate_command_line(args)

            # Print the command in the terminal
            print(f"Generated Command: {command}")

        except Exception as e:
            messagebox.showerror("Error", str(e))

        finally:
            # Re-enable the "Run Simulation" button after generating the command
            self.run_button.config(state="normal")

    def get_args(self):
        class Args:
            pass

        args = Args()
        args.optim = self.options["optim"].get()
        args.optimizer = self.options["optimizer"].get()
        args.symmetry = self.options["symmetry"].get()
        args.blank = self.options["blank"].get()
        args.sources = self.parse_sources()
        # Pass resolution as the divider input
        args.res = self.options["res"].get() if self.options["res"].get() > 0 else None
        args.visualize = self.get_visualize_options()
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

        return args

    def parse_sources(self):
        sources_list = []
        for source in self.options["sources"]:
            pos = source["position"].get()
            heat = source["heat"].get()
            if pos == "" or heat == "":
                continue  # Skip empty entries
            try:
                pos = float(pos)
                heat = float(heat)
                if pos < 0 or pos > 1:
                    raise ValueError("Source positions must be between 0 and 1 (normalized).")
                sources_list.extend([pos, heat])
            except ValueError as e:
                raise ValueError(f"Invalid source input: {e}")
        return sources_list if sources_list else None

    def get_visualize_options(self):
        selected_options = []
        for option, var in self.options["visualize"].items():
            if var.get():
                selected_options.append(option)
        if "none" in selected_options and len(selected_options) > 1:
            raise ValueError("Cannot combine 'none' with other visualization options.")
        return selected_options

    def toggle_optimizer(self):
        """Toggle the visibility of the optimizer selection based on the 'Run Optimization' checkbox."""
        if self.options["optim"].get():
            self.optimizer_label.grid()  # Show the optimizer label
            self.optimizer_menu.grid()   # Show the optimizer menu
        else:
            self.optimizer_label.grid_remove()  # Hide the optimizer label
            self.optimizer_menu.grid_remove()   # Hide the optimizer menu

    def toggle_volume_fraction(self):
        if self.options["vf_enabled"].get():
            self.vf_entry.config(state="normal")
        else:
            self.vf_entry.config(state="disabled")

    def update_plot_mode_visibility(self):
        """Show or hide the plot mode options based on visualization selections."""
        # Check if any visualization option is selected
        any_selected = any(var.get() for var in self.options["visualize"].values())

        if any_selected:
            self.plot_mode_frame.grid()  # Show the plotting mode options
        else:
            self.plot_mode_frame.grid_remove()  # Hide the plotting mode options

    def update_latent_size(self, *args):
        """Update the latent variables and entries when the latent size changes."""
        size = self.options["latent_size"].get()
        self.options["latent"] = [tk.DoubleVar() for _ in range(size)]
        self.update_latent_entries()

    def update_latent_method(self, *args):
        """Update the visibility of latent entries based on the selected method."""
        self.update_latent_entries()

    def update_latent_entries(self):
        """Create or update the latent entries based on the latent size and method."""
        # Clear the current latent entries
        for widget in self.latent_frame.winfo_children():
            widget.destroy()

        method = self.options["latent_method"].get()
        if method == "manual":
            tk.Label(self.latent_frame, text="Latent Values").grid(row=0, column=0, sticky="w")
            latent_values_frame = tk.Frame(self.latent_frame)
            latent_values_frame.grid(row=0, column=1, columnspan=self.options["latent_size"].get())
            for i in range(self.options["latent_size"].get()):
                tk.Entry(
                    latent_values_frame,
                    textvariable=self.options["latent"][i],
                    width=5
                ).grid(row=0, column=i)
        elif method == "random":
            tk.Label(self.latent_frame, text="Latent vector will be randomly generated").grid(row=0, column=0, sticky="w")
        elif method == "preloaded":
            tk.Label(self.latent_frame, text="Latent vector will be loaded from 'best_latent_vector.npy'").grid(row=0, column=0, sticky="w")


if __name__ == "__main__":
    root = tk.Tk()
    app = SimulationConfigGUI(root)
    root.mainloop()
