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
            "optimizer": tk.StringVar(value="bayesian"),  # Default is bayesian
            "latent": [tk.DoubleVar() for _ in range(4)],
            "symmetry": tk.BooleanVar(),
            "blank": tk.BooleanVar(),
            "sources": [],  # List to hold source entries
            "res": tk.DoubleVar(),
            "visualize": {},  # Dictionary to hold visualization options
            "vf": tk.DoubleVar(value=0.2),
            "ssh_mode": tk.BooleanVar(),
            "interactive_viz": tk.BooleanVar(),
        }

        # Visualization options
        self.visualize_options = ["gamma", "temperature", "flux", "profiles", "all", "pregamma"]

        # Create the GUI components
        self.create_widgets()

    def create_widgets(self):
        # Optimization option
        tk.Checkbutton(
            self.root,
            text="Run Optimization",
            variable=self.options["optim"],
            command=self.toggle_optimizer,
        ).grid(row=0, column=0, sticky="w")

        # Optimizer choice (Dropdown) - initially hidden
        self.optimizer_label = tk.Label(self.root, text="Select Optimizer")
        self.optimizer_label.grid(row=1, column=0, sticky="w")
        self.optimizer_label.grid_remove()  # Hide it initially

        self.optimizer_menu = tk.OptionMenu(
            self.root, self.options["optimizer"], "bayesian", "cmaes"
        )
        self.optimizer_menu.grid(row=1, column=1)
        self.optimizer_menu.grid_remove()  # Hide it initially

        # Latent values input
        tk.Label(self.root, text="Latent Values (z1, z2, z3, z4)").grid(
            row=2, column=0, sticky="w"
        )
        latent_frame = tk.Frame(self.root)
        latent_frame.grid(row=2, column=1, columnspan=4)
        for i in range(4):
            tk.Entry(
                latent_frame, textvariable=self.options["latent"][i], width=5
            ).grid(row=0, column=i)

        # Symmetry option
        tk.Checkbutton(
            self.root, text="Enable Symmetry", variable=self.options["symmetry"]
        ).grid(row=3, column=0, sticky="w")

        # Blank option
        tk.Checkbutton(
            self.root, text="Run with Blank Image", variable=self.options["blank"]
        ).grid(row=4, column=0, sticky="w")

        # Sources input
        tk.Label(self.root, text="Sources (Position, Heat)").grid(
            row=5, column=0, sticky="w"
        )
        self.sources_frame = tk.Frame(self.root)
        self.sources_frame.grid(row=5, column=1, columnspan=4, sticky="w")

        # Add Source Button
        tk.Button(
            self.root, text="Add Source", command=self.add_source_row
        ).grid(row=6, column=0, sticky="w")

        # Resolution input (as a divider of LENGTH)
        tk.Label(self.root, text="Mesh Resolution: Length / [] ").grid(
            row=7, column=0, sticky="w"
        )
        tk.Entry(self.root, textvariable=self.options["res"], width=10).grid(
            row=7, column=1
        )

        # Visualization options
        tk.Label(self.root, text="Visualization Options").grid(
            row=8, column=0, sticky="nw"
        )
        self.visualize_frame = tk.Frame(self.root)
        self.visualize_frame.grid(row=8, column=1, sticky="w")

        for option in self.visualize_options:
            self.options["visualize"][option] = tk.BooleanVar(
                value=(option == "all")
            )
            tk.Checkbutton(
                self.visualize_frame, text=option, variable=self.options["visualize"][option]
            ).pack(anchor="w")

        # Volume Fraction input
        tk.Label(self.root, text="Volume Fraction").grid(
            row=9, column=0, sticky="w"
        )
        tk.Entry(self.root, textvariable=self.options["vf"], width=10).grid(
            row=9, column=1
        )

        # SSH Mode option
        tk.Checkbutton(
            self.root,
            text="SSH Mode",
            variable=self.options["ssh_mode"],
            command=self.update_visualization_options,
        ).grid(row=10, column=0, sticky="w")

        # Interactive Visualization option
        tk.Checkbutton(
            self.root,
            text="Interactive Visualization",
            variable=self.options["interactive_viz"],
            command=self.update_visualization_options,
        ).grid(row=11, column=0, sticky="w")

        # Submit and Generate buttons
        button_frame = tk.Frame(self.root)
        button_frame.grid(row=12, column=0, columnspan=2, pady=10)

        # "Run Simulation" button
        self.run_button = tk.Button(
            button_frame, text="Run Simulation", command=self.run_simulation
        )
        self.run_button.pack(side="left", padx=5)

        # "Generate Command" button
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

    def update_visualization_options(self):
        # Ensure mutual exclusivity between SSH Mode and Interactive Visualization
        if self.options["ssh_mode"].get():
            self.options["interactive_viz"].set(False)
        elif self.options["interactive_viz"].get():
            self.options["ssh_mode"].set(False)

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
            command = self.generate_command_line(args)

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
        args.latent = [val.get() for val in self.options["latent"]]
        args.optimizer = self.options["optimizer"].get()
        args.symmetry = self.options["symmetry"].get()
        args.blank = self.options["blank"].get()
        args.sources = self.parse_sources()
        # Pass resolution as the divider input
        args.res = self.options["res"].get() if self.options["res"].get() > 0 else None
        args.visualize = self.get_visualize_options()
        args.vf = self.options["vf"].get()
        args.ssh_mode = self.options["ssh_mode"].get()
        args.interactive_viz = self.options["interactive_viz"].get()

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
        if "all" in selected_options and len(selected_options) > 1:
            selected_options.remove("all")
        if "none" in selected_options and len(selected_options) > 1:
            raise ValueError("Cannot combine 'none' with other visualization options.")
        return selected_options

    def generate_command_line(self, args):
        command = "python src/main.py"

        if args.optim:
            command += " --optim"
            if args.optimizer:
                command += f" --optimizer {args.optimizer}"
        if args.symmetry:
            command += " --symmetry"
        if args.blank:
            command += " --blank"
        if args.latent and any(args.latent):
            command += f" --latent {' '.join(map(str, args.latent))}"
        if args.sources:
            command += f" --sources {' '.join(map(str, args.sources))}"
        if args.res:
            command += f" --res {args.res}"
        if args.visualize:
            command += f" --visualize {' '.join(args.visualize)}"
        if args.vf is not None:
            command += f" --vf {args.vf}"

        return command

    def toggle_optimizer(self):
        """Toggle the visibility of the optimizer selection based on the 'Run Optimization' checkbox."""
        if self.options["optim"].get():
            self.optimizer_label.grid()  # Show the optimizer label
            self.optimizer_menu.grid()   # Show the optimizer menu
        else:
            self.optimizer_label.grid_remove()  # Hide the optimizer label
            self.optimizer_menu.grid_remove()   # Hide the optimizer menu


if __name__ == "__main__":
    root = tk.Tk()
    app = SimulationConfigGUI(root)
    root.mainloop()
