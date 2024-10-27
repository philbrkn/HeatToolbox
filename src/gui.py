import tkinter as tk
from tkinter import messagebox
from tkinter import ttk


class SimulationConfigGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulation Configuration")

        # Command-line equivalent options
        self.options = {
            "optim": tk.BooleanVar(),
            "latent": [tk.DoubleVar() for _ in range(4)],
            "symmetry": tk.BooleanVar(),
            "blank": tk.BooleanVar(),
            "sources": tk.StringVar(),
            "res": tk.DoubleVar(),
            "visualize": tk.StringVar(value="all"),
            "vf": tk.DoubleVar(value=0.2),
        }

        # Create the GUI components
        self.create_widgets()

    def create_widgets(self):
        # Optimization option
        tk.Checkbutton(self.root, text="Run Optimization", variable=self.options["optim"]).grid(row=0, column=0, sticky="w")

        # Latent values input
        tk.Label(self.root, text="Latent Values (z1, z2, z3, z4)").grid(row=1, column=0, sticky="w")
        latent_frame = tk.Frame(self.root)
        latent_frame.grid(row=1, column=1, columnspan=4)
        for i in range(4):
            tk.Entry(latent_frame, textvariable=self.options["latent"][i], width=5).grid(row=0, column=i)

        # Symmetry option
        tk.Checkbutton(self.root, text="Enable Symmetry", variable=self.options["symmetry"]).grid(row=2, column=0, sticky="w")

        # Blank option
        tk.Checkbutton(self.root, text="Run with Blank Image", variable=self.options["blank"]).grid(row=3, column=0, sticky="w")

        # Sources input
        tk.Label(self.root, text="Sources (Position, Heat)").grid(row=4, column=0, sticky="w")
        tk.Entry(self.root, textvariable=self.options["sources"], width=30).grid(row=4, column=1, columnspan=4, sticky="w")

        # Resolution input
        tk.Label(self.root, text="Mesh Resolution").grid(row=5, column=0, sticky="w")
        tk.Entry(self.root, textvariable=self.options["res"], width=10).grid(row=5, column=1)

        # Visualization option
        tk.Label(self.root, text="Visualization Options").grid(row=6, column=0, sticky="w")
        visualize_menu = ttk.Combobox(self.root, textvariable=self.options["visualize"])
        visualize_menu['values'] = ["none", "gamma", "temperature", "flux", "all", "pregamma"]
        visualize_menu.grid(row=6, column=1)

        # Volume Fraction input
        tk.Label(self.root, text="Volume Fraction").grid(row=7, column=0, sticky="w")
        tk.Entry(self.root, textvariable=self.options["vf"], width=10).grid(row=7, column=1)

        # Submit button
        tk.Button(self.root, text="Run Simulation", command=self.run_simulation).grid(row=8, column=0, columnspan=2)

    def run_simulation(self):
        try:
            # Extracting the options
            args = self.get_args()

            # Initialize SimulationConfig with the extracted args
            config = SimulationConfig(args)

            messagebox.showinfo("Configuration Loaded", "Simulation Configuration loaded successfully.")
            # You can now pass `config` to your main simulation function or proceed with the rest of your application.

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def get_args(self):
        # Mimic argparse.Namespace for easy passing to SimulationConfig
        class Args:
            pass

        args = Args()
        args.optim = self.options["optim"].get()
        args.latent = [val.get() for val in self.options["latent"]]
        args.symmetry = self.options["symmetry"].get()
        args.blank = self.options["blank"].get()
        args.sources = self.parse_sources(self.options["sources"].get())
        args.res = self.options["res"].get() if self.options["res"].get() else None
        args.visualize = [self.options["visualize"].get()]
        args.vf = self.options["vf"].get()

        return args

    @staticmethod
    def parse_sources(sources_string):
        if not sources_string:
            return None
        try:
            sources_list = list(map(float, sources_string.split()))
            if len(sources_list) % 2 != 0:
                raise ValueError("Each source must have a position and heat value. Provide pairs of values.")
            return sources_list
        except ValueError as e:
            raise ValueError(f"Invalid source input: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = SimulationConfigGUI(root)
    root.mainloop()
