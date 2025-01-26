import tkinter as tk
from tkinter import ttk


class OptimizationFrame(ttk.LabelFrame):
    def __init__(self, parent, options):
        super().__init__(parent, text="Optimization Options")
        self.options = options
        self.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        self.create_widgets()

        self.options["optim"].trace_add("write", self._on_optim_changed)

    def create_widgets(self):
        # Run Optimization Checkbox
        tk.Checkbutton(
            self,
            text="Run Optimization",
            variable=self.options["optim"]
            # command=self.toggle_optimizer_options,
        ).grid(row=0, column=0, sticky="w")

        # Optimizer Selection
        self.optimizer_label = tk.Label(self, text="Select Optimizer")
        self.optimizer_menu = tk.OptionMenu(
            self, self.options["optimizer"], "bayesian", "cmaes"
        )

        # Population Size
        self.popsize_label = tk.Label(self, text="Population Size:")
        self.popsize_entry = tk.Entry(self, textvariable=self.options["popsize"], width=10)

        # Bounds
        self.bounds_lower_label = tk.Label(self, text="Lower Bound:")
        self.bounds_lower_entry = tk.Entry(self, textvariable=self.options["bounds_lower"], width=10)

        self.bounds_upper_label = tk.Label(self, text="Upper Bound:")
        self.bounds_upper_entry = tk.Entry(self, textvariable=self.options["bounds_upper"], width=10)

        # Number of Iterations
        self.n_iter_label = tk.Label(self, text="Number of Iterations:")
        self.n_iter_entry = tk.Entry(self, textvariable=self.options["n_iter"], width=10)

        # Initially hide all optimizer-related options
        self.hide_optimizer_options()

    def toggle_optimizer_options(self):
        """Show or hide optimizer options based on the Run Optimization checkbox."""
        if self.options["optim"].get():
            # Show optimizer-related options
            self.optimizer_label.grid(row=1, column=0, sticky="w")
            self.optimizer_menu.grid(row=1, column=1, sticky="w")
            self.popsize_label.grid(row=2, column=0, sticky="w")
            self.popsize_entry.grid(row=2, column=1, sticky="w")
            self.bounds_lower_label.grid(row=3, column=0, sticky="w")
            self.bounds_lower_entry.grid(row=3, column=1, sticky="w")
            self.bounds_upper_label.grid(row=4, column=0, sticky="w")
            self.bounds_upper_entry.grid(row=4, column=1, sticky="w")
            self.n_iter_label.grid(row=5, column=0, sticky="w")
            self.n_iter_entry.grid(row=5, column=1, sticky="w")
        else:
            self.hide_optimizer_options()

    def hide_optimizer_options(self):
        """Hide all optimizer-related options."""
        self.optimizer_label.grid_remove()
        self.optimizer_menu.grid_remove()
        self.popsize_label.grid_remove()
        self.popsize_entry.grid_remove()
        self.bounds_lower_label.grid_remove()
        self.bounds_lower_entry.grid_remove()
        self.bounds_upper_label.grid_remove()
        self.bounds_upper_entry.grid_remove()
        self.n_iter_label.grid_remove()
        self.n_iter_entry.grid_remove()
    
    def _on_optim_changed(self, *args):
        """Callback whenever self.options['hpc_enabled'] is set (True/False)."""
        # Just reuse your existing logic:
        self.toggle_optimizer_options()
