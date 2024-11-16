import tkinter as tk
from tkinter import ttk


class OptimizationFrame(ttk.LabelFrame):
    def __init__(self, parent, options):
        super().__init__(parent, text="Optimization Options")
        self.options = options
        self.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        self.create_widgets()

    def create_widgets(self):
        tk.Checkbutton(
            self,
            text="Run Optimization",
            variable=self.options["optim"],
            command=self.toggle_optimizer_callback,
        ).grid(row=0, column=0, sticky="w")

        self.optimizer_label = tk.Label(self, text="Select Optimizer")
        self.optimizer_label.grid(row=1, column=0, sticky="w")
        self.optimizer_label.grid_remove()

        self.optimizer_menu = tk.OptionMenu(
            self, self.options["optimizer"], "bayesian", "cmaes"
        )
        self.optimizer_menu.grid(row=1, column=1)
        self.optimizer_menu.grid_remove()

    def toggle_optimizer_callback(self):
        """Toggle the visibility of the optimizer selection based
        on the 'Run Optimization' checkbox."""
        if self.options["optim"].get():
            self.optimizer_label.grid()  # Show the optimizer label
            self.optimizer_menu.grid()  # Show the optimizer menu
        else:
            self.optimizer_label.grid_remove()  # Hide the optimizer label
            self.optimizer_menu.grid_remove()  # Hide the optimizer menu
