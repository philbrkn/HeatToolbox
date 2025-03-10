import customtkinter as ctk


class OptimizationFrame(ctk.CTkFrame):
    def __init__(self, parent, options):
        super().__init__(parent)
        self.options = options

        self.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure(99, weight=1)  # Prevent bottom empty space

        self.create_widgets()
        # Automatically update UI when "optim" checkbox is changed
        self.options["optim"].trace_add("write", self._on_optim_changed)

    def create_widgets(self):
        """Create UI elements with CustomTkinter."""
        # Run Optimization Checkbox
        self.optim_checkbox = ctk.CTkCheckBox(
            self, text="Run Optimization", variable=self.options["optim"]
        )
        self.optim_checkbox.grid(row=0, column=0, columnspan=2, pady=(5, 10), sticky="w")

        # Optimizer Selection
        self.optimizer_label = ctk.CTkLabel(self, text="Select Optimizer:")
        self.optimizer_menu = ctk.CTkOptionMenu(
            self, variable=self.options["optimizer"], values=["bayesian", "cmaes", "nsga2"]
        )

        # Population Size
        self.popsize_label = ctk.CTkLabel(self, text="Population Size:")
        self.popsize_entry = ctk.CTkEntry(self, textvariable=self.options["popsize"], width=100)

        # Bounds
        self.bounds_lower_label = ctk.CTkLabel(self, text="Lower Bound:")
        self.bounds_lower_entry = ctk.CTkEntry(self, textvariable=self.options["bounds_lower"], width=100)

        self.bounds_upper_label = ctk.CTkLabel(self, text="Upper Bound:")
        self.bounds_upper_entry = ctk.CTkEntry(self, textvariable=self.options["bounds_upper"], width=100)

        # Number of Iterations
        self.n_iter_label = ctk.CTkLabel(self, text="Number of Iterations:")
        self.n_iter_entry = ctk.CTkEntry(self, textvariable=self.options["n_iter"], width=100)

        self.maxtime_label = ctk.CTkLabel(self, text="Max Time (seconds):")
        self.maxtime_entry = ctk.CTkEntry(self, textvariable=self.options["maxtime"], width=100)
        
        # Initially hide all optimizer-related options
        self.hide_optimizer_options()

    def toggle_optimizer_options(self):
        """Show or hide optimizer options based on the Run Optimization checkbox."""
        if self.options["optim"].get():
            # Show optimizer-related options
            self.optimizer_label.grid(row=1, column=0, sticky="w")
            self.optimizer_menu.grid(row=1, column=1, padx=5, pady=5, sticky="w")
            self.popsize_label.grid(row=2, column=0, sticky="w")
            self.popsize_entry.grid(row=2, column=1, sticky="w")
            self.bounds_lower_label.grid(row=3, column=0, sticky="w")
            self.bounds_lower_entry.grid(row=3, column=1, sticky="w")
            self.bounds_upper_label.grid(row=4, column=0, sticky="w")
            self.bounds_upper_entry.grid(row=4, column=1, sticky="w")
            self.n_iter_label.grid(row=5, column=0, sticky="w")
            self.n_iter_entry.grid(row=5, column=1, sticky="w")

            self.maxtime_label.grid(row=6, column=0, sticky="w")
            self.maxtime_entry.grid(row=6, column=1, sticky="w")
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
        self.maxtime_label.grid_remove()
        self.maxtime_entry.grid_remove()

    def _on_optim_changed(self, *args):
        """Callback whenever self.options['hpc_enabled'] is set (True/False)."""
        # Just reuse your existing logic:
        self.toggle_optimizer_options()
