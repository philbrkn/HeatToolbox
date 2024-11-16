import tkinter as tk
from tkinter import ttk


class VisualizationFrame(ttk.LabelFrame):
    def __init__(self, parent, options, visualize_options):
        super().__init__(parent, text="Visualization Options")
        self.options = options
        self.visualize_options = visualize_options
        self.grid(row=1, column=1, padx=10, pady=5, sticky="nsew")
        self.create_widgets()

    def create_widgets(self):
        # Visualization checkboxes
        tk.Label(self, text="Visualization Options").grid(row=0, column=0, sticky="nw")
        self.visualize_frame = tk.Frame(self)
        self.visualize_frame.grid(row=0, column=1, sticky="w")

        for option in self.visualize_options:
            self.options["visualize"][option] = tk.BooleanVar()
            tk.Checkbutton(
                self.visualize_frame,
                text=option,
                variable=self.options["visualize"][option],
                command=self.update_plot_mode_visibility,  # Attach callback
            ).pack(anchor="w")

        # Plotting mode
        tk.Label(self, text="Plotting Mode").grid(row=1, column=0, sticky="w")
        self.plot_mode_frame = tk.Frame(self)
        self.plot_mode_frame.grid(row=1, column=1, sticky="w")
        self.plot_mode_frame.grid_remove()  # Hidden initially

        tk.Radiobutton(
            self.plot_mode_frame,
            text="Save Screenshots",
            variable=self.options["plot_mode"],
            value="screenshot",
        ).grid(row=1, column=1, sticky="w")

        tk.Radiobutton(
            self.plot_mode_frame,
            text="Interactive Plotting",
            variable=self.options["plot_mode"],
            value="interactive",
        ).grid(row=2, column=1, sticky="w")

    def update_plot_mode_visibility(self):
        """Show or hide the plot mode options based on visualization selections."""
        # Check if any visualization option is selected
        any_selected = any(var.get() for var in self.options["visualize"].values())

        if any_selected:
            self.plot_mode_frame.grid()  # Show the plotting mode options
        else:
            self.plot_mode_frame.grid_remove()  # Hide the plotting mode options
