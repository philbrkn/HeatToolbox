import tkinter as tk
from tkinter import ttk


class SolvingFrame(ttk.LabelFrame):
    def __init__(self, parent, options):
        super().__init__(parent, text="Solving Options")
        self.options = options
        self.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        self.create_widgets()

    def create_widgets(self):
        tk.Label(self, text="Mesh Resolution: Length / [] ").grid(
            row=0, column=0, sticky="w"
        )
        tk.Entry(self, textvariable=self.options["res"], width=10).grid(
            row=0, column=1
        )

        tk.Checkbutton(
            self,
            text="Enable Volume Fraction Control",
            variable=self.options["vf_enabled"],
            command=self.toggle_volume_fraction,
        ).grid(row=1, column=0, sticky="w")

        tk.Label(self, text="Volume Fraction").grid(
            row=2, column=0, sticky="w"
        )
        self.vf_entry = tk.Entry(self, textvariable=self.options["vf_value"], width=10)
        self.vf_entry.grid(row=2, column=1)
        self.toggle_volume_fraction()

        # Logging Toggle
        tk.Checkbutton(
            self,
            text="Enable Logging",
            variable=self.options["logging_enabled"],
        ).grid(row=3, column=0, sticky="w")

    def toggle_volume_fraction(self):
        if self.options["vf_enabled"].get():
            self.vf_entry.config(state="normal")
        else:
            self.vf_entry.config(state="disabled")
