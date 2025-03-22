import customtkinter as ctk
import tkinter as tk
import os


class MaterialFrame(ctk.CTkFrame):
    def __init__(self, parent, options):
        super().__init__(parent)
        self.options = options
        self.grid(row=0, column=1, padx=10, pady=5, sticky="nsew")
        self.create_widgets()

    def create_widgets(self):
        # Load CMA-ES Config Button
        row = 0
        # select solver type, gke vs fourier
        ctk.CTkLabel(self, text="Solver Type").grid(row=row, column=0, sticky="w")
        self.solver_type_menu = ctk.CTkOptionMenu(
            self,
            variable=self.options["solver_type"],
            values=["gke", "fourier", "joule"],
            command=self.update_solver_type,
        ).grid(row=row, column=1, padx=5, pady=5, sticky="ew")

        # write knudsen number
        row += 1
        ctk.CTkLabel(self, text="Knudsen Number").grid(row=row, column=0, sticky="w")
        ctk.CTkEntry(self, textvariable=self.options["knudsen"], width=100).grid(
            row=row, column=1
        )

        # latent size
        row += 1
        ctk.CTkLabel(self, text="Latent Size").grid(row=row, column=0, sticky="w")
        ctk.CTkOptionMenu(
            self,
            variable=self.options["latent_size"],
            values=["2", "4", "8", "16"],  # Convert numbers to strings
            command=self.update_latent_size,
        ).grid(row=row, column=1, padx=5, pady=5, sticky="ew")

        # Latent Method
        row += 1
        ctk.CTkLabel(self, text="Latent Method").grid(row=row, column=0, sticky="w")
        ctk.CTkOptionMenu(
            self,
            variable=self.options["latent_method"],
            values=["manual", "random", "preloaded"],
            command=self.update_latent_method,
        ).grid(row=row, column=1, padx=5, pady=5, sticky="ew")
        self.options["latent_method"].set("preloaded")  # Set a default value

        # Latent Values Entry (will be updated based on latent size and method)
        row += 1
        self.latent_frame = ctk.CTkFrame(self)
        self.latent_frame.grid(row=row, column=0, columnspan=2, sticky="w")
        self.update_latent_method()  # Initialize latent entries

        row += 1
        # "Enable Symmetry" Checkbutton (moved to row 3)
        ctk.CTkCheckBox(
            self, text="Enable Symmetry", variable=self.options["symmetry"]
        ).grid(row=row, column=0, sticky="w")

        row += 1
        # "Run with Blank Image" Checkbutton (moved to row 4)
        ctk.CTkCheckBox(
            self, text="Run with Blank Image", variable=self.options["blank"]
        ).grid(row=row, column=0, sticky="w")

        row += 1
        # "Sources (Position, Heat)" Label (moved to row 5)
        ctk.CTkLabel(self, text="Sources (Position, Heat)").grid(
            row=row, column=0, sticky="w"
        )
        self.row = row

    def update_latent_size(self, choice):
        """Update the latent variables and entries when the latent size changes."""
        size = int(choice)  # Convert from string to integer
        self.options["latent_size"].set(size)  # Update the variable
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
            ctk.CTkLabel(self.latent_frame, text="Latent Values").grid(
                row=0, column=0, sticky="w"
            )
            latent_values_frame = ctk.CTkFrame(self.latent_frame)
            latent_values_frame.grid(
                row=0, column=1, columnspan=self.options["latent_size"].get()
            )
            for i in range(self.options["latent_size"].get()):
                ctk.CTkEntry(
                    latent_values_frame, textvariable=self.options["latent"][i], width=100
                ).grid(row=0, column=i)
        elif method == "random":
            ctk.CTkLabel(
                self.latent_frame, text="Latent vector will be randomly generated"
            ).grid(row=0, column=0, sticky="w")
        elif method == "preloaded":
            ctk.CTkLabel(
                self.latent_frame,
                text="Latent vector will be loaded from 'best_latent_vector.npy'",
            ).grid(row=0, column=0, sticky="w")

    def update_solver_type(self, *args):
        solver_type = self.options["solver_type"].get()

        # Update defaults based on solver type
        if solver_type == "gke":
            self.options["knudsen"].set(1.0)
            # Additional GKE-specific defaults
        elif solver_type == "fourier":
            self.options["knudsen"].set(None)  # Not used for Fourier
            # Fourier-specific defaults
        elif solver_type == "joule":
            self.options["knudsen"].set(None)  # Not applicable
            # Add Joule-specific defaults here

        # Trigger updates for dependent widgets
        self.update_latent_entries()
