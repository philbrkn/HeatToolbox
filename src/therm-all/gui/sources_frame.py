import tkinter as tk
import customtkinter as ctk


class SourcesFrame(ctk.CTkFrame):
    def __init__(self, parent, options, material_frame):
        super().__init__(parent)
        self.options = options
        self.material_frame = material_frame
        self.row = self.material_frame.row
        # self.grid(row=2, column=1, columnspan=4, sticky="w")
        # Store this frame in options so it can be accessed in utils_config.py
        self.options["sources_frame"] = self

        self.create_widgets()
        self.add_source_row()

    def create_widgets(self):
        ctk.CTkLabel(self.material_frame, text="Sources (Position, Heat)").grid(
            row=self.row, column=0, sticky="w"
        )
        self.sources_frame = ctk.CTkFrame(self.material_frame)
        self.sources_frame.grid(row=self.row + 1, column=1, columnspan=4, sticky="w")

        ctk.CTkButton(
            self.material_frame,
            text="Add Source",
            command=self.add_source_row,
        ).grid(row=self.row + 2, column=0, sticky="w")

    def add_source_row(self):
        source_row = {}
        row_frame = ctk.CTkFrame(self.sources_frame)
        row_frame.pack(anchor="w", pady=2)

        # Position Entry
        ctk.CTkLabel(row_frame, text="Position:").pack(side="left")
        position_var = tk.DoubleVar(value=0.5)
        ctk.CTkEntry(row_frame, textvariable=position_var, width=50).pack(side="left")

        # Heat Value Entry
        ctk.CTkLabel(row_frame, text="Heat:").pack(side="left")
        heat_var = tk.DoubleVar(value=80)
        ctk.CTkEntry(row_frame, textvariable=heat_var, width=50).pack(side="left")

        # Remove Button
        remove_button = ctk.CTkButton(
            row_frame, text="Remove", command=lambda: self.remove_source_row(source_row)
        )
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
