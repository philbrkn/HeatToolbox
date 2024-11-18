import tkinter as tk


class SourcesFrame(tk.Frame):
    def __init__(self, parent, options, material_frame):
        super().__init__(parent)
        self.options = options
        self.material_frame = material_frame
        self.grid(row=6, column=1, columnspan=4, sticky="w")
        self.create_widgets()
        self.add_source_row()

    def create_widgets(self):
        tk.Label(self.material_frame, text="Sources (Position, Heat)").grid(
            row=5, column=0, sticky="w"
        )
        self.sources_frame = tk.Frame(self.material_frame)
        self.sources_frame.grid(row=6, column=1, columnspan=4, sticky="w")

        tk.Button(
            self.material_frame,
            text="Add Source",
            command=self.add_source_row,
        ).grid(row=7, column=0, sticky="w")

    def add_source_row(self):
        source_row = {}
        row_frame = tk.Frame(self.sources_frame)
        row_frame.pack(anchor="w", pady=2)

        # Position Entry
        tk.Label(row_frame, text="Position:").pack(side="left")
        position_var = tk.DoubleVar(value=0.5)
        tk.Entry(row_frame, textvariable=position_var, width=5).pack(side="left")

        # Heat Value Entry
        tk.Label(row_frame, text="Heat:").pack(side="left")
        heat_var = tk.DoubleVar(value=80)
        tk.Entry(row_frame, textvariable=heat_var, width=5).pack(side="left")

        # Remove Button
        remove_button = tk.Button(
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
