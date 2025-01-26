import customtkinter as ctk


class SolvingFrame(ctk.CTkFrame):
    def __init__(self, parent, options):
        super().__init__(parent)
        self.options = options
        self.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        self.create_widgets()

    def create_widgets(self):
        ctk.CTkLabel(self, text="Mesh Resolution: Length / [] ").grid(
            row=0, column=0, sticky="w"
        )
        ctk.CTkEntry(self, textvariable=self.options["res"], width=10).grid(
            row=0, column=1
        )

        ctk.CTkCheckBox(
            self,
            text="Enable Volume Fraction Control",
            variable=self.options["vf_enabled"],
            command=self.toggle_volume_fraction,
        ).grid(row=1, column=0, sticky="w")

        ctk.CTkLabel(self, text="Volume Fraction").grid(
            row=2, column=0, sticky="w"
        )
        self.vf_entry = ctk.CTkEntry(self, textvariable=self.options["vf_value"], width=10)
        self.vf_entry.grid(row=2, column=1)
        self.toggle_volume_fraction()

        # Logging Toggle
        ctk.CTkCheckBox(
            self,
            text="Enable Logging",
            variable=self.options["logging_enabled"],
        ).grid(row=3, column=0, sticky="w")

        ctk.CTkLabel(self, text="Log File Name").grid(row=4, column=0, sticky="w")
        ctk.CTkEntry(self, textvariable=self.options["log_name"], width=20).grid(row=4, column=1, sticky="w")

    def toggle_volume_fraction(self):
        if self.options["vf_enabled"].get():
            self.vf_entry.configure(state="normal")  #  Use configure() instead of config()
        else:
            self.vf_entry.configure(state="disabled")  #  Correct way to disable