from .solving_frame import SolverFrame


class JouleFrame(ttk.LabelFrame):
    def __init__(self, parent, options):
        super().__init__(parent, text="Joule Heating Configuration")
        self.options = options
        self.grid(row=0, column=1, padx=10, pady=5, sticky="nsew")
        self.create_widgets()