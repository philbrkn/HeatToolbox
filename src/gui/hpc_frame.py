import tkinter as tk
from tkinter import ttk


class HPCFrame(ttk.LabelFrame):
    def __init__(self, parent, options):
        super().__init__(parent, text="HPC Script Options")
        self.options = options
        self.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        self.create_widgets()

    def create_widgets(self):
        row = 0
        # Number of Nodes
        tk.Label(self, text="Number of Nodes").grid(row=row, column=0, sticky="w")
        tk.Entry(self, textvariable=self.options["nodes"], width=10).grid(row=row, column=1, sticky="w")
        row += 1

        # Number of CPUs per Node
        tk.Label(self, text="CPUs per Node (ncpus)").grid(row=row, column=0, sticky="w")
        tk.Entry(self, textvariable=self.options["ncpus"], width=10).grid(row=row, column=1, sticky="w")
        row += 1

        # Memory per Node
        tk.Label(self, text="Memory per Node (GB)").grid(row=row, column=0, sticky="w")
        tk.Entry(self, textvariable=self.options["mem"], width=10).grid(row=row, column=1, sticky="w")
        row += 1

        # Walltime
        tk.Label(self, text="Walltime (HH:MM:SS)").grid(row=row, column=0, sticky="w")
        tk.Entry(self, textvariable=self.options["walltime"], width=10).grid(row=row, column=1, sticky="w")
        row += 1

        # Timeout
        # tk.Label(self, text="Timeout (HH:MM:SS)").grid(row=row, column=0, sticky="w")
        # tk.Entry(self, textvariable=self.options["timeout"], width=10).grid(row=row, column=1, sticky="w")
        # row += 1

        # Parallelization Toggle
        tk.Checkbutton(
            self,
            text="Enable Parallelization (MPI)",
            variable=self.options["parallelize"],
            command=self.toggle_mpi_options
        ).grid(row=row, column=0, sticky="w")
        row += 2

        # MPI Processes per Node
        self.mpiprocs_label = tk.Label(self, text="MPI Processes per Node (mpiprocs)")
        self.mpiprocs_entry = tk.Entry(self, textvariable=self.options["mpiprocs"], width=10)#.grid(row=row, column=1, sticky="w")
        row += 1

        # Conda Environment Path
        tk.Label(self, text="Conda Environment Path").grid(row=row, column=0, sticky="w")
        tk.Entry(self, textvariable=self.options["conda_env_path"], width=30).grid(row=row, column=1, sticky="w")
        row += 1

        # Conda Environment Name
        tk.Label(self, text="Conda Environment Name").grid(row=row, column=0, sticky="w")
        tk.Entry(self, textvariable=self.options["conda_env_name"], width=20).grid(row=row, column=1, sticky="w")

        row = 0
        # HPC user, host, and directory
        tk.Label(self, text="HPC User").grid(row=row, column=2, sticky="w")
        tk.Entry(self, textvariable=self.options["hpc_user"], width=10).grid(row=row, column=3, sticky="w")
        row += 1
        tk.Label(self, text="HPC Host").grid(row=row, column=2, sticky="w")
        tk.Entry(self, textvariable=self.options["hpc_host"], width=10).grid(row=row, column=3, sticky="w")
        row += 1
        tk.Label(self, text="HPC Directory").grid(row=row, column=2, sticky="w")
        tk.Entry(self, textvariable=self.options["hpc_dir"], width=10).grid(row=row, column=3, sticky="w")

        # Initially hide MPI options if parallelization is not enabled
        self.toggle_mpi_options()

    def toggle_mpi_options(self):
        if self.options["parallelize"].get():
            self.mpiprocs_label.grid(row=6, column=0, sticky="w")
            self.mpiprocs_entry.grid(row=6, column=1, sticky="w")
        else:
            self.mpiprocs_label.grid_remove()
            self.mpiprocs_entry.grid_remove()
