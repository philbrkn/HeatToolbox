import customtkinter as ctk

class HPCFrame(ctk.CTkFrame):
    def __init__(self, parent, options):
        super().__init__(parent)
        self.options = options

        self.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        self.create_widgets()

        # Attach a trace to update GUI when hpc_enabled changes
        self.options["hpc_enabled"].trace_add("write", self._on_hpc_enabled_changed)

    def create_widgets(self):
        # Enable HPC Checkbox
        self.hpc_checkbox = ctk.CTkCheckBox(
            self,
            text="Enable HPC",
            variable=self.options["hpc_enabled"]
            # command=self.toggle_hpc_options,
        )
        self.hpc_checkbox.grid(row=0, column=0, sticky="w")

        # HPC Options
        self.nodes_label = ctk.CTkLabel(self, text="Number of Nodes:")
        self.nodes_entry = ctk.CTkEntry(self, textvariable=self.options["nodes"], width=50)

        self.ncpus_label = ctk.CTkLabel(self, text="CPUs per Node (ncpus):")
        self.ncpus_entry = ctk.CTkEntry(self, textvariable=self.options["ncpus"], width=5)

        self.mem_label = ctk.CTkLabel(self, text="Memory per Node (GB):")
        self.mem_entry = ctk.CTkEntry(self, textvariable=self.options["mem"], width=5)

        self.walltime_label = ctk.CTkLabel(self, text="Walltime (HH:MM:SS):")
        self.walltime_entry = ctk.CTkEntry(self, textvariable=self.options["walltime"], width=5)

        self.parallelize_checkbutton = ctk.CTkCheckBox(
            self,
            text="Enable Parallelization (MPI)",
            variable=self.options["parallelize"],
            command=self.toggle_mpi_options,
        )

        self.mpiprocs_label = ctk.CTkLabel(self, text="MPI Processes per Node (mpiprocs):")
        self.mpiprocs_entry = ctk.CTkEntry(self, textvariable=self.options["mpiprocs"], width=10)

        self.conda_path_label = ctk.CTkLabel(self, text="Conda Environment Path:")
        self.conda_path_entry = ctk.CTkEntry(self, textvariable=self.options["conda_env_path"], width=30)

        self.conda_name_label = ctk.CTkLabel(self, text="Conda Environment Name:")
        self.conda_name_entry = ctk.CTkEntry(self, textvariable=self.options["conda_env_name"], width=20)

        self.hpc_user_label = ctk.CTkLabel(self, text="HPC User:")
        self.hpc_user_entry = ctk.CTkEntry(self, textvariable=self.options["hpc_user"], width=10)

        self.hpc_host_label = ctk.CTkLabel(self, text="HPC Host:")
        self.hpc_host_entry = ctk.CTkEntry(self, textvariable=self.options["hpc_host"], width=10)

        self.hpc_dir_label = ctk.CTkLabel(self, text="HPC Directory:")
        self.hpc_dir_entry = ctk.CTkEntry(self, textvariable=self.options["hpc_dir"], width=10)

        # Initially hide all HPC-related options
        self.hide_hpc_options()

    def toggle_hpc_options(self):
        """Show or hide HPC options based on the Enable HPC checkbox."""
        if self.options["hpc_enabled"].get():
            # Show HPC-related options
            self.nodes_label.grid(row=1, column=0, sticky="w")
            self.nodes_entry.grid(row=1, column=1, sticky="w")
            self.ncpus_label.grid(row=2, column=0, sticky="w")
            self.ncpus_entry.grid(row=2, column=1, sticky="w")
            self.mem_label.grid(row=3, column=0, sticky="w")
            self.mem_entry.grid(row=3, column=1, sticky="w")
            self.walltime_label.grid(row=4, column=0, sticky="w")
            self.walltime_entry.grid(row=4, column=1, sticky="w")
            self.parallelize_checkbutton.grid(row=5, column=0, sticky="w")
            self.toggle_mpi_options()  # Show or hide MPI options based on the checkbox
            self.hpc_user_label.grid(row=1, column=2, sticky="w")
            self.hpc_user_entry.grid(row=1, column=3, sticky="w")
            self.hpc_host_label.grid(row=2, column=2, sticky="w")
            self.hpc_host_entry.grid(row=2, column=3, sticky="w")
            self.hpc_dir_label.grid(row=3, column=2, sticky="w")
            self.hpc_dir_entry.grid(row=3, column=3, sticky="w")
            self.conda_path_label.grid(row=4, column=2, sticky="w")
            self.conda_path_entry.grid(row=4, column=3, sticky="w")
            self.conda_name_label.grid(row=5, column=2, sticky="w")
            self.conda_name_entry.grid(row=5, column=3, sticky="w")
        else:
            self.hide_hpc_options()

    def hide_hpc_options(self):
        """Hide all HPC-related options."""
        self.nodes_label.grid_remove()
        self.nodes_entry.grid_remove()
        self.ncpus_label.grid_remove()
        self.ncpus_entry.grid_remove()
        self.mem_label.grid_remove()
        self.mem_entry.grid_remove()
        self.walltime_label.grid_remove()
        self.walltime_entry.grid_remove()
        self.parallelize_checkbutton.grid_remove()
        self.mpiprocs_label.grid_remove()
        self.mpiprocs_entry.grid_remove()
        self.conda_path_label.grid_remove()
        self.conda_path_entry.grid_remove()
        self.conda_name_label.grid_remove()
        self.conda_name_entry.grid_remove()
        self.hpc_user_label.grid_remove()
        self.hpc_user_entry.grid_remove()
        self.hpc_host_label.grid_remove()
        self.hpc_host_entry.grid_remove()
        self.hpc_dir_label.grid_remove()
        self.hpc_dir_entry.grid_remove()

    def toggle_mpi_options(self):
        if self.options["parallelize"].get():
            self.mpiprocs_label.grid(row=6, column=0, sticky="w")
            self.mpiprocs_entry.grid(row=6, column=1, sticky="w")
        else:
            self.mpiprocs_label.grid_remove()
            self.mpiprocs_entry.grid_remove()

    def _on_hpc_enabled_changed(self, *args):
        """Callback whenever self.options['hpc_enabled'] is set (True/False)."""
        # Just reuse your existing logic:
        self.toggle_hpc_options()
