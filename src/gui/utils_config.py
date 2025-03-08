import tkinter as tk
from tkinter import messagebox
import os
import json


def initialize_options():
    """Initialize all Tkinter options with default values."""
    return {
        "optim": tk.BooleanVar(),
        "optimizer": tk.StringVar(value="cmaes"),
        "latent_size": tk.IntVar(value=4),
        "latent_method": tk.StringVar(value="preloaded"),
        "latent": [tk.DoubleVar() for _ in range(4)],
        "symmetry": tk.BooleanVar(),
        "blank": tk.BooleanVar(),
        "sources": [],
        "res": tk.DoubleVar(value=12),
        "visualize": {},
        "vf_enabled": tk.BooleanVar(value=True),
        "vf_value": tk.DoubleVar(value=0.2),
        "plot_mode": tk.StringVar(value="screenshot"),
        "logging_enabled": tk.BooleanVar(value=True),
        # HPC options
        "nodes": tk.IntVar(value=1),
        "ncpus": tk.IntVar(value=4),
        "mem": tk.IntVar(value=8),
        "walltime": tk.StringVar(value="03:00:00"),
        "parallelize": tk.BooleanVar(value=False),
        "mpiprocs": tk.IntVar(value=4),
        "conda_env_path": tk.StringVar(value="~/miniforge3/bin/conda"),
        "conda_env_name": tk.StringVar(value="fenicsx_torch"),
        "log_name": tk.StringVar(value=""),
        "hpc_user": tk.StringVar(value="pt721"),
        "hpc_host": tk.StringVar(value="login.cx3.hpc.ic.ac.uk"),
        "hpc_dir": tk.StringVar(value="~/BTE-NO"),
        # CMAES
        "popsize": tk.IntVar(value=8),  # New parameter for population size
        "bounds_lower": tk.DoubleVar(value=-2.5),  # New lower bound for CMA-ES
        "bounds_upper": tk.DoubleVar(value=2.5),  # New upper bound for CMA-ES
        "n_iter": tk.IntVar(value=100),  # New parameter for the number of iterations
        "hpc_enabled": tk.BooleanVar(value=False),
        "load_cmaes_config": tk.BooleanVar(value=False),
        # "timeout": tk.StringVar(),
        "knudsen": tk.DoubleVar(value=1),
        "solver_type": tk.StringVar(value="gke"),
        "maxtime": tk.IntVar(value=3600),  # Max optimization time for NSGA2
    }


def get_config_dict(options):
    """Extract configuration from Tkinter options."""
    return {
        "optim": options["optim"].get(),
        "optimizer": options["optimizer"].get(),
        "latent_size": options["latent_size"].get(),
        "latent_method": options["latent_method"].get(),
        "latent": [val.get() for val in options["latent"]],
        "symmetry": options["symmetry"].get(),
        "blank": options["blank"].get(),
        "sources": parse_sources(options["sources"]),
        "res": options["res"].get(),
        "visualize": {k: v.get() for k, v in options["visualize"].items()},  # Convert BooleanVars to actual values
        "vf_enabled": options["vf_enabled"].get(),
        "vf_value": options["vf_value"].get(),
        "plot_mode": options["plot_mode"].get(),
        "logging_enabled": options["logging_enabled"].get(),
        "nodes": options["nodes"].get(),
        "ncpus": options["ncpus"].get(),
        "mem": options["mem"].get(),
        "walltime": options["walltime"].get(),
        # "timeout": options["timeout"].get(),
        "parallelize": options["parallelize"].get(),
        "mpiprocs": options["mpiprocs"].get(),
        "conda_env_path": options["conda_env_path"].get(),
        "conda_env_name": options["conda_env_name"].get(),
        "log_name": options["log_name"].get(),
        "hpc_user": options["hpc_user"].get(),
        "hpc_host": options["hpc_host"].get(),
        "hpc_dir": options["hpc_dir"].get(),
        "popsize": options["popsize"].get(),  # New parameter
        "bounds": [options["bounds_lower"].get(), options["bounds_upper"].get()],  # New parameter
        "n_iter": options["n_iter"].get(),  # New parameter
        "hpc_enabled": options["hpc_enabled"].get(),
        "load_cmaes_config": options["load_cmaes_config"].get(),
        "knudsen": options["knudsen"].get(),
        "solver_type": options["solver_type"].get(),
    }


def save_config(options, log_dir):
    """Save the current configuration to a JSON file."""
    print("SAVING CONFIG")
    config = get_config_dict(options)
    os.makedirs(log_dir, exist_ok=True)
    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    messagebox.showinfo("Success", f"Configuration saved to {config_path}")


def load_config(options, file_path):
    """Load configuration from a JSON file and update the options."""
    with open(file_path, "r") as f:
        config = json.load(f)
    set_options_from_config(options, config)
    messagebox.showinfo("Success", f"Configuration loaded from {file_path}")


def set_options_from_config(options, config):
    """Update Tkinter options with values from a configuration."""
    for key, value in config.items():
        if key in options:
            if key == "sources":  # Special handling for sources
                update_sources_from_config(options, value)
            elif isinstance(options[key], list):  # Handle lists like "latent"
                if isinstance(value, list):
                    for i, val in enumerate(value):
                        if i < len(options[key]) and hasattr(options[key][i], 'set'):
                            options[key][i].set(val)
                else:
                    raise ValueError(f"Expected a list for key '{key}', but got {type(value)}.")
            elif isinstance(options[key], dict):  # Handle dictionaries like "visualize"
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if subkey in options[key]:
                            if hasattr(options[key][subkey], 'set'):
                                options[key][subkey].set(subvalue)
                            else:
                                options[key][subkey] = subvalue
                        else:
                            # Handle new keys that might not exist in options
                            options[key][subkey] = tk.BooleanVar(value=subvalue)
                else:
                    raise ValueError(f"Expected a dict for key '{key}', but got {type(value)}.")
            elif hasattr(options[key], 'set'):  # For Tkinter variables
                options[key].set(value)
            else:  # Handle non-Tkinter variables (e.g., plain dicts or other types)
                options[key] = value
        else:
            # Handle new keys that might not exist in options
            options[key] = value


def parse_sources(tk_sources_dict):
    sources_list = []
    for source in tk_sources_dict:
        pos = source["position"].get()
        heat = source["heat"].get()
        if pos == "" or heat == "":
            continue  # Skip empty entries
        try:
            pos = float(pos)
            heat = float(heat)
            if pos < 0 or pos > 1:
                raise ValueError("Source positions must be between 0 and 1 (normalized).")
            sources_list.extend([pos, heat])
        except ValueError as e:
            raise ValueError(f"Invalid source input: {e}")
    return sources_list if sources_list else None


def update_sources_from_config(options, sources_list):
    """Ensure the correct number of source rows are added to match the loaded config."""
    sources_frame = options["sources_frame"]  # This should be a reference to the GUI's SourcesFrame

    # Clear all existing sources
    for source in options["sources"]:
        sources_frame.remove_source_row(source)
    
    options["sources"].clear()  # Reset source list
    
    # Add sources from the loaded config
    for i in range(0, len(sources_list), 2):  # Assuming sources are stored as [pos, heat, pos, heat, ...]
        if i + 1 < len(sources_list):
            position, heat = sources_list[i], sources_list[i + 1]
            sources_frame.add_source_row()  # Add a new source row dynamically

            # Update the last added source with correct values
            options["sources"][-1]["position"].set(position)
            options["sources"][-1]["heat"].set(heat)
