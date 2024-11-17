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
        "timeout": tk.StringVar(value="2:55:00"),
        "parallelize": tk.BooleanVar(value=False),
        "mpiprocs": tk.IntVar(value=4),
        "conda_env_path": tk.StringVar(value="~/miniforge3/bin/conda"),
        "conda_env_name": tk.StringVar(value="fenicsx_torch"),
        "log_name": tk.StringVar(value=""),
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
        "sources": options["sources"],
        "res": options["res"].get(),
        "visualize": options["visualize"],
        "vf_enabled": options["vf_enabled"].get(),
        "vf_value": options["vf_value"].get(),
        "plot_mode": options["plot_mode"].get(),
        "logging_enabled": options["logging_enabled"].get(),
        "nodes": options["nodes"].get(),
        "ncpus": options["ncpus"].get(),
        "mem": options["mem"].get(),
        "walltime": options["walltime"].get(),
        "timeout": options["timeout"].get(),
        "parallelize": options["parallelize"].get(),
        "mpiprocs": options["mpiprocs"].get(),
        "conda_env_path": options["conda_env_path"].get(),
        "conda_env_name": options["conda_env_name"].get(),
        "log_name": options["log_name"].get(),
    }


def save_config(options, log_name=None):
    """Save the current configuration to a JSON file."""
    config = get_config_dict(options)
    log_name = log_name or config.get("log_name") or "default_log"
    log_dir = os.path.join("logs", log_name)
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
            if isinstance(options[key], list):  # For "latent" and similar lists
                for i, val in enumerate(value):
                    options[key][i].set(val)
            else:
                options[key].set(value)