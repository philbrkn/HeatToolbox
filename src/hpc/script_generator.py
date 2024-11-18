
def generate_hpc_script(config, config_path):
    """
    Generate the HPC script based on a configuration dictionary.

    Args:
        config (dict): Configuration dictionary containing HPC and simulation parameters.
        config_path (str): Path to the configuration JSON file.

    Returns:
        str: The content of the HPC script.
    """
    # Start the script with the shebang and PBS directives
    script_lines = ["#!/bin/bash"]

    # Select resources
    select_line = f"#PBS -l select={config['nodes']}:ncpus={config['ncpus']}:mem={config['mem']}gb"

    # If parallelization is enabled, add mpiprocs
    if config.get("parallelize", False):
        select_line += f":mpiprocs={config['mpiprocs']}"

    script_lines.append(select_line)

    # Add walltime
    script_lines.append(f"#PBS -l walltime={config['walltime']}")

    # Load necessary modules and activate the conda environment
    script_lines.extend(
        [
            "",
            "module load tools/prod",
            f'eval "$({config["conda_env_path"]} shell.bash hook)"',
            f"conda activate {config['conda_env_name']}",
            "",
            "# Copy input file to $TMPDIR",
            "cp -r $HOME/BTE-NO $TMPDIR/",
            "",
            "cd $TMPDIR/BTE-NO",
            "",
        ]
    )

    # Generate the command line using the config path
    command = f"python src/main.py --config {config_path}"

    # Calculate timeout in hours
    if config.get("timeout"):
        timeout_parts = list(map(int, config["timeout"].split(":")))
        script_timeout = (
            timeout_parts[0] + timeout_parts[1] / 60 + timeout_parts[2] / 3600
        )
    else:
        timeout_parts = list(map(int, config["walltime"].split(":")))
        script_timeout = (
            timeout_parts[0] + timeout_parts[1] / 60 + timeout_parts[2] / 3600
        )
        script_timeout *= 0.98  # Use 98% of walltime as timeout buffer

    timeout_line = f"timeout {round(script_timeout, 2)}h {command}"

    # Use mpirun if parallelization is enabled
    if config.get("parallelize", False):
        mpirun_command = (
            f"mpirun -np {config['mpiprocs'] * config['nodes']} {timeout_line}"
        )
        script_lines.append("# Run application with MPI")
        script_lines.append(mpirun_command)
    else:
        script_lines.append("# Run application")
        script_lines.append(timeout_line)

    script_lines.extend(
        [
            "",
            "# Copy required files back",
            "cp -r $TMPDIR/BTE-NO/logs/* $HOME/BTE-NO/logs/",
            "",
        ]
    )

    # Join the script lines into a single string
    script_content = "\n".join(script_lines)

    return script_content
