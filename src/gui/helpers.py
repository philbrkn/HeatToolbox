
def generate_hpc_script(args):
    # Start the script with the shebang and PBS directives
    script_lines = ["#!/bin/bash"]

    # Select resources
    select_line = f"#PBS -l select={args.nodes}:ncpus={args.ncpus}:mem={args.mem}gb"

    # If parallelization is enabled, add mpiprocs
    if args.parallelize:
        select_line += f":mpiprocs={args.mpiprocs}"

    script_lines.append(select_line)

    # Add walltime
    script_lines.append(f"#PBS -l walltime={args.walltime}")

    # Load necessary modules and activate conda environment
    script_lines.extend(
        [
            "",
            "module load tools/prod",
            f'eval "$({args.conda_env_path} shell.bash hook)"',
            f"conda activate {args.conda_env_name}",
            "",
            "# Copy input file to $TMPDIR",
            "cp -r $HOME/BTE-NO $TMPDIR/",
            "",
            "cd $TMPDIR/BTE-NO",
            "",
        ]
    )

    # Generate the command line using your existing function
    command = generate_command_line(args)

    # Add timeout before walltime
    # convert args.timeout, which is in HH:MM:SS, to H.[MM/60+SS/3600]
    if args.timeout:
        timeout_parts = list(map(int, args.timeout.split(":")))
        script_timeout = timeout_parts[0] + timeout_parts[1] / 60 + timeout_parts[2] / 3600
        script_timeout = round(script_timeout, 2)
    else:
        timeout_parts = list(map(int, args.walltime.split(":")))
        script_timeout = timeout_parts[0] + timeout_parts[1] / 60 + timeout_parts[2] / 3600
        script_timeout = round(0.98 * script_timeout, 2)
    timeout_line = f"timeout {script_timeout} {command}"

    # If parallelization is enabled, use mpirun
    if args.parallelize:
        mpirun_command = f"mpirun -np {args.total_procs} {timeout_line}"
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


def parse_sources(source_entries):
    sources_list = []
    for source in source_entries:
        pos = source["position"].get()
        heat = source["heat"].get()
        if pos == "" or heat == "":
            continue  # Skip empty entries
        try:
            pos = float(pos)
            heat = float(heat)
            if pos < 0 or pos > 1:
                raise ValueError(
                    "Source positions must be between 0 and 1 (normalized)."
                )
            sources_list.extend([pos, heat])
        except ValueError as e:
            raise ValueError(f"Invalid source input: {e}")
    return sources_list if sources_list else None


def get_visualize_options(visualize_vars):
    selected_options = []
    for option, var in visualize_vars.items():
        if var.get():
            selected_options.append(option)
    if "none" in selected_options and len(selected_options) > 1:
        raise ValueError("Cannot combine 'none' with other visualization options.")
    return selected_options


def generate_command_line(args):
    command = "python src/main.py"

    # Optimization options
    if args.optim:
        command += " --optim"
        if args.optimizer:
            command += f" --optimizer {args.optimizer}"

    # Latent vector options
    if args.latent_method == "manual" and args.latent and any(args.latent):
        command += " --latent " + " ".join(map(str, args.latent))
    if args.latent_size != 4:  # Only add if different from default
        command += f" --latent-size {args.latent_size}"
    if args.latent_method != "manual":  # Only add if different from default
        command += f" --latent-method {args.latent_method}"

    # Physical and simulation options
    if args.symmetry:
        command += " --symmetry"
    if args.blank:
        command += " --blank"
    if args.sources:
        # Ensure sources are paired correctly
        sources_str = " ".join(map(str, args.sources))
        command += f" --sources {sources_str}"
    if args.res:
        command += f" --res {args.res}"
    if args.visualize:
        command += " --visualize " + " ".join(args.visualize)
    if args.vf is not None:
        command += f" --vf {args.vf}"

    # Plotting mode (only if visualization options are selected)
    if args.visualize:
        command += f" --plot-mode {args.plot_mode}"

    # Logging option
    if args.no_logging:
        command += " --no-logging"

    return command
