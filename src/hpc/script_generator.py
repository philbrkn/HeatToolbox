from datetime import datetime


def generate_hpc_script(args, config_path):
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
    command = f"python src/main.py --config {config_path}"

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
    timeout_line = f"timeout {script_timeout}h {command}"

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
