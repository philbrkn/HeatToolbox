# utils.py


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
