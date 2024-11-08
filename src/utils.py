# utils.py


def generate_command_line(args):
    command = "python src/main.py"

    if args.optim:
        command += " --optim"
        if args.optimizer:
            command += f" --optimizer {args.optimizer}"
    if args.latent and any(args.latent):
        command += " --latent " + " ".join(map(str, args.latent))
    if args.symmetry:
        command += " --symmetry"
    if args.blank:
        command += " --blank"
    if args.sources:
        command += " --sources " + " ".join(map(str, args.sources))
    if args.res:
        command += f" --res {args.res}"
    if args.visualize:
        command += " --visualize " + " ".join(args.visualize)
    if args.vf is not None:
        command += f" --vf {args.vf}"
    if args.plot_mode:
        command += f" --plot-mode {args.plot_mode}"
    return command
