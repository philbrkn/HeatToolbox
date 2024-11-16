

def main(config):
    # Load VAE model
    comm = MPI.COMM_WORLD
    rank = MPI.COMM_WORLD.rank
    model = load_vae_model(rank)

    # Initialize logging module
    logger = LoggingModule(config) if config.logging_enabled else None

    # Mesh generation
    mesh_generator = MeshGenerator(config)
    time1 = MPI.Wtime()

    comm.barrier()
    msh, cell_markers, facet_markers = dolfinx.io.gmshio.read_from_msh("domain_with_extrusions.msh", MPI.COMM_SELF, gdim=2)

    # create solver instance
    solver = Solver(msh, facet_markers, config)
    if config.optim:
        # Run optimization
        optimizer = None
        if config.optimizer == "cmaes":
            optimizer = CMAESModule(
                solver, model, torch.device("cpu"), rank, config, logger=logger
            )
            best_z_list = optimizer.optimize(n_iter=100)  # Adjust iterations as needed
        elif config.optimizer == "bayesian":
            optimizer = BayesianModule(
                solver, model, torch.device("cpu"), rank, config, logger=logger
            )
            best_z_list = optimizer.optimize(init_points=10, n_iter=100)

        latent_vectors = best_z_list
        # Optional: Save the best_z to a file for future solving
        if rank == 0:
            np.save("best_latent_vector.npy", best_z_list)

    # Run solving based on provided latent vector
    else:
        # Handle latent vector based on the selected method
        latent_vectors = []
        if config.latent_method == "manual":
            # Use the latent vector provided in args.latent
            z = np.array(config.latent)
            if len(z) != config.latent_size:
                raise ValueError(f"Expected latent vector of size {config.latent_size}, got {len(z)}.")
            latent_vectors = [z] * len(config.source_positions)
        elif config.latent_method == "random":
            # Generate random latent vectors
            for _ in range(len(config.source_positions)):
                z = np.random.randn(config.latent_size)
                latent_vectors.append(z)
        elif config.latent_method == "preloaded":
            # Load latent vectors from file
            try:
                best_z_list = np.load("best_latent_vector.npy", allow_pickle=True)
                print("Opening best vector from file")
                latent_vectors = best_z_list
            except FileNotFoundError:
                raise FileNotFoundError("No saved latent vectors found. Please provide a valid file.")

    # Generate image from latent vector
    img_list = []
    for z in latent_vectors:
        if config.blank:
            img = np.zeros((128, 128))
        else:
            # Ensure z is reshaped correctly if needed
            img = z_to_img(z.reshape(1, -1), model, config.vol_fraction)
        img_list.append(img)

    # Apply symmetry to each image if enabled
    if config.symmetry:
        img_list = [img[:, : img.shape[1] // 2] for img in img_list]

    if "pregamma" in config.visualize:
        plot_image_list(img_list, config, logger=logger)
    # Solve the image using the solver
    avg_temp_global = solver.solve_image(img_list)
    time2 = MPI.Wtime()
    if rank == 0:
        print(f"Average temperature: {avg_temp_global} K")
        print(f"Time taken to solve: {time2 - time1:.3f} seconds")

    # Optional Post-processing
    if "none" not in config.visualize:
        post_processor = PostProcessingModule(rank, config, logger=logger)
        post_processor.postprocess_results(solver.U, solver.msh, solver.gamma)

    if config.optim:
        # After optimization completes
        final_results = {
            "average_temperature": avg_temp_global,  # Replace with actual metric
            "best_latent_vector": best_z_list  # Best solution from the optimizer
        }
        if logger:
            logger.log_results(final_results)
    else:
        # Results if running without optimization
        results = {
            "average_temperature": avg_temp_global,
            "runtime": time2 - time1
        }
        if logger:
            logger.log_results(results)

