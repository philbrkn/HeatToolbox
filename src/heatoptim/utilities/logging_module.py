import os
import json
import datetime
import numpy as np
from mpi4py import MPI


class LoggingModule:
    def __init__(self, config):
        self.rank = MPI.COMM_WORLD.rank
        self.config = config

        # Determine the log directory name
        if config.log_name:
            # Use the provided log name
            self.log_dir = os.path.join("logs", config.log_name)
        else:
            # Generate a unique directory name based on timestamp and descriptors
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if config.optim:
                optimizer = config.optimizer
            else:
                optimizer = "solve-only"
            num_sources = (
                len(config.source_positions)
                if hasattr(config, "source_positions")
                else "unknown"
            )
            self.log_dir = f"logs/{timestamp}_{optimizer}_sources_{num_sources}"

        # Create the log directory if it doesn't exist
        if self.rank == 0:
            os.makedirs(self.log_dir, exist_ok=True)
            # Initialize logging
            self._initialize_logging()

    def _initialize_logging(self):
        # Save the configuration file in the log directory
        config_path = os.path.join(self.log_dir, "config.json")
        # Only create/write config.json if it doesn't already exist
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump(self.config.config, f, indent=4)
        else:
            print(f"Skipping config.json write; already exists at {config_path}")

        # Save a command-line script to reproduce the run
        run_script_path = os.path.join(self.log_dir, "run_simulation.sh")
        command = f"python -m heatoptim.main --config {config_path}"
        with open(run_script_path, "w") as f:
            f.write(command + "\n")

        # Save the HPC script if generated
        if hasattr(self.config, "hpc_script_content"):
            hpc_script_path = os.path.join(self.log_dir, "hpc_run.sh")
            with open(hpc_script_path, "w") as f:
                f.write(self.config.hpc_script_content)

    def log_generation_data(self, generation, data):
        """Log generation data during optimization."""
        if self.rank == 0:
            with open(os.path.join(self.log_dir, "generation_data.txt"), "a") as f:
                timestamp = datetime.datetime.now().isoformat()
                f.write(f"{timestamp} - Generation {generation}: {data}\n")

    def save_optimized_latent_vectors(self, latent_vectors):
        """Save the best latent vectors from optimization."""
        if self.rank == 0:
            # Save the latent vectors in the optimization_results subdirectory
            opt_results_dir = os.path.join(self.log_dir, "optimization_results")
            os.makedirs(opt_results_dir, exist_ok=True)
            latent_vectors_path = os.path.join(opt_results_dir, "best_latent_vectors.npy")
            np.save(latent_vectors_path, latent_vectors)

    def save_image(self, fig, filename, latent_vector_name=None):
        """Save visualization images in the appropriate directory."""
        if self.rank == 0:
            if latent_vector_name:
                # Create a subdirectory for the specific latent vector
                vis_dir = os.path.join(self.log_dir, "visualization", latent_vector_name)
            else:
                vis_dir = os.path.join(self.log_dir, "visualization")
            os.makedirs(vis_dir, exist_ok=True)
            filepath = os.path.join(vis_dir, filename)
            fig.savefig(filepath)
            print(f"Visualization saved to {filepath}")
