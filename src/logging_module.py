# logging_module.py
import os
import json
import datetime
from mpi4py import MPI

from utils import generate_command_line


class LoggingModule:
    def __init__(self, config):
        self.rank = MPI.COMM_WORLD.rank
        self.config = config
        self.log_dir = "logs"

        # Generate a unique directory name based on date, time, optimizer, and source count
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

        # Create a directory for logs, e.g., logs/20240101_123456_optimizer_cmaes_sources_3
        self.log_dir = f"logs/{timestamp}_{optimizer}_sources_{num_sources}"

        # Create a directory for logs if it doesn’t exist
        if self.rank == 0:
            os.makedirs(self.log_dir, exist_ok=True)
            # Initialize log files with configurations and other settings
            self._initialize_logging()

    def _initialize_logging(self):
        """Initialize logging by saving configurations and creating necessary files."""
        # Save the command-line arguments as a JSON file for replication
        with open(os.path.join(self.log_dir, "config.json"), "w") as f:
            json.dump(vars(self.config.args), f, indent=4)

        # Save a shell script that re-runs the current configuration
        command_line = generate_command_line(self.config.args)
        with open(os.path.join(self.log_dir, "run_command.sh"), "w") as f:
            f.write(command_line + "\n")

    def log_generation_data(self, generation, data):
        """Log generation data during optimization."""
        if self.rank == 0:
            with open(os.path.join(self.log_dir, "generation_data.txt"), "a") as f:
                timestamp = datetime.datetime.now().isoformat()
                f.write(f"{timestamp} - Generation {generation}: {data}\n")

    def save_image(self, fig, filename):
        """Save an image (e.g., a plot) if the plot mode is 'screenshot'."""
        if self.rank == 0:
            filepath = os.path.join(self.log_dir, filename)
            fig.savefig(filepath)
            print(f"Screenshot saved to {filepath}")
    
    def log_results(self, results):
        """Log final results after the simulation or optimization."""
        if self.rank == 0:
            with open(os.path.join(self.log_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=4)
            print("Results saved to results.json")
