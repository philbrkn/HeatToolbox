# import os
import numpy as np


def read_last_latent_vector(cma_log_file, z_dim, num_sources):
    """
    Read the last latent vector(s) from the CMA-ES log file.

    Parameters:
    - cma_log_file (str): Path to the CMA-ES log file.
    - z_dim (int): Dimension of the latent vector.
    - num_sources (int): Number of sources (latent vectors).

    Returns:
    - latent_vectors (list of numpy arrays): List containing the latent vectors.
    """
    with open(cma_log_file, 'r') as f:
        lines = f.readlines()

    # Filter out comment lines (starting with '%')
    data_lines = [line for line in lines if not line.strip().startswith('%') and line.strip()]
    if not data_lines:
        raise ValueError("No data lines found in the CMA-ES log file.")

    last_line = data_lines[-1]
    # Split the line into components
    parts = last_line.strip().split()
    # The latent vector starts from the 6th element (index 5)
    latent_values = [float(x) for x in parts[5:]]

    # Check if the length matches expected size
    expected_length = z_dim * num_sources
    if len(latent_values) != expected_length:
        raise ValueError(f"Expected {expected_length} latent values, but got {len(latent_values)}.")

    # Split the latent_values into individual latent vectors
    latent_vectors = []
    for i in range(num_sources):
        start_idx = i * z_dim
        end_idx = (i + 1) * z_dim
        latent_vector = np.array(latent_values[start_idx:end_idx])
        latent_vectors.append(latent_vector)

    return latent_vectors
