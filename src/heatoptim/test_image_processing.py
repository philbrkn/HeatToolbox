'''
test functions in the image_processing.py file
'''
import numpy as np
from heatoptim.utilities.image_processing import generate_images
from heatoptim.utilities.vae_module import load_vae_model
from heatoptim.config.sim_config import SimulationConfig
import matplotlib.pyplot as plt
import os


def test_generate_images(config, latent_size, num_vizs=1):
    """
    Test the generate_images function by generating random latent vectors,
    creating images, and plotting them.

    Args:
        config_path (str): Path to the configuration file.
        latent_size (int): Size of the latent vector.
    """
    model = load_vae_model(rank=0, z_dim=latent_size)
    for _ in range(num_vizs):
        # Generate random latent vectors
        latent_vectors = [np.random.randn(latent_size) for _ in range(len(config.source_positions))]
        config.symmetry = False
        # Generate images from latent vectors
        img_list = generate_images(config, latent_vectors, model)
        # Plot the generated images
        for i, img in enumerate(img_list):
            plt.figure()
            plt.imshow(img, cmap='gray')
            plt.title(f"Generated Image {i+1}")
            plt.axis('off')
        plt.show()
        config.symmetry = True
        # Generate images from latent vectors
        img_list = generate_images(config, latent_vectors, model)
        # Plot the generated images
        for i, img in enumerate(img_list):
            plt.figure()
            plt.imshow(img, cmap='gray')
            plt.title(f"Generated Image {i+1}")
            plt.axis('off')
        plt.show()


if __name__ == '__main__':
    ITER_PATH = "logs/_ONE_SOURCE_NSGA2/test_nsga_10mpi_z8"
    config_file = os.path.join(ITER_PATH, "config.json")
    config = SimulationConfig(config_file)
    test_generate_images(config=config, latent_size=4, num_vizs=10)
