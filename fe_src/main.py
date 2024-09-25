# main.py

import numpy as np
from fenicsx_solver import solve_pde  # Import the mesh from fenics_solver
from vae_model import VAE, z_to_img, Flatten, UnFlatten
# from utils import save_for_modulus
import torch
import time

# Set parameters
num_samples = 1  # Number of simulations
latent_size = 4     # Size of the latent vector
device = torch.device("cpu")  # Change to "cuda" if using GPU

# Load the pre-trained VAE model
model = VAE()
model = torch.load('./model/model', map_location=torch.device('cpu'))
# model.load_state_dict(torch.load('model.pth', map_location=device))
model = model.to(device)
model.eval()


for i in range(num_samples):
    # measure time for simulation
    time_start = time.time()

    # Step 1: Generate image using VAE
    z = np.random.randn(latent_size)
    img = z_to_img(z, model, device)

    # Step 2: Solve PDE using the image
    u, p = solve_pde(img)

    # Step 3: Save the output data
    filename = f'simulation_data_{i}.h5'
    # save_for_modulus(filename, mesh, p, u)

    # Optional: Visualize and verify
    # plot(p)
    # plt.show()

    time_end = time.time()
    print(f"Completed simulation {i+1}/{num_samples} in {time_end - time_start:.2f} seconds")
