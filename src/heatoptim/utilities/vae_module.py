# vae_model.py

import torch
from torch import nn
# from mpi4py import MPI
import heatoptim
import os

# REMOVE LATER?
image_size = 128
hidden_size = 1024
latent_size = 4

device = torch.device("cpu")  # Change to "cuda" if using GPU


def _load_vae_model_depreciated(rank, z_dim=latent_size):
    '''DEPRECIATE'''
    model = VAE(z_dim=z_dim)
    # model = torch.load("./model/model", map_location=torch.device("cpu"))
    model_path = f"models/128latent{z_dim}epochs200Alldict" if z_dim != 16 else f"models/128latent{z_dim}epochs500Alldict"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def load_vae_model(rank, z_dim=latent_size):
    model = VAE(z_dim=z_dim)

    # Determine the base directory of the installed package
    package_dir = os.path.dirname(heatoptim.__file__)

    # Choose the correct folder based on z_dim
    model_dict = f"128latent{z_dim}epochs200Alldict" if z_dim != 16 else f"128latent{z_dim}epochs500Alldict"

    # Construct the full path to the model file
    # Adjust "model.pth" to the actual filename you need to load
    model_path = os.path.join(package_dir, "models", model_dict)

    # Load the state dict from the file
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=hidden_size):
        return input.view(input.size(0), size, 1, 1)


class VAE(nn.Module):
    def __init__(self, image_channels=1, h_dim=hidden_size, z_dim=latent_size):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten(),
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # eps = torch.randn_like(std)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

