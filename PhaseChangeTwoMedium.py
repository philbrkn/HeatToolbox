
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
# import torchshow as ts
from scipy.ndimage import gaussian_filter, zoom
import numpy as np
# from skimage.filters import gaussian
from skimage.transform import resize

from PIL import Image

from io import BytesIO


from fenics import *
import fenics
from mshr import Circle, generate_mesh

from mshr import Rectangle, generate_mesh

import matplotlib.pyplot as plt
import ufl
import numpy as np
from deap import base, creator, tools, algorithms
from deap.tools import sortNondominated
from skimage.transform import resize

from skimage.morphology import binary_dilation, binary_erosion, remove_small_objects  

from datetime import datetime

import random
from math import *
import sys
import os

from PIL import Image

parameters["refinement_algorithm"] = "plaza_with_parent_facets"

device = torch.device("cpu")


image_size = 128
hidden_size = 1024
latent_size = 16
batch_size = 128
log_interval = 10

epochs = 500

n_interp = 8

which = 'All'


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
            Flatten()
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
        # return torch.normal(mu, std)
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


    
model = VAE().to(device)
model.load_state_dict(torch.load(f'./model/{latent_size}', map_location=torch.device('cpu')))
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def gaussian_blur(img, sigma=1):
    """Apply Gaussian blur using PyTorch."""
    # Create a 2D Gaussian kernel
    kernel_size = int(4 * sigma + 1)
    x = torch.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    gauss = torch.exp(-(x**2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    gauss_kernel = gauss[:, None] * gauss[None, :]
    gauss_kernel = gauss_kernel.expand(
        1, 1, *gauss_kernel.shape
    )  # Shape to match convolution input

    # Add batch and channel dimensions to the image for convolution
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()

    # Apply 2D convolution for Gaussian blur
    blurred_img = F.conv2d(img_tensor, gauss_kernel, padding=kernel_size // 2)

    # Remove extra dimensions
    return blurred_img.squeeze().numpy()


def apply_volume_fraction(img, vf):
    # Flatten the image to a 1D array
    img_flat = img.flatten()
    # Sort the pixels
    sorted_pixels = np.sort(img_flat)
    # Determine the threshold that will result in the desired volume fraction
    num_pixels = img_flat.size
    k = int((1 - vf) * num_pixels)
    threshold = (sorted_pixels[k] + sorted_pixels[k - 1]) / 2.0
    # Apply the threshold
    img_binary = (img >= threshold).astype(np.float32)
    return img_binary


def upsample_image(img, zoom_factor):
    return zoom(img, zoom_factor, order=3)  # Cubic interpolation


def z_to_img(z):
    z = torch.from_numpy(z).float().to(device)
    model.eval()
    with torch.no_grad():
        sample = model.decode(z).cpu()
        sample = torch.reshape(sample,(image_size, -1))
        sample = torch.from_numpy((sample.detach().numpy())).view(1, image_size, image_size)
    
    img = np.array(sample)


    img = gaussian_filter(sample, sigma=1)
    img = upsample_image(img, 3)

    img = (img + np.flip(img, axis=2)) / 2

    img = apply_volume_fraction(img, 0.15)

    return img

def filter(u_):
    """
    Applies ering to a field by solving af PDEfilter type PDE 
    based on the work of B.S. Lazarov and O. Sigmund DOI: 10.1002/nme.3072
    
    Inputs: u_ is the unfiltered field, r_ is the characteristic length
    Outputs: filtered field u_tilde
    """
    Vold = u_.function_space()            # Saving function space from u_
    VCG = FunctionSpace(mesh_, "CG", 1)    # Defining continous function space

    r = Vold.mesh().hmin()/20

    # Defining variational problem
    u_tilde = TrialFunction(VCG)
    vf = TestFunction(VCG)
    # HELMHOLTZ-TYPE PDE (u_ is projected to a continouos function space)
    F = u_tilde*vf*dx + r_**2*dot(grad(u_tilde), grad(vf))*dx - project(u_, VCG)*vf*dx(mesh_)
    # a: unknowns, L: knowns
    a, L = lhs(F), rhs(F)

    u_temp = Function(VCG)
    u_tilde = Function(Vold)

    # Compute solution and project onto inital function space
    solve(a == L, u_temp)
    u_tilde.assign(project(u_temp, Vold))

    return u_tilde

def exponential_mapping(z):
    return np.exp(z)


dx = dx(metadata={'quadrature_degree': 8})

def exponential_mapping(z):
    return np.exp(z)

def generate_circle_image(square_grid):
    square_grid = (square_grid > 0.5).astype(int)  # Ensure binary values
    square_grid = square_grid[::3, ::3]  # Downsample for faster computation
    size = square_grid.shape[0]  # Size of the square grid
    
    x = np.linspace(-1, 1, size)
    y_min, y_max = -np.pi / 4, np.pi / 1  # Define angular opening range
    y = np.linspace(y_min, y_max, size)
    X, Y = np.meshgrid(x, y)
    X, Y = -Y, X  # Swap axes to correct inversion
    
    Z = np.array([[X[i, j] + 1j * Y[i, j] if square_grid[i, j] == 1 else 0 
                   for j in range(size)] for i in range(size)])
    W = np.array([[exponential_mapping(Z[i, j]) if Z[i, j] != 0 else 0 
                   for j in range(size)] for i in range(size)])
    
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    mapped_points = W[np.nonzero(W)]
    mapped_points = np.concatenate((mapped_points, -mapped_points))
    ax.plot(mapped_points.real, mapped_points.imag, '.', color='white', markersize=5)
    ax.plot(mapped_points.imag, mapped_points.real, '.', color='white', markersize=5)
    
    max_radius = np.max(np.abs(mapped_points))
    circle = plt.Circle((0, 0), max_radius, color='black', zorder=0)
    ax.add_artist(circle)
    
    ax.set_facecolor('white')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal', 'box')
    fig.patch.set_facecolor('white')
    ax.axis('off')
    
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)
    
    return Image.fromarray(img)


def generate_q2(square_grid):
    square_grid = (square_grid > 0.5).astype(int)  # Ensure binary values
    square_grid = square_grid[::3, ::3]  # Downsample for faster computation
    size = square_grid.shape[0]  # Size of the square grid
    
    x = np.linspace(-1, 1, size)
    y_min, y_max = -np.pi / 3, np.pi / 3  # Define angular opening range
    y = np.linspace(y_min, y_max, size)
    X, Y = np.meshgrid(x, y)
    X, Y = -Y, X  # Swap axes to correct inversion
    
    Z = np.array([[X[i, j] + 1j * Y[i, j] if square_grid[i, j] == 1 else 0 
                   for j in range(size)] for i in range(size)])
    W = np.array([[exponential_mapping(Z[i, j]) if Z[i, j] != 0 else 0 
                   for j in range(size)] for i in range(size)])
    
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    mapped_points = W[np.nonzero(W)]
    # mapped_points = np.concatenate((mapped_points, -mapped_points))
    # ax.plot(mapped_points.real, mapped_points.imag, '.', color='white', markersize=5)
    ax.plot(mapped_points.imag, mapped_points.real, '.', color='white', markersize=5)
    
    max_radius = np.max(np.abs(mapped_points))
    circle = plt.Circle((0, 0), max_radius, color='black', zorder=0)
    ax.add_artist(circle)
    
    ax.set_facecolor('white')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal', 'box')
    fig.patch.set_facecolor('white')
    ax.axis('off')
    
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)
    
    return Image.fromarray(img)

def generate_q4(square_grid):
    square_grid = (square_grid > 0.5).astype(int)  # Ensure binary values
    square_grid = square_grid[::3, ::3]  # Downsample for faster computation
    size = square_grid.shape[0]  # Size of the square grid
    
    x = np.linspace(-1, 1, size)
    y_min, y_max = -np.pi / 3, np.pi / 3  # Define angular opening range
    y = np.linspace(y_min, y_max, size)
    X, Y = np.meshgrid(x, y)
    X, Y = -Y, X  # Swap axes to correct inversion
    
    Z = np.array([[X[i, j] + 1j * Y[i, j] if square_grid[i, j] == 1 else 0 
                   for j in range(size)] for i in range(size)])
    W = np.array([[exponential_mapping(Z[i, j]) if Z[i, j] != 0 else 0 
                   for j in range(size)] for i in range(size)])
    
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    mapped_points = -W[np.nonzero(W)]
    # mapped_points = np.concatenate((mapped_points, -mapped_points))
    # ax.plot(mapped_points.real, mapped_points.imag, '.', color='white', markersize=5)
    ax.plot(mapped_points.imag, mapped_points.real, '.', color='white', markersize=5)
    
    max_radius = np.max(np.abs(mapped_points))
    circle = plt.Circle((0, 0), max_radius, color='black', zorder=0)
    ax.add_artist(circle)
    
    ax.set_facecolor('white')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal', 'box')
    fig.patch.set_facecolor('white')
    ax.axis('off')
    
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)

    
    return Image.fromarray(img)

def generate_nonuniform_circle(grids):


    # Generate each quadrant using the respective grid
    q2 = generate_q2(grids[0])
    # q4 = generate_q4(grids[1])
    q4 = generate_q4(grids[1])


    q2 = crop_to_circle(q2)
    q4 = crop_to_circle(q4)

    # plt.imshow(q4)
    # plt.show()

    # plt.imshow(q4[250:-30,100:250])
    # plt.show()

    new_q4 = np.zeros_like(q4)
    new_q4[200:-70,100:260] = q4[240:-30,100:260]

    # plt.imshow(new_q4)
    # plt.show()

    q2 = resize(np.array(q2), (512, 512))
    q4 = resize(np.array(new_q4), (512, 512))


    # full_image = q1+q2+q3+q4
    full_image = q2 + q4

    full_image = full_image.reshape(512, 512, 1)

    # Convert the numpy array to an image if necessary
    final_image = Image.fromarray(full_image.squeeze())


    return final_image



def enforce_volume_fraction(img, vf):
    # Calculate the current volume fraction
    current_vf = np.sum(img) / img.size

    # print(current_vf)

    # Iteratively adjust the image to match the desired volume fraction
    adjusted_img = img.copy()
    niter = 0
    while (not np.isclose(current_vf, vf, atol=5e-2) or niter < 50):
        if current_vf < vf:
            # Increase the volume fraction by dilating the image
            adjusted_img = binary_dilation(adjusted_img)
        elif current_vf > vf:
            # Decrease the volume fraction by eroding the image
            adjusted_img = binary_erosion(adjusted_img)
        
        # Recalculate the volume fraction
        current_vf = np.sum(adjusted_img) / adjusted_img.size

        # print(current_vf)

        niter += 1
    # convert to uint8
    # adjusted_img = (adjusted_img * 255).astype(np.uint8)
    adjusted_img = remove_small_objects(adjusted_img, min_size=1000)
    
    return adjusted_img


def crop_to_circle(img):
    """Crop the image to tightly fit the circle."""
    img_gray = img.convert("L")  # Convert to grayscale
    img_np = np.array(img_gray)  # Convert to numpy array
    
    # Find non-white pixels (assuming white background)
    non_white = np.where(img_np < 255)
    if non_white[0].size == 0 or non_white[1].size == 0:
        return img  # Return original if nothing to crop
    
    # Determine bounding box
    y_min, y_max = non_white[0].min(), non_white[0].max()
    x_min, x_max = non_white[1].min(), non_white[1].max()
    
    # Crop image
    cropped_img = img.crop((x_min, y_min, x_max, y_max))

    if len(np.array(cropped_img).shape) == 2:
        circle = np.array(cropped_img)

    else:
        circle = np.array(cropped_img)[:,:,0]

    circle = enforce_volume_fraction(circle, 0.3)
    return circle


def SIMP(gamma, ks, kf, p):
    return ks + (kf - ks) * gamma * (1+p) / (gamma + p)

# PCM - Paraffin Wax
Pr = Constant(56.49)
Ra = Constant(19.15e3)
Ste = Constant(1/15)


T_r = Constant(1e-3)
r = Constant(0.025)

def phi(T):
    
    return 0.5*(1. + ufl.tanh((T_r - T)/r))

def np_phi(T):
        
    return 0.5*(1. + np.tanh((1e-2 - T)/0.025))

T_m = Constant(0.2)
delta_T = Constant(1)

def B(T):
    return 1/2 + (T-T_m) / delta_T

# def B(T):
#     return phi(T)

def A(T, gamma, factor=1):
    return 1e6 * (1 - B(T))**2 / (1e-3 + B(T)**3) + factor*ramp(gamma) 

def ramp(gamma, min = 0, max = 1e9, qa = 1):
    return min + (max - min)*gamma/(1+qa*(1-gamma))

def mu(T, gamma):
    return mu_L * (1 + A(T,gamma))

mu_L = Constant(1)
mu_S = Constant(1.e8)

# def mu(phi, gamma):
#      return mu_L + (mu_S - mu_L)*phi + mu_S*gamma


def C(gamma, Cmin = 0.2, Cmax = 1, qa = 100):
    return Cmin + (Cmax - Cmin)*(1-gamma)/(1+qa*gamma)

def K(gamma, Kmin = 1, Kmax = 200, qa = 100):
    return Kmin + (Kmax - Kmin)*gamma/(1+qa*(1-gamma))

def alpha(gamma, alphamin = 0.0, alphamax=1e6, qa=10):
    return alphamin + (alphamax - alphamin)*gamma/(1+qa*(1-gamma))


n = 32

outer_radius, inner_radius = 1.0, 0.35

vol_frac = (1 - (inner_radius/outer_radius)**2) * 0.5

print(vol_frac)

center = Point(0.0, 0.0)
outer_circle = Circle(center, outer_radius)
inner_circle = Circle(center, inner_radius)

domain = outer_circle - inner_circle

# Generate the mesh with a specified resolution
resolution = n  # Higher n -> finer mesh

# Define the FE_image class to map the image onto the mesh
class FE_image(UserExpression):
    def __init__(self, img, Nx, Ny, **kwargs):
        super().__init__(**kwargs)
        self.img = img
        self.Nx = Nx
        self.Ny = Ny

    def eval_cell(self, value, x, ufc_cell):
        # Map x, y coordinates to pixel indices
        px = int((x[0] + 1.0) / 2.0 * (self.Nx+1))  # Normalize x[0] to [0, Nx-1]
        py = int((x[1] + 1.0) / 2.0 * (self.Ny+1))  # Normalize x[1] to [0, Ny-1]
        # Safeguard for out-of-bound indices
        if 0 <= px < self.Nx and 0 <= py < self.Ny:
            value[0] = self.img[py, px]
        else:
            value[0] = 0.0

    def value_shape(self):
        return ()

# Define the function that maps the image to gamma
def img_to_gamma(img, mesh):
    Nx, Ny = img.shape[1], img.shape[0]

    img = img | np.flip(img, axis=1)
    # img = enforce_volume_fraction(img, 0.3)

    # Create an instance of FE_image
    y = FE_image(img, Nx, Ny, degree=1)

    # Define the FunctionSpace on the mesh
    V = FunctionSpace(mesh, 'P', 1)

    # Create a Function in the FunctionSpace
    u0 = Function(V)

    # Interpolate the image onto the mesh
    u0.interpolate(y)

    return u0


def project_gamma(gamma, slope = 8, point = 0.75):
    return ( np.tanh(slope*(gamma.vector() - point)) + np.tanh(slope*point) ) / ( np.tanh(slope*(1 - point)) + np.tanh(slope*point) )

def project_B(gamma, slope = 8, point = 0.5):
    return ( ufl.tanh(slope*(gamma - point)) + ufl.tanh(slope*point) ) / ( ufl.tanh(slope*(1 - point)) + ufl.tanh(slope*point) )

def filter(rho_n, r_min, projection = False, name="Filtered"):
    V = rho_n.function_space()

    rho = TrialFunction(V)
    w = TestFunction(V)

    a = (r_min**2)*inner(grad(rho), grad(w))*dx + rho*w*dx
    L = rho_n*w*dx

    A, b = assemble_system(a, L)
    rho = Function(V, name=name)
    solve(A, rho.vector(), b)

    if projection == True:
        rho_p = project_gamma(rho)
        rho.vector()[:] = rho_p

    return rho


def go(circle):
    initial_hot_wall_refinement_cycles = 3

    mesh = generate_mesh(domain, resolution)
    mesh_plot = generate_mesh(domain, 30)

    # markers = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    mesh = refine(mesh)


    class OuterCircle(SubDomain):
        def inside(self, x, on_boundary):
            return (x[0]**2 + x[1]**2 >= 0.4**2) and on_boundary  # Equation of unit circle

    class InnerCircle(SubDomain):
        def inside(self, x, on_boundary):
            return (x[0]**2 + x[1]**2 <= 0.4**2) and on_boundary  # Equation of unit circle
    
    class InnerCircleNeighborhood(UserExpression):
        def __init__(self, T_h, T_c, transition_width=0.05, R_inner = 0.27, **kwargs):
            super().__init__(**kwargs)
            self.T_h = T_h
            self.T_c = T_c
            self.transition_width = transition_width  # Controls the smoothness of transition
            self.R_inner = R_inner

        def eval(self, value, x):
            r = (x[0]**2 + x[1]**2)**0.5  # Compute distance from the origin
            if r <= self.R_inner:  # Near the outer circle
                value[0] = self.T_h
            elif r <= self.R_inner + self.transition_width:  # Transition zone
                alpha = (r - self.R_inner) / self.transition_width
                value[0] = (1 - alpha) * self.T_h + alpha * self.T_c
            else:  # Far from the outer circle
                value[0] = self.T_c

        def value_shape(self):
            return ()



    # Mark boundaries
    boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 9999)

    InnerCircle().mark(boundary_markers, 0)
    OuterCircle().mark(boundary_markers, 1)

    #Redefine boundary integration measure
    ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

    hot = 0.5
    T_h = Constant(hot)

    cold = -0.01
    T_c = Constant(cold)

    hconv = Constant(5)



    P1 = FiniteElement('P', mesh.ufl_cell(), 1)
    P2 = VectorElement('P', mesh.ufl_cell(), 2)
    mixed_element = MixedElement([P1, P2, P1])

    W = FunctionSpace(mesh, mixed_element)
    W_plot = FunctionSpace(mesh_plot, mixed_element)


    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)

    OuterCircle().mark(boundaries, 1)
    InnerCircle().mark(boundaries, 2)

    dS_cooling = ds(subdomain_id=1, subdomain_data=boundaries)
    dS_heating = ds(subdomain_id=2, subdomain_data=boundaries)

    q_elec = Expression("1+0.5*sin(200*pi*t)", t = 0, degree = 0)

    

    psi_p, psi_u, psi_T = TestFunctions(W)

    w = Function(W)

    p, u, T = split(w)


    W_u = W.sub(1)
    W_T = W.sub(2)



    gamma = img_to_gamma(circle, mesh)

    gamma = filter(gamma, r_min = mesh.hmin()*2, projection = True)
    # plt.clf()
    # plot(gamma).set_cmap('gray')
    # # plt.colorbar(plot(gamma))
    # plt.axis('off')
    # plt.savefig(f'./gamma.png')
    # plt.show()
    boundary_conditions = [DirichletBC(W.sub(1), (0., 0.), boundaries, 1), DirichletBC(W.sub(1), (0., 0.), boundaries, 2)]

    t_i = 1./2.**(initial_hot_wall_refinement_cycles - 1)

    T_n_expr = InnerCircleNeighborhood(T_h=hot, T_c=cold, degree=1)
    T_n_projected = interpolate(T_n_expr, W_T.collapse())  # Project onto the function space

    w_n = interpolate(
        Expression(("0.", "0.", "0.", "T_n"),
                    T_n=T_n_projected,
                    element=mixed_element),
        W)


    p_n, u_n, T_n = fenics.split(w_n)

    plot(T_n)
    plt.colorbar(plot(T_n))
    # plt.show()
    plt.clf()


    dt = Constant(10)
    u_t = (u - u_n)/dt
    T_t = (T - T_n)/dt
    phi_t = (phi(T) - phi(T_n))/dt

    f_B = Ra/Pr*T*Constant((0., -1.))



    mass = -psi_p*div(u)

    # momentum = dot(psi_u, u_t + dot(grad(u), u) + f_B) - div(psi_u)*p \
    #     + 2.*mu(phi(T), gamma)*inner(sym(grad(psi_u)), sym(grad(u))) + inner(psi_u*alpha(gamma), u)

    momentum = dot(psi_u, u_t + dot(grad(u), u) + f_B) - div(psi_u)*p \
        + 2.*mu(T, gamma)*inner(sym(grad(psi_u)), sym(grad(u))) + inner(psi_u*A(T, gamma, factor=1e3), u)

    # enthalpy = psi_T*(T_t - 1./Ste*phi_t) + dot(grad(psi_T), 1./Pr*grad(T) - T*u) # sign swap due to integration by parts
    enthalpy = C(gamma) * psi_T*(T_t - 1./Ste*phi_t) + dot(grad(psi_T), K(gamma) / Pr*grad(T) - T*u)
            
    F = (mass + momentum + enthalpy)*dx + hconv*(T-T_c)*psi_T*dS_cooling - Constant(5e-1)*q_elec*psi_T*dS_heating
    rho = Constant(1.e-7)

    F += -psi_p*rho*p*dx


    JF = derivative(F, w, TrialFunction(W))


        
    problem = NonlinearVariationalProblem(F, w, boundary_conditions, JF)


    M = phi_t*dx
    print(type(M))
    epsilon_M = 4.e-2


    solver = NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['absolute_tolerance'] = 1e-3
    solver.parameters['newton_solver']['relative_tolerance'] = 1e-3

    
    w.leaf_node().vector()[:] = w_n.leaf_node().vector()
    # w.vector()[:] = w_n.vector()

    t = 0
    q_elec.t = t

    solver.solve()


    # p, u, T = w.split()
    p, u, T = w.leaf_node().split()

    T_source, T_domain = [], []

    plt.clf()
    for timestep in range(10):
        t += float(dt)
        # mesh = refine(mesh)
        
        solver.solve()

        # p, u, T = w.split()
        p, u, T = w.leaf_node().split()

        mesh = w.function_space().mesh()

        boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        boundaries.set_all(0)
        OuterCircle().mark(boundaries, 1)
        dS = Measure('ds', domain = mesh, subdomain_data = boundaries) 

        T_source.append(assemble(T * dS(1)))
        T_domain.append(assemble(inner(grad(T), grad(T)) * dx))

        if timestep <= 9:
            time = datetime.now().strftime("%m-%d_%H:%M:%S")
            u.set_allow_extrapolation(True)
            T.set_allow_extrapolation(True)
            u_plot = interpolate(u, W_plot.sub(1).collapse())
            T_plot = interpolate(T, W_plot.sub(2).collapse())

            u_plot *=  project_B(B(T_plot))

            plt.clf()

            plt.set_cmap('Reds')
            plot(T).set_alpha(0.01)
            plt.colorbar(plot(T))

            plt.set_cmap('gray_r')
            plot(gamma).set_alpha(0.1)

            plt.set_cmap('viridis')
            plot(u_plot, mode = 'glyphs', scale = 10).set_alpha(1)
            plt.colorbar(plot(u_plot))
            plt.axis('off')

            plt.savefig(f"./TwoMedium/z{latent_size}/u_{time}.png")

            # store the plot
            # Get the current figure
            plt.axis('off')
            fig = plt.gcf()

            # Save the figure to a buffer
            buf = BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)

            # Open the buffer as an image and convert to NumPy array
            image = np.array(Image.open(buf))

            
            # plt.show()

        w_n.leaf_node().vector()[:] = w.leaf_node().vector()
        # w_n.vector()[:] = w.vector()


    T_source_mean = np.mean(T_source)
    T_domain_mean = np.mean(T_domain)

    melt = assemble(phi(T)*dx)

    T_source_var, T_domain_var = [], []

    for timestep in range(len(T_source)):
        T_source_var.append((T_source[timestep] - T_source_mean)**2)
        T_domain_var.append((T_domain[timestep] - T_domain_mean)**2)
    return melt, sum(T_domain_var) / len(T_domain_var), image

def evaluate(z):
    """
    Takes a flattened latent vector of size 4 * latent_size,
    splits it into 4 separate latent vectors, generates 4 images,
    creates a non-uniform circle, and evaluates it.
    """
    print(z.shape)  # Debugging

    z = z.reshape(2, latent_size)  # Split into 4 separate latent vectors

    img1 = z_to_img(z[0].reshape(1, latent_size))[0]
    img2 = z_to_img(z[1].reshape(1, latent_size))[0]


    # Create a non-uniform circle
    circle = generate_nonuniform_circle([img1, img2])
    print(type(circle))
    circle = crop_to_circle(circle)
    # print(type(circle))
    circle = np.array(circle)[:,:]
    


    # Evaluate the resulting image
    a = go(circle)

    return a[0], a[1], a[2]  # Multi-objective evaluation

pareto_front = np.load(f'./TwoMedium/latents/Pareto_{latent_size}.npy')

pareto_front = np.unique(pareto_front, axis=0)


# Save Pareto front solutions
pareto_solutions = np.array([ind for ind in pareto_front])

objectives = []

# Generate and save images from Pareto front
for i, ind in enumerate(pareto_solutions):
    pareto_z = np.array(ind).reshape(2, latent_size)
    pareto_imgs = [z_to_img(pareto_z[j].reshape(1, latent_size))[0] for j in range(2)]
    pareto_circle = crop_to_circle(generate_nonuniform_circle(pareto_imgs))

    # plt.imshow(pareto_circle, cmap='gray')
    # plt.savefig(f"./results/Pareto_{latent_size}_{i}.png")
    # plt.show()

    f1, f2, f3 = go(pareto_circle)

    print(f3.shape)

    time = datetime.now().strftime("%m-%d_%H:%M:%S")

    plt.clf()
    plt.imshow(f3)
    plt.axis('off')
    # write f1 and f2 on the image
    plt.text(10, 10, f'f1: {f1:.3f}, f2: {f2:.2f}', color='black')
    plt.savefig(f"./TwoMedium/z{latent_size}/u_{time}.png")
    # plt.show()

    # # save the objective values
    # objectives.append([f1, f2])

# save the objective values
# np.save(f'./TwoMedium/Pareto_{latent_size}_objectives.npy', objectives)