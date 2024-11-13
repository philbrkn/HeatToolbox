'''
defining boundary conditions via gmsh
'''
import os
import time

import numpy as np
import ufl
import gmsh
from mpi4py import MPI
from petsc4py import PETSc
import torch

from dolfinx import fem, io, la, mesh
import basix.ufl

import dolfinx.fem.petsc  # ghost import

from utils import gather_mesh_on_rank0, alpha_function
from vae_model import VAE, z_to_img, Flatten, UnFlatten

os.environ["OMP_NUM_THREADS"] = "1"  # Use one thread per process

# Physical constants
C = PETSc.ScalarType(1.0)  # Slip parameter for fully diffusive boundaries
T_ISO = PETSc.ScalarType(0.0)  # Isothermal temperature, K
Q_L = 50
Q = PETSc.ScalarType(Q_L)

ELL_SI = PETSc.ScalarType(196e-9)  # Non-local length, m
ELL_DI = PETSc.ScalarType(196e-8)
KAPPA_SI = PETSc.ScalarType(141.0)  # W/mK, thermal conductivity
KAPPA_DI = PETSc.ScalarType(600.0)
LENGTH = 0.439e-6  # Characteristic length, adjust as necessary

L_X = 5 * LENGTH
L_Y = 2.5 * LENGTH
SOURCE_WIDTH = LENGTH * 0.5
SOURCE_HEIGHT = LENGTH * 0.25 * 0.5
R_TOL = LENGTH * 1e-3
RESOLUTION = LENGTH / 10  # Adjust mesh resolution as needed

# MPI initialization
comm = MPI.COMM_WORLD
RANK = comm.Get_rank()
SIZE = comm.Get_size()

# LOAD VAE #
# Set parameters
num_samples = 1  # Number of simulations
latent_size = 4     # Size of the latent vector
device = torch.device("cpu")  # Change to "cuda" if using GPU
if RANK == 0:
    # Load the pre-trained VAE model
    model = VAE()
    model = torch.load('./model/model', map_location=torch.device('cpu'))
    # model.load_state_dict(torch.load('model.pth', map_location=device))
    model = model.to(device)
    model.eval()


def isothermal_boundary(x):
    return np.isclose(x[1], 0.0, rtol=R_TOL)


def source_boundary(x):
    return np.isclose(x[1], L_Y + SOURCE_HEIGHT, rtol=R_TOL)


def slip_boundary(x):
    on_left = np.isclose(x[0], 0.0)
    on_right = np.isclose(x[0], L_X)
    on_top = np.isclose(x[1], L_Y)
    return np.logical_or.reduce((on_left, on_right, on_top))


def main():
    print(f"Rank {RANK} of {SIZE} is running.")

    if RANK == 0:
        time1 = time.time()
    else:
        time1 = None

    # Create mesh
    msh, cell_markers, facet_markers = create_mesh(L_X, L_Y, SOURCE_WIDTH, SOURCE_HEIGHT, RESOLUTION)

    if RANK == 0:
        # z = np.random.randn(latent_size)
        # print(f"z: {z}")
        z = np.array([0.57852902, -0.75218827,  0.07094553, -0.40801165])
        img = z_to_img(z, model, device)
    else:
        img = None

    img = comm.bcast(img, root=0)

    # Define function spaces and functions
    W, U, dU, v, s = define_function_spaces(msh)

    # set gamma as a function equal to 0
    V_gamma = fem.functionspace(msh, ("CG", 1))
    gamma = fem.Function(V_gamma)  # Material field

    gamma_expr = img_to_gamma_expression(img, msh)
    gamma.interpolate(gamma_expr)

    alpha = alpha_function(gamma, msh)
    # facet_tags = tag_boundary_facets_based_on_gamma(gamma, msh)
    # facet_tag_function = map_facet_tags_to_function(facet_tags, msh)
    
    # Post-process results
    # import pyvista as pv
    # grid = pv.UnstructuredGrid(*dolfinx.plot.vtk_mesh(msh, 2))
    # grid.point_data["gamma"] = alpha.x.array.real
    # grid.set_active_scalars("gamma")

    # # Plot the scalar field
    # plotter = pv.Plotter()
    # plotter.add_mesh(grid, show_edges=True)
    # plotter.view_xy()
    # plotter.show()

    postprocess_results(U, msh, img, gamma, alpha, time1)


def map_facet_tags_to_function(facet_tags, msh):
    """
    Maps facet tags to a function defined on the mesh vertices.

    Parameters:
    - facet_tags: dolfinx.MeshTags object containing facet tags.
    - msh: dolfinx mesh.

    Returns:
    - node_function: fem.Function with values 1.0 at nodes connected to tagged facets, 0.0 elsewhere.
    """
    tdim = msh.topology.dim
    fdim = tdim - 1  # Facet dimension

    # Create a scalar function space on the mesh
    V = fem.functionspace(msh, ("CG", 1))

    # Create a function to hold the node values
    node_function = fem.Function(V)
    node_values = node_function.vector.array
    node_values.fill(0.0)  # Initialize all node values to 0.0

    # Get the indices of the tagged facets
    tagged_facets = facet_tags.indices[facet_tags.values == 1]

    # Create connectivity between facets and vertices
    msh.topology.create_connectivity(fdim, 0)
    f_to_v = msh.topology.connectivity(fdim, 0)

    # Create connectivity between vertices and facets (needed for efficient mapping)
    msh.topology.create_connectivity(0, fdim)
    v_to_f = msh.topology.connectivity(0, fdim)

    # Get the dof map
    dofs = V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts

    # Create an array to keep track of which nodes are connected to tagged facets
    node_markers = np.zeros(dofs, dtype=np.int32)

    # Loop over the tagged facets and mark connected nodes
    for facet in tagged_facets:
        vertices = f_to_v.links(facet)
        node_markers[vertices] = 1  # Mark nodes connected to tagged facets

    # Assign the node markers to the function values
    node_values[:] = node_markers.astype(np.float64)

    # Update the ghost values
    node_function.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                     mode=PETSc.ScatterMode.FORWARD)

    return node_function


def tag_boundary_facets_based_on_gamma(gamma, msh, threshold=1, tolerance=0.1):
    """
    Tags facets (edges/faces) on the boundary where gamma transitions from a value close to 1 to a different value.
    
    Parameters:
    - gamma: fem.Function, the gamma function.
    - msh: dolfinx mesh.
    - threshold: float, the value of gamma to tag (default = 1.0).
    - tolerance: float, the tolerance for comparison (default = 1e-6).
    
    Returns:
    - meshtags: A dolfinx.MeshTags object with the tagged facets (edges in 2D, faces in 3D).
    """
    tdim = msh.topology.dim
    fdim = tdim - 1  # Facets (edges in 2D, faces in 3D)
    
    # Connectivity between facets and cells (create if needed)
    msh.topology.create_connectivity(fdim, tdim)
    
    # Number of facets
    num_facets = msh.topology.index_map(fdim).size_local
    
    # Array to store tags
    facet_tags = np.zeros(num_facets, dtype=np.int32)
    V_gamma = gamma.function_space
    # Iterate through each facet
    for facet in range(num_facets):
        # Get the cells that share this facet
        cell_dofs = msh.topology.connectivity(fdim, tdim).links(facet)
        
        # Get gamma values for the cells connected to the facet
        gamma_values_in_cells = [gamma.vector.array[V_gamma.dofmap.cell_dofs(cell)] for cell in cell_dofs]
        
        # Check if there is a transition from gamma = 1 to something else across the facet
        gamma_mean_values = [np.mean(gamma_values) for gamma_values in gamma_values_in_cells]
        if any(np.abs(gamma_mean - threshold) < tolerance for gamma_mean in gamma_mean_values) and \
           any(np.abs(gamma_mean - threshold) >= tolerance for gamma_mean in gamma_mean_values):
            facet_tags[facet] = 1  # Tag this facet as part of the contour

    # Create a MeshTags object for facets
    facet_tags_msh = mesh.meshtags(msh, fdim, np.arange(num_facets), facet_tags)

    return facet_tags_msh


def create_mesh(L_x, L_y, source_width, source_height, resolution):
    gdim = 2
    if RANK == 0:
        gmsh.initialize()
        gmsh.model.add("domain_with_extrusion")

        # Define points for the base rectangle
        p0 = gmsh.model.geo.addPoint(0, 0, 0, meshSize=resolution)
        p1 = gmsh.model.geo.addPoint(L_x, 0, 0, meshSize=resolution)
        p2 = gmsh.model.geo.addPoint(L_x, L_y, 0, meshSize=resolution)
        p3 = gmsh.model.geo.addPoint(0, L_y, 0, meshSize=resolution)

        # Define lines for the base rectangle
        l0 = gmsh.model.geo.addLine(p0, p1)
        l1 = gmsh.model.geo.addLine(p1, p2)
        l2 = gmsh.model.geo.addLine(p2, p3)
        l3 = gmsh.model.geo.addLine(p3, p0)

        # Define points for the extrusion (source region)
        x_min = 0.5 * L_x - 0.5 * source_width
        x_max = 0.5 * L_x + 0.5 * source_width
        p4 = gmsh.model.geo.addPoint(x_min, L_y, 0, meshSize=resolution)
        p5 = gmsh.model.geo.addPoint(x_max, L_y, 0, meshSize=resolution)
        p6 = gmsh.model.geo.addPoint(x_max, L_y + source_height, 0, meshSize=resolution)
        p7 = gmsh.model.geo.addPoint(x_min, L_y + source_height, 0, meshSize=resolution)

        # Define lines for the extrusion
        l4 = gmsh.model.geo.addLine(p4, p5)
        l5 = gmsh.model.geo.addLine(p5, p6)
        l6 = gmsh.model.geo.addLine(p6, p7)
        l7 = gmsh.model.geo.addLine(p7, p4)

        # Connect the extrusion to the base rectangle
        l8 = gmsh.model.geo.addLine(p3, p4)
        l9 = gmsh.model.geo.addLine(p5, p2)

        # Define curve loops
        loop_combined = gmsh.model.geo.addCurveLoop([l0, l1, -l9, l5, l6, l7, -l8, l3])
        surface = gmsh.model.geo.addPlaneSurface([loop_combined])

        gmsh.model.geo.synchronize()
        # Isothermal boundary
        gmsh.model.addPhysicalGroup(1, [l0], tag=1)
        gmsh.model.setPhysicalName(1, 1, "IsothermalBoundary")
        # Source boundary
        gmsh.model.addPhysicalGroup(1, [l6], tag=3)
        gmsh.model.setPhysicalName(1, 3, "TopBoundary")
        # Slip boundary
        gmsh.model.addPhysicalGroup(1, [l1, l9, l5, l7, l8, l3], tag=2)
        gmsh.model.setPhysicalName(1, 2, "SlipBoundary")
        # Define physical groups for domains (if needed)
        gmsh.model.addPhysicalGroup(2, [surface], tag=1)
        gmsh.model.setPhysicalName(2, 1, "Domain")

        # Generate the mesh
        gmsh.model.mesh.generate(2)

    # Convert GMSH model to DOLFINx mesh and distribute it across all ranks
    comm = MPI.COMM_WORLD
    msh, cell_markers, facet_markers = io.gmshio.model_to_mesh(gmsh.model, comm, rank=0, gdim=gdim)

    if RANK == 0:
        gmsh.finalize()

    return msh, cell_markers, facet_markers


def define_function_spaces(msh):

    # Function spaces for temperature and heat flux
    # Create the Taylor-Hood function space
    P2 = basix.ufl.element("CG", msh.basix_cell(), 2, shape=(msh.geometry.dim,))
    P1 = basix.ufl.element("CG", msh.basix_cell(), 1)  # for temperature
    TH = basix.ufl.mixed_element([P2, P1])
    W = fem.functionspace(msh, TH)

    # Define functions
    U = fem.Function(W)  # Current solution (q, T)
    dU = ufl.TrialFunction(W)  # Increment (delta q, delta T)
    (v, s) = ufl.TestFunctions(W)  # Test functions

    # Initialize U with zeros or an appropriate initial guess
    U.x.array[:] = 0.0

    return W, U, dU, v, s


def img_to_gamma_expression(img, domain, mask_extrusion=True):
    # Precompute local min and max coordinates of the mesh
    x = domain.geometry.x
    x_local_min = np.min(x[:, 0]) if x.size > 0 else np.inf
    x_local_max = np.max(x[:, 0]) if x.size > 0 else -np.inf
    y_local_min = np.min(x[:, 1]) if x.size > 0 else np.inf
    y_local_max = np.max(x[:, 1]) if x.size > 0 else -np.inf

    # Compute global min and max coordinates
    comm = domain.comm
    x_min = comm.allreduce(x_local_min, op=MPI.MIN)
    x_max = comm.allreduce(x_local_max, op=MPI.MAX)
    y_min = comm.allreduce(y_local_min, op=MPI.MIN)
    y_max = comm.allreduce(y_local_max, op=MPI.MAX)

    img_height, img_width = img.shape

    x_range = x_max - x_min
    y_range = y_max - y_min

    y_min += SOURCE_HEIGHT  # to make image higher
    x_min -= x_range/100  # to move image  to the left

    # Avoid division by zero in case the mesh or image has no range
    if x_range == 0:
        x_range = 1.0
    if y_range == 0:
        y_range = 1.0

    # define the extrusion region
    y_min_extrusion = L_Y - SOURCE_HEIGHT
    x_min_extrusion = L_X / 2 - SOURCE_WIDTH / 2
    x_max_extrusion = L_X / 2 + SOURCE_WIDTH / 2

    def gamma_expression(x_input):
        # x_input is of shape (gdim, N)
        x_coords = x_input[0, :]
        y_coords = x_input[1, :]

        # Initialize gamma_values with zeros
        gamma_values = np.zeros_like(x_coords)

        # For all points in the mesh, scale the image to the full size of the mesh
        x_norm = (x_coords - x_min) / x_range  # Normalize x coordinates to the range of the mesh
        y_norm = (y_coords - y_min) / y_range  # Normalize y coordinates to the range of the mesh

        x_indices = np.clip((x_norm * (img_width - 1)).astype(int), 0, img_width - 1)
        y_indices = np.clip(((1 - y_norm) * (img_height - 1)).astype(int), 0, img_height - 1)

        gamma_values = img[y_indices, x_indices]  # Map image values to the mesh

        # Mask the top extrusion if requested
        if mask_extrusion:
            # above Ly and between Lx/2 - w/2 and Lx/2 + w/2
            in_extrusion = np.logical_and(
                np.logical_and(y_coords > y_min_extrusion, x_coords >= x_min_extrusion),
                x_coords <= x_max_extrusion,
            )
            gamma_values[in_extrusion] = 1.0

        return gamma_values

    return gamma_expression


def postprocess_results(U, msh, img, gamma, alpha, time1):

    V1, _ = U.function_space.sub(1).collapse()
    global_top, global_geom, global_ct, global_gamma = gather_mesh_on_rank0(msh, V1, gamma)
    _, _, _, global_alpha = gather_mesh_on_rank0(msh, V1, alpha)

    if RANK == 0:

        time2 = time.time()
        print(f"Time taken: {time2 - time1}")

        import pyvista as pv

        # plot gamma
        if global_gamma is not None:
            grid = pv.UnstructuredGrid(global_top, global_ct, global_geom)
            grid.point_data["gamma"] = global_gamma.real
            grid.set_active_scalars("gamma")

            # Plot the scalar field
            plotter = pv.Plotter()
            plotter.add_mesh(grid, show_edges=True)
            plotter.view_xy()
            plotter.show()
            
        # plot alpha
        if global_alpha is not None:
            grid = pv.UnstructuredGrid(global_top, global_ct, global_geom)
            grid.point_data["gamma"] = global_alpha.real
            grid.set_active_scalars("gamma")

            # Plot the scalar field
            plotter = pv.Plotter()
            plotter.add_mesh(grid, show_edges=True)
            plotter.view_xy()
            plotter.show()


if __name__ == "__main__":
    main()
