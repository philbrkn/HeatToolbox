# utils.py

import numpy as np

# import h5py
import pyvista as pv
from dolfinx.plot import vtk_mesh
from mpi4py import MPI
import dolfinx.fem
from dolfinx import fem
from ufl import TestFunction, TrialFunction, inner, grad, form, derivative, inner

import ufl
from dolfinx import cpp as _cpp
from dolfinx import la
from dolfinx.fem import (Function, FunctionSpace, dirichletbc, form,
                         locate_dofs_geometrical)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_matrix, create_vector, set_bc)

from mpi4py import MPI
from petsc4py import PETSc


# Map image to gamma
def img_to_gamma(img, mesh, dx):
    # Map the image to the gamma field on the mesh
    x = mesh.geometry.x
    x_coords = x[:, 0]
    y_coords = x[:, 1]
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    x_norm = (x_coords - x_min) / (x_max - x_min)
    y_norm = (y_coords - y_min) / (y_max - y_min)
    img_height, img_width = img.shape
    x_indices = (x_norm * (img_width - 1)).astype(int)
    y_indices = ((1 - y_norm) * (img_height - 1)).astype(int)
    gamma_values = img[y_indices, x_indices]
    return gamma_values


# Apply filter function
def filter_function(rho_n, mesh, dx, projection=True, name="Filtered"):
    # Find the minimum size (hmin) across all processes
    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local

    # Compute the size of all cells in the mesh
    h_sizes = mesh.h(tdim, np.arange(num_cells))

    # Find the minimum size (hmin) across all processes
    hmin = mesh.comm.allreduce(np.min(h_sizes), op=MPI.MIN)
    # Compute the filter radius
    r_min = hmin / 20

    V = rho_n.function_space

    # Define trial and test functions
    rho = TrialFunction(V)
    w = TestFunction(V)

    # Define the bilinear and linear forms
    a = (r_min**2) * inner(grad(rho), grad(w)) * dx + rho * w * dx
    L = rho_n * w * dx

    # Create the solution function
    rho_filtered = fem.Function(V, name=name)
    # Create the linear problem and solve
    problem = dolfinx.fem.petsc.LinearProblem(
        a,
        L,
        u=rho_filtered,
        bcs=[],
        petsc_options={"ksp_type": "cg", "pc_type": "sor"},
    )
    rho_filtered = problem.solve()

    # Optional projection
    # if projection:
    #     rho_p = project_gamma(rho_filtered)
    #     rho_filtered.x.array[:] = rho_p

    return rho_filtered


class NonlinearPDE_SNESProblem:
    def __init__(self, F, u, bc):
        V = u.function_space
        du = TrialFunction(V)
        self.L = form(F)
        self.a = form(derivative(F, u, du))
        self.bc = bc
        self._F, self._J = None, None
        self.u = u

    def F(self, snes, x, F):
        """Assemble residual vector."""
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.vector)
        self.u.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        with F.localForm() as f_local:
            f_local.set(0.0)
        assemble_vector(F, self.L)
        apply_lifting(F, [self.a], bcs=[[self.bc]], x0=[x], scale=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(F, [self.bc], x, -1.0)

    def J(self, snes, x, J, P):
        """Assemble Jacobian matrix."""
        J.zeroEntries()
        assemble_matrix(J, self.a, bcs=[self.bc])
        J.assemble()


def plot_mesh(mesh):
    # Create a PyVista mesh
    topology, cell_types, geometry = vtk_mesh(mesh, mesh.topology.dim)
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)

    # Plot the mesh using PyVista
    plotter = pv.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    plotter.show()


def plot_boundaries(mesh, facet_markers):
    # Create a PyVista plotter
    plotter = pv.Plotter()

    # Extract the boundary facets based on the facet markers
    topology, cell_types, geometry = vtk_mesh(mesh, mesh.topology.dim)
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)

    # Add the boundary markers as a new cell array
    grid.cell_data["Boundary Markers"] = facet_markers.values

    # Plot boundaries with different colors for each marker
    plotter.add_mesh(
        grid, scalars="Boundary Markers", show_edges=True, show_scalar_bar=True
    )
    plotter.show()


def plot_subdomains(mesh, cell_markers):
    plotter = pv.Plotter()

    for marker in np.unique(cell_markers.values):
        cells = np.where(cell_markers.values == marker)[0]
        topology, cell_types, geometry = vtk_mesh(mesh, mesh.topology.dim, cells)
        subdomain_grid = pv.UnstructuredGrid(topology, cell_types, geometry)
        plotter.add_mesh(subdomain_grid, show_edges=True, label=f"Subdomain {marker}")

    plotter.add_legend()
    plotter.show()


def plot_scalar_field(mesh, scalar_field):
    topology, cell_types, geometry = vtk_mesh(mesh, mesh.topology.dim)
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)
    grid.point_data["T"] = scalar_field.x.array.real
    grid.set_active_scalars("T")

    # Plot the scalar field
    plotter = pv.Plotter()
    plotter.add_mesh(grid, cmap="coolwarm", show_edges=True)
    plotter.show()


def plot_vector_field(mesh, vector_field, Q):
    top_imap = mesh.topology.index_map(mesh.topology.dim)
    num_cells = top_imap.size_local + top_imap.num_ghosts
    midpoints = dolfinx.mesh.compute_midpoints(mesh, mesh.topology.dim, np.arange(num_cells, dtype=np.int32))
    num_dofs = Q.dofmap.index_map.size_local + Q.dofmap.index_map.num_ghosts
    # topology, cell_types, x 
    grid = pv.UnstructuredGrid(*vtk_mesh(mesh, mesh.topology.dim, np.arange(num_cells, dtype=np.int32)))

    print(num_cells, num_dofs)
    assert num_cells == num_dofs
    values = np.zeros((num_dofs, 3), dtype=np.float64)
    values[:, :mesh.geometry.dim] = vector_field.x.array.real.reshape(num_dofs, Q.dofmap.index_map_bs)

    cloud = pv.PolyData(midpoints)
    cloud["qw"] = values
    cloud["[W/m2]"] = np.linalg.norm(values, axis=1)

    # FOR NORMAL FLUX FIGURES
    glyphs = cloud.glyph("qw", scale=False, factor=5e-2)
    # glyphs = cloud.glyph("qw", scale=True, factor=3.5e-6)

    # THRESHOLD
    plotter = pv.Plotter()
    sargs = dict(height=0.1, vertical=False, position_x=0.22, position_y=0.05, n_labels=2, fmt="%.3g")
    actor2 = plotter.add_mesh(glyphs, cmap=plt.cm.jet, scalar_bar_args=sargs,
                                    scalars="[W/m2]", show_scalar_bar=show_scalar_bar,
                                    clim =(0,1e4))
    actor = plotter.add_mesh(grid, color="white", show_edges=False)  #, opacity=0.2)
    plotter.view_xy()
    plotter.show()


def plot_material_distribution(mesh, material_field):
    topology, cell_types, geometry = vtk_mesh(mesh, mesh.topology.dim)
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)
    # grid.point_data["Material"] = material_field.x.array

    # Plot the material distribution
    plotter = pv.Plotter()
    plotter.add_mesh(grid, cmap="coolwarm", show_edges=True)
    plotter.show()


def project_gamma(gamma, slope=10, point=0.5):
    return (np.tanh(slope * (gamma.vector() - point)) + np.tanh(slope * point)) / (
        np.tanh(slope * (1 - point)) + np.tanh(slope * point)
    )


def save_for_modulus(filename, mesh, temperature_field, flux_field):
    """
    Save the mesh and fields (temperature, flux)
    into an HDF5 file compatible with Modulus.

    Parameters:
    - filename: Name of the HDF5 file to create.
    - mesh: FEniCS mesh object.
    - temperature_field: FEniCS Function representing the temperature.
    - flux_field: FEniCS Function representing the flux.
    """
    coordinates = mesh.coordinates()
    connectivity = mesh.cells()
    temperature_values = temperature_field.vector().get_local()
    flux_values = flux_field.vector().get_local()

    with h5py.File(filename, "w") as hdf5_file:
        # Save mesh data
        hdf5_file.create_dataset("mesh/coordinates", data=coordinates)
        hdf5_file.create_dataset("mesh/connectivity", data=connectivity)
        # Save field data
        hdf5_file.create_dataset("fields/temperature", data=temperature_values)
        hdf5_file.create_dataset("fields/flux", data=flux_values)
