# utils.py

import numpy as np

# import h5py
import pyvista as pv
from dolfinx.plot import vtk_mesh
from mpi4py import MPI
import dolfinx.fem
from dolfinx import fem
import ufl
from ufl import TestFunction, TrialFunction, inner, grad, form, derivative, inner
import dolfinx
from petsc4py import PETSc


def gather_mesh_on_rank0(mesh, V, function, root=0):
    """
    Gathers mesh data (topology, cell types, geometry) and solution data (u) from all ranks to rank 0.

    Parameters:
    mesh: dolfinx.Mesh
        The distributed mesh.
    V: dolfinx.FunctionSpace
        The function space associated with the function.
    function: dolfinx.Function
        The function whose data needs to be gathered (e.g., temperature field).
    root: int
        The rank on which to gather data (default is 0).

    Returns:
    On rank 0:
        - root_top: np.ndarray (global topology)
        - root_geom: np.ndarray (global geometry)
        - root_ct: np.ndarray (global cell types)
        - root_vals: np.ndarray (global function values)
    On other ranks:
        - None, None, None, None
    """
    comm = mesh.comm
    rank = comm.rank

    # Create local VTK mesh data structures
    topology, cell_types, geometry = vtk_mesh(mesh, mesh.topology.dim)

    # Get the number of cells and DOFs (degrees of freedom) for local partition
    num_cells_local = mesh.topology.index_map(mesh.topology.dim).size_local
    num_dofs_local = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

    # Number of DOFs per cell (assuming uniform cells)
    num_dofs_per_cell = topology[0]

    # Get the DOF indices from the topology array
    topology_dofs = (np.arange(len(topology)) % (num_dofs_per_cell + 1)) != 0

    # Map to global DOF indices
    global_dofs = V.dofmap.index_map.local_to_global(topology[topology_dofs].copy())

    # Replace local DOF indices with global DOF indices
    topology[topology_dofs] = global_dofs

    # Gather mesh and function data on the root process
    global_topology = comm.gather(
        topology[: (num_dofs_per_cell + 1) * num_cells_local], root=root
    )
    global_geometry = comm.gather(
        geometry[: V.dofmap.index_map.size_local, :], root=root
    )
    global_ct = comm.gather(cell_types[:num_cells_local], root=root)
    global_vals = comm.gather(function.x.array[:num_dofs_local], root=root)

    if rank == root:
        # Stack the data from all ranks on the root process
        root_geom = np.vstack(global_geometry)
        root_top = np.concatenate(global_topology)
        root_ct = np.concatenate(global_ct)
        root_vals = np.concatenate(global_vals)

        return root_top, root_geom, root_ct, root_vals

    return None, None, None, None


def project_gamma(gamma, slope=10, point=0.5):
    gamma_values = gamma.vector.array

    return (np.tanh(slope * (gamma_values - point)) + np.tanh(slope * point)) / (
        np.tanh(slope * (1 - point)) + np.tanh(slope * point)
    )

    
def alpha_function(gamma, msh, alphamin=0.0125, alphamax=1, qa=1):
    comm = msh.comm
    gamma.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    interface = ufl.inner(ufl.grad(gamma), ufl.grad(gamma))

    V_g = gamma.function_space
    interface_g = fem.Function(V_g)
    inter_expr = fem.Expression(interface, V_g.element.interpolation_points())
    interface_g.interpolate(inter_expr)
    # interface = project(interface, gamma.function_space())

    interface_filtered = filter_function(interface_g, msh, rad_div=10)
    # Normalize the filtered interface
    # interface_g *= 1 / (interface_g.vector.max() + 1e-10)
    interface_values = interface_filtered.vector.array
    
    # filter out small values
    interface_values = interface_filtered.vector.array
    global_max = comm.allreduce(np.max(interface_values), op=MPI.MAX)
    if global_max > 1e-10:
        interface_values /= global_max
    interface_filtered.vector.array[:] = interface_values
    interface_filtered.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    # interface_filtered.vector.set(interface_values)
    # inter_expr = fem.Expression(interface_filtered, V_g.element.interpolation_points())
    # interface_filtered.interpolate(inter_expr)
    # interface_filtered.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    # interface_gg = project(interface, gamma.function_space())
    interface_filtered_2 = filter_function(interface_filtered, msh, rad_div=15)
    interface_filtered_2.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    # Compute the global mean
    interface_values = interface_filtered_2.vector.array.copy()
    local_sum = np.sum(interface_values)
    local_count = len(interface_values)
    global_sum = comm.allreduce(local_sum, op=MPI.SUM)
    global_count = comm.allreduce(local_count, op=MPI.SUM)

    global_mean = global_sum / global_count

    inter_proj_gamma = project_gamma(interface_filtered_2, point=global_mean)
    interface_filtered_2.vector.array = inter_proj_gamma
    interface_filtered_2.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    # interface_g.vector().set_local(project_gamma(interface, point=mean))

    interface_rev = 1 - interface_filtered_2

    alpha = alphamin + (alphamax - alphamin) * (1 - interface_rev) / (
        1 + qa * interface_rev
    )

    # project alpha
    alpha_g = fem.Function(V_g)
    alpha_expr = fem.Expression(alpha, V_g.element.interpolation_points())
    alpha_g.interpolate(alpha_expr)
    alpha_g.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    alpha_clamp = alpha_g.vector.array.copy()
    alpha_clamp = np.clip(alpha_clamp, alphamin, alphamax)

    alpha_g.vector.array[:] = alpha_clamp
    alpha_g.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    return alpha_g


# Apply filter function
def filter_function(rho_n, mesh, rad_div=20, projection=True, name="Filtered"):
    # Find the minimum size (hmin) across all processes
    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local

    # Compute the size of all cells in the mesh
    h_sizes = mesh.h(tdim, np.arange(num_cells))

    # Find the minimum size (hmin) across all processes
    hmin = mesh.comm.allreduce(np.min(h_sizes), op=MPI.MIN)
    # Compute the filter radius
    r_min = hmin / rad_div

    V = rho_n.function_space

    # Define trial and test functions
    rho = TrialFunction(V)
    w = TestFunction(V)

    # Define the bilinear and linear forms
    a = (r_min**2) * inner(grad(rho), grad(w)) * ufl.dx + rho * w * ufl.dx
    L = rho_n * w * ufl.dx

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
    rho_filtered.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    return rho_filtered


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


def gather_vector_data_on_rank0(mesh, function, root=0):
    """
    Gathers cell centers and vector function data from all ranks to rank 0.

    Parameters:
    mesh: dolfinx.Mesh
        The distributed mesh.
    function: dolfinx.Function
        The vector function (e.g., q_dg) to gather.
    root: int
        The rank on which to gather data (default is 0).

    Returns:
    On rank 0:
        - cell_centers: np.ndarray (global cell centers)
        - function_values: np.ndarray (global vector function values at cell centers)
    On other ranks:
        - None, None
    """
    comm = mesh.comm
    rank = comm.rank

    # Get local cell centers
    num_cells_local = mesh.topology.index_map(mesh.topology.dim).size_local
    # Coordinates of cell vertices
    cell_vertices = mesh.geometry.x[
        mesh.topology.connectivity(mesh.topology.dim, 0).array
    ]
    # Reshape to (num_cells_local, num_vertices_per_cell, gdim)
    num_vertices_per_cell = cell_vertices.shape[0] // num_cells_local
    cell_vertices = cell_vertices.reshape((num_cells_local, num_vertices_per_cell, -1))
    # Compute cell centers
    cell_centers_local = cell_vertices.mean(axis=1)

    # Evaluate function at cell centers
    function_values_local = function.eval(cell_centers_local, mesh.cells)

    # Gather data on root
    all_cell_centers = comm.gather(cell_centers_local, root=root)
    all_function_values = comm.gather(function_values_local, root=root)

    if rank == root:
        # Concatenate data from all ranks
        cell_centers = np.vstack(all_cell_centers)
        function_values = np.vstack(all_function_values)
        return cell_centers, function_values
    else:
        return None, None


def gather_solution_on_rank0(function, mesh):
    """
    Gathers a distributed solution field from all MPI ranks and orders it in a single global array on rank 0.

    Parameters:
    function: dolfinx.Function
        The function (solution) to be gathered from all ranks.
    mesh: dolfinx.Mesh
        The mesh associated with the function space.

    Returns:
    global_array (numpy.ndarray): Gathered solution array on rank 0 (None on other ranks).
    global_x (numpy.ndarray): Gathered coordinates of DOFs on rank 0 (None on other ranks).
    """
    comm = mesh.comm
    rank = comm.rank

    # Ensure data is up-to-date
    function.x.scatter_forward()

    # Get the index map and local ranges
    imap = function.function_space.dofmap.index_map
    local_range = (
        np.asarray(imap.local_range, dtype=np.int32)
        * function.function_space.dofmap.index_map_bs
    )
    size_global = imap.size_global * function.function_space.dofmap.index_map_bs

    # Gather local ranges and solution data on rank 0
    ranges = comm.gather(local_range, root=0)
    data = comm.gather(function.vector.array, root=0)

    # Gather local coordinates of degrees of freedom (DOFs)
    x = function.function_space.tabulate_dof_coordinates()[: imap.size_local]
    x_glob = comm.gather(x, root=0)

    if rank == 0:
        # Create global array for solution values
        global_array = np.zeros(size_global)
        for r, d in zip(ranges, data):
            global_array[r[0] : r[1]] = d

        # Create global array for coordinates of DOFs
        global_x = np.zeros((size_global, x.shape[1]))
        for r, x_ in zip(ranges, x_glob):
            global_x[r[0] : r[1], :] = x_

        return global_array, global_x
    else:
        return None, None
