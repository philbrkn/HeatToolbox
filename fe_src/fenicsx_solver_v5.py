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

from utils import gather_mesh_on_rank0
from vae_model import VAE, z_to_img, Flatten, UnFlatten

os.environ["OMP_NUM_THREADS"] = "1"  # Use one thread per process

# Physical constants
C = PETSc.ScalarType(1.0)  # Slip parameter for fully diffusive boundaries
T_ISO = PETSc.ScalarType(0.0)  # Isothermal temperature, K
# Q_L = 50
Q_L = 120
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
RESOLUTION = LENGTH / 15  # Adjust mesh resolution as needed

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
        # z = np.array([0.57852902, -0.75218827,  0.07094553, -0.40801165])  # tree
        # z = np.array([-0.28166873, -0.25455361, -1, 1])
        # z = np.array([0.90583324, -0.70455073, -0.1075645,  -1])
        # z = np.array([0.63012329, -0.5772154,  -0.86185697,  0.09507076])
        z = np.array([[0.91578013,  0.06633388,  0.3837567,  -0.36896428]])  # tree sym
        img = z_to_img(z, model, device)
        # symmetrize and de symmetrize
        img = img[:, :img.shape[1] // 2]  # Take the left half of the image
        img = np.concatenate((img, img[:, ::-1]), axis=1)  # Symmetrize the image
    else:
        img = None

    img = comm.bcast(img, root=0)
    # Define boundary conditions and measures

    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_markers)
    ds_bottom = ds(1)
    ds_slip = ds(2)
    ds_top = ds(3)
    
    check_form0 = fem.form(PETSc.ScalarType(1) * ds_top)
    check_local0 = fem.assemble_scalar(check_form0)  # assemble over cell
    totat_check0 = msh.comm.allreduce(check_local0, op=MPI.SUM)
    if RANK == 0:
        print(f"Integral of 1 over source line: {totat_check0}")
        print(f"Should be: {SOURCE_WIDTH}")

    # # CHECK ISOTHERMAL LINE
    check_form2 = fem.form(PETSc.ScalarType(1) * ds_bottom)
    check_local2 = fem.assemble_scalar(check_form2)  # assemble over cell
    totat_check2 = msh.comm.allreduce(check_local2, op=MPI.SUM)
    if RANK == 0:
        print(f"Integral of 1 over isothermal line: {totat_check2}")
        print(f"Should be: {L_X}")

    # # CHECK SLIP LINE
    check_form3 = fem.form(PETSc.ScalarType(1) * ds_slip)
    check_local3 = fem.assemble_scalar(check_form3)  # assemble over cell
    totat_check3 = msh.comm.allreduce(check_local3, op=MPI.SUM)
    if RANK == 0:
        print(f"Integral of 1 over slip line: {totat_check3}")
        print(f"Should be: {L_Y * 2 + (L_X-SOURCE_WIDTH)}")

    # Define function spaces and functions
    W, U, dU, v, s = define_function_spaces(msh)

    # set gamma as a function equal to 0
    V_gamma = fem.functionspace(msh, ("CG", 1))
    gamma = fem.Function(V_gamma)  # Material field

    gamma_expr = img_to_gamma_expression(img, msh)
    gamma.interpolate(gamma_expr)

    # Define variational forms
    n = ufl.FacetNormal(msh)
    F = define_variational_form(
        U, v, s, KAPPA_SI, ELL_SI, KAPPA_DI, ELL_DI, n, ds_slip, ds_top, Q, gamma
    )

    # Solve the problem
    solve_problem(U, dU, F, W)

    # Post-process results
    postprocess_results(U, msh, img, gamma, time1)


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


def define_variational_form(U, v, s, kappa_si, ell_si, kappa_di, ell_di, n, ds_slip, ds_top, Q, gamma):

    q, T = ufl.split(U)

    def ramp(gamma, a_min, a_max, qa=200):
        return a_min + (a_max - a_min) * gamma / (1 + qa * (1 - gamma))

    ramp_kappa = ramp(gamma, kappa_si, kappa_di)
    ramp_ell = ramp(gamma, ell_si, ell_di)
    # ramp_kappa = ramp(gamma, kappa_di, kappa_si)
    # ramp_ell = ramp(gamma, ell_di, ell_si)

    viscous_term = (
        ramp_ell ** 2
        * (
            ufl.inner(ufl.grad(q), ufl.grad(v))
            + 2 * ufl.inner(ufl.div(q) * ufl.Identity(2), ufl.grad(v))
        )
        * ufl.dx
    )

    # Define flux continuity (∇⋅q = 0)
    flux_continuity = -ufl.div(q) * s * ufl.dx

    # Define pressure term (related to ∇T)
    pressure_term = - ramp_kappa * T * ufl.div(v) * ufl.dx

    # Define flux term (q ⋅ v)
    flux_term = ufl.inner(q, v) * ufl.dx

    # Tangential component of q
    def u_t(q):
        return q - ufl.dot(q, n) * n

    # Slip boundary condition term
    def t(q, T):
        return (
            ramp_ell**2 * (ufl.grad(q) + 2 * ufl.div(q) * ufl.Identity(2))
            - ramp_kappa * T * ufl.Identity(2)
        ) * n

    F = (
        flux_continuity  # Continuity: ∇⋅q = 0
        + flux_term  # Flux term: q ⋅ v
        + viscous_term  # Viscous-like term
        + pressure_term  # Pressure-like term from ∇T
        - ufl.dot(n, t(q, T)) * ufl.dot(v, n) * ds_slip  # Slip boundary condition term
        - ufl.dot(q, n) * ufl.dot(n, t(v, s)) * ds_slip  # Slip boundary condition term
        + ramp_ell * ufl.dot(u_t(q), u_t(v)) * ds_slip  # Slip boundary stabilization term
        + ufl.dot(q, n) * ufl.dot(v, n) * ds_slip  # Additional stabilization
        + Q * ufl.dot(v, n) * ds_top  # Source term at the top boundary
    )

    return F


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

    # ranges in mesh dimensions
    x_range = x_max - x_min
    y_range = y_max - y_min

    y_min += SOURCE_HEIGHT  # to make image higher
    # center the image

    # Avoid division by zero in case the mesh or image has no range
    if x_range == 0:
        x_range = 1.0
    if y_range == 0:
        y_range = 1.0

    # define the extrusion region for masking
    y_min_extrusion = L_Y - SOURCE_HEIGHT * 2
    x_min_extrusion = L_X / 2 - SOURCE_WIDTH * 0.6
    x_max_extrusion = L_X / 2 + SOURCE_WIDTH * 0.6

    def gamma_expression(x_input):
        # x_input is of shape (gdim, N)
        x_coords = x_input[0, :]
        y_coords = x_input[1, :]

        # Initialize gamma_values with zeros
        gamma_values = np.zeros_like(x_coords)

        x_mesh_center = x_range / 2
        # For all points in the mesh, scale the image to the full size of the mesh
        x_norm = (x_coords - x_mesh_center) / x_range + 0.5
        y_norm = (y_coords - y_min) / y_range  # Normalize y coordinates to the range of the mesh

        # Map the normalized coordinates to the image
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


def solve_problem(U, dU, F, W):

    residual = fem.form(F)
    J = ufl.derivative(F, U, dU)
    jacobian = fem.form(J)

    # Set up boundary conditions
    W1 = W.sub(1)
    Q1, _ = W1.collapse()
    noslip = fem.Function(Q1)
    facets = mesh.locate_entities(U.function_space.mesh, 1, isothermal_boundary)
    dofs = fem.locate_dofs_topological((W1, Q1), 1, facets)
    bc0 = fem.dirichletbc(noslip, dofs, W1)

    # Create matrix and vector
    A = fem.petsc.create_matrix(jacobian)
    L = fem.petsc.create_vector(residual)

    solver = PETSc.KSP().create(U.function_space.mesh.comm)
    solver.setOperators(A)
    solver.setType("minres")
    solver.setTolerances(rtol=1e-6, atol=1e-13, max_it=1000)
    pc = solver.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")

    # Set MUMPS options
    factor_mat = pc.getFactorMatrix()
    factor_mat.setMumpsIcntl(24, 1)  # Support solving singular matrix
    factor_mat.setMumpsIcntl(25, 0)  # Support solving singular matrix
    factor_mat.setMumpsIcntl(13, 1)  # Enable parallel root node factorization
    factor_mat.setMumpsIcntl(28, 2)  # Use parallel analysis
    factor_mat.setMumpsIcntl(14, 100)  # Increase MUMPS working memory
    factor_mat.setMumpsIcntl(22, 0)  # Disable out-of-core factorization
    factor_mat.setMumpsCntl(1, 1e-6)  # Relative pivoting scale
    factor_mat.setMumpsCntl(3, 1e-6)  # Absolute pivoting scale
    factor_mat.setMumpsIcntl(1, -1)  # Print all error messages
    factor_mat.setMumpsIcntl(2, 3)  # Enable diagnostic printing stats and warnings
    factor_mat.setMumpsIcntl(4, 0)  # Set print level verbosity (0-4)

    solver.setFromOptions()

    du = fem.Function(W)

    i = 0
    max_iterations = 5
    while i < max_iterations:
        # Assemble Jacobian and residual
        with L.localForm() as loc_L:
            loc_L.set(0)
        A.zeroEntries()
        fem.petsc.assemble_matrix(A, jacobian, bcs=[bc0])
        A.assemble()
        fem.petsc.assemble_vector(L, residual)
        L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        L.scale(-1)

        # Apply boundary conditions
        fem.petsc.apply_lifting(L, [jacobian], [[bc0]], x0=[U.vector], scale=1)
        fem.petsc.set_bc(L, [bc0], U.vector, 1.0)
        L.ghostUpdate(
            addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
        )

        # Solve linear problem
        solver.solve(L, du.vector)
        du.x.scatter_forward()

        # Update solution
        U.x.array[:] += du.x.array
        i += 1

        # Check convergence
        correction_norm = du.vector.norm(0)
        if U.function_space.mesh.comm.rank == 0:
            print(f"Iteration {i}: Correction norm {correction_norm}")
        if correction_norm < 1e-5:
            break


def postprocess_results(U, msh, img, gamma, time1):

    q, T = U.sub(0).collapse(), U.sub(1).collapse()
    norm_q, norm_T = la.norm(q.x), la.norm(T.x)

    V1, _ = U.function_space.sub(1).collapse()
    global_top, global_geom, global_ct, global_vals = gather_mesh_on_rank0(msh, V1, T)
    _, _, _, global_gamma = gather_mesh_on_rank0(msh, V1, gamma)

    if RANK == 0:
        print(f"(D) Norm of flux coefficient vector (monolithic, direct): {norm_q}")
        print(f"(D) Norm of temp coefficient vector (monolithic, direct): {norm_T}")

        time2 = time.time()
        print(f"Time taken: {time2 - time1}")

        import pyvista as pv

        if global_vals is not None:
            grid = pv.UnstructuredGrid(global_top, global_ct, global_geom)
            grid.point_data["T"] = global_vals.real
            grid.set_active_scalars("T")

            # Plot the scalar field
            plotter = pv.Plotter()
            plotter.add_mesh(grid, cmap="coolwarm", show_edges=False)
            plotter.view_xy()
            plotter.show()

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

    # Vector field
    if RANK == 0:
        gdim = msh.geometry.dim
        V_dg = fem.functionspace(msh, ("DG", 2, (gdim,)))
        q_dg = fem.Function(V_dg)
        q_copy = q.copy()
        q_dg.interpolate(q_copy)

        with io.VTXWriter(msh.comm, "flux.bp", q_dg) as vtx:
            vtx.write(0.0)

        V_cells, V_types, V_x = dolfinx.plot.vtk_mesh(V_dg)
        V_grid = pv.UnstructuredGrid(V_cells, V_types, V_x)
        Esh_values = np.zeros((V_x.shape[0], 3), dtype=np.float64)
        Esh_values[:, :msh.topology.dim] = q_dg.x.array.reshape(V_x.shape[0], msh.topology.dim).real
        V_grid.point_data["u"] = Esh_values

        plotter = pv.Plotter()
        plotter.add_text("magnitude", font_size=12, color="black")
        plotter.add_mesh(V_grid.copy(), show_edges=False)
        plotter.view_xy()
        plotter.link_views()
        plotter.show()


if __name__ == "__main__":
    main()
