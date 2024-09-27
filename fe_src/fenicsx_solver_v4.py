import os
import time

import numpy as np
import ufl
import gmsh
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, io, la, mesh
import basix.ufl

from utils import gather_mesh_on_rank0

os.environ["OMP_NUM_THREADS"] = "1"  # Use one thread per process

# Physical constants
C = PETSc.ScalarType(1.0)  # Slip parameter for fully diffusive boundaries
T_ISO = PETSc.ScalarType(0.0)  # Isothermal temperature, K
Q_L = 90
Q = PETSc.ScalarType(Q_L)

ELL = PETSc.ScalarType(196e-9)  # Non-local length, m
KAPPA = PETSc.ScalarType(141.0)  # W/mK, thermal conductivity

LENGTH = 0.439e-6  # Characteristic length, adjust as necessary

L_X = 25 * LENGTH
L_Y = 12.5 * LENGTH
SOURCE_WIDTH = LENGTH
SOURCE_HEIGHT = LENGTH * 0.25
R_TOL = LENGTH * 1e-3
RESOLUTION = LENGTH / 2  # Adjust mesh resolution as needed


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
    # MPI initialization
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print(f"Rank {rank} of {size} is running.")

    if rank == 0:
        time1 = time.time()
    else:
        time1 = None

    # Create mesh
    msh = create_mesh(L_X, L_Y, SOURCE_WIDTH, SOURCE_HEIGHT, RESOLUTION)

    # Define boundary conditions and measures
    ds, ds_bottom, ds_slip, ds_top = define_boundary_conditions(msh)

    # Define function spaces and functions
    W, U, dU, v, s = define_function_spaces(msh)

    # Define variational forms
    n = ufl.FacetNormal(msh)
    F = define_variational_form(U, v, s, KAPPA, ELL, n, ds_slip, ds_top, Q)

    # Solve the problem
    solve_problem(U, dU, F, W)
        
    # Post-process results
    postprocess_results(U, msh, time1)


def create_mesh(L_x, L_y, source_width, source_height, resolution):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    gdim = 2
    if rank == 0:
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

        # Define physical groups for domains (if needed)
        gmsh.model.addPhysicalGroup(2, [surface], tag=1)
        gmsh.model.setPhysicalName(2, 1, "Domain")

        # Generate the mesh
        gmsh.model.mesh.generate(2)

    # Convert GMSH model to DOLFINx mesh and distribute it across all ranks
    comm = MPI.COMM_WORLD
    msh, _, _ = io.gmshio.model_to_mesh(gmsh.model, comm, rank=0, gdim=gdim)

    if rank == 0:
        gmsh.finalize()

    return msh


def define_boundary_conditions(msh):

    boundaries = [
        (1, isothermal_boundary),
        (2, slip_boundary),
        (3, source_boundary),
    ]

    facet_indices, facet_markers = [], []
    fdim = msh.topology.dim - 1
    for marker, locator in boundaries:
        facets = mesh.locate_entities(msh, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = mesh.meshtags(
        msh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets]
    )

    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tag)
    ds_bottom = ds(1)
    ds_slip = ds(2)
    ds_top = ds(3)

    return ds, ds_bottom, ds_slip, ds_top


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


def define_variational_form(U, v, s, kappa, ell, n, ds_slip, ds_top, Q):

    q, T = ufl.split(U)

    viscous_term = (
        ell**2
        * (
            ufl.inner(ufl.grad(q), ufl.grad(v))
            + 2 * ufl.inner(ufl.div(q) * ufl.Identity(2), ufl.grad(v))
        )
        * ufl.dx
    )

    # Define flux continuity (∇⋅q = 0)
    flux_continuity = -ufl.div(q) * s * ufl.dx

    # Define pressure term (related to ∇T)
    pressure_term = -kappa * T * ufl.div(v) * ufl.dx

    # Define flux term (q ⋅ v)
    flux_term = ufl.inner(q, v) * ufl.dx

    def ramp(gamma, a_min, a_max, qa=200):
        return a_min + (a_max - a_min) * gamma / (1+qa*(1-gamma))

    # Tangential component of q
    def u_t(q):
        return q - ufl.dot(q, n) * n

    # Slip boundary condition term
    def t(q, T):
        return (
            ell**2 * (ufl.grad(q) + 2 * ufl.div(q) * ufl.Identity(2))
            - kappa * T * ufl.Identity(2)
        ) * n

    F = (
        flux_continuity  # Continuity: ∇⋅q = 0
        + flux_term  # Flux term: q ⋅ v
        + viscous_term  # Viscous-like term
        + pressure_term  # Pressure-like term from ∇T
        - ufl.dot(n, t(q, T)) * ufl.dot(v, n) * ds_slip  # Slip boundary condition term
        - ufl.dot(q, n) * ufl.dot(n, t(v, s)) * ds_slip  # Slip boundary condition term
        + ell * ufl.dot(u_t(q), u_t(v)) * ds_slip  # Slip boundary stabilization term
        + ufl.dot(q, n) * ufl.dot(v, n) * ds_slip  # Additional stabilization
        + Q * ufl.dot(v, n) * ds_top  # Source term at the top boundary
    )

    return F


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


def postprocess_results(U, msh, time1):

    q, T = U.sub(0).collapse(), U.sub(1).collapse()
    norm_q, norm_T = la.norm(q.x), la.norm(T.x)

    V1, _ = U.function_space.sub(1).collapse()
    global_top, global_geom, global_ct, global_vals = gather_mesh_on_rank0(msh, V1, T)

    if msh.comm.rank == 0:
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
            plotter.show()


if __name__ == "__main__":
    main()
