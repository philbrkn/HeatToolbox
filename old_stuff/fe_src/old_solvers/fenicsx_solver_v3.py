'''
tring to get parrallelization working. 
command: mpirun -n 4 python3 fe_src/fenicsx_solver_v3.py
'''
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io, la, log, mesh
from generate_mesh import create_mesh_v2  # , create_mesh_scaled
import basix.ufl
import dolfinx.fem.petsc
import os
from utils import gather_mesh_on_rank0
import gmsh
import time
os.environ["OMP_NUM_THREADS"] = "1"  # Use one thread per process

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"Rank {rank} of {size} is running.")

if rank == 0:
    time1 = time.time()
# Define dimensions based on the provided figure
# Define physical constants from the problem description
c = PETSc.ScalarType(1.63e6)  # J/m^3K, specific heat capacity
kappa = PETSc.ScalarType(141.0)  # W/mK, thermal conductivity
ell = PETSc.ScalarType(196e-9)  # Non-local length, m
# tau = 1e-12  # Relaxation time, s
# alpha = 1.0  # Dimensionless coefficient (alpha term)
T_iso = PETSc.ScalarType(0.0)  # Isothermal temperature, K
R = PETSc.ScalarType(2e-9)  # Kapitza resistance, m^2K/W
C = PETSc.ScalarType(1.0)  # Slip parameter for fully diffusive boundaries
Length = 0.439e-6  # Characteristic length, adjust as necessary

Lx = 25 * Length
Ly = 12.5 * Length
source_width = Length
source_height = Length * 0.25
r_tol = Length * 1e-3
# Calculate the line heat source q_l
# delta_T = 0.5  # K
# q_l = kappa * delta_T / Length  # W/
# print(f"Line heat source q_l = {q_l:.3e} W/m")
q_l = 90
Q = PETSc.ScalarType(q_l)


############################################
#---------- 4 Mesh Creation ---------------#
############################################

gdim = 2
# Define parameters
resolution = Length / 2  # Adjust mesh resolution as needed

if rank == 0:
    gmsh.initialize()
    gmsh.model.add("domain_with_extrusion")

    # Define points for the base rectangle
    p0 = gmsh.model.geo.addPoint(0, 0, 0, meshSize=resolution)
    p1 = gmsh.model.geo.addPoint(Lx, 0, 0, meshSize=resolution)
    p2 = gmsh.model.geo.addPoint(Lx, Ly, 0, meshSize=resolution)
    p3 = gmsh.model.geo.addPoint(0, Ly, 0, meshSize=resolution)

    # Define lines for the base rectangle
    l0 = gmsh.model.geo.addLine(p0, p1)
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p0)

    # Define points for the extrusion (source region)
    x_min = 0.5 * Lx - 0.5 * source_width
    x_max = 0.5 * Lx + 0.5 * source_width
    p4 = gmsh.model.geo.addPoint(x_min, Ly, 0, meshSize=resolution)
    p5 = gmsh.model.geo.addPoint(x_max, Ly, 0, meshSize=resolution)
    p6 = gmsh.model.geo.addPoint(x_max, Ly + source_height, 0, meshSize=resolution)
    p7 = gmsh.model.geo.addPoint(x_min, Ly + source_height, 0, meshSize=resolution)

    # Define lines for the extrusion
    l4 = gmsh.model.geo.addLine(p4, p5)
    l5 = gmsh.model.geo.addLine(p5, p6)
    l6 = gmsh.model.geo.addLine(p6, p7)
    l7 = gmsh.model.geo.addLine(p7, p4)

    # Connect the extrusion to the base rectangle
    l8 = gmsh.model.geo.addLine(p3, p4)
    l9 = gmsh.model.geo.addLine(p5, p2)

    # Define curve loops
    loop_combined = gmsh.model.geo.addCurveLoop(
        [l0, l1, -l9, l5, l6, l7, -l8, l3]
    )
    surface = gmsh.model.geo.addPlaneSurface([loop_combined])

    gmsh.model.geo.synchronize()

    # Define physical groups for domains (if needed)
    gmsh.model.addPhysicalGroup(2, [surface], tag=1)
    gmsh.model.setPhysicalName(2, 1, "Domain")

    # Generate the mesh
    gmsh.model.mesh.generate(2)


# Convert GMSH model to DOLFINx mesh and distribute it across all ranks
msh, _, _ = io.gmshio.model_to_mesh(gmsh.model, comm, rank=0, gdim=gdim)

if rank == 0:
    gmsh.finalize()

################################


def isothermal_boundary(x):
    return np.isclose(x[1], 0.0, rtol=r_tol)


def source_boundary(x):
    # return np.logical_and(
    #     np.isclose(x[1], Ly + source_height, rtol=r_tol),  # Ensure the boundary is at the top (y)
    #     np.logical_and(
    #         x[0] >= 0.5 * Lx - 0.5 * source_width,  # Ensure the boundary is within the source width
    #         x[0] <= 0.5 * Lx + 0.5 * source_width
    #     )
    # )
    return np.isclose(x[1], Ly + source_height, rtol=r_tol)


def slip_boundary(x):
    on_left = np.isclose(x[0], 0.0)
    on_right = np.isclose(x[0], Lx)
    on_top = np.isclose(x[1], Ly)
    return np.logical_or.reduce((on_left, on_right, on_top))


boundaries = [
    (1, isothermal_boundary),
    (2, slip_boundary),
    (3, source_boundary),
]


facet_indices, facet_markers = [], []
fdim = msh.topology.dim - 1
for (marker, locator) in boundaries:
    facets = mesh.locate_entities(msh, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = mesh.meshtags(msh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tag)
ds_bottom = ds(1)
ds_slip = ds(2)
ds_top = ds(3)

# ds_bottom = ufl.Measure("ds", domain=msh, subdomain_data=facet_markers, subdomain_id=1)
# dx = ufl.Measure("dx", domain=msh, subdomain_data=cell_markers)
# dx_source = dx(2)


# CHECK DS SOURCE
check_form0 = fem.form(PETSc.ScalarType(1) * ds_top)
check_local0 = fem.assemble_scalar(check_form0)  # assemble over cell
totat_check0 = msh.comm.allreduce(check_local0, op=MPI.SUM)
if rank == 0:
    print(f"Integral of 1 over source line: {totat_check0}")
    print(f"Should be: {source_width}")

# # CHECK ISOTHERMAL LINE
check_form2 = fem.form(PETSc.ScalarType(1) * ds_bottom)
check_local2 = fem.assemble_scalar(check_form2)  # assemble over cell
totat_check2 = msh.comm.allreduce(check_local2, op=MPI.SUM)
if rank == 0:
    print(f"Integral of 1 over isothermal line: {totat_check2}")
    print(f"Should be: {Lx}")

# # CHECK SLIP LINE
check_form3 = fem.form(PETSc.ScalarType(1) * ds_slip)
check_local3 = fem.assemble_scalar(check_form3)  # assemble over cell
totat_check3 = msh.comm.allreduce(check_local3, op=MPI.SUM)
if rank == 0:
    print(f"Integral of 1 over slip line: {totat_check3}")
    print(f"Should be: {Ly * 2 + (Lx-source_width)}")


# Function spaces for temperature and heat flux
# Create the Taylot-Hood function space
P2 = basix.ufl.element("CG", msh.basix_cell(), 2, shape=(msh.geometry.dim,))
P1 = basix.ufl.element("CG", msh.basix_cell(), 1)  # for temperature
TH = basix.ufl.mixed_element([P2, P1])
W = fem.functionspace(msh, TH)

# Define functions
U = fem.Function(W)            # Current solution (q, T)
dU = ufl.TrialFunction(W)      # Increment (delta q, delta T)
(v, s) = ufl.TestFunctions(W)  # Test functions

# Initialize U with zeros or an appropriate initial guess
U.x.array[:] = 0.0

# Split U into q and T
q, T = ufl.split(U)

# Define the variational forms
n = ufl.FacetNormal(msh)

# Define viscous term (for hydrodynamic model)
def viscous_term(q, v):
    return ell**2 * (ufl.inner(ufl.grad(q), ufl.grad(v)) + 2 * ufl.inner(ufl.div(q) * ufl.Identity(2), ufl.grad(v))) * ufl.dx

# Define flux continuity (∇⋅q = 0)
def flux_continuity(q, s):
    return - ufl.div(q) * s * ufl.dx

# Define pressure term (related to ∇T)
def pressure_term(T, q, v):
    return - kappa * T * ufl.div(v) * ufl.dx

# Define flux term (q ⋅ v)
def flux_term(q, v):
    return ufl.inner(q, v) * ufl.dx

# Tangential component of q
def u_t(q):
    return q - ufl.dot(q, n) * n

# Slip boundary condition term
def t(q, T):
    return (ell**2 * (ufl.grad(q) + 2 * ufl.div(q) * ufl.Identity(2)) - kappa * T * ufl.Identity(2)) * n

# Boundary term for isothermal condition
# T_iso_form = fem.Constant(msh, T_iso)
# F_isothermal = (T - T_iso + R * ufl.dot(q, n)) * s * ds_bottom


# Define the variational form
F = (
    flux_continuity(q, s)                 # Continuity: ∇⋅q = 0
    + flux_term(q, v)                     # Flux term: q ⋅ v
    + viscous_term(q, v)                  # Viscous-like term
    + pressure_term(T, q, v)              # Pressure-like term from ∇T
    - ufl.dot(n, t(q, T)) * ufl.dot(v, n) * ds_slip   # Slip boundary condition term
    - ufl.dot(q, n) * ufl.dot(n, t(v, s)) * ds_slip   # Slip boundary condition term
    + ell * ufl.dot(u_t(q), u_t(v)) * ds_slip    # Slip boundary stabilization term
    + ufl.dot(q, n) * ufl.dot(v, n) * ds_slip        # Additional stabilization
    + Q * ufl.dot(v, n) * ds_top  # Source term applied at the bottom boundary
    # + F_isothermal
)

# Temperature boundary condition
W1 = W.sub(1)
Q1, _ = W1.collapse()
noslip = fem.Function(Q1)
facets = mesh.locate_entities(msh, 1, isothermal_boundary)
dofs = fem.locate_dofs_topological((W1, Q1), 1, facets)
bc0 = fem.dirichletbc(noslip, dofs, W1)

# define residual
residual = fem.form(F)

# define function and jacobian
J = ufl.derivative(F, U, dU)

jacobian = fem.form(J)

A = dolfinx.fem.petsc.create_matrix(jacobian)
L = dolfinx.fem.petsc.create_vector(residual)
solver = PETSc.KSP().create(msh.comm)
solver.setOperators(A)
# solver.setType("minres")  # for umpfpack

solver.setType("minres")  # for mumps
solver.setTolerances(rtol=1e-6, atol=1e-13, max_it=1000)
pc = solver.getPC()
pc.setType("lu")
pc.setFactorSolverType("mumps")

# Set sequential analysis in MUMPS by controlling its options
# A = pc.getFactorMatrix()
pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)  # Option to support solving a singular matrix (pressure nullspace)
pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)  # Option to support solving a singular matrix (pressure nullspace)

# Parallel options
pc.getFactorMatrix().setMumpsIcntl(icntl=13, ival=1)  # Enable parallel root node factorization
pc.getFactorMatrix().setMumpsIcntl(icntl=28, ival=2)  # Use sequential analysis

# memory allocation
pc.getFactorMatrix().setMumpsIcntl(icntl=14, ival=100)  # Increase MUMPS working memory
pc.getFactorMatrix().setMumpsIcntl(22, 0)  # Disable out-of-core factorization
# pc.getFactorMatrix().setMumpsIcntl(icntl=22, ival=1)  # Out of core factorization
# pc.getFactorMatrix().setMumpsIcntl(icntl=23, ival=10000)  # Increase MUMPS working memory

# small scale
pc.getFactorMatrix().setMumpsCntl(1, 1e-6)  # relative pivoting scale
pc.getFactorMatrix().setMumpsCntl(3, 1e-6)  # absolute pivoting scale

# Enable detailed MUMPS diagnostic outputs
pc.getFactorMatrix().setMumpsIcntl(icntl=1, ival=-1)  # Print all error messages
pc.getFactorMatrix().setMumpsIcntl(icntl=2, ival=3)  # Enable diagnostic printing, statistics, and warnings
pc.getFactorMatrix().setMumpsIcntl(icntl=4, ival=0)  # Set print level to maximum verbosity (0-4)

# After the first analysis, set ICNTL(1) = -1 to skip further analyses
pc.getFactorMatrix().setMumpsIcntl(1, -1)  # Suppress analysis phase in subsequent calls

# Apply the options
solver.setFromOptions()

du = dolfinx.fem.Function(W)

# print(f"Rank {rank} is handling {msh.num_cells()} cells.")
# print(f"Rank {rank} is assembling a matrix with {A.getLocalSize()} local rows.")

i = 0
max_iterations = 5

while i < max_iterations:
    if rank == 0:
        print("Iteration ", i, " starting")
    # Assemble Jacobian and residual
    with L.localForm() as loc_L:
        loc_L.set(0)
    A.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(A, jacobian, bcs=[bc0])
    A.assemble()
    dolfinx.fem.petsc.assemble_vector(L, residual)
    L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    L.scale(-1)

    # Compute b - J(u_D-u_(i-1))
    dolfinx.fem.petsc.apply_lifting(L, [jacobian], [[bc0]], x0=[U.vector], scale=1)
    # Set du|_bc = u_{i-1}-u_D
    dolfinx.fem.petsc.set_bc(L, [bc0], U.vector, 1.0)
    L.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

    # Solve linear problem
    solver.solve(L, du.vector)
    du.x.scatter_forward()

    # Update u_{i+1} = u_i + delta u_i
    U.x.array[:] += du.x.array
    i += 1

    # Compute norm of update
    correction_norm = du.vector.norm(0)

    # du_norm.append(correction_norm)

    if rank == 0:
        print(f"Iteration {i}: Correction norm {correction_norm}")
    if correction_norm < 1e-5:
        break
    # solutions[i, :] = U.x.array[sort_order]

# Split the mixed solution and collapse
q, T = U.sub(0).collapse(), U.sub(1).collapse()
# Compute norms
norm_q, norm_T = la.norm(q.x), la.norm(T.x)

V1, _ = W.sub(1).collapse()
global_top, global_geom, global_ct, global_vals = gather_mesh_on_rank0(msh, V1, T)

if rank == 0:
    print(f"(D) Norm of flux coefficient vector (monolithic, direct): {norm_q}")
    print(f"(D) Norm of temp coefficient vector (monolithic, direct): {norm_T}")

    time2 = time.time()
    print(f"Time taken: {time2 - time1}")
    # with io.XDMFFile(msh.comm, "scalar_field.xdmf", "w") as xdmf:
    #     # Write the mesh to the XDMF file
    #     xdmf.write_mesh(msh)

    #     # Write the scalar field to the XDMF file
    #     xdmf.write_function(p)

    import pyvista as pv
    from dolfinx.plot import vtk_mesh

    # global_q, global_coords_q = gather_solution_on_rank0(q, msh)

    # error if NaN or Inf is detected in the solution or in norm
    # if np.any(np.isnan(T.x.array)) or np.any(np.isinf(T.x.array)) or np.any(np.isnan(q.x.array)) or np.any(np.isinf(q.x.array)):
    #     print("Warning: NaN or Inf detected in the solution!")
    #     throw error
    #     exit(1)

    if global_vals is not None:  # and global_q is not None:
        grid = pv.UnstructuredGrid(global_top, global_ct, global_geom)
        # if T.function_space.mesh.comm.size > 1:
        # Gather data from all ranks on rank 0
        # T.x.scatter_forward()
        # T = msh.comm.gather(T, root=0)
        grid.point_data["T"] = global_vals.real
        grid.set_active_scalars("T")

        # Plot the scalar field
        plotter = pv.Plotter()
        plotter.add_mesh(grid, cmap="coolwarm", show_edges=False)
        plotter.show()
        
    # Vector field
    # gdim = msh.geometry.dim
    # V_dg = fem.functionspace(msh, ("DG", 2, (gdim,)))
    # q_dg = fem.Function(V_dg)
    # q_copy = q.copy()
    # q_dg.interpolate(q_copy)

    # with io.VTXWriter(msh.comm, "flux.bp", q_dg) as vtx:
    #     vtx.write(0.0)

    # V_cells, V_types, V_x = dolfinx.plot.vtk_mesh(V_dg)
    # V_grid = pv.UnstructuredGrid(V_cells, V_types, V_x)
    # Esh_values = np.zeros((V_x.shape[0], 3), dtype=np.float64)
    # Esh_values[:, :msh.topology.dim] = q_dg.x.array.reshape(V_x.shape[0], msh.topology.dim).real
    # V_grid.point_data["u"] = Esh_values

    # plotter = pv.Plotter()
    # plotter.add_text("magnitude", font_size=12, color="black")
    # plotter.add_mesh(V_grid.copy(), show_edges=False)
    # plotter.view_xy()
    # plotter.link_views()
    # plotter.show()

