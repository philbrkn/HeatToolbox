'''
kinda works temp distribution is just off
'''
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io, la, log, mesh
from generate_mesh import create_mesh  # , create_mesh_scaled
import basix.ufl
import dolfinx.fem.petsc


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"Rank {rank} of {size} is running.")

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
# Length = 1

Lx = 25 * Length
Ly = 12.5 * Length
source_width = Length * 0.5
source_height = Length * 0.25
r_tol = Length / 1e5
# Calculate the line heat source q_l
# delta_T = 0.5  # K
# q_l = kappa * delta_T / Length  # W/
# print(f"Line heat source q_l = {q_l:.3e} W/m")
q_l = 20
Q = PETSc.ScalarType(q_l)


with io.XDMFFile(MPI.COMM_WORLD, "fe_src/mesh_matei/mesh.xdmf", "r") as xdmf:
    msh = xdmf.read_mesh(name="mesh")


def isothermal_boundary(x):
    return np.isclose(x[1], 0.0, rtol=r_tol)


def source_boundary(x):
    # return np.logical_and(
    #     np.logical_and(
    #         x[0] >= 0.5 * Lx - 0.5 * source_width,
    #         x[0] <= 0.5 * Lx + 0.5 * source_width
    #     ),
    #     np.isclose(x[1], Ly + source_height)
    # )
    return np.isclose(x[1], Ly + source_height, rtol=r_tol)


def slip_boundary(x):
    on_left = np.isclose(x[0], 0.0)
    on_right = np.isclose(x[0], Lx)
    on_top = np.isclose(x[1], Ly)
    in_source_region = np.logical_and(
        np.logical_and(
            x[0] >= 0.5 * Lx - 0.5 * source_width,
            x[0] <= 0.5 * Lx + 0.5 * source_width
        ),
        np.isclose(x[1], Ly + source_height)
    )
    return np.logical_or.reduce((on_left, on_right, on_top, in_source_region))


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


# # Check DX total
# check_form = fem.form(PETSc.ScalarType(1) * dx)
# check_local = fem.assemble_scalar(check_form)  # assemble over cell
# totat_check = msh.comm.allreduce(check_local, op=MPI.SUM)
# print(f"Integral of 1 over total region: {totat_check}")
# print(f"Should be: {Length ** 2 * 0.25 + 25 * 12.5 * Length ** 2}")

# CHECK DX SOURCE
check_form0 = fem.form(PETSc.ScalarType(1) * ds_top)
check_local0 = fem.assemble_scalar(check_form0)  # assemble over cell
totat_check0 = msh.comm.allreduce(check_local0, op=MPI.SUM)
print(f"Integral of 1 over source line: {totat_check0}")
print(f"Should be: {source_width}")

# # CHECK ISOTHERMAL LINE
check_form2 = fem.form(PETSc.ScalarType(1) * ds_bottom)
check_local2 = fem.assemble_scalar(check_form2)  # assemble over cell
totat_check2 = msh.comm.allreduce(check_local2, op=MPI.SUM)
print(f"Integral of 1 over isothermal line: {totat_check2}")
print(f"Should be: {Lx}")

# # CHECK SLIP LINE
check_form3 = fem.form(PETSc.ScalarType(1) * ds_slip)
check_local3 = fem.assemble_scalar(check_form3)  # assemble over cell
totat_check3 = msh.comm.allreduce(check_local3, op=MPI.SUM)
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
T_iso_form = fem.Constant(msh, T_iso)
F_isothermal = (T - T_iso + R * ufl.dot(q, n)) * s * ds_bottom


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
    + F_isothermal
)


# define residual
residual = fem.form(F)

# define function and jacobian
J = ufl.derivative(F, U, dU)

jacobian = fem.form(J)

A = dolfinx.fem.petsc.create_matrix(jacobian)
L = dolfinx.fem.petsc.create_vector(residual)
solver = PETSc.KSP().create(msh.comm)
solver.setOperators(A)
solver.setType("minres")
solver.setTolerances(rtol=1e-6)

pc = solver.getPC()
pc.setType("lu")
pc.setFactorSolverType("umfpack")
pc.setFactorSetUpSolverType()
# pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
# pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)
du = dolfinx.fem.Function(W)

i = 0
max_iterations = 10
# coords = W.tabulate_dof_coordinates()[:, 0]
# sort_order = np.argsort(coords)
# solutions = np.zeros((max_iterations + 1, len(coords)))
# solutions[0] = U.x.array[sort_order]

while i < max_iterations:
    print("Iteration", i)
    # Assemble Jacobian and residual
    with L.localForm() as loc_L:
        loc_L.set(0)
    A.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(A, jacobian)
    A.assemble()
    dolfinx.fem.petsc.assemble_vector(L, residual)
    L.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    # Scale residual by -1
    L.scale(-1)
    L.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

    # Solve linear problem
    print("starting solve")
    solver.solve(L, du.vector)
    print("finished solve")
    du.x.scatter_forward()
    # Update u_{i+1} = u_i + delta u_i
    U.x.array[:] += du.x.array
    i += 1

    # Compute norm of update
    correction_norm = du.vector.norm(0)
    print(f"Iteration {i}: Correction norm {correction_norm}")
    if correction_norm < 1e-5:
        break
    # solutions[i, :] = U.x.array[sort_order]

# Split the mixed solution and collapse
q, T = U.sub(0).collapse(), U.sub(1).collapse()
MPI.COMM_WORLD.barrier()
# Compute norms
norm_q, norm_T = la.norm(q.x), la.norm(T.x)

if MPI.COMM_WORLD.rank == 0:
    print(f"(D) Norm of flux coefficient vector (monolithic, direct): {norm_q}")
    print(f"(D) Norm of temp coefficient vector (monolithic, direct): {norm_T}")

    # Ensure that plotting is only done on rank 0 (the root process)

    # Create an XDMF file writer
    # with io.XDMFFile(msh.comm, "scalar_field.xdmf", "w") as xdmf:
    #     # Write the mesh to the XDMF file
    #     xdmf.write_mesh(msh)

    #     # Write the scalar field to the XDMF file
    #     xdmf.write_function(p)

    # # vector field plotting
    import pyvista as pv
    from dolfinx.plot import vtk_mesh
    # error if NaN or Inf is detected in the solution or in norm
    if np.any(np.isnan(T.x.array)) or np.any(np.isinf(T.x.array)) or np.any(np.isnan(q.x.array)) or np.any(np.isinf(q.x.array)):
        print("Warning: NaN or Inf detected in the solution!")
        # throw error
        exit(1)

    topology, cell_types, geometry = vtk_mesh(msh, msh.topology.dim)
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)
    grid.point_data["T"] = T.x.array.real
    grid.set_active_scalars("T")

    # Plot the scalar field
    plotter = pv.Plotter()
    plotter.add_mesh(grid, cmap="coolwarm", show_edges=False)
    plotter.show()
    
    # Vector field
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

