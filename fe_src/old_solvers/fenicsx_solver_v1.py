'''
kinda works temp distribution is just off
'''
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io, la, log
from generate_mesh_v2 import create_mesh
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
Length = 1  # Characteristic length, adjust as necessary

msh, cell_markers, facet_markers = create_mesh(Length)
# Scale the mesh coordinates back to original size
# scaling_factor = 1e-6
# msh.geometry.x[:] /= scaling_factor

ds_slip = ufl.Measure("ds", domain=msh, subdomain_data=facet_markers, subdomain_id=2)
ds_bottom = ufl.Measure("ds", domain=msh, subdomain_data=facet_markers, subdomain_id=1)
dx = ufl.Measure("dx", domain=msh, subdomain_data=cell_markers)
dx_source = dx(2)

# # CHECK DX SOURCE
# check_form = fem.form(PETSc.ScalarType(1) * dx_source)
# check_local = fem.assemble_scalar(check_form)  # assemble over cell
# totat_check = msh.comm.allreduce(check_local, op=MPI.SUM)
# print(f"Integral of 1 over source region: {totat_check}")
# print(f"Should be: {1 * 0.25}")

# # CHECK ISOTHERMAL LINE
# check_form = fem.form(PETSc.ScalarType(1) * ds_bottom)
# check_local = fem.assemble_scalar(check_form)  # assemble over cell
# totat_check = msh.comm.allreduce(check_local, op=MPI.SUM)
# print(f"Integral of 1 over isothermal line: {totat_check}")
# print(f"Should be: {25 * 1}")

# # CHECK SLIP LINE
# check_form = fem.form(PETSc.ScalarType(1) * ds_slip)
# check_local = fem.assemble_scalar(check_form)  # assemble over cell
# totat_check = msh.comm.allreduce(check_local, op=MPI.SUM)
# print(f"Integral of 1 over slip line: {totat_check}")
# print(f"Should be: {25 + 12.5 * 2}")

# Function spaces for temperature and heat flux
# Create the Taylot-Hood function space
P2 = basix.ufl.element("CG", msh.basix_cell(), 2, shape=(msh.geometry.dim,))
P1 = basix.ufl.element("CG", msh.basix_cell(), 1)  # for temperature
TH = basix.ufl.mixed_element([P2, P1])
W = fem.functionspace(msh, TH)

# Define trial and test functions
# Define variational problem
(q, T) = ufl.TrialFunctions(W)  # q is heat flux, T is temperature
(v, s) = ufl.TestFunctions(W)  # s is scalar, v is vector

# Bilinear form components

# Term: ∫ q ⋅ ∇ s dΩ
# a_continuity = ufl.inner(ufl.div(q), s) * ufl.dx
a_continuity = ufl.dot(q, ufl.grad(s)) * ufl.dx

# Term: ∫ v ⋅ q dΩ
# a_flux = ufl.inner(v, q) * ufl.dx
a_flux = ufl.dot(v, q) * ufl.dx

# Term: κ ∫ v ⋅ ∇ T dΩ
# a_conduction = T * ufl.div(v) * ufl.dx
a_conduction = -kappa * T * ufl.div(v) * ufl.dx

# Term: ℓ² ∫ ∇ v : ∇ q dΩ
a_laplacian_q = ell**2 * ufl.inner(ufl.grad(v), ufl.grad(q)) * ufl.dx

# Combine all terms to form the bilinear form a(U, V)
a = a_continuity + a_flux + a_conduction + a_laplacian_q


n = ufl.FacetNormal(msh)
# SLIP CONDITION #
# Tangential component of v and q
# v_t = v - ufl.outer(v, n) * n  # Project v onto the tangent plane
# q_t = q - ufl.outer(q, n) * n  # Project q onto the tangent plane

# Slip boundary term
# slip_q_t = (ell ** 2 * (ufl.grad(q) + 2 * ufl.div(q) * ufl.Identity(2))
#             - kappa * T * ufl.Identity(2)) * n
# slip_v_s = (ell ** 2 * (ufl.grad(v) + 2 * ufl.div(v) * ufl.Identity(2))
#             - kappa * s * ufl.Identity(2)) * n
# slip_term = (- ufl.dot(n, slip_q_t) * ufl.dot(v, n) * ds_slip
#              - ufl.dot(q, n) * ufl.dot(n, slip_v_s) * ds_slip
#              + ell * ufl.dot(q_t, v_t) * ds_slip
#              + ell**2 * ufl.dot(q, n) * ufl.dot(v, n) * ds_slip
#              )

v_t = v - ufl.dot(v, n) * n
q_t = q - ufl.dot(q, n) * n
sigma = ell**2 * (ufl.grad(q) + (2/3) * ufl.div(q) * ufl.Identity(msh.topology.dim)) - kappa * T * ufl.Identity(msh.topology.dim)
lambda_ = ufl.dot(sigma, n)
slip_term = (
    - ufl.dot(lambda_, v) * ds_slip
    - ufl.dot(q, n) * ufl.dot(v, n) * ds_slip
    + ell * ufl.dot(q_t, v_t) * ds_slip
)
# Add the slip term to the bilinear form
a += slip_term


# ISOTHERMAL TERM #
# Normal component of q
q_n = ufl.dot(q, n)
# Boundary term for isothermal condition
T_iso_form = fem.Constant(msh, T_iso)
isothermal_term = (T - T_iso + R * q_n) * s * ds_bottom
# Add the isothermal term to the bilinear form
# a += isothermal_term


# SOURCE TERM #
# Q = PETSc.ScalarType(1e5)  # Adjust Q as necessary
Q = PETSc.ScalarType(-1e4)  # Adjust Q as necessary
L = s * Q * dx_source


# BOUNDARY CONDITIONS #
# Lid velocity
def t_val_expression(x):
    return np.full(x.shape[1], T_iso)


# set up boundary conditions
W1 = W.sub(1)
Q_t, _ = W1.collapse()
t_zero = fem.Function(Q_t)
# interpolate 0 onto t_zero
t_zero.interpolate(t_val_expression)
isothermal_facets = facet_markers.find(1)
t_dofs = fem.locate_dofs_topological((W1, Q_t), 1, isothermal_facets)
isothermal_bc = fem.dirichletbc(t_zero, t_dofs, W1)
# bcs = []
bcs = [isothermal_bc]


a = fem.form(a)
L = fem.form(L)
# Assemble LHS matrix and RHS vector
A = fem.petsc.assemble_matrix(a, bcs=bcs)
A.assemble()
b = fem.petsc.assemble_vector(L)

fem.petsc.apply_lifting(b, [a], bcs=[bcs])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

# Set Dirichlet boundary condition values in the RHS
fem.petsc.set_bc(b, bcs)

# Create and configure solver
ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")

log.set_log_level(log.LogLevel.INFO)

# Configure MUMPS to handle pressure nullspace
pc = ksp.getPC()
pc.setType("lu")
pc.setFactorSolverType("mumps")
pc.setFactorSetUpSolverType()
pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)

# Compute the solution
U = fem.Function(W)
MPI.COMM_WORLD.barrier()
try:
    ksp.solve(b, U.x.petsc_vec)
except PETSc.Error as e:
    if e.ierr == 92:
        print("The required PETSc solver/preconditioner is not available. Exiting.")
        print(e)
        exit(0)
    else:
        raise e

# Split the mixed solution and collapse
u, p = U.sub(0).collapse(), U.sub(1).collapse()
MPI.COMM_WORLD.barrier()
# Compute norms
norm_u, norm_p = la.norm(u.x), la.norm(p.x)

if MPI.COMM_WORLD.rank == 0:
    print(f"(D) Norm of velocity coefficient vector (monolithic, direct): {norm_u}")
    print(f"(D) Norm of pressure coefficient vector (monolithic, direct): {norm_p}")

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
    if np.any(np.isnan(p.x.array)) or np.any(np.isinf(p.x.array)) or np.any(np.isnan(u.x.array)) or np.any(np.isinf(u.x.array)):
        print("Warning: NaN or Inf detected in the solution!")
        # throw error
        exit(1)

    topology, cell_types, geometry = vtk_mesh(msh, msh.topology.dim)
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)
    grid.point_data["T"] = p.x.array.real
    grid.set_active_scalars("T")

    # Plot the scalar field
    plotter = pv.Plotter()
    plotter.add_mesh(grid, cmap="coolwarm", show_edges=False)
    plotter.show()
    
    # Vector field
    gdim = msh.geometry.dim
    V_dg = fem.functionspace(msh, ("DG", 2, (gdim,)))
    q_dg = fem.Function(V_dg)
    q_copy = u.copy()
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
