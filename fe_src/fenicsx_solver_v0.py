import numpy as np
from mpi4py import MPI
from dolfinx import fem, log
from dolfinx.fem import Function, functionspace, dirichletbc, locate_dofs_topological
from dolfinx.io import XDMFFile
from basix.ufl import element, mixed_element
import ufl
from petsc4py import PETSc
import dolfinx.fem.petsc
import dolfinx.nls.petsc
from generate_mesh import create_mesh
from utils import (
    img_to_gamma,
    filter_function,
    plot_mesh,
    plot_boundaries,
    plot_subdomains,
    plot_scalar_field,
    plot_vector_field,
    plot_material_distribution,
)


class NonlinearPDE_SNESProblem:
    def __init__(self, F, u, bc):
        V = u.function_space
        du = ufl.TrialFunction(V)
        self.L = fem.form(F)
        self.a = fem.form(ufl.derivative(F, u, du))
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
        fem.petsc.assemble_vector(F, self.L)
        fem.petsc.apply_lifting(F, [self.a], bcs=[[self.bc]], x0=[x], scale=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(F, [self.bc], x, -1.0)

    def J(self, snes, x, J, P):
        """Assemble Jacobian matrix."""
        J.zeroEntries()
        fem.petsc.assemble_matrix(J, self.a, bcs=[self.bc])
        J.assemble()


# Load mesh
domain, cell_markers, facet_markers = create_mesh()

# plot_mesh(domain)
# plot_boundaries(domain, facet_markers)
# plot_subdomains(domain, cell_markers)

# Define material properties and constants
L = 439e-9  # Characteristic length
k_si = PETSc.ScalarType(141)  # Silicon thermal conductivity
k_di = PETSc.ScalarType(600)  # Diamond thermal conductivity
ell_si = PETSc.ScalarType(L / np.sqrt(5))
ell_di = PETSc.ScalarType(10 * L / np.sqrt(5))
source_value = PETSc.ScalarType(10)

# Define finite elements
# For vector field u
q_cg2 = element("CG", domain.ufl_cell().cellname(), 2, shape=(domain.topology.dim,))
# For scalar field p
v_cg1 = element("CG", domain.ufl_cell().cellname(), 1)

# Create mixed element and function space
w_el = mixed_element([q_cg2, v_cg1])
W = functionspace(domain, w_el)

# Define boundary conditions
# Isothermal boundary condition (p = 0 on boundary 1)
p_zero = PETSc.ScalarType(0)
isothermal_facets = facet_markers.find(1)
p_dofs = locate_dofs_topological(W.sub(1), domain.topology.dim - 1, isothermal_facets)
isothermal_bc = dirichletbc(p_zero, p_dofs, W.sub(1))

# CHECK PRESSURE IS 0

# Collect boundary conditions
# bcs = [isothermal_bc]

# Define measures with the appropriate markers
dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_markers)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_markers)

# Define boundary measures
# ds_slip = ds(2)  # Slip boundary
# ds_source = ds(3)  # Source boundary

# integrate 1 across ds_slip:
ds_slip = ufl.Measure("ds", domain=domain, subdomain_data=facet_markers, subdomain_id=2)
ds_source = ufl.Measure(
    "ds", domain=domain, subdomain_data=facet_markers, subdomain_id=3
)

# Define the normal vector
n = ufl.FacetNormal(domain)

# Define the variational problem components that are independent of 'img'
# Trial and test functions
up = Function(W)
v_q = ufl.TestFunction(W)
u, p = ufl.split(up)
v, q = ufl.split(v_q)

# Define function space for gamma
V_gamma = functionspace(domain, ("CG", 1))
gamma = Function(V_gamma)  # material field
gamma_bar = Function(V_gamma)  # filtered material field


# Define Material Properties and Equation
def ramp(gamma, a_min, a_max, qa=200):
    """
    interpolation function for material properties
    between two states (no heavy side step)
    """
    return a_min + (a_max - a_min) * gamma / (1 + qa * (1 - gamma))


# Function for t
def t_func(gamma, u, p, n):
    return (
        ramp(gamma, ell_si, ell_di) ** 2
        * (ufl.grad(u) + 2 * ufl.div(u) * ufl.Identity(2))
        - ramp(gamma, k_si, k_di) * p * ufl.Identity(2)
    ) * n


# Function for u_t
def u_t(u, n):
    return u - ufl.dot(u, n) * n


def solve_pde(img):
    # Map the image to gamma
    gamma_values = img_to_gamma(img, domain, dx)
    gamma.vector.array[:] = gamma_values
    gamma.x.scatter_forward()

    # gamma_bar = filter_function(gamma, domain, dx=dx)
    gamma_bar = gamma

    # Material properties based on gamma_bar
    kappa_expr = ramp(gamma_bar, k_si, k_di)
    ell_expr = ramp(gamma_bar, ell_si, ell_di)

    # Define the variational problem using gamma_bar
    # Continuity equation
    flux_continuity = -ufl.div(u) * q * dx

    # Pressure term
    pressure_term = -kappa_expr * p * ufl.div(v) * dx

    # Flux term
    flux_term = ufl.inner(v, u) * dx

    # Viscous term
    viscous_term = (
        ell_expr**2
        * (
            ufl.inner(ufl.grad(u), ufl.grad(v))
            + 2 * ufl.inner(ufl.div(u) * ufl.Identity(2), ufl.grad(v))
        )
        * dx
    )

    # Boundary terms
    boundary_terms = (
        -ufl.dot(n, t_func(gamma_bar, u, p, n)) * ufl.dot(v, n) * ds_slip
        - ufl.dot(u, n) * ufl.dot(n, t_func(gamma_bar, v, q, n)) * ds_slip
        + ell_expr * ufl.dot(u_t(u, n), u_t(v, n)) * ds_slip
        + ufl.dot(u, n) * ufl.dot(v, n) * ds_slip
        + source_value * ufl.dot(v, n) * ds_source
    )

    # Total variational form
    F = flux_continuity + flux_term + viscous_term + pressure_term + boundary_terms

    # Initialize the solution vector (initial guess)
    up.x.array[:] = 0.0  # Reset to zero before setting initial guesses

    # Set initial conditions for u and p separately
    u_initial = lambda x: (
        -10 * np.ones_like(x[0]),
        -10 * np.ones_like(x[0]),
    )  # Example initial condition for u
    p_initial = lambda x: 0.4 * np.ones_like(x[0])  # Example initial condition for p

    up.sub(0).interpolate(u_initial)
    up.sub(1).interpolate(p_initial)

    up.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    # Define the nonlinear problem
    problem = dolfinx.fem.petsc.NonlinearProblem(F, up, bcs=[isothermal_bc])

    # Create the Newton solver
    solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-6
    solver.report = True  # Set to True to see solver progress

    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()

    opts[f"{option_prefix}ksp_type"] = "gmres"
    opts[f"{option_prefix}pc_type"] = "lu"  # LU decomposition
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "umfpack"  # Use MUMPS solver
    ksp.setFromOptions()

    # Solve the problem
    log.set_log_level(log.LogLevel.INFO)
    n_iter, converged = solver.solve(up)
    if not converged:
        print("Newton solver did not converge after", n_iter, "iterations")
    else:
        print("Newton solver converged in", n_iter, "iterations")

    # Extract solutions
    u_h, p_h = up.split()
    u_h.name = "u"
    p_h.name = "p"

    # Create an XDMF file writer
    with XDMFFile(domain.comm, "scalar_field.xdmf", "w") as xdmf:
        # Write the mesh to the XDMF file
        xdmf.write_mesh(domain)

        # Write the scalar field to the XDMF file
        xdmf.write_function(p_h)

    plot_material_distribution(domain, gamma_bar)

    V_p = fem.functionspace(domain, ("CG", 1))
    p_h_proj = Function(V_p)
    p_h_proj.interpolate(p_h)
    plot_scalar_field(domain, p_h_proj)

    # q_cg2 = element("CG", domain.ufl_cell().cellname(), 1, shape=(domain.topology.dim,))
    # V_u = fem.functionspace(domain, q_cg2)
    # u_h_proj = project_vector_field(domain, u_h)
    # plot_vector_field(domain, u_h_proj, V_u)

    return u_h, p_h


def project_vector_field(domain, u_h):
    # Define a separate CG1 vector function space for u
    q_cg2 = element("CG", domain.ufl_cell().cellname(), 1, shape=(domain.topology.dim,))
    V_u = fem.functionspace(domain, q_cg2)

    # Define trial and test functions
    u_trial = ufl.TrialFunction(V_u)
    v_test = ufl.TestFunction(V_u)

    # Define the variational problem for projection (L2 projection)
    a = ufl.inner(u_trial, v_test) * ufl.dx
    L = ufl.inner(u_h, v_test) * ufl.dx

    # Assemble the system
    A = fem.assemble_matrix(a)
    A.assemble()
    b = fem.assemble_vector(L)

    # Create a Function to hold the projected vector field
    u_proj = Function(V_u)

    # Solve the linear system
    problem = dolfinx.fem.petsc.LinearProblem(
        a, L, u_proj, petsc_options={"ksp_type": "cg", "pc_type": "jacobi"}
    )
    u_proj = problem.solve()

    return u_proj


# Example usage
if __name__ == "__main__":
    # Create a sample image (for testing purposes)
    img_height, img_width = 100, 100
    img = np.zeros((img_height, img_width))

    # First call to solve_pde (the setup is done before this)
    u_h, p_h = solve_pde(img)


"""
    # Define the nonlinear problem
    problem = dolfinx.fem.petsc.NonlinearProblem(F, up, bcs)

    # Create the Newton solver
    solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-6
    solver.report = True  # Set to True to see solver progress

    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()

    opts[f"{option_prefix}ksp_type"] = "cg"
    opts[f"{option_prefix}pc_type"] = "lu"  # LU decomposition
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "umfpack"  # Use MUMPS solver
    ksp.setFromOptions()

    # Solve the problem
    log.set_log_level(log.LogLevel.INFO)
    n_iter, converged = solver.solve(up)
    if not converged:
        print("Newton solver did not converge after", n_iter, "iterations")
    else:
        print("Newton solver converged in", n_iter, "iterations")
"""
