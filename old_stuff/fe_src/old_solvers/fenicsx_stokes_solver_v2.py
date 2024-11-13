from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import ufl
from basix.ufl import element, mixed_element
from dolfinx import fem, la
from dolfinx.fem import (
    Constant,
    Function,
    dirichletbc,
    extract_function_spaces,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_rectangle, locate_entities_boundary
from ufl import div, dx, grad, inner
from generate_mesh import create_mesh
from utils import img_to_gamma

L = 439e-9  # Characteristic length
# Create mesh
msh, cell_markers, facet_markers = create_mesh(L=L)

ds_slip = ufl.Measure("ds", domain=msh, subdomain_data=facet_markers, subdomain_id=2)
ds_source = ufl.Measure(
    "ds", domain=msh, subdomain_data=facet_markers, subdomain_id=3
)

# Create the Taylot-Hood function space
P2 = element("Lagrange", msh.basix_cell(), 2, shape=(msh.geometry.dim,))
P1 = element("Lagrange", msh.basix_cell(), 1)
TH = mixed_element([P2, P1])
W = functionspace(msh, TH)

# No slip boundary condition
W1 = W.sub(1)
# Q is P2 (u), V is P1 (p)
Q, _ = W1.collapse()


# Define variational problem
(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)
# f = Function(Q)


# Lid velocity
def t_zero_expression(x):
    return np.stack(300*np.ones(x.shape[0]))


# set up boundary conditions
t_zero = Function(Q)
# interpolate 0 onto t_zero
t_zero.interpolate(t_zero_expression)
isothermal_facets = facet_markers.find(1)
t_dofs = locate_dofs_topological((W1, Q), 1, isothermal_facets)
isothermal_bc = dirichletbc(t_zero, t_dofs, W1)
bcs = [isothermal_bc]

k_si = PETSc.ScalarType(141)  # Silicon thermal conductivity
k_di = PETSc.ScalarType(600)  # Diamond thermal conductivity
ell_si = PETSc.ScalarType(L / np.sqrt(5))
ell_di = PETSc.ScalarType(10 * L / np.sqrt(5))
source_value = PETSc.ScalarType(10)


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


# Define function space for gamma
V_gamma = functionspace(msh, ("CG", 1))
gamma = Function(V_gamma)  # material field
img = np.zeros((100, 100))
gamma_values = img_to_gamma(img, msh, dx)
gamma.vector.array[:] = gamma_values
gamma.x.scatter_forward()

# Material properties based on gamma_bar
kappa_expr = ramp(gamma, k_si, k_di)
ell_expr = ramp(gamma, ell_si, ell_di)

n = ufl.FacetNormal(msh)
# (u, p) = ufl.TrialFunctions(W)
# (v, q) = ufl.TestFunctions(W)
# Bilinear form 'a'
_a = (
    # Flux continuity term
    - inner(ufl.div(u), q) * dx
    # Flux term
    + ufl.inner(v, u) * dx
    # Viscous term
    + ell_expr**2 * (
        ufl.inner(ufl.grad(u), ufl.grad(v))
        + 2 * ufl.inner(ufl.div(u) * ufl.Identity(2), ufl.grad(v))
    ) * dx
    # Pressure term
    - kappa_expr * p * ufl.div(v) * dx
    # Boundary terms (excluding the source term)
    - ufl.dot(n, t_func(gamma, u, p, n)) * ufl.dot(v, n) * ds_slip
    - ufl.dot(u, n) * ufl.dot(n, t_func(gamma, v, q, n)) * ds_slip
    + ell_expr * ufl.dot(u_t(u, n), u_t(v, n)) * ds_slip
    + ufl.dot(u, n) * ufl.dot(v, n) * ds_slip
)

# Linear form 'L'
_L = (
    # Source term
    source_value * ufl.dot(v, n) * ds_source
)


def mixed_direct():

    a = fem.form(_a)
    L = fem.form(_L)

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

    # Configure MUMPS to handle pressure nullspace
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    pc.setFactorSetUpSolverType()
    pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
    pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)

    # Compute the solution
    U = Function(W)
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

    # Compute norms
    norm_u, norm_p = la.norm(u.x), la.norm(p.x)
    if MPI.COMM_WORLD.rank == 0:
        print(f"(D) Norm of velocity coefficient vector (monolithic, direct): {norm_u}")
        print(f"(D) Norm of pressure coefficient vector (monolithic, direct): {norm_p}")

    return norm_u, norm_u


norm_u_3, norm_p_3 = mixed_direct()

