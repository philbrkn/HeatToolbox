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


P2 = element("Lagrange", msh.basix_cell(), 2, shape=(msh.geometry.dim,))
P1 = element("Lagrange", msh.basix_cell(), 1)
V, Q = functionspace(msh, P2), functionspace(msh, P1)


# Lid velocity
def t_zero_expression(x):
    return np.stack(np.zeros(x.shape[0]))


# set up boundary conditions
t_zero = Function(Q)
# interpolate 0 onto t_zero
t_zero.interpolate(t_zero_expression)
isothermal_facets = facet_markers.find(1)
t_dofs = locate_dofs_topological(Q, 1, isothermal_facets)
isothermal_bc = dirichletbc(t_zero, t_dofs)
bcs = [isothermal_bc]

# integrate 1 across ds_slip:
ds_slip = ufl.Measure("ds", domain=msh, subdomain_data=facet_markers, subdomain_id=2)
ds_source = ufl.Measure("ds", domain=msh, subdomain_data=facet_markers, subdomain_id=3)

# Define variational problem
(u, p) = ufl.TrialFunction(V), ufl.TrialFunction(Q)
(v, q) = ufl.TestFunction(V), ufl.TestFunction(Q)

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

t_u = (
    ell_expr**2 * (ufl.grad(u) + 2 * ufl.div(u) * ufl.Identity(2))
) * n

t_p = (
    - kappa_expr * p * ufl.Identity(2)
) * n
t_v = (
    ell_expr**2 * (ufl.grad(v) + 2 * ufl.div(v) * ufl.Identity(2))
) * n
t_q = (
    - kappa_expr * q * ufl.Identity(2)
) * n
a_ = [
    [
        # a[0][0]: Terms involving u (trial) and v (test)
        (
            # Flux term
            ufl.inner(v, u) * dx
            # Viscous term
            + ell_expr**2 * (
                ufl.inner(ufl.grad(u), ufl.grad(v))
                + 2 * ufl.inner(ufl.div(u) * ufl.Identity(2), ufl.grad(v))
            ) * dx
            # Boundary terms
            - ufl.dot(n, t_u) * ufl.dot(v, n) * ds_slip
            - ufl.dot(u, n) * ufl.dot(n, t_v) * ds_slip
            + ell_expr * ufl.dot(u_t(u, n), u_t(v, n)) * ds_slip
            + ufl.dot(u, n) * ufl.dot(v, n) * ds_slip
        ),
        # a[0][1]: Terms involving p (trial) and v (test)
        (
            # Pressure term
            - kappa_expr * p * ufl.div(v) * dx
            # Boundary term
            - ufl.dot(n, t_p) * ufl.dot(v, n) * ds_slip
        ),
    ],
    [
        # a[1][0]: Terms involving u (trial) and q (test)
        (
            # Flux continuity term
            - ufl.div(u) * q * dx
            # Boundary term
            - ufl.dot(u, n) * ufl.dot(n, t_q) * ds_slip
        ),
        # a[1][1]: Terms involving p (trial) and q (test)
        inner(p, q) * dx  # Define a suitable bilinear form for p-q
    ],
]
eps = 1e-8  # Small parameter to regularize the block
a_[1][1] = eps * ufl.inner(p, q) * dx

# Linear form 'L' as a block vector
L_ = [
    # L[0]: Right-hand side for the first equation (test function v)
    source_value * ufl.dot(v, n) * ds_source,
    # L[1]: Right-hand side for the second equation (test function q)
    inner(Constant(msh, PETSc.ScalarType(0)), q) * dx  # Assuming zero for the second equation
]

# Create the bilinear and linear forms
a_form = fem.form(a_)
L_form = fem.form(L_)


a_p11 = form(inner(p, q) * dx)
a_p = [[a_form[0][0], None], [None, a_p11]]


def block_operators():
    """Return block operators and block RHS vector for the Stokes
    problem"""

    # Assembler matrix operator, preconditioner and RHS vector into
    # single objects but preserving block structure
    A = assemble_matrix_block(a_form, bcs=bcs)
    A.assemble()
    P = assemble_matrix_block(a_p, bcs=bcs)
    P.assemble()
    b = assemble_vector_block(L_form, a_form, bcs=bcs)

    # Set the nullspace for pressure (since pressure is determined only
    # up to a constant)
    null_vec = A.createVecLeft()
    offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    null_vec.array[offset:] = 1.0
    null_vec.normalize()
    nsp = PETSc.NullSpace().create(vectors=[null_vec])
    assert nsp.test(A)
    A.setNullSpace(nsp)

    return A, P, b


def block_iterative_solver():
    """Solve the Stokes problem using blocked matrices and an iterative
    solver."""

    # Assembler the operators and RHS vector
    A, P, b = block_operators()

    # Build PETSc index sets for each field (global dof indices for each
    # field)
    V_map = V.dofmap.index_map
    Q_map = Q.dofmap.index_map
    offset_u = V_map.local_range[0] * V.dofmap.index_map_bs + Q_map.local_range[0]
    offset_p = offset_u + V_map.size_local * V.dofmap.index_map_bs
    is_u = PETSc.IS().createStride(
        V_map.size_local * V.dofmap.index_map_bs, offset_u, 1, comm=PETSc.COMM_SELF
    )
    is_p = PETSc.IS().createStride(Q_map.size_local, offset_p, 1, comm=PETSc.COMM_SELF)

    # Create a MINRES Krylov solver and a block-diagonal preconditioner
    # using PETSc's additive fieldsplit preconditioner
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A, P)
    ksp.setTolerances(rtol=1e-9)
    ksp.setType("minres")
    ksp.getPC().setType("fieldsplit")
    ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
    ksp.getPC().setFieldSplitIS(("u", is_u), ("p", is_p))

    # Configure velocity and pressure sub-solvers
    ksp_u, ksp_p = ksp.getPC().getFieldSplitSubKSP()
    ksp_u.setType("preonly")
    ksp_u.getPC().setType("gamg")
    ksp_p.setType("preonly")
    ksp_p.getPC().setType("jacobi")

    # The matrix A combined the vector velocity and scalar pressure
    # parts, hence has a block size of 1. Unlike the MatNest case, GAMG
    # cannot infer the correct near-nullspace from the matrix block
    # size. Therefore, we set block size on the top-left block of the
    # preconditioner so that GAMG can infer the appropriate near
    # nullspace.
    ksp.getPC().setUp()
    Pu, _ = ksp_u.getPC().getOperators()
    Pu.setBlockSize(msh.topology.dim)

    # Create a block vector (x) to store the full solution and solve
    x = A.createVecRight()
    ksp.solve(b, x)

    # Create Functions to split u and p
    u, p = Function(V), Function(Q)
    offset = V_map.size_local * V.dofmap.index_map_bs
    u.x.array[:offset] = x.array_r[:offset]
    p.x.array[: (len(x.array_r) - offset)] = x.array_r[offset:]

    # Compute the $L^2$ norms of the solution vectors
    norm_u, norm_p = la.norm(u.x), la.norm(p.x)
    if MPI.COMM_WORLD.rank == 0:
        print(f"(B) Norm of velocity coefficient vector (blocked, iterative): {norm_u}")
        print(f"(B) Norm of pressure coefficient vector (blocked, iterative): {norm_p}")

    return norm_u, norm_p


# Solve using PETSc MatNest
norm_u_0, norm_p_0 = block_iterative_solver()
