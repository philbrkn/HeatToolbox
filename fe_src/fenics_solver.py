from fenics import (
    SubDomain,
    FunctionSpace,
    Function,
    near,
    between,
    inner,
    grad,
    div,
    dot,
    DirichletBC,
    NonlinearVariationalProblem,
    NonlinearVariationalSolver,
    derivative,
    FacetNormal,
    MeshFunction,
    Point,
    VectorElement,
    FiniteElement,
    MixedElement,
    refine,
    Constant,
    Identity,
    split,
    TestFunction,
    sqrt,
    Measure,
    DOLFIN_EPS,
    plot,
    interpolate
)
from mshr import Rectangle, generate_mesh
# from fe_src.utils import img_to_gamma, filter_function
import matplotlib.pyplot as plt

# Define constants
L = 439e-9  # Characteristic length
Lx, Ly = 25 * L, 12.5 * L
width, height = 0.5 * L, 0.25 * L
resolution = 64
niter = 500
k_si = Constant(141)  # Silicon thermal conductivity
k_di = Constant(600)  # Diamond thermal conductivity
ell_si = L / sqrt(5)
ell_di = 10 * L / sqrt(5)
source_value = Constant(10)

# Define the domain and source
domain = Rectangle(Point(0, 0), Point(Lx, Ly))
source_region = Rectangle(
    Point(0.5 * Lx - 0.5 * width, Ly), Point(0.5 * Lx + 0.5 * width, Ly + height)
)
domain += source_region

# # Generate and refine mesh
mesh = generate_mesh(domain, resolution)
mesh = refine(mesh)


# SubDomain for the base domain
class BaseDomain(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] <= Ly + DOLFIN_EPS  # Base domain is up to y = Ly


# SubDomain for the source region
class SourceRegion(SubDomain):
    def inside(self, x, on_boundary):
        x_min = 0.5 * Lx - 0.5 * width
        x_max = 0.5 * Lx + 0.5 * width
        y_min = Ly
        y_max = Ly + height
        return between(x[0], (x_min, x_max)) and between(
            x[1], (y_min, y_max + DOLFIN_EPS)
        )


# Initialize MeshFunction for subdomains
subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
subdomains.set_all(0)  # Assign default value of 0 to all cells

# Mark subdomains
base_domain = BaseDomain()
source_region = SourceRegion()

base_domain.mark(subdomains, 1)  # Mark base domain with 1
source_region.mark(subdomains, 2)  # Mark source region with 2


# Define SubDomains
class SlipBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return (
            on_boundary
            and near(x[0], 0)
            or near(x[0], Lx)
            or near(x[1], Ly)
            or (
                between(x[0], (0.5 * Lx - 0.5 * width, 0.5 * Lx + 0.5 * width))
                and near(x[1], Ly + height)
            )
        )


class IsothermalBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0)


class SourceBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[0], (0.5 * Lx - 0.5 * width, 0.5 * Lx + 0.5 * width)) and near(
            x[1], Ly + height
        )


class CentralSection(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[0], (0, Lx)) and between(x[1], (0, Ly))


# Mark boundaries and subdomains
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
slip_boundary = SlipBoundary()
isothermal_boundary = IsothermalBoundary()
source_boundary = SourceBoundary()

slip_boundary.mark(boundaries, 1)
isothermal_boundary.mark(boundaries, 2)
source_boundary.mark(boundaries, 3)

sections = MeshFunction("size_t", mesh, mesh.topology().dim())
sections.set_all(0)
CentralSection().mark(sections, 1)

# Plot subdomains
# plot(subdomains)
# plt.title("Subdomains")
# plt.show()

# # Plot boundaries
# plot(boundaries)
# plt.title("Boundaries")
# plt.show()

# Define finite elements
Q = VectorElement("CG", mesh.ufl_cell(), 2)  # For vector field u
V = FiniteElement("CG", mesh.ufl_cell(), 1)  # For scalar field p

# Create mixed element and function space
W_elem = MixedElement([Q, V])
W = FunctionSpace(mesh, W_elem)

# Define boundary conditions and measures
isothermal_bc = DirichletBC(W.sub(1), Constant(0), boundaries, 2)
bcs = [isothermal_bc]

dx = Measure("dx", domain=mesh, subdomain_data=subdomains)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

ds_slip = ds(1)
ds_isothermal = ds(2)
ds_source = ds(3)

# dx_domain = dx(1)

n = FacetNormal(mesh)


# Define Material Properties and Equation
def ramp(gamma, a_min, a_max, qa=200):
    return a_min + (a_max - a_min) * gamma / (1 + qa * (1 - gamma))


# function for t
def t(gamma, u, p):
    return (
        ramp(gamma, ell_si, ell_di) ** 2 * (grad(u) + 2 * div(u) * Identity(2))
        + -ramp(gamma, k_si, k_di) * p * Identity(2)
    ) * n


# function for u_t
def u_t(u):
    return u - dot(u, n) * n


def solve_pde(img):  # GO IMAGE INSTEAD
    # Map image to gamma
    # gamma = img_to_gamma(img, mesh)
    # gamma_bar = filter_function(gamma, r_min=mesh.hmin() / 20, projection=True)
    gamma = interpolate(Constant(0.0), W.sub(1).collapse())
    # Define variational problem
    up = Function(W)
    v_q = TestFunction(W)
    u, p = split(up)
    v, q = split(v_q)

    # Material properties interpolation
    k = ramp(gamma, k_si, k_di)
    ell = ramp(gamma, ell_si, ell_di)

    # Define the PDE using gamma_bar
    # Continuity equation
    flux_continuity = -div(u) * q * dx
    # Pressure term
    pressure_term = -k * p * div(v) * dx
    # Flux term
    flux_term = inner(u, v) * dx
    # Viscous term
    viscous_term = (
        ell**2
        * (inner(grad(u), grad(v)) + 2 * inner(div(u) * Identity(2), grad(v)))
        * dx
    )
    # Boundary terms
    boundary_terms = (
        -dot(n, t(gamma, u, p)) * dot(v, n) * ds_slip
        - dot(u, n) * dot(n, t(gamma, v, q)) * ds_slip
        + ell * dot(u_t(u), u_t(v)) * ds_slip
        + dot(u, n) * dot(v, n) * ds_slip
        + source_value * dot(v, n) * ds_source
    )

    F = flux_continuity + flux_term + viscous_term + pressure_term + boundary_terms

    J = derivative(F, up)
    problem = NonlinearVariationalProblem(F, up, bcs, J)
    solver = NonlinearVariationalSolver(problem)
    solver.parameters["nonlinear_solver"] = "snes"
    solver.parameters["snes_solver"]["linear_solver"] = "umfpack"

    solver.solve()

    u_h, p_h = up.split(deepcopy=True)

    return u_h, p_h


if __name__ == "__main__":
    u_h, p_h = solve_pde(None)

    plot(p_h)
    plt.title("temp Field")
    plt.show()
