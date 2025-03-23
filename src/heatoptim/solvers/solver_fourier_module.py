import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh
import dolfinx.fem.petsc  # ghost import
from heatoptim.utilities.image_processing import img_list_to_gamma_expression
import numpy as np


class FourierSolver:
    def __init__(self, msh, facet_markers, config):
        self.msh = msh
        self.config = config
        self.facet_markers = facet_markers

        # Set up function space and function for temperature T
        self.V = fem.functionspace(msh, ("CG", 1))
        self.T = fem.Function(self.V)
        self.T.name = "Temperature"
        self.T.x.array[:] = 0.0  # Initialize T

        # Set up boundary conditions and measures
        self.define_boundary_conditions()

        # Function for material property gamma
        V_gamma = fem.functionspace(msh, ("CG", 1))
        self.gamma = fem.Function(V_gamma)

    def define_boundary_conditions(self):
        # Define boundary conditions and measures
        self.ds = ufl.Measure("ds", domain=self.msh, subdomain_data=self.facet_markers)
        self.ds_bottom = self.ds(1)  # Isothermal Boundary
        self.ds_slip = self.ds(2)    # Slip Boundary

        # Collect top boundary measures based on source positions
        num_sources = len(self.config.source_positions)
        top_tags = list(range(3, 3 + num_sources))  # Tags start at 3
        self.ds_tops = [self.ds(tag) for tag in top_tags]

        if self.config.symmetry:
            self.ds_symmetry = self.ds(4)

        # Set up Dirichlet boundary condition at the bottom (isothermal boundary)
        T_D_bottom = fem.Constant(self.msh, 0.0)
        facets_bottom = mesh.locate_entities_boundary(
            self.msh, self.msh.topology.dim - 1, lambda x: np.isclose(x[1], 0.0, atol=1e-8)
        )
        dofs_bottom = fem.locate_dofs_topological(self.V, self.msh.topology.dim - 1, facets_bottom)
        self.bc_bottom = fem.dirichletbc(T_D_bottom, dofs_bottom, self.V)

        # Collect all Dirichlet boundary conditions
        self.bcs = [self.bc_bottom]

    def define_variational_form(self):
        T = self.T  # Current temperature solution
        v = ufl.TestFunction(self.V)
        n = ufl.FacetNormal(self.msh)

        def ramp(gamma, a_min, a_max, qa=200):
            return a_min + (a_max - a_min) * gamma / (1 + qa * (1 - gamma))

        ramp_kappa = ramp(self.gamma, self.config.KAPPA_SI, self.config.KAPPA_DI)

        # Variational form for Fourier's heat conduction
        F = ramp_kappa * ufl.inner(ufl.grad(T), ufl.grad(v)) * ufl.dx

        # Include Neumann boundary conditions (source terms) on top boundaries
        source_term = sum(
            Q_i * v * ds_top_i
            for Q_i, ds_top_i in zip(self.config.Q_sources, self.ds_tops)
        )
        F += source_term

        return F

    def solve_image(self, img_list):
        # Reset temperature field T to zero to avoid accumulation from previous evaluations
        self.T.x.array[:] = 0.0

        gamma_expr = img_list_to_gamma_expression(img_list, self.config)
        self.gamma.interpolate(gamma_expr)

        # Define variational forms
        F = self.define_variational_form()
        residual = fem.form(F)
        J = ufl.derivative(F, self.T)
        jacobian = fem.form(J)

        # Solve the problem
        self.solve_problem(residual, jacobian)

        temp_form = fem.form(self.T * ufl.dx)
        temp_local = fem.assemble_scalar(temp_form)
        temp_global = temp_local
        area = self.config.L_X * self.config.L_Y + self.config.SOURCE_WIDTH * self.config.SOURCE_HEIGHT
        avg_temp_global = temp_global / area

        return avg_temp_global

    def get_std_dev(self):
        T = self.T
        # Compute the mean temperature
        temp_form = fem.form(T * ufl.dx)
        temp_local = fem.assemble_scalar(temp_form)
        temp_global = temp_local
        area = self.config.L_X * self.config.L_Y + self.config.SOURCE_WIDTH * self.config.SOURCE_HEIGHT
        mean_temp = temp_global / area

        # Compute the variance
        variance_form = fem.form((T - mean_temp) ** 2 * ufl.dx)
        variance_local = fem.assemble_scalar(variance_form)
        variance_global = variance_local / area

        # Standard deviation is the square root of the variance
        std_dev = np.sqrt(variance_global)

        return std_dev

    def solve_problem(self, residual, jacobian):
        # Create matrix and vector
        A = fem.petsc.create_matrix(jacobian)
        L = fem.petsc.create_vector(residual)

        solver = PETSc.KSP().create(self.T.function_space.mesh.comm)
        solver.setOperators(A)
        solver.setType("cg")  # Conjugate Gradient for symmetric positive-definite
        solver.setTolerances(rtol=1e-6, atol=1e-13, max_it=1000)
        pc = solver.getPC()
        pc.setType("ilu")  # Incomplete LU factorization

        solver.setFromOptions()

        # Assemble system
        with L.localForm() as loc_L:
            loc_L.set(0)
        A.zeroEntries()
        fem.petsc.assemble_matrix(A, jacobian, bcs=self.bcs)
        A.assemble()
        fem.petsc.assemble_vector(L, residual)
        fem.petsc.apply_lifting(L, [jacobian], [self.bcs])
        L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(L, self.bcs)
        L.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

        # Solve linear problem
        solver.solve(L, self.T.x.petsc_vec)
        self.T.x.scatter_forward()
