import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh
import dolfinx.fem.petsc  # ghost import
from image_processing import img_list_to_gamma_expression
import basix.ufl
import numpy as np


class Solver:
    def __init__(self, msh, facet_markers, config):
        self.msh = msh
        self.config = config
        self.facet_markers = facet_markers

        # Set up function spaces and functions
        P2 = basix.ufl.element("CG", msh.basix_cell(), 2, shape=(msh.geometry.dim,))
        P1 = basix.ufl.element("CG", msh.basix_cell(), 1)
        TH = basix.ufl.mixed_element([P2, P1])
        self.W = fem.functionspace(msh, TH)

        self.U = fem.Function(self.W)  # Current solution (q, T)
        self.dU = ufl.TrialFunction(self.W)  # Increment (delta q, delta T)
        (self.v, self.s) = ufl.TestFunctions(self.W)
        self.U.x.array[:] = 0.0  # Initialize U

        # Set up boundary conditions and measures
        self.define_boundary_conditions()
        self.perform_mesh_checks()

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

        # Set up boundary condition functions and Dirichlet boundary conditions
        W1 = self.W.sub(1)
        Q1, _ = W1.collapse()
        noslip = fem.Function(Q1)  # Set a constant function for noslip condition

        rtol = self.config.LENGTH * 1e-3
        facets = mesh.locate_entities(self.msh, 1, lambda x: np.isclose(x[1], 0.0, rtol=rtol))
        dofs = fem.locate_dofs_topological((W1, Q1), 1, facets)
        self.bc0 = fem.dirichletbc(noslip, dofs, W1)

    def define_variational_form(self):
        q, T = ufl.split(self.U)
        n = ufl.FacetNormal(self.msh)

        def ramp(gamma, a_min, a_max, qa=200):
            return a_min + (a_max - a_min) * gamma / (1 + qa * (1 - gamma))

        ramp_kappa = ramp(self.gamma, self.config.KAPPA_SI, self.config.KAPPA_DI)
        ramp_ell = ramp(self.gamma, self.config.ELL_SI, self.config.ELL_DI)

        viscous_term = (
            ramp_ell ** 2
            * (
                ufl.inner(ufl.grad(q), ufl.grad(self.v))
                + 2 * ufl.inner(ufl.div(q) * ufl.Identity(2), ufl.grad(self.v))
            )
            * ufl.dx
        )

        flux_continuity = -ufl.div(q) * self.s * ufl.dx
        pressure_term = - ramp_kappa * T * ufl.div(self.v) * ufl.dx
        flux_term = ufl.inner(q, self.v) * ufl.dx

        # Source term at the top boundaries
        source_term = sum(
            Q_i * ufl.dot(self.v, n) * ds_top_i
            for Q_i, ds_top_i in zip(self.config.Q_sources, self.ds_tops)
        )

        def u_t(q):
            return q - ufl.dot(q, n) * n

        def t(q, T):
            return (
                ramp_ell ** 2 * (ufl.grad(q) + 2 * ufl.div(q) * ufl.Identity(2))
                - ramp_kappa * T * ufl.Identity(2)
            ) * n

        F = (
            flux_continuity  # Continuity: ∇⋅q = 0
            + flux_term  # Flux term: q ⋅ v
            + viscous_term  # Viscous-like term
            + pressure_term  # Pressure-like term from ∇T
            - ufl.dot(n, t(q, T)) * ufl.dot(self.v, n) * self.ds_slip  # Slip boundary condition term
            - ufl.dot(q, n) * ufl.dot(n, t(self.v, self.s)) * self.ds_slip  # Slip boundary condition term
            + ramp_ell * ufl.dot(u_t(q), u_t(self.v)) * self.ds_slip  # Slip boundary stabilization term
            + ufl.dot(q, n) * ufl.dot(self.v, n) * self.ds_slip  # Additional stabilization
            + source_term
        )

        # Symmetry boundary condition if enabled
        if self.config.symmetry:
            F += ufl.dot(q, n) * ufl.dot(self.v, n) * self.ds_symmetry

        return F

    def solve_image(self, img_list):
        gamma_expr = img_list_to_gamma_expression(img_list, self.config)
        # gamma_expr = img_to_gamma_expression(img_list[0], self.config)
        self.gamma.interpolate(gamma_expr)

        # Define variational forms
        F = self.define_variational_form()

        # Solve the problem
        self.solve_problem(F)

        q, T = self.U.sub(0).collapse(), self.U.sub(1).collapse()

        temp_form = fem.form(T * ufl.dx)
        temp_local = fem.assemble_scalar(temp_form)
        temp_global = self.msh.comm.allreduce(temp_local, op=MPI.SUM)
        area = self.config.L_X * self.config.L_Y + self.config.SOURCE_WIDTH * self.config.SOURCE_HEIGHT
        avg_temp_global = temp_global / area
        return avg_temp_global

    def solve_problem(self, F):
        residual = fem.form(F)
        J = ufl.derivative(F, self.U, self.dU)
        jacobian = fem.form(J)

        # Create matrix and vector
        A = dolfinx.fem.petsc.create_matrix(jacobian)
        L = dolfinx.fem.petsc.create_vector(residual)

        solver = PETSc.KSP().create(self.U.function_space.mesh.comm)
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
        factor_mat.setMumpsCntl(1, 1e-9)  # Relative pivoting scale
        factor_mat.setMumpsCntl(3, 1e-9)  # Absolute pivoting scale
        factor_mat.setMumpsIcntl(1, -1)  # Print all error messages
        factor_mat.setMumpsIcntl(2, 3)  # Enable diagnostic printing stats and warnings
        factor_mat.setMumpsIcntl(4, 0)  # Set print level verbosity (0-4)

        solver.setFromOptions()

        du = fem.Function(self.W)

        i = 0
        max_iterations = 5
        while i < max_iterations:
            # Assemble Jacobian and residual
            with L.localForm() as loc_L:
                loc_L.set(0)
            A.zeroEntries()
            fem.petsc.assemble_matrix(A, jacobian, bcs=[self.bc0])
            A.assemble()
            fem.petsc.assemble_vector(L, residual)
            L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            L.scale(-1)

            # Apply boundary conditions
            fem.petsc.apply_lifting(L, [jacobian], [[self.bc0]], x0=[self.U.vector], scale=1)
            fem.petsc.set_bc(L, [self.bc0], self.U.vector, 1.0)
            L.ghostUpdate(
                addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
            )

            # Solve linear problem
            solver.solve(L, du.vector)
            du.x.scatter_forward()

            # Update solution
            self.U.x.array[:] += du.x.array
            i += 1

            # Check convergence
            correction_norm = du.vector.norm(0)
            if correction_norm < 1e-5:
                break

    def perform_mesh_checks(self):
        rank = MPI.COMM_WORLD.rank
        # Check measures over specific boundaries
        checks = [
            # ("source line", self.ds_top, self.config.SOURCE_WIDTH),
            ("isothermal line", self.ds_bottom, self.config.L_X),
            ("slip line", self.ds_slip, self.config.L_Y + (self.config.L_X - self.config.SOURCE_WIDTH) + self.config.SOURCE_HEIGHT)
        ]

        for name, ds_measure, expected_value in checks:
            check_form = fem.form(PETSc.ScalarType(1) * ds_measure)
            check_local = fem.assemble_scalar(check_form)  # Assemble over cell
            total_check = self.msh.comm.allreduce(check_local, op=MPI.SUM)
            # if rank == 0:
            #     # perform assertions
            #     assert np.isclose(total_check, expected_value, rtol=1e-6), f"Mesh check failed for {name}"
