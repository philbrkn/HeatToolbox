'''module docstring'''
import numpy as np
from ..config import config
# import time

import dolfinx
import dolfinx.fem.petsc
from dolfinx import fem
from dolfinx.cpp import log
from dolfinx.mesh import meshtags, compute_midpoints, locate_entities
from dolfinx.fem import (Function, FunctionSpace)
from dolfinx.io import gmshio, XDMFFile, VTXWriter
from dolfinx import geometry
import ufl
from ufl import (ds, dx, grad, inner, TrialFunction, TestFunction)
try:
    from dolfinx.plot import create_vtk_mesh
except ImportError:
    from dolfinx.plot import vtk_mesh

# from memory_profiler import profile
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from petsc4py import PETSc
try:
    import pyvista as pv
    import matplotlib.pyplot as plt
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    # print("PyVista is not available in this environment.")


class FenicsxSolver():

    def __init__(self, config=config, img=None):
        # self.img = img
        self.config = config
        self.settings = config['fenics']
        bcs = config['bcs']
        # self.boundary_conditions = bc_dict
        for key, value in self.settings.items():
            setattr(self, key, value)
        # self.thermal_ks = {'Fe': 67, 'PDMS': 0.15, 'Cu': 386}
        self.msh, cell_markers, self.facet_markers = gmshio.read_from_msh("mesh_with_two_circles.msh", MPI.COMM_WORLD, gdim=2)
        self.V = FunctionSpace(self.msh, ("CG", 1))
        self.Th = Function(self.V)
        self.Th.name = "Th"

        # locate DOFs associated w boundary
        Tl = bcs['left']['Dirichlet']
        Tr = bcs['right']['Dirichlet']
        # use dirichletbc to create a DirichletBCMetaClass class that represents the bc
        dofs_L = fem.locate_dofs_geometrical(self.V, lambda x: np.isclose(x[0], 0))
        bc_L = fem.dirichletbc(value=ScalarType(Tl), dofs=dofs_L, V=self.V)
        dofs_R = fem.locate_dofs_geometrical(self.V, lambda x: np.isclose(x[0], 1))
        bc_R = fem.dirichletbc(value=ScalarType(Tr), dofs=dofs_R, V=self.V)
        self.bcs = [bc_R, bc_L]

        f = fem.Constant(self.msh, ScalarType(0))
        g = fem.Constant(self.msh, ScalarType(0))
        self.v = TestFunction(self.V)
        self.T = TrialFunction(self.V)
        L = f * self.v * dx + g * self.v * ds  # No source term

        # Prepare linear algebra structures for iterative problem
        # bilinear_form = fem.form(self.a)
        self.linear_form = fem.form(L)

        # L does not change with img, whereas matrix 'a' does
        # Thus only assemble L once, and create matrix A to be updated based on 'a'
        self.b = dolfinx.fem.petsc.create_vector(self.linear_form)

        # Using petsc4py to create a linear solver
        self.solver = PETSc.KSP().create(self.msh.comm)
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)

        self.T_target, self.dX = self.get_T_target()
        dX_local = fem.assemble_scalar(fem.form(1*self.dX(1)))
        self.dX_total = self.msh.comm.allreduce(dX_local, op=MPI.SUM)  # sum all procs

        # Getting normilzation values for scaling
        # self.normalized_values()

    def solve_heat(self, img):
        '''solving the heat equation'''

        self.kappa = self.get_kappa(img)

        a = inner(self.kappa * grad(self.T), grad(self.v)) * dx  # kappa outside or inside inner?
        bilinear_form = fem.form(a)
        A = fem.petsc.assemble_matrix(bilinear_form, bcs=self.bcs)
        A.assemble()

        # assign matrix A to linear algebra solver
        self.solver.setOperators(A)

        # Apply Dirichlet boundary condition to the vector
        # Update the right hand side reusing the initial vector
        with self.b.localForm() as loc_b:
            loc_b.set(0)
        fem.petsc.assemble_vector(self.b, self.linear_form)
        fem.petsc.apply_lifting(self.b, [bilinear_form], [self.bcs])
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(self.b, self.bcs)

        # Solving the heat equation
        self.solver.solve(self.b, self.Th.vector)
        self.Th.x.scatter_forward()

    def get_kappa(self, img):
        # !!! could move Q to init !!!
        Q = FunctionSpace(self.msh, ("DG", 0))
        k_vals = self.thermal_ks

        def thermal_conductivity(x):
            ''' x contains geometrical coordinates in mesh
            '''
            n_image_res = img.shape[0]
            values = np.zeros(x.shape[1], dtype=ScalarType)
            for idx in range(x.shape[1]):
                i = int(min(x[0, idx]*n_image_res, n_image_res - 1))
                j = int(min(x[1, idx]*n_image_res, n_image_res - 1))

                if img[-(j+1), i] == 0:  # img = 0, black, most present
                    values[idx] = k_vals['Cu']
                elif img[-(j+1), i] == 1:  # img = 1, grey
                    values[idx] = k_vals['PDMS']
                elif img[-(j+1), i] == 2:  # img = 2, background
                    values[idx] = k_vals['Fe']
                else:
                    raise ValueError('Image value not 0, 1, or 2')
            return values

        kappa = Function(Q)
        kappa.interpolate(thermal_conductivity)
        # kappa = 1.0  # for debugging
        return kappa

    def get_heat_flux(self, bnd_tag, square=True):
        '''get heat flux loss (q^2). units are (W/m^2)^2
        bnd_tag is defined in create_gmsh
        '''
        arc_indices = self.facet_markers.indices[self.facet_markers.values == bnd_tag]
        arc_tags = meshtags(self.msh, self.msh.topology.dim-1, arc_indices, bnd_tag)
        dS = ufl.Measure("dS", domain=self.msh, subdomain_data=arc_tags)

        # Check facet
        # msh.topology.create_connectivity(msh.topology.dim-1, msh.topology.dim)
        # with XDMFFile(msh.comm, "facet_tags.xdmf", "w") as xdmf:
        #     xdmf.write_mesh(msh)
        #     xdmf.write_meshtags(arc_tags)

        n = ufl.FacetNormal(self.msh)  # normal vector
        Q = fem.VectorFunctionSpace(self.msh, ("DG", 0))
        qw = Function(Q)
        qw_expr = fem.Expression(self.kappa*grad(self.Th), Q.element.interpolation_points())
        qw.interpolate(qw_expr)  # [W/m^2]
        flux = ufl.dot(qw, n('-'))

        if square:
            flux_form = fem.form(np.power(flux, 2) * dS(bnd_tag))  # fem form
        else:
            flux_form = fem.form(flux * dS(bnd_tag))  # fem form

        flux_local = fem.assemble_scalar(flux_form)  # assemble over cell
        total_flux = self.msh.comm.allreduce(flux_local, op=MPI.SUM)  # sum all procs
        if bnd_tag == 4:
            ri = self.radius_inner_z
        if bnd_tag == 3:
            ri = self.radius_inner_m
        total_flux = total_flux / (2 * np.pi * ri)
        return total_flux

    def get_total_flux_domain(self):
        ''' get the total heat flux integrated over the domain
        '''
        n = ufl.FacetNormal(self.msh)  # normal vector
        Q = fem.VectorFunctionSpace(self.msh, ("DG", 0))
        qw = Function(Q)
        qw_expr = fem.Expression(self.kappa*grad(self.Th), Q.element.interpolation_points())
        qw.interpolate(qw_expr)  # [W/m^2]
        flux = ufl.dot(qw, n('-'))

        flux_form = fem.form(flux * dx)
        flux_local = fem.assemble_scalar(flux_form)  # assemble over cell
        total_flux = self.msh.comm.allreduce(flux_local, op=MPI.SUM)  # sum all procs
        return total_flux

    def get_T_target(self):

        def outside_outer_circle(x):
            return np.array((x.T[0] - 0.5)**2 + (x.T[1]-0.5)**2 >= 0.45**2, dtype=np.int32)

        num_cells = self.msh.topology.index_map(self.msh.topology.dim).size_local
        midpoints = compute_midpoints(self.msh, self.msh.topology.dim, list(np.arange(num_cells, dtype=np.int32)))
        outside_tags = meshtags(self.msh, self.msh.topology.dim, np.arange(num_cells), outside_outer_circle(midpoints))
        dX = ufl.Measure("dx", domain=self.msh, subdomain_data=outside_tags)

        T, v = TrialFunction(self.V), TestFunction(self.V)
        a = inner(self.thermal_ks['Fe'] * grad(T), grad(v)) * dx
        f = fem.Constant(self.msh, ScalarType(0))
        g = fem.Constant(self.msh, ScalarType(0))
        L = f * v * dx + g * v * ds  # No source term

        problem = fem.petsc.LinearProblem(a, L, bcs=self.bcs,
                                          petsc_options={"ksp_type": "preonly",  # preonly / gmres
                                                         "pc_type": "lu"})  # lu / ilu
        T_target = problem.solve()
        return T_target, dX

    def get_tsub_loss(self, square=True):
        Tsub = Function(self.V)
        if square is False:
            Tsub.x.array[:] = self.Th.x.array - self.T_target.x.array
        else:
            Tsub.x.array[:] = np.power(self.Th.x.array - self.T_target.x.array, 2)
            
        Tsub_form = fem.form(Tsub * self.dX(1))  # fem form
        Tsub_local = fem.assemble_scalar(Tsub_form)  # assemble over cell
        Tsub_total = self.msh.comm.allreduce(Tsub_local, op=MPI.SUM)  # sum all procs
        # dX1_should_be = 1*1 - np.pi*0.45**2
        # print(Tsub_total/dX_total, Tsub_total)
        return Tsub_total / self.dX_total

    def get_entropy_generation(self):
        ''' get the entropy generation as defined by Bejan
        '''

        # NEED TO MAKE Q P1 or P2 BECAUSE TAKING DIV
        Q = fem.VectorFunctionSpace(self.msh, ("DG", 1))
        qw = Function(Q)
        qw_expr = fem.Expression(-self.kappa*grad(self.Th), Q.element.interpolation_points())
        qw.interpolate(qw_expr)  # [W/m^2]

        delT = Function(Q)
        delT_expr = fem.Expression(grad(self.Th), Q.element.interpolation_points())
        delT.interpolate(delT_expr)  # [K/m]

        # not vector function space because scalar field
        W = fem.FunctionSpace(self.msh, ("DG", 0))  # 0th order because very discontinuous
        ufl_div_qw = ufl.div(qw)
        divq_expr = fem.Expression(ufl_div_qw, W.element.interpolation_points())
        divq = Function(W)
        divq.interpolate(divq_expr)  # [W/m^3]

        # entropy generation
        # put inside function so it can be plotted, otherwise not needed
        T_squared = np.power(self.Th, 2)
        S_gen_expr = fem.Expression((1 / self.Th) * divq - (1 / T_squared) * ufl.dot(qw, delT), W.element.interpolation_points())
        S_gen = Function(W)
        S_gen.interpolate(S_gen_expr)  # [W/m^3 K]
        # print(type(S_gen))

        # integrate S_gen over domain
        S_gen_form = fem.form(S_gen * dx)  # fem form
        S_gen_local = fem.assemble_scalar(S_gen_form)  # assemble over cell
        S_gen_total = self.msh.comm.allreduce(S_gen_local, op=MPI.SUM)  # sum all procs

        self.S_gen = S_gen
        self.S_gen_sum = S_gen_total
        return S_gen_total

    def get_temperature_line(self, y_value=0.5, num_points=100):
        ''' has changed between fenicsx060 and fenicsx070
        '''
        # Create a 1D mesh along the line where we want to evaluate the temperature
        tol = 0.001  # Avoid hitting the outside of the domain
        x = np.linspace(-1 + tol, 1 - tol, num_points)
        points = np.zeros((3, num_points))
        points[0] = x
        points[1] = y_value
        # T_values = []

        # bb_tree = geometry.bb_tree(self.msh, self.msh.topology.dim)
        bb_tree = geometry.BoundingBoxTree(self.msh, self.msh.topology.dim)
        
        # Find cells whose bounding-box collide with the the points
        # cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
        cell_candidates = geometry.compute_collisions(bb_tree, points.T)
        
        # Choose one of the cells that contains the point
        colliding_cells = geometry.compute_colliding_cells(self.msh, cell_candidates, points.T)

        cells = []
        points_on_proc = []
        for i, point in enumerate(points.T):
            if len(colliding_cells.links(i)) > 0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])

        points_on_proc = np.array(points_on_proc, dtype=np.float64)
        T_values = self.Th.eval(points_on_proc, cells)
        
        # Return the coordinates and their corresponding temperature values
        return T_values

    def plot_conductivity(self, subplotter, show_scalar_bar=False):
        num_cells = self.msh.topology.index_map(self.msh.topology.dim).size_local
        topology, cell_types, x = create_vtk_mesh(self.msh, self.msh.topology.dim, np.arange(num_cells, dtype=np.int32))
        # MATERIAL ON MESH DISTRIBUTION
        subplotter.subplot(0, 0)
        # subplotter.add_text("Mesh with cond. markers", font_size=14, color="black", position="upper_edge")
        grid = pv.UnstructuredGrid(topology, cell_types, x)
        grid.cell_data["Conductivity"] = self.kappa.vector
        # grid.cell_data["outside_markers"] = outside_tags.values
        # grid.set_active_scalars("cond")
        grid.set_active_scalars("Conductivity")
        subplotter.add_mesh(grid, show_edges=True, show_scalar_bar=show_scalar_bar)
        subplotter.view_xy()

    def plot_sgen_distribution(self, subplotter, show_scalar_bar=True):
        '''Similar to how flux is done, but for scalar distribution
        instead of vector field
        '''
        msh = self.msh  # normal mesh
        W = fem.FunctionSpace(msh, ("DG", 0))  # from get_entropy_generation()

        top_imap = msh.topology.index_map(msh.topology.dim)
        num_cells = top_imap.size_local + top_imap.num_ghosts
        midpoints = compute_midpoints(msh, msh.topology.dim, range(num_cells))

        num_dofs = W.dofmap.index_map.size_local + W.dofmap.index_map.num_ghosts
        # topology, cell_types, x 
        sgen_grid = pv.UnstructuredGrid(*create_vtk_mesh(msh, msh.topology.dim, np.arange(num_cells, dtype=np.int32)))

        assert num_cells == num_dofs

        values = self.S_gen.x.array.real

        # Create a PolyData object using the midpoints
        cloud = pv.PolyData(midpoints)
        cloud["[W/m3K]"] = values

        SCALAR_BAR_RANGE = (0, 50)

        # Scalar bar arguments
        sargs = dict(height=0.1, vertical=False, position_x=0.22, position_y=0.05, n_labels=2, fmt="%.3g")  #, color='white')

        # Add meshes
        # subplotter.add_mesh(sgen_grid, color="white", show_edges=False)
        subplotter.add_mesh(cloud, cmap="viridis", scalar_bar_args=sargs,
                            scalars="[W/m3K]", show_scalar_bar=show_scalar_bar,
                            clim=SCALAR_BAR_RANGE)

        # Add title with entropy gen sum
        subplotter.add_title(str(self.S_gen_sum), font_size=10, color="black")
        subplotter.view_xy()

    def plot_temperature_distribution(self, subplotter, show_scalar_bar=True):
        grid_uh = pv.UnstructuredGrid(*create_vtk_mesh(self.V))
        grid_uh.point_data["  "] = self.Th.x.array.real
        grid_uh.set_active_scalars("  ")

        # Scalar bar arguments
        # SCALAR_BAR_RANGE = (293, 393)
        sargs = dict(height=0.1, vertical=False, position_x=0.22, position_y=0.05, n_labels=5, fmt="%.1f")
        subplotter.add_mesh(grid_uh, show_edges=False, scalar_bar_args=sargs,
                            show_scalar_bar=show_scalar_bar, cmap="turbo")

        subplotter.view_xy()

    def plot_tsub_distribution(self, subplotter, show_scalar_bar=True):
        # [Your existing code for plotting Tsub distribution]
        Tsub = Function(self.V)
        Tsub.x.array[:] = self.Th.x.array - self.T_target.x.array

        # subplotter.add_text("Tsub Distribution", font_size=14, color="black", position="upper_edge")
        grid_sub = pv.UnstructuredGrid(*create_vtk_mesh(self.V))
        grid_sub.point_data["[K]"] = Tsub.x.array.real
        grid_sub.set_active_scalars("[K]")

        # Scalar bar arguments
        SCALAR_BAR_RANGE = (-2, 2)
        sargs = dict(height=0.1, vertical=False, position_x=0.22, position_y=0.05, n_labels=2)
        subplotter.add_mesh(grid_sub, show_edges=False, scalar_bar_args=sargs,
                            show_scalar_bar=show_scalar_bar, clim=SCALAR_BAR_RANGE,
                            cmap="turbo")
        center_o = self.config['assembly']['obj']['outer'][1]
        circle_o = pv.Circle(self.radius_outer, resolution=100)
        circle_o.translate([0.5, 0.5, 0.0], inplace=True)
        # subplotter.add_mesh(circle_o, color="white", show_edges=False)
        subplotter.view_xy()

    def plot_flux_vector_field(self, subplotter, show_scalar_bar=True):

        # We include ghosts cells as we access all degrees of freedom (including ghosts) on each process
        Q = fem.VectorFunctionSpace(self.msh, ("DG", 0))
        qw = Function(Q)
        qw_expr = fem.Expression(-self.kappa*grad(self.Th), Q.element.interpolation_points())
        qw.interpolate(qw_expr)

        # Interpolate onto coarser mesh for clarity
        # coarse_msh, cell_markers, facet_markers = gmshio.read_from_msh("really_coarse_mesh_with_two_circles.msh", MPI.COMM_WORLD, gdim=2)
        coarse_msh, cell_markers, facet_markers = gmshio.read_from_msh("semi_coarse_mesh_with_two_circles.msh", MPI.COMM_WORLD, gdim=2)
        # coarse_msh = self.msh
        Q = fem.VectorFunctionSpace(coarse_msh, ("DG", 0))
        qw_coarse = Function(Q)
        qw_coarse.interpolate(qw)
        # coarse_msh=self.msh
        # qw_coarse=qw

        top_imap = coarse_msh.topology.index_map(coarse_msh.topology.dim)
        num_cells = top_imap.size_local + top_imap.num_ghosts
        midpoints = compute_midpoints(coarse_msh, coarse_msh.topology.dim, range(num_cells))
        num_dofs = Q.dofmap.index_map.size_local + Q.dofmap.index_map.num_ghosts
        # topology, cell_types, x 
        grid = pv.UnstructuredGrid(*create_vtk_mesh(coarse_msh, coarse_msh.topology.dim, np.arange(num_cells, dtype=np.int32)))

        assert num_cells == num_dofs
        values = np.zeros((num_dofs, 3), dtype=np.float64)
        values[:, :coarse_msh.geometry.dim] = qw_coarse.x.array.real.reshape(num_dofs, Q.dofmap.index_map_bs)

        cloud = pv.PolyData(midpoints)
        cloud["qw"] = values

        # remove specific value for sketch purposes
        # index = np.argwhere(np.linalg.norm(values, axis=1) == np.sort(np.linalg.norm(values, axis=1))[-5])
        # values[index] = np.array([0, 0, 0])
        # index2 = np.argwhere(np.linalg.norm(values, axis=1) == np.sort(np.linalg.norm(values, axis=1))[-2])
        # index3 = np.argwhere(np.linalg.norm(values, axis=1) == np.sort(np.linalg.norm(values, axis=1))[-1])
        # values[index2] = values[index3]
        # values[index2, 1] *= -1

        # print(np.sort(np.linalg.norm(values, axis=1)))
        # print("NYM VALUES", len(values))
        cloud["[W/m2]"] = np.linalg.norm(values, axis=1)
        
        # FOR FLUX SCKETCH
        # show_scalar_bar = True
        # glyphs = cloud.glyph("qw", scale=True, factor=4e-6)
        # sargs = dict(height=0.1, vertical=False, position_x=0.22, position_y=0.05, n_labels=2, fmt="%.3g", color='white')
        # actor2 = subplotter.add_mesh(glyphs, cmap=plt.cm.jet, scalar_bar_args=sargs, scalars="[W/m2]", show_scalar_bar=show_scalar_bar)
        # ^ FOR FLUX SKETCH

        # FOR NORMAL FLUX FIGURES
        glyphs = cloud.glyph("qw", scale=False, factor=5e-2)
        # glyphs = cloud.glyph("qw", scale=True, factor=3.5e-6)

        # THRESHOLD
        # threshold = glyphs.threshold(value=3e4)
        # print("THRESHOLD", glyphs)
        # glyphs = threshold

        sargs = dict(height=0.1, vertical=False, position_x=0.22, position_y=0.05, n_labels=2, fmt="%.3g")
        actor2 = subplotter.add_mesh(glyphs, cmap=plt.cm.jet, scalar_bar_args=sargs,
                                     scalars="[W/m2]", show_scalar_bar=show_scalar_bar,
                                     clim =(0,1e4))
        actor = subplotter.add_mesh(grid, color="white", show_edges=False) #, opacity=0.2)
        subplotter.view_xy()

    def plot_solutions(self, off_screen=False, file_path=None, color_bars=True,
                       separate_plots=False, funcs=None, add_name=""):
        '''Plot solutions using pyvista 
        '''
        if not PYVISTA_AVAILABLE:
            print("PyVista is not installed. Cannot plot solutions.")
            return

        pv.OFF_SCREEN = off_screen
        # make circles:
        ri_z = self.radius_inner_z
        ri_m = self.radius_inner_m
        ro = self.radius_outer
        center_o = self.config['assembly']['obj']['outer'][1]
        center_i1 = self.config['assembly']['obj']['inner'][0][1]
        center_i2 = self.config['assembly']['obj']['inner'][1][1]
        circle_o = pv.CircularArc([center_o[0], center_o[1]+ro, 0],
                                  [center_o[0], center_o[1]+ro, 0],
                                  [center_o[0], center_o[1], 0], negative=True)
        circle_i1 = pv.CircularArc([center_i1[0], center_i1[1]+ri_z, 0],
                                   [center_i1[0], center_i1[1]+ri_z, 0],
                                   [center_i1[0], center_i1[1], 0], negative=True)
        circle_i2 = pv.CircularArc([center_i2[0], center_i2[1]+ri_m, 0],
                                   [center_i2[0], center_i2[1]+ri_m, 0],
                                   [center_i2[0], center_i2[1], 0], negative=True)

        if separate_plots:
            # Plot each subplot in a separate window or save as a separate file
            for plot_func in funcs:
                subplotter = pv.Plotter(window_size=[500, 600])
                plot_func(subplotter, color_bars)
                subplotter.add_mesh(circle_o, color="black", line_width=2, opacity=1)
                subplotter.add_mesh(circle_i1, color="black", line_width=2)
                subplotter.add_mesh(circle_i2, color="black", line_width=2)
                if off_screen:
                    subplotter.screenshot(f"{file_path}/{plot_func.__name__}{add_name}.png")
                else:
                    subplotter.show()
                
        else:
            # Plot all subplots together
            subplotter = pv.Plotter(shape=(1, 4), window_size=[2000, 600])
            self.plot_conductivity(subplotter, color_bars)
            self.plot_temperature_distribution(subplotter, color_bars)
            self.plot_tsub_distribution(subplotter, color_bars)
            self.plot_flux_vector_field(subplotter, color_bars)
            if off_screen:
                subplotter.screenshot(f"{file_path}/fenicsx_plots.png")
            else:
                subplotter.show()

        # SAVE TEMP VTK FILE
        # from dolfinx import io
        # from pathlib import Path
        # results_folder = Path("results")
        # filename_vtk = f"{file_path}/temp_dist_VTK.bp"
        # with VTXWriter(self.msh.comm, filename_vtk, [self.Th]) as vtx:
        #     vtx.write(0.0)
        filename_xdmf = f"{file_path}/temp_dist_VTK_{add_name}.xdmf"
        with XDMFFile(self.msh.comm, filename_xdmf, "w") as xdmf:
            xdmf.write_mesh(self.msh)
            xdmf.write_function(self.Th)

    def set_normalized_values(self, obj_config=None, PDMS_grid=None, Fe_grid=None):
        '''call this in beginning to acquire scaled values for flux and tsub
        import distribution of material either all PDMS or all Cu.
        import config for boundary values of circles.
        '''
        # PDMS_grid = np.loadtxt('PDMS_grid.txt')
        # Cu_grid = np.loadtxt("Cu_grid.txt")

        # OUTER CIRCLE CLOAKING #
        # material is all PDMS
        self.solve_heat(PDMS_grid)
        self.cloak_norm = self.get_tsub_loss()

        # INNER CIRCLE ZERO FLUX #
        # material is all PDMS
        # right circle: bnd tag 4
        self.solve_heat(Fe_grid)
        bnd_tag = obj_config['inner'][0][2]
        self.zero_flux_norm = self.get_heat_flux(bnd_tag)

        # INNER CIRCLE MAX FLUX #
        # material is all Cu

        # left circle: bnd tag 3
        bnd_tag = obj_config['inner'][1][2]
        self.max_flux_norm = self.get_heat_flux(bnd_tag)

        print("Normalization values: ", self.cloak_norm, self.zero_flux_norm, self.max_flux_norm)


# @profile
def debug_solver_loop():
    ''' run pseudo optimaztion by iterating through solver with
    random images
    '''
    num_iter = 1000
    fs = FenicsxSolver(config)
    for i in range(num_iter):
        img = np.random.randint(3, size=(100, 100))
        fs.solve_heat(img)
        flux = fs.get_heat_flux()
        tsub = fs.get_tsub_loss()
        print(f"Total flux: {flux} W")
        print(f"Tsub loss: {tsub} K")


def main():
    # img = np.loadtxt('data/sample_trimaterial_grid.txt')
    # img = np.loadtxt('logs/20231009_2038_8cells/material_grid_minFlux.txt')
    img = np.loadtxt('logs/20231012_1337_8cells/material_grid_minZeroFlux.txt')
    fs = FenicsxSolver(config)
    fs.solve_heat(img)
    tsub = fs.get_tsub_loss()
    print(f"Tsub loss: {tsub} K")
    for i in range(3, 5):
        flux = fs.get_heat_flux(i)
        print(f"Total flux: {flux} W")

    fs.plot_solutions()


if __name__ == "__main__":
    main()