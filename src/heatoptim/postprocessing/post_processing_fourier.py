# post_processing.py

import numpy as np
from dolfinx import fem, la, plot, geometry
import matplotlib.pyplot as plt
import ufl
# from mpi4py import MPI
import os
try:
    import pyvista as pv
except ImportError:
    pv = None


class PostProcessingFourier:
    def __init__(self, rank, config, logger=None):
        self.rank = rank
        self.config = config
        self.logger = logger
        self.is_off_screen = self.config.plot_mode == "screenshot"

        # make sure self.logger.log_dir / visualization exists
        if self.rank == 0 and self.logger:
            os.makedirs(os.path.join(self.logger.log_dir, "visualization"), exist_ok=True)

    def postprocess_results(self, T, V, msh, gamma, fileadd=""):
        global_top, global_geom, global_ct, global_vals = self.gather_mesh_on_rank0(msh, V, T)
        _, _, _, global_gamma = self.gather_mesh_on_rank0(msh, V, gamma)
        
        # Print the knudsen numbers
        if self.rank == 0:
            print(f"For a length of {self.config.LENGTH} m:")
            # print(f"Knudsen number (silicon): {self.config.KNUDSEN_SI}")
            # print(f"Knudsen number (diamond): {self.config.KNUDSEN_DI}")

        # if q is not None:
        #     # Calculate eff_cond once at the top-level here
        #     eff_cond = self.calculate_eff_thermal_cond(q, T, msh)
        #     eff_cond_CG, V_CG = self.project_to_CG_space(eff_cond, msh)
        #     _, _, _, global_eff_cond = self.gather_mesh_on_rank0(msh, V_CG, eff_cond_CG)
        # else:
        #     global_eff_cond = None

        viz = self.config.visualize
        if self.rank == 0:
            # print(f"(D) Norm of flux coefficient vector (monolithic, direct): {norm_q}")
            # print(f"(D) Norm of temp coefficient vector (monolithic, direct): {norm_T}")

            if viz["gamma"] and global_gamma is not None:
                self.plot_scalar_field(
                    global_top,
                    global_ct,
                    global_geom,
                    global_gamma,
                    field_name=fileadd+"gamma",
                    show_edges=False,
                )

            if viz["temperature"] and global_vals is not None:
                self.plot_scalar_field(
                    global_top,
                    global_ct,
                    global_geom,
                    global_vals,
                    field_name=fileadd+"T",
                    clim=[0, 0.5],
                )

            # ADD THIS BLOCK TO VISUALIZE EFFECTIVE CONDUCTIVITY
            # if viz["effective_conductivity"] and global_eff_cond is not None:
            #     self.plot_scalar_field(
            #         global_top, global_ct, global_geom, global_eff_cond,
            #         field_name="Effective Conductivity", clim=[0,2500], show_edges=False,
            #     )

        # Code for flux plotting (only runs when not in parallel)
        if msh.comm.size == 1:
            # if viz["flux"]:
            #     self.plot_vector_field(q, msh)
            if viz["profiles"]:
                self.plot_profiles(T, msh)
        else:
            if self.rank == 0:
                print("Flux and profiles plotting is disabled when running in parallel.")
                print("Please run the code in serial to enable flux and profiles plotting.")

    def plot_profiles(self, T, msh):
        # GET TEMPERATURE and FLUX PROFILES #
        x_char = self.config.L_X if self.config.symmetry else self.config.L_X / 2
        # horizontal line:
        x_end = x_char
        y_val = self.config.L_Y - 4 * self.config.LENGTH / 8
        (x_vals, T_x) = self.get_temperature_line(T, msh, "horizontal", start=0, end=x_end, value=y_val)

        # vertical line:
        y_end = self.config.L_Y + self.config.SOURCE_HEIGHT
        x_val = x_char - self.config.LENGTH / 8
        (y_vals, T_y) = self.get_temperature_line(T, msh, "vertical", start=0, end=y_end, value=x_val)
        # normalize x and y vals by config.ell_si
        x_vals = (x_vals[-1] - x_vals) / self.config.ELL_SI
        y_vals = (y_vals[-1] - y_vals) / self.config.ELL_SI

        # PLOT TEMP PROFILES #
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle("Temperature profiles")
        ax.plot(x_vals, T_x, color='red', label="T(x) - Horizontal Line")
        ax.plot(y_vals, T_y, color='blue', label="T(y) - Vertical Line")
        ax.set_xlabel("Position (normalized)")
        ax.set_ylabel("Temperature (T)")
        ax.legend()
        print(self.is_off_screen, "OFF SCREN")
        if self.is_off_screen:
            if self.logger:
                self.logger.save_image(fig, "temperature_profiles.png")
            else:
                plt.savefig("temperature_profiles.png")
            plt.close(fig)
        else:
            plt.show()

    def gather_mesh_on_rank0(self, mesh, V, function, root=0):
        """
        Gathers mesh data (topology, cell types, geometry) and solution data (u) from all ranks to rank 0.

        Parameters:
        mesh: dolfinx.Mesh
            The distributed mesh.
        V: dolfinx.FunctionSpace
            The function space associated with the function.
        function: dolfinx.Function
            The function whose data needs to be gathered (e.g., temperature field).
        root: int
            The rank on which to gather data (default is 0).

        Returns:
        On rank 0:
            - root_top: np.ndarray (global topology)
            - root_geom: np.ndarray (global geometry)
            - root_ct: np.ndarray (global cell types)
            - root_vals: np.ndarray (global function values)
        On other ranks:
            - None, None, None, None
        """
        comm = mesh.comm
        rank = comm.rank

        # Create local VTK mesh data structures
        topology, cell_types, geometry = plot.vtk_mesh(mesh, mesh.topology.dim)

        # Get the number of cells and DOFs (degrees of freedom) for local partition
        num_cells_local = mesh.topology.index_map(mesh.topology.dim).size_local
        num_dofs_local = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

        # Number of DOFs per cell (assuming uniform cells)
        num_dofs_per_cell = topology[0]

        # Get the DOF indices from the topology array
        topology_dofs = (np.arange(len(topology)) % (num_dofs_per_cell + 1)) != 0

        # Map to global DOF indices
        global_dofs = V.dofmap.index_map.local_to_global(topology[topology_dofs].copy())

        # Replace local DOF indices with global DOF indices
        topology[topology_dofs] = global_dofs

        # Gather mesh and function data on the root process
        global_topology = comm.gather(
            topology[: (num_dofs_per_cell + 1) * num_cells_local], root=root
        )
        global_geometry = comm.gather(
            geometry[: V.dofmap.index_map.size_local, :], root=root
        )
        global_ct = comm.gather(cell_types[:num_cells_local], root=root)
        global_vals = comm.gather(function.x.array[:num_dofs_local], root=root)

        if rank == root:
            # Stack the data from all ranks on the root process
            root_geom = np.vstack(global_geometry)
            root_top = np.concatenate(global_topology)
            root_ct = np.concatenate(global_ct)
            root_vals = np.concatenate(global_vals)

            return root_top, root_geom, root_ct, root_vals

        return None, None, None, None

    def plot_scalar_field(
        self,
        global_top,
        global_ct,
        global_geom,
        values,
        field_name="field",
        clim=None,
        show_edges=False,
    ):
        grid = pv.UnstructuredGrid(global_top, global_ct, global_geom)
        grid.point_data[field_name] = values.real
        grid.set_active_scalars(field_name)

        # Plot the scalar field
        plotter = pv.Plotter(off_screen=self.is_off_screen)
        plotter.add_mesh(grid, cmap="coolwarm", show_edges=show_edges, clim=clim)
        plotter.view_xy()
        if self.is_off_screen:
            if self.logger:
                filepath = os.path.join(self.logger.log_dir, "visualization", f"{field_name}_field.png")
                plotter.screenshot(
                    filepath,
                    transparent_background=False,
                    window_size=[1000, 1000],
                )
            else:
                plotter.screenshot(
                    f"{field_name}_field.png",
                    transparent_background=False,
                    window_size=[1000, 1000],
                )
        else:
            print("Plotting the scalar field interactively.")
            # Display interactively if interactive mode
            plotter.show()

    def plot_vector_field(self, q, msh):
        gdim = msh.geometry.dim
        V_dg = fem.functionspace(msh, ("DG", 2, (gdim,)))
        q_dg = fem.Function(V_dg)
        q_copy = q.copy()
        q_dg.interpolate(q_copy)

        V_cells, V_types, V_x = plot.vtk_mesh(V_dg)
        V_grid = pv.UnstructuredGrid(V_cells, V_types, V_x)
        Esh_values = np.zeros((V_x.shape[0], 3), dtype=np.float64)
        Esh_values[:, : msh.topology.dim] = q_dg.x.array.reshape(
            V_x.shape[0], msh.topology.dim
        ).real
        V_grid.point_data["u"] = Esh_values

        plotter = pv.Plotter(off_screen=self.is_off_screen)
        plotter.add_text("magnitude", font_size=12, color="black")
        # plotter.add_mesh(V_grid.copy(), show_edges=False)
        plotter.add_mesh(V_grid.copy(), show_edges=False, clim=[0,3000])
        
        plotter.view_xy()
        plotter.link_views()
        if self.is_off_screen:
            if self.logger:
                filepath = os.path.join(self.logger.log_dir, "visualization", "flux_field.png")
                plotter.screenshot(
                    filepath,
                    transparent_background=False,
                    window_size=[1000, 1000],
                )
            else:
                plotter.screenshot(
                    "flux_field.png",
                    transparent_background=False,
                    window_size=[1000, 1000],
                )
        else:
            print("Plotting the vector field interactively.")
            # Display interactively if interactive mode
            plotter.show()

    def get_temperature_line(
        self,
        T_results,
        msh,
        line_orientation="horizontal",
        start=0.0,
        end=1.0,
        value=0.5,
        num_points=500,
    ):
        # tol = 1e-9  # Small tolerance to avoid hitting boundaries exactly
        tol = self.config.LENGTH * 1e-3  # Small tolerance to avoid hitting boundaries exactly
        points = np.zeros((3, num_points))

        if line_orientation == "horizontal":
            x_coords = np.linspace(start + tol, end - tol, num_points)
            points[0] = x_coords
            points[1] = value
        elif line_orientation == "vertical":
            y_coords = np.linspace(start + tol, end - tol, num_points)
            points[0] = value
            points[1] = y_coords
        else:
            raise ValueError(
                "line_orientation must be either 'horizontal' or 'vertical'"
            )

        # Create bounding box tree for the mesh
        bb_tree = geometry.bb_tree(msh, msh.topology.dim)

        # Find cells whose bounding-box collide with the the points
        cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)

        # Choose one of the cells that contains the point
        colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points.T)

        cells = []
        points_on_proc = []
        # Get the temperature values at the points
        for i, point in enumerate(points.T):
            if len(colliding_cells.links(i)) > 0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])

        points_on_proc = np.array(points_on_proc, dtype=np.float64)
        T_values = T_results.eval(points_on_proc, cells)

        if line_orientation == "horizontal":
            # Return the coordinates and their corresponding temperature values
            return (points[0], T_values)
        elif line_orientation == "vertical":
            # Return the coordinates and their corresponding temperature values
            return (points[1], T_values)

    def calculate_curl(self, q, msh, plot_curl=False):

        def curl_2d(q: fem.Function):
            # Returns the z-component of the 2D curl as a scalar
            return q[1].dx(0) - q[0].dx(1)

        V_curl = fem.functionspace(msh, ("DG", 1))  # DG space for scalar curl
        curl_function = fem.Function(V_curl)

        curl_flux_calculator = fem.Expression(curl_2d(q), V_curl.element.interpolation_points())
        curl_function.interpolate(curl_flux_calculator)

        if plot_curl:
            V_cells, V_types, V_x = plot.vtk_mesh(V_curl)
            curl_values = curl_function.x.array
            curl_grid = pv.UnstructuredGrid(V_cells, V_types, V_x)
            curl_grid.point_data["Curl"] = curl_values
            curl_grid.set_active_scalars("Curl")
            # Plot the curl field
            plotter = pv.Plotter()
            plotter.add_mesh(curl_grid, cmap="coolwarm", show_edges=False)
            plotter.view_xy()
            plotter.show()

        return curl_function

    def calculate_eff_thermal_cond(self, q, T, msh):
        def heat_flux_magnitude(q):
            q_x, q_y = q[0], q[1]
            return ufl.sqrt(q_x**2 + q_y**2)

        def temperature_gradient_magnitude(T):
            grad_T = ufl.grad(T)
            return ufl.sqrt(grad_T[0]**2 + grad_T[1]**2)

        def k_cond(q, T):
            q_magnitude = heat_flux_magnitude(q)
            grad_T_magnitude = temperature_gradient_magnitude(T)
            return q_magnitude / (grad_T_magnitude)

        V_cond = fem.functionspace(msh, ("DG", 1))  # DG space for scalar conductivity
        cond_function = fem.Function(V_cond)
        cond_expr = fem.Expression(k_cond(q, T), V_cond.element.interpolation_points())
        cond_function.interpolate(cond_expr)

        return cond_function

    def project_to_CG_space(self, function_DG, msh, degree=1):
        V_CG = fem.functionspace(msh, ("CG", 1))
        function_CG = fem.Function(V_CG)
        function_CG.interpolate(function_DG)
        return function_CG, V_CG
