# post_processing.py

import numpy as np
import pyvista as pv
from dolfinx import fem, la, plot, geometry
import matplotlib.pyplot as plt


class PostProcessingModule:
    def __init__(self, rank, config):
        self.rank = rank
        self.config = config

    def postprocess_results(self, U, msh, img, gamma):
        q, T = U.sub(0).collapse(), U.sub(1).collapse()
        norm_q, norm_T = la.norm(q.x), la.norm(T.x)

        V1, _ = U.function_space.sub(1).collapse()
        global_top, global_geom, global_ct, global_vals = self.gather_mesh_on_rank0(
            msh, V1, T
        )
        _, _, _, global_gamma = self.gather_mesh_on_rank0(msh, V1, gamma)

        if self.rank == 0:
            print(f"(D) Norm of flux coefficient vector (monolithic, direct): {norm_q}")
            print(f"(D) Norm of temp coefficient vector (monolithic, direct): {norm_T}")

            if global_vals is not None:
                self.plot_scalar_field(
                    global_top,
                    global_ct,
                    global_geom,
                    global_vals,
                    field_name="T",
                    # clim=(0, 0.5),
                )

            if global_gamma is not None:
                self.plot_scalar_field(
                    global_top,
                    global_ct,
                    global_geom,
                    global_gamma,
                    field_name="gamma",
                    show_edges=True,
                )

            self.plot_vector_field(q, msh)

            x_char = self.config.L_X if self.config.symmetry else self.config.L_X / 2
            # horizontal line:
            x_end = x_char
            y_val = self.config.L_Y - self.config.LENGTH / 8
            (x_vals, T_x) = self.get_temperature_line(T, msh, "horizontal", start=0, end=x_end, value=y_val)
            # vertical line:
            y_end = self.config.L_Y + self.config.SOURCE_HEIGHT
            x_val = x_char - self.config.LENGTH * 3 / 8
            (y_vals, T_y) = self.get_temperature_line(T, msh, "vertical", start=0, end=y_end, value=x_val)

            # normalize x and y vals by config.ell_si
            x_vals = (x_end - x_vals) / self.config.ELL_SI
            y_vals = (y_end - y_vals) / self.config.ELL_SI

            # figure with a single plot
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.suptitle("Temperature profiles")

            # horizontal line in red
            ax.plot(x_vals, T_x, color='red', label="T(x) - Horizontal Line")

            # vertical line in blue
            ax.plot(y_vals, T_y, color='blue', label="T(y) - Vertical Line")

            # Set labels and legend
            ax.set_xlabel("Position (normalized)")
            ax.set_ylabel("Temperature (T)")
            ax.legend()

            # Display the plot
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
        plotter = pv.Plotter()
        plotter.add_mesh(grid, cmap="coolwarm", show_edges=show_edges, clim=clim)
        plotter.view_xy()
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

        plotter = pv.Plotter()
        plotter.add_text("magnitude", font_size=12, color="black")
        plotter.add_mesh(V_grid.copy(), show_edges=False)
        plotter.view_xy()
        plotter.link_views()
        plotter.show()

    def get_temperature_line(
        self,
        T_results,
        msh,
        line_orientation="horizontal",
        start=0.0,
        end=1.0,
        value=0.5,
        num_points=200,
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
