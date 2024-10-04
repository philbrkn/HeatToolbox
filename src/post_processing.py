import time
import numpy as np
import pyvista as pv
from dolfinx import fem, la, plot


class PostProcessingModule:
    def __init__(self, rank):
        self.rank = rank

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
