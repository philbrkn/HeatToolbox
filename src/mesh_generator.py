import gmsh
from mpi4py import MPI
from dolfinx import io


class MeshGenerator:
    def __init__(self, config):
        self.config = config

    def create_mesh(self):
        comm = MPI.COMM_SELF
        rank = comm.rank

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("domain_with_extrusions")

        L_X = self.config.L_X
        L_Y = self.config.L_Y
        res = self.config.RESOLUTION
        source_width = self.config.SOURCE_WIDTH
        source_height = self.config.SOURCE_HEIGHT

        # Define points for the base rectangle
        p0 = gmsh.model.geo.addPoint(0, 0, 0, meshSize=res)  # Bottom left
        p1 = gmsh.model.geo.addPoint(L_X, 0, 0, meshSize=res)  # Bottom right
        p2 = gmsh.model.geo.addPoint(L_X, L_Y, 0, meshSize=res)  # Top right
        p3 = gmsh.model.geo.addPoint(0, L_Y, 0, meshSize=res)  # Top left

        # Define lines for the base rectangle
        l0 = gmsh.model.geo.addLine(p0, p1)  # Bottom
        l1 = gmsh.model.geo.addLine(p1, p2)  # Right
        l2 = gmsh.model.geo.addLine(p2, p3)  # Top
        l3 = gmsh.model.geo.addLine(p3, p0)  # Left

        # Initialize list to hold all extrusion curve loops
        extrusion_loops = []

        # Physical groups for boundaries
        slip_boundaries = [l1, l3]
        top_boundaries = []
        connect_rect_extrusion_point = p3
        rect_extr_lines = []
        # Iterate over each source position to create extrusions
        for idx, pos in enumerate(self.config.source_positions):
            x_pos = pos * L_X
            x_min = x_pos - source_width / 2
            x_max = x_pos + source_width / 2
            y_min = L_Y
            y_max = L_Y + source_height

            # Define points for the extrusion (source region)
            p4 = gmsh.model.geo.addPoint(x_min, y_min, 0, meshSize=res)  # Bottom left of extrusion
            p5 = gmsh.model.geo.addPoint(x_max, y_min, 0, meshSize=res)  # Bottom right of extrusion
            p6 = gmsh.model.geo.addPoint(x_max, y_max, 0, meshSize=res)  # Top right of extrusion
            p7 = gmsh.model.geo.addPoint(x_min, y_max, 0, meshSize=res)  # Top left of extrusion

            # Define lines for the extrusion
            # l4 = gmsh.model.geo.addLine(p4, p5)  # Bottom of extrusion
            l5 = gmsh.model.geo.addLine(p5, p6)  # Right of extrusion
            l6 = gmsh.model.geo.addLine(p6, p7)  # Top of extrusion
            l7 = gmsh.model.geo.addLine(p7, p4)  # Left of extrusion

            # To connect extrusion to the base rectangle (For curve loop and slip line)
            l_connect = gmsh.model.geo.addLine(connect_rect_extrusion_point, p4)
            connect_rect_extrusion_point = p5
            rect_extr_lines.append(l_connect)

            # Define curve loops for the extrusion (for curve loop)
            extrusion_loop = [l5, l6, l7]
            extrusion_loops.append(extrusion_loop)  # Use this for curve loop later

            # Physical groups for boundaries
            slip_boundaries.extend([l5, l7, l_connect])
            top_boundaries.append(l6)  # Top of each extrusion as a separate boundary

        l_connect = gmsh.model.geo.addLine(connect_rect_extrusion_point, p2)
        slip_boundaries.append(l_connect)
        rect_extr_lines.append(l_connect)
        all_loops = [l0, l1]
        # reverse order of extrusion loop and rect_extr_lines:
        rect_extr_lines = rect_extr_lines[::-1]
        extrusion_loops = extrusion_loops[::-1]
        for i in range(len(extrusion_loops)):
            all_loops.append(-rect_extr_lines[i])
            all_loops.extend(extrusion_loops[i])
        # add last rect_extr_line
        all_loops.extend([-rect_extr_lines[-1], l3])

        # create gmsh loop and surface
        loop_combined = gmsh.model.geo.addCurveLoop(all_loops)
        surface = gmsh.model.geo.addPlaneSurface([loop_combined])

        gmsh.model.geo.synchronize()

        # Define physical groups
        # Isothermal Boundary
        gmsh.model.addPhysicalGroup(1, [l0], tag=1)
        gmsh.model.setPhysicalName(1, 1, "IsothermalBoundary")

        # Slip Boundaries
        gmsh.model.addPhysicalGroup(1, slip_boundaries, tag=2)
        gmsh.model.setPhysicalName(1, 2, "SlipBoundary")

        # Top Boundaries
        for idx, l_top in enumerate(top_boundaries):
            tag = 3 + idx  # Starting from tag 3
            gmsh.model.addPhysicalGroup(1, [l_top], tag=tag)
            gmsh.model.setPhysicalName(1, tag, f"TopBoundary_{idx}")

        # Domain
        gmsh.model.addPhysicalGroup(2, [surface], tag=1)
        gmsh.model.setPhysicalName(2, 1, "Domain")

        # Generate the mesh
        gmsh.model.mesh.generate(2)

        # save the mesh because of mpi
        gmsh.write("domain_with_extrusions.msh")

        # Convert to Dolfinx mesh
        # msh, cell_markers, facet_markers = io.gmshio.model_to_mesh(
        #     gmsh.model,
        #     comm=comm,
        #     rank=rank,
        #     gdim=2
        # )

        gmsh.finalize()

        # return msh, cell_markers, facet_markers

    def sym_create_mesh(self):
        """
        Symmetric version: 
        - The right boundary (x = L_X) is flagged as 'Symmetry'.
            This boundary is set as L_X / 2 in sim_config.py
        - We create a single source 'extrusion' region at the top-right corner.
        """
        comm = MPI.COMM_SELF
        rank = comm.rank

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("domain_with_extrusions")

        L_X = self.config.L_X
        L_Y = self.config.L_Y
        res = self.config.RESOLUTION
        source_width = self.config.SOURCE_WIDTH
        source_height = self.config.SOURCE_HEIGHT

        # -------------------------------------------------
        # 1) Define base rectangle (0 <= x <= L_X, 0 <= y <= L_Y)
        # -------------------------------------------------
        #
        #    p3-------- l2? --------p2
        #    |                       |
        #   l3                      l1  (Symmetry)
        #    |                       |
        #    p0--------- l0 --------p1
        #        
        p0 = gmsh.model.geo.addPoint(0, 0, 0, meshSize=res)  # Bottom left
        p1 = gmsh.model.geo.addPoint(L_X, 0, 0, meshSize=res)  # Bottom right
        p2 = gmsh.model.geo.addPoint(L_X, L_Y, 0, meshSize=res)  # Top right
        p3 = gmsh.model.geo.addPoint(0, L_Y, 0, meshSize=res)  # Top left

        # Define lines for the base rectangle
        l0 = gmsh.model.geo.addLine(p0, p1)  # Bottom
        l1 = gmsh.model.geo.addLine(p1, p2)  # Right
        l2 = gmsh.model.geo.addLine(p2, p3)  # Top
        l3 = gmsh.model.geo.addLine(p3, p0)  # Left

        # -------------------------------------------------
        # 2) Extrusions: from y=L_Y up to y=L_Y+source_height
        #    for each source in 'source_positions'
        # -------------------------------------------------
        #
        # We'll connect them from left to right, building lines that
        # run along the top of the base rectangle (p3 => first block => ... => p2).
        #
        # - "connect_rect_extrusion_point" is the last x on the top row that we reached
        #   (starting from p3).
        # - For each source:
        #       x_pos  = fraction * L_X
        #       x_min  = x_pos - source_width/2
        #       x_max  = x_pos + source_width/2
        #   Then create p4, p5, p6, p7 for that block
        #   Connect previous chunk to p4 with a new line
        # -------------------------------------------------

        # We store the vertical/diagonal lines that connect from one block to the next
        rect_extr_lines = []
        # Initialize list to hold all extrusion curve loops
        extrusion_loops = []

        # Physical groups for boundaries
        isothermal_boundaries = [l0]  # bottom
        symmetry_boundaries = [l1]  # right
        slip_boundaries = [l3]  # left
        top_boundaries = []  # top

        # We'll connect from p3 (top-left) across to p2 (top-right).
        connect_rect_extrusion_point = p3

        # Sort sources by ascending x-position (if needed)
        sorted_positions = sorted(self.config.source_positions)
        # Iterate over each source position to create extrusions
        for idx, pos in enumerate(sorted_positions):
            x_pos = pos * L_X * 2
            x_min = x_pos - source_width / 2
            x_max = x_pos + source_width / 2

            # If you want to clamp to [0, L_X] so blocks cannot go out of domain:
            if x_min < 0.0:
                x_min = 0.0
            if x_max > L_X:
                x_max = L_X

            y_min = L_Y
            y_max = L_Y + source_height

            # Define points for the extrusion (source region)
            p4 = gmsh.model.geo.addPoint(x_min, y_min, 0, meshSize=res)  # Bottom left of extrusion
            p5 = gmsh.model.geo.addPoint(x_max, y_min, 0, meshSize=res)  # Bottom right of extrusion
            p6 = gmsh.model.geo.addPoint(x_max, y_max, 0, meshSize=res)  # Top right of extrusion
            p7 = gmsh.model.geo.addPoint(x_min, y_max, 0, meshSize=res)  # Top left of extrusion

            # Define lines for the extrusion
            # l4 = gmsh.model.geo.addLine(p4, p5)  # Bottom of extrusion
            l5 = gmsh.model.geo.addLine(p5, p6)  # Right of extrusion
            l6 = gmsh.model.geo.addLine(p6, p7)  # Top of extrusion
            l7 = gmsh.model.geo.addLine(p7, p4)  # Left of extrusion

            # To connect extrusion to the base rectangle (For curve loop and slip line)
            l_connect = gmsh.model.geo.addLine(connect_rect_extrusion_point, p4)
            # Update the "current" top reference to the bottom-right corner of this block
            connect_rect_extrusion_point = p5
            rect_extr_lines.append(l_connect)

            # Define curve loops for the extrusion (for curve loop)
            extrusion_loop = [l5, l6, l7]
            extrusion_loops.append(extrusion_loop)  # Use this for curve loop later

            # Physical groups for boundaries
            # Right edge:
            if abs(x_max - L_X) < 1e-14:
                symmetry_boundaries.append(l5)
            else:
                slip_boundaries.append(l5)

            # The connecting line from previous top corner to p4:
            # Check if it's at x=0 or x=L_X for the entire line (unlikely),
            # else we default it to slip.
            slip_boundaries.append(l_connect)

            slip_boundaries.extend([l5, l7, l_connect])
            top_boundaries.append(l6)  # Top of each extrusion as a separate boundary

        l_connect = gmsh.model.geo.addLine(connect_rect_extrusion_point, p2)
        slip_boundaries.append(l_connect)
        rect_extr_lines.append(l_connect)
        
        # -------------------------------------------------
        # 3) Build the final curve loop
        # -------------------------------------------------
        # Following the same orientation logic as your original code:
        #
        #   all_loops starts with [l0, l1] => bottom & right
        #   then we reverse rect_extr_lines and extrusion_loops
        #   for each block we do -l_connect, +[l5, l6, l7]
        #   at the end we add -rect_extr_lines[-1], l3
        #
        all_loops = [l0, l1]   # bottom + right
        # reverse order of extrusion loop and rect_extr_lines:
        rect_extr_lines = rect_extr_lines[::-1]
        extrusion_loops = extrusion_loops[::-1]
        for i in range(len(extrusion_loops)):
            all_loops.append(-rect_extr_lines[i])
            all_loops.extend(extrusion_loops[i])
        # add last rect_extr_line
        all_loops.extend([-rect_extr_lines[-1], l3])

        # create gmsh loop and surface
        loop_combined = gmsh.model.geo.addCurveLoop(all_loops)
        surface = gmsh.model.geo.addPlaneSurface([loop_combined])

        gmsh.model.geo.synchronize()

        # Define physical groups
        # Isothermal Boundary
        gmsh.model.addPhysicalGroup(1, isothermal_boundaries, tag=1)
        gmsh.model.setPhysicalName(1, 1, "IsothermalBoundary")

        # Slip Boundaries
        gmsh.model.addPhysicalGroup(1, slip_boundaries, tag=2)
        gmsh.model.setPhysicalName(1, 2, "SlipBoundary")

        # Top Boundaries
        for idx, l_top in enumerate(top_boundaries):
            tag = 3 + idx  # Starting from tag 3
            gmsh.model.addPhysicalGroup(1, [l_top], tag=tag)
            gmsh.model.setPhysicalName(1, tag, f"TopBoundary_{idx}")

        gmsh.model.addPhysicalGroup(1, [l1], tag=99)
        gmsh.model.setPhysicalName(1, 99, "Symmetry")

        # Domain
        gmsh.model.addPhysicalGroup(2, [surface], tag=1)
        gmsh.model.setPhysicalName(2, 1, "Domain")
        # -------------------------
        #  5) Generate and convert
        # -------------------------
        gmsh.model.mesh.generate(2)

        # save the mesh because of mpi
        gmsh.write("domain_with_extrusions.msh")

        # Convert to Dolfinx mesh
        # msh, cell_markers, facet_markers = io.gmshio.model_to_mesh(
        #     gmsh.model,
        #     comm=comm,
        #     rank=rank,
        #     gdim=2
        # )

        gmsh.finalize()

        # return msh, cell_markers, facet_markers

    def old_sym_create_mesh(self):
        comm = MPI.COMM_SELF
        if comm.rank == 0:
            gmsh.initialize()
            gmsh.model.add("domain_with_extrusion")

            L_X = self.config.L_X
            L_Y = self.config.L_Y
            res = self.config.RESOLUTION

            y_max = L_Y + self.config.SOURCE_HEIGHT
            # Define points for the base rectangle
            p0 = gmsh.model.geo.addPoint(0, 0, 0, meshSize=res)  # bottom left
            p1 = gmsh.model.geo.addPoint(L_X, 0, 0, meshSize=res)  # bottom right
            p2 = gmsh.model.geo.addPoint(L_X, y_max, 0, meshSize=res)  # top right
            p3 = gmsh.model.geo.addPoint(0, L_Y, 0, meshSize=res)  # tl

            # Define lines for the base rectangle
            l0 = gmsh.model.geo.addLine(p0, p1)  # bottom of base rectangle
            l1 = gmsh.model.geo.addLine(p1, p2)  # right of base rectangle
            l3 = gmsh.model.geo.addLine(p3, p0)  # left of base rectangle

            # Define points for the extrusion (source region)
            x_min = L_X - self.config.SOURCE_WIDTH
            # bottom left of extrusion:
            p4 = gmsh.model.geo.addPoint(x_min, L_Y, 0, meshSize=res)
            # top left of extrusion:
            p7 = gmsh.model.geo.addPoint(x_min, y_max, 0, meshSize=res)

            # Define lines for the extrusion
            l6 = gmsh.model.geo.addLine(p2, p7)  # top of extrusion
            l7 = gmsh.model.geo.addLine(p7, p4)  # left of extrusion

            # Connect the extrusion to the base rectangle
            l8 = gmsh.model.geo.addLine(p3, p4)  # top of base rectangle

            # Define curve loops
            loop_combined = gmsh.model.geo.addCurveLoop([l0, l1, l6, l7, -l8, l3])
            surface = gmsh.model.geo.addPlaneSurface([loop_combined])

            gmsh.model.geo.synchronize()
            # Physical groups for boundaries
            gmsh.model.addPhysicalGroup(1, [l0], tag=1)
            gmsh.model.setPhysicalName(1, 1, "IsothermalBoundary")
            gmsh.model.addPhysicalGroup(1, [l6], tag=3)
            gmsh.model.setPhysicalName(1, 3, "TopBoundary")
            gmsh.model.addPhysicalGroup(1, [l7, l8, l3], tag=2)
            gmsh.model.setPhysicalName(1, 2, "SlipBoundary")
            # gmsh.model.addPhysicalGroup(1, [l7, l6], tag=3)
            # gmsh.model.setPhysicalName(1, 3, "TopBoundary")
            # gmsh.model.addPhysicalGroup(1, [l8, l3], tag=2)
            # gmsh.model.setPhysicalName(1, 2, "SlipBoundary")
            gmsh.model.addPhysicalGroup(1, [l1], tag=4)
            gmsh.model.setPhysicalName(1, 4, "Symmetry")
            gmsh.model.addPhysicalGroup(2, [surface], tag=1)
            gmsh.model.setPhysicalName(2, 1, "Domain")

            gmsh.model.mesh.generate(2)

        msh, cell_markers, facet_markers = io.gmshio.model_to_mesh(gmsh.model, comm, rank=0, gdim=2)

        if comm.rank == 0:
            gmsh.finalize()

        return msh, cell_markers, facet_markers
