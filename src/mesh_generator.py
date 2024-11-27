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

        # Optionally, save the mesh for debugging
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
