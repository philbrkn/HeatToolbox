import gmsh
from mpi4py import MPI
from dolfinx import io


class MeshGenerator:
    def __init__(self, config):
        self.config = config

    def create_mesh(self):
        if MPI.COMM_WORLD.rank == 0:
            gmsh.initialize()
            gmsh.model.add("domain_with_extrusion")

            L_X = self.config.L_X
            L_Y = self.config.L_Y
            res = self.config.RESOLUTION

            if self.config.symmetry:
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
            elif self.config.two_sources:
                # Define points for the base rectangle
                p0 = gmsh.model.geo.addPoint(0, 0, 0, meshSize=res)  # bottom left
                p1 = gmsh.model.geo.addPoint(L_X, 0, 0, meshSize=res)  # bottom right
                p2 = gmsh.model.geo.addPoint(L_X, L_Y, 0, meshSize=res)  # tr
                p3 = gmsh.model.geo.addPoint(0, L_Y, 0, meshSize=res)  # tl

                # Define lines for the base rectangle
                l0 = gmsh.model.geo.addLine(p0, p1)  # bottom of base rectangle
                l1 = gmsh.model.geo.addLine(p1, p2)  # right of base rectangle
                l2 = gmsh.model.geo.addLine(p2, p3)  # top of base rectangle
                l3 = gmsh.model.geo.addLine(p3, p0)  # left of base rectangle

                # Define points for the left extrusion
                x_min_l = 0.25 * L_X - 0.5 * self.config.SOURCE_WIDTH
                x_max_l = 0.25 * L_X + 0.5 * self.config.SOURCE_WIDTH
                p4 = gmsh.model.geo.addPoint(x_min_l, L_Y, 0, meshSize=res)  # BL, left ex.
                p5 = gmsh.model.geo.addPoint(x_max_l, L_Y, 0, meshSize=res)  # BR, left ex.
                p6 = gmsh.model.geo.addPoint(x_max_l, L_Y + self.config.SOURCE_HEIGHT, 0, meshSize=res)  # TR, left ex.
                p7 = gmsh.model.geo.addPoint(x_min_l, L_Y + self.config.SOURCE_HEIGHT, 0, meshSize=res)  # TL, left ex.
                # Lines for left extrusion
                # l4 = gmsh.model.geo.addLine(p4, p5)
                l5 = gmsh.model.geo.addLine(p5, p6)  # right of left extrusion
                l6 = gmsh.model.geo.addLine(p6, p7)  # top of left extrusion
                l7 = gmsh.model.geo.addLine(p7, p4)  # left of left extrusion

                # Define points for the right extrusion
                x_min_r = 0.75 * L_X - 0.5 * self.config.SOURCE_WIDTH
                x_max_r = 0.75 * L_X + 0.5 * self.config.SOURCE_WIDTH
                p8 = gmsh.model.geo.addPoint(x_min_r, L_Y, 0, meshSize=res)  # BL, right ex.
                p9 = gmsh.model.geo.addPoint(x_max_r, L_Y, 0, meshSize=res)  # BR, right ex.
                p10 = gmsh.model.geo.addPoint(x_max_r, L_Y + self.config.SOURCE_HEIGHT, 0, meshSize=res)  # TR, left ex.
                p11 = gmsh.model.geo.addPoint(x_min_r, L_Y + self.config.SOURCE_HEIGHT, 0, meshSize=res)  # TL, left ex.
                # Lines for right extrusion
                # l4 = gmsh.model.geo.addLine(p8, p9)
                l8 = gmsh.model.geo.addLine(p9, p10)  # right of right extrusion
                l9 = gmsh.model.geo.addLine(p10, p11)  # top of right extrusion
                l10 = gmsh.model.geo.addLine(p11, p8)  # left of right extrusion

                # Connect the left extrusion to the base rectangle
                l11 = gmsh.model.geo.addLine(p3, p4)
                # Connect the left extrusion to the right extrusion
                l12 = gmsh.model.geo.addLine(p5, p8)
                # connect the right extrusion to the base rectangle
                l13 = gmsh.model.geo.addLine(p9, p2)

                # Define curve loops
                loop_combined = gmsh.model.geo.addCurveLoop(
                    [l0, l1, -l13, l8, l9, l10, -l12, l5, l6, l7, -l11, l3]
                    )
                surface = gmsh.model.geo.addPlaneSurface([loop_combined])

                gmsh.model.geo.synchronize()
                # Physical groups for boundaries
                gmsh.model.addPhysicalGroup(1, [l0], tag=1)
                gmsh.model.setPhysicalName(1, 1, "IsothermalBoundary")
                gmsh.model.addPhysicalGroup(1, [l6], tag=3)
                gmsh.model.setPhysicalName(1, 3, "TopBoundaryLeft")
                gmsh.model.addPhysicalGroup(1, [l9], tag=4)
                gmsh.model.setPhysicalName(1, 4, "TopBoundaryRight")
                gmsh.model.addPhysicalGroup(1, [l1, l13, l8, l10, l12, l5, l7, l11, l3], tag=2)
                gmsh.model.setPhysicalName(1, 2, "SlipBoundary")
                gmsh.model.addPhysicalGroup(2, [surface], tag=1)
                gmsh.model.setPhysicalName(2, 1, "Domain")

            else:
                # Define points for the base rectangle
                p0 = gmsh.model.geo.addPoint(0, 0, 0, meshSize=res)
                p1 = gmsh.model.geo.addPoint(L_X, 0, 0, meshSize=res)
                p2 = gmsh.model.geo.addPoint(L_X, L_Y, 0, meshSize=res)
                p3 = gmsh.model.geo.addPoint(0, L_Y, 0, meshSize=res)

                # Define lines for the base rectangle
                l0 = gmsh.model.geo.addLine(p0, p1)
                l1 = gmsh.model.geo.addLine(p1, p2)
                l2 = gmsh.model.geo.addLine(p2, p3)
                l3 = gmsh.model.geo.addLine(p3, p0)

                # Define points for the extrusion (source region)
                x_min = 0.5 * L_X - 0.5 * self.config.SOURCE_WIDTH
                x_max = 0.5 * L_X + 0.5 * self.config.SOURCE_WIDTH
                p4 = gmsh.model.geo.addPoint(x_min, L_Y, 0, meshSize=res)
                p5 = gmsh.model.geo.addPoint(x_max, L_Y, 0, meshSize=res)
                p6 = gmsh.model.geo.addPoint(x_max, L_Y + self.config.SOURCE_HEIGHT, 0, meshSize=res)
                p7 = gmsh.model.geo.addPoint(x_min, L_Y + self.config.SOURCE_HEIGHT, 0, meshSize=res)

                # Define lines for the extrusion
                l4 = gmsh.model.geo.addLine(p4, p5)
                l5 = gmsh.model.geo.addLine(p5, p6)
                l6 = gmsh.model.geo.addLine(p6, p7)
                l7 = gmsh.model.geo.addLine(p7, p4)

                # Connect the extrusion to the base rectangle
                l8 = gmsh.model.geo.addLine(p3, p4)
                l9 = gmsh.model.geo.addLine(p5, p2)

                # Define curve loops
                loop_combined = gmsh.model.geo.addCurveLoop(
                    [l0, l1, -l9, l5, l6, l7, -l8, l3]
                    )
                surface = gmsh.model.geo.addPlaneSurface([loop_combined])

                gmsh.model.geo.synchronize()
                # Physical groups for boundaries
                gmsh.model.addPhysicalGroup(1, [l0], tag=1)
                gmsh.model.setPhysicalName(1, 1, "IsothermalBoundary")
                gmsh.model.addPhysicalGroup(1, [l6], tag=3)
                gmsh.model.setPhysicalName(1, 3, "TopBoundary")
                gmsh.model.addPhysicalGroup(1, [l1, l9, l5, l7, l8, l3], tag=2)
                gmsh.model.setPhysicalName(1, 2, "SlipBoundary")
                gmsh.model.addPhysicalGroup(2, [surface], tag=1)
                gmsh.model.setPhysicalName(2, 1, "Domain")

            # Generate the mesh
            gmsh.model.mesh.generate(2)

        comm = MPI.COMM_WORLD
        msh, cell_markers, facet_markers = io.gmshio.model_to_mesh(gmsh.model, comm, rank=0, gdim=2)

        if comm.rank == 0:
            gmsh.finalize()

        return msh, cell_markers, facet_markers
