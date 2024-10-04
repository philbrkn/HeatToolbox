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
            y_max = self.config.L_Y + self.config.SOURCE_HEIGHT
            res = self.config.RESOLUTION

            if self.config.symmetry:
                # Define points for the base rectangle
                p0 = gmsh.model.geo.addPoint(0, 0, 0, meshSize=res)
                p1 = gmsh.model.geo.addPoint(L_X, 0, 0, meshSize=res)
                p2 = gmsh.model.geo.addPoint(L_X, y_max, 0, meshSize=res)
                p3 = gmsh.model.geo.addPoint(0, self.config.L_Y, 0, meshSize=res)

                # Define lines for the base rectangle
                l0 = gmsh.model.geo.addLine(p0, p1)
                l1 = gmsh.model.geo.addLine(p1, p2)
                l3 = gmsh.model.geo.addLine(p3, p0)

                # Define points for the extrusion (source region)
                x_min = L_X - self.config.SOURCE_WIDTH
                p4 = gmsh.model.geo.addPoint(x_min, self.config.L_Y, 0, meshSize=res)
                p7 = gmsh.model.geo.addPoint(x_min, y_max, 0, meshSize=res)

                # Define lines for the extrusion
                l6 = gmsh.model.geo.addLine(p2, p7)
                l7 = gmsh.model.geo.addLine(p7, p4)

                # Connect the extrusion to the base rectangle
                l8 = gmsh.model.geo.addLine(p3, p4)

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
                gmsh.model.addPhysicalGroup(1, [l1], tag=4)
                gmsh.model.setPhysicalName(1, 4, "Symmetry")
                gmsh.model.addPhysicalGroup(2, [surface], tag=1)
                gmsh.model.setPhysicalName(2, 1, "Domain")
            else:
                # Define points for the base rectangle
                p0 = gmsh.model.geo.addPoint(0, 0, 0, meshSize=res)
                p1 = gmsh.model.geo.addPoint(L_X, 0, 0, meshSize=res)
                p2 = gmsh.model.geo.addPoint(L_X, self.config.L_Y, 0, meshSize=res)
                p3 = gmsh.model.geo.addPoint(0, self.config.L_Y, 0, meshSize=res)

                # Define lines for the base rectangle
                l0 = gmsh.model.geo.addLine(p0, p1)
                l1 = gmsh.model.geo.addLine(p1, p2)
                l2 = gmsh.model.geo.addLine(p2, p3)
                l3 = gmsh.model.geo.addLine(p3, p0)

                # Define points for the extrusion (source region)
                x_min = 0.5 * L_X - 0.5 * self.config.SOURCE_WIDTH
                x_max = 0.5 * L_X + 0.5 * self.config.SOURCE_WIDTH
                p4 = gmsh.model.geo.addPoint(x_min, self.config.L_Y, 0, meshSize=res)
                p5 = gmsh.model.geo.addPoint(x_max, self.config.L_Y, 0, meshSize=res)
                p6 = gmsh.model.geo.addPoint(x_max, self.config.L_Y + self.config.SOURCE_HEIGHT, 0, meshSize=res)
                p7 = gmsh.model.geo.addPoint(x_min, self.config.L_Y + self.config.SOURCE_HEIGHT, 0, meshSize=res)

                # Define lines for the extrusion
                l4 = gmsh.model.geo.addLine(p4, p5)
                l5 = gmsh.model.geo.addLine(p5, p6)
                l6 = gmsh.model.geo.addLine(p6, p7)
                l7 = gmsh.model.geo.addLine(p7, p4)

                # Connect the extrusion to the base rectangle
                l8 = gmsh.model.geo.addLine(p3, p4)
                l9 = gmsh.model.geo.addLine(p5, p2)

                # Define curve loops
                loop_combined = gmsh.model.geo.addCurveLoop([l0, l1, -l9, l5, l6, l7, -l8, l3])
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