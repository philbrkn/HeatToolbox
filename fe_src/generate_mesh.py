import gmsh
from mpi4py import MPI
from dolfinx.io import gmshio


def create_mesh(L):
    # generate_mesh.py

    # Initialize Gmsh
    gmsh.initialize()
    gmsh.model.add("domain_with_extrusion")

    # Define parameters
    Lx = 5 * L
    Ly = 2.5 * L
    source_width = L*0.5
    source_height = L * 0.25
    resolution = L / 64  # Adjust mesh resolution as needed

    # Define points for the base rectangle
    p0 = gmsh.model.geo.addPoint(0, 0, 0, meshSize=resolution)
    p1 = gmsh.model.geo.addPoint(Lx, 0, 0, meshSize=resolution)
    p2 = gmsh.model.geo.addPoint(Lx, Ly, 0, meshSize=resolution)
    p3 = gmsh.model.geo.addPoint(0, Ly, 0, meshSize=resolution)

    # Define lines for the base rectangle
    l0 = gmsh.model.geo.addLine(p0, p1)
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p0)

    # Define points for the extrusion (source region)
    x_min = 0.5 * Lx - 0.5 * source_width
    x_max = 0.5 * Lx + 0.5 * source_width
    p4 = gmsh.model.geo.addPoint(x_min, Ly, 0, meshSize=resolution)
    p5 = gmsh.model.geo.addPoint(x_max, Ly, 0, meshSize=resolution)
    p6 = gmsh.model.geo.addPoint(x_max, Ly + source_height, 0, meshSize=resolution)
    p7 = gmsh.model.geo.addPoint(x_min, Ly + source_height, 0, meshSize=resolution)

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

    # Define physical groups for boundaries
    # Isothermal boundary
    gmsh.model.addPhysicalGroup(1, [l0], tag=1)
    gmsh.model.setPhysicalName(1, 1, "IsothermalBoundary")
    # Source boundary
    gmsh.model.addPhysicalGroup(1, [l6], tag=3)
    gmsh.model.setPhysicalName(1, 3, "TopBoundary")
    # Slip boundary
    gmsh.model.addPhysicalGroup(1, [l1, l9, l5, l7, l8, l3], tag=2)
    gmsh.model.setPhysicalName(1, 2, "SlipBoundary")


    # Define physical groups for domains (if needed)
    gmsh.model.addPhysicalGroup(2, [surface], tag=1)
    gmsh.model.setPhysicalName(2, 1, "Domain")

    # Generate the mesh
    gmsh.model.mesh.generate(2)

    # Write mesh to file
    # gmsh.write("domain_with_extrusion.msh")
    domain, cell_markers, facet_markers = gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, rank=0, gdim=2
    )

    # Finalize Gmsh
    gmsh.finalize()

    return domain, cell_markers, facet_markers
