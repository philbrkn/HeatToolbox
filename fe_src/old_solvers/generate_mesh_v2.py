import gmsh
from mpi4py import MPI
from dolfinx.io import gmshio


def create_mesh(L):
    gmsh.initialize()
    gmsh.model.add("domain_with_extrusion")

    # Define parameters
    Lx = 25 * L
    Ly = 12.5 * L
    source_width = L
    source_height = 0.25 * L
    resolution = L / 10  # Mesh resolution

    # Set mesh element size
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", resolution)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", resolution)
    
    # Create the main rectangular domain
    main_domain = gmsh.model.occ.addRectangle(0, 0, 0, Lx, Ly)

    # Create the extrusion (source region) as a rectangle
    x_min = 0.5 * Lx - 0.5 * source_width
    x_max = 0.5 * Lx + 0.5 * source_width

    extrusion = gmsh.model.occ.addRectangle(x_min, Ly, 0, source_width, source_height)
    
    # Fragment the two rectangles to ensure shared nodes and edges
    gmsh.model.occ.fragment([(2, main_domain)], [(2, extrusion)])
    gmsh.model.occ.synchronize()
    
    surfaces = gmsh.model.occ.getEntities(dim=2)
    # Initialize lists to hold surface tags for the rectangle and extrusion
    rectangle_surfaces = []
    extrusion_surfaces = []

    # Categorize surfaces based on their center of mass
    for dim, tag in surfaces:
        x, y, z = gmsh.model.occ.getCenterOfMass(dim, tag)
        if y < Ly + source_height / 2:
            rectangle_surfaces.append(tag)
        else:
            extrusion_surfaces.append(tag)

    # Define physical groups for the rectangle and the extrusion
    rectangle_group = gmsh.model.addPhysicalGroup(2, rectangle_surfaces, tag=1)
    extrusion_group = gmsh.model.addPhysicalGroup(2, extrusion_surfaces, tag=2)

    # Retrieve the lines created after fragmentation
    lines = gmsh.model.occ.getEntities(dim=1)

    # Initialize lists to hold tags for the bottom line and the rest of the lines
    bottom_line = []
    rest_of_lines = []
    top_line = []
    # Categorize lines based on their geometric properties
    for dim, tag in lines:
        x, y, z = gmsh.model.occ.getCenterOfMass(dim, tag)

        # Bottom line (where y = 0)
        if abs(y) < (resolution/1e-6):   # Small tolerance to account for floating point precision
            bottom_line.append(tag)
        # top line (where y = Ly + source_height)
        elif abs(y - (Ly + source_height)) < (resolution/1e-6):
            top_line.append(tag)
        else:
            rest_of_lines.append(tag)

    # Define physical groups for the bottom line and other lines
    isothermal_line = gmsh.model.addPhysicalGroup(1, bottom_line, tag=1)
    slip_lines = gmsh.model.addPhysicalGroup(1, rest_of_lines, tag=2)
    top_line = gmsh.model.addPhysicalGroup(1, top_line, tag=3)
    gmsh.model.mesh.generate(2)

    # Write mesh to file
    # gmsh.write("domain_with_extrusion.msh")
    domain, cell_markers, facet_markers = gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, rank=0, gdim=2
    )

    # Finalize Gmsh
    gmsh.finalize()

    return domain, cell_markers, facet_markers


def create_mesh_scaled(L, scaling_factor):
    gmsh.initialize()
    gmsh.model.add("domain_with_extrusion")

    # Define parameters
    Lx = 25 * L * scaling_factor
    Ly = 12.5 * L * scaling_factor
    source_width = L * scaling_factor
    source_height = 0.25 * L * scaling_factor
    resolution = L / 10 * scaling_factor  # Mesh resolution

    # Set mesh element size
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", resolution)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", resolution)
    
    # Create the main rectangular domain
    main_domain = gmsh.model.occ.addRectangle(0, 0, 0, Lx, Ly)

    # Create the extrusion (source region) as a rectangle
    x_min = 0.5 * Lx - 0.5 * source_width
    x_max = 0.5 * Lx + 0.5 * source_width

    extrusion = gmsh.model.occ.addRectangle(x_min, Ly, 0, source_width, source_height)
    
    # Fragment the two rectangles to ensure shared nodes and edges
    gmsh.model.occ.fragment([(2, main_domain)], [(2, extrusion)])
    gmsh.model.occ.synchronize()
    
    surfaces = gmsh.model.occ.getEntities(dim=2)
    # Initialize lists to hold surface tags for the rectangle and extrusion
    rectangle_surfaces = []
    extrusion_surfaces = []

    # Categorize surfaces based on their center of mass
    for dim, tag in surfaces:
        x, y, z = gmsh.model.occ.getCenterOfMass(dim, tag)
        if y < Ly + source_height / 2:
            rectangle_surfaces.append(tag)
        else:
            extrusion_surfaces.append(tag)

    # Define physical groups for the rectangle and the extrusion
    rectangle_group = gmsh.model.addPhysicalGroup(2, rectangle_surfaces, tag=1)
    extrusion_group = gmsh.model.addPhysicalGroup(2, extrusion_surfaces, tag=2)

    # Retrieve the lines created after fragmentation
    lines = gmsh.model.occ.getEntities(dim=1)

    # Initialize lists to hold tags for the bottom line and the rest of the lines
    bottom_line = []
    rest_of_lines = []
    top_lines = []
    # Categorize lines based on their geometric properties
    for dim, tag in lines:
        x, y, z = gmsh.model.occ.getCenterOfMass(dim, tag)

        # Bottom line (where y = 0)
        if abs(y) < (L/1e-7):   # Small tolerance to account for floating point precision
            bottom_line.append(tag)
        # top line (where y = Ly + source_height)
        elif abs(y - (Ly + source_height)) < (L/1e-7):
            top_lines.append(tag)
        else:
            rest_of_lines.append(tag)

    # Define physical groups for the bottom line and other lines
    isothermal_line = gmsh.model.addPhysicalGroup(1, bottom_line, tag=1)
    slip_lines = gmsh.model.addPhysicalGroup(1, rest_of_lines, tag=2)
    source_line = gmsh.model.addPhysicalGroup(1, top_lines, tag=3)

    gmsh.model.mesh.generate(2)

    # Write mesh to file
    # gmsh.write("domain_with_extrusion.msh")
    domain, cell_markers, facet_markers = gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, rank=0, gdim=2
    )

    # Scale the mesh coordinates back to original size
    domain.geometry.x[:] /= scaling_factor

    # Finalize Gmsh
    gmsh.finalize()

    return domain, cell_markers, facet_markers
