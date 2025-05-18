import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from montecarlo.solver_grey_mc import Material, GreyMC
from heatoptim.config.sim_config import SimulationConfig
import pyvista as pv
from dolfinx import plot
import gmsh


def build_split_domain_msh(path: str, Lx: float, Ly: float, h: float):
    """Return (msh, cell_tags, facet_tags) for a Si (left) | Diamond (right) bar.

    A vertical interface at x = Lx/2 is marked with a 1‑element BoundaryLayer
    so that the jump in material properties happens over *exactly* one element.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("split_domain")

    # geometry ---------------------------------------------------------
    p0 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, h)  # bottom left
    p1 = gmsh.model.geo.addPoint(Lx, 0.0, 0.0, h)  # bottom right
    p2 = gmsh.model.geo.addPoint(Lx, Ly, 0.0, h)  # top right
    p3 = gmsh.model.geo.addPoint(0.0, Ly, 0.0, h)  # top left

    # Left boundary, bottom, top
    l_bottom = gmsh.model.geo.addLine(p0, p1)  # bottom
    l_right = gmsh.model.geo.addLine(p1, p2)  # right
    l_top = gmsh.model.geo.addLine(p2, p3)  # top
    l_left = gmsh.model.geo.addLine(p3, p0)  # left

    # interface line x = Lx/2
    p4 = gmsh.model.geo.addPoint(Lx / 2, 0.0, 0.0, h)  # bottom midpoint
    p5 = gmsh.model.geo.addPoint(Lx / 2, Ly, 0.0, h)  # top midpoint
    l_if = gmsh.model.geo.addLine(p4, p5)  # interface

    # Split the rectangle into two surfaces (left and right)
    # Build a curve loop for the left block (p0‑p4‑p5‑p3)
    l0 = gmsh.model.geo.addLine(p0, p4)  # bottom-left to bottom midpoint
    l1 = gmsh.model.geo.addLine(p4, p5)  # bottom midpoint to top midpoint
    l2 = gmsh.model.geo.addLine(p5, p3)  # top midpoint to top-left

    # loop : left line, bottom line , midline, top line
    loop_left = gmsh.model.geo.addCurveLoop([l_left, l0, l1, l2])
    surf_left = gmsh.model.geo.addPlaneSurface([loop_left])

    # right block uses existing lines: p4‑p1‑p2‑p5
    l3 = gmsh.model.geo.addLine(p4, p1)  # bottom midpoint to bottom right
    l4 = gmsh.model.geo.addLine(p5, p2)  # top midpoint to top right
    loop_right = gmsh.model.geo.addCurveLoop([l3, l_right, -l4, -l1])
    surf_right = gmsh.model.geo.addPlaneSurface([loop_right])

    gmsh.model.geo.synchronize()

    # --- mesh‑size field: 1‑element BoundaryLayer at the interface ----
    field = gmsh.model.mesh.field
    bl = field.add("BoundaryLayer")
    field.setNumbers(bl, "CurvesList", [l_if])
    field.setNumber(bl, "NbLayers", 1)
    field.setNumber(bl, "Size", h)           # interface cell height
    field.setNumber(bl, "SizeFar", 5 * h)
    field.setNumber(bl, "Ratio", 1.0)
    field.setAsBackgroundMesh(bl)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)

    # physical tags ----------------------------------------------------
    gmsh.model.addPhysicalGroup(2, [surf_left], tag=11)
    gmsh.model.setPhysicalName(2, 11, "Si")
    gmsh.model.addPhysicalGroup(2, [surf_right], tag=12)
    gmsh.model.setPhysicalName(2, 12, "Diamond")

    gmsh.model.addPhysicalGroup(1, [l_bottom], tag=1)  # isothermal

    heater_lines = [l2, l4]
    gmsh.model.addPhysicalGroup(1, heater_lines, tag=3)
    gmsh.model.setPhysicalName(1, 3, "HeaterBoundary")
    gmsh.model.addPhysicalGroup(1, [l_left, l_right], tag=2)  # slip/adiabatic

    gmsh.model.mesh.generate(2)

    gmsh.write(path)

    gmsh.finalize()



# -----------------------------------------------------------------------------
# 1) Configuration (tweak these to speed up / slow down)
# -----------------------------------------------------------------------------
h      = 0.25e-6           # element size
Lx, Ly = 20*h, 10*h        # domain dims
npart  = 20000             # number of MC particles
nsteps = 500              # time steps
dt     = 5e-12             # MC time step
Qheat  = 80.0              # heater flux [W/m]
heater_tag = 3             # as in your mesh generator

# pick your two materials from sim config
cfg = SimulationConfig({
    "solver_type": "gke",   # unused here, just to pull constants
    "KAPPA_SI": None,       # these will be filled in below
    "ELL_SI":   None,
    "KAPPA_DI": None,
    "ELL_DI":   None,
    "log_name": "verify",
})
# override with real numbers (from your sim_config defaults)
cfg.KAPPA_SI, cfg.ELL_SI = 149.0, 300e-9    # example: Si
cfg.KAPPA_DI, cfg.ELL_DI = 2000.0, 1000e-9  # example: Diamond
c_si = 1.63e6   # J/m3K (from Tur-Prats et al.)

# -----------------------------------------------------------------------------
# 2) Build the split‐domain mesh (Si on left, diamond on right)
# -----------------------------------------------------------------------------
mshfile = "split_verify.msh"
build_split_domain_msh(mshfile, Lx, Ly, h)
msh, cell_tags, facet_tags = \
    __import__('dolfinx.io').io.gmshio.read_from_msh(mshfile, MPI.COMM_SELF, gdim=2)

# -----------------------------------------------------------------------------
# 3) Set up Grey–MC solver and run
# -----------------------------------------------------------------------------
# create a 'gamma' that is 0 in Si, 1 in diamond
from dolfinx import fem
V_gamma = fem.functionspace(msh, ("CG", 1))   # scalar, CG-1
gamma   = fem.Function(V_gamma)

# tabulate the coordinates of each DOF that lives on *this* process
coords = V_gamma.tabulate_dof_coordinates()

# build the local array (same length as coords[:,0])
gamma.x.array[:] = np.where(coords[:, 0] >= Lx/2, 1.0, 0.0)

# make sure ghost values are updated before you use gamma in forms
gamma.x.scatter_forward()

matA = Material(cfg.KAPPA_SI, cfg.ELL_SI,   c_si)
matB = Material(cfg.KAPPA_DI, cfg.ELL_DI,   c_si)
mc   = GreyMC(msh, facet_tags, gamma, matA, matB, nparticles=npart, seed=42)

vmax = max(matA.vg, matB.vg)
t_travel = (Ly - mc.ymin)/vmax
needed_steps = int(np.ceil(t_travel/dt)) + 10
print("need at least", needed_steps, "steps to reach bottom")

T_mc, q_mc = mc.run(heater_tag=heater_tag, q_heater=Qheat, dt=dt, nsteps=nsteps)

# right after T_mc, q_mc = mc.run(…)
print("ΔT min:", T_mc.x.array.min(), "ΔT max:", T_mc.x.array.max())
# how much energy did we deposit at all?
print("Total deposited energy:", mc.E.sum())
print("  number of cells with nonzero E:", np.count_nonzero(mc.E))


# --- 1) Extract VTK mesh data for a DG0 space (same for T_mc and q_mc)
V0 = T_mc.function_space  # DG0 scalar for T_mc
cells, cell_types, geometry = plot.vtk_mesh(msh, msh.topology.dim)

# 2) Build a PyVista unstructured grid
grid = pv.UnstructuredGrid(cells, cell_types, geometry)

# 3) Attach your MC temperature as a cell‐data array
#    T_mc.x.array is length = n_cells
grid.cell_data["T_MC"] = np.asarray(T_mc.x.array, dtype=float)

# 4) Attach your MC flux as two cell‐data arrays
#    q_mc.x.array is flat [qx0, qy0, qx1, qy1, …]
qvals = np.asarray(q_mc.x.array).reshape(-1, 2)
grid.cell_data["q_MC_x"] = qvals[:, 0]
grid.cell_data["q_MC_y"] = qvals[:, 1]

# --- 5) Plot with PyVista ---
plotter = pv.Plotter()
# Temperature contour
plotter.add_mesh(grid, scalars="T_MC", cmap="coolwarm", show_edges=False)
# plotter.add_scalar_bar("ΔT [K]")
plotter.view_xy()
plotter.show()

# Flux vector field
# Create glyphs (arrows) to visualize q⃗
# arrows = grid.glyph(
#     orient="q_MC_x", scale=False,  # orient by x‐component alone won't work; instead:
#     vector="q_MC",                 # but PyVista needs a single named vector, so:
# )
# # Alternatively combine into a 3‐component array with zero z:
# vec3 = np.column_stack([qvals, np.zeros(qvals.shape[0])])
# grid.cell_data["q_MC"] = vec3
# arrows = grid.glyph(orient="q_MC", scale=False, factor=0.5*h)

# plotter = pv.Plotter()
# plotter.add_mesh(grid, show_edges=False)
# plotter.add_mesh(arrows, color="black")
# plotter.view_xy()
# plotter.show()