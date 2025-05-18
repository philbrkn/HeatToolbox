# run_mc_gke_verification.py
"""Minimal driver that cross‑checks the FEniCSx Guyer–Krumhansl solver
with the Grey deviational Monte‑Carlo reference on a sharp Si | Diamond
interface.

Usage (single process works fine for the verification size)::

    mpirun -n 4 python run_verification.py  # optional MPI

The script prints the L2‑norm error in temperature between the two
solutions and exits with code 0 if the error < 5 %.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import gmsh
import numpy as np
from mpi4py import MPI

import ufl
from dolfinx import fem, mesh, io
from dolfinx.fem.petsc import (create_matrix, create_vector, apply_lifting,
                               assemble_matrix, assemble_vector, set_bc)
from dolfinx.io import gmshio
# import geometry
from dolfinx import geometry


# --- project imports --------------------------------------------------
from heatoptim.solvers.solver_gke_module import GKESolver
from heatoptim.config.sim_config import SimulationConfig
from montecarlo.solver_grey_mc import Material, GreyMC  # module added previously
from dolfinx.mesh import Mesh, MeshTags
# import tuple
from typing import Tuple


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


def main():
    h = 0.25e-6
    Lx, Ly = 20*h, 10*h
    mshfile = "split.msh"
    build_split_domain_msh(mshfile, Lx, Ly, h)

    msh, cell_tags, facet_tags = gmshio.read_from_msh(mshfile, MPI.COMM_SELF, gdim=2)

    # GMC solver
    cfg = SimulationConfig({"solver_type":"gke","blank":True,"log_name":"verify"})
    gk  = GKESolver(msh, facet_tags, cfg)
    gk.gamma.interpolate(lambda x: np.where(x[0]<Lx/2,0.0,1.0))
    F = gk.define_variational_form()
    gk.solve_problem(F)
    q_gk, T_gk = gk.U.sub(0).collapse(), gk.U.sub(1).collapse()

    # Grey MC ref
    matA = Material(cfg.KAPPA_SI, cfg.ELL_SI, 1.63e6)
    matB = Material(cfg.KAPPA_DI, cfg.ELL_DI, 1.63e6)
    mc   = GreyMC(msh, facet_tags, gk.gamma, matA, matB, nparticles=50_000)
    T_mc, q_mc = mc.run(heater_tag=3, q_heater=80.0, dt=5e-12, nsteps=200)

    # write XDMF
    with io.XDMFFile(msh.comm, "verify.xdmf","w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(T_gk, "T_GK")
        xdmf.write_function(q_gk, "q_GK")
        xdmf.write_function(T_mc, "T_MC")
        xdmf.write_function(q_mc, "q_MC")

    # L2 error
    e2 = fem.assemble_scalar(fem.form(((T_gk-T_mc)**2)*ufl.dx(domain=msh)))
    n2 = fem.assemble_scalar(fem.form((T_mc**2)*ufl.dx(domain=msh)))
    rel = math.sqrt(e2/n2)
    print(f"Relative L2 error: {rel:.2%}")
    sys.exit(0 if rel<0.05 else 1)

    # sample horizontal mid-height line
    from dolfinx import geometry
    y_mid = 0.5*Ly
    x_vals = np.linspace(0, Lx, 400)
    pts = np.column_stack((x_vals, np.full_like(x_vals, y_mid), np.zeros_like(x_vals)))
    cells = geometry.compute_colliding_cells(msh,
            geometry.compute_collisions_points(geometry.bb_tree(msh, 2), pts), pts)
    Tg = T_gk.eval(pts, cells)
    Tm = T_mc.eval(pts, cells)

    import matplotlib.pyplot as plt
    plt.plot(x_vals*1e6, Tg, label="GK")
    plt.plot(x_vals*1e6, Tm, '--', label="MC")
    plt.xlabel("x [µm]"); plt.ylabel("ΔT [K]")
    plt.legend(); plt.show()


if __name__ == "__main__":
    main()
