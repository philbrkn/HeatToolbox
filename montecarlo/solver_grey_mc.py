# solver_grey_mc.py

from __future__ import annotations
import math, random, time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from dolfinx import geometry, fem, mesh as dmesh
from dolfinx.mesh import MeshTags
import ufl
import basix.ufl


@dataclass(slots=True)
class Material:
    """Grey‐BTE material parameters."""
    kappa: float  # W·m⁻¹·K⁻¹
    ell:   float  # nonlocal length ℓ (m)
    c:     float  # volumetric heat capacity (J·m⁻³·K⁻¹)

    @property
    def vg(self) -> float:
        Lam = math.sqrt(5.0) * self.ell
        return 3.0 * self.kappa / (self.c * Lam)

    @property
    def tau(self) -> float:
        return math.sqrt(5.0) * self.ell / self.vg


class GreyMC:
    """Serial 2-D steady-state grey-phonon Monte-Carlo."""

    def __init__(
        self,
        msh: dmesh.Mesh,
        facet_tags: MeshTags,
        gamma: fem.Function,
        mat_A: Material,
        mat_B: Material,
        nparticles: int = 200_000,
        seed: int | None = None,
    ):
        self.msh = msh
        self.facets = facet_tags
        self.gamma = gamma
        self.mat_A = mat_A
        self.mat_B = mat_B
        self.nparticles = nparticles

        # RNG
        if seed is None:
            seed = int(time.time())
        random.seed(seed)
        np.random.seed(seed)

        # particle buffers (serial => all in one rank)
        self.n = nparticles
        self.x = np.zeros((self.n, 2))
        self.u = np.zeros((self.n, 2))
        self.alive = np.zeros(self.n, dtype=bool)
        self.ω = 1.0  # will scale in run()

        # geometry helpers
        self.bb = geometry.bb_tree(msh, msh.topology.dim)
        coords = msh.geometry.x
        self.xmin, self.xmax = coords[:,0].min(), coords[:,0].max()
        self.ymin, self.ymax = coords[:,1].min(), coords[:,1].max()

        # tallies per cell
        nc = msh.topology.index_map(2).size_local
        self.E  = np.zeros(nc)    # J
        self.Fx = np.zeros(nc)
        self.Fy = np.zeros(nc)

    def run(
        self,
        heater_tag: int,
        q_heater: float,
        *,
        dt: float = 5e-12,
        nsteps: int = 200,
    ) -> Tuple[fem.Function, fem.Function]:
        # compute heater total length
        Lh = self._heater_length(heater_tag)
        if Lh == 0:
            raise RuntimeError(f"No facets tagged {heater_tag}")
        # choose ω such that total energy injected per dt = q_heater * Lh
        self.ω = q_heater * Lh * dt / self.n

        self._emit(heater_tag)
        for _ in range(nsteps):
            self._advect(dt)
            self._scatter(dt)

        return self._reduce()

    # --- helpers ------------------------------------------------------
    def _heater_length(self, tag: int) -> float:
        mask    = (self.facets.values == tag)
        fids    = self.facets.indices[mask]
        self.msh.topology.create_connectivity(1, 0)
        conn    = self.msh.topology.connectivity(1, 0)
        coords  = self.msh.geometry.x
        total_L = 0.0
        for f in fids:
            v = conn.links(f)
            total_L += np.linalg.norm(coords[v[1]] - coords[v[0]])
        return total_L

    def _emit(self, tag: int):
        mask = (self.facets.values == tag)
        fids = self.facets.indices[mask]
        # prepare midpoints & normals
        self.msh.topology.create_connectivity(1, 0)
        conn   = self.msh.topology.connectivity(1, 0)
        coords = self.msh.geometry.x
        mids, normals = [], []
        for f in fids:
            v = conn.links(f)
            p0, p1 = coords[v]
            mids.append(0.5*(p0+p1))
            t = p1 - p0
            n = np.array([-t[1], t[0]]) / np.linalg.norm(t)
            normals.append(n)
        mids = np.array(mids)
        normals = np.array(normals)
        # drop the z‐component for a 2D run
        mids    = mids[:, :2]
        normals = normals[:, :2]
        # assign each particle
        for i in range(self.n):
            f = i % len(mids)
            self.x[i, :] = mids[f]
            # hemi-isotropic
            r1, r2 = random.random(), random.random()
            θ = 2*math.pi*r1
            μ = math.sqrt(r2)
            v = np.array([
                math.sqrt(1-μ*μ)*math.cos(θ),
                math.sqrt(1-μ*μ)*math.sin(θ),
            ])
            if np.dot(v, normals[f]) < 0:
                v *= -1
            self.u[i] = v
            self.alive[i] = True

    def _advect(self, dt: float):
        idx = np.where(self.alive)[0]
        if idx.size == 0:
            return
        mat_idx = self._mat(idx)
        speeds  = np.where(mat_idx==0, self.mat_A.vg, self.mat_B.vg)
        self.x[idx] += speeds[:,None] * dt * self.u[idx]

        # bottom absorption → deposit
        bottom = self.x[idx,1] < self.ymin
        if bottom.any():
            self._deposit(idx[bottom])
            self.alive[idx[bottom]] = False

        # top diffuse reflect
        top = self.x[idx,1] > self.ymax
        if top.any():
            self.u[idx[top],1]  *= -1
            self.x[idx[top],1]  = 2*self.ymax - self.x[idx[top],1]

        # left/right reflect
        left  = self.x[idx,0] < self.xmin
        right = self.x[idx,0] > self.xmax
        lr = left | right
        if lr.any():
            self.u[idx[lr],0] *= -1
            self.x[idx[left],0]  = 2*self.xmin - self.x[idx[left],0]
            self.x[idx[right],0] = 2*self.xmax - self.x[idx[right],0]

    def _deposit(self, part_idx: np.ndarray):
        pts2d = self.x[part_idx]
        pts = np.hstack([pts2d, np.zeros((pts2d.shape[0], 1))])  # (N, 3)
        cps = geometry.compute_collisions_points(self.bb, pts)
        cells = geometry.compute_colliding_cells(self.msh, cps, pts)
        for k, p in enumerate(part_idx):
            links = cells.links(k)
            if len(links) == 0:
                # this particle fell outside the mesh—skip depositing
                continue
            cell = links[0]        
            self.E[cell]  += self.ω
            self.Fx[cell] += self.ω * self.u[p, 0]
            self.Fy[cell] += self.ω * self.u[p, 1]

    def _scatter(self, dt: float):
        idx = np.where(self.alive)[0]
        if idx.size == 0:
            return
        taus = np.where(self._mat(idx)==0, self.mat_A.tau, self.mat_B.tau)
        prob = 1 - np.exp(-dt/taus)
        coll = np.random.random(idx.size) < prob
        sel = idx[coll]
        θ = 2*math.pi*np.random.random(sel.size)
        self.u[sel] = np.column_stack((np.cos(θ), np.sin(θ)))

    def _mat(self, idx: np.ndarray) -> np.ndarray:
        # take the 2D (x,y) positions and promote to 3D (x,y,0)
        pts2 = self.x[idx]                         # shape (npts,2)
        pts  = np.hstack([pts2, np.zeros((pts2.shape[0],1))])  # shape (npts,3)

        cps = geometry.compute_collisions_points(self.bb, pts)
        cells = geometry.compute_colliding_cells(self.msh, cps, pts)
        out = np.zeros(idx.size, dtype=np.int8)
        for k in range(idx.size):
            links = cells.links(k)
            # only assign if this point actually landed in a cell
            if len(links) > 0:
                cell_index = links[0]
                out[k] = int(self.gamma.x.array[cell_index] > 0.5)
        return out


    def _reduce(self) -> tuple[fem.Function, fem.Function]:
        # --- 1) Build DG0 spaces for T (scalar) and q (2-vector) ---
        P0   = basix.ufl.element("DG", self.msh.basix_cell(), 0)
        V_T  = fem.functionspace(self.msh, P0)
        T    = fem.Function(V_T)

        P0v  = basix.ufl.element("DG", self.msh.basix_cell(), 0, shape=(2,))
        V_q  = fem.functionspace(self.msh, P0v)
        q    = fem.Function(V_q)

        # --- 2) Assemble the cell-area vector via ∫ v * dx over each cell ---
        v_test = ufl.TestFunction(V_T)
        dx     = ufl.dx(domain=self.msh)
        area_form = fem.form(v_test * dx)

        # assemble_vector gives you a PETSc Vec whose i-th entry is ∫_cell_i v_i dx = area(cell_i)
        area_vec = fem.assemble_vector(area_form)
        # get it as a NumPy array
        vol = area_vec.array   # shape = (n_local_cells,)

        # --- 3) Build c per cell from gamma (also DG0) ---
        # gamma.x.array is length = n_local_cells
        c_cells = np.where(self.gamma.x.array < 0.5,
                        self.mat_A.c,
                        self.mat_B.c)

        # --- 4) Fill in T and q from your tallies ---
        # E, Fx, Fy are length = n_local_cells
        T.x.array[:] = self.E / (c_cells * vol)

        # q.x.array is flat length = 2 * n_local_cells: reshape to (ncells,2)
        qval = q.x.array.reshape(-1, 2)
        qval[:, 0] = self.Fx / vol
        qval[:, 1] = self.Fy / vol
        q.x.array[:] = qval.reshape(-1)

        return T, q