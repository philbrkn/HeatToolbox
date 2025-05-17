import math
import random
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from mpi4py import MPI
from dolfinx.cpp.mesh import entities_to_geometry
from dolfinx.fem import Function
from dolfinx.mesh import Mesh
from dolfinx import geometry
from petsc4py import PETSc

__all__ = [
    "Material",
    "GreyMC",
]


@dataclass
class Material:
    """Thermal properties required by the grey deviational Monte‑Carlo solver."""

    kappa: float  # W m⁻¹ K⁻¹
    ell: float  # non‑local length ℓ (m)
    c: float  # volumetric heat capacity (J m⁻³ K⁻¹)

    @property
    def velocity(self) -> float:
        """Group velocity recovered from Fourier conductivity (κ = ⅓ c v Λ)."""
        lam = math.sqrt(5.0) * self.ell  # Λ = √5 ℓ for G‑K ↔ BTE consistency
        return 3.0 * self.kappa / (self.c * lam)

    @property
    def relaxation_time(self) -> float:
        return math.sqrt(5.0) * self.ell / self.velocity


class GreyMC:
    """Grey deviatonal Monte‑Carlo solver for 2‑D steady‑state heat conduction.

    The algorithm follows Tur‑Prats *et al.* (2024) and Minasian & Chen (2015).
    Each particle carries a fixed energy *weight* ω with sign ±1 and propagates
    ballistically with speed |v| = v_g until a Poisson‑distributed scattering
    time τ. Upon scattering the direction is re‑randomised (isotropic). Bottom
    boundary is isothermal/absorbing; all others are perfectly diffusive.
    """

    def __init__(
        self,
        msh: Mesh,
        gamma: Function,
        mat_A: Material,
        mat_B: Material,
        nparticles: int = 200_000,
        seed: int | None = None,
    ) -> None:
        self.msh = msh
        self.gamma = gamma
        self.mat_A = mat_A
        self.mat_B = mat_B
        self.comm = msh.comm
        self.rank = self.comm.rank
        self.world = MPI.COMM_WORLD

        self.np_total = nparticles
        if seed is None:
            seed = int(time.time()) + self.rank
        random.seed(seed)
        np.random.seed(seed)

        # Split particle batch among MPI ranks (simple round‑robin)
        counts = self._even_split(self.np_total, self.world.size)
        self.n_local = counts[self.rank]

        # Prepare particle arrays
        self._x = np.empty((self.n_local, 2), dtype=np.float64)
        self._dir = np.empty_like(self._x)
        self._alive = np.zeros(self.n_local, dtype=np.uint8)
        self._weight = np.ones(self.n_local, dtype=np.int8)  # +1 or −1

        # Pre‑compute cell bounding boxes for point‑location (serial per rank)
        self.bounding_boxes = geometry.compute_collisions(
            geometry.compute_entity_bounding_boxes(self.msh, 2),
            np.zeros((1, 2), dtype=np.float64),  # dummy, we supply points later
        )[0]

        # Per‑cell tallies
        ncells = self.msh.topology.index_map(2).size_local + self.msh.topology.index_map(2).num_ghosts
        self._energy = np.zeros(ncells, dtype=np.float64)
        self._flux = np.zeros((ncells, 2), dtype=np.float64)

    # ---------------------------------------------------------------------
    # Public interface
    # ---------------------------------------------------------------------

    def run(
        self,
        heater_facets: List[int],
        q_heater: float,
        dt: float,
        nsteps: int,
        omega: float,
        progress: bool = True,
    ) -> Tuple[Function, Function]:
        """Execute the MC simulation.

        Parameters
        ----------
        heater_facets : list[int]
            Index list (in facet marker order) defining the top heater.
        q_heater : float
            Power density (W m⁻²) deposited uniformly on *heater_facets*.
        dt : float
            Time step for flight segments (s).
        nsteps : int
            Total number of flight / collision iterations.
        omega : float
            Energy weight carried by each particle (J).
        progress : bool, optional
            Print log every 10 % of total steps if *True*.

        Returns
        -------
        T, q : dolfinx.fem.Function
            Temperature rise (K) and heat‑flux vector field (W m⁻²).
        """
        if progress and self.rank == 0:
            PETSc.Sys.Print("Launching %d particles per rank" % self.n_local)

        # Emit initial particles from the heater
        self._emit_particles(heater_facets, q_heater, omega)

        # Main loop
        for istep in range(1, nsteps + 1):
            self._advect(dt)
            self._scatter(dt)
            if progress and istep % max(1, nsteps // 10) == 0 and self.rank == 0:
                PETSc.Sys.Print(f"MC {istep / nsteps:5.1%} complete")

        # Gather tallies and form FEM fields
        return self._reduce_to_functions()

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _even_split(N: int, P: int) -> List[int]:
        base = N // P
        rem = N % P
        return [base + (1 if r < rem else 0) for r in range(P)]

    # ------------------------------------------------------------------
    # Particle operations
    # ------------------------------------------------------------------

    def _emit_particles(self, heater_facets: List[int], q_heater: float, omega: float):
        """Spawn particles on the heater facets according to *q_heater*."""
        # Determine total heater length on this rank
        mesh_facets = self.msh.topology.index_map(1).size_local
        coord = self.msh.geometry.x
        indices = []
        for f in heater_facets:
            if f < mesh_facets:
                indices.append(f)
        if not indices:
            return  # nothing on this rank

        # Simple uniform distribution along each facet midpoint
        midpts = []
        normals = []
        for f in indices:
            cells = entities_to_geometry(self.msh, 1)[f]
            v = self.msh.topology.connectivity(1, 0).links(f)
            pts = coord[v]
            midpts.append(pts.mean(axis=0))
            # outward normal (approx.)
            dx = pts[1] - pts[0]
            normals.append(np.array([-dx[1], dx[0]]) / np.linalg.norm(dx))
        midpts = np.asarray(midpts)
        normals = np.asarray(normals)

        # Number of particles per facet proportional to q_heater * length
        lengths = np.linalg.norm(midpts[1:] - midpts[:-1], axis=1, dtype=np.float64)
        if len(lengths) == 0:
            lengths = np.array([np.linalg.norm(self.msh.geometry.x[self.msh.topology.connectivity(1, 0).links(indices[0])[1]] -
                                             self.msh.geometry.x[self.msh.topology.connectivity(1, 0).links(indices[0])[0]])])
        lengths = np.repeat(lengths, repeats=1)
        power_local = (q_heater * lengths).sum()
        np_emit = int(power_local / omega)
        if np_emit == 0:
            return

        for i in range(min(np_emit, self.n_local)):
            f = random.randrange(len(midpts))
            self._x[i] = midpts[f]
            # Cosine law for hemi‑isotropic emission into domain
            phi = math.acos(math.sqrt(random.random()))
            theta = 2.0 * math.pi * random.random()
            dir_vec = np.array([math.sin(phi) * math.cos(theta), math.sin(phi) * math.sin(theta)])
            # Ensure inward pointing
            if np.dot(dir_vec, normals[f]) < 0:
                dir_vec *= -1.0
            self._dir[i] = dir_vec
            self._alive[i] = 1
            self._weight[i] = 1

    def _advect(self, dt: float):
        """Ballistic flight segments."""
        active = self._alive == 1
        if not np.any(active):
            return
        x = self._x[active]
        d = self._dir[active]
        mats = self._materials(x)
        v = np.where(mats == 0, self.mat_A.velocity, self.mat_B.velocity)
        self._x[active] += dt * v[:, None] * d

        # Boundary handling (2‑D rectangle assumed)
        coords = self._x[active]
        # Bottom (isothermal)
        bottom = coords[:, 1] < 0.0
        idx_bottom = np.where(bottom)[0]
        if idx_bottom.size:
            self._record_absorption(active.nonzero()[0][idx_bottom])
        # Top
        top = coords[:, 1] > self.msh.geometry.x[:, 1].max()
        idx_top = np.where(top)[0]
        self._dir[active][idx_top][:, 1] *= -1.0  # diffuse reflect (flip y)
        # Left & right
        left = coords[:, 0] < 0.0
        right = coords[:, 0] > self.msh.geometry.x[:, 0].max()
        idx_lr = np.logical_or(left, right)
        if np.any(idx_lr):
            self._dir[active][idx_lr][:, 0] *= -1.0

    def _scatter(self, dt: float):
        """Isotropic scattering with probability 1‑exp(‑dt/τ)."""
        active = self._alive == 1
        if not np.any(active):
            return
        mats = self._materials(self._x[active])
        tau = np.where(mats == 0, self.mat_A.relaxation_time, self.mat_B.relaxation_time)
        p_scat = 1.0 - np.exp(-dt / tau)
        rand = np.random.random(size=tau.size)
        coll = rand < p_scat
        if np.any(coll):
            idx = np.where(active)[0][coll]
            # Isotropic re‑direction
            phi = 2.0 * math.pi * np.random.random(len(idx))
            self._dir[idx] = np.column_stack((np.cos(phi), np.sin(phi)))

    def _record_absorption(self, global_indices: np.ndarray):
        """Deposit energy carried by particles hitting the isothermal boundary."""
        # For simplicity, kill particles
        self._alive[global_indices] = 0
        # Energy deposition not counted for steady‑state deviational scheme

    # ------------------------------------------------------------------
    # Mesh helpers
    # ------------------------------------------------------------------

    def _materials(self, pts: np.ndarray) -> np.ndarray:
        """Return 0 for material A, 1 for B for each query point."""
        cells = geometry.compute_collisions_points(self.msh, pts, self.bounding_boxes)[1]
        gamma_vals = self.gamma.x.array[cells]
        return (gamma_vals > 0.5).astype(np.int8)

    # ------------------------------------------------------------------
    # Reduction to Finite‑Element fields
    # ------------------------------------------------------------------

    def _reduce_to_functions(self):
        # Gather energy and flux tallies (currently empty)
        energy_tot = self.comm.allreduce(self._energy, op=MPI.SUM)
        flux_tot = self.comm.allreduce(self._flux, op=MPI.SUM)

        V_T = Function(self.msh, ("DG", 0))
        V_q = Function(self.msh, ("DG", 0), shape=(self.msh.geometry.dim,))
        c_field = np.where(self.gamma.x.array < 0.5, self.mat_A.c, self.mat_B.c)
        vol = geometry.cell_volume(self.msh)
        V_T.x.array[:] = energy_tot / (c_field * vol)
        V_q.x.array[:] = flux_tot.ravel()
        return V_T, V_q
