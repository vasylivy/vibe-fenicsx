#!/usr/bin/env python3
"""Rayleigh-Darcy thermal convection in a laterally periodic porous layer.

Dimensionless Darcy-Oberbeck-Boussinesq flow on Omega = [0,L] x [0,1]:

    u = -grad p + Ra T z_hat,    div u = 0,    dT/dt + u.grad T = lap T

Pressure is eliminated with a streamfunction psi (u = psi_z, w = -psi_x), so
incompressibility is automatic and the curl of Darcy's law gives a Poisson
problem  lap psi = -Ra dT/dx  (psi = 0 on the impermeable top/bottom walls,
periodic in x).  Each backward-Euler step is two scalar solves on a shared P1
space carrying one periodic MultiPointConstraint (dolfinx_mpc):

  1. streamfunction:  lap psi = -Ra dT/dx                     (CG + AMG, SPD)
  2. temperature:     (T-Tn)/dt + u.grad T - lap T = 0        (GMRES + AMG)

with u frozen at the latest psi (advection linearized).

Run (MPI parallel is the default for the showcase):

    ./run_porous.sh                                   # Ra=4000 coarse showcase
    ./run_porous.sh 2000 96                           # custom Ra / resolution
    mpirun -np 4 python examples/porous_convection.py # equivalent, default knobs

A live temperature plot refreshes on rank 0 as the run advances: a window when a
display is available, otherwise a results/porous/live.png snapshot that is
rewritten in place each refresh (./run_porous.sh opens it in VS Code, where the
preview reloads on disk change).

Output (all under results/porous/, override with --outdir or POROUS_OUTDIR):
    frame_<step:05d>.npz   points (N,2), T (N,), u (N,2), t, Ra, L
    live.png               latest temperature field (headless live view)

The frame .npz layout matches scripts/porous_convection/plot_porous.py.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx_mpc
from dolfinx import fem, mesh

# Onset of convection (Horton-Rogers-Lapwood): Ra_c = 4 pi^2.
RA_C = 4.0 * np.pi**2

# Krylov + algebraic-multigrid options.  The streamfunction operator is a
# constant SPD Laplacian (CG + AMG); the temperature operator is a nonsymmetric
# advection-diffusion (GMRES + AMG).  Tight rtol per the spec.
_PSI_OPTS = {
    "ksp_type": "cg",
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
    "ksp_rtol": 1.0e-9,
    "ksp_atol": 1.0e-14,
}
_T_OPTS = {
    "ksp_type": "gmres",
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
    "ksp_gmres_restart": 50,
    "ksp_rtol": 1.0e-9,
    "ksp_atol": 1.0e-14,
}


@dataclass
class Params:
    Ra: float = 4000.0
    N: int = 64             # vertical resolution; horizontal is round(L*N).
                            # Coarse on purpose so the showcase runs in < 1 min;
                            # raise it for a finer, quantitatively converged run.
    L: float = 3.0          # several critical wavelengths (lambda_c = 2) wide
    t_end: float = 0.05     # captures spin-up + a short developed-plume tail
    cfl: float = 1.0        # advection is implicit, so C ~ 1 is stable
    dt_max: float = 2.0e-3  # cap so the slow conductive spin-up stays sampled
    dt_init: float = 1.0e-4
    n_frames: int = 120
    eps: float = 0.05       # perturbation amplitude
    n_modes: int = 6
    seed: int = 1234
    write_frames: bool = True
    outdir: str = "results/porous"
    live: bool = True       # refresh a temperature plot on rank 0 during the run
    live_every: int = 20    # ... every this many steps


def perturbation_coeffs(p: Params):
    """Fixed, seeded multi-mode coefficients so runs are reproducible."""
    rng = np.random.default_rng(p.seed)
    amps = rng.uniform(0.5, 1.0, size=p.n_modes)
    amps /= np.linalg.norm(amps)            # normalize so eps controls amplitude
    phases = rng.uniform(0.0, 2.0 * np.pi, size=p.n_modes)
    return amps, phases


def initial_temperature(x, p: Params, amps, phases):
    """Conduction profile T = 1 - z plus a small periodic multi-mode seed."""
    T = 1.0 - x[1]
    pert = np.zeros_like(x[0])
    for n in range(1, p.n_modes + 1):
        a, phi = amps[n - 1], phases[n - 1]
        pert += a * np.cos(2.0 * np.pi * n * x[0] / p.L + phi)
    T = T + p.eps * pert * np.sin(np.pi * x[1])
    return T


class PorousConvection:
    """Coupled streamfunction-temperature solver on a periodic porous layer."""

    def __init__(self, p: Params, comm=None):
        self.p = p
        self.comm = comm or MPI.COMM_WORLD
        nx = max(1, round(p.L * p.N))
        self.mesh = mesh.create_rectangle(
            self.comm,
            [np.array([0.0, 0.0]), np.array([p.L, 1.0])],
            [nx, p.N],
            cell_type=mesh.CellType.triangle,
        )
        # One shared scalar P1 space for both psi and T.
        self.V = fem.functionspace(self.mesh, ("Lagrange", 1))
        # Vector P1 space for vertex-sampled velocity output / CFL.
        self.W = fem.functionspace(self.mesh, ("Lagrange", 1, (2,)))

        self.psi = fem.Function(self.V, name="psi")
        self.T = fem.Function(self.V, name="T")
        self.T_n = fem.Function(self.V, name="T_n")
        self.u_vec = fem.Function(self.W, name="u")

        self.dt = fem.Constant(self.mesh, PETSc.ScalarType(p.dt_init))
        self.Ra = fem.Constant(self.mesh, PETSc.ScalarType(p.Ra))

        self._build_bcs()
        self._build_mpc()
        self._build_problems()

        self._u_expr = fem.Expression(
            ufl.as_vector([self.psi.dx(1), -self.psi.dx(0)]),
            self.W.element.interpolation_points,
        )
        # Local cell size for the CFL limit.
        self.h_min = self._min_cell_size()

    # ---- setup -----------------------------------------------------------
    def _build_bcs(self):
        def bottom(x):
            return np.isclose(x[1], 0.0)

        def top(x):
            return np.isclose(x[1], 1.0)

        dofs_b = fem.locate_dofs_geometrical(self.V, bottom)
        dofs_t = fem.locate_dofs_geometrical(self.V, top)
        zero = PETSc.ScalarType(0.0)
        one = PETSc.ScalarType(1.0)
        # Streamfunction: psi = 0 on both walls.
        self.bcs_psi = [
            fem.dirichletbc(zero, dofs_b, self.V),
            fem.dirichletbc(zero, dofs_t, self.V),
        ]
        # Temperature: T = 1 (hot, bottom), T = 0 (cold, top).
        self.bcs_T = [
            fem.dirichletbc(one, dofs_b, self.V),
            fem.dirichletbc(zero, dofs_t, self.V),
        ]

    def _build_mpc(self):
        L = self.p.L

        def on_right(x):
            return np.isclose(x[0], L)

        def map_r2l(x):
            out = x.copy()
            out[0] = x[0] - L
            return out

        # One constraint on the shared space; the top/bottom Dirichlet set is
        # the same for psi and T, so passing those dofs keeps the x=L corners
        # Dirichlet rather than turning them into periodic slaves.
        self.mpc = dolfinx_mpc.MultiPointConstraint(self.V)
        self.mpc.create_periodic_constraint_geometrical(
            self.V, on_right, map_r2l, self.bcs_psi
        )
        self.mpc.finalize()

    def _build_problems(self):
        p_tr, q = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)

        # Streamfunction Poisson:  int grad psi . grad q = int Ra dT/dx q
        a_psi = ufl.inner(ufl.grad(p_tr), ufl.grad(q)) * ufl.dx
        L_psi = self.Ra * self.T.dx(0) * q * ufl.dx
        self.prob_psi = dolfinx_mpc.LinearProblem(
            a_psi, L_psi, self.mpc, bcs=self.bcs_psi,
            petsc_options=_PSI_OPTS,
            petsc_options_prefix="porous_psi_",
        )

        # Temperature advection-diffusion, backward Euler, u frozen at psi.
        u = ufl.as_vector([self.psi.dx(1), -self.psi.dx(0)])
        T_tr = ufl.TrialFunction(self.V)
        a_T = (
            (T_tr / self.dt) * q * ufl.dx
            + ufl.dot(u, ufl.grad(T_tr)) * q * ufl.dx
            + ufl.dot(ufl.grad(T_tr), ufl.grad(q)) * ufl.dx
        )
        L_T = (self.T_n / self.dt) * q * ufl.dx
        self.prob_T = dolfinx_mpc.LinearProblem(
            a_T, L_T, self.mpc, bcs=self.bcs_T,
            petsc_options=_T_OPTS,
            petsc_options_prefix="porous_T_",
        )

    def _min_cell_size(self):
        tdim = self.mesh.topology.dim
        self.mesh.topology.create_connectivity(tdim, tdim)
        ncells = self.mesh.topology.index_map(tdim).size_local
        if ncells == 0:
            local = np.inf
        else:
            local = float(self.mesh.h(tdim, np.arange(ncells, dtype=np.int32)).min())
        return self.comm.allreduce(local, op=MPI.MIN)

    # ---- helpers ---------------------------------------------------------
    @property
    def _n_local(self):
        return self.V.dofmap.index_map.size_local

    def _copy_back(self, src: fem.Function, dst: fem.Function):
        """Copy owned dofs from an mpc-space solution into a clean V function.

        dolfinx_mpc solves into a function on mpc.function_space whose ghost
        layout differs from V; copy the owned slice and re-scatter ghosts.
        """
        n = self._n_local
        dst.x.array[:n] = src.x.array[:n]
        dst.x.scatter_forward()

    # ---- stages ----------------------------------------------------------
    def set_initial_condition(self):
        amps, phases = perturbation_coeffs(self.p)
        self.T.interpolate(lambda x: initial_temperature(x, self.p, amps, phases))
        self.T.x.scatter_forward()
        self.T_n.x.array[:] = self.T.x.array
        self.T_n.x.scatter_forward()

    def solve_streamfunction(self):
        sol = self.prob_psi.solve()
        self._copy_back(sol, self.psi)
        return self.psi

    def solve_temperature(self):
        sol = self.prob_T.solve()
        self._copy_back(sol, self.T)
        return self.T

    def update_velocity(self):
        self.u_vec.interpolate(self._u_expr)
        self.u_vec.x.scatter_forward()

    def velocity_max(self):
        n = self.W.dofmap.index_map.size_local * self.W.dofmap.index_map_bs
        arr = self.u_vec.x.array[:n].reshape(-1, 2)
        local = float(np.hypot(arr[:, 0], arr[:, 1]).max()) if arr.size else 0.0
        return self.comm.allreduce(local, op=MPI.MAX)

    def pick_dt(self, t):
        umax = self.velocity_max()
        if umax > 1.0e-12:
            dt = self.p.cfl * self.h_min / umax
        else:
            dt = self.p.dt_max
        dt = min(dt, self.p.dt_max)
        dt = min(dt, self.p.t_end - t)        # land exactly on t_end
        return max(dt, 1.0e-12)

    # ---- frame output ----------------------------------------------------
    def _gather_owned(self, local):
        """Gather per-rank owned arrays to rank 0 and concatenate (axis 0)."""
        chunks = self.comm.gather(np.ascontiguousarray(local), root=0)
        if self.comm.rank == 0:
            return np.concatenate(chunks, axis=0)
        return None

    def write_frame(self, step, t):
        n = self._n_local
        coords = self.V.tabulate_dof_coordinates()[:n, :2]
        Tloc = self.T.x.array[:n]
        nv = self.W.dofmap.index_map.size_local * self.W.dofmap.index_map_bs
        uloc = self.u_vec.x.array[:nv].reshape(-1, 2)

        pts = self._gather_owned(coords)
        Tg = self._gather_owned(Tloc)
        ug = self._gather_owned(uloc)
        if self.comm.rank == 0:
            os.makedirs(self.p.outdir, exist_ok=True)
            path = os.path.join(self.p.outdir, f"frame_{step:05d}.npz")
            np.savez(path, points=pts, T=Tg, u=ug,
                     t=float(t), Ra=float(self.p.Ra), L=float(self.p.L))

    # ---- live view -------------------------------------------------------
    def _ensure_live_fig(self):
        """Create the rank-0 figure on first use; pick window vs headless PNG."""
        if getattr(self, "_live_fig", None) is not None:
            return
        import matplotlib
        # No DISPLAY (the usual container case) -> Agg + a refreshed live.png.
        self._live_headless = not os.environ.get("DISPLAY")
        if self._live_headless:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not self._live_headless:
            plt.ion()
        self._plt = plt
        self._live_fig, self._live_ax = plt.subplots(figsize=(7.0, 3.2))
        self._live_cbar = None

    def _live_update(self, t):
        """Refresh the temperature view. Collective gather; only rank 0 draws.

        The gather must run on every rank (see velocity_max); guard only the
        drawing so multi-rank runs do not deadlock.
        """
        n = self._n_local
        coords = self.V.tabulate_dof_coordinates()[:n, :2]
        pts = self._gather_owned(coords)
        Tg = self._gather_owned(self.T.x.array[:n])
        if self.comm.rank != 0:
            return
        self._ensure_live_fig()
        ax = self._live_ax
        ax.clear()
        tcf = ax.tricontourf(pts[:, 0], pts[:, 1], Tg,
                             levels=np.linspace(0.0, 1.0, 21), cmap="RdBu_r")
        if self._live_cbar is None:
            self._live_cbar = self._live_fig.colorbar(tcf, ax=ax, label="T")
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_title(f"Ra={self.p.Ra:g}   t={t:.4f}")
        if self._live_headless:
            os.makedirs(self.p.outdir, exist_ok=True)
            self._live_fig.savefig(os.path.join(self.p.outdir, "live.png"), dpi=110)
        else:
            self._live_fig.canvas.draw_idle()
            self._plt.pause(0.001)

    # ---- driver ----------------------------------------------------------
    def run(self, verbose=True):
        p = self.p
        self.set_initial_condition()
        self.solve_streamfunction()
        self.update_velocity()

        t = 0.0
        step = 0
        frame = 0
        frame_dt = p.t_end / p.n_frames
        next_frame_t = 0.0

        if p.write_frames:
            self.write_frame(frame, t)
            frame += 1
            next_frame_t = frame * frame_dt

        while t < p.t_end - 1.0e-12:
            dt = self.pick_dt(t)
            self.dt.value = dt

            # advance temperature with u frozen at current psi, then refresh psi
            self.solve_temperature()
            self.solve_streamfunction()
            self.update_velocity()

            t += dt
            step += 1
            self.T_n.x.array[:] = self.T.x.array
            self.T_n.x.scatter_forward()

            if p.write_frames and (t >= next_frame_t - 1.0e-12 or t >= p.t_end - 1.0e-12):
                self.write_frame(frame, t)
                frame += 1
                next_frame_t = frame * frame_dt

            if p.live and step % p.live_every == 0:
                self._live_update(t)

            if verbose and step % 50 == 0:
                # velocity_max() is collective (allreduce) -> all ranks must call
                # it; guard only the print, never the reduction.
                umax = self.velocity_max()
                if self.comm.rank == 0:
                    print(f"  step {step:5d}  t={t:.5f}  dt={dt:.2e}  "
                          f"umax={umax:.1f}", flush=True)

        if p.live:
            self._live_update(t)

        if verbose and self.comm.rank == 0:
            print(f"  done: {step} steps, {frame} frames", flush=True)
        return {"steps": step, "frames": frame}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _params_from_args(args):
    return Params(
        Ra=args.ra, N=args.N, L=args.L, t_end=args.t_end,
        cfl=args.cfl, n_frames=args.frames, seed=args.seed,
        write_frames=not args.no_frames, outdir=args.outdir,
        live=not args.no_live, live_every=args.live_every,
    )


def main(argv=None):
    env = os.environ.get
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ra", type=float, default=float(env("POROUS_RA", 4000.0)))
    ap.add_argument("--N", type=int, default=int(env("POROUS_N", 64)))
    ap.add_argument("--L", type=float, default=float(env("POROUS_L", 3.0)))
    ap.add_argument("--t-end", dest="t_end", type=float,
                    default=float(env("POROUS_TEND", 0.05)))
    ap.add_argument("--cfl", type=float, default=float(env("POROUS_CFL", 1.0)))
    ap.add_argument("--frames", type=int, default=int(env("POROUS_FRAMES", 120)))
    ap.add_argument("--seed", type=int, default=int(env("POROUS_SEED", 1234)))
    ap.add_argument("--outdir", default=env("POROUS_OUTDIR", "results/porous"))
    ap.add_argument("--no-frames", action="store_true")
    ap.add_argument("--no-live", action="store_true",
                    help="disable the live rank-0 temperature plot")
    ap.add_argument("--live-every", type=int,
                    default=int(env("POROUS_LIVE_EVERY", 20)),
                    help="refresh the live plot every this many steps")
    args = ap.parse_args(argv)

    comm = MPI.COMM_WORLD
    p = _params_from_args(args)
    if comm.rank == 0:
        regime = "convecting" if p.Ra > RA_C else "sub-critical (conduction)"
        print(f"Rayleigh-Darcy porous convection: Ra={p.Ra:g} ({regime}), "
              f"L={p.L}, N={p.N}, t_end={p.t_end}", flush=True)
    PorousConvection(p, comm).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
