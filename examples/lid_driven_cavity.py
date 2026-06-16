#!/usr/bin/env python3
"""Transient lid-driven cavity by IPCS pressure-correction splitting.

Incompressible Navier-Stokes on the unit square Omega = [0,1]^2, unit density:

    du/dt + (u.grad) u = -grad p + nu lap u,   div u = 0,   nu = 1/Re

The top lid (y=1) moves at (1,0); no-slip on the other three walls; the flow
starts from rest and spins up a recirculating vortex.  Each step is three linear
solves (incremental pressure-correction splitting, IPCS) on Taylor-Hood
P2 velocity / P1 pressure:

  1. tentative velocity u*   (semi-implicit convection, implicit diffusion; BiCGStab)
  2. pressure correction     lap p^{n+1} from div u*           (CG + AMG, SPD)
  3. velocity correction     u^{n+1} = u* - dt grad(p^{n+1}-p^n)  (CG + Jacobi)

Convection is semi-implicit (advecting velocity u_n, advected velocity the
unknown u*), which stays stable at high Re where an explicit term would blow
up.  That puts u_n inside the step-1 operator, so it is reassembled each step
while the pressure-Poisson and mass operators are assembled once.  The step is
fixed (the operator carries the mass term M/dt), set from the CFL limit at the
lid speed.  Pressure is fixed up to a constant, so one pressure dof is pinned.

Run (MPI parallel is the default for the showcase):

    ./run_cavity.sh                                    # Re=1000 showcase
    ./run_cavity.sh 400 48                             # custom Re / resolution
    mpirun -np 4 python examples/lid_driven_cavity.py  # equivalent, default knobs

A live velocity-magnitude plot refreshes on rank 0 as the run advances: a window
when a display is available, otherwise a results/cavity/live.png snapshot that is
rewritten in place each refresh (./run_cavity.sh opens it in VS Code, where the
preview reloads on disk change).

Output (all under results/cavity/, override with --outdir or CAVITY_OUTDIR):
    frame_<step:05d>.npz   points (N,2), u (N,2), p (N,), t
    live.png               latest velocity field (headless live view)

The frame .npz layout matches scripts/lid_driven_cavity/plot_cavity.py.  The
steady centerline extraction and the Ghia validation sweep are *not* done here —
they are the analyst activity (see prompts/lid_driven_cavity.md).
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, mesh
from dolfinx.fem.petsc import (
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    create_vector,
    set_bc,
)

# Krylov + preconditioner options.  Step 1 (tentative velocity) carries a
# semi-implicit convection term, so its operator is non-symmetric -> BiCGStab.
# Step 2 is a pressure Poisson solve (CG + AMG); step 3 is a mass-matrix
# projection, SPD (CG + Jacobi).
_MOM_OPTS = {"ksp_type": "bcgs", "pc_type": "jacobi",
             "ksp_rtol": 1.0e-8, "ksp_atol": 1.0e-12}
_P_OPTS = {"ksp_type": "cg", "pc_type": "hypre", "pc_hypre_type": "boomeramg",
           "ksp_rtol": 1.0e-8, "ksp_atol": 1.0e-12}
_VEL_OPTS = {"ksp_type": "cg", "pc_type": "jacobi",
             "ksp_rtol": 1.0e-8, "ksp_atol": 1.0e-12}


@dataclass
class Params:
    Re: float = 1000.0
    N: int = 64             # cells per side; P2 velocity -> ~2N nodes per side.
                            # Coarse-ish so the showcase runs quickly; raise it
                            # for a finer, more converged steady state.
    t_end: float = 60.0     # generous cap; the run stops earlier at steady state
    steady_tol: float = 2.0e-4   # stop when ||u^{n+1}-u^n|| / dt drops below this
    cfl: float = 1.0        # convection is semi-implicit (stable); C sets accuracy
    dt_max: float = 0.02    # the fixed step is min(cfl*h, dt_max)
    n_frames: int = 120
    write_frames: bool = True
    outdir: str = "results/cavity"
    live: bool = True       # refresh a velocity plot on rank 0 during the run
    live_every: int = 25    # ... every this many steps


class LidDrivenCavity:
    """IPCS Taylor-Hood Navier-Stokes solver on the unit square."""

    def __init__(self, p: Params, comm=None):
        self.p = p
        self.comm = comm or MPI.COMM_WORLD
        self.mesh = mesh.create_unit_square(
            self.comm, p.N, p.N, cell_type=mesh.CellType.triangle
        )
        # Taylor-Hood: P2 velocity, P1 pressure.
        self.V = fem.functionspace(self.mesh, ("Lagrange", 2, (2,)))
        self.Q = fem.functionspace(self.mesh, ("Lagrange", 1))
        # P1 vector space for vertex-sampled velocity output / live view / CFL.
        self.Wout = fem.functionspace(self.mesh, ("Lagrange", 1, (2,)))

        self.u_n = fem.Function(self.V, name="u_n")   # velocity at step n
        self.u_s = fem.Function(self.V, name="u_star")  # tentative velocity
        self.u_ = fem.Function(self.V, name="u")      # velocity at step n+1
        self.p_n = fem.Function(self.Q, name="p_n")
        self.p_ = fem.Function(self.Q, name="p")
        self.u_out = fem.Function(self.Wout, name="u_out")

        # Fixed step set from the CFL limit at the lid speed (U_ref = 1).  It
        # MUST be fixed: the tentative-velocity operator A1 carries the mass
        # term M/dt and is assembled once, so dt cannot change mid-run without
        # also rebuilding A1.  The lid bounds the velocity scale throughout, so
        # one CFL-safe step serves the whole spin-up.
        self.h_min = self._min_cell_size()
        self.dt_value = min(p.cfl * self.h_min, p.dt_max)
        self.dt = fem.Constant(self.mesh, PETSc.ScalarType(self.dt_value))
        self.nu = fem.Constant(self.mesh, PETSc.ScalarType(1.0 / p.Re))

        self._build_bcs()
        self._build_forms()
        self._build_solvers()

        self._u_out_expr = fem.Expression(
            self.u_, self.Wout.element.interpolation_points
        )

    # ---- setup -----------------------------------------------------------
    def _build_bcs(self):
        def walls(x):       # left, right, bottom: no-slip
            return (
                np.isclose(x[0], 0.0)
                | np.isclose(x[0], 1.0)
                | np.isclose(x[1], 0.0)
            )

        def lid(x):         # top: moving wall
            return np.isclose(x[1], 1.0)

        V = self.V
        dofs_w = fem.locate_dofs_geometrical(V, walls)
        dofs_l = fem.locate_dofs_geometrical(V, lid)
        zero = fem.Constant(self.mesh, PETSc.ScalarType((0.0, 0.0)))
        one_x = fem.Constant(self.mesh, PETSc.ScalarType((1.0, 0.0)))
        # The lid wins at the two top corners (consistent driven-lid convention).
        self.bcu = [
            fem.dirichletbc(zero, dofs_w, V),
            fem.dirichletbc(one_x, dofs_l, V),
        ]

        # Pin one pressure dof (corner) since velocity is prescribed everywhere.
        def corner(x):
            return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)

        dof_p = fem.locate_dofs_geometrical(self.Q, corner)
        self.bcp = [fem.dirichletbc(PETSc.ScalarType(0.0), dof_p, self.Q)]

    def _build_forms(self):
        u, v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        p, q = ufl.TrialFunction(self.Q), ufl.TestFunction(self.Q)
        k, nu = self.dt, self.nu
        n = ufl.FacetNormal(self.mesh)

        def eps(w):
            return ufl.sym(ufl.nabla_grad(w))

        def sigma(w, pr):
            return 2.0 * nu * eps(w) - pr * ufl.Identity(len(w))

        U = 0.5 * (self.u_n + u)   # Crank-Nicolson diffusion
        # Step 1: tentative velocity.  Convection is semi-implicit -- the
        # advecting velocity is the known u_n but the advected velocity is the
        # unknown u -- which keeps the march unconditionally stable at high Re
        # (an explicit convection term blows up once viscosity stops damping
        # it).  This puts u_n inside the operator, so A1 is reassembled each
        # step (see step()).
        F1 = (
            ufl.dot((u - self.u_n) / k, v) * ufl.dx
            + ufl.dot(ufl.dot(self.u_n, ufl.nabla_grad(u)), v) * ufl.dx
            + ufl.inner(sigma(U, self.p_n), eps(v)) * ufl.dx
            + ufl.dot(self.p_n * n, v) * ufl.ds
            - ufl.dot(nu * ufl.nabla_grad(U) * n, v) * ufl.ds
        )
        self.a1 = fem.form(ufl.lhs(F1))
        self.L1 = fem.form(ufl.rhs(F1))

        # Step 2: pressure correction (Poisson for the new pressure).
        self.a2 = fem.form(ufl.dot(ufl.grad(p), ufl.grad(q)) * ufl.dx)
        self.L2 = fem.form(
            ufl.dot(ufl.grad(self.p_n), ufl.grad(q)) * ufl.dx
            - (1.0 / k) * ufl.div(self.u_s) * q * ufl.dx
        )

        # Step 3: velocity correction.
        self.a3 = fem.form(ufl.dot(u, v) * ufl.dx)
        self.L3 = fem.form(
            ufl.dot(self.u_s, v) * ufl.dx
            - k * ufl.dot(ufl.grad(self.p_ - self.p_n), v) * ufl.dx
        )

    def _build_solvers(self):
        # A2 (pressure Poisson) and A3 (mass) are constant -> assemble once.
        # A1 carries the semi-implicit convection, so it is reassembled in place
        # each step; the first assembly here sizes the matrix and fixes the
        # sparsity pattern.
        self.A1 = assemble_matrix(self.a1, bcs=self.bcu)
        self.A1.assemble()
        self.A2 = assemble_matrix(self.a2, bcs=self.bcp)
        self.A2.assemble()
        self.A3 = assemble_matrix(self.a3)
        self.A3.assemble()
        self.b1 = create_vector(self.V)
        self.b2 = create_vector(self.Q)
        self.b3 = create_vector(self.V)

        self.ksp1 = self._ksp(self.A1, _MOM_OPTS, "cavity_mom_")
        self.ksp2 = self._ksp(self.A2, _P_OPTS, "cavity_p_")
        self.ksp3 = self._ksp(self.A3, _VEL_OPTS, "cavity_vel3_")

    def _assemble_A1(self):
        """Reassemble the tentative-velocity operator (depends on u_n)."""
        self.A1.zeroEntries()
        assemble_matrix(self.A1, self.a1, bcs=self.bcu)
        self.A1.assemble()

    def _ksp(self, A, opts, prefix):
        ksp = PETSc.KSP().create(self.comm)
        ksp.setOperators(A)
        ksp.setOptionsPrefix(prefix)
        optdb = PETSc.Options()
        optdb.prefixPush(prefix)
        for key, val in opts.items():
            optdb[key] = val
        optdb.prefixPop()
        ksp.setFromOptions()
        return ksp

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
    def update_velocity_out(self):
        self.u_out.interpolate(self._u_out_expr)
        self.u_out.x.scatter_forward()

    def steady_residual(self, dt):
        """||u^{n+1} - u^n||_L2 / dt, reduced across ranks (collective)."""
        diff = self.u_.x.array - self.u_n.x.array
        form = fem.form(ufl.dot(self.u_ - self.u_n, self.u_ - self.u_n) * ufl.dx)
        local = fem.assemble_scalar(form)
        l2 = np.sqrt(self.comm.allreduce(local, op=MPI.SUM))
        return l2 / dt

    # ---- IPCS step -------------------------------------------------------
    def step(self):
        # Step 1: tentative velocity.  Reassemble A1 (it depends on u_n).
        self._assemble_A1()
        with self.b1.localForm() as loc:
            loc.set(0.0)
        assemble_vector(self.b1, self.L1)
        apply_lifting(self.b1, [self.a1], [self.bcu])
        self.b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                            mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b1, self.bcu)
        self.ksp1.solve(self.b1, self.u_s.x.petsc_vec)
        self.u_s.x.scatter_forward()

        # Step 2: pressure correction.
        with self.b2.localForm() as loc:
            loc.set(0.0)
        assemble_vector(self.b2, self.L2)
        apply_lifting(self.b2, [self.a2], [self.bcp])
        self.b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                            mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b2, self.bcp)
        self.ksp2.solve(self.b2, self.p_.x.petsc_vec)
        self.p_.x.scatter_forward()

        # Step 3: velocity correction.
        with self.b3.localForm() as loc:
            loc.set(0.0)
        assemble_vector(self.b3, self.L3)
        self.b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                            mode=PETSc.ScatterMode.REVERSE)
        self.ksp3.solve(self.b3, self.u_.x.petsc_vec)
        self.u_.x.scatter_forward()

    # ---- frame output ----------------------------------------------------
    def _gather_owned(self, local):
        chunks = self.comm.gather(np.ascontiguousarray(local), root=0)
        if self.comm.rank == 0:
            return np.concatenate(chunks, axis=0)
        return None

    def write_frame(self, step, t):
        nq = self.Q.dofmap.index_map.size_local
        coords = self.Q.tabulate_dof_coordinates()[:nq, :2]
        ploc = self.p_n.x.array[:nq]
        nv = self.Wout.dofmap.index_map.size_local * self.Wout.dofmap.index_map_bs
        uloc = self.u_out.x.array[:nv].reshape(-1, 2)

        pts = self._gather_owned(coords)
        ug = self._gather_owned(uloc)
        pg = self._gather_owned(ploc)
        if self.comm.rank == 0:
            os.makedirs(self.p.outdir, exist_ok=True)
            path = os.path.join(self.p.outdir, f"frame_{step:05d}.npz")
            np.savez(path, points=pts, u=ug, p=pg, t=float(t))

    # ---- live view -------------------------------------------------------
    def _ensure_live_fig(self):
        if getattr(self, "_live_fig", None) is not None:
            return
        import matplotlib
        self._live_headless = not os.environ.get("DISPLAY")
        if self._live_headless:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not self._live_headless:
            plt.ion()
        self._plt = plt
        self._live_fig, self._live_ax = plt.subplots(figsize=(5.0, 4.4))
        self._live_cbar = None

    def _live_update(self, t):
        """Refresh the velocity-magnitude view. Collective gather; rank 0 draws."""
        nq = self.Q.dofmap.index_map.size_local
        coords = self.Q.tabulate_dof_coordinates()[:nq, :2]
        nv = self.Wout.dofmap.index_map.size_local * self.Wout.dofmap.index_map_bs
        uloc = self.u_out.x.array[:nv].reshape(-1, 2)
        pts = self._gather_owned(coords)
        ug = self._gather_owned(uloc)
        if self.comm.rank != 0:
            return
        self._ensure_live_fig()
        ax = self._live_ax
        ax.clear()
        speed = np.hypot(ug[:, 0], ug[:, 1])
        tcf = ax.tricontourf(pts[:, 0], pts[:, 1], speed,
                             levels=np.linspace(0.0, 1.0, 21), cmap="viridis",
                             extend="max")
        if self._live_cbar is None:
            self._live_cbar = self._live_fig.colorbar(tcf, ax=ax, label="|u|")
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Re={self.p.Re:g}   t={t:.2f}")
        if self._live_headless:
            os.makedirs(self.p.outdir, exist_ok=True)
            self._live_fig.savefig(os.path.join(self.p.outdir, "live.png"), dpi=110)
        else:
            self._live_fig.canvas.draw_idle()
            self._plt.pause(0.001)

    # ---- driver ----------------------------------------------------------
    def run(self, verbose=True):
        p = self.p
        self.update_velocity_out()

        t = 0.0
        step = 0
        frame = 0
        frame_dt = p.t_end / p.n_frames
        next_frame_t = 0.0

        if p.write_frames:
            self.write_frame(frame, t)
            frame += 1
            next_frame_t = frame * frame_dt

        dt = self.dt_value          # fixed step (see __init__)
        residual = np.inf
        while t < p.t_end - 1.0e-12:
            self.step()
            self.update_velocity_out()

            t += dt
            step += 1
            residual = self.steady_residual(dt)

            # roll forward: u_n <- u_, p_n <- p_
            self.u_n.x.array[:] = self.u_.x.array
            self.u_n.x.scatter_forward()
            self.p_n.x.array[:] = self.p_.x.array
            self.p_n.x.scatter_forward()

            if p.write_frames and (t >= next_frame_t - 1.0e-12 or t >= p.t_end - 1.0e-12):
                self.write_frame(frame, t)
                frame += 1
                next_frame_t = frame * frame_dt

            if p.live and step % p.live_every == 0:
                self._live_update(t)

            if verbose and step % 50 == 0 and self.comm.rank == 0:
                print(f"  step {step:5d}  t={t:.3f}  dt={dt:.2e}  "
                      f"res={residual:.2e}", flush=True)

            if residual < p.steady_tol and step > 5:
                break

        if p.write_frames:    # always capture the final (steady) frame
            self.write_frame(frame, t)
            frame += 1
        if p.live:
            self._live_update(t)

        if verbose and self.comm.rank == 0:
            reached = "steady" if residual < p.steady_tol else "t_end"
            print(f"  done ({reached}): {step} steps, {frame} frames, "
                  f"final res={residual:.2e}", flush=True)
        return {"steps": step, "frames": frame, "residual": float(residual),
                "steady": bool(residual < p.steady_tol)}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _params_from_args(args):
    return Params(
        Re=args.re, N=args.N, t_end=args.t_end, steady_tol=args.steady_tol,
        cfl=args.cfl, n_frames=args.frames, write_frames=not args.no_frames,
        outdir=args.outdir, live=not args.no_live, live_every=args.live_every,
    )


def main(argv=None):
    env = os.environ.get
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--re", type=float, default=float(env("CAVITY_RE", 1000.0)))
    ap.add_argument("--N", type=int, default=int(env("CAVITY_N", 64)))
    ap.add_argument("--t-end", dest="t_end", type=float,
                    default=float(env("CAVITY_TEND", 60.0)))
    ap.add_argument("--steady-tol", type=float,
                    default=float(env("CAVITY_STEADY_TOL", 2.0e-4)))
    ap.add_argument("--cfl", type=float, default=float(env("CAVITY_CFL", 1.0)))
    ap.add_argument("--frames", type=int, default=int(env("CAVITY_FRAMES", 120)))
    ap.add_argument("--outdir", default=env("CAVITY_OUTDIR", "results/cavity"))
    ap.add_argument("--no-frames", action="store_true")
    ap.add_argument("--no-live", action="store_true",
                    help="disable the live rank-0 velocity plot")
    ap.add_argument("--live-every", type=int,
                    default=int(env("CAVITY_LIVE_EVERY", 25)),
                    help="refresh the live plot every this many steps")
    args = ap.parse_args(argv)

    comm = MPI.COMM_WORLD
    p = _params_from_args(args)
    if comm.rank == 0:
        print(f"Lid-driven cavity: Re={p.Re:g}, N={p.N} (Taylor-Hood P2/P1), "
              f"t_end={p.t_end}, steady_tol={p.steady_tol}", flush=True)
    LidDrivenCavity(p, comm).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
