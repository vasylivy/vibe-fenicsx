"""Tests for the lid-driven cavity IPCS solver.

Run serially and under MPI:

    pytest tests/test_cavity.py
    mpirun -np 4 python -m pytest tests/test_cavity.py

All tests are collective, so every rank must execute every test.
"""
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.lid_driven_cavity import LidDrivenCavity, Params  # noqa: E402


def _umax(sim):
    """Max vertex speed over the P1 output field, reduced across ranks."""
    n = sim.Wout.dofmap.index_map.size_local * sim.Wout.dofmap.index_map_bs
    arr = sim.u_out.x.array[:n].reshape(-1, 2)
    local = float(np.hypot(arr[:, 0], arr[:, 1]).max()) if arr.size else 0.0
    return MPI.COMM_WORLD.allreduce(local, op=MPI.MAX)


def test_residual_decays_and_lid_drives_flow():
    """From rest, ||u^{n+1}-u^n||/dt decays and the lid drives an O(1) flow.

    A short Re=100 march on a coarse mesh: the steady residual must fall over
    the run (relaxing toward steady), and the lid boundary condition must show
    up as a near-unit maximum speed in the field.
    """
    p = Params(Re=100.0, N=24, write_frames=False, live=False)
    sim = LidDrivenCavity(p)
    dt = sim.dt_value
    sim.dt.value = dt

    res_hist = []
    for k in range(40):
        sim.step()
        sim.update_velocity_out()
        res_hist.append(sim.steady_residual(dt))
        sim.u_n.x.array[:] = sim.u_.x.array
        sim.u_n.x.scatter_forward()
        sim.p_n.x.array[:] = sim.p_.x.array
        sim.p_n.x.scatter_forward()

    # The transient relaxes: the late residual is well below the early one.
    assert res_hist[-1] < 0.25 * res_hist[1]
    # The lid (|u|=1) is active and develops the flow.
    assert 0.9 < _umax(sim) < 1.5


def test_parallel_frame(tmp_path):
    """A written frame has the full global vertex count and finite fields.

    Confirms the owned-dof gather is correct under MPI: no missing subdomains
    and no duplicated ghost dofs.
    """
    outdir = str(tmp_path / "cavity") if MPI.COMM_WORLD.rank == 0 else ""
    outdir = MPI.COMM_WORLD.bcast(outdir, root=0)
    p = Params(Re=100.0, N=20, t_end=0.1, n_frames=3, steady_tol=0.0,
               outdir=outdir, live=False)
    sim = LidDrivenCavity(p)
    n_global = sim.Q.dofmap.index_map.size_global   # P1 -> vertices
    sim.run(verbose=False)

    if MPI.COMM_WORLD.rank == 0:
        frames = sorted(Path(outdir).glob("frame_*.npz"))
        assert frames, "no frames written"
        d = np.load(frames[-1])
        assert d["points"].shape == (n_global, 2)
        assert d["u"].shape == (n_global, 2)
        assert d["p"].shape == (n_global,)
        assert np.isfinite(d["u"]).all() and np.isfinite(d["p"]).all()
        # lid speed sets the scale; nothing should exceed it by much
        assert np.hypot(d["u"][:, 0], d["u"][:, 1]).max() < 1.5


def test_pressure_pinned():
    """The pinned corner dof keeps the pressure finite (no nullspace drift)."""
    p = Params(Re=100.0, N=16, write_frames=False, live=False)
    sim = LidDrivenCavity(p)
    sim.dt.value = sim.dt_value
    for _ in range(10):
        sim.step()
        sim.u_n.x.array[:] = sim.u_.x.array
        sim.u_n.x.scatter_forward()
        sim.p_n.x.array[:] = sim.p_.x.array
        sim.p_n.x.scatter_forward()
    n = sim.Q.dofmap.index_map.size_local
    local = float(np.abs(sim.p_n.x.array[:n]).max()) if n else 0.0
    pmax = MPI.COMM_WORLD.allreduce(local, op=MPI.MAX)
    assert np.isfinite(pmax) and pmax < 1.0e3
