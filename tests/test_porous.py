"""Tests for the Rayleigh-Darcy porous-convection solver.

Run serially and under MPI:

    pytest tests/test_porous.py
    mpirun -np 4 python -m pytest tests/test_porous.py

All tests are collective, so every rank must execute every test.
"""
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.porous_convection import (  # noqa: E402
    RA_C,
    Params,
    PorousConvection,
)


def test_periodic():
    """Streamfunction Poisson on a coarse grid is periodic across x=0 / x=L.

    With the MPC wired up, the x=L dofs are slaves of their x=0 masters, so the
    solved psi must match across the seam to round-off for every interior z.
    """
    p = Params(Ra=1000.0, N=16, L=3.0, write_frames=False)
    sim = PorousConvection(p)
    # Seed a temperature with genuine x-structure so dT/dx (the psi source) != 0.
    sim.T.interpolate(
        lambda x: 1.0 - x[1] + 0.2 * np.cos(2.0 * np.pi * x[0] / p.L) * np.sin(np.pi * x[1])
    )
    sim.T.x.scatter_forward()
    sim.solve_streamfunction()

    # Compare psi at matching (0,z) and (L,z) vertices, gathered to all ranks.
    n = sim.V.dofmap.index_map.size_local
    coords = sim.V.tabulate_dof_coordinates()[:n]
    psi = sim.psi.x.array[:n]
    comm = MPI.COMM_WORLD
    coords = np.concatenate(comm.allgather(coords), axis=0)
    psi = np.concatenate(comm.allgather(psi), axis=0)

    left = np.abs(coords[:, 0] - 0.0) < 1e-9
    right = np.abs(coords[:, 0] - p.L) < 1e-9
    # match by z coordinate
    zl, pl = coords[left, 1], psi[left]
    zr, pr = coords[right, 1], psi[right]
    ol, orr = np.argsort(zl), np.argsort(zr)
    assert len(zl) == len(zr) and len(zl) > 0
    np.testing.assert_allclose(zl[ol], zr[orr], atol=1e-12)
    np.testing.assert_allclose(pl[ol], pr[orr], atol=1e-10)


def test_parallel_frame(tmp_path):
    """A written frame has the full global vertex count and T in [0,1].

    Confirms the owned-dof gather is correct: no missing subdomains and no
    duplicated ghost dofs.
    """
    outdir = str(tmp_path / "porous") if MPI.COMM_WORLD.rank == 0 else ""
    outdir = MPI.COMM_WORLD.bcast(outdir, root=0)
    p = Params(Ra=1000.0, N=20, L=3.0, t_end=0.004, n_frames=3,
               dt_max=1e-3, outdir=outdir, live=False)
    sim = PorousConvection(p)
    n_global = sim.V.dofmap.index_map.size_global
    sim.run(verbose=False)

    if MPI.COMM_WORLD.rank == 0:
        frames = sorted(Path(outdir).glob("frame_*.npz"))
        assert frames, "no frames written"
        d = np.load(frames[-1])
        assert d["points"].shape == (n_global, 2)
        assert d["T"].shape == (n_global,)
        assert d["u"].shape == (n_global, 2)
        assert d["T"].min() >= -1e-9 and d["T"].max() <= 1.0 + 1e-9


def _max_dev_from_conduction(sim):
    """Max |T - (1 - z)| over owned dofs, reduced across ranks (collective)."""
    n = sim.V.dofmap.index_map.size_local
    x = sim.V.tabulate_dof_coordinates()[:n]
    dev = np.abs(sim.T.x.array[:n] - (1.0 - x[:, 1]))
    local = float(dev.max()) if dev.size else 0.0
    return MPI.COMM_WORLD.allreduce(local, op=MPI.MAX)


def test_conduction():
    """Below onset (Ra < 4 pi^2) the seeded perturbation decays toward the
    linear conduction profile T = 1 - z (no convection)."""
    assert 30.0 < RA_C  # sanity: 30 is sub-critical
    p = Params(Ra=30.0, N=24, L=3.0, t_end=0.05, dt_max=2e-3,
               write_frames=False, live=False)
    sim = PorousConvection(p)

    sim.set_initial_condition()
    dev0 = _max_dev_from_conduction(sim)   # the seeded perturbation amplitude
    sim.run(verbose=False)                 # re-seeds the same IC, then integrates
    dev1 = _max_dev_from_conduction(sim)   # departure after the run

    assert dev0 > 1e-3         # the perturbation really was seeded
    assert dev1 < 0.5 * dev0   # decayed clearly (sub-critical: convection is off)
