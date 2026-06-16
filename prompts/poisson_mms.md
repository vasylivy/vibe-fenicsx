# Example 1 — Poisson MMS verification & solution contour (FEniCSx)

Implement a **3D Poisson** finite-element solve in **FEniCSx / DOLFINx** (Python)
from this spec. No solver code exists yet — you write it. **Plan first, involve
the user in the planning, then build in milestones.**

The PDE is deliberately classic. The point of the exercise is the end-to-end
analyst workflow: **build** the solver, **verify** it against a checked-in gold
reference by a method-of-manufactured-solutions (MMS) convergence study, and
**visualize** the solution field.

> **Background:** the PDE, weak form, and convergence theory are written up in
> [`docs/poisson_mms.md`](../docs/poisson_mms.md).

## Environment & conventions

- DOLFINx, UFL, Basix, PETSc (`petsc4py`) and MPI (`mpi4py`) are preinstalled in
  the container. NumPy and matplotlib are available.
- Layout:
  - solver code in **`examples/`** (an importable module is fine), e.g.
    `examples/poisson_mms.py`;
  - tests in `tests/` (pytest);
  - the provided plotting CLIs in `scripts/poisson_mms/`;
  - all generated output goes to `results/` (gitignored);
  - the checked-in pass/fail data lives in `gold/` (read-only — never write there).
- **MPI:** the solve must be correct both at `np=1` (`python examples/poisson_mms.py`)
  and `np=4` (`mpirun -np 4 python examples/poisson_mms.py`). DOLFINx partitions the
  mesh automatically; reduce any integrated quantity with
  `MPI.COMM_WORLD.allreduce(..., op=MPI.SUM)`.
- Let DOLFINx assemble — do not hand-loop over dofs/elements in Python.

## Problem

`-∇·∇u = f` on the unit cube `[0,1]³`, homogeneous Dirichlet `u = 0` on all six
faces. Manufactured solution `u = sin(πx)·sin(πy)·sin(πz)`, source `f = 3π²·u`.

## Discretization — must match the gold reference exactly

- Mesh: `dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n,
  dolfinx.mesh.CellType.hexahedron)` — a uniform `n×n×n` hex grid.
- Space: `dolfinx.fem.functionspace(msh, ("Lagrange", 1))` — trilinear **Q1**.
- Quadrature: build the measure as
  `dx = ufl.Measure("dx", domain=msh, metadata={"quadrature_degree": 3})` and use
  it for the bilinear form, the load, **and both error norms**. On a hexahedron
  `quadrature_degree=3` is the 2-points-per-direction (8-point) **2×2×2
  Gauss–Legendre** rule — the integration points the reference data was computed
  with.
- BCs: locate the boundary facets on all six faces, then the dofs on them, and
  apply `dolfinx.fem.dirichletbc(0.0, dofs, V)`.
- Solver: the operator is a symmetric positive-definite Poisson matrix, so
  **always** use **CG + an algebraic-multigrid PC** (PETSc `cg` + `hypre`/`gamg`)
  — not a direct LU factorization, which does not scale with `n`. (A direct LU is
  acceptable only as a throwaway sanity check at the smallest `n`.) Use
  `rtol ≤ 1e-10` so `u_h` is the exact Galerkin solution.

## Error norms (both, on the physical domain, with the degree-3 measure)

- Represent the exact `u` as a **UFL expression of `ufl.SpatialCoordinate`** —
  do **not** interpolate it into a finite-element space (that injects
  interpolation error and breaks the gold match).
- `L²  = sqrt(∫ (u_h − u)² dx)`, `H¹-seminorm = sqrt(∫ |∇u_h − ∇u|² dx)`.
- Compute each via `assemble_scalar(form(...))`, `allreduce(SUM)`, then `sqrt`.

## Sweep & CSV output

- Resolutions `n ∈ {10, 20, 40, 80}`.
- Emit a CSV with header `n,l2_error,h1_error` to the path in env var
  `MMS_CONVERGENCE_CSV` (conventional `results/poisson_mms_q1_convergence.csv`);
  skip writing when the env var is unset.
- **Write LF-terminated rows.** Python's `csv.writer` defaults to `\r\n`; pass
  `lineterminator="\n"` so the file matches the LF gold CSV byte-for-byte and a
  plain `diff` against gold is clean.
- Plot it: `python scripts/poisson_mms/plot_mms_convergence.py <csv>
  results/poisson_mms_q1_convergence.png`. Then open the figure so it previews in
  VS Code — the user should not have to open it themselves:
  `code results/poisson_mms_q1_convergence.png`.

## Solution contour (visualize the steady-heat field)

Poisson is the steady heat equation, so `u_h` is a steady temperature field.
Export a 2D slice and contour it:

- When env var `POISSON_SLICE_NPZ` is set (serial runs only), also write the
  `z = 0.5` mid-plane of `u_h` to that path as an `.npz` with arrays:
  - `points` — `(M, 2)` float, the `(x, y)` of the mid-plane nodes,
  - `u` — `(M,)` float, `u_h` on the plane,
  - `u_exact` — `(M,)` float, `sin(πx)·sin(πy)·sin(π/2)` (optional but nice),
  - `error` — `(M,)` float, `u_h − u_exact` (optional),
  - `n` — the mesh resolution used (so the plot can be labelled).
  Pick an **even** `n` (e.g. `40`) so the `z = 0.5` plane lands on a mesh layer;
  P1 dofs sit on vertices, so the plane nodes are just the dof coordinates with
  `z ≈ 0.5`.
- Contour it: `python scripts/poisson_mms/plot_poisson_slice.py <slice.npz> [out.png]` —
  filled contour of `u_h` plus a pointwise-error panel.

## Acceptance

- Per consecutive pair, `log2(e_coarse/e_fine)`: L² rate ∈ `[1.8, 2.2]`,
  H¹ rate ∈ `[0.8, 1.2]` (theory: 2.0 and 1.0).
- `diff` the emitted CSV against `gold/poisson_mms_q1_convergence.csv`; with the
  LF fix it should be **byte-identical** (values certainly agree to 2–3
  significant figures).
- The slice contour shows the `sin·sin` bump: `u_h ≈ 1` at the plane centre,
  zero on the boundary; error `O(10⁻³)` at `n = 40`.
- Passes at `np=1` and `np=4`.

## Test suite (pytest; correct at `np=1` and `np=4`)

- `test_patch` — every linear field `u = a + b·x + c·y + d·z` prescribed as
  Dirichlet data with `f = 0` is reproduced to `1e-10` (L∞ nodal). Constant and
  linear must both pass.
- `test_mms_convergence` — runs the sweep, checks the per-pair rates, writes the
  CSV, and diffs it against gold (2–3 sig figs).
- `test_slice` *(optional)* — the slice `.npz` is written with the documented
  keys and `max(u_h)` on the plane is close to 1.

## Process

- Plan first and clear up any uncertainties with the user. Group the work into
  milestones: **(a)** skeleton + solve reproducing gold at `np=1`/`np=4`;
  **(b)** the slice export + contour figure; **(c)** tests + docs.
- Iterate with fast feedback: run only the focused test(s) for what you just
  changed; run the full suite and regenerate the figures at milestone
  boundaries, then commit.
- Regenerate `CLAUDE.md` at each milestone so a fresh session has the current
  layout, run commands, and conventions.
- Default to pausing for user review at each milestone boundary unless told to
  run straight through.
