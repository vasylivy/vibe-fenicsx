# Example 2 — Lid-driven cavity: transient spin-up & Ghia validation (FEniCSx)

A working solver already ships in `examples/lid_driven_cavity.py`. Unlike
Example 1, you do not write it from scratch — you drive it and extend it through
two activities: run and watch the vortex spin up from rest, then quantify the
steady flow and *validate* it against the classic Ghia benchmark.

The maths and physics — the incompressible Navier–Stokes equations, the IPCS
pressure-correction splitting, Taylor–Hood elements, and the Ghia et al. (1982)
benchmark — are written up in
[`docs/lid_driven_cavity.md`](../docs/lid_driven_cavity.md). Read that for
background; this prompt is about the activities.

## The shipped solver

`examples/lid_driven_cavity.py` is a transient IPCS Navier–Stokes solver on the
unit square `[0,1]²`: the lid (`y=1`) moves at `(1,0)`, no-slip on the other
three walls, the flow starts from rest. It uses **Taylor–Hood P2/P1** elements
and three Krylov solves per step — tentative velocity (semi-implicit convection,
so it stays stable at `Re=1000`; **BiCGStab**), a pressure-correction Poisson
solve (**CG + AMG**), and a velocity correction (**CG + Jacobi**). One pressure
dof is pinned. It provides:

- a fixed CFL-safe time step and a steady-state stop when
  `‖uⁿ⁺¹ − uⁿ‖ / Δt` falls below `--steady-tol`;
- `results/cavity/frame_<step:05d>.npz` velocity/pressure frames sampled at the
  P1 vertices (`points`, `u`, `p`, `t` — the layout the plotter expects),
  gathered to rank 0 so they are correct under MPI;
- a live velocity-magnitude plot on rank 0: a `results/cavity/live.png` rewritten
  in place each refresh, which the launcher opens in VS Code where the preview
  reloads on disk change;
- a launcher `./run_cavity.sh [RE] [N] [extra args]` (defaults `Re=1000`,
  `N=64`, MPI on 4 ranks, reaches steady in well under a minute);
- sanity tests `test_residual_decays_and_lid_drives_flow`, `test_parallel_frame`,
  `test_pressure_pinned`.

## Activity 1 — Run and watch

Drive the shipped solver and confirm it behaves.

- Launch the showcase **through the launcher**: `./run_cavity.sh` (or
  `./run_cavity.sh 400 48` for a different `Re`/resolution). Run it *via this
  script* — do not hand-roll your own `mpirun` invocation, and do **not** pass
  `--no-live`: the live view is the point of this activity. It should reach steady
  state in well under a minute on 4 ranks and print step progress (`t`, `dt`,
  `res`).
- Watch the live plot as the lid shear spins the fluid up from rest into the
  primary recirculating vortex (and, at `Re=1000`, the secondary corner
  vortices). The launcher writes `results/cavity/live.png`, which refreshes in
  place as the run advances; **open it in VS Code so it previews**:
  `code results/cavity/live.png`. Tune `--live-every` as you like. (The live PNG is
  a required artifact of this activity — it is rendered for the **user** to watch
  the run evolve in VS Code, so always produce and open it rather than disabling
  it.)
- Run the sanity tests serially and under MPI:
  `python -m pytest tests/test_cavity.py` and
  `mpirun -np 4 python -m pytest tests/test_cavity.py`.
- Render the figures from the frames:
  `python scripts/lid_driven_cavity/plot_cavity.py results/cavity results/cavity`
  (montage, GIF, final steady field). The centerline-comparison panel fills in
  once Activity 2 produces the centerline data.
- When the activity is done, open the figures so they preview in VS Code:
  `code results/cavity_montage.png results/cavity.gif results/cavity_steady.png`.

## Activity 2 — Quantify and validate against Ghia

Add the steady centerline diagnostic the baseline omits, then sweep it over `Re`
and check it against the canonical benchmark.

- Add centerline extraction: once steady, sample `u_x` along the **vertical**
  centerline `x=0.5` (as a function of `y`) and `u_y` along the **horizontal**
  centerline `y=0.5` (as a function of `x`). Point-evaluate the FE velocity
  (bounding-box tree → colliding cells → `u.eval`) on a line of points — don't
  assume a structured dof layout. Under MPI the sampled points live on whichever
  ranks own them, so gather/`allreduce` across **every** rank — never inside an
  `if rank == 0` guard, or the run deadlocks. Write
  `results/cavity/centerlines_re<Re>.npz` with `y`, `u_vert`, `x`, `v_horiz`,
  `Re`, `N` — the layout the plotter's centerline panel expects.
- Add a `--sweep` mode that solves `Re ∈ {100, 400, 1000}` (frames off), each to
  steady, writing one `centerlines_re<Re>.npz` per `Re`. A coarser `N` is fine
  for the lower `Re`.
- Compare to **Ghia, Ghia & Shin (1982)**, *J. Comput. Phys.* 48:387–411 — the
  checked-in `gold/ghia_re100.csv`, `gold/ghia_re400.csv`, `gold/ghia_re1000.csv`
  (columns `y,u,x,v`). Expect a match within a few percent on `N=64` (P2 gives a
  `≈128²`-node velocity field); the gap widens with `Re` as the wall boundary
  layers thin and corner vortices sharpen. Cite the benchmark.
- Plot it:
  `python scripts/lid_driven_cavity/plot_cavity.py results/cavity results/cavity`
  draws your `u(y)` and `v(x)` against the Ghia points, one row per `Re`.
- Add an opt-in/slow test (e.g. behind `CAVITY_RUN_SLOW=1`): a short `Re=400`
  run gives centerline profiles within ≈15% of `gold/ghia_re400.csv`.
- When the activity is done, open the comparison so it previews in VS Code:
  `code results/cavity_centerlines.png`.

## Activity 3 — Watch it converge with resolution

Activity 2 shows one mesh lands within a few percent of Ghia. Now *watch the
discretization error shrink*: hold `Re=1000` fixed and refine the mesh.

- Add a `--converge` mode that solves `N ∈ {16, 32, 64, 128}` (frames off)
  to steady, reusing your Activity 2 extraction to write one
  `results/cavity/converge_re1000_N<N>.npz` per `N` (same keys; the `converge_`
  prefix keeps them out of the per-`Re` panel). `N=128` is the heavy run, still
  fine on 4 ranks. Keep the `allreduce` on **every** rank, as in Activity 2.
- Add a convergence plot to `scripts/lid_driven_cavity/plot_cavity.py` (reuse
  `_read_ghia`): a **single** figure, `u(y)` and `v(x)` panels, one curve per `N`
  graded light→dark over the `gold/ghia_re1000.csv` points, emitted as
  `<out_prefix>_convergence.png` when `converge_*.npz` exist. The coarse meshes
  sit off the benchmark; the curves tighten onto it as `N` grows, `N=128` nearly
  on top.
- Optional slow check (`CAVITY_RUN_SLOW=1`): the finest `N` is closer to Ghia
  (RMS over the sample points) than the coarsest.
- When the activity is done, open the convergence figure so it previews in VS
  Code: `code results/cavity_convergence.png`.

## Process

- Plan first and clear up uncertainties with the user. The two activities are
  the milestones; develop on a coarse mesh / short horizon, then run the
  showcase resolution at the boundary.
- Pause for user review at each milestone boundary unless told to run straight
  through.
