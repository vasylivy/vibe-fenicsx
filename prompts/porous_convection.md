# Example 3 — Rayleigh–Darcy convection in a periodic porous layer (FEniCSx)

A working solver already ships in `examples/porous_convection.py`. Like Example 2
(and unlike Example 1), you do not write it from scratch — you drive it and extend it
through two activities: run and watch the plumes form, then quantify and *validate* the
heat transport.

The maths and physics — the Darcy–Oberbeck–Boussinesq equations, the
streamfunction–temperature formulation, the onset value `Ra_c = 4π²`, and the
`Nu ≈ 0.0069·Ra` scaling — are written up in
[`docs/porous_convection.md`](../docs/porous_convection.md). Read that for background;
this prompt is about the activities.

## The shipped solver

`examples/porous_convection.py` is a segregated streamfunction–temperature solver on
a laterally periodic layer `[0,L]×[0,1]`. Each backward-Euler step does one
streamfunction Poisson solve (CG + AMG) and one advection–diffusion temperature solve
(GMRES + AMG), with the velocity frozen at the latest streamfunction. It provides:

- a shared P1 space with one periodic `MultiPointConstraint` reused for both solves;
- adaptive CFL time stepping and `results/porous/frame_<step:05d>.npz` temperature
  frames (`points`, `T`, `u`, `t`, `Ra`, `L` — the layout the plotter expects);
- a live temperature plot on rank 0: a refreshing `results/porous/live.png`,
  rewritten in place each refresh, which the launcher opens in VS Code;
- a launcher `./run_porous.sh [RA] [N] [extra args]` (defaults `Ra=4000`, `N=64`,
  tuned to finish in under a minute on 4 ranks);
- sanity tests `test_periodic`, `test_parallel_frame`, `test_conduction`.

## Activity 1 — Run and watch

Drive the shipped solver and confirm it behaves.

- Launch the showcase **through the launcher**: `./run_porous.sh` (or
  `./run_porous.sh 2000 96` for a different `Ra`/resolution). Run it *via this
  script* — do not hand-roll your own `mpirun` invocation, and do **not** pass
  `--no-live`: the live view is the point of this activity. It should finish in
  under a minute on 4 ranks and print step progress (`t`, `dt`, `umax`).
- Watch the live temperature plot as protoplumes form at the walls and merge into
  domain-spanning fingers. The launcher writes the refreshing
  `results/porous/live.png`; **open it in VS Code so it previews**:
  `code results/porous/live.png`. Tune `--live-every` as you like. (The live PNG is
  a required artifact of this activity — it is rendered for the **user** to watch
  the run evolve in VS Code, so always produce and open it rather than disabling
  it.)
- Run the sanity tests serially and under MPI:
  `python -m pytest tests/test_porous.py` and
  `mpirun -np 4 python -m pytest tests/test_porous.py`.
- Render the figures from the frames:
  `python scripts/porous_convection/plot_porous.py results/porous results/porous`
  (montage, GIF, final field). The `Nu(t)` and Nu–Ra panels fill in once Activity 2
  produces the heat-transport data.

## Activity 2 — Quantify and validate heat transport

Add the heat-transport diagnostic the baseline omits, then sweep it over `Ra` and
check it against the published correlation.

- Add the Nusselt number: `Nu(t) = 1 + (1/L)∫_Ω wT dΩ` (with `w = −ψ_x`), assembled
  with `fem.assemble_scalar` and reduced across ranks with `allreduce`. Like every
  collective, that reduction must run on every rank — never inside an `if rank == 0`
  guard, or the run deadlocks. Track `Nu` over the run and write
  `results/porous/nu_timeseries.npz` (`t`, `nu`, `Ra`, `nu_avg`, averaged over the
  statistically-steady second half) — the layout the plotter's `Nu(t)` panel expects.
- Add a `--sweep` mode that runs `Ra ∈ {500, 1000, 2000}` (frames off), time-averages
  `Nu` for each, and writes `results/porous/nu_scaling.csv` with columns `Ra,Nu`. A
  coarser `N` / short horizon is fine for a trend check.
- Compare to the published correlation `Nu ≈ 0.0069·Ra + 2.75` (Hewitt, Neufeld &
  Lister, *Phys. Rev. Lett.* 108:224503, 2012; the docs explain what the slope and
  intercept mean). There is no checked-in gold to match — `Nu` is a time-average of
  an unsteady, chaotic signal, reproducible only to a few percent — so expect your
  swept points within ≈15% of the line (e.g. ≈9.65 at `Ra=1000`).
- Plot it:
  `python scripts/porous_convection/plot_porous.py results/porous results/porous results/porous/nu_scaling.csv`
  draws your `Nu(Ra)` points against the correlation.
- Add an opt-in/slow test `test_nu_scaling` (e.g. behind `POROUS_RUN_SLOW=1`): a short
  `Ra=1000` run gives a time-averaged `Nu` within ≈15% of the correlation.
- When the activity is done, open the Nu–Ra figure so it previews in VS Code:
  `code results/porous_nu_scaling.png`.

## Process

- Plan first and clear up uncertainties with the user. The two activities are the
  milestones; develop on a coarse mesh / short horizon, then run the showcase
  resolution at the boundary.
- Pause for user review at each milestone boundary unless told to run straight through.
