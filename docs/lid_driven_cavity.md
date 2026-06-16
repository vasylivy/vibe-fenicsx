# The lid-driven cavity and its Ghia validation

Companion notes for [Example 2](../prompts/lid_driven_cavity.md). This is the
maths the solver discretizes and how the benchmark validates it.

## Strong form (incompressible Navier–Stokes)

On the unit square $`\Omega = (0,1)^2`$, with unit density, solve for the velocity
$`\mathbf{u}(x,t)`$ and pressure $`p(x,t)`$:

```math
\frac{\partial \mathbf{u}}{\partial t}
+ (\mathbf{u}\cdot\nabla)\mathbf{u}
= -\nabla p + \nu\,\nabla^2 \mathbf{u},
\qquad
\nabla\cdot\mathbf{u} = 0 .
```

The momentum equation balances inertia and convection against the pressure
gradient and viscous diffusion; the second equation enforces incompressibility.
The kinematic viscosity is $`\nu = 1/\mathrm{Re}`$.

The **Reynolds number** $`\mathrm{Re} = UL/\nu`$ compares convection to diffusion.
With lid speed $`U=1`$ and cavity side $`L=1`$, $`\mathrm{Re}=1000`$ means
$`\nu = 10^{-3}`$ — convection-dominated, with thin wall boundary layers and
secondary corner vortices.

## Boundary and initial conditions

- **Lid** ($`y=1`$): $`\mathbf{u} = (1, 0)`$ — the moving wall that drives the flow.
- **Other three walls**: no-slip, $`\mathbf{u} = (0,0)`$.
- **Initial**: rest, $`\mathbf{u} = \mathbf{0}`$.
- With velocity prescribed on the *entire* boundary, pressure is determined only
  up to an additive constant; we **pin one pressure dof** (or attach the constant
  nullspace) to make it unique.

## IPCS time splitting

Solving the coupled velocity–pressure system monolithically each step is
expensive, so we use **incremental pressure-correction splitting (IPCS)**. With
step $`\Delta t`$ and previous fields $`\mathbf{u}^n, p^n`$, each step is three linear
solves.

**Step 1 — tentative velocity** $`\mathbf{u}^{\ast}`$ (momentum with the old
pressure; convection semi-implicit, diffusion Crank–Nicolson):

```math
\frac{\mathbf{u}^{\ast} - \mathbf{u}^n}{\Delta t}
+ (\mathbf{u}^n \cdot \nabla)\mathbf{u}^{\ast}
- \nu\,\nabla^2\!\big(\tfrac{1}{2}(\mathbf{u}^n+\mathbf{u}^{\ast})\big)
+ \nabla p^n = \mathbf{0},
\qquad \mathbf{u}^{\ast} = \text{BC on } \partial\Omega .
```

The convection is **semi-implicit**: the advecting velocity is the known
$`\mathbf{u}^n`$ but the advected velocity is the unknown $`\mathbf{u}^{\ast}`$, which
keeps the march stable at $`\mathrm{Re}=1000`$ where a fully explicit
$`(\mathbf{u}^n\cdot\nabla)\mathbf{u}^n`$ would not be.

Lagging the advecting velocity makes the march first-order in time overall, so the
Crank–Nicolson diffusion is a cheap (it rides the operator the convection already
reassembles each step), low-dissipation default rather than a bid for second-order
accuracy. Either way it does not affect the validated result: the run is integrated
to steady state, where the time-derivative drops out and the converged field is
independent of the stepping scheme.

**Step 2 — pressure correction.** Requiring $`\nabla\cdot\mathbf{u}^{n+1}=0`$ with
$`\mathbf{u}^{n+1} = \mathbf{u}^{\ast} - \Delta t\,\nabla(p^{n+1}-p^n)`$ gives a
Poisson problem for the pressure increment:

```math
\nabla^2\big(p^{n+1}-p^n\big) = \frac{1}{\Delta t}\,\nabla\cdot\mathbf{u}^{\ast} .
```

**Step 3 — velocity correction.** Project the tentative velocity onto the
divergence-free space:

```math
\mathbf{u}^{n+1} = \mathbf{u}^{\ast} - \Delta t\,\nabla\big(p^{n+1}-p^n\big).
```

## Spatial discretization

**Taylor–Hood** elements: continuous **$`P_2`$ velocity / $`P_1`$ pressure** on
triangles. The velocity space being one order richer than the pressure space
satisfies the inf–sup (LBB) condition, which rules out spurious checkerboard
pressure modes that equal-order pairs suffer from.

## From rest to steady state

Starting from rest, the lid's shear spins up a **primary recirculating vortex**;
the field evolves until $`\lVert \mathbf{u}^{n+1}-\mathbf{u}^n\rVert/\Delta t`$ drops
below a tolerance (steady). Although the end state is steady, the solver dumps the
transient as frames so the spin-up can be animated. At $`\mathrm{Re}=1000`$ the
steady flow also develops **secondary counter-rotating vortices** in the bottom
corners.

## Validation: Ghia et al. (1982)

The canonical benchmark is Ghia, Ghia & Shin, *J. Comput. Phys.* **48**:387–411
(1982), who tabulated centerline velocities from a coupled-multigrid solver (a
$`129^2`$ uniform grid at $`\mathrm{Re}=1000`$; finer grids for higher $`\mathrm{Re}`$).
We compare the steady profiles along the two centerlines:

- $`u_x`$ along the **vertical** centerline $`x=0.5`$, as a function of $`y`$;
- $`u_y`$ along the **horizontal** centerline $`y=0.5`$, as a function of $`x`$;

against the checked-in benchmarks `gold/ghia_re100.csv`, `gold/ghia_re400.csv`
and `gold/ghia_re1000.csv`. Tracking those points to within a few percent — at
$`\mathrm{Re}=1000`$ including the $`u`$-profile's minimum near $`y\approx0.17`$ and the
$`v`$-profile's deep trough near $`x\approx0.91`$ — validates the solver
(`gold/cavity_ghia_re1000.png`). The match is tightest at $`\mathrm{Re}=100`$ and
loosens as $`\mathrm{Re}`$ rises and the wall boundary layers thin, so a fixed mesh
needs more resolution at higher $`\mathrm{Re}`$.
