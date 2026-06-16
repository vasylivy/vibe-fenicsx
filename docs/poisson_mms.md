# The Poisson problem and its MMS verification

Companion notes for [Example 1](../prompts/poisson_mms.md). This is the maths the
solver discretizes and why the convergence study verifies it.

## Strong form

On the unit cube $\Omega = (0,1)^3$ with boundary $\partial\Omega$, solve for the
scalar field $u$:

$$
-\nabla\!\cdot\!(\nabla u) \;=\; -\Delta u \;=\; f \quad\text{in }\Omega,
\qquad u = 0 \;\text{ on }\partial\Omega .
$$

This is the steady-state **heat / diffusion** equation with unit conductivity:
$u$ is temperature, $f$ a volumetric source. Homogeneous Dirichlet data fixes the
temperature to zero on all six faces.

## Manufactured solution

The **method of manufactured solutions** (MMS) turns "is my solver correct?" into
an exact-error measurement: pick a smooth $u$, substitute it into the PDE to
*derive* the source $f$, then check that the solver recovers $u$ at the expected
rate. We choose

$$
u(x,y,z) = \sin(\pi x)\,\sin(\pi y)\,\sin(\pi z).
$$

Each second derivative contributes a factor $-\pi^2$, so

$$
-\Delta u = 3\pi^2 \sin(\pi x)\sin(\pi y)\sin(\pi z) = 3\pi^2 u \;=:\; f .
$$

$u$ vanishes on every face, so the homogeneous Dirichlet condition is satisfied
*exactly* — no boundary error is introduced.

## Weak form

Multiply by a test function $v \in H^1_0(\Omega)$ and integrate by parts (the
boundary term drops because $v|_{\partial\Omega}=0$):

$$
\int_\Omega \nabla u \cdot \nabla v \,\mathrm{d}x
\;=\;
\int_\Omega f\, v \,\mathrm{d}x
\qquad \forall\, v \in H^1_0(\Omega).
$$

The discrete problem replaces $H^1_0$ by a finite-dimensional subspace $V_h$ of
**trilinear $Q_1$** functions on a uniform $n\times n\times n$ hexahedral mesh
($h = 1/n$), and seeks $u_h \in V_h$ satisfying the same identity for all
$v_h \in V_h$ (Galerkin).

## Error measures and expected rates

Two norms of the error $e = u - u_h$ on the physical domain:

$$
\|e\|_{L^2} = \Big(\!\int_\Omega e^2\,\mathrm{d}x\Big)^{1/2},
\qquad
|e|_{H^1} = \Big(\!\int_\Omega |\nabla e|^2\,\mathrm{d}x\Big)^{1/2}.
$$

For $Q_1$ elements and smooth $u$, Céa's lemma gives the energy-norm rate and an
Aubin–Nitsche duality argument lifts the $L^2$ rate by one power of $h$:

$$
|u-u_h|_{H^1} = \mathcal{O}(h), \qquad \|u-u_h\|_{L^2} = \mathcal{O}(h^2).
$$

The sweep over $n \in \{10,20,40,80\}$ estimates the observed order from
consecutive rungs, $\log_2\!\big(e_{\text{coarse}}/e_{\text{fine}}\big)$, which
should land near $1.0$ (H¹) and $2.0$ (L²). Matching those slopes — and
reproducing the checked-in `gold/` values — is the verification.

## Two details that protect the rates

- **Exact solution as an expression, not an interpolant.** $u$ is evaluated
  through `ufl.SpatialCoordinate`. Interpolating it into $V_h$ first would inject
  its own $\mathcal{O}(h^2)$ interpolation error and spoil the measured $L^2$
  rate.
- **Consistent quadrature.** A degree-3 measure (the $2\times2\times2$
  Gauss–Legendre rule on the hexahedron) integrates the forms *and* the error
  norms, so the reported numbers are the canonical $Q_1$ values.

## Visualizing the field

The solver exports the $z = 0.5$ mid-plane, where
$u(x,y,0.5) = \sin(\pi x)\sin(\pi y)$ — a smooth bump peaking at $1$ in the centre
and vanishing on the edges. `scripts/poisson_mms/plot_poisson_slice.py` contours
it alongside the pointwise error (see `gold/poisson_slice.png`).
