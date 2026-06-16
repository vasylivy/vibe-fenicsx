# Rayleigh–Darcy convection and its fingering plumes

Companion notes for [Example 3](../prompts/porous_convection.md). This is the
mathematics the solver discretizes and the physics it should reproduce: thermal
convection in a fluid-saturated porous layer heated from below and cooled from
above — the **Horton–Rogers–Lapwood**, or **Rayleigh–Darcy**, problem.

## Strong form (Darcy–Oberbeck–Boussinesq)

On the laterally periodic layer $\Omega = (0,L)\times(0,1)$ we solve for the Darcy
velocity $\mathbf{u}=(u,w)$, pressure $p$, and temperature $T$. In dimensionless
form, with lengths scaled by the layer depth $H$, velocity by $\kappa/H$ (thermal
diffusivity over depth), and temperature by the top-to-bottom difference
$\Delta T$:

$$
\mathbf{u} = -\nabla p + \mathrm{Ra}\,T\,\hat{\mathbf{z}},
\qquad
\nabla\cdot\mathbf{u} = 0,
\qquad
\frac{\partial T}{\partial t} + \mathbf{u}\cdot\nabla T = \nabla^2 T .
$$

The first equation is **Darcy's law** with a **Boussinesq** buoyancy body force:
the porous medium resists flow linearly (momentum is not advected — there is no
$(\mathbf{u}\cdot\nabla)\mathbf{u}$), and warm fluid is lighter. The third
equation advects and diffuses heat. The single governing parameter is the

$$
\textbf{Rayleigh–Darcy number}\qquad
\mathrm{Ra} = \frac{g\,\beta\,\Delta T\,k\,H}{\nu\,\kappa},
$$

built from gravity $g$, thermal expansion $\beta$, permeability $k$, kinematic
viscosity $\nu$ and diffusivity $\kappa$. It measures buoyant driving against
diffusive–viscous damping.

## Boundary and initial conditions

- **Bottom** ($z=0$): hot, $T = 1$. **Top** ($z=1$): cold, $T = 0$.
- **Top and bottom are impermeable**: no normal flow, $w = 0$.
- **Lateral**: all fields are **periodic in $x$** (period $L$).
- **Initial**: the conductive profile $T = 1 - z$ plus a small multi-mode
  perturbation that seeds the instability.

## Onset of convection

The motionless **conductive base state** $T = 1-z$, $\mathbf{u}=\mathbf 0$ solves
the equations for any $\mathrm{Ra}$. Linearizing about it, a perturbation with
horizontal wavenumber $a$ and the gravest vertical structure grows at rate

$$
\sigma(a) = \frac{a^2}{a^2+\pi^2}\,\mathrm{Ra} - (a^2+\pi^2),
$$

so it is marginal ($\sigma=0$) when $\mathrm{Ra} = (a^2+\pi^2)^2/a^2$. Minimizing
over $a$ gives the classical result

$$
\boxed{\mathrm{Ra}_c = 4\pi^2 \approx 39.48}, \qquad a_c = \pi,
$$

(Horton & Rogers 1945; Lapwood 1948): below $\mathrm{Ra}_c$ disturbances decay and
heat crosses by conduction alone; above it, convection rolls grow with a preferred
wavelength $\lambda_c = 2\pi/a_c = 2$ (two layer-depths wide). Choosing the domain
width $L$ to be a few multiples of $\lambda_c$ lets several plumes coexist.

## Streamfunction–temperature formulation

In 2-D it is convenient to eliminate pressure. Incompressibility is satisfied
identically by a **streamfunction** $\psi$ with

$$
u = \frac{\partial\psi}{\partial z}, \qquad w = -\frac{\partial\psi}{\partial x}.
$$

Taking the curl of Darcy's law removes $\nabla p$ and leaves a Poisson problem
linking the streamfunction to horizontal temperature gradients — the buoyancy
source of vorticity:

$$
\nabla^2\psi = -\,\mathrm{Ra}\,\frac{\partial T}{\partial x},
\qquad \psi = 0 \ \text{on } z=0,1, \quad \psi \text{ periodic in } x .
$$

Impermeability becomes the simple Dirichlet condition $\psi=\text{const}$ (taken
$0$) on the horizontal walls — no pressure null space to fix. Each time step is
then two scalar solves: update $\psi$ from the current $T$, then advect–diffuse
$T$ with $\mathbf{u}=(\psi_z,-\psi_x)$.

## Heat transport and the fingering regime

The **Nusselt number** measures total heat flux relative to pure conduction.
Averaging the temperature equation gives the exact identity

$$
\mathrm{Nu} = 1 + \langle w\,T\rangle,
$$

with $\langle\cdot\rangle$ the volume average over $\Omega$; $\mathrm{Nu}=1$ is the
conductive state. Just above onset, steady counter-rotating rolls carry a modest
extra flux. As $\mathrm{Ra}$ increases the rolls give way to a population of
narrow rising and sinking **plumes ("fingers")**: thermal boundary layers on the
hot and cold walls shed small **protoplumes** that merge into domain-spanning
**megaplumes** (Otero et al. 2004; Hewitt, Neufeld & Lister 2012). In this
high-$\mathrm{Ra}$ regime 2-D simulations follow a near-linear law whose slope is
the robust feature,

$$
\mathrm{Nu} \approx 0.0069\,\mathrm{Ra} + 2.75 ,
$$

the trend the solver is validated against here. Both constants are Hewitt,
Neufeld & Lister's: $0.0069$ is the asymptotic high-$\mathrm{Ra}$ slope and
$+2.75$ a finite-$\mathrm{Ra}$ correction that fades relative to the linear term
as $\mathrm{Ra}\to\infty$, so at the moderate $\mathrm{Ra}$ sampled here the
points sit a little below the bare-slope line. The flow is statistically steady
but unsteady in detail, so $\mathrm{Nu}$ is time-averaged once the plumes are
established. This convective fingering is the porous-medium
cousin of the
viscous/density fingering seen when buoyancy drives one fluid through another —
the mechanism behind, e.g., dissolution trapping of CO₂ in saline aquifers.

## References

- C. W. Horton & F. T. Rogers, *Convection currents in a porous medium*,
  J. Appl. Phys. **16** (1945) 367.
- E. R. Lapwood, *Convection of a fluid in a porous medium*,
  Proc. Camb. Phil. Soc. **44** (1948) 508.
- J. Otero et al., *High-Rayleigh-number convection in a fluid-saturated porous
  layer*, J. Fluid Mech. **500** (2004) 263.
- D. R. Hewitt, J. A. Neufeld & J. R. Lister, *Ultimate regime of high Rayleigh
  number convection in a porous medium*, Phys. Rev. Lett. **108** (2012) 224503.
- D. A. Nield & A. Bejan, *Convection in Porous Media*, Springer.
