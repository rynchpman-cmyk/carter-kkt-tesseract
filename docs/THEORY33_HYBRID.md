# Theory 3.3 physical hybrid mode

## Outcome

`Theory33HybridTesseract` is a self-contained physical two-current mode for
the Carter/KKT recurrence. It replaces the original synthetic current
generator and learned constitutive baseline with:

- future-timelike baryon and carrier currents;
- the analytic M1 master function;
- Carter conjugate momenta and generalized pressure;
- the mixed-index Hilbert stress;
- thermodynamic and Legendre convexity gates;
- a differentiated local characteristic symbol; and
- the existing hard KKT, branch, parity, and boundary machinery.

The analytic master and constitutive coefficients also have a native
Slang/Vulkan implementation with generated reverse mode.

The literal recurrence now uses a fitted state-dependent \(\kappa_n\) on the
two reachable odd-parity capture intervals. This expands the qualified
initialization basin from four locked states to the compact interval
\(z_0\in[-1,1]\) without changing the analytic Carter master.

The module has no source, runtime, data, or build dependency on an external
solver repository. It contains only the local constitutive mathematics needed
by Tesseract.

## Physical KKT map

The four bounded KKT variables are interpreted as

\[
x_{\rm KKT}
=\left(\log(n/n_0),\log(d/(ny_0)),\eta_N,\eta_D\right).
\]

Therefore

\[
n=n_0e^{x_0},
\qquad
d=ny_0e^{x_1},
\]

and the two rapidities construct normalized four-velocities in the local ADM
frame. The physical currents are

\[
N^\mu=nU_N^\mu,
\qquad
C_D^\mu=dU_D^\mu.
\]

This parameterization enforces

\[
X_+=n^2>0,
\qquad
X_-=d^2>0,
\qquad
X_{+-}=nd\cosh(\eta_N-\eta_D)\ge nd
\]

by construction. Positivity and timelike orientation are no longer expected
to emerge accidentally from arbitrary four-vectors.

## Analytic M1 master

Define

\[
\mathcal W=X_{+-}-\sqrt{X_+X_-}.
\]

The default master is

\[
\Lambda_{\rm M1}
=-\rho_0(n,d,\sigma)
-a_1\mathcal W
-\frac{a_2}{2M_D^4}\mathcal W^2,
\]

with

\[
\rho_0
=n+\frac{e^{(\gamma-1)\sigma}n^\gamma}{\gamma-1}
+K_Dd^{1+c_D^2}
+\frac{(d-y_\star n)^2}{2\chi_D}.
\]

As in the original Tesseract, every constitutive coefficient comes from this
one scalar:

\[
\mathcal B_N=-2\Lambda_{,X_+},
\qquad
\mathcal B_D=-2\Lambda_{,X_-},
\qquad
\mathcal A=-\Lambda_{,X_{+-}}.
\]

The momenta, generalized pressure, and stress remain

\[
\mu_\nu=\mathcal B_NN_\nu+\mathcal AC_{D\nu},
\qquad
\chi_\nu=\mathcal B_DC_{D\nu}+\mathcal AN_\nu,
\]

\[
\Psi=\Lambda-N^\rho\mu_\rho-C_D^\rho\chi_\rho,
\]

\[
T^\mu{}_\nu
=\Psi\delta^\mu{}_\nu
+N^\mu\mu_\nu
+C_D^\mu\chi_\nu.
\]

An optional neural residual is present behind an exactly zero-initialized
scale:

\[
\Lambda=\Lambda_{\rm M1}+\epsilon_\phi\Lambda_\phi.
\]

It is disabled by default. Enabling it requires renewed convexity and
characteristic qualification.

## Hard physical diagnostics

Every step checks both branch copies for:

- positive baryon and carrier densities;
- positive temperature and Eulerian energy;
- one-negative/three-positive metric signature;
- normalized timelike current identities;
- relative Lorentz factor inside the configured domain;
- positive current-to-momentum Legendre map;
- positive thermodynamic Hessian;
- subluminal constituent sound-speed screen; and
- valid bounded-KKT certificates.

The optional full audit differentiates the exact local 1D conservation and
momentum-vorticity constitutive vectors, builds the principal symbol, and
checks:

- real characteristic roots;
- a complete, acceptably conditioned eigenbasis;
- positive Legendre and thermodynamic eigenvalues; and
- all characteristic speeds inside the metric light cone.

Classifier and KKT selections remain hard and piecewise constant. Reverse
mode differentiates only through the selected continuous calculation.

## Broad-basin dynamical conditioning

The parity-zero core at \(c=0.05\) has the stable fixed point

\[
z_\star=0.1019493853295916.
\]

States near the edge of its basin enter one of two reachable odd cells:

\[
I_-=[-1,-0.716\overline6],
\qquad
I_+=[0.616\overline6,1.05].
\]

Part 6 already allows \(\kappa_n\) to vary by step. The conditioned mode uses

\[
\kappa_n
=\kappa_0+\sum_{j=0}^{5}a_j^{(\pm)}T_j(\xi_\pm(z_n)),
\]

where \(T_j\) are Chebyshev polynomials and \(\xi_\pm\) maps each capture
interval to \([-1,1]\). The twelve coefficients are the deterministic
least-squares projection of the exact gain that sends an odd-cell state to
\(z_\star\) in one step. The maximum one-step projection error is
`1.27e-5`.

This controller changes only the recurrence gain. It does not alter
\(\Lambda_{\rm M1}\), the Carter momenta, the stress tensor, the Legendre
map, or the characteristic symbol. The failed alternative—putting the same
capacity into an unconstrained \(\Lambda_\phi\)—was rejected because it
stabilized the scalar map while making the principal symbol non-hyperbolic.

## Global absorbing gate

The literal parity map cannot be globally bounded on its own. Whenever
\(P_n=0\), its correction vanishes and arbitrarily large states follow the
quadratic \(u_n^2+c\). The global mode therefore applies an explicit outer
projection before any KKT or Carter evaluation:

\[
\widehat z_n=
\begin{cases}
z_n, & |z_n|\le 1,\\
\operatorname{sign}(z_n), & |z_n|>1.
\end{cases}
\]

The physical recurrence evaluates \(F(\widehat z_n)\). Consequently:

- the literal conditioned map is bit-for-bit unchanged throughout
  \([-1,1]\);
- every finite outer input enters the qualified inner enclosure in one step;
- KKT and Carter quantities are never evaluated at an unqualified or
  overflow-scale coordinate;
- the selected outer derivative is exactly zero; and
- NaN and infinite inputs are rejected rather than silently projected.

This is a visible hard safety decision, reported as `outer_safety_gate`,
`projected_input`, and `projection_distance`. It is not represented as
physics and is not hidden inside the master function.

## Literal validation

Run:

```powershell
.\run_theory33_hybrid.ps1
```

The default command executes 128 literal scale-1 steps and performs the full
characteristic audit at every step. The locked validation states are

\[
z_0=(-0.8,-0.2,0.2,0.8).
\]

The current reference result is:

| Property | Result |
|---|---:|
| KKT validity | all steps |
| Physical validity | all steps |
| Full characteristic validity | all audited states |
| Maximum \(|z|\) | `0.8` |
| Maximum Lyapunov increment | `0.0` |
| Minimum Legendre margin | `0.923796` |
| Minimum thermodynamic margin | `0.841667` |
| Minimum causal screen margin | `0.1` |
| Maximum characteristic speed | `0.965279` |
| Maximum current normalization error | `1.39e-17` |
| Terminal state | `0.1019493853` |

The same command additionally evolves 1,025 evenly spaced initial states over
\([-1,1]\) for 128 literal steps:

| Broad-basin property | Result |
|---|---:|
| KKT validity | all states and steps |
| Physical validity | all states and steps |
| Sampled characteristic validity | both branches over the enclosure |
| Maximum \(|z|\) | `1.045708` |
| Minimum KKT margin | `5.06e-5` |
| Minimum physical margin | `8.87e-10` |
| Maximum characteristic speed | `0.966170` |
| Converged grid fraction | `1.0` |
| Terminal maximum error | `6.94e-17` |

The global safety audit additionally uses 302 signed, logarithmically spaced
outer states through magnitude \(10^{300}\):

| Global property | Result |
|---|---:|
| All outer gates activated | yes |
| Finite through 128 steps | all states |
| KKT and physical validity | all states and steps |
| Maximum projected input | `1.0` |
| Maximum \(|z|\) after projection | `0.101953` |
| Maximum outer derivative | `0.0` |
| Terminal maximum error | `6.94e-17` |

Validate the native GPU constitutive path with:

```powershell
.\run_theory33_slang.ps1
```

The current Vulkan comparison against an independent float32 reference gives:

| Property | Maximum error |
|---|---:|
| M1 constitutive outputs | `1.91e-6` |
| Native reverse gradient | `9.54e-7` |

The Slang kernel currently covers the local analytic master and constitutive
coefficients. The complete hard-KKT recurrence still uses the PyTorch
reference in this mode.

## Scope boundary

This mode is a differentiable, homogeneous/local constitutive simulator. It
does not claim to be a complete numerical-relativity evolution. In
particular, it does not include:

- a KKT objective derived from multifluid primitive recovery or constrained
  thermodynamic equilibrium—the current box QP remains the Tesseract
  controller, although its outputs now parameterize admissible primitives;
- a claim that scalar recurrence time is physical coordinate or proper time;
- a complete Slang port of the hard-KKT physical recurrence;
- an interpretation of the outer projection as a physical law—it is a
  numerical absorbing gate around the qualified literal map;
- finite-volume current conservation on a spatial grid;
- spacetime geometry evolution;
- vector-field evolution;
- primitive recovery from conserved PDE variables;
- shocks, boundaries, AMR, or distributed execution.

Those require a separate grid-state design and validation campaign. The
present result establishes the physical Carter closure and its differentiable
hard-KKT integration before that larger step. The outer projection provides a
global absorbing construction for every finite scalar input; inside the
compact scalar domain, the dense qualification is empirical rather than a
formal interval proof. The exact repelling parity-zero fixed point
\(z=0.2424950591\), and its exact preimages, remain bounded and physical but
do not approach \(z_\star\); they form a measure-zero exception to the
attractor statement, not to boundedness.
