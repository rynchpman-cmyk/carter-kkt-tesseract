# Frozen constitutive promotion and first continuum ladder

## Outcome

The qualified frozen Theory 3.3 master is now an active production option in
the standalone Tesseract NR solver. Its serialized invariant gradient drives
both:

1. the nonlinear conserved-to-primitive inversion; and
2. the Carter stress, canonical carrier momentum, path-conservative carrier
   equation, geometry sources, and matter fluxes used by the coupled PDE RHS.

The analytic M1 master remains available as a selectable control. The two
paths share one constitutive interface and no longer contain independent
copies of the Carter coefficients.

The first nested-grid comparison at 8, 16, and 32 points passed every gate
defined for this promotion experiment. The complete machine-readable result
is in `tesseract_nr_convergence_results.json`.

## Constitutive lowering

The frozen ABI returns

\[
  \left(
    \Lambda,
    \frac{\partial\Lambda}{\partial n^2},
    \frac{\partial\Lambda}{\partial d^2},
    \frac{\partial\Lambda}{\partial x^2},
    \frac{\partial\Lambda}{\partial \sigma}
  \right).
\]

The production master converts these derivatives into the Carter
coefficients

\[
  B_N=-2\Lambda_{,n^2},\qquad
  B_D=-2\Lambda_{,d^2},\qquad
  A=-\Lambda_{,x^2}.
\]

It then constructs

\[
  \Psi=\Lambda+B_N n^2+B_D d^2+2A x^2
\]

and uses the same \(B_N\), \(B_D\), and \(A\) in the Hilbert stress,
energy-momentum density, carrier canonical momentum, drag inertia, and
principal-symbol audit. Consequently, primitive recovery and forward
evolution now see the same learned material law.

The analytic control is lowered through the identical interface. Its
invariant gradient is computed analytically, so switching between models does
not switch recovery or PDE algorithms.

## Nonsmooth branch treatment

The phase classifier remains hard. During each local Newton solve, its phase
is classified once from the current primitive seed and then treated as
piecewise constant by:

- the residual evaluation;
- centered numerical recovery Jacobians;
- entropy-manifold projection; and
- the implicit drag solve.

After recovery, the ordinary classifier is evaluated again on the physical
state. The solver reports the active-phase fraction and minimum distance from
the learned switching surface. A local solve that actually changes phase is
rejected instead of returning conserved variables built from one branch and
stress from the other. The frozen artifact ABI was extended with an explicit
branch-evaluation method without changing the original `evaluate` contract.

Second thermodynamic derivatives for the frozen model are computed from
branch-frozen derivatives of the learned invariant gradient. This keeps a
Newton perturbation from differentiating through the discrete phase decision.

## Active neural PDE experiment

The comparison uses a smooth periodic counterflow inside the artifact's
training enclosure:

- baryon density near `0.1`;
- carrier fraction near `0.05`;
- specific internal energy `0.15`;
- relative Lorentz factor near `1.008`;
- all cells on the learned second phase;
- minimum switching margin approximately `1.84e-3`.

Each model is evaluated on nested 8-, 16-, and 32-point grids. The study
compares both the semidiscrete full-state RHS and a short RK4 evolution at a
common final time of `1e-5`. Timesteps halve with resolution.

The initially proposed counterflow was rejected: although it passed recovery,
conservation, convexity, and switching tests, its frozen characteristic audit
produced a maximum speed of `1.0477`. The accepted experiment uses a lower
counterflow and audits the principal symbol at every cell on every level.

## Results

| Measurement | Analytic control | Qualified frozen |
|---|---:|---:|
| Semidiscrete observed order | `0.902414` | `0.902413` |
| Evolved-state observed order | `0.902399` | `0.902399` |
| Fine-grid continuum error estimate | `1.70723e-9` | `1.70723e-9` |
| Fine-grid combined-energy relative change | `1.72822e-11` | `1.72823e-11` |
| Recovery failures | `0` | `0` |
| Carrier-number relative change | `0.0` | `0.0` |
| Minimum Legendre eigenvalue | `0.989288` | `0.969017` |
| Minimum thermodynamic eigenvalue | `0.900849` | `0.900849` |
| Minimum causal margin | `3.52e-2` | `9.21e-3` |
| Minimum switching margin | not applicable | `1.84150e-3` |
| Active learned-phase fraction | `0.0` | `1.0` |

At the finest level, the learned response changes the full PDE RHS by an
aggregate `L2` norm of `3.76836e-8` and the evolved state by `8.54837e-9`.
The neural path is therefore active rather than merely loading or replaying
an unused artifact.

The approximately first-order rate is consistent with the current
unreconstructed Rusanov/path-conservative production flux. The important
result at this stage is that both the analytic and learned paths approach a
continuum limit at the same measured rate while remaining measurably
different.

## Acceptance gates

The generated report requires all of the following:

- no primitive-recovery failure;
- baryon and carrier relative balance below `1e-12`;
- combined-energy relative change below `1e-9`;
- nondecreasing total entropy within tolerance;
- learned phase active in more than half the cells;
- switching margin above `1e-3`;
- positive Legendre and thermodynamic margins;
- strong hyperbolicity and causal characteristic speeds at every sampled
  cell;
- finite positive self-convergence;
- and a nonzero learned modification of the PDE RHS.

All gates passed on the recorded ladder.

## Scope and remaining boundary

This is the first active-neural resolution ladder, not yet a strong-field
continuum solution. Its geometry starts flat while the material source is
nonzero, so the reported Hamiltonian constraint is intentionally not used as
a continuum qualification claim.

The next honest ladder is:

1. construct resolution-matched CMC/CTT data for the active learned phase;
2. converge the elliptic and evolved CCZ4 constraints together;
3. extend the evolution time and resolution range;
4. replace first-order interface states with qualified higher-order
   reconstruction;
5. scan the learned phase up to, but not through, its measured causal edge;
6. compare against an independent implementation;
7. only then lower the qualified frozen path to Slang/Vulkan.

The learned PSD mobility is still diagnostic at this milestone. Promoting it
into the dissipative source operator is a separate thermodynamic change and
will require its own entropy and stiff-limit qualification.
