# Theory 3.3 nonsmooth and neural-constitutive experiment report

## Executive result

Five literature-motivated experiments were implemented and run in dependency
order against the public, self-contained `Theory33HybridTesseract` model:

1. a certified monotone neural capture controller;
2. a conservative, jump-aware VJP at hard switching boundaries;
3. a convex neural dissipation potential;
4. a partially convex neural residual in the Carter master function; and
5. an explicit continuous-potential, two-phase Carter master.

All five experiments pass their declared acceptance criteria. The three
experiments that alter the physical recurrence also pass the existing
128-step broad-basin, KKT, physical, characteristic, and global finite-input
safety audits at literal integration scale `1.0`.

These are controlled numerical experiments, not a claim that a new physical
theory has been established. In particular, the learned Carter correction is
deliberately small, the phase data are synthetic, and the stability evidence
is empirical on the audited domains. No external private solver was imported,
copied, referenced, or required.

The machine-readable measurements are preserved in
[`theory33_experiment_results.json`](theory33_experiment_results.json).

## Literature map

The implementation takes architectural ideas from the following primary
sources while retaining the project's hard KKT and Carter structure:

| Project experiment | Relevant literature | Adopted idea |
|---|---|---|
| Certified neural controller | You et al., [Deep Lattice Networks and Partial Monotonic Functions](https://proceedings.neurips.cc/paper_files/paper/2017/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) | Piecewise-linear learned responses with monotonicity enforced by parameterization |
| Boundary VJP | Kong et al., [Saltation Matrices: The Essential Tool for Linearizing Hybrid Dynamical Systems](https://doi.org/10.1109/JPROC.2024.3440211) | Sensitivities of hybrid systems must account for state-dependent jumps |
| Convex potentials | Amos, Xu, and Kolter, [Input Convex Neural Networks](https://proceedings.mlr.press/v70/amos17b.html) | Nonnegative weights and convex activations can enforce input convexity |
| Dissipation learning | Huang et al., [Variational Onsager Neural Networks](https://arxiv.org/abs/2112.09085) | Learn evolution through free-energy and dissipation potentials while imposing thermodynamic structure |
| Neural constitutive response | Masi et al., [Thermodynamics-based Artificial Neural Networks for constitutive modeling](https://doi.org/10.1016/j.jmps.2020.104277) | Generate constitutive quantities from learned thermodynamic potentials rather than unconstrained outputs |
| Carter master function | Andersson and Comer, [Relativistic Fluid Dynamics: Physics for Many Different Scales](https://doi.org/10.12942/lrr-2007-1) | Currents, entrainment, momenta, and stress are generated from derivatives of a scalar multifluid master function |

The boundary experiment is *saltation-like*, not a literal continuous-time
saltation matrix. It evaluates the distributional jump term of this
discrete-time, piecewise recurrence. Likewise, the compact convex networks
use the ICNN design principle but are specialized one- and two-dimensional
architectures rather than reproductions of the cited general networks.

## Shared protocol

Every model-changing experiment is isolated from the validated default:

- a fresh `Theory33HybridTesseract` is constructed;
- only the experimental controller or constitutive module is substituted;
- parity, classifier, KKT active-set, phase, and outer-gate decisions remain
  hard and are treated as piecewise constant;
- the physical recurrence remains at literal scale `1.0`;
- no experiment mutates the shipped default parameters or artifacts.

The common acceptance audit contains:

- 128 recurrence steps from a sampled initialization basin;
- exact KKT certificate validity;
- future-timelike current and thermodynamic admissibility checks;
- differentiated characteristic validation and a subluminal speed bound;
- convergence to the conditioned attractor;
- a global absorbing-gate test using signed finite `float64` values through
  magnitude `1e300`;
- zero sensitivity outside the absorbing interval.

PyTorch's float64 path is used for these measurements. Seeds `3301`, `3302`,
and `3303` make the learned fits deterministic.

## Experiment 1: certified monotone neural capture controller

### Question

Can the fitted Chebyshev odd-cell controller be replaced by a learned response
without surrendering its useful monotonic shape or destabilizing the literal
recurrence?

### Construction

The negative and positive odd cells receive separate neural lattice branches.
For equally spaced coordinates \(\xi_i\), their vertex values are

\[
v_0=a,\qquad
v_i=v_{i-1}-\Delta_{\max}\,\sigma(\theta_i).
\]

Linear interpolation between vertices makes every branch nonincreasing by
construction. Since every decrement is bounded by
\(\Delta_{\max}=1.5\,\Delta\xi\), the absolute branch slope is certified not
to exceed `1.5`. A 24-vertex branch is trained to distill the existing compact
Chebyshev controller. This learned controller still enters only through
\(\kappa_n\); hard parity and all feasibility decisions are unchanged.

### Result

| Metric | Result |
|---|---:|
| Epochs | `800` |
| Samples per branch | `1,024` |
| Initial RMSE | `4.37110e-2` |
| Final RMSE | `1.65996e-4` |
| Maximum teacher error | `6.31845e-4` |
| Negative maximum absolute slope | `0.618920` |
| Positive maximum absolute slope | `0.757661` |
| Certified slope bound | `1.5` |
| Monotonicity test | passed |
| 128-step physical audit | passed |
| Converged basin fraction | `1.0` |

This establishes a narrow but useful result: the existing stabilizing
controller can be represented by a learned module whose monotonicity and
slope limit are architectural facts rather than soft penalties.

## Experiment 2: conservative and jump-aware boundary VJP

### Question

What derivative should reverse mode expose when the objective crosses a hard
parity, KKT, classifier, or safety boundary?

### Construction

For the one-step objective

\[
L(z)=\frac12\left(F(z)-z_\star\right)^2
\]

and a boundary \(b\), the experiment compares three objects over
\([b-\delta,b+\delta]\), with \(\delta=0.002\):

1. the mean ordinary branch derivative;
2. the two-sided finite-volume reference
   \[
   \bar L'=\frac{L(b+\delta)-L(b-\delta)}{2\delta};
   \]
3. a jump-aware derivative
   \[
   \bar L'_{\mathrm{jump}}
   =
   \operatorname{mean}(L'_{\mathrm{branch}})
   +\frac{L(b^+)-L(b^-)}{2\delta}.
   \]

The side derivatives and their midpoint are also recorded as a conservative
generalized-gradient diagnostic. No switching decision is softened.

### Result

Five boundaries were found: two outer safety boundaries, two discontinuous
parity boundaries, and one continuous KKT active-set boundary.

| Metric | Branchwise | Jump-aware |
|---|---:|---:|
| Mean error against finite-volume reference | `44.9394275` | `5.42579e-4` |
| Error at negative parity boundary | `112.348569` | `1.35621e-3` |
| Error at positive parity boundary | `112.348569` | `1.35614e-3` |

The two parity loss jumps are approximately `+0.4493997` and `-0.4493997`.
The jump correction improves both discontinuities. At the continuous outer
and KKT boundaries, the branchwise errors remain tiny; the KKT jump estimate
is numerical rather than a material discontinuity.

This is the clearest outcome of the suite. Ordinary BPTT is valid inside a
fixed switching sector, but it omits a dominant distributional term when a
training objective averages across a discontinuous parity boundary. A future
optimizer should therefore expose both:

- the ordinary piecewise VJP for trajectories that do not move the event; and
- an optional jump-aware estimator for objectives that intentionally average
  or randomize across boundary location.

Using the jump term indiscriminately would be wrong. Its scale depends on the
chosen smoothing or uncertainty width \(\delta\).

## Experiment 3: convex neural dissipation potential

### Question

Can a learned nonlinear relaxation law be generated from a convex potential,
with zero response at equilibrium, and remain safe inside the full recurrence?

### Construction

The two generalized forces are a normalized chemical imbalance and a
relative-rapidity drag force. The potential has the form

\[
\mathcal D_\phi(f)
=
\frac12\sum_i d_i f_i^2
+
\sum_j w_j\left[
\operatorname{softplus}(r_j)
+\operatorname{softplus}(-r_j)
-2\log 2
\right],
\]

where \(d_i>0\), \(w_j>0\), and \(r_j\) is a learned linear projection. The
even softplus term is convex and has zero value and zero gradient at the
origin. The relaxation response is generated by
\(\nabla_f\mathcal D_\phi\), not emitted as an unconstrained vector.

The synthetic teacher is

\[
\mathcal D_{\mathrm{teacher}}(f)
=\frac12\lVert f\rVert^2+5(f_0-f_1)^4.
\]

Training matches both potential values and their force derivatives.

### Result

| Metric | Result |
|---|---:|
| Epochs / samples | `700` / `2,048` |
| Initial response RMSE | `7.30791e-2` |
| Final response RMSE | `2.41018e-2` |
| Final potential RMSE | `1.91528e-3` |
| Minimum sampled potential | `7.95749e-8` |
| Minimum sampled Hessian eigenvalue | `0.914538` |
| 128-step physical audit | passed |
| Converged basin fraction | `1.0` |

The positive sampled Hessian margin confirms more than nonnegative
dissipation on the test set: the learned potential is strongly convex at the
probed points. The architecture provides global convexity, while the Hessian
test catches implementation mistakes and degenerate fits.

The scalar recurrence currently consumes the negative sum of the two
potential-generated responses. This is an experimental closure, not yet a
complete covariant Onsager mobility tensor.

## Experiment 4: partially convex Carter master residual

### Question

Can a learned constitutive correction enter the Carter master itself, so that
momenta, entrainment, generalized pressure, and stress are still generated by
automatic derivatives of one scalar?

### Construction

The analytic M1 master remains the reference closure. A convex residual is
added only in the dimensionless relative drift

\[
r=\frac{X_{+-}}{\sqrt{X_+X_-}}-1,
\qquad y=\frac{r}{0.02},
\]

using

\[
\Lambda_{\mathrm{hybrid}}
=
\Lambda_{\mathrm{M1}}
-a\,10^{-8}\,N_\phi(y).
\]

The network is a positive quadratic plus a nonnegative softplus Bregman
expansion. Therefore

\[
N_\phi(0)=0,\qquad N_\phi'(0)=0,\qquad N_\phi''(y)\ge 0.
\]

It is trained against \(N(y)=\tfrac12y^2\). The outer amplitude \(a\) is
qualified by a physical homotopy, beginning at `1.0`.

### Result

| Metric | Result |
|---|---:|
| Epochs | `700` |
| Final value RMSE | `2.84589e-4` |
| Final derivative RMSE | `1.02001e-3` |
| Minimum discrete second difference | `9.29625e-7` |
| Origin value / derivative | `0.0` / `0.0` |
| Energy scale | `1.0e-8` |
| Maximum scaled residual energy on fit domain | `4.99826e-9` |
| Selected amplitude | `1.0` on first attempt |
| Maximum characteristic speed | `0.972502` |
| Minimum physical margin | `1.05079e-9` |
| 128-step and global audits | passed |

The scale is scientifically important. An initial `1.0e-3` energy scale
failed the physical and characteristic gates even after reducing the
amplitude to `0.01`. Relative drift is normalized by small current
invariants, so apparently modest energy derivatives become very large
constitutive derivatives. The accepted `1.0e-8` scale is a nonzero
proof-of-integration, not evidence that a large neural Carter correction is
safe.

The physical margin is also thin—about `1.05e-9`. Increasing constitutive
authority should use a continuation on physical margins, not simply increase
the network width or optimizer learning rate.

## Experiment 5: explicit two-phase Carter master

### Question

Can the master retain a continuous potential while introducing a hard phase
decision and an honest constitutive derivative jump?

### Construction

At a learned threshold \(r_c\), the master switches from the analytic branch
to

\[
\Lambda_2
=
\Lambda_{\mathrm{M1}}
-a\,10^{-8}
\left[
N_\phi(r/0.02)-N_\phi(r_c/0.02)
\right].
\]

Subtracting the threshold value makes \(\Lambda_1(r_c)=\Lambda_2(r_c)\).
The derivative is not matched, so momenta and stress can jump. The phase mask
is detached and treated as piecewise constant, matching the treatment of the
parity, classifier, and KKT decisions.

A synthetic phase dataset was generated with \(r_c=0.006\), and a grid search
over candidate thresholds was used to recover it.

### Result

| Metric | Result |
|---|---:|
| Recovered phase threshold | `0.006` |
| Threshold fit MSE | `0.0` |
| Phase-one / phase-two samples | `796` / `230` |
| Potential jump | `1.49880e-15` |
| Constitutive gradient jump norm | `2.09798e-3` |
| Maximum characteristic speed | `0.972502` |
| Minimum physical margin | `1.05079e-9` |
| 128-step and global audits | passed |

This crosses the intended structural boundary: the code contains a genuine
hard constitutive phase edge, both phases are exercised, the scalar master is
continuous to numerical precision, and its generated constitutive response
is nonsmooth.

The exact threshold recovery is not a generalization result because the
training data were generated by the same response family. The useful result
is architectural and numerical: this kind of hard two-phase master can
coexist with the Carter derivatives and all current safety gates.

## Cross-experiment physical audit

| Variant | Basin points | Max \(|z|\) | Min KKT margin | Min physical margin | Max characteristic speed | Terminal max error | Global terminal error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Neural controller | 129 | `1.03836` | `3.50588e-4` | `4.68097e-8` | `0.966170` | `6.93889e-17` | `6.76039e-9` |
| Neural dissipation | 129 | `1.03836` | `3.50588e-4` | `4.68097e-8` | `0.966170` | `6.93889e-17` | `5.44859e-11` |
| Partially convex master | 97 | `1.02516` | `6.17814e-4` | `1.05079e-9` | `0.972502` | `6.93889e-17` | `3.13312e-11` |
| Explicit two-phase master | 97 | `1.02516` | `6.17814e-4` | `1.05079e-9` | `0.972502` | `6.93889e-17` | `3.13311e-11` |

All basin samples converged. All global tests fired the outer gate for the
intended extreme inputs, remained finite and physical, and reported zero
outer sensitivity.

The maximum transient Lyapunov-growth diagnostics are `0.615925` for the
controller/dissipation variants and `0.577521` for the master variants. Thus
these experiments demonstrate controlled eventual capture, not a
per-step-monotone Lyapunov certificate for every tested initial condition.

## What failed before the final run

Two failures materially changed the design:

1. A smooth monotone softplus-mixture controller fit the teacher only to about
   `0.16` worst-case error in the short run. It was replaced with the
   monotone piecewise-linear lattice, which gives an explicit slope
   certificate and `6.32e-4` worst-case full-fit error.
2. The first Carter residual used energy scale `1.0e-3`. Every tested
   homotopy amplitude down through `0.01` violated physical or
   characteristic validity. Adding an exact positive quadratic channel made
   the convex fit well-conditioned; reducing the energy scale to `1.0e-8`
   then passed at amplitude `1.0`.

These rejected settings are not hidden numerical noise. They show where the
current physical closure is sensitive and why the accepted model is
conservative.

## Reproduction

Install the training dependencies, then run:

```powershell
.\run_theory33_experiments.ps1 `
    -Output .\theory33_experiment_results.json
```

For a shorter development fit with the same acceptance audits:

```powershell
.\run_theory33_experiments.ps1 -Quick
```

The ordered Python entry point is:

```powershell
python .\run_theory33_experiments.py `
    --output .\theory33_experiment_results.json
```

Relevant files:

| Path | Purpose |
|---|---|
| `theory33_experiments.py` | Networks, hybrid masters, boundary estimator, training, and audits |
| `run_theory33_experiments.py` | Ordered CLI and JSON serialization |
| `run_theory33_experiments.ps1` | PowerShell entry point |
| `theory33_experiment_results.json` | Deterministic full-run evidence |
| `tests/test_theory33_experiments.py` | Structural convexity, normalization, monotonicity, and artifact tests |

## Recommended next stage

The most defensible progression is:

1. serialize learned weights so individual trained variants can be replayed
   without fitting;
2. replace synthetic constitutive targets with independently generated
   equation-of-state or transport data;
3. train the two-phase threshold on held-out noisy trajectories and report
   uncertainty, not only point recovery;
4. replace the scalar dissipation readout with a covariant positive-semidefinite
   mobility operator;
5. incorporate event-location derivatives into multi-step BPTT and compare
   ordinary, conservative, randomized-smoothing, and jump-aware estimators;
6. continue the Carter residual scale upward while optimizing explicit KKT,
   Legendre, thermodynamic, characteristic, switching, and Lyapunov margins;
7. port only a qualified frozen experimental model into the Slang/Vulkan
   module.

The current suite is a successful integration ladder. It shows that learned
monotone control, potential-generated dissipation, a convex Carter residual,
and an explicit nonsmooth phase edge can all be inserted without weakening
the public model's hard structural checks. The next scientific question is no
longer whether these components can run. It is how much constitutive authority
they can safely earn from real or independently simulated data.
