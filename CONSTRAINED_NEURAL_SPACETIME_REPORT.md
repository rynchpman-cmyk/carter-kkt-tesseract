# Constraint-converged active-neural spacetime report

## Outcome

Tesseract now constructs resolution-matched periodic constant-mean-curvature
(CMC) initial data with the frozen Theory 3.3 phase active, evolves the full
52-component CCZ4/matter/vector state, and demonstrates joint convergence of:

- the elliptic CMC residuals;
- the CCZ4 Hamiltonian and momentum constraints;
- the complete evolved state;
- and the geometric-work energy rate.

The recorded 8/16/32 ladder passes every defined recovery, conservation,
entropy, switching, convexity, hyperbolicity, causality, and convergence gate.
The machine-readable evidence is in
`tesseract_nr_constrained_results.json`.

## Periodic compatibility construction

The periodic CMC momentum equation has a zero-mode compatibility condition.
A globally one-directional carrier flow violates that condition even if an
unweighted mean momentum is manually removed, because the conformal equation
contains the weighted source \(\psi^6 S_i\).

The new construction instead solves for the baryon velocity in every cell so
that

\[
  S_i^{\rm Carter}(x)=0
\]

pointwise while the two currents retain nonzero relative flow. The carrier
velocity stays near `0.125`, and the baryon counterflow is the unique bracketed
root of the frozen or analytic Carter momentum. This gives:

- pointwise momentum residual below `3.4e-21`;
- learned phase two active in every cell;
- switching margin above `2.0e-3`;
- a nontrivial canonical carrier momentum and transport flux.

The matter and geometry are then Picard-iterated:

1. evaluate the physical metric;
2. rebalance the Carter currents on that metric;
3. recover the complete material stress;
4. solve the periodic CMC Hamiltonian and momentum equations;
5. convert the resulting ADM geometry to CCZ4;
6. repeat until the material source and geometry agree.

Three Picard passes reduce the final source change to approximately
`2e-15`.

## Time integration boundary

The first direct attempt used one large `1e-5` step. It was rejected because
the split drag solve predicted a local entropy decrease. A `1e-6` step was
also outside the local entropy-qualified bound.

Repeated `1e-7` steps at eight points are entropy-positive. The production
ladder therefore scales

\[
  \Delta t_{\max}(N)=10^{-7}\frac{8}{N}
\]

with resolution. The recorded final time `1e-6` uses 10, 20, and 40 steps on
the 8-, 16-, and 32-point grids. This preserves the complete drag and
projection map rather than disabling difficult physics to obtain a longer
nominal horizon.

## Recorded results

| Measurement | Qualified frozen result |
|---|---:|
| Finest elliptic Hamiltonian residual | `1.97670e-8` |
| Finest elliptic momentum residual | `7.31e-20` |
| Initial Hamiltonian convergence order | `1.98560` |
| Initial momentum convergence order | `1.98594` |
| Evolved Hamiltonian convergence order | `1.98565` |
| Evolved momentum convergence order | `1.98593` |
| Complete evolved-state order | `2.04205` |
| Geometric-work energy-rate order | `2.04947` |
| Finest evolved Hamiltonian \(L_2\) | `3.93151e-4` |
| Finest evolved momentum \(L_2\) | `9.00026e-8` |
| Baryon relative balance | `0.0` |
| Carrier relative balance | `1.72e-16` |
| Recovery failures | `0` |
| Active learned-phase fraction | `1.0` |
| Minimum switching margin | `2.06723e-3` |
| Minimum Legendre eigenvalue | `0.968406` |
| Minimum thermodynamic eigenvalue | `0.900848` |
| Minimum causal margin | `1.05681e-2` |

The frozen and analytic models remain measurably distinct on the finest grid:

- full RHS difference: `6.26956e-8`;
- evolved-state difference: `1.47249e-8`.

Every sampled frozen principal symbol remains strongly hyperbolic and causal.
Total entropy increases on every accepted step.

## Why matter energy is not held constant

The CMC data have nonzero trace \(K\), so matter exchanges energy with the
evolving geometry. Requiring the coordinate-volume matter energy to remain
constant would be incorrect.

The qualification instead measures

\[
  \frac{E(t_f)-E(0)}{t_f}
\]

and requires this geometric-work rate to converge across resolutions. The
frozen rate converges at order `2.04947`.

## Qualification boundary

This result crosses the constraint-converged active-neural spacetime
boundary for a smooth one-dimensional periodic CMC problem. It does not yet
establish:

- long-time nonlinear stability;
- shocks or phase-boundary crossings on a constrained spacetime;
- three-dimensional strong-field collapse or binaries;
- moving-box AMR convergence;
- independent-code agreement;
- or accelerator parity.

The next campaign should extend the qualified timestep horizon, introduce
higher-order reconstructed fluxes, move to multidimensional constrained data,
and compare invariant observables rather than only grid-state norms.
