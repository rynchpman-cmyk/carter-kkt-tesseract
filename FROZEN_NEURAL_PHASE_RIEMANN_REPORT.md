# Frozen-Neural Phase Riemann Frontier

## Result

Tesseract now evolves a discontinuous two-current Riemann state whose two
halves select opposite branches of the qualified frozen neural constitutive
model. The complete Carter flux/path eigensystem remains real, causal,
recoverable, and convex throughout a 16/32/64/128-cell convergence ladder.
At the finest resolution, four cells dynamically cross the learned hard
classifier rather than merely transporting an unchanged phase mask.

The machine-readable record is
[`tesseract_neural_phase_riemann_results.json`](tesseract_neural_phase_riemann_results.json).
All 23 qualification gates pass. The frozen artifact digest is
`e3a8e89ca635cfe68d5bd963152097f85ec47b7960819830c9513f45b7bcf108`.

## Initial data

The periodic domain contains two interfaces. Both sides use

\[
n=0.1,\qquad \epsilon=0.12,\qquad d=0.005,\qquad v_N^x=0,
\]

while the carrier velocity jumps from

\[
v_D^x=0.112\quad\hbox{to}\quad v_D^x=0.103.
\]

The left state selects learned phase two and has signed switching margin
\(+6.6819\times10^{-4}\). The right state selects phase one and has margin
\(-3.1636\times10^{-4}\). This is a deliberately narrow phase-contact
problem: it crosses the learned nonsmooth constitutive surface without
leaving the causal basin of the frozen model.

The full nine-field Carter principal system is evolved with its
eleven-field entropy/tracer reconstruction extension, conditioned complete
eigensystem Roe-Rusanov fluctuations, straight-state DLM midpoint path
operator, SSP-RK3 time integration, and stage-local constitutive projection.
The learned classifier is hard and is held piecewise constant inside every
branchwise constitutive derivative and nonlinear recovery solve.

## Conserved-path basin audit

Before evolving the jump, Tesseract samples 129 points on the exact straight
line between the two endpoint **conserved** states. This is the same
state-space path selected for the DLM fluctuation; checking only a primitive
interpolation would not qualify the path actually used by the solver.

| Measurement | Value |
|---|---:|
| hard learned-phase transitions | 1 |
| maximum physical characteristic speed | 0.9824085 |
| maximum characteristic imaginary part | \(3.29\times10^{-18}\) |
| maximum eigenvector condition number | 284.98 |
| maximum primitive-recovery residual | \(4.11\times10^{-9}\) |
| minimum Legendre eigenvalue | 0.97485 |
| minimum thermodynamic eigenvalue | 0.90233 |
| minimum path-symbol norm | 0.09385 |

The closest sampled point lies \(4.50\times10^{-6}\) from the classifier
surface, as expected for a path deliberately crossing it. The endpoint
margins, not that unavoidable interior crossing distance, are the
admissibility margins. There is exactly one classifier transition along the
path.

## Resolution convergence

At \(t=0.02\) with CFL 0.12:

| Cells | Phase-two fraction | Cells changing phase | Minimum final switching distance |
|---:|---:|---:|---:|
| 16 | 0.50000 | 0 | \(2.12\times10^{-4}\) |
| 32 | 0.50000 | 0 | \(1.33\times10^{-4}\) |
| 64 | 0.50000 | 0 | \(2.80\times10^{-5}\) |
| 128 | 0.53125 | 4 | \(1.03\times10^{-4}\) |

| Grid pair | Successive normalized \(L_1\) error | Observed order |
|---|---:|---:|
| 16 to 32 | 0.00957931 | 0.9558 |
| 32 to 64 | 0.00493851 | 0.6534 |
| 64 to 128 | 0.00313989 | — |

The successive error decreases at every refinement. The measured order is
consistent with a resolved nonsmooth solution: high-order reconstruction
cannot retain its formal smooth order at a discontinuity.

Across every level:

- primitive recovery reports zero failed cells;
- both learned phases and both periodic interfaces remain present;
- the full characteristic spectra remain real and causal;
- the minimum Legendre and thermodynamic eigenvalues remain positive;
- both material densities remain positive without invoking the positivity
  limiter; and
- periodic baryon and carrier-number relative balance errors remain below
  \(5.6\times10^{-16}\).

The largest recorded physical speed is 0.9824085, the largest eigenvector
condition number is 334.67, and the largest recovery residual is
\(1.95\times10^{-9}\).

## What changed

The new frontier adds:

- the `neural_phase_contact` Riemann state;
- an exact conserved-state corridor audit;
- cellwise phase-mask change tracking from the initial state;
- phase-interface, final switching-distance, and periodic number-balance
  diagnostics;
- a dedicated frozen-neural resolution runner and PowerShell entrypoint; and
- regression coverage for the corridor, a short physical ladder, the frozen
  artifact digest, and the tracked 128-cell phase crossing.

## Reproduction

Run the recorded campaign:

```powershell
.\run_tesseract_neural_phase_riemann.ps1
```

Run the faster regression ladder:

```powershell
.\run_tesseract_neural_phase_riemann.ps1 `
    -Resolutions "16,32,64" `
    -FinalTime 0.005 `
    -CorridorSamples 33
```

The short ladder qualifies the causal path and convergence machinery but is
not expected to change a cell's classifier state. The tracked 128-cell run
is therefore the evidence for dynamic learned-phase crossing.

## Honest scope

This closes the previous analytic-M1-only convergence boundary for one
qualified frozen-neural discontinuity. It does not establish universal
learned two-fluid shock robustness:

- the causal learned-phase corridor is narrow and the jump is mild;
- convergence is successive-grid convergence because no closed-form
  nonlinear two-current phase-transition solution is available;
- the constitutive master value is continuous at the learned threshold, but
  its selected derivative branch is nonsmooth;
- the DLM path is the explicitly selected straight conserved-state path with
  midpoint quadrature;
- spacetime and mixed-vector fields are frozen to isolate the Carter
  principal subsystem; and
- stronger phase jumps must be rejected when their full-symbol path leaves
  the frozen model's trained causal basin.

The next boundary is broader shock-accessible neural training, higher-order
DLM path quadrature, kinetic or microphysical reference data, and coupled
CCZ4 evolution of converged learned phase fronts.
