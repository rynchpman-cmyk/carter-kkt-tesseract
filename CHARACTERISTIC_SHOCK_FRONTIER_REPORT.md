# Characteristic Shock and Phase-Boundary Frontier

## Result

Tesseract now has a qualified periodic finite-volume path for coupled
characteristic WENO-Z reconstruction, conservative positivity preservation,
and multidimensional hard constitutive phase crossings.

The tracked evidence is
[`tesseract_nr_shock_results.json`](tesseract_nr_shock_results.json). All 16
qualification gates pass.

## Numerical changes

### Coupled characteristic reconstruction

The former WENO5-Z path reconstructed every conserved component separately.
The new `characteristic_weno5_z` mode instead applies local
Lax--Friedrichs flux splitting and freezes one basis at each interface for the
entire five-point WENO stencil.

The material block

\[
  (D,S_x,S_y,S_z,E,\mathcal S,\mathcal T)
\]

uses a normalized relativistic acoustic/contact basis. The normal momentum,
energy, and density waves are coupled; transverse momentum, entropy, and
tracer remain linearly degenerate fields.

The Carter carrier block

\[
  (D_D,P^{(D)}_x,P^{(D)}_y,P^{(D)}_z)
\]

uses a frozen charge/canonical-momentum basis whose impedance comes from
\(B_D=-2\partial\Lambda/\partial d^2\) and whose wave scale uses the qualified
carrier sound speed. The Carter path-conservative canonical-momentum products
remain explicit after the conservative flux divergence.

The largest measured basis condition numbers were 6.77 in the physical
phase-crossing run and 10.42 in the positivity torture test.

### Conservative positivity preservation

For every periodic face, the solver computes:

1. the high-order characteristic WENO-Z flux;
2. the monotone first-order local Lax--Friedrichs control flux;
3. a forward-Euler admissibility coefficient for each adjacent cell; and
4. one face coefficient equal to the minimum of its two cell coefficients.

The final flux is

\[
  F_{i+1/2}=F^{L}_{i+1/2}
  +\theta_{i+1/2}\left(F^{H}_{i+1/2}-F^{L}_{i+1/2}\right),
  \qquad 0\leq\theta_{i+1/2}\leq1 .
\]

Because the same face coefficient is used on both sides and across each
coupled block, the limiter is conservative. It enforces material density,
carrier density, and a material dominant-energy interior proxy. Recovery
failure at an RK stage also activates a bounded global stage contraction as a
last-resort admissibility fallback; it was not needed in the recorded runs.

The falsification control matters. On the same 16x16 two-dimensional
rarefaction corner:

| Quantity | Unlimited characteristic WENO | Protected |
|---|---:|---:|
| minimum material/floor | 0.9874893 | 1.0000000 |
| minimum carrier/floor | 0.9810071 | 1.0000000 |
| limited faces | 0 | 40 |
| minimum face coefficient | 1.0 | 0.3641649 |

Only 40 of 512 axis-faces were limited, so this did not silently reduce the
whole update to first order.

### Hard phase crossings

Newton iterations still treat classifier and constitutive branch decisions as
piecewise constant. A converged primitive is then reclassified outside the
Newton solve. If its branch changed, recovery restarts on the new frozen
branch, with a bounded number of outer updates.

The 8x8 checkerboard experiment contains both learned phases. An intentionally
stale opposite-phase primitive cache forces 64 outer branch updates. Recovery
reconstructs the exact original hard phase map with a maximum residual of
approximately \(9\times10^{-9}\). A complete Strang-split CCZ4 + Carter step
then moves three cells across the learned phase threshold while retaining:

- zero recovery failures;
- \(1.73\times10^{-9}\) maximum final recovery residual;
- 0.9253 minimum Legendre eigenvalue;
- 0.8338 minimum thermodynamic eigenvalue;
- zero baryon balance error to printed precision;
- \(1.64\times10^{-16}\) relative carrier-number balance error; and
- positive total entropy production.

This is a real spatial threshold crossing, not just a classifier unit test.

## Reproduction

```powershell
.\run_tesseract_nr_shocks.ps1
```

The runner writes `tesseract_nr_shock_results.json`. The regression test is:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m unittest tests.nr.test_shock_frontier
```

## Honest scope

This result crosses the requested implementation boundary, but it is not yet
a proof of shock convergence:

- the current material and carrier bases are audited frozen approximations,
  not the complete numerical eigensystem of the full two-current Carter
  Jacobian;
- the local face limiter is currently qualified on periodic grids; physical
  boundaries retain the existing characteristic boundary machinery;
- the positivity torture uses an elevated configured floor so the frozen
  constitutive model stays inside a meaningful test regime while the WENO
  undershoot remains measurable;
- one 8x8 physical step demonstrates multidimensional branch crossing, not a
  resolution-converged Riemann solution; and
- entropy stability is measured, not established by a discrete entropy proof.

The next boundary is therefore clear: derive or numerically differentiate the
full Carter flux/path Jacobian, build its complete left/right eigensystem, and
run resolution ladders against relativistic multifluid Riemann and oblique
phase-interface benchmarks.
