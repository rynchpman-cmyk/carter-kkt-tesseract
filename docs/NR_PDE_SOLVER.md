# Tesseract numerical-relativity PDE solver

## Status

Tesseract now contains a standalone float64 CPU reference backend for the
coupled system

```text
CCZ4 geometry
+ relativistic conservative matter
+ Carter Theory 3.3 baryon/carrier currents
+ mixed massive-vector fields
+ entropy-positive carrier drag
```

The backend is a one-way source transplant into this public repository. It
has no runtime, build, test, data, or source dependency on another project.
All public development occurs inside `tesseract_nr/`.

## State contract

The method-of-lines state contains 24 component-leading arrays, representing
52 scalar fields per cell:

| Sector | Scalar components |
|---|---:|
| CCZ4 geometry and gauge | 25 |
| Conservative material variables | 7 |
| Mixed-vector fields and cleaners | 16 |
| Carrier charge and canonical momentum | 4 |
| **Total** | **52** |

The material state evolves total energy and momentum. Exchange between the
Carter matter and vector fields is therefore internal to the combined balance
law.

## Numerical formulation

- Covariant CCZ4 with algebraic determinant and trace-free projection.
- Advective 1+log or harmonic lapse and Gamma-driver shift.
- Fourth- or second-order finite differences for geometry.
- Valencia conservative hydrodynamics with HLL fluxes.
- Conservative baryon and carrier-number fluxes.
- Path-conservative carrier canonical-momentum evolution.
- Stage-local nonlinear multifluid primitive recovery.
- Stage-local projection of redundant entropy onto the master manifold.
- Exact split mixed-vector damping and implicit entropy-positive carrier drag.
- Periodic or finite Cartesian grids with explicit physical boundaries.

The existing scalar Tesseract recurrence is not used as coordinate time. It
remains a constitutive-controller and event-derivative experiment. The PDE
solver consumes local Carter stress, momenta, characteristic bounds, and
admissibility margins.

`tesseract_nr.frozen_closure.QualifiedFrozenClosure` loads the qualified
digest-protected artifact without fitting and exposes its master value,
invariant gradient, learned phase decision, threshold, and PSD mobility to
batched NR states. The production RHS retains the analytic M1 closure until
this frozen response also passes primitive-recovery and conservative PDE
promotion gates; artifact availability is not silently treated as authority.

## Run

From the repository root:

```powershell
.\run_tesseract_nr.ps1 -Points 16 -Steps 4 -Dt 1e-5
```

Save a restartable checkpoint with:

```powershell
python run_tesseract_nr.py `
    --points 16 `
    --steps 4 `
    --dt 1e-5 `
    --checkpoint tesseract_nr_smoke.npz
```

The runner reports conservation errors, CCZ4 and Gauss constraints, recovery
residuals, convexity margins, relative counterflow, entropy production, and
whether the evolved state remains qualified.

## Validation boundary

The transplanted CPU backend includes regression coverage for:

- exact Minkowski evolution and CCZ4 algebraic constraints;
- fourth-order gauge-wave convergence;
- second-order smooth hydrodynamic convergence;
- relativistic shock recovery;
- baryon, carrier, charge, and energy balances;
- stiff entropy-positive drag;
- Gauss-compatible vector damping and finite boundaries;
- isolated and periodic constraint construction;
- checkpoint/restart equivalence;
- AMR conservation and reflux primitives;
- Schwarzschild horizon/Hawking-mass controls and flat-space `Psi4`.

This establishes a transparent NR reference solver, not yet a campaign-scale
HPC code. Moving-box AMR, nonlinear outer-boundary qualification, long
strong-field binaries, accelerator kernel parity, and independent-code
comparison remain promotion gates.
