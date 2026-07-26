# Tesseract NR PDE transplant report

## Outcome

Tesseract now contains a standalone CPU numerical-relativity reference
backend under `tesseract_nr/`. The backend advances a coupled 52-component
state containing CCZ4 geometry, conservative relativistic matter, Carter
Theory 3.3 baryon/carrier currents, and mixed massive-vector fields.

The source transfer was one-way. The private reference tree was treated as
read-only, and its aggregate 1,463-file SHA-256 manifest remained

```text
bdb70022e547e2d6b4a380902312663ddbbd6f2d45edb93902f886851ce0ed3a
```

before and after the transplant. The public package contains no import,
runtime, test, build, or data dependency on that tree.

## Public package

The dependency-closed NR slice contains:

| Module | Responsibility |
|---|---|
| `adm.py` | ADM geometry, curvature, and constraints |
| `ccz4.py` | conformal covariant Z4 evolution and gauge |
| `grid.py` | periodic and finite Cartesian differential operators |
| `grhd.py` | Valencia recovery, HLL fluxes, and hydrodynamics |
| `theory33.py` | Carter M1 master, stress, and characteristic audit |
| `production33.py` | coupled Theory 3.3 production evolution |
| `theory3.py` | mixed-vector fields, Gauss constraints, and damping |
| `initial_constraints.py` | isolated CTT and periodic CMC data |
| `boundary.py` | outflow, radiative, and incoming-Z4 treatment |
| `amr.py` | conservative transfer, subcycling, and reflux primitives |
| `backend.py` | array, decomposition, and local exponential contracts |
| `nr_diagnostics.py` | horizons, Hawking mass, and finite-radius `Psi4` |
| `kkt_horizon.py` | constrained spectral MOTS observer |
| `io.py` | portable checkpoint/restart storage |
| `frozen_closure.py` | qualified artifact-to-NR constitutive ABI |

Checkpoint format identifiers were renamed to the public `tesseract.*`
namespace. All copied regression imports now target `tesseract_nr`.

## Evolved state

The method-of-lines state is packed into 24 component-leading arrays:

| Sector | Fields | Scalar components |
|---|---|---:|
| Geometry | conformal metric/A, conformal factor, \(K\), \(\Theta\), \(\hat\Gamma^i\), lapse, shift, shift driver | 25 |
| Matter | baryon density, total momentum/energy, entropy, tracer | 7 |
| Vector | two spatial potentials, two electric momenta, longitudinal charges, cleaners | 16 |
| Carrier | source charge and canonical spatial momentum | 4 |
| **Total** |  | **52** |

The production path uses:

- RK4 method-of-lines evolution;
- stage-local CCZ4 algebraic projection;
- stage-local nonlinear multifluid recovery;
- stage-local entropy-manifold projection;
- conservative baryon and carrier-number fluxes;
- a path-conservative carrier canonical-momentum product;
- exact split mixed-vector damping;
- and an implicit entropy-positive carrier-drag collision map.

## Frozen constitutive bridge

`QualifiedFrozenClosure` reads `theory33_frozen_variants.json` directly and
verifies its variant digest. It evaluates, without fitting or PyTorch:

- the qualified two-phase master value;
- its branchwise analytic gradient with respect to all four invariants;
- the hard learned phase selection and contextual threshold;
- and the learned \(2\times2\) positive-definite mobility.

A first implementation attempted to load PyTorch into the NumPy NR process.
The local Windows environment exposed duplicate OpenMP runtimes, so that
design was rejected rather than hidden behind `KMP_DUPLICATE_LIB_OK`. The
current bridge lowers the serialized networks to pure NumPy and avoids that
runtime ambiguity.

The frozen closure is available at the NR boundary but does not silently
replace the analytic production master. Promotion into primitive recovery and
the PDE RHS requires new recovery, conservation, convexity, characteristic,
and continuum gates.

## Coupled smoke evolution

The public runner constructs a smooth periodic two-current state, performs
periodic CMC constraint construction with two matter/geometry Picard passes,
and advances the complete coupled state:

```powershell
.\run_tesseract_nr.ps1 -Points 8 -Steps 1 -Dt 1e-5
```

The reference result is:

| Diagnostic | Result |
|---|---:|
| CMC Hamiltonian residual | `1.78901e-7` |
| CMC momentum residual | `7.15627e-8` |
| Recovery failures | `0` |
| Baryon relative balance error | `0.0` |
| Carrier relative balance error | `0.0` |
| Combined-energy relative change | `1.74566e-7` |
| Final entropy change | `0.0` |
| Maximum relative Lorentz factor | `1.00005627` |
| Minimum Legendre eigenvalue | `2.36935` |
| Minimum thermodynamic eigenvalue | `2.01759` |
| A-field Gauss \(L_2\) | `1.43e-39` |
| B-field Gauss \(L_2\) | `2.37e-28` |

The coarse-grid CCZ4 constraint diagnostic after converting and recovering
the CMC state is larger than the elliptic residual
(`H_L2 = 5.85e-5`, `M_L2 = 1.19e-5`). Both values are reported rather than
conflated. This smoke run proves coupled advancement and local admissibility;
it is not a continuum strong-field claim.

## Regression evidence

The transplanted NR suite contains 87 passing tests covering:

- CCZ4 fixed points, projection, and fourth-order gauge-wave convergence;
- GRHD primitive recovery, shocks, conservation, and second-order convergence;
- periodic and physical finite boundaries;
- Theory 3.3 recovery, conservation, drag, damping, and restart;
- characteristic, causal, Legendre, and thermodynamic gates;
- CTT/CMC constraint construction;
- AMR conservation and reflux primitives;
- horizons, Hawking mass, and `Psi4` controls;
- frozen artifact digest, PSD mobility, and invariant-gradient checks;
- and the end-to-end coupled runner.

The existing 27 constitutive/recurrence tests also continue to pass in a
separate process. Separation is intentional on Windows because the PyTorch
and NumPy test environments may load different OpenMP implementations.

## Qualification boundary and next gate

This milestone establishes a working, standalone NR reference backend. It
does not yet establish:

- frozen-master authority inside nonlinear primitive recovery;
- continuum convergence of the new coupled smoke problem;
- nonlinear outer-boundary reflection bounds;
- long strong-field or binary stability;
- full-state moving-box AMR;
- native Slang/Vulkan NR kernels;
- or differentiable multi-step PDE adjoints.

The next promotion milestone is to make the production recovery consume the
frozen invariant gradient, compare every recovered conservative state against
the analytic M1 control, and accept the frozen path only when conservation,
entropy, KKT, Legendre, thermodynamic, characteristic, and continuum margins
all pass.
