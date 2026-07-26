# Neural spacetime frontier report

## Outcome

Tesseract has crossed the next numerical-relativity boundary:

- the coupled conservative and carrier-current fluxes now have
  piecewise-constant, MUSCL-MC, and WENO5-Z reconstruction modes;
- the split carrier-drag map has an explicit second-order Strang path;
- the constraint-compatible Carter counterflow construction works in one,
  two, and three spatial dimensions with nonzero transverse relative flow;
- the frozen neural closure completes 128 constrained physical steps;
- a 6x6, 12x12, 24x24 active-neural spacetime ladder converges;
- and the complete 6x6x6 three-dimensional constraint/evolution path passes.

Every campaign-level gate in `tesseract_nr_frontier_results.json` is true.

## Coupled reconstructed fluxes

The earlier production solver evaluated a local Lax--Friedrichs interface
directly from adjacent cell states. That first-order path remains available
as `piecewise_constant`.

The new `muscl_mc` path reconstructs conserved states and their physical
fluxes with a monotonized-central slope. The new `weno5_z` path performs
fifth-order WENO-Z reconstruction on both sides of every interface before
applying the same conservative local wave-speed bound. Both paths operate on:

- baryon density;
- total material/vector momentum;
- total energy;
- entropy and tracer;
- carrier charge;
- and all three components of carrier canonical momentum.

The smooth periodic transport audit gives:

| Flux path | Finest error at 128 points | Finest observed order |
|---|---:|---:|
| Piecewise constant | `1.09037e-1` | `0.999710` |
| MUSCL-MC | `9.65456e-3` | `1.49512` |
| WENO5-Z | `2.10959e-8` | `4.99840` |

The MUSCL global L2 order is reduced at smooth extrema by the limiter, as
expected. WENO5-Z is the qualified smooth-spacetime path.

This fifth-order measurement applies to the reconstructed flux operator. The
complete PDE remains second-order because the CCZ4 geometry and source
derivatives use `fd2` in this campaign.

## Second-order source splitting

The carrier drag solve is nonlinear, entropy producing, and branch frozen.
Applying it once after the RK4 transport update is a first-order Lie split.
`Theory3ProductionParameters.source_splitting` now makes that choice
explicit:

- `lie` preserves the original production behavior;
- `strang` applies half a drag/damping map, a complete projected RK4 update,
  and the second half map.

The frontier runner uses Strang splitting with reconstructed fluxes. Hard
constitutive phase decisions remain piecewise constant inside every local
drag and recovery solve.

## Long-horizon constrained evolution

The recorded horizon evolves separate analytic-M1 and qualified-frozen CMC
states with:

```text
points       = 8
steps        = 128
dt           = 1e-7
final time   = 1.28e-5
flux         = WENO5-Z
source split = Strang
```

This is 12.8 times the previous recorded constrained physical time.

| Frozen measurement | Result |
|---|---:|
| Recovery failures | `0` |
| Baryon relative balance | `4.16e-16` |
| Carrier relative balance | `1.72e-16` |
| Minimum accepted step entropy change | `3.77504e-13` |
| Initial/final Hamiltonian L2 | `5.99096e-3` / `5.99140e-3` |
| Initial/final momentum L2 | `1.36946e-6` / `1.38948e-6` |
| Final switching margin | `2.06646e-3` |
| Final minimum Legendre eigenvalue | `0.968401` |
| Final minimum thermodynamic eigenvalue | `0.900842` |
| Final causal margin | `1.05734e-2` |
| Frozen/analytic final-state difference | `1.47252e-8` |

The constraint norms drift smoothly rather than jumping: Hamiltonian L2
changes by about `7.2e-5` relatively and momentum L2 by about `1.46%` over
the full horizon. Every sampled principal symbol stays strongly hyperbolic
and causal, and the learned phase remains active in every cell.

## Two-dimensional constraint convergence

The generalized initial data use density modes in every active coordinate,
including a cross mode, and a carrier velocity with genuine transverse
components. The baryon velocity is solved as a local counterflow multiplier
so the complete Carter momentum vanishes pointwise. This meets the periodic
CMC zero-mode condition without removing relative transport.

The frozen 2D ladder uses 6x6, 12x12, and 24x24 grids, 4/8/16 Strang steps,
and a common final time `2e-7`. An analytic-M1 control is run at 24x24.

| Measurement | Observed order |
|---|---:|
| Initial Hamiltonian constraint | `1.97488` |
| Initial momentum constraint | `1.97478` |
| Evolved Hamiltonian constraint | `1.97488` |
| Evolved momentum constraint | `1.97477` |
| Complete 52-component state | `2.07513` |
| Geometric-work energy rate | `1.92600` |

Additional finest-grid results:

- elliptic Hamiltonian residual: `1.83624e-8`;
- elliptic momentum residual: `8.64887e-20`;
- pointwise Carter momentum residual: `5.04e-21`;
- maximum transverse carrier speed: `0.035`;
- recovery failures: `0`;
- baryon relative balance: `0`;
- carrier relative balance: `0`;
- switching margin: `2.07026e-3`;
- causal margin: `9.78119e-3`;
- frozen/analytic final-state difference: `1.53779e-8`.

Nested comparisons use the actual periodic nodal hierarchy in
`PeriodicGrid`: every second fine node coincides with a coarse node in every
spatial direction.

## Three-dimensional path

The complete 6x6x6 frozen-neural state contains 216 cells and 11,232 evolved
scalars. It completed two WENO5-Z/Strang steps with:

- elliptic Hamiltonian residual `1.37915e-8`;
- elliptic momentum residual `7.69880e-20`;
- pointwise Carter momentum residual `4.69e-21`;
- maximum transverse carrier speed `0.03165`;
- zero recovery failures;
- no rejected entropy step;
- phase two active in every cell;
- switching margin `2.13178e-3`;
- causal margin `9.51300e-3`.

This is a code-path and physics-gate qualification, not a three-dimensional
resolution-convergence claim.

## Reproduce

From the repository root:

```powershell
.\run_tesseract_nr_frontier.ps1 `
    -HorizonSteps 128 `
    -Output tesseract_nr_frontier_results.json
```

The command runs the reconstruction audit, analytic/frozen long horizon, 2D
convergence ladder, and 3D smoke. The recorded CPU run takes several minutes.

## Honest boundary

This campaign establishes smooth multidimensional constrained evolution with
an active frozen neural constitutive response. It does not yet establish:

- characteristic-space WENO or shock robustness;
- positivity preservation under reconstructed discontinuities;
- phase-boundary crossings inside multidimensional spacetime;
- three-dimensional resolution convergence;
- strong-field collapse or binary evolution;
- moving-box AMR convergence;
- independent-code agreement;
- or Slang/Vulkan parity for the NR kernels.

The clean next campaign is a shock/phase-interface program with
characteristic reconstruction and a positivity limiter, followed by a
6x6x6/12x12x12/24x24x24 constrained ladder when accelerator kernels make
that scale practical.
