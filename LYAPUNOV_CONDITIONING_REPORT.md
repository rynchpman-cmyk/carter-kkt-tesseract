# Literal-map Lyapunov conditioning

## Milestone

The conditioned Carter/KKT model now runs the literal recurrence for 128
steps with:

- integration scale exactly `1.0`;
- valid active and opposite KKT certificates at every step;
- minimum KKT feasibility margin `0.0375403`;
- minimum classifier margin `0.0267592`;
- minimum parity-boundary distance `0.125`;
- maximum selected-system condition estimate `1.47269`;
- no positive per-lane Lyapunov increment;
- convergence of all lanes to `0.1019493853`.

No continuation, damping, state clamp, or gain backtracking is active in this
test.

## Intrinsic attractor

Inside an even parity cell, every learned-closure contribution vanishes and
the literal recurrence reduces to

```text
F(z) = 2.25 * (z + c)^2 + c
```

Conditioning fixes `c = 0.05`. The stable root is

```text
z* = (1 - 4.5*c - sqrt(1 - 18*c)) / 4.5
   = 0.1019493853
```

and its intrinsic local gain is

```text
|F'(z*)| = |4.5 * (z* + c)| = 0.683772234
```

The neural Carter master and the remaining recurrence coefficients learn a
one-step capture map for the two outer initial states. Once captured, the
trajectory evolves under the literal parity-zero core and converges to the
analytical fixed point.

## Lyapunov objective

Training uses

```text
V(z) = (z - z*)^2
```

and penalizes

```text
relu(V(z_next) - 0.9 * V(z))
```

alongside:

- distance to the attractor;
- KKT, classifier, and parity margins below `0.02`;
- local gains above `1.0`;
- a horizon curriculum of `4, 8, 16, 32`.

The literal 128-step PyTorch audit reports maximum Lyapunov growth `0.0` and
terminal Lyapunov value `3.08e-33`.

## Vulkan results

| Horizon | Status | Maximum `|z|` | Maximum Lyapunov growth | Maximum Lyapunov ratio | Minimum integration scale |
|---:|---|---:|---:|---:|---:|
| 1 | valid | 0.190625 | -1.75e-3 | 0.817914 | 1.0 |
| 4 | valid | 0.158219 | -4.45e-8 | 0.817914 | 1.0 |
| 16 | valid | 0.103161 | -4.92e-12 | 0.817914 | 1.0 |
| 64 | valid | 0.101949 | 0.0 | 1.0 at float32 fixed point | 1.0 |
| 128 | valid | 0.101949 | 0.0 | 1.0 at float32 fixed point | 1.0 |

The maximum whole-trajectory Jacobian is `6.15849` during the learned capture
step. After capture it is at most `1.08281` and converges toward `0.68377`.
This distinction matters: the capture is locally expansive but strictly
Lyapunov-decreasing on the validated initial basin.

The sampled BPTT finite-difference relative errors are `0.11%`, `0.16%`, and
`0.30%` at horizons 1, 4, and 16. At horizons 64 and 128 the terminal objective
has reached float32 precision and both analytical and numerical gradients
vanish.

## Artifacts

- `hybrid_tesseract_conditioned.pt`: conditioned PyTorch checkpoint.
- `hybrid_model_conditioned.json`: portable weights, dynamics, and audit
  metrics.
- `carter_tesseract_conditioned.slang`: generated conditioned Slang module.

## Run

Train and export:

```powershell
cd path\to\carter-kkt-tesseract
.\run_train_conditioned.ps1
```

Compile and run the literal Vulkan audit:

```powershell
.\run_conditioned_literal.ps1
```
