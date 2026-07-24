# Progressive full-physics unroll

The Carter/KKT recurrence was unrolled on Vulkan from
`z0 = [-0.8, -0.2, 0.2, 0.8]` with terminal target `0.2`.

Every step evaluates the literal full-physics map `F(z)`. Progressive
continuation advances with

```text
z_next = z + scale * (F(z) - z)
```

using base scale `0.05`. The scale is halved as a hard, piecewise-constant
decision until the measured local gain is at most `1.1`. This is a
residual-flow integration of the complete recurrence; no state clamp is used.

## Implementation

- The 81 KKT candidates are enumerated and ranked as hard decisions.
- Only the winning `4 x 4` system is rebuilt for its exact tangent solve.
- The PyTorch reference uses the corresponding transposed system as a custom
  implicit VJP.
- The tape records both active sets, both classifiers, boundary iota, parity
  cell, KKT feasibility margins, classifier margins, parity distance, and
  selected-system condition estimates.
- Centered temporal probes must retain the complete hard signature. They
  shrink automatically if a probe approaches a switching boundary.
- The learned Carter closure uses the analytical mixed-direction neural VJP.

## Stabilized results

| Horizon | Status | Max terminal `|z|` | Loss | Max local gain | Initial adjoint norm | Weight-gradient norm | Sampled FD relative |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | valid | 0.852560 | 0.186519 | 1.09160 | 0.297382 | 2.1207e-5 | 1.53% |
| 2 | valid | 0.910093 | 0.186016 | 1.09762 | 0.300634 | 4.6494e-5 | 1.10% |
| 4 | valid | 0.975109 | 0.189129 | 1.09762 | 0.320912 | 7.7230e-5 | 1.62% |
| 8 | valid | 1.12859 | 0.206687 | 1.09762 | 0.419188 | 1.6183e-4 | 1.43% |
| 16 | valid | 1.56317 | 0.309238 | 1.09762 | 1.01762 | 4.2164e-4 | 1.45% |
| 32 | valid | 2.76195 | 0.881618 | 1.09816 | 5.44269 | 2.3840e-3 | 1.11% |
| 64 | valid | 8.39905 | 8.45954 | 1.09816 | 149.765 | 6.6706e-2 | not sampled |
| 128 | valid, low margin | 76.7222 | 732.014 | 1.09925 | 107,221 | 47.7505 | not sampled |

At horizon 32 the minimum integration scale is `0.00625`. At horizon 128 it
has fallen to `0.0001953125`, and the minimum classifier margin is
`8.38e-6`. The runner therefore labels the 128-step gradient low-confidence
even though all KKT certificates remain valid.

The system is now mechanically unrollable well past step three, but the raw
physical map remains expansive. Adaptive continuation delays that instability;
it does not make the original map contractive.

That diagnostic directly motivated the separate Lyapunov-conditioned model.
The conditioned model runs the literal map at scale `1.0` for 128 validated
steps with healthy switching margins and no positive Lyapunov increment. See
[LYAPUNOV_CONDITIONING_REPORT.md](LYAPUNOV_CONDITIONING_REPORT.md).

## Original literal-map result

With integration scale `1` and gain control disabled, the original result is
unchanged: horizons one through three reach maximum local gains of roughly
`4.66`, `14.17`, and `102.58`; one lane loses its KKT certificate at step four.

## Run

```powershell
cd path\to\carter-kkt-tesseract
.\run_progressive_full.ps1
```

Deeper diagnostic without finite-difference sampling:

```powershell
.\run_progressive_full.ps1 `
    -Horizons '64,128' `
    -FiniteDifferenceWeights 0
```

Original literal-map behavior:

```powershell
.\run_progressive_full.ps1 `
    -Horizons '1,2,3,4' `
    -IntegrationScale 1.0 `
    -MaxLocalGain 0
```
