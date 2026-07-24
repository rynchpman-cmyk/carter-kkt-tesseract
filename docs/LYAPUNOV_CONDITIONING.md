# Intrinsic Lyapunov conditioning

## Why the original map fails

At literal integration scale, the baseline trajectory from

```text
[-0.8, -0.2, 0.2, 0.8]
```

has maximum local gains:

```text
4.66 → 14.17 → 102.58
```

The third-step adjoint norm reaches approximately \(1.98\times10^6\), and the
fourth step loses a KKT certificate in one lane.

Residual-flow continuation can mechanically unroll this model, but by horizon
128 it needs a minimum scale of `0.0001953125`. That is numerical survival,
not intrinsic conditioning.

## Analytical basin

In an even parity cell, all learned corrections vanish:

\[
F(z)=2.25(z+c)^2+c.
\]

Choosing \(c=0.05\) yields the stable fixed point

\[
z_\*=0.1019493853295916
\]

with asymptotic gain

\[
|F'(z_\*)|=0.683772234.
\]

The conditioning problem is therefore divided into:

1. learn a parity-one capture map that sends the outer initial states into the
   parity-zero basin;
2. preserve the intrinsic stable core once captured.

## Objective

The Lyapunov candidate is

\[
V(z)=(z-z_\*)^2.
\]

Training penalizes positive violations of

\[
V(F(z))\le 0.9V(z),
\]

using

\[
\operatorname{ReLU}(V(F(z))-0.9V(z)).
\]

The full objective also includes:

- log-scaled distance to the attractor;
- penalties when KKT, classifier, or parity margin falls below `0.02`;
- a local-gain penalty above `1.0`;
- gradient clipping;
- bounded recurrence coefficients.

## Curriculum

The first phase trains two literal steps on capture states lying in odd parity
cells around the outer basin.

The second phase refines perturbed versions of the four validation states at
horizons

```text
4 → 8 → 16 → 32.
```

The final audit advances detached literal steps for 128 iterations while
measuring KKT validity, all switching margins, condition estimates, local
gain, and Lyapunov increments.

## Result

Both the PyTorch reference and Vulkan shader converge all four lanes to
\(z_\*\).

| Measurement | Result |
|---|---:|
| Literal steps | `128` |
| Integration scale | `1.0` |
| Minimum KKT margin | `0.0375403` |
| Minimum classifier margin | `0.0267592` |
| Minimum parity margin | `0.125` |
| Maximum KKT condition estimate | `1.47269` |
| Maximum Lyapunov growth | `0.0` |
| Terminal Lyapunov value, PyTorch | `3.08e-33` |

The capture step has maximum local gain `6.15849`, but it is strictly
Lyapunov-decreasing on the validated states. Post-capture gain is at most
`1.08281` and converges toward `0.68377`.

The distinction is important: the result is not a global contraction proof.
It is a learned capture into an analytically stable intrinsic basin, supported
by explicit margin and Lyapunov audits.
