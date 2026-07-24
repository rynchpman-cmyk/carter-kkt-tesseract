# Differentiation through the hybrid recurrence

## Piecewise derivative contract

This simulator is not globally smooth. Parity, branch selection, classifiers,
and KKT active sets are hard decisions. The derivative implemented here is the
ordinary derivative inside one selected smooth region.

A local derivative is accepted only when perturbations retain the full hard
signature:

```text
iota
sigma
active KKT mask
active classifier mask
opposite KKT mask
opposite classifier mask
parity cell
```

Near a switching surface, the runner shrinks its probe or marks the gradient
low-confidence.

## Selected KKT implicit VJP

Candidate enumeration and ranking run outside autodiff. For the selected
system,

\[
A(\theta)x(\theta)=b(\theta).
\]

Given an upstream adjoint \(\bar{x}\), solve

\[
A^\mathsf{T}\lambda=\bar{x}.
\]

Then

\[
\bar{A}=-\lambda x^\mathsf{T},
\qquad
\bar{b}=\lambda.
\]

The selected row state maps these adjoints back to \(H,f,\ell,h\):

- free row: \(A_i=H_i,\ b_i=-f_i\);
- lower-active row: \(A_i=e_i^\mathsf{T},\ b_i=\ell_i\);
- upper-active row: \(A_i=e_i^\mathsf{T},\ b_i=h_i\).

`_SelectedBoxQPSolve` in `hybrid_tesseract.py` implements this transpose solve.
The shader rebuilds only the winning system for its forward tangent.

## Learned master function

\(\Lambda_\phi\) is a scalar MLP with a learned linear skip path. The Carter
coefficients require first feature derivatives of \(\Lambda_\phi\), while the
recurrence contains \(dM/dz\), which introduces mixed second derivatives.

The Slang reverse kernel uses a compact jet with components

```text
(f, D_u f, D_c f, D_v f, D_c D_v f)
```

and manually reverses the two tanh layers. This contracts all eight closure
adjoints without constructing a large synthesized higher-order reverse graph.

## Full temporal reverse

For scalar independent lanes,

\[
\bar{z}_t
=\bar{z}_{t+1}
\frac{\partial z_{t+1}}{\partial z_t},
\]

and shared weight gradients accumulate as

\[
\bar{w}
=\sum_t
\bar{z}_{t+1}
\frac{\partial z_{t+1}}{\partial w}.
\]

The temporal Jacobian is measured with centered Vulkan probes inside the
frozen hard region. The learned-closure weight VJP is analytical.

In an even parity cell, \(P_n=0\), so the exact learned-weight VJP is zero.
The runner enforces this identity instead of retaining float32 cancellation
noise from the physical closure probe.

## Why a small finite-difference component remains

The current Slang SPIR-V reverse emitter omits or mis-scales components when
lowering the nested dual-valued physical closure. The reliable production
path therefore uses:

- exact Slang forward physics;
- exact hard tapes and KKT tangents;
- exact analytical neural mixed-jet VJP;
- centered Vulkan calls for the small physical closure Jacobian and scalar
  temporal Jacobian;
- whole-trajectory finite differences as an external audit.

The reported conditioned BPTT relative errors are approximately:

| Horizon | Sampled relative error |
|---:|---:|
| 1 | `0.11%` |
| 4 | `0.16%` |
| 16 | `0.30%` |

At horizons 64 and 128 the terminal objective reaches float32 fixed-point
precision and both analytical and numerical gradients vanish.
