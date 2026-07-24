# Vulkan visualization instrument

Run:

```powershell
.\run_visualization.ps1
```

The visualization is backed by batched evaluations of the actual baseline and
conditioned Slang modules.

## Panel 1 — basin/time atlas

- horizontal axis: initial state \(z_0\);
- vertical axis: literal recurrence step;
- color: \(\operatorname{asinh}(z_n)\);
- black: non-finite state, state-limit exit, or invalid KKT certificate.

The nonlinear color transform preserves the explosive baseline without
flattening the conditioned basin.

## Panel 2 — literal cobweb

The panel plots \(F(z)\), the diagonal \(F(z)=z\), and cobweb trajectories for

```text
z0 = [-0.8, -0.2, 0.2, 0.8].
```

The conditioned view exposes the learned capture step followed by collapse
onto the intrinsic fixed point.

## Panel 3 — KKT phase atlas

Each valid pixel is colored by its exact selected active-set mask. Black
columns expose initial conditions whose trajectories leave the valid regime.
Fine filaments reveal changes in active-set history across nearby initial
conditions.

## Panel 4 — projected Carter tesseract

The 16 shader vertices are embedded at \(\{-1,+1\}^4\), rotated in multiple
4D coordinate planes, projected to 3D, and projected again to the screen.

`carterDiagnosticKernel` drives the display with four channels per vertex:

- scalar value;
- scalar tangent;
- vector norm;
- matrix norm.

The selected `z0=+0.8` lane also displays current state, KKT mask, classifier,
parity cell, switching margins, condition estimate, and Lyapunov value.

## Controls

- **Conditioned / Baseline:** swap the entire shader-backed atlas.
- **Step slider:** inspect one recurrence time.
- **Play:** animate recurrence time and the 4D projection.
- **Snapshot:** write `tesseract_conditioning_atlas.png`.

Headless render:

```powershell
.\run_visualization.ps1 -Samples 320 -Steps 64 -NoShow
```

The measured 320-sample domain scan over \([-1,1]\) retained `81.2%` valid
conditioned trajectories at horizon 64. The baseline retained `0%`.
