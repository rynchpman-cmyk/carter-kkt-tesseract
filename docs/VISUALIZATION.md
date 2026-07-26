# Tesseract visualization instruments

## Full-stack neural-spacetime observatory

Run the default live render:

```powershell
.\run_visualization.ps1
```

This is the README dashboard and the visual summary of the complete qualified
stack. It does not paint arbitrary decorative fields: the renderer rebuilds a
real 24x24 frozen-neural periodic CMC state and combines it with the tracked
machine-readable frontier results.

The five views are:

1. **Live 2D neural CMC slice:** the local Hamiltonian field, conformal
   contours, and transverse carrier current from the active frozen closure.
2. **Origin-to-spacetime pipeline:** scaled addition, parity/KKT, the learned
   Carter master, physical certificates, WENO/Strang transport, and CCZ4.
3. **Numerical ladders:** the measured piecewise, MUSCL-MC, and WENO5-Z
   transport errors plus evolved Hamiltonian/momentum convergence.
4. **Constrained horizon:** analytic and frozen constraint drift over all 128
   accepted physical steps, with the learned switching margin.
5. **Qualification tesseract:** four colored edge directions for hard KKT,
   learned Carter constitution, conservative PDE transport, and CCZ4
   spacetime geometry.

The generated image is `tesseract_full_stack_dashboard.png`.

Fast artifact-only render:

```powershell
.\run_visualization.ps1 -Fast -NoShow
```

The fast path reconstructs a deterministic spatial profile from the recorded
finest-grid metrics. It is useful for layout development; the tracked README
image is produced with the live CMC path.

## Interactive Vulkan literal microscope

Run:

```powershell
.\run_visualization.ps1 -Mode literal
```

This visualization is backed by batched evaluations of the actual baseline
and conditioned Slang modules.

### Panel 1 — basin/time atlas

- horizontal axis: initial state \(z_0\);
- vertical axis: literal recurrence step;
- color: \(\operatorname{asinh}(z_n)\);
- black: non-finite state, state-limit exit, or invalid KKT certificate.

The nonlinear color transform preserves the explosive baseline without
flattening the conditioned basin.

### Panel 2 — literal cobweb

The panel plots \(F(z)\), the diagonal \(F(z)=z\), and cobweb trajectories for

```text
z0 = [-0.8, -0.2, 0.2, 0.8].
```

The conditioned view exposes the learned capture step followed by collapse
onto the intrinsic fixed point.

### Panel 3 — KKT phase atlas

Each valid pixel is colored by its exact selected active-set mask. Black
columns expose initial conditions whose trajectories leave the valid regime.
Fine filaments reveal changes in active-set history across nearby initial
conditions.

### Panel 4 — projected Carter tesseract

The 16 shader vertices are embedded at \(\{-1,+1\}^4\), rotated in multiple
4D coordinate planes, projected to 3D, and projected again to the screen.

`carterDiagnosticKernel` drives the display with four channels per vertex:

- scalar value;
- scalar tangent;
- vector norm;
- matrix norm.

The selected `z0=+0.8` lane also displays current state, KKT mask, classifier,
parity cell, switching margins, condition estimate, and Lyapunov value.

### Controls

- **Conditioned / Baseline:** swap the entire shader-backed atlas.
- **Step slider:** inspect one recurrence time.
- **Play:** animate recurrence time and the 4D projection.
- **Snapshot:** write `tesseract_conditioning_atlas.png`.

Headless render:

```powershell
.\run_visualization.ps1 `
    -Mode literal `
    -Samples 320 `
    -Steps 64 `
    -NoShow
```

The measured 320-sample domain scan over \([-1,1]\) retained `81.2%` valid
conditioned trajectories at horizon 64. The baseline retained `0%`.
