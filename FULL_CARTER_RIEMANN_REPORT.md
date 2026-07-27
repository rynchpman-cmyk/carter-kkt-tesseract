# Full Carter Eigensystem and Riemann Frontier

## Result

Tesseract now constructs the complete numerical principal eigensystem of its
literal two-current Carter PDE and uses it in a path-conservative
characteristic solver. Two discontinuous relativistic multifluid problems
converge across 16/32/64/128-cell ladders.

The machine-readable record is
[`tesseract_carter_riemann_results.json`](tesseract_carter_riemann_results.json).
All 28 qualification gates pass.

## Why the path term belongs in the symbol

The production Carter sector evolves the nine independent physical fields

\[
U=(D_N,D_D,S_x,S_y,S_z,E,P_x^{D},P_y^{D},P_z^{D}).
\]

Entropy and tracer are transported as redundant/passive extensions, producing
an eleven-field reconstruction block.

For a spatial direction \(k\), the carrier canonical-momentum equation
contains the explicit right-hand-side product

\[
C(U)\left[\partial_k\chi_0+
v_D^k\partial_k\chi_k\right].
\]

Consequently the quasilinear matrix is not merely the flux Jacobian. With
primitive variables

\[
p=(\log n,q_N^i,\log d,q_D^i,\sigma),
\]

Tesseract evaluates

\[
A_k =
\left[
\frac{\partial F_k}{\partial p}
-C\left(
\frac{\partial\chi_0}{\partial p}
+v_D^k\frac{\partial\chi_k}{\partial p}
\right)
\right]
\left(\frac{\partial U}{\partial p}\right)^{-1}.
\]

The minus sign is required because the path product is implemented on the
right-hand side of the evolution equation. Constitutive phase decisions are
frozen during every centered numerical derivative.

This follows the path-conservative framework of Parés: a nonconservative weak
solution requires a specified state-space path, and the numerical
fluctuations must be consistent with it. The implementation currently uses a
straight-state DLM path with a midpoint path matrix. See
[Parés (2006)](https://doi.org/10.1137/050628052) and the high-order
nonconservative WENO construction of
[Ren and Parés (2024)](https://arxiv.org/abs/2407.04401).

## Complete eigensystem

The numerical process is:

1. evaluate \(U(p)\), \(F_k(p)\), \(\chi_0(p)\), and \(\chi_k(p)\);
2. compute centered branchwise derivatives with respect to all nine
   primitives;
3. assemble \(A_k\) including the path contribution;
4. compute all nine physical eigenvalues and left/right eigenvectors;
5. reconstruct repeated eigenspaces with an SVD nullspace basis;
6. normalize the eigenvectors deterministically;
7. extend physical modes consistently into entropy and tracer; and
8. reject complex, inaccurate, or ill-conditioned decompositions.

The SVD step is important. Transverse current modes are genuinely degenerate.
Blindly accepting the raw basis returned by a generic eigensolver can produce
condition numbers near machine singularity even when the invariant subspace
is complete. Tesseract orthonormalizes only eigenvalue clusters and then
checks the resulting eigenpair residual, so numerical regularization cannot
silently hide a defective symbol.

### Recorded audit

| Measurement | Value |
|---|---:|
| physical modes | 9 |
| maximum imaginary part | 0 |
| maximum eigenpair residual | \(1.66\times10^{-14}\) |
| left/right biorthogonality residual | \(3.89\times10^{-15}\) |
| eigenvector condition number | 214.80 |
| path-symbol norm | 0.07268 |
| independent acoustic-speed difference | \(2.42\times10^{-11}\) |

The four longitudinal sound speeds are

\[
(-0.94407544,-0.36648495,0.40059916,0.95910449),
\]

matching the older independent current/momentum characteristic audit. The
remaining modes are the entropy/contact and transverse-current families. The
appearance of two sound families is consistent with relativistic
two-constituent shock theory; see
[Vlasov (1998)](https://arxiv.org/abs/hep-ph/9808250).

On a symmetric oblique state, the complete x- and y-direction spectra agree
to \(8.88\times10^{-16}\). The maximum oblique eigenpair residual is
\(5.02\times10^{-15}\), and the largest condition number is 263.15.

## Path-conservative Riemann solver

At each face, the solver forms a conditioned Roe/Rusanov blend in the complete
eigenbasis. The conservative flux is accompanied by the straight-path jump

\[
\mathcal P_{i+1/2}
\approx \frac{1}{2}(B_i+B_{i+1})(U_{i+1}-U_i),
\]

distributed as path-conservative left/right fluctuations. SSP-RK3 advances
the homogeneous special-relativistic Carter subsystem, and every stage is
projected onto the constitutive manifold. The existing conservative
positivity limiter remains active.

### Counterflow shock

| Cells | Successive normalized \(L_1\) error |
|---:|---:|
| 16 to 32 | 0.0272004 |
| 32 to 64 | 0.0134487 |
| 64 to 128 | 0.00676812 |

Observed orders are 1.016 and 0.991, as expected for a converged
shock-containing solution. At 128 cells:

- recovery residual: \(1.73\times10^{-9}\);
- minimum Legendre eigenvalue: 0.9652;
- minimum thermodynamic eigenvalue: 0.8750; and
- maximum physical characteristic speed: 0.9710.

### Colliding streams

| Cells | Successive normalized \(L_1\) error |
|---:|---:|
| 16 to 32 | 0.0247592 |
| 32 to 64 | 0.0135845 |
| 64 to 128 | 0.00735735 |

Observed orders are 0.866 and 0.885. At 128 cells:

- recovery residual: \(1.98\times10^{-9}\);
- minimum Legendre eigenvalue: 0.9507;
- minimum thermodynamic eigenvalue: 0.8527; and
- maximum physical characteristic speed: 0.9771.

All evolved densities remain positive and every physical cell symbol remains
real and causal.

## Reproduction

```powershell
.\run_tesseract_carter_riemann.ps1
```

For the faster regression ladder:

```powershell
.\run_tesseract_carter_riemann.ps1 `
    -Resolutions "16,32,64" `
    -FinalTime 0.01
```

## Honest scope

This crosses the numerical-eigensystem and Riemann-convergence boundary, with
four deliberate limits:

- the resolution ladders use the analytic M1 master function. The qualified
  frozen neural closure is included in the eigensystem audit and the earlier
  phase-crossing experiment, but discontinuous neural phase mixtures are not
  yet claimed to be resolution-converged;
- no closed-form nonlinear two-current Riemann solution is available here,
  so convergence is measured by nested successive-grid \(L_1\) differences;
- the DLM path uses midpoint quadrature and is therefore shock-consistent only
  with the explicitly selected straight-state path; and
- the Riemann experiments freeze spacetime and vector fields to isolate the
  homogeneous Carter principal subsystem. The complete coupled CCZ4 path
  remains covered by the smooth/full-step regression tests.

The next honest boundary is to qualify the frozen neural master throughout a
shock-accessible causal basin, upgrade the path integral beyond midpoint
quadrature, and compare against independently generated kinetic or
microphysical two-fluid shock data.
