# Mathematical model

## Literal recurrence

The scalar state update is

\[
F(z_n)=u_n^2+c_n+
\frac{P_n}{N_n}
\left[
\kappa_nu_n
-\alpha_n\partial_zJ_n^{(L)}
+\chi_nR_n
-\zeta_n
\left(D_{(z,\sigma_n)}^{(C)}M_n
+\omega_\Delta\iota_n[M_n]_S\right)
\right],
\]

where

\[
u_n=\frac32(z_n+c_n),\qquad
P_n=\frac{1-(-1)^{\lfloor|u_n|\rfloor}}2,\qquad
N_n=\sqrt{z_n^2+c_n^2+\varepsilon}.
\]

The literal map advances \(z_{n+1}=F(z_n)\). The progressive diagnostic can
instead integrate the residual flow

\[
z_{n+1}=z_n+s_n(F(z_n)-z_n),
\]

but the conditioned milestone uses \(s_n=1\) at every step.

## Hard decisions

The following values are discrete and are not smoothed:

- branch \(\sigma_n\);
- parity cell \(\lfloor|u_n|\rfloor\);
- four-bit classifier mask;
- four-variable KKT active state;
- boundary-crossing indicator \(\iota_n\).

Derivatives are branchwise. A derivative probe is accepted only when the
entire hard signature is unchanged.

## Bounded KKT sector

The response \(x=(q,\theta_1,\theta_2,\theta_3)\) solves

\[
\min_x\quad \frac12x^\mathsf{T}Hx+f^\mathsf{T}x
\qquad\text{subject to}\qquad
\ell\le x\le h.
\]

Each variable can be lower-active, free, or upper-active, producing
\(3^4=81\) candidates. The shader:

1. constructs the candidate linear system;
2. solves it with pivoted elimination;
3. checks primal feasibility, free-variable stationarity, and multiplier sign;
4. ranks valid candidates by objective;
5. applies a deterministic one-sided tie convention;
6. rebuilds only the winning system for its tangent.

For a selected active state,

\[
A(z)x(z)=b(z)
\]

and therefore

\[
A\,\frac{dx}{dz}
=\frac{db}{dz}-\frac{dA}{dz}x.
\]

## Carter sector

For currents \(J_+^A,J_-^A\) and metric \(G_{AB}\),

\[
X_+=-G_{AB}J_+^AJ_+^B,\qquad
X_-=-G_{AB}J_-^AJ_-^B,
\]

\[
X_{+-}=-G_{AB}J_+^AJ_-^B.
\]

The learned scalar master function is

\[
\Lambda_\phi(X_+,X_-,X_{+-},\Gamma,q).
\]

All constitutive coefficients are derivatives of this one scalar:

\[
B_+=-2\Lambda_{,X_+},\qquad
B_-=-2\Lambda_{,X_-},\qquad
A=-\Lambda_{,X_{+-}}.
\]

This keeps the neural component inside explicit Carter structure rather than
learning independent momenta or stress components.

## Boundary certificate

The shader records four continuous classifier margins:

\[
z,\qquad \Gamma-1,\qquad X_{+-},\qquad q-\theta_1.
\]

A step is marked as crossing when any of the following changes between the
start, midpoint, and end:

- KKT active mask;
- classifier mask or classifier-margin sign;
- parity cell.

The hard tape additionally stores feasibility margins, parity distance, and a
selected-system condition estimate.

## Intrinsic conditioned attractor

In an even parity cell, \(P_n=0\), so every learned correction disappears:

\[
F(z)=2.25(z+c)^2+c.
\]

With \(c=0.05\), the stable root is

\[
z_\*=
\frac{1-4.5c-\sqrt{1-18c}}{4.5}
=0.1019493853,
\]

with local gain

\[
|F'(z_\*)|
=|4.5(z_\*+c)|
=0.683772234.
\]

The learned odd-cell response acts as a capture map into this intrinsic basin.
