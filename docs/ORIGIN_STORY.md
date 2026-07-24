# From scaled addition to a Carter–KKT tesseract

This project did not begin as a physics simulator. It began with a deliberately
simple binary operation and escalated one structural question at a time.

## Stage 1 — scaled addition

\[
a\oplus b=\frac32(a+b).
\]

The factor \(3/2\) is the oldest surviving component of the final system.

## Stage 2 — parity-sensitive normalization

The result was given a second behavior when it was odd:

\[
a\oplus^\* b=
\frac{\frac32(a+b)}
     {\sqrt{a^2+b^2}}.
\]

This introduced two ideas that remain fundamental:

1. a discrete parity decision;
2. a normalized continuous correction.

## Stage 3 — scalar recurrence

The operation became a state update:

\[
u_n=\frac32(z_n+c),
\qquad
v_n=\frac{u_n}{\sqrt{z_n^2+c^2+10^{-8}}},
\]

\[
z_{n+1}
=u_n^2+c-
\begin{cases}
0,&\lfloor |u_n|\rfloor\text{ even},\\
\mu v_n,&\lfloor |u_n|\rfloor\text{ odd}.
\end{cases}
\]

Writing

\[
P_n=\frac{1-(-1)^{\lfloor |u_n|\rfloor}}{2}
\]

turns the branch into a hard gate \(P_n\in\{0,1\}\).

## Stage 4 — a differentiated correction

The odd-cell correction gained a trainable drift and a derivative of a
structured objective:

\[
z_{n+1}
=\frac94(z_n+c)^2+c
-P_n
\frac{
  \frac32\lambda(z_n+c)-\alpha\,dJ_n^{(L)}/dz_n
}{
  \sqrt{z_n^2+c^2+\varepsilon}
}.
\]

At this point the map was no longer just arithmetic. It had become a hybrid
dynamical system whose continuous update depended on the derivative of another
calculation.

## Stage 5 — constrained optimal response

The correction was generalized to

\[
z_{n+1}
=u_n^2+c_n+
\frac{P_n}{N_n}
\left[
\kappa_nu_n
-\alpha_n\frac{dJ_n^{(L)}}{dz_n}
+\chi_nR_n
\right],
\]

with

\[
N_n=\sqrt{z_n^2+c_n^2+\varepsilon}.
\]

The response parameter became the solution of a constrained problem:

\[
\theta_n^\*=\arg\min_\theta
\left\{
J_n^{(L)}(\theta)
+\frac{\omega_g}{2}
\left\|
\Phi(\operatorname{Bifurcate}(f,\Theta(\theta),z_n))
-\Gamma_{\mathrm{target}}
\right\|_W^2
+\frac{\omega_\theta}{2}\|\theta-\theta_n\|^2
\right\}
\]

subject to inequality, equality, and box constraints.

That change introduced KKT structure, active sets, feasibility certificates,
and piecewise-smooth derivatives.

## Stage 6 — Carter-coupled free energy

The final recurrence is

\[
z_{n+1}
=u_n^2+c_n+
\frac{P_n}{N_n}
\left[
\kappa_nu_n
-\alpha_n\frac{dJ_n^{(L)}}{dz_n}
+\chi_nR_n
-\zeta_n
\left(
D_{(z,\sigma_n)}^{(C)}M_n
+\omega_\Delta\iota_n[M_n]_S
\right)
\right].
\]

Here:

- \(D_{(z,\sigma_n)}^{(C)}M_n\) is the branchwise Carter-sector pullback;
- \([M_n]_S=M_n(C_n(z_n^+))-M_n(C_n(z_n^-))\) is the side jump;
- \(\iota_n\) records a parity, classifier, or KKT boundary crossing.

The free energy is

\[
M_n
=\omega_EE_n
+\frac{\omega_T}{2}\|T_n-T_{\mathrm{target}}\|_W^2
+\frac{\omega_\pi}{2}
\|\pi_n^{(+)}-\pi_n^{(-)}\|_G^2
+\omega_AA_nX_{+-}.
\]

One scalar master function generates the constitutive response:

\[
\Lambda_n=\Lambda(X_+,X_-,X_{+-};\Gamma_n,q_n^{(L)}),
\]

\[
B_+=-2\frac{\partial\Lambda_n}{\partial X_+},
\qquad
B_-=-2\frac{\partial\Lambda_n}{\partial X_-},
\qquad
A_n=-\frac{\partial\Lambda_n}{\partial X_{+-}}.
\]

The Carter momenta and stress follow:

\[
\pi_A^{(+)}=B_+J_A^{(+)}+A_nJ_A^{(-)},
\qquad
\pi_A^{(-)}=B_-J_A^{(-)}+A_nJ_A^{(+)},
\]

\[
\Psi_n
=\Lambda_n
-J_+^A\pi_A^{(+)}
-J_-^A\pi_A^{(-)},
\]

\[
T^A_{\ B}
=\Psi_n\delta^A_{\ B}
+J_+^A\pi_B^{(+)}
+J_-^A\pi_B^{(-)}.
\]

## What survived

The finished simulator is far removed from the opening line, but its core is
still visible:

```text
3/2 scaled addition
    → u_n = 3/2 (z_n + c_n)
    → parity cell
    → normalized correction
    → differentiated objective
    → constrained KKT response
    → Carter free-energy pullback
    → differentiable Vulkan recurrence
    → Lyapunov-conditioned literal map
```

The complexity was accumulated rather than substituted. The original
operation remains the first instruction in the final recurrence.
