"""Conformal covariant Z4 evolution with matter sources.

Equations follow Alic et al., Phys. Rev. D 85, 064040 (2012), Eqs. 14--23,
using signature (-,+,+,+), K_ij = -L_n gamma_ij/2, and Eulerian matter
sources (rho, S_i, S_ij). Spatial derivatives are supplied by ``PeriodicGrid``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .adm import (
    ADMState,
    christoffel,
    covariant_hessian,
    inverse_metric,
    ricci_tensor,
)
from .grid import Array, PeriodicGrid
from .integrators import rk4_arrays

Matter = tuple[Array, Array, Array]


@dataclass(frozen=True)
class CCZ4Parameters:
    kappa_gravity: float = 8.0 * np.pi
    cosmological_constant: float = 0.0
    kappa1: float = 0.1
    kappa2: float = 0.0
    kappa3: float = 1.0
    shift_speed: float = 1.0
    eta_shift: float = 1.0
    slicing: str = "1+log"
    evolve_shift: bool = True
    radiative_boundaries: bool = True

    def __post_init__(self) -> None:
        if self.kappa_gravity <= 0.0 or self.kappa1 < 0.0:
            raise ValueError("kappa_gravity must be positive and kappa1 nonnegative")
        if self.kappa2 <= -1.0:
            raise ValueError("constraint damping requires kappa2 > -1")
        if self.shift_speed <= 0.0 or self.eta_shift < 0.0:
            raise ValueError("shift_speed must be positive and eta_shift nonnegative")
        if self.slicing not in {"1+log", "harmonic"}:
            raise ValueError("slicing must be '1+log' or 'harmonic'")


@dataclass
class CCZ4State:
    conformal_metric: Array
    conformal_A: Array
    conformal_factor: Array
    trace_K: Array
    theta: Array
    gamma_hat: Array
    lapse: Array
    shift: Array
    shift_driver: Array
    time: float = 0.0

    def copy(self) -> "CCZ4State":
        return CCZ4State(
            self.conformal_metric.copy(),
            self.conformal_A.copy(),
            self.conformal_factor.copy(),
            self.trace_K.copy(),
            self.theta.copy(),
            self.gamma_hat.copy(),
            self.lapse.copy(),
            self.shift.copy(),
            self.shift_driver.copy(),
            self.time,
        )


@dataclass(frozen=True)
class CCZ4Diagnostics:
    time: float
    hamiltonian_l2: float
    momentum_l2: float
    theta_l2: float
    z_l2: float
    determinant_linf: float
    trace_A_linf: float
    minimum_lapse: float
    minimum_metric_eigenvalue: float


class CCZ4Solver:
    def __init__(self, grid: PeriodicGrid, parameters: CCZ4Parameters | None = None) -> None:
        self.grid = grid
        self.parameters = parameters or CCZ4Parameters()

    def flat_state(self) -> CCZ4State:
        metric = self.grid.zeros((3, 3))
        for i in range(3):
            metric[i, i] = 1.0
        return CCZ4State(
            conformal_metric=metric,
            conformal_A=self.grid.zeros((3, 3)),
            conformal_factor=np.ones(self.grid.shape),
            trace_K=self.grid.zeros(),
            theta=self.grid.zeros(),
            gamma_hat=self.grid.zeros((3,)),
            lapse=np.ones(self.grid.shape),
            shift=self.grid.zeros((3,)),
            shift_driver=self.grid.zeros((3,)),
        )

    def physical_geometry(self, state: CCZ4State) -> tuple[Array, Array]:
        factor_sq = state.conformal_factor**2
        h = state.conformal_metric / factor_sq[None, None, ...]
        K = (
            state.conformal_A
            + (state.trace_K / 3.0)[None, None, ...] * state.conformal_metric
        ) / factor_sq[None, None, ...]
        return h, K

    def from_adm(
        self,
        geometry: ADMState,
        *,
        lapse: Array | None = None,
        shift: Array | None = None,
    ) -> CCZ4State:
        h_inv, determinant, _ = inverse_metric(geometry.h)
        factor = determinant ** (-1.0 / 6.0)
        conformal_metric = factor[None, None, ...] ** 2 * geometry.h
        trace_K = np.einsum("ij...,ij...->...", h_inv, geometry.K)
        conformal_A = factor[None, None, ...] ** 2 * (
            geometry.K - geometry.h * (trace_K / 3.0)[None, None, ...]
        )
        conformal_inverse = inverse_metric(conformal_metric)[0]
        conformal_gamma = christoffel(self.grid, conformal_metric, conformal_inverse)
        gamma_tilde = np.einsum(
            "jk...,ijk...->i...", conformal_inverse, conformal_gamma
        )
        result = CCZ4State(
            conformal_metric,
            conformal_A,
            factor,
            trace_K,
            self.grid.zeros(),
            gamma_tilde,
            np.ones(self.grid.shape) if lapse is None else np.asarray(lapse, dtype=float),
            self.grid.zeros((3,)) if shift is None else np.asarray(shift, dtype=float),
            self.grid.zeros((3,)),
            geometry.time,
        )
        return self.project_algebraic(result)

    def to_adm(self, state: CCZ4State) -> ADMState:
        h, K = self.physical_geometry(state)
        return ADMState(h, K, state.time)

    def conformal_connection(self, conformal_metric: Array) -> tuple[Array, Array, Array]:
        inverse = inverse_metric(conformal_metric)[0]
        gamma = christoffel(self.grid, conformal_metric, inverse)
        contracted = np.einsum("jk...,ijk...->i...", inverse, gamma)
        return inverse, gamma, contracted

    def z_covector(
        self,
        conformal_metric: Array,
        gamma_hat: Array,
        gamma_tilde: Array | None = None,
    ) -> Array:
        if gamma_tilde is None:
            _, _, gamma_tilde = self.conformal_connection(conformal_metric)
        return 0.5 * np.einsum(
            "ij...,j...->i...", conformal_metric, gamma_hat - gamma_tilde
        )

    def project_algebraic(self, state: CCZ4State) -> CCZ4State:
        matrices = np.moveaxis(state.conformal_metric, (0, 1), (-2, -1))
        determinant = np.linalg.det(matrices)
        if np.any(determinant <= 0.0):
            raise FloatingPointError("conformal metric lost positive definiteness")
        rescale = determinant ** (-1.0 / 3.0)
        metric = state.conformal_metric * rescale[None, None, ...]
        metric = 0.5 * (metric + np.swapaxes(metric, 0, 1))
        inverse = inverse_metric(metric)[0]
        trace_A = np.einsum("ij...,ij...->...", inverse, state.conformal_A)
        A = state.conformal_A - metric * (trace_A / 3.0)[None, None, ...]
        A = 0.5 * (A + np.swapaxes(A, 0, 1))
        if np.any(state.conformal_factor <= 0.0) or np.any(state.lapse <= 0.0):
            raise FloatingPointError("conformal factor and lapse must remain positive")
        return CCZ4State(
            metric,
            A,
            state.conformal_factor,
            state.trace_K,
            state.theta,
            state.gamma_hat,
            state.lapse,
            state.shift,
            state.shift_driver,
            state.time,
        )

    def _derivative_or_zero(self, field: Array, axis: int) -> Array:
        return self.grid.derivative(field, axis) if axis < self.grid.ndim else np.zeros_like(field)

    def _gradient_component(self, scalar: Array, axis: int) -> Array:
        return self._derivative_or_zero(scalar, axis)

    def rhs(self, time: float, values: tuple[Array, ...], matter: Matter) -> tuple[Array, ...]:
        del time
        (
            gt,
            At,
            phi,
            K,
            theta,
            gamma_hat,
            alpha,
            beta,
            driver,
        ) = values
        rho, momentum, stress = matter
        p = self.parameters
        gt_inv, gt_gamma, gamma_tilde = self.conformal_connection(gt)
        Z_down = self.z_covector(gt, gamma_hat, gamma_tilde)
        Z_up_physical = phi[None, ...] ** 2 * np.einsum(
            "ij...,j...->i...", gt_inv, Z_down
        )
        h = gt / phi[None, None, ...] ** 2
        h_inv, _, _ = inverse_metric(h)
        h_gamma = christoffel(self.grid, h, h_inv)
        ricci = ricci_tensor(self.grid, h, h_inv, h_gamma)
        scalar_ricci = np.einsum("ij...,ij...->...", h_inv, ricci)
        hessian_alpha = covariant_hessian(self.grid, alpha, h_gamma)
        laplace_alpha = np.einsum("ij...,ij...->...", h_inv, hessian_alpha)
        stress_trace = np.einsum("ij...,ij...->...", h_inv, stress)

        DZ = self.grid.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                DZ[i, j] = self._derivative_or_zero(Z_down[j], i)
                for k in range(3):
                    DZ[i, j] -= h_gamma[k, i, j] * Z_down[k]
        div_Z = np.einsum("ij...,ij...->...", h_inv, DZ)

        beta_div = self.grid.divergence(beta)
        dgt = self.grid.zeros((3, 3))
        dAt = self.grid.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                dgt[i, j] = -2.0 * alpha * At[i, j] - (2.0 / 3.0) * gt[i, j] * beta_div
                dAt[i, j] = -(2.0 / 3.0) * At[i, j] * beta_div
                for k in range(3):
                    dgt[i, j] += beta[k] * self._derivative_or_zero(gt[i, j], k)
                    dgt[i, j] += gt[k, i] * self._gradient_component(beta[k], j)
                    dgt[i, j] += gt[k, j] * self._gradient_component(beta[k], i)
                    dAt[i, j] += beta[k] * self._derivative_or_zero(At[i, j], k)
                    dAt[i, j] += At[k, i] * self._gradient_component(beta[k], j)
                    dAt[i, j] += At[k, j] * self._gradient_component(beta[k], i)

        core = -hessian_alpha + alpha[None, None, ...] * (
            ricci + DZ + np.swapaxes(DZ, 0, 1) - p.kappa_gravity * stress
        )
        core_trace = np.einsum("ij...,ij...->...", h_inv, core)
        core_tf = core - h * (core_trace / 3.0)[None, None, ...]
        dAt += phi[None, None, ...] ** 2 * core_tf
        At_mixed = np.einsum("ik...,kj...->ij...", gt_inv, At)
        dAt += alpha[None, None, ...] * (
            At * (K - 2.0 * theta)[None, None, ...]
            - 2.0 * np.einsum("ik...,kj...->ij...", At, At_mixed)
        )

        adv_phi = sum(
            beta[k] * self._derivative_or_zero(phi, k) for k in range(3)
        )
        dphi = (alpha * K - beta_div) * phi / 3.0 + adv_phi
        adv_K = sum(beta[k] * self._derivative_or_zero(K, k) for k in range(3))
        dK = (
            -laplace_alpha
            + alpha
            * (
                scalar_ricci
                + 2.0 * div_Z
                + K**2
                - 2.0 * theta * K
                - 3.0 * p.kappa1 * (1.0 + p.kappa2) * theta
                - 3.0 * p.cosmological_constant
            )
            + adv_K
            + 0.5 * p.kappa_gravity * alpha * (stress_trace - 3.0 * rho)
        )
        At_squared = np.einsum(
            "ij...,ik...,jl...,kl...->...", At, gt_inv, gt_inv, At
        )
        adv_theta = sum(
            beta[k] * self._derivative_or_zero(theta, k) for k in range(3)
        )
        Z_grad_alpha = sum(
            Z_up_physical[i] * self._gradient_component(alpha, i) for i in range(3)
        )
        dtheta = (
            0.5
            * alpha
            * (
                scalar_ricci
                + 2.0 * div_Z
                - At_squared
                + (2.0 / 3.0) * K**2
                - 2.0 * theta * K
                - 2.0 * p.cosmological_constant
            )
            - Z_grad_alpha
            + adv_theta
            - alpha * p.kappa1 * (2.0 + p.kappa2) * theta
            - p.kappa_gravity * alpha * rho
        )

        At_up = np.einsum("ik...,jl...,kl...->ij...", gt_inv, gt_inv, At)
        dgamma = self.grid.zeros((3,))
        gradient_phi = self.grid.gradient(phi)
        gradient_K = self.grid.gradient(K)
        gradient_theta = self.grid.gradient(theta)
        gradient_alpha = self.grid.gradient(alpha)
        Z_tilde_up = np.einsum("ij...,j...->i...", gt_inv, Z_down)
        for i in range(3):
            first = np.zeros(self.grid.shape)
            for j in range(3):
                for k in range(3):
                    first += gt_gamma[i, j, k] * At_up[j, k]
                first -= 3.0 * At_up[i, j] * gradient_phi[j] / phi
                first -= (2.0 / 3.0) * gt_inv[i, j] * gradient_K[j]
            dgamma[i] = 2.0 * alpha * first
            for k in range(3):
                dgamma[i] += 2.0 * gt_inv[k, i] * (
                    alpha * gradient_theta[k]
                    - theta * gradient_alpha[k]
                    - (2.0 / 3.0) * alpha * K * Z_down[k]
                )
                dgamma[i] -= 2.0 * At_up[i, k] * gradient_alpha[k]
                for ell in range(3):
                    dgamma[i] += gt_inv[k, ell] * self._derivative_or_zero(
                        self._gradient_component(beta[i], ell), k
                    )
                dgamma[i] += (1.0 / 3.0) * gt_inv[i, k] * self._derivative_or_zero(
                    beta_div, k
                )
                dgamma[i] -= gamma_tilde[k] * self._gradient_component(beta[i], k)
                dgamma[i] += beta[k] * self._derivative_or_zero(gamma_hat[i], k)
                dgamma[i] += 2.0 * p.kappa3 * (
                    (2.0 / 3.0) * Z_tilde_up[i] * self._gradient_component(beta[k], k)
                    - Z_tilde_up[k] * self._gradient_component(beta[i], k)
                )
            dgamma[i] += (2.0 / 3.0) * gamma_tilde[i] * beta_div
            dgamma[i] -= 2.0 * alpha * p.kappa1 * Z_tilde_up[i]
            dgamma[i] -= 2.0 * p.kappa_gravity * alpha * np.einsum(
                "j...,j...->...", gt_inv[i], momentum
            )

        adv_alpha = sum(
            beta[k] * self._derivative_or_zero(alpha, k) for k in range(3)
        )
        if p.slicing == "harmonic":
            dalpha = -(alpha**2) * (K - 2.0 * theta) + adv_alpha
        else:
            dalpha = -2.0 * alpha * (K - 2.0 * theta) + adv_alpha
        if p.evolve_shift:
            dbeta = p.shift_speed * driver.copy()
            ddriver = dgamma.copy()
            for i in range(3):
                for k in range(3):
                    dbeta[i] += beta[k] * self._derivative_or_zero(beta[i], k)
                    ddriver[i] -= beta[k] * self._derivative_or_zero(gamma_hat[i], k)
                    ddriver[i] += beta[k] * self._derivative_or_zero(driver[i], k)
                ddriver[i] -= p.eta_shift * driver[i]
        else:
            dbeta = np.zeros_like(beta)
            ddriver = np.zeros_like(driver)

        result = (dgt, dAt, dphi, dK, dtheta, dgamma, dalpha, dbeta, ddriver)
        if not getattr(self.grid, "is_periodic", True) and p.radiative_boundaries:
            from .boundary import RadiativeBoundary

            boundary_state = CCZ4State(
                gt, At, phi, K, theta, gamma_hat, alpha, beta, driver
            )
            result = RadiativeBoundary(self.grid).ccz4_rhs(boundary_state, result)
        return result

    def step(self, state: CCZ4State, dt: float, matter: Matter | None = None) -> CCZ4State:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if matter is None:
            matter = (
                self.grid.zeros(),
                self.grid.zeros((3,)),
                self.grid.zeros((3, 3)),
            )
        values = (
            state.conformal_metric,
            state.conformal_A,
            state.conformal_factor,
            state.trace_K,
            state.theta,
            state.gamma_hat,
            state.lapse,
            state.shift,
            state.shift_driver,
        )
        evolved = rk4_arrays(
            values,
            state.time,
            dt,
            lambda time, stage: self.rhs(time, stage, matter),
        )
        result = CCZ4State(*evolved, state.time + dt)
        return self.project_algebraic(result)

    def constraints(self, state: CCZ4State, matter: Matter | None = None) -> tuple[Array, Array]:
        if matter is None:
            matter = (
                self.grid.zeros(),
                self.grid.zeros((3,)),
                self.grid.zeros((3, 3)),
            )
        from .adm import ADMSolver, ADMParameters

        adm = ADMSolver(
            self.grid,
            ADMParameters(
                self.parameters.kappa_gravity, self.parameters.cosmological_constant
            ),
        )
        return adm.constraints(self.to_adm(state), matter)

    def diagnostics(self, state: CCZ4State, matter: Matter | None = None) -> CCZ4Diagnostics:
        hamiltonian, momentum = self.constraints(state, matter)
        gt_inv, _, gamma_tilde = self.conformal_connection(state.conformal_metric)
        Z_down = self.z_covector(state.conformal_metric, state.gamma_hat, gamma_tilde)
        z_sq = np.einsum("ij...,i...,j...->...", gt_inv, Z_down, Z_down)
        matrices = np.moveaxis(state.conformal_metric, (0, 1), (-2, -1))
        determinant = np.linalg.det(matrices)
        eigenvalues = np.linalg.eigvalsh(matrices)
        trace_A = np.einsum(
            "ij...,ij...->...", gt_inv, state.conformal_A
        )
        return CCZ4Diagnostics(
            time=state.time,
            hamiltonian_l2=float(np.sqrt(np.mean(hamiltonian**2))),
            momentum_l2=float(np.sqrt(np.mean(np.sum(momentum**2, axis=0)))),
            theta_l2=float(np.sqrt(np.mean(state.theta**2))),
            z_l2=float(np.sqrt(np.mean(np.maximum(z_sq, 0.0)))),
            determinant_linf=float(np.max(np.abs(determinant - 1.0))),
            trace_A_linf=float(np.max(np.abs(trace_A))),
            minimum_lapse=float(np.min(state.lapse)),
            minimum_metric_eigenvalue=float(np.min(eigenvalues)),
        )
