"""Conformal transverse-traceless initial-data solver on isolated patches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .adm import ADMState
from .grid import Array, CartesianGrid, PeriodicGrid


@dataclass(frozen=True)
class CTTParameters:
    kappa: float = 8.0 * np.pi
    maximum_iterations: int = 8000
    tolerance: float = 2.0e-7
    relaxation: float = 0.35
    minimum_conformal_factor: float = 0.1
    maximum_conformal_factor: float = 20.0
    report_interval: int = 50

    def __post_init__(self) -> None:
        if self.kappa <= 0.0 or self.maximum_iterations < 1:
            raise ValueError("kappa and maximum_iterations must be positive")
        if self.tolerance <= 0.0 or not 0.0 < self.relaxation <= 1.0:
            raise ValueError("invalid CTT tolerance or relaxation")
        if self.maximum_conformal_factor <= self.minimum_conformal_factor:
            raise ValueError("maximum conformal factor must exceed the minimum")


@dataclass(frozen=True)
class CTTReport:
    converged: bool
    iterations: int
    hamiltonian_residual_l2: float
    momentum_residual_l2: float
    minimum_conformal_factor: float


@dataclass
class CTTResult:
    geometry: ADMState
    conformal_factor: Array
    vector_potential: Array
    conformal_extrinsic_curvature: Array
    report: CTTReport


@dataclass(frozen=True)
class PeriodicCMCReport:
    converged: bool
    iterations: int
    hamiltonian_residual_l2: float
    momentum_residual_l2: float
    trace_K: float
    minimum_conformal_factor: float
    maximum_conformal_factor: float


@dataclass
class PeriodicCMCResult:
    geometry: ADMState
    conformal_factor: Array
    vector_potential: Array
    conformal_extrinsic_curvature: Array
    report: PeriodicCMCReport


class PeriodicCMCInitialDataSolver:
    """Conformally flat periodic CMC data in one, two, or three dimensions.

    A compact positive-energy source cannot satisfy the periodic Hamiltonian
    constraint on a maximal slice.  The constant mean curvature supplies the
    required zero mode.  Its magnitude is updated from the integrated
    Lichnerowicz equation while the mean conformal factor and vector potential
    are fixed to one and zero, respectively.
    """

    def __init__(
        self,
        grid: PeriodicGrid,
        parameters: CTTParameters | None = None,
        *,
        expanding: bool = True,
    ) -> None:
        self.grid = grid
        self.parameters = parameters or CTTParameters()
        self.expanding = expanding

    def longitudinal(self, vector: Array) -> Array:
        divergence = self.grid.divergence(vector)
        result = self.grid.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                if j < self.grid.ndim:
                    result[i, j] += self.grid.derivative(vector[i], j)
                if i < self.grid.ndim:
                    result[i, j] += self.grid.derivative(vector[j], i)
                if i == j:
                    result[i, j] -= (2.0 / 3.0) * divergence
        return result

    def vector_laplacian(self, vector: Array) -> Array:
        return self.grid.laplacian(vector) + self.grid.gradient(
            self.grid.divergence(vector)
        ) / 3.0

    def residuals(
        self, psi: Array, vector: Array, rho: Array, momentum_lower: Array
    ) -> tuple[Array, Array, Array, float]:
        Abar = self.longitudinal(vector)
        Abar_squared = np.einsum("ij...,ij...->...", Abar, Abar)
        psi5 = psi**5
        denominator = max(float(np.mean(psi5)), 1.0e-300)
        trace_K_squared = (
            1.5 * float(np.mean(Abar_squared * psi ** -7))
            + 3.0 * self.parameters.kappa * float(np.mean(rho * psi5))
        ) / denominator
        trace_K = (-1.0 if self.expanding else 1.0) * np.sqrt(
            max(trace_K_squared, 0.0)
        )
        hamiltonian = (
            self.grid.laplacian(psi)
            + 0.125 * Abar_squared * psi ** -7
            - (trace_K_squared / 12.0) * psi5
            + 0.25 * self.parameters.kappa * rho * psi5
        )
        momentum = self.vector_laplacian(vector) - (
            self.parameters.kappa * psi**6
        )[None, ...] * momentum_lower
        return hamiltonian, momentum, Abar, trace_K

    def solve(
        self,
        rho: Array,
        momentum_lower: Array,
        conformal_factor: Array | None = None,
        vector_potential: Array | None = None,
    ) -> PeriodicCMCResult:
        rho = np.asarray(rho, dtype=float)
        momentum_lower = np.asarray(momentum_lower, dtype=float)
        if rho.shape != self.grid.shape or momentum_lower.shape != (3,) + self.grid.shape:
            raise ValueError("periodic CMC matter source shape does not match the grid")
        if np.any(rho < 0.0):
            raise ValueError("periodic CMC energy density must be nonnegative")
        source_mean = np.mean(momentum_lower, axis=tuple(range(1, momentum_lower.ndim)))
        source_scale = max(float(np.sqrt(np.mean(momentum_lower**2))), 1.0e-300)
        if float(np.max(np.abs(source_mean))) > max(
            self.parameters.tolerance, 1.0e-10 * source_scale
        ):
            raise ValueError("periodic CMC momentum source must have zero spatial mean")

        psi = (
            np.ones(self.grid.shape)
            if conformal_factor is None
            else np.array(conformal_factor, dtype=float, copy=True)
        )
        vector = (
            self.grid.zeros((3,))
            if vector_potential is None
            else np.array(vector_potential, dtype=float, copy=True)
        )
        psi /= max(float(np.mean(psi)), 1.0e-300)
        vector -= np.mean(
            vector, axis=tuple(range(1, vector.ndim)), keepdims=True
        )
        p = self.parameters
        dt = p.relaxation * min(self.grid.spacing) ** 2 / (2.0 * self.grid.ndim)
        converged = False
        h_norm = m_norm = np.inf
        trace_K = 0.0
        for iteration in range(1, p.maximum_iterations + 1):
            h_residual, m_residual, _, trace_K = self.residuals(
                psi, vector, rho, momentum_lower
            )
            vector += dt * m_residual
            vector -= np.mean(
                vector, axis=tuple(range(1, vector.ndim)), keepdims=True
            )
            psi += dt * h_residual
            np.maximum(psi, p.minimum_conformal_factor, out=psi)
            psi /= max(float(np.mean(psi)), 1.0e-300)
            if (
                not np.all(np.isfinite(psi))
                or float(np.max(psi)) > p.maximum_conformal_factor
            ):
                break
            if iteration == 1 or iteration % p.report_interval == 0:
                h_residual, m_residual, _, trace_K = self.residuals(
                    psi, vector, rho, momentum_lower
                )
                h_norm = float(np.sqrt(np.mean(h_residual**2)))
                m_norm = float(np.sqrt(np.mean(np.sum(m_residual**2, axis=0))))
                if max(h_norm, m_norm) <= p.tolerance:
                    converged = True
                    break

        h_residual, m_residual, Abar, trace_K = self.residuals(
            psi, vector, rho, momentum_lower
        )
        h_norm = float(np.sqrt(np.mean(h_residual**2)))
        m_norm = float(np.sqrt(np.mean(np.sum(m_residual**2, axis=0))))
        h = self.grid.zeros((3, 3))
        for i in range(3):
            h[i, i] = psi**4
        K = psi[None, None, ...] ** -2 * Abar + (
            trace_K / 3.0
        ) * h
        report = PeriodicCMCReport(
            converged,
            iteration,
            h_norm,
            m_norm,
            trace_K,
            float(np.min(psi)),
            float(np.max(psi)),
        )
        return PeriodicCMCResult(
            ADMState(h, K), psi, vector, Abar, report
        )


class CTTInitialDataSolver:
    """Solve maximal, conformally-flat CTT data with isolated outer data.

    The supplied ``rho`` and covariant ``momentum`` are physical 3+1 matter
    projections.  At a finite outer box we impose psi=1 and W^i=0, the leading
    asymptotically-flat values.  The equations solved are

      Delta psi + Abar^2 psi^-7 / 8 + kappa rho psi^5 / 4 = 0,
      Delta_L W^i = kappa psi^6 S_i.

    The psi^6 factor converts a physical lower-index momentum source to the
    flat conformal vector equation.
    """

    def __init__(self, grid: CartesianGrid, parameters: CTTParameters | None = None) -> None:
        if grid.ndim != 3:
            raise ValueError("the CTT solver requires a three-dimensional Cartesian grid")
        self.grid = grid
        self.parameters = parameters or CTTParameters()
        self._interior = tuple(slice(1, -1) for _ in range(3))
        self._interior_vector = (slice(None),) + self._interior

    def _set_outer(self, field: Array, value: float) -> None:
        leading = field.ndim - 3
        for axis in range(3):
            array_axis = leading + axis
            lower = [slice(None)] * field.ndim
            upper = [slice(None)] * field.ndim
            lower[array_axis] = 0
            upper[array_axis] = -1
            field[tuple(lower)] = value
            field[tuple(upper)] = value

    def longitudinal(self, vector: Array) -> Array:
        divergence = self.grid.divergence(vector)
        result = self.grid.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                result[i, j] = self.grid.derivative(vector[j], i)
                result[i, j] += self.grid.derivative(vector[i], j)
                if i == j:
                    result[i, j] -= (2.0 / 3.0) * divergence
        return result

    def vector_laplacian(self, vector: Array) -> Array:
        return self.grid.laplacian(vector) + self.grid.gradient(
            self.grid.divergence(vector)
        ) / 3.0

    def residuals(
        self, psi: Array, vector: Array, rho: Array, momentum_lower: Array
    ) -> tuple[Array, Array, Array]:
        Abar = self.longitudinal(vector)
        Abar_squared = np.einsum("ij...,ij...->...", Abar, Abar)
        hamiltonian = (
            self.grid.laplacian(psi)
            + 0.125 * Abar_squared * psi ** -7
            + 0.25 * self.parameters.kappa * rho * psi**5
        )
        momentum = self.vector_laplacian(vector) - (
            self.parameters.kappa * psi**6
        )[None, ...] * momentum_lower
        return hamiltonian, momentum, Abar

    def solve(
        self,
        rho: Array,
        momentum_lower: Array,
        conformal_factor: Array | None = None,
        vector_potential: Array | None = None,
    ) -> CTTResult:
        rho = np.asarray(rho, dtype=float)
        momentum_lower = np.asarray(momentum_lower, dtype=float)
        if rho.shape != self.grid.shape or momentum_lower.shape != (3,) + self.grid.shape:
            raise ValueError("CTT matter source shape does not match the grid")
        if np.any(rho < 0.0):
            raise ValueError("CTT energy density must be nonnegative")
        psi = np.ones(self.grid.shape) if conformal_factor is None else np.array(conformal_factor, dtype=float, copy=True)
        vector = self.grid.zeros((3,)) if vector_potential is None else np.array(vector_potential, dtype=float, copy=True)
        self._set_outer(psi, 1.0)
        self._set_outer(vector, 0.0)
        p = self.parameters
        dt = p.relaxation * min(self.grid.spacing) ** 2 / 6.0
        converged = False
        h_norm = m_norm = np.inf

        for iteration in range(1, p.maximum_iterations + 1):
            h_residual, m_residual, _ = self.residuals(psi, vector, rho, momentum_lower)
            # Diffusive pseudo-time has the elliptic operators as its stable part.
            vector[self._interior_vector] += dt * m_residual[self._interior_vector]
            psi[self._interior] += dt * h_residual[self._interior]
            np.maximum(psi, p.minimum_conformal_factor, out=psi)
            if not np.all(np.isfinite(psi)) or float(np.max(psi)) > p.maximum_conformal_factor:
                # A Lichnerowicz solve may genuinely have no regular solution for
                # overly compact freely specified physical matter.  Return a
                # controlled nonconvergence report instead of propagating NaNs.
                np.nan_to_num(
                    psi,
                    copy=False,
                    nan=p.maximum_conformal_factor,
                    posinf=p.maximum_conformal_factor,
                    neginf=p.minimum_conformal_factor,
                )
                np.clip(
                    psi,
                    p.minimum_conformal_factor,
                    p.maximum_conformal_factor,
                    out=psi,
                )
                break
            self._set_outer(psi, 1.0)
            self._set_outer(vector, 0.0)
            if iteration == 1 or iteration % p.report_interval == 0:
                h_residual, m_residual, _ = self.residuals(psi, vector, rho, momentum_lower)
                h_norm = float(np.sqrt(np.mean(h_residual[self._interior] ** 2)))
                m_core = m_residual[self._interior_vector]
                m_norm = float(np.sqrt(np.mean(np.sum(m_core**2, axis=0))))
                if max(h_norm, m_norm) <= p.tolerance:
                    converged = True
                    break

        h_residual, m_residual, Abar = self.residuals(psi, vector, rho, momentum_lower)
        h_norm = float(np.sqrt(np.mean(h_residual[self._interior] ** 2)))
        m_core = m_residual[self._interior_vector]
        m_norm = float(np.sqrt(np.mean(np.sum(m_core**2, axis=0))))
        h = self.grid.zeros((3, 3))
        K = self.grid.zeros((3, 3))
        for i in range(3):
            h[i, i] = psi**4
        K[:] = psi[None, None, ...] ** -2 * Abar
        report = CTTReport(
            converged, iteration, h_norm, m_norm, float(np.min(psi))
        )
        return CTTResult(ADMState(h, K), psi, vector, Abar, report)
