"""Experimental raw-ADM geometry evolution and constraint diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .grid import Array, PeriodicGrid

Matter = tuple[Array, Array, Array]
GaugeScalar = Callable[[float, Array, Array], Array]
GaugeVector = Callable[[float, Array, Array], Array]


@dataclass(frozen=True)
class ADMParameters:
    kappa: float = 8.0 * np.pi
    cosmological_constant: float = 0.0

    def __post_init__(self) -> None:
        if self.kappa <= 0.0:
            raise ValueError("kappa must be positive")


@dataclass
class ADMState:
    h: Array
    K: Array
    time: float = 0.0


def flat_metric(grid: PeriodicGrid) -> Array:
    h = grid.zeros((3, 3))
    for i in range(3):
        h[i, i] = 1.0
    return h


def inverse_metric(h: Array) -> tuple[Array, Array, Array]:
    """Return inverse, determinant, and square-root determinant pointwise."""
    matrices = np.moveaxis(h, (0, 1), (-2, -1))
    determinant = np.linalg.det(matrices)
    if np.any(determinant <= 0.0):
        raise FloatingPointError("spatial metric lost positive definiteness")
    inverse = np.moveaxis(np.linalg.inv(matrices), (-2, -1), (0, 1))
    return inverse, determinant, np.sqrt(determinant)


def christoffel(grid: PeriodicGrid, h: Array, h_inv: Array | None = None) -> Array:
    h_inv = inverse_metric(h)[0] if h_inv is None else h_inv
    dh = grid.zeros((3, 3, 3))  # derivative axis, metric indices
    for axis in range(grid.ndim):
        dh[axis] = grid.derivative(h, axis)
    gamma = grid.zeros((3, 3, 3))
    for k in range(3):
        for i in range(3):
            for j in range(3):
                for ell in range(3):
                    gamma[k, i, j] += 0.5 * h_inv[k, ell] * (
                        dh[i, ell, j] + dh[j, ell, i] - dh[ell, i, j]
                    )
    return gamma


def ricci_tensor(
    grid: PeriodicGrid, h: Array, h_inv: Array | None = None, gamma: Array | None = None
) -> Array:
    h_inv = inverse_metric(h)[0] if h_inv is None else h_inv
    gamma = christoffel(grid, h, h_inv) if gamma is None else gamma
    ricci = grid.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if k < grid.ndim:
                    ricci[i, j] += grid.derivative(gamma[k, i, j], k)
                if j < grid.ndim:
                    ricci[i, j] -= grid.derivative(gamma[k, i, k], j)
                for ell in range(3):
                    ricci[i, j] += (
                        gamma[k, i, j] * gamma[ell, k, ell]
                        - gamma[k, i, ell] * gamma[ell, j, k]
                    )
    return 0.5 * (ricci + np.swapaxes(ricci, 0, 1))


def covariant_hessian(
    grid: PeriodicGrid, scalar: Array, gamma: Array
) -> Array:
    gradient = grid.gradient(scalar)
    result = grid.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i < grid.ndim and j < grid.ndim:
                result[i, j] = grid.derivative(gradient[j], i)
            for k in range(3):
                result[i, j] -= gamma[k, i, j] * gradient[k]
    return result


def lie_metric(grid: PeriodicGrid, tensor: Array, shift: Array) -> Array:
    result = grid.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(grid.ndim):
                result[i, j] += shift[k] * grid.derivative(tensor[i, j], k)
                result[i, j] += tensor[k, j] * grid.derivative(shift[k], i) if i < grid.ndim else 0.0
                result[i, j] += tensor[i, k] * grid.derivative(shift[k], j) if j < grid.ndim else 0.0
    return result


class ADMSolver:
    """Raw ADM evolution with prescribed lapse and shift.

    This module prioritizes transparent equations and diagnostics. Raw ADM is
    not a production-stable free-evolution formulation; long strong-field runs
    should replace it with BSSN or Z4c while retaining the matter interface.
    """

    def __init__(
        self,
        grid: PeriodicGrid,
        parameters: ADMParameters | None = None,
        lapse: GaugeScalar | None = None,
        shift: GaugeVector | None = None,
    ) -> None:
        self.grid = grid
        self.parameters = parameters or ADMParameters()
        self.lapse_function = lapse
        self.shift_function = shift

    def flat_state(self) -> ADMState:
        return ADMState(flat_metric(self.grid), self.grid.zeros((3, 3)))

    def gauge(self, time: float, h: Array, K: Array) -> tuple[Array, Array]:
        lapse = (
            np.ones(self.grid.shape)
            if self.lapse_function is None
            else np.asarray(self.lapse_function(time, h, K), dtype=float)
        )
        shift = (
            self.grid.zeros((3,))
            if self.shift_function is None
            else np.asarray(self.shift_function(time, h, K), dtype=float)
        )
        if lapse.shape != self.grid.shape or shift.shape != (3,) + self.grid.shape:
            raise ValueError("gauge function returned an incompatible shape")
        if np.any(lapse <= 0.0):
            raise FloatingPointError("lapse must remain positive")
        return lapse, shift

    def rhs(self, time: float, h: Array, K: Array, matter: Matter) -> tuple[Array, Array]:
        rho, momentum, stress = matter
        del momentum  # momentum enters constraints, not the K_ij evolution directly
        p = self.parameters
        h_inv, _, _ = inverse_metric(h)
        gamma = christoffel(self.grid, h, h_inv)
        ricci = ricci_tensor(self.grid, h, h_inv, gamma)
        lapse, shift = self.gauge(time, h, K)
        hessian_lapse = covariant_hessian(self.grid, lapse, gamma)
        K_up_one = np.einsum("ik...,kj...->ij...", h_inv, K)
        trace_K = np.einsum("ij...,ij...->...", h_inv, K)
        K_squared = np.einsum("ik...,kj...->ij...", K, K_up_one)
        trace_stress = np.einsum("ij...,ij...->...", h_inv, stress)

        dh = -2.0 * lapse[None, None, ...] * K + lie_metric(self.grid, h, shift)
        dK = -hessian_lapse + lie_metric(self.grid, K, shift)
        dK += lapse[None, None, ...] * (
            ricci
            + trace_K[None, None, ...] * K
            - 2.0 * K_squared
            - p.cosmological_constant * h
            + 0.5
            * p.kappa
            * (
                h * (trace_stress - rho)[None, None, ...]
                - 2.0 * stress
            )
        )
        return 0.5 * (dh + np.swapaxes(dh, 0, 1)), 0.5 * (
            dK + np.swapaxes(dK, 0, 1)
        )

    def constraints(self, state: ADMState, matter: Matter) -> tuple[Array, Array]:
        rho, momentum_lower, _ = matter
        p = self.parameters
        h_inv, _, _ = inverse_metric(state.h)
        gamma = christoffel(self.grid, state.h, h_inv)
        ricci = ricci_tensor(self.grid, state.h, h_inv, gamma)
        scalar_ricci = np.einsum("ij...,ij...->...", h_inv, ricci)
        trace_K = np.einsum("ij...,ij...->...", h_inv, state.K)
        K_up = np.einsum("ik...,jl...,kl...->ij...", h_inv, h_inv, state.K)
        KijKij = np.einsum("ij...,ij...->...", state.K, K_up)
        hamiltonian = (
            scalar_ricci
            + trace_K**2
            - KijKij
            - 2.0 * p.cosmological_constant
            - 2.0 * p.kappa * rho
        )

        tensor = K_up - h_inv * trace_K[None, None, ...]
        momentum = self.grid.zeros((3,))
        for i in range(3):
            for j in range(3):
                if j < self.grid.ndim:
                    momentum[i] += self.grid.derivative(tensor[i, j], j)
                for k in range(3):
                    momentum[i] += gamma[i, j, k] * tensor[k, j]
                    momentum[i] += gamma[j, j, k] * tensor[i, k]
        momentum_up = np.einsum("ij...,j...->i...", h_inv, momentum_lower)
        momentum -= p.kappa * momentum_up
        return hamiltonian, momentum
