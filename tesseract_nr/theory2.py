"""Source, target, and interaction interfaces fixed by Theory 2.0."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .adm import inverse_metric
from .grid import Array, PeriodicGrid
from .integrators import rk4_arrays
from .matter import FluidPrimitive, IdealGasEOS, perfect_fluid_stress_energy


@dataclass(frozen=True)
class Theory2Parameters:
    Z: float = 1.0
    mass: float = 0.5
    coupling: float = 0.1
    tau_star: float = 1.0
    ell_sigma: float = 1.0
    gamma_ad: float = 5.0 / 3.0

    def __post_init__(self) -> None:
        if self.Z <= 0.0 or self.mass <= 0.0 or self.tau_star <= 0.0:
            raise ValueError("Z, mass, and tau_star must be positive")
        if self.coupling < 0.0 or self.ell_sigma < 0.0:
            raise ValueError("coupling and ell_sigma must be nonnegative")
        if self.gamma_ad <= 1.0:
            raise ValueError("gamma_ad must exceed one")


@dataclass(frozen=True)
class TargetData:
    Aeq_space: Array
    Phi_eq: Array
    q: Array
    Phi_star: Array
    target_rhs: Array
    target_constraint: Array


@dataclass(frozen=True)
class SourceData:
    effective_mass_squared: Array
    J_star_normal_density: Array
    J_star_space_lower: Array
    rho_star: Array
    target_coupling: Array


@dataclass(frozen=True)
class StressEnergy3p1:
    rho: Array
    momentum: Array
    stress: Array
    trace: Array


def asinh_over_x(x: Array | float) -> Array:
    x = np.asarray(x, dtype=float)
    x2 = x * x
    series = 1.0 - x2 / 6.0 + 3.0 * x2 * x2 / 40.0 - 5.0 * x2**3 / 112.0
    safe = np.where(np.abs(x) < 1.0e-8, 1.0, x)
    return np.where(np.abs(x) < 1.0e-4, series, np.arcsinh(x) / safe)


def phi_star(velocity: Array, astar_lower: Array) -> Array:
    return np.einsum("i...,i...->...", velocity, astar_lower)


def equilibrium_target(
    grid: PeriodicGrid,
    h: Array,
    fluid: FluidPrimitive,
    ell_sigma: float = 1.0,
) -> tuple[Array, Array, Array]:
    """Construct Aeq from an advected scalar using only slice data.

    Advection supplies ``n^mu grad_mu sigma = -v^i D_i sigma``. Therefore the
    spatial covector is ``Q_i = ell D_i sigma`` and
    ``q^2 = Q_i Q^i - (v^i Q_i)^2``.
    """
    h_inv, _, _ = inverse_metric(h)
    Q = ell_sigma * grid.gradient(fluid.sigma)
    Q_up = np.einsum("ij...,j...->i...", h_inv, Q)
    Q_phi = np.einsum("i...,i...->...", fluid.velocity, Q)
    q_squared = np.einsum("i...,i...->...", Q, Q_up) - Q_phi**2
    if np.any(q_squared < -1.0e-10):
        raise FloatingPointError("projected sigma gradient became timelike")
    q = np.sqrt(np.maximum(q_squared, 0.0))
    factor = asinh_over_x(q)
    return factor[None, ...] * Q, factor * Q_phi, q


def target_data(
    grid: PeriodicGrid,
    h: Array,
    fluid: FluidPrimitive,
    astar_lower: Array,
    parameters: Theory2Parameters,
) -> TargetData:
    Aeq, Phi_eq, q = equilibrium_target(grid, h, fluid, parameters.ell_sigma)
    Phi_star = phi_star(fluid.velocity, astar_lower)
    rhs = -(astar_lower - Aeq) / parameters.tau_star
    # Reconstruction makes the continuum orthogonality residual identically zero.
    constraint = Phi_star - np.einsum("i...,i...->...", fluid.velocity, astar_lower)
    return TargetData(Aeq, Phi_eq, q, Phi_star, rhs, constraint)


def source_data(
    h: Array,
    fluid: FluidPrimitive,
    astar_lower: Array,
    parameters: Theory2Parameters,
) -> SourceData:
    h_inv, _, _ = inverse_metric(h)
    coupling = parameters.coupling * fluid.baryon_density
    Phi_star = phi_star(fluid.velocity, astar_lower)
    astar_up = np.einsum("ij...,j...->i...", h_inv, astar_lower)
    coefficient = coupling / parameters.Z
    return SourceData(
        effective_mass_squared=parameters.mass**2 + coefficient,
        J_star_normal_density=coefficient * Phi_star,
        J_star_space_lower=coefficient[None, ...] * astar_lower,
        rho_star=coefficient * Phi_star,
        target_coupling=coupling,
    )


def interaction_stress_energy(
    h: Array,
    fluid: FluidPrimitive,
    a_lower: Array,
    phi: Array,
    astar_lower: Array,
    parameters: Theory2Parameters,
) -> tuple[Array, Array, Array]:
    h_inv, _, _ = inverse_metric(h)
    w = a_lower - astar_lower
    W_phi = phi - phi_star(fluid.velocity, astar_lower)
    w_squared = np.einsum("ij...,i...,j...->...", h_inv, w, w)
    coefficient = parameters.coupling * fluid.baryon_density
    rho = 0.5 * coefficient * (W_phi**2 + w_squared)
    momentum = coefficient[None, ...] * W_phi[None, ...] * w
    stress = coefficient[None, None, ...] * (
        w[:, None, ...] * w[None, :, ...]
        + 0.5 * h * (W_phi**2 - w_squared)[None, None, ...]
    )
    return rho, momentum, stress


def combine_stress_energy(h: Array, *parts: tuple[Array, Array, Array]) -> StressEnergy3p1:
    if not parts:
        raise ValueError("at least one stress-energy contribution is required")
    rho = sum((part[0] for part in parts), np.zeros_like(parts[0][0]))
    momentum = sum((part[1] for part in parts), np.zeros_like(parts[0][1]))
    stress = sum((part[2] for part in parts), np.zeros_like(parts[0][2]))
    h_inv, _, _ = inverse_metric(h)
    trace = np.einsum("ij...,ij...->...", h_inv, stress)
    return StressEnergy3p1(rho, momentum, stress, trace)


class TargetEvolution:
    """Specified Cartesian advection-relaxation implementation for ``a*_i``.

    The full covariant projected derivative contains acceleration/connection
    terms not expanded in Theory 2.0. This class implements the document's
    explicit interface RHS plus coordinate advection.
    """

    def __init__(self, grid: PeriodicGrid, parameters: Theory2Parameters) -> None:
        self.grid = grid
        self.parameters = parameters

    def rhs(self, time: float, values: tuple[Array], h: Array, fluid: FluidPrimitive) -> tuple[Array]:
        del time
        (astar,) = values
        data = target_data(self.grid, h, fluid, astar, self.parameters)
        rhs = data.target_rhs.copy()
        for axis in range(self.grid.ndim):
            rhs -= fluid.velocity[axis][None, ...] * self.grid.derivative(astar, axis)
        return (rhs,)

    def step(self, astar: Array, time: float, dt: float, h: Array, fluid: FluidPrimitive) -> Array:
        return rk4_arrays(
            (astar,), time, dt, lambda t, y: self.rhs(t, y, h, fluid)
        )[0]


def total_stress_energy(
    grid: PeriodicGrid,
    h: Array,
    fluid: FluidPrimitive,
    proca_part: tuple[Array, Array, Array],
    a_lower: Array,
    phi: Array,
    astar_lower: Array,
    parameters: Theory2Parameters,
) -> StressEnergy3p1:
    fluid_part = perfect_fluid_stress_energy(
        grid, h, fluid, IdealGasEOS(parameters.gamma_ad)
    )
    interaction_part = interaction_stress_energy(
        h, fluid, a_lower, phi, astar_lower, parameters
    )
    return combine_stress_energy(h, fluid_part, proca_part, interaction_part)
