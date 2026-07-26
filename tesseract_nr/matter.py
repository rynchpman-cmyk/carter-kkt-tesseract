"""Theory 2.0 ideal-fluid primitives and Eulerian stress-energy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .adm import inverse_metric
from .grid import Array, PeriodicGrid


@dataclass(frozen=True)
class IdealGasEOS:
    gamma_ad: float = 5.0 / 3.0

    def __post_init__(self) -> None:
        if self.gamma_ad <= 1.0:
            raise ValueError("gamma_ad must exceed one")

    def pressure(self, baryon_density: Array, specific_internal_energy: Array) -> Array:
        return (self.gamma_ad - 1.0) * baryon_density * specific_internal_energy

    def energy_density(self, baryon_density: Array, specific_internal_energy: Array) -> Array:
        return baryon_density * (1.0 + specific_internal_energy)

    def specific_enthalpy(self, specific_internal_energy: Array) -> Array:
        return 1.0 + self.gamma_ad * specific_internal_energy


@dataclass
class FluidPrimitive:
    baryon_density: Array
    specific_internal_energy: Array
    velocity: Array
    sigma: Array

    def copy(self) -> "FluidPrimitive":
        return FluidPrimitive(
            self.baryon_density.copy(),
            self.specific_internal_energy.copy(),
            self.velocity.copy(),
            self.sigma.copy(),
        )


@dataclass(frozen=True)
class FluidDiagnostics:
    minimum_density: float
    minimum_internal_energy: float
    maximum_speed: float
    normalization_linf: float
    baryon_number: float


def validate_fluid(grid: PeriodicGrid, h: Array, fluid: FluidPrimitive) -> None:
    if fluid.baryon_density.shape != grid.shape or fluid.specific_internal_energy.shape != grid.shape:
        raise ValueError("fluid scalar fields do not match the grid")
    if fluid.sigma.shape != grid.shape or fluid.velocity.shape != (3,) + grid.shape:
        raise ValueError("fluid sigma or velocity does not match the grid")
    if np.any(fluid.baryon_density < 0.0) or np.any(fluid.specific_internal_energy < 0.0):
        raise ValueError("fluid density and internal energy must be nonnegative")
    speed_sq = np.einsum("ij...,i...,j...->...", h, fluid.velocity, fluid.velocity)
    if np.any(speed_sq >= 1.0):
        raise ValueError("fluid velocity must be subluminal")


def lorentz_factor(h: Array, velocity: Array) -> Array:
    speed_sq = np.einsum("ij...,i...,j...->...", h, velocity, velocity)
    if np.any(speed_sq >= 1.0):
        raise ValueError("fluid velocity must be subluminal")
    return 1.0 / np.sqrt(1.0 - np.maximum(speed_sq, 0.0))


def perfect_fluid_stress_energy(
    grid: PeriodicGrid,
    h: Array,
    fluid: FluidPrimitive,
    eos: IdealGasEOS | None = None,
) -> tuple[Array, Array, Array]:
    """Return the Theory 2.0 perfect-fluid ``(rho, S_i, S_ij)``."""
    eos = eos or IdealGasEOS()
    validate_fluid(grid, h, fluid)
    pressure = eos.pressure(fluid.baryon_density, fluid.specific_internal_energy)
    energy = eos.energy_density(fluid.baryon_density, fluid.specific_internal_energy)
    W = lorentz_factor(h, fluid.velocity)
    velocity_lower = np.einsum("ij...,j...->i...", h, fluid.velocity)
    prefactor = (energy + pressure) * W**2
    rho = prefactor - pressure
    momentum = prefactor[None, ...] * velocity_lower
    stress = prefactor[None, None, ...] * (
        velocity_lower[:, None, ...] * velocity_lower[None, :, ...]
    ) + pressure[None, None, ...] * h
    return rho, momentum, stress


def fluid_diagnostics(
    grid: PeriodicGrid, h: Array, fluid: FluidPrimitive
) -> FluidDiagnostics:
    validate_fluid(grid, h, fluid)
    _, _, sqrt_h = inverse_metric(h)
    speed_sq = np.einsum("ij...,i...,j...->...", h, fluid.velocity, fluid.velocity)
    W = 1.0 / np.sqrt(1.0 - speed_sq)
    normalization = -W**2 + W**2 * speed_sq + 1.0
    return FluidDiagnostics(
        minimum_density=float(np.min(fluid.baryon_density)),
        minimum_internal_energy=float(np.min(fluid.specific_internal_energy)),
        maximum_speed=float(np.sqrt(np.max(speed_sq))),
        normalization_linf=float(np.max(np.abs(normalization))),
        baryon_number=grid.integrate(sqrt_h * fluid.baryon_density * W),
    )
