"""Spherical Tolman--Oppenheimer--Volkoff equilibrium generator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class PolytropicEOS:
    K: float = 100.0
    gamma: float = 2.0

    def __post_init__(self) -> None:
        if self.K <= 0.0 or self.gamma <= 1.0:
            raise ValueError("polytropic K must be positive and gamma exceed one")

    def pressure(self, density: Array | float) -> Array:
        return self.K * np.asarray(density, dtype=float) ** self.gamma

    def density_from_pressure(self, pressure: Array | float) -> Array:
        return np.maximum(np.asarray(pressure, dtype=float), 0.0) ** (1.0 / self.gamma) / self.K ** (1.0 / self.gamma)

    def energy_density_from_pressure(self, pressure: Array | float) -> Array:
        pressure = np.maximum(np.asarray(pressure, dtype=float), 0.0)
        density = self.density_from_pressure(pressure)
        return density + pressure / (self.gamma - 1.0)


@dataclass(frozen=True)
class TOVSolution:
    radius: Array
    density: Array
    pressure: Array
    energy_density: Array
    enclosed_mass: Array
    lapse: Array
    radial_metric: Array
    surface_radius: float
    gravitational_mass: float
    baryonic_mass: float


def solve_tov(
    central_density: float,
    eos: PolytropicEOS | None = None,
    *,
    dr: float = 0.01,
    pressure_floor_fraction: float = 1.0e-12,
    maximum_radius: float = 100.0,
) -> TOVSolution:
    """Integrate a cold spherical star in areal-radius coordinates, G=c=1."""
    eos = eos or PolytropicEOS()
    if central_density <= 0.0 or dr <= 0.0 or maximum_radius <= dr:
        raise ValueError("TOV integration controls are invalid")
    central_pressure = float(eos.pressure(central_density))
    pressure_floor = pressure_floor_fraction * central_pressure
    central_energy = float(eos.energy_density_from_pressure(central_pressure))
    r = 0.5 * dr
    mass = (4.0 * np.pi / 3.0) * central_energy * r**3
    baryon_mass = (4.0 * np.pi / 3.0) * central_density * r**3
    pressure = central_pressure
    nu = 0.0
    rows: list[tuple[float, float, float, float, float]] = []

    def derivatives(radius: float, values: np.ndarray) -> np.ndarray:
        m, p, mb, potential = values
        p = max(float(p), 0.0)
        density = float(eos.density_from_pressure(p))
        energy = float(eos.energy_density_from_pressure(p))
        compact = max(radius * (radius - 2.0 * m), 1.0e-300)
        gravity = (m + 4.0 * np.pi * radius**3 * p) / compact
        metric_factor = max(1.0 - 2.0 * m / radius, 1.0e-14)
        return np.array(
            [
                4.0 * np.pi * radius**2 * energy,
                -(energy + p) * gravity,
                4.0 * np.pi * radius**2 * density / np.sqrt(metric_factor),
                2.0 * gravity,
            ]
        )

    while pressure > pressure_floor and r < maximum_radius and r > 2.0 * mass:
        density = float(eos.density_from_pressure(pressure))
        energy = float(eos.energy_density_from_pressure(pressure))
        rows.append((r, density, pressure, energy, mass))
        y = np.array([mass, pressure, baryon_mass, nu])
        k1 = derivatives(r, y)
        k2 = derivatives(r + 0.5 * dr, y + 0.5 * dr * k1)
        k3 = derivatives(r + 0.5 * dr, y + 0.5 * dr * k2)
        k4 = derivatives(r + dr, y + dr * k3)
        y = y + (dr / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        mass, pressure, baryon_mass, nu = map(float, y)
        r += dr
    if not rows or r >= maximum_radius or r <= 2.0 * mass:
        raise FloatingPointError("TOV integration did not reach a regular surface")
    radius, density, pressures, energy, masses = map(np.asarray, zip(*rows))
    surface_radius = float(radius[-1])
    gravitational_mass = float(masses[-1])
    # Shift nu so the interior lapse matches Schwarzschild at the surface.
    nu_surface_target = np.log(1.0 - 2.0 * gravitational_mass / surface_radius)
    # Reintegrate nu samples by cumulative trapezoid from the stored profiles.
    nu_gradient = np.zeros_like(radius)
    for index, (rr, pp, mm) in enumerate(zip(radius, pressures, masses)):
        nu_gradient[index] = 2.0 * (mm + 4.0 * np.pi * rr**3 * pp) / (
            rr * (rr - 2.0 * mm)
        )
    nu_profile = np.zeros_like(radius)
    if radius.size > 1:
        nu_profile[1:] = np.cumsum(
            0.5 * (nu_gradient[1:] + nu_gradient[:-1]) * np.diff(radius)
        )
    nu_profile += nu_surface_target - nu_profile[-1]
    lapse = np.exp(0.5 * nu_profile)
    radial_metric = 1.0 / (1.0 - 2.0 * masses / radius)
    return TOVSolution(
        radius,
        density,
        pressures,
        energy,
        masses,
        lapse,
        radial_metric,
        surface_radius,
        gravitational_mass,
        float(baryon_mass),
    )
