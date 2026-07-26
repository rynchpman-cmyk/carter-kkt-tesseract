"""Reduced-Hamiltonian Proca evolution on a periodic Minkowski slice."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .grid import Array, PeriodicGrid
from .integrators import rk4_arrays

Source = Callable[[float], tuple[Array, Array]]


@dataclass(frozen=True)
class ProcaParameters:
    Z: float = 1.0
    mass: float = 1.0

    def __post_init__(self) -> None:
        if self.Z <= 0.0:
            raise ValueError("Z must be positive")
        if self.mass <= 0.0:
            raise ValueError("mass must be positive; the reduced system is singular at m=0")


@dataclass
class ProcaState:
    a: Array
    pi: Array
    time: float = 0.0

    def copy(self) -> "ProcaState":
        return ProcaState(self.a.copy(), self.pi.copy(), self.time)


@dataclass(frozen=True)
class ProcaDiagnostics:
    time: float
    energy: float
    electric: float
    magnetic: float
    mass: float
    longitudinal: float
    lorenz_l2: float
    max_amplitude: float


class ProcaSolver:
    """Evolve all three massive-vector polarizations.

    The normal component is eliminated with

    ``Phi = (div(pi) + rho_source) / (Z m^2)``.

    External sources use the interaction ``A_mu J^mu`` and are supplied as
    ``(rho_source, spatial_current)``. Energy conservation diagnostics are
    meaningful only for a vanishing source.
    """

    def __init__(
        self,
        grid: PeriodicGrid,
        parameters: ProcaParameters | None = None,
        source: Source | None = None,
    ) -> None:
        self.grid = grid
        self.parameters = parameters or ProcaParameters()
        self.source = source

    def zero_state(self) -> ProcaState:
        return ProcaState(self.grid.zeros((3,)), self.grid.zeros((3,)))

    def validate_state(self, state: ProcaState) -> None:
        expected = (3,) + self.grid.shape
        if state.a.shape != expected or state.pi.shape != expected:
            raise ValueError(f"a and pi must both have shape {expected}")
        if not np.all(np.isfinite(state.a)) or not np.all(np.isfinite(state.pi)):
            raise FloatingPointError("Proca state contains a non-finite value")

    def _source(self, time: float) -> tuple[Array, Array]:
        if self.source is None:
            return self.grid.zeros(), self.grid.zeros((3,))
        rho, current = self.source(time)
        rho = np.asarray(rho, dtype=float)
        current = np.asarray(current, dtype=float)
        if rho.shape != self.grid.shape or current.shape != (3,) + self.grid.shape:
            raise ValueError("source returned arrays with incompatible shapes")
        return rho, current

    def phi(self, state: ProcaState, time: float | None = None) -> Array:
        rho, _ = self._source(state.time if time is None else time)
        p = self.parameters
        return (self.grid.divergence(state.pi) + rho) / (p.Z * p.mass**2)

    def rhs(self, time: float, values: tuple[Array, Array]) -> tuple[Array, Array]:
        a, pi = values
        p = self.parameters
        rho, current = self._source(time)
        phi = (self.grid.divergence(pi) + rho) / (p.Z * p.mass**2)
        da = pi / p.Z - self.grid.gradient(phi)
        magnetic_operator = self.grid.laplacian(a) - self.grid.gradient(
            self.grid.divergence(a)
        )
        dpi = p.Z * magnetic_operator - p.Z * p.mass**2 * a + current
        return da, dpi

    def step(self, state: ProcaState, dt: float) -> ProcaState:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.validate_state(state)
        a, pi = rk4_arrays((state.a, state.pi), state.time, dt, self.rhs)
        result = ProcaState(a, pi, state.time + dt)
        self.validate_state(result)
        return result

    def recommended_dt(self, cfl: float = 0.15) -> float:
        if cfl <= 0.0:
            raise ValueError("cfl must be positive")
        spatial = cfl * min(self.grid.spacing) / np.sqrt(self.grid.ndim)
        mass = cfl / self.parameters.mass
        return min(spatial, mass)

    def energy_density(self, state: ProcaState) -> dict[str, Array]:
        p = self.parameters
        phi = self.phi(state)
        electric = np.sum(state.pi * state.pi, axis=0) / (2.0 * p.Z)
        magnetic_field = self.grid.curl(state.a)
        magnetic = 0.5 * p.Z * np.sum(magnetic_field * magnetic_field, axis=0)
        mass = 0.5 * p.Z * p.mass**2 * np.sum(state.a * state.a, axis=0)
        longitudinal = 0.5 * p.Z * p.mass**2 * phi * phi
        return {
            "electric": electric,
            "magnetic": magnetic,
            "mass": mass,
            "longitudinal": longitudinal,
            "total": electric + magnetic + mass + longitudinal,
        }

    def stress_energy_3p1(self, state: ProcaState) -> tuple[Array, Array, Array]:
        """Return Eulerian ``(rho, S_i, S_ij)`` in Minkowski coordinates."""
        p = self.parameters
        # The canonical momentum follows document 1's n^mu F_{mu i} convention.
        # Theory 2.0 instead defines the physical E_i = F_{i mu} n^mu, hence
        # E_i = -pi_i/Z on a Minkowski slice.
        electric = -state.pi / p.Z
        magnetic = self.grid.curl(state.a)
        phi = self.phi(state)
        e2 = np.sum(electric * electric, axis=0)
        b2 = np.sum(magnetic * magnetic, axis=0)
        a2 = np.sum(state.a * state.a, axis=0)
        rho = 0.5 * p.Z * (e2 + b2 + p.mass**2 * (phi * phi + a2))
        momentum = p.Z * np.cross(
            np.moveaxis(electric, 0, -1), np.moveaxis(magnetic, 0, -1)
        )
        momentum = np.moveaxis(momentum, -1, 0) + p.Z * p.mass**2 * phi * state.a

        stress = self.grid.zeros((3, 3))
        isotropic = 0.5 * (e2 + b2) - 0.5 * p.mass**2 * (a2 - phi * phi)
        for i in range(3):
            for j in range(3):
                stress[i, j] = p.Z * (
                    -electric[i] * electric[j]
                    - magnetic[i] * magnetic[j]
                    + p.mass**2 * state.a[i] * state.a[j]
                )
                if i == j:
                    stress[i, j] += p.Z * isotropic
        return rho, momentum, stress

    def diagnostics(self, state: ProcaState, previous: ProcaState | None = None) -> ProcaDiagnostics:
        pieces = self.energy_density(state)
        lorenz_l2 = float("nan")
        if previous is not None and state.time != previous.time:
            phi_dot = (self.phi(state) - self.phi(previous)) / (state.time - previous.time)
            residual = phi_dot + self.grid.divergence(state.a)
            lorenz_l2 = float(np.sqrt(np.mean(residual * residual)))
        return ProcaDiagnostics(
            time=state.time,
            energy=self.grid.integrate(pieces["total"]),
            electric=self.grid.integrate(pieces["electric"]),
            magnetic=self.grid.integrate(pieces["magnetic"]),
            mass=self.grid.integrate(pieces["mass"]),
            longitudinal=self.grid.integrate(pieces["longitudinal"]),
            lorenz_l2=lorenz_l2,
            max_amplitude=float(max(np.max(np.abs(state.a)), np.max(np.abs(state.pi)))),
        )
