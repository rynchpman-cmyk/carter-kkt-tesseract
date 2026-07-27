"""Closed fixed-geometry driver for the deterministic Theory 2.0 sector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .adm import flat_metric
from .causal_map import causal_current, causal_margin, project_spatial
from .curved_proca import Theory2ProcaSystem
from .grid import Array, PeriodicGrid
from .integrators import rk4_arrays
from .matter import FluidPrimitive, fluid_diagnostics
from .theory2 import (
    StressEnergy3p1,
    TargetEvolution,
    Theory2Parameters,
    target_data,
    total_stress_energy,
)


@dataclass
class Theory2FixedState:
    a: Array
    pi: Array
    astar: Array
    time: float = 0.0


class Theory2FixedSolver:
    """Evolve ``A_i``, canonical ``pi^i``, and ``Astar_i`` on fixed geometry.

    The fluid primitive fields are prescribed. This is the fully specified
    validation system prior to choosing an explicit fluid exchange four-force.
    """

    def __init__(
        self,
        grid: PeriodicGrid,
        fluid: FluidPrimitive,
        parameters: Theory2Parameters | None = None,
        h: Array | None = None,
        lapse: Array | None = None,
        shift: Array | None = None,
    ) -> None:
        self.grid = grid
        self.fluid = fluid
        self.parameters = parameters or Theory2Parameters()
        self.h = flat_metric(grid) if h is None else np.asarray(h, dtype=float)
        self.lapse = np.ones(grid.shape) if lapse is None else np.asarray(lapse, dtype=float)
        self.shift = grid.zeros((3,)) if shift is None else np.asarray(shift, dtype=float)
        fluid_diagnostics(grid, self.h, fluid)
        self.proca = Theory2ProcaSystem(grid, self.parameters)
        self.target = TargetEvolution(grid, self.parameters)

    def rhs(self, time: float, values: tuple[Array, ...]) -> tuple[Array, ...]:
        a, pi, astar = values
        da, dpi = self.proca.rhs_theory2(
            self.h,
            a,
            pi,
            self.fluid,
            astar,
            self.lapse,
            self.shift,
        )
        (dastar,) = self.target.rhs(time, (astar,), self.h, self.fluid)
        return da, dpi, dastar

    def step(self, state: Theory2FixedState, dt: float) -> Theory2FixedState:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        a, pi, astar = rk4_arrays(
            (state.a, state.pi, state.astar), state.time, dt, self.rhs
        )
        return Theory2FixedState(a, pi, astar, state.time + dt)

    def phi(self, state: Theory2FixedState) -> Array:
        return self.proca.constrained_phi(
            self.h, state.pi, self.fluid, state.astar
        )

    def stress_energy(self, state: Theory2FixedState) -> StressEnergy3p1:
        proca = self.proca.stress_energy_theory2(
            self.h, state.a, state.pi, self.fluid, state.astar
        )
        return total_stress_energy(
            self.grid,
            self.h,
            self.fluid,
            proca,
            state.a,
            self.phi(state),
            state.astar,
            self.parameters,
        )

    def diagnostics(self, state: Theory2FixedState) -> dict[str, float]:
        gauss = self.proca.gauss_constraint(
            self.h, state.a, state.pi, self.fluid, state.astar, self.phi(state)
        )
        target = target_data(
            self.grid, self.h, self.fluid, state.astar, self.parameters
        )
        stress = self.stress_energy(state)
        mismatch = state.a - state.astar
        mismatch_l2 = np.sqrt(np.mean(np.sum(mismatch * mismatch, axis=0)))
        return {
            "time": state.time,
            "gauss_l2": float(np.sqrt(np.mean(gauss**2))),
            "gauss_linf": float(np.max(np.abs(gauss))),
            "target_constraint_linf": float(np.max(np.abs(target.target_constraint))),
            "target_mismatch_l2": float(mismatch_l2),
            "total_eulerian_energy": self.grid.integrate(stress.rho),
        }

    def minkowski_current(self, state: Theory2FixedState) -> tuple[Array, Array]:
        """Return physical current and positive timelike margin for flat ``h``."""
        speed_sq = np.sum(self.fluid.velocity**2, axis=0)
        W = 1.0 / np.sqrt(1.0 - speed_sq)
        u = np.concatenate((W[None, ...], W[None, ...] * self.fluid.velocity), axis=0)
        phi = self.phi(state)
        A_contrav = np.concatenate((phi[None, ...], state.a), axis=0)
        xi = project_spatial(A_contrav, u)
        current, _, _ = causal_current(self.fluid.baryon_density, u, xi)
        return current, causal_margin(current)
