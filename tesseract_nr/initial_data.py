"""Theory 2.0 benchmark and constraint-construction helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .adm import ADMState, flat_metric
from .curved_proca import Theory2ProcaSystem
from .fixed_theory2 import Theory2FixedState
from .grid import Array, CartesianGrid, PeriodicGrid
from .initial_constraints import CTTInitialDataSolver, CTTParameters, CTTReport
from .matter import FluidPrimitive, IdealGasEOS, perfect_fluid_stress_energy
from .theory2 import (
    Theory2Parameters,
    combine_stress_energy,
    interaction_stress_energy,
)


@dataclass(frozen=True)
class TwoLobeParameters:
    separation: float = 2.0
    width: float = 0.8
    peak_density: float = 1.0e-3
    density_floor: float = 1.0e-11
    specific_internal_energy: float = 0.05
    matter_speed: float = 0.25
    beta_target: float = 1.76

    def __post_init__(self) -> None:
        if self.separation <= 0.0 or self.width <= 0.0 or self.peak_density < 0.0:
            raise ValueError("lobe geometry and density are invalid")
        if self.density_floor < 0.0 or self.specific_internal_energy < 0.0:
            raise ValueError("fluid floors must be nonnegative")
        if not 0.0 <= self.matter_speed < 1.0 or self.beta_target < 0.0:
            raise ValueError("matter_speed must be subluminal and beta_target nonnegative")


@dataclass
class Theory2InitialData:
    geometry: ADMState
    fluid: FluidPrimitive
    repair: Theory2FixedState
    phi: Array
    phi_star: Array
    gravitational_constraints_solved: bool
    constraint_report: CTTReport | None = None


def rotating_two_lobe_data(
    grid: PeriodicGrid,
    theory: Theory2Parameters | None = None,
    benchmark: TwoLobeParameters | None = None,
) -> Theory2InitialData:
    """Construct the document's rotating two-lobe material and Proca data.

    The Proca Gauss constraint is solved algebraically. The supplied document
    does not specify an elliptic GR initial-data algorithm, so the geometry is
    conformally flat with K_ij=0 and is explicitly marked unsolved.
    ``sigma=0`` because no benchmark sigma profile is supplied; consequently
    the prescribed initial Astar is an off-equilibrium transient target.
    """
    theory = theory or Theory2Parameters()
    benchmark = benchmark or TwoLobeParameters()
    origins = getattr(grid, "origin", tuple(0.0 for _ in range(grid.ndim)))
    centered = [
        coordinate - (origin + 0.5 * length)
        for coordinate, origin, length in zip(grid.coordinates(), origins, grid.lengths)
    ]
    while len(centered) < 3:
        centered.append(np.zeros(grid.shape))
    x, y, z = centered
    R0 = 0.5 * benchmark.separation
    r_plus_sq = (x - R0) ** 2 + y**2 + z**2
    r_minus_sq = (x + R0) ** 2 + y**2 + z**2
    plus = np.exp(-r_plus_sq / benchmark.width**2)
    minus = np.exp(-r_minus_sq / benchmark.width**2)
    density = benchmark.density_floor + benchmark.peak_density * (plus + minus)
    internal = np.full(grid.shape, benchmark.specific_internal_energy)
    signed_weight = (plus - minus) / np.maximum(plus + minus, 1.0e-300)
    velocity = grid.zeros((3,))
    velocity[2] = benchmark.matter_speed * signed_weight
    sigma = np.zeros(grid.shape)
    fluid = FluidPrimitive(density, internal, velocity, sigma)

    astar = grid.zeros((3,))
    astar[2] = np.arcsinh(benchmark.beta_target) * (plus - minus)
    a = astar.copy()
    pi = grid.zeros((3,))  # E^i=0 in the Theory 2.0 convention.
    geometry = ADMState(flat_metric(grid), grid.zeros((3, 3)))
    system = Theory2ProcaSystem(grid, theory)
    phi = system.constrained_phi(geometry.h, pi, fluid, astar)
    phi_star = np.einsum("i...,i...->...", velocity, astar)
    return Theory2InitialData(
        geometry=geometry,
        fluid=fluid,
        repair=Theory2FixedState(a, pi, astar),
        phi=phi,
        phi_star=phi_star,
        gravitational_constraints_solved=False,
    )


def constraint_solved_rotating_two_lobe_data(
    grid: CartesianGrid,
    theory: Theory2Parameters | None = None,
    benchmark: TwoLobeParameters | None = None,
    ctt: CTTParameters | None = None,
    picard_iterations: int = 3,
) -> Theory2InitialData:
    """Construct two-lobe data and solve the coupled GR constraints.

    A short Picard iteration recomputes the fluid, bare-Proca, and interaction
    stress projections on the latest conformal geometry before each CTT solve.
    The Proca Gauss constraint is then reconstructed on the final slice.
    """
    if picard_iterations < 1:
        raise ValueError("picard_iterations must be positive")
    theory = theory or Theory2Parameters()
    data = rotating_two_lobe_data(grid, theory, benchmark)
    elliptic = CTTInitialDataSolver(grid, ctt)
    geometry = data.geometry
    psi = None
    vector = None
    result = None
    system = Theory2ProcaSystem(grid, theory)
    eos = IdealGasEOS(theory.gamma_ad)
    for _ in range(picard_iterations):
        phi = system.constrained_phi(
            geometry.h, data.repair.pi, data.fluid, data.repair.astar
        )
        proca = system.stress_energy_theory2(
            geometry.h,
            data.repair.a,
            data.repair.pi,
            data.fluid,
            data.repair.astar,
        )
        interaction = interaction_stress_energy(
            geometry.h,
            data.fluid,
            data.repair.a,
            phi,
            data.repair.astar,
            theory,
        )
        fluid_part = perfect_fluid_stress_energy(grid, geometry.h, data.fluid, eos)
        total = combine_stress_energy(geometry.h, fluid_part, proca, interaction)
        result = elliptic.solve(total.rho, total.momentum, psi, vector)
        geometry = result.geometry
        psi = result.conformal_factor
        vector = result.vector_potential
    assert result is not None
    phi = system.constrained_phi(
        geometry.h, data.repair.pi, data.fluid, data.repair.astar
    )
    return Theory2InitialData(
        geometry=geometry,
        fluid=data.fluid,
        repair=data.repair,
        phi=phi,
        phi_star=np.einsum(
            "i...,i...->...", data.fluid.velocity, data.repair.astar
        ),
        gravitational_constraints_solved=result.report.converged,
        constraint_report=result.report,
    )
