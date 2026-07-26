"""Constraint-converged active-neural spacetime evolution studies."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .convergence import (
    _observed_order,
    _same_grid_difference_norm,
    _tuple_difference_norm,
)
from .frozen_closure import QualifiedFrozenClosure
from .grid import Array, PeriodicGrid
from .initial_constraints import CTTParameters, PeriodicCMCInitialDataSolver
from .matter import FluidPrimitive
from .production33 import Theory33ProductionSolver
from .theory33 import CarrierPrimitive


@dataclass(frozen=True)
class ConstrainedLevel:
    points: int
    steps: int
    dt: float
    elliptic_report: dict[str, float | int | bool]
    initial_diagnostics: dict[str, object]
    final_diagnostics: dict[str, object]
    relative_balance_errors: dict[str, float]
    energy_change_rate: float
    rhs: tuple[Array, ...]
    final_state: tuple[Array, ...]


def balanced_active_counterflow(
    solver: Theory33ProductionSolver,
    h: Array,
) -> tuple[FluidPrimitive, CarrierPrimitive, float]:
    """Return phase-two primitives with pointwise-zero total momentum.

    The carrier keeps a positive coordinate flow. The baryon velocity is
    solved independently in every cell so the Carter material momentum
    vanishes before the CMC solve. This supplies the periodic momentum
    compatibility condition without removing the relative current.
    """
    grid = solver.grid
    (coordinate,) = grid.coordinates()
    phase = 2.0 * np.pi * coordinate / grid.lengths[0]
    density = 0.1 * (1.0 + 0.01 * np.cos(phase))
    carrier_density = 0.005 * (1.0 + 0.01 * np.cos(phase))
    internal = np.full(grid.shape, 0.15)
    carrier_velocity = grid.zeros((3,))
    carrier_velocity[0] = 0.125 + 0.001 * np.cos(phase)
    carrier = CarrierPrimitive(carrier_density, carrier_velocity)

    def momentum(fluid_speed: Array) -> Array:
        velocity = grid.zeros((3,))
        velocity[0] = fluid_speed
        fluid = FluidPrimitive(
            density,
            internal,
            velocity,
            grid.zeros(),
        )
        return solver.master.evaluate(h, fluid, carrier).stress.momentum[0]

    lower = np.full(grid.shape, -0.05)
    upper = np.full(grid.shape, 0.01)
    lower_momentum = momentum(lower)
    upper_momentum = momentum(upper)
    if np.any(lower_momentum >= 0.0) or np.any(upper_momentum <= 0.0):
        raise FloatingPointError(
            "active counterflow momentum root is not bracketed"
        )
    for _ in range(60):
        middle = 0.5 * (lower + upper)
        middle_momentum = momentum(middle)
        upper = np.where(middle_momentum > 0.0, middle, upper)
        lower = np.where(middle_momentum > 0.0, lower, middle)

    velocity = grid.zeros((3,))
    velocity[0] = 0.5 * (lower + upper)
    fluid = FluidPrimitive(
        density,
        internal,
        velocity,
        grid.zeros(),
    )
    residual = float(
        np.max(
            np.abs(
                solver.master.evaluate(
                    h,
                    fluid,
                    carrier,
                ).stress.momentum
            )
        )
    )
    return fluid, carrier, residual


def build_constraint_converged_state(
    solver: Theory33ProductionSolver,
    *,
    picard_iterations: int = 3,
    elliptic_tolerance: float = 2.0e-8,
) -> tuple[object, dict[str, float | int | bool]]:
    """Construct a matter/geometry-consistent periodic CMC state."""
    if picard_iterations < 2:
        raise ValueError("at least two matter/geometry Picard iterations are required")
    grid = solver.grid
    geometry = solver.ccz4.flat_state()
    conformal_factor = None
    vector_potential = None
    constraint_solver = PeriodicCMCInitialDataSolver(
        grid,
        CTTParameters(
            kappa=solver.ccz4.parameters.kappa_gravity,
            maximum_iterations=9000,
            tolerance=elliptic_tolerance,
            relaxation=0.3,
            report_interval=20,
        ),
        expanding=False,
    )
    latest = None
    momentum_balance = np.inf
    previous_rho = None
    source_change = np.inf
    for _ in range(picard_iterations):
        h, _ = solver.ccz4.physical_geometry(geometry)
        fluid, carrier, momentum_balance = balanced_active_counterflow(
            solver,
            h,
        )
        state = solver.initialize(geometry, fluid, carrier=carrier)
        recovery = solver.recover(state)
        if previous_rho is not None:
            source_change = float(
                np.sqrt(np.mean((recovery.total_stress.rho - previous_rho) ** 2))
                / max(
                    float(np.sqrt(np.mean(recovery.total_stress.rho**2))),
                    1.0e-300,
                )
            )
        previous_rho = recovery.total_stress.rho.copy()
        latest = constraint_solver.solve(
            recovery.total_stress.rho,
            recovery.total_stress.momentum,
            conformal_factor,
            vector_potential,
        )
        if not latest.report.converged:
            raise FloatingPointError(
                "active-neural periodic CMC solve did not converge: "
                f"{latest.report}"
            )
        conformal_factor = latest.conformal_factor
        vector_potential = latest.vector_potential
        geometry = solver.ccz4.from_adm(latest.geometry)

    if latest is None:
        raise AssertionError("unreachable empty CMC iteration")
    h, _ = solver.ccz4.physical_geometry(geometry)
    fluid, carrier, momentum_balance = balanced_active_counterflow(solver, h)
    state = solver.initialize(geometry, fluid, carrier=carrier)
    recovery = solver.recover(state)
    h_residual, m_residual, _, trace_K = constraint_solver.residuals(
        conformal_factor,
        vector_potential,
        recovery.total_stress.rho,
        recovery.total_stress.momentum,
    )
    h_norm = float(np.sqrt(np.mean(h_residual**2)))
    m_norm = float(np.sqrt(np.mean(np.sum(m_residual**2, axis=0))))
    return state, {
        "converged": (
            latest.report.converged
            and h_norm <= 1.1 * elliptic_tolerance
            and m_norm <= 1.1 * elliptic_tolerance
        ),
        "iterations": latest.report.iterations,
        "picard_iterations": picard_iterations,
        "hamiltonian_residual_l2": h_norm,
        "momentum_residual_l2": m_norm,
        "trace_K": float(trace_K),
        "minimum_conformal_factor": float(np.min(conformal_factor)),
        "maximum_conformal_factor": float(np.max(conformal_factor)),
        "pointwise_momentum_balance": momentum_balance,
        "final_source_relative_change": source_change,
    }


def _characteristic_diagnostics(
    solver: Theory33ProductionSolver,
    state: object,
) -> dict[str, float | bool]:
    recovery = solver.recover(state)
    audits = [
        solver.master.characteristic_audit(
            float(recovery.fluid.baryon_density[index]),
            float(recovery.fluid.specific_internal_energy[index]),
            float(recovery.fluid.velocity[0][index]),
            float(recovery.carrier.number_density[index]),
            float(recovery.carrier.velocity[0][index]),
        )
        for index in np.ndindex(solver.grid.shape)
    ]
    maximum = max(audit.maximum_absolute_speed for audit in audits)
    return {
        "characteristic_maximum_absolute_speed": maximum,
        "characteristic_minimum_causal_margin": 1.0 - maximum,
        "characteristic_maximum_eigenvector_condition": max(
            audit.eigenvector_condition for audit in audits
        ),
        "characteristic_all_strongly_hyperbolic": all(
            audit.strongly_hyperbolic for audit in audits
        ),
        "characteristic_all_causal": all(audit.causal for audit in audits),
    }


def _run_level(
    points: int,
    *,
    model: str,
    final_time: float,
    artifact_path: str | Path,
    picard_iterations: int,
    elliptic_tolerance: float,
) -> ConstrainedLevel:
    grid = PeriodicGrid((points,), (1.0,), method="fd2")
    closure = (
        QualifiedFrozenClosure(artifact_path)
        if model == "qualified_frozen"
        else None
    )
    solver = Theory33ProductionSolver(
        grid,
        constitutive_closure=closure,
    )
    state, elliptic = build_constraint_converged_state(
        solver,
        picard_iterations=picard_iterations,
        elliptic_tolerance=elliptic_tolerance,
    )
    initial = solver.diagnostics(state)
    initial.update(_characteristic_diagnostics(solver, state))
    rhs = solver.rhs(state.time, solver._pack(state))
    # The complete split drag/projection map has a stricter qualified step
    # than the conservative CFL limit on this CMC background. Scale that
    # measured bound with resolution so temporal error also converges.
    maximum_dt = 1.0e-7 * 8.0 / points
    steps = max(1, int(math.ceil(final_time / maximum_dt)))
    dt = final_time / steps
    for _ in range(steps):
        state = solver.step(state, dt)
    final = solver.diagnostics(state)
    final.update(_characteristic_diagnostics(solver, state))

    def relative_balance(name: str) -> float:
        before = float(initial[name])
        after = float(final[name])
        return abs(after - before) / max(abs(before), 1.0e-300)

    return ConstrainedLevel(
        points=points,
        steps=steps,
        dt=dt,
        elliptic_report=elliptic,
        initial_diagnostics=initial,
        final_diagnostics=final,
        relative_balance_errors={
            "baryon_mass": relative_balance("baryon_mass"),
            "carrier_number": relative_balance("carrier_number"),
            "combined_energy": relative_balance("combined_energy"),
        },
        energy_change_rate=(
            float(final["combined_energy"])
            - float(initial["combined_energy"])
        )
        / final_time,
        rhs=tuple(np.asarray(value).copy() for value in rhs),
        final_state=tuple(
            np.asarray(value).copy() for value in solver._pack(state)
        ),
    )


def _constraint_orders(
    levels: tuple[ConstrainedLevel, ...],
    key: str,
    *,
    final: bool,
) -> dict[str, float]:
    diagnostics = (
        [level.final_diagnostics for level in levels]
        if final
        else [level.initial_diagnostics for level in levels]
    )
    errors = [float(values[key]) for values in diagnostics]
    return {
        "coarse": errors[0],
        "medium": errors[1],
        "fine": errors[2],
        "coarse_medium_order": _observed_order(errors[0], errors[1]),
        "medium_fine_order": _observed_order(errors[1], errors[2]),
    }


def _serialize_model(
    model: str,
    levels: tuple[ConstrainedLevel, ...],
) -> dict[str, object]:
    ratio0 = levels[1].points // levels[0].points
    ratio1 = levels[2].points // levels[1].points
    evolved_cm = _tuple_difference_norm(
        levels[0].final_state,
        levels[1].final_state,
        ratio0,
    )
    evolved_mf = _tuple_difference_norm(
        levels[1].final_state,
        levels[2].final_state,
        ratio1,
    )
    energy_rate_cm = abs(
        levels[0].energy_change_rate - levels[1].energy_change_rate
    )
    energy_rate_mf = abs(
        levels[1].energy_change_rate - levels[2].energy_change_rate
    )
    return {
        "model": model,
        "levels": [
            {
                "points": level.points,
                "steps": level.steps,
                "dt": level.dt,
                "elliptic_report": level.elliptic_report,
                "initial_diagnostics": level.initial_diagnostics,
                "final_diagnostics": level.final_diagnostics,
                "relative_balance_errors": level.relative_balance_errors,
                "energy_change_rate": level.energy_change_rate,
            }
            for level in levels
        ],
        "initial_constraints": {
            "hamiltonian": _constraint_orders(
                levels, "hamiltonian_l2", final=False
            ),
            "momentum": _constraint_orders(
                levels, "momentum_l2", final=False
            ),
        },
        "final_constraints": {
            "hamiltonian": _constraint_orders(
                levels, "hamiltonian_l2", final=True
            ),
            "momentum": _constraint_orders(
                levels, "momentum_l2", final=True
            ),
        },
        "evolved_state": {
            "coarse_medium_error": evolved_cm,
            "medium_fine_error": evolved_mf,
            "observed_order": _observed_order(evolved_cm, evolved_mf),
        },
        "geometric_work_energy_rate": {
            "coarse": levels[0].energy_change_rate,
            "medium": levels[1].energy_change_rate,
            "fine": levels[2].energy_change_rate,
            "coarse_medium_difference": energy_rate_cm,
            "medium_fine_difference": energy_rate_mf,
            "observed_order": _observed_order(
                energy_rate_cm,
                energy_rate_mf,
            ),
        },
    }


def run_constrained_convergence(
    resolutions: Iterable[int] = (8, 16, 32),
    *,
    final_time: float = 1.0e-6,
    artifact_path: str | Path = "theory33_frozen_variants.json",
    picard_iterations: int = 3,
    elliptic_tolerance: float = 2.0e-8,
) -> dict[str, object]:
    """Run the analytic/frozen constraint-converged spacetime ladder."""
    values = tuple(int(value) for value in resolutions)
    if len(values) != 3 or any(value < 4 for value in values):
        raise ValueError("exactly three resolutions of at least four points are required")
    if values[1] != 2 * values[0] or values[2] != 2 * values[1]:
        raise ValueError("resolutions must form a factor-two nested triplet")
    if final_time <= 0.0 or elliptic_tolerance <= 0.0:
        raise ValueError("time and elliptic tolerance must be positive")

    analytic_levels = tuple(
        _run_level(
            points,
            model="analytic_m1",
            final_time=final_time,
            artifact_path=artifact_path,
            picard_iterations=picard_iterations,
            elliptic_tolerance=elliptic_tolerance,
        )
        for points in values
    )
    frozen_levels = tuple(
        _run_level(
            points,
            model="qualified_frozen",
            final_time=final_time,
            artifact_path=artifact_path,
            picard_iterations=picard_iterations,
            elliptic_tolerance=elliptic_tolerance,
        )
        for points in values
    )
    analytic = _serialize_model("analytic_m1", analytic_levels)
    frozen = _serialize_model("qualified_frozen", frozen_levels)
    finest_analytic = analytic_levels[-1]
    finest_frozen = frozen_levels[-1]
    rhs_difference = _same_grid_difference_norm(
        finest_analytic.rhs,
        finest_frozen.rhs,
    )
    evolved_difference = _same_grid_difference_norm(
        finest_analytic.final_state,
        finest_frozen.final_state,
    )

    all_levels = analytic_levels + frozen_levels
    frozen_final = finest_frozen.final_diagnostics
    frozen_initial_constraints = frozen["initial_constraints"]
    frozen_final_constraints = frozen["final_constraints"]
    qualification = {
        "elliptic_constraints_converged": all(
            bool(level.elliptic_report["converged"])
            for level in all_levels
        ),
        "pointwise_momentum_compatible": all(
            float(level.elliptic_report["pointwise_momentum_balance"])
            < 1.0e-14
            for level in all_levels
        ),
        "initial_ccz4_constraints_second_order": (
            float(
                frozen_initial_constraints["hamiltonian"][
                    "medium_fine_order"
                ]
            )
            > 1.8
            and float(
                frozen_initial_constraints["momentum"][
                    "medium_fine_order"
                ]
            )
            > 1.8
        ),
        "evolved_ccz4_constraints_second_order": (
            float(
                frozen_final_constraints["hamiltonian"][
                    "medium_fine_order"
                ]
            )
            > 1.8
            and float(
                frozen_final_constraints["momentum"][
                    "medium_fine_order"
                ]
            )
            > 1.8
        ),
        "recovery_currents_and_entropy": all(
            int(level.final_diagnostics["recovery_failures"]) == 0
            and level.relative_balance_errors["baryon_mass"] < 1.0e-12
            and level.relative_balance_errors["carrier_number"] < 1.0e-12
            and float(level.final_diagnostics["step_entropy_change"])
            >= -1.0e-10
            for level in all_levels
        ),
        "geometric_work_energy_rate_converges": (
            float(
                frozen["geometric_work_energy_rate"]["observed_order"]
            )
            > 1.8
        ),
        "frozen_phase_and_switching": (
            float(frozen_final["constitutive_phase_two_fraction"]) > 0.99
            and float(
                frozen_final["constitutive_minimum_switching_margin"]
            )
            > 1.0e-3
        ),
        "frozen_convexity": (
            float(frozen_final["minimum_legendre_eigenvalue"]) > 0.0
            and float(frozen_final["minimum_thermodynamic_eigenvalue"]) > 0.0
        ),
        "frozen_hyperbolicity_and_causality": (
            bool(frozen_final["characteristic_all_strongly_hyperbolic"])
            and bool(frozen_final["characteristic_all_causal"])
            and float(frozen_final["characteristic_minimum_causal_margin"])
            > 0.0
        ),
        "positive_evolved_state_convergence": (
            np.isfinite(float(frozen["evolved_state"]["observed_order"]))
            and float(frozen["evolved_state"]["observed_order"]) > 0.0
        ),
        "neural_spacetime_response_is_active": rhs_difference > 1.0e-14,
    }
    return {
        "format": "tesseract.nr.constrained-neural-spacetime.v1",
        "scope": (
            "periodic CMC matter/geometry Picard data with active frozen "
            "counterflow and coupled CCZ4 evolution"
        ),
        "resolutions": list(values),
        "final_time": final_time,
        "picard_iterations": picard_iterations,
        "elliptic_tolerance": elliptic_tolerance,
        "analytic_control": analytic,
        "qualified_frozen": frozen,
        "finest_model_difference": {
            "rhs_l2": rhs_difference,
            "evolved_state_l2": evolved_difference,
        },
        "qualification": qualification,
        "qualified": all(qualification.values()),
    }
