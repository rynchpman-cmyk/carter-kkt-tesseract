"""Paired analytic/frozen resolution studies for the active Theory 3.3 PDE."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .frozen_closure import QualifiedFrozenClosure
from .grid import Array, PeriodicGrid
from .matter import FluidPrimitive
from .production33 import Theory33ProductionSolver
from .theory33 import CarrierPrimitive


@dataclass(frozen=True)
class ResolutionLevel:
    points: int
    steps: int
    dt: float
    initial_diagnostics: dict[str, object]
    diagnostics: dict[str, float | int | str]
    relative_balance_errors: dict[str, float]
    rhs: tuple[Array, ...]
    final_state: tuple[Array, ...]


@dataclass(frozen=True)
class ModelConvergence:
    model: str
    levels: tuple[ResolutionLevel, ...]
    rhs_coarse_medium_error: float
    rhs_medium_fine_error: float
    rhs_observed_order: float
    evolved_coarse_medium_error: float
    evolved_medium_fine_error: float
    evolved_observed_order: float
    evolved_continuum_error_estimate: float


def build_qualified_counterflow(
    solver: Theory33ProductionSolver,
) -> object:
    """Build a smooth state inside the frozen artifact's qualified enclosure."""
    grid = solver.grid
    (coordinate,) = grid.coordinates()
    phase = 2.0 * np.pi * coordinate / grid.lengths[0]
    fluid_velocity = grid.zeros((3,))
    carrier_velocity = grid.zeros((3,))
    fluid_velocity[0] = 0.002 * np.sin(phase)
    carrier_velocity[0] = 0.125 + 0.002 * np.cos(phase)
    fluid = FluidPrimitive(
        0.1 * (1.0 + 0.01 * np.sin(phase)),
        np.full(grid.shape, 0.15),
        fluid_velocity,
        grid.zeros(),
    )
    carrier = CarrierPrimitive(
        0.005 * (1.0 + 0.01 * np.cos(phase)),
        carrier_velocity,
    )
    return solver.initialize(
        solver.ccz4.flat_state(),
        fluid,
        carrier=carrier,
    )


def _restricted(field: Array, ratio: int) -> Array:
    if ratio < 1:
        raise ValueError("restriction ratio must be positive")
    return np.asarray(field)[..., ::ratio]


def _tuple_difference_norm(
    coarse: tuple[Array, ...],
    fine: tuple[Array, ...],
    ratio: int,
) -> float:
    squared = 0.0
    count = 0
    for coarse_field, fine_field in zip(coarse, fine, strict=True):
        difference = np.asarray(coarse_field) - _restricted(fine_field, ratio)
        squared += float(np.sum(difference**2))
        count += difference.size
    return math.sqrt(squared / max(count, 1))


def _same_grid_difference_norm(
    left: tuple[Array, ...],
    right: tuple[Array, ...],
) -> float:
    return _tuple_difference_norm(left, right, 1)


def _observed_order(coarse_medium: float, medium_fine: float) -> float:
    if coarse_medium <= 0.0 or medium_fine <= 0.0:
        return float("nan")
    return math.log(coarse_medium / medium_fine, 2.0)


def _run_level(
    points: int,
    *,
    model: str,
    final_time: float,
    artifact_path: str | Path,
) -> ResolutionLevel:
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
    state = build_qualified_counterflow(solver)
    initial = solver.diagnostics(state)
    rhs = solver.rhs(state.time, solver._pack(state))
    steps = max(1, points // 8)
    dt = final_time / steps
    for _ in range(steps):
        state = solver.step(state, dt)
    diagnostics = solver.diagnostics(state)
    recovery = solver.recover(state)
    characteristic_audits = [
        solver.master.characteristic_audit(
            float(recovery.fluid.baryon_density[index]),
            float(recovery.fluid.specific_internal_energy[index]),
            float(recovery.fluid.velocity[0][index]),
            float(recovery.carrier.number_density[index]),
            float(recovery.carrier.velocity[0][index]),
        )
        for index in np.ndindex(grid.shape)
    ]
    characteristic_maximum = max(
        audit.maximum_absolute_speed for audit in characteristic_audits
    )
    diagnostics.update(
        {
            "characteristic_maximum_absolute_speed": (
                characteristic_maximum
            ),
            "characteristic_minimum_causal_margin": (
                1.0 - characteristic_maximum
            ),
            "characteristic_maximum_eigenvector_condition": max(
                audit.eigenvector_condition
                for audit in characteristic_audits
            ),
            "characteristic_all_strongly_hyperbolic": all(
                audit.strongly_hyperbolic
                for audit in characteristic_audits
            ),
            "characteristic_all_causal": all(
                audit.causal for audit in characteristic_audits
            ),
        }
    )

    def relative_balance(name: str) -> float:
        before = float(initial[name])
        after = float(diagnostics[name])
        return abs(after - before) / max(abs(before), 1.0e-300)

    return ResolutionLevel(
        points=points,
        steps=steps,
        dt=dt,
        initial_diagnostics=initial,
        diagnostics=diagnostics,
        relative_balance_errors={
            "baryon_mass": relative_balance("baryon_mass"),
            "carrier_number": relative_balance("carrier_number"),
            "combined_energy": relative_balance("combined_energy"),
        },
        rhs=tuple(np.asarray(value).copy() for value in rhs),
        final_state=tuple(
            np.asarray(value).copy() for value in solver._pack(state)
        ),
    )


def _model_convergence(
    model: str,
    resolutions: tuple[int, int, int],
    *,
    final_time: float,
    artifact_path: str | Path,
) -> ModelConvergence:
    levels = tuple(
        _run_level(
            points,
            model=model,
            final_time=final_time,
            artifact_path=artifact_path,
        )
        for points in resolutions
    )
    ratio0 = resolutions[1] // resolutions[0]
    ratio1 = resolutions[2] // resolutions[1]
    rhs_cm = _tuple_difference_norm(levels[0].rhs, levels[1].rhs, ratio0)
    rhs_mf = _tuple_difference_norm(levels[1].rhs, levels[2].rhs, ratio1)
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
    evolved_order = _observed_order(evolved_cm, evolved_mf)
    denominator = 2.0**evolved_order - 1.0
    continuum_error = (
        evolved_mf / denominator
        if np.isfinite(denominator) and denominator > 0.0
        else float("inf")
    )
    return ModelConvergence(
        model=model,
        levels=levels,
        rhs_coarse_medium_error=rhs_cm,
        rhs_medium_fine_error=rhs_mf,
        rhs_observed_order=_observed_order(rhs_cm, rhs_mf),
        evolved_coarse_medium_error=evolved_cm,
        evolved_medium_fine_error=evolved_mf,
        evolved_observed_order=evolved_order,
        evolved_continuum_error_estimate=continuum_error,
    )


def run_constitutive_convergence(
    resolutions: Iterable[int] = (8, 16, 32),
    *,
    final_time: float = 1.0e-5,
    artifact_path: str | Path = "theory33_frozen_variants.json",
) -> dict[str, object]:
    """Run paired self-convergence and analytic-control comparisons."""
    values = tuple(int(value) for value in resolutions)
    if len(values) != 3 or any(value < 4 for value in values):
        raise ValueError("exactly three resolutions of at least four points are required")
    if values[1] != 2 * values[0] or values[2] != 2 * values[1]:
        raise ValueError("resolutions must form a factor-two nested triplet")
    if final_time <= 0.0:
        raise ValueError("final_time must be positive")

    analytic = _model_convergence(
        "analytic_m1",
        values,
        final_time=final_time,
        artifact_path=artifact_path,
    )
    frozen = _model_convergence(
        "qualified_frozen",
        values,
        final_time=final_time,
        artifact_path=artifact_path,
    )
    finest_analytic = analytic.levels[-1]
    finest_frozen = frozen.levels[-1]
    rhs_model_difference = _same_grid_difference_norm(
        finest_analytic.rhs,
        finest_frozen.rhs,
    )
    evolved_model_difference = _same_grid_difference_norm(
        finest_analytic.final_state,
        finest_frozen.final_state,
    )

    def serialize(model: ModelConvergence) -> dict[str, object]:
        return {
            "model": model.model,
            "levels": [
                {
                    "points": level.points,
                    "steps": level.steps,
                    "dt": level.dt,
                    "initial_diagnostics": level.initial_diagnostics,
                    "diagnostics": level.diagnostics,
                    "relative_balance_errors": (
                        level.relative_balance_errors
                    ),
                }
                for level in model.levels
            ],
            "rhs": {
                "coarse_medium_error": model.rhs_coarse_medium_error,
                "medium_fine_error": model.rhs_medium_fine_error,
                "observed_order": model.rhs_observed_order,
            },
            "evolved": {
                "coarse_medium_error": model.evolved_coarse_medium_error,
                "medium_fine_error": model.evolved_medium_fine_error,
                "observed_order": model.evolved_observed_order,
                "continuum_error_estimate": (
                    model.evolved_continuum_error_estimate
                ),
            },
        }

    all_levels = analytic.levels + frozen.levels
    conservation_limit = 1.0e-9
    conservation_qualified = all(
        abs(float(level.diagnostics["carrier_number_balance_residual"]))
        < conservation_limit
        and level.relative_balance_errors["baryon_mass"] < 1.0e-12
        and level.relative_balance_errors["carrier_number"] < 1.0e-12
        and level.relative_balance_errors["combined_energy"] < 1.0e-9
        and int(level.diagnostics["recovery_failures"]) == 0
        and float(level.diagnostics["step_entropy_change"]) >= -1.0e-10
        for level in all_levels
    )
    frozen_finest = finest_frozen.diagnostics
    qualification = {
        "recovery_and_conservation": conservation_qualified,
        "frozen_phase_active": (
            float(frozen_finest["constitutive_phase_two_fraction"]) > 0.5
        ),
        "healthy_switching_margin": (
            float(frozen_finest["constitutive_minimum_switching_margin"])
            > 1.0e-3
        ),
        "positive_legendre_margin": (
            float(frozen_finest["minimum_legendre_eigenvalue"]) > 0.0
        ),
        "positive_thermodynamic_margin": (
            float(frozen_finest["minimum_thermodynamic_eigenvalue"]) > 0.0
        ),
        "strongly_hyperbolic_and_causal": (
            bool(
                frozen_finest[
                    "characteristic_all_strongly_hyperbolic"
                ]
            )
            and bool(frozen_finest["characteristic_all_causal"])
            and float(
                frozen_finest["characteristic_minimum_causal_margin"]
            )
            > 0.0
        ),
        "finite_positive_self_convergence": (
            np.isfinite(frozen.evolved_observed_order)
            and frozen.evolved_observed_order > 0.0
        ),
        "neural_response_changes_rhs": rhs_model_difference > 1.0e-14,
    }
    return {
        "format": "tesseract.nr.constitutive-convergence.v1",
        "scope": (
            "smooth periodic active-branch semidiscrete and short-time "
            "self-convergence from flat geometric data"
        ),
        "continuum_claim": (
            "first convergence ladder; not yet a constraint-converged "
            "strong-field continuum solution"
        ),
        "resolutions": list(values),
        "final_time": final_time,
        "analytic_control": serialize(analytic),
        "qualified_frozen": serialize(frozen),
        "finest_model_difference": {
            "rhs_l2": rhs_model_difference,
            "evolved_state_l2": evolved_model_difference,
        },
        "qualification": qualification,
        "qualified": all(qualification.values()),
    }
