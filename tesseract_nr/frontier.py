"""Long-horizon, reconstructed-flux, multidimensional NR qualification."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .constrained_convergence import (
    _characteristic_diagnostics,
    balanced_active_counterflow,
    build_constraint_converged_state,
)
from .frozen_closure import QualifiedFrozenClosure
from .grid import Array, PeriodicGrid
from .production3 import Theory3ProductionParameters
from .production33 import Theory33ProductionSolver


def _observed_order(coarse: float, fine: float) -> float:
    if coarse <= 0.0 or fine <= 0.0:
        return float("nan")
    return math.log(coarse / fine, 2.0)


def _nested_restrict(field: Array, ratio: int, ndim: int) -> Array:
    """Restrict the package's nested periodic nodal grids in every axis."""
    result = np.asarray(field)
    if ratio < 1:
        raise ValueError("restriction ratio must be positive")
    for cells in result.shape[-ndim:]:
        if cells % ratio:
            raise ValueError("fine shape is not divisible by restriction ratio")
    return result[
        (slice(None),) * (result.ndim - ndim)
        + (slice(None, None, ratio),) * ndim
    ]


def _state_difference(
    coarse: tuple[Array, ...],
    fine: tuple[Array, ...],
    ratio: int,
    ndim: int,
) -> float:
    squared = 0.0
    count = 0
    for coarse_field, fine_field in zip(coarse, fine, strict=True):
        difference = np.asarray(coarse_field) - _nested_restrict(
            fine_field, ratio, ndim
        )
        squared += float(np.sum(difference**2))
        count += difference.size
    return math.sqrt(squared / max(count, 1))


def _same_grid_difference(
    left: tuple[Array, ...], right: tuple[Array, ...]
) -> float:
    squared = 0.0
    count = 0
    for left_field, right_field in zip(left, right, strict=True):
        difference = np.asarray(left_field) - np.asarray(right_field)
        squared += float(np.sum(difference**2))
        count += difference.size
    return math.sqrt(squared / max(count, 1))


def _solver(
    shape: tuple[int, ...],
    *,
    model: str,
    reconstruction: str,
    artifact_path: str | Path,
) -> Theory33ProductionSolver:
    closure = (
        QualifiedFrozenClosure(artifact_path)
        if model == "qualified_frozen"
        else None
    )
    return Theory33ProductionSolver(
        PeriodicGrid(shape, (1.0,) * len(shape), method="fd2"),
        production=Theory3ProductionParameters(
            flux_reconstruction=reconstruction,
            source_splitting=(
                "strang" if reconstruction != "piecewise_constant" else "lie"
            ),
        ),
        constitutive_closure=closure,
    )


def reconstruction_accuracy(
    resolutions: Iterable[int] = (16, 32, 64, 128),
) -> dict[str, object]:
    """Measure the coupled interface operators on periodic advection."""
    values = tuple(int(value) for value in resolutions)
    if len(values) < 3 or any(value < 8 for value in values):
        raise ValueError("at least three resolutions of eight cells are required")
    schemes: dict[str, object] = {}
    for scheme in ("piecewise_constant", "muscl_mc", "weno5_z"):
        errors: list[float] = []
        for points in values:
            solver = _solver(
                (points,),
                model="analytic_m1",
                reconstruction=scheme,
                artifact_path="theory33_frozen_variants.json",
            )
            (coordinate,) = solver.grid.coordinates()
            state = np.sin(2.0 * np.pi * coordinate)
            speed = np.ones_like(state)
            interface = solver._rusanov_interface(
                state, state, speed, 0
            )
            numerical = solver._interface_divergence(interface, 0)
            exact = -2.0 * np.pi * np.cos(2.0 * np.pi * coordinate)
            errors.append(float(np.sqrt(np.mean((numerical - exact) ** 2))))
        schemes[scheme] = {
            "errors": errors,
            "orders": [
                _observed_order(errors[index], errors[index + 1])
                for index in range(len(errors) - 1)
            ],
        }
    weno = schemes["weno5_z"]
    control = schemes["piecewise_constant"]
    return {
        "resolutions": list(values),
        "schemes": schemes,
        "qualification": {
            "piecewise_constant_control_is_first_order": (
                0.9 < float(control["orders"][-1]) < 1.1
            ),
            "weno5_z_is_high_order": float(weno["orders"][-1]) > 4.5,
            "weno5_z_reduces_finest_error": (
                float(weno["errors"][-1])
                < 1.0e-4 * float(control["errors"][-1])
            ),
        },
    }


def _diagnostics(
    solver: Theory33ProductionSolver, state: object
) -> dict[str, object]:
    result = solver.diagnostics(state)
    result.update(_characteristic_diagnostics(solver, state))
    return result


def _balance_errors(
    initial: dict[str, object], final: dict[str, object]
) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in ("baryon_mass", "carrier_number"):
        before = float(initial[name])
        after = float(final[name])
        result[name] = abs(after - before) / max(abs(before), 1.0e-300)
    return result


def _run_horizon_model(
    *,
    model: str,
    points: int,
    steps: int,
    dt: float,
    reconstruction: str,
    artifact_path: str | Path,
    diagnostic_stride: int,
) -> tuple[dict[str, object], tuple[Array, ...]]:
    solver = _solver(
        (points,),
        model=model,
        reconstruction=reconstruction,
        artifact_path=artifact_path,
    )
    state, elliptic = build_constraint_converged_state(solver)
    initial = _diagnostics(solver, state)
    snapshots = [
        {
            "step": 0,
            "time": 0.0,
            "hamiltonian_l2": initial["hamiltonian_l2"],
            "momentum_l2": initial["momentum_l2"],
            "entropy_integral": initial["entropy_integral"],
            "switching_margin": initial[
                "constitutive_minimum_switching_margin"
            ],
        }
    ]
    minimum_step_entropy_change = float("inf")
    maximum_recovery_failures = int(initial["recovery_failures"])
    for step in range(1, steps + 1):
        state = solver.step(state, dt)
        minimum_step_entropy_change = min(
            minimum_step_entropy_change, solver.last_step_entropy_change
        )
        if step % diagnostic_stride == 0 or step == steps:
            values = _diagnostics(solver, state)
            maximum_recovery_failures = max(
                maximum_recovery_failures,
                int(values["recovery_failures"]),
            )
            snapshots.append(
                {
                    "step": step,
                    "time": state.time,
                    "hamiltonian_l2": values["hamiltonian_l2"],
                    "momentum_l2": values["momentum_l2"],
                    "entropy_integral": values["entropy_integral"],
                    "switching_margin": values[
                        "constitutive_minimum_switching_margin"
                    ],
                }
            )
    final = _diagnostics(solver, state)
    return (
        {
            "model": model,
            "points": points,
            "steps": steps,
            "dt": dt,
            "final_time": state.time,
            "reconstruction": reconstruction,
            "elliptic_report": elliptic,
            "initial_diagnostics": initial,
            "final_diagnostics": final,
            "relative_balance_errors": _balance_errors(initial, final),
            "minimum_step_entropy_change": minimum_step_entropy_change,
            "maximum_recovery_failures": maximum_recovery_failures,
            "snapshots": snapshots,
        },
        tuple(np.asarray(value).copy() for value in solver._pack(state)),
    )


def run_long_horizon(
    *,
    points: int = 8,
    steps: int = 128,
    dt: float = 1.0e-7,
    reconstruction: str = "weno5_z",
    artifact_path: str | Path = "theory33_frozen_variants.json",
    diagnostic_stride: int = 16,
) -> dict[str, object]:
    """Evolve analytic and frozen constrained states for 128+ split steps."""
    if points < 8 or steps < 1 or dt <= 0.0:
        raise ValueError("invalid long-horizon controls")
    analytic, analytic_state = _run_horizon_model(
        model="analytic_m1",
        points=points,
        steps=steps,
        dt=dt,
        reconstruction=reconstruction,
        artifact_path=artifact_path,
        diagnostic_stride=diagnostic_stride,
    )
    frozen, frozen_state = _run_horizon_model(
        model="qualified_frozen",
        points=points,
        steps=steps,
        dt=dt,
        reconstruction=reconstruction,
        artifact_path=artifact_path,
        diagnostic_stride=diagnostic_stride,
    )
    final_difference = _same_grid_difference(analytic_state, frozen_state)
    frozen_final = frozen["final_diagnostics"]
    qualification = {
        "at_least_128_steps": steps >= 128,
        "elliptic_constraints_converged": (
            bool(analytic["elliptic_report"]["converged"])
            and bool(frozen["elliptic_report"]["converged"])
        ),
        "no_recovery_failures": (
            int(analytic["maximum_recovery_failures"]) == 0
            and int(frozen["maximum_recovery_failures"]) == 0
        ),
        "currents_conserved": all(
            float(run["relative_balance_errors"][name]) < 1.0e-12
            for run in (analytic, frozen)
            for name in ("baryon_mass", "carrier_number")
        ),
        "entropy_gate_held": all(
            float(run["minimum_step_entropy_change"]) >= -1.0e-10
            for run in (analytic, frozen)
        ),
        "frozen_switching_and_convexity_held": (
            float(
                frozen_final["constitutive_minimum_switching_margin"]
            )
            > 1.0e-3
            and float(frozen_final["minimum_legendre_eigenvalue"]) > 0.0
            and float(frozen_final["minimum_thermodynamic_eigenvalue"]) > 0.0
        ),
        "frozen_hyperbolicity_and_causality_held": (
            bool(frozen_final["characteristic_all_strongly_hyperbolic"])
            and bool(frozen_final["characteristic_all_causal"])
        ),
        "neural_response_remains_distinct": final_difference > 1.0e-14,
    }
    return {
        "analytic_control": analytic,
        "qualified_frozen": frozen,
        "final_state_model_difference_l2": final_difference,
        "qualification": qualification,
        "qualified": all(qualification.values()),
    }


def _run_multidimensional_level(
    points: int,
    *,
    ndim: int,
    final_time: float,
    model: str,
    reconstruction: str,
    artifact_path: str | Path,
    base_steps: int,
) -> dict[str, object]:
    solver = _solver(
        (points,) * ndim,
        model=model,
        reconstruction=reconstruction,
        artifact_path=artifact_path,
    )
    state, elliptic = build_constraint_converged_state(solver)
    initial = _diagnostics(solver, state)
    recovery = solver.recover(state)
    transverse_speed = (
        max(
            float(np.max(np.abs(recovery.carrier.velocity[axis])))
            for axis in range(1, ndim)
        )
        if ndim > 1
        else 0.0
    )
    steps = base_steps * points // 6
    dt = final_time / steps
    minimum_step_entropy_change = float("inf")
    for _ in range(steps):
        state = solver.step(state, dt)
        minimum_step_entropy_change = min(
            minimum_step_entropy_change, solver.last_step_entropy_change
        )
    final = _diagnostics(solver, state)
    return {
        "points_per_axis": points,
        "shape": list(solver.grid.shape),
        "cells": int(np.prod(solver.grid.shape)),
        "steps": steps,
        "dt": dt,
        "elliptic_report": elliptic,
        "initial_diagnostics": initial,
        "final_diagnostics": final,
        "relative_balance_errors": _balance_errors(initial, final),
        "minimum_step_entropy_change": minimum_step_entropy_change,
        "maximum_transverse_carrier_speed": transverse_speed,
        "energy_change_rate": (
            float(final["combined_energy"])
            - float(initial["combined_energy"])
        )
        / final_time,
        "final_state": tuple(
            np.asarray(value).copy() for value in solver._pack(state)
        ),
    }


def _constraint_series(
    levels: tuple[dict[str, object], ...],
    key: str,
    stage: str,
) -> dict[str, object]:
    errors = [
        float(level[f"{stage}_diagnostics"][key]) for level in levels
    ]
    return {
        "errors": errors,
        "orders": [
            _observed_order(errors[index], errors[index + 1])
            for index in range(len(errors) - 1)
        ],
    }


def run_multidimensional_convergence(
    resolutions: Iterable[int] = (6, 12, 24),
    *,
    ndim: int = 2,
    final_time: float = 2.0e-7,
    base_steps: int = 4,
    reconstruction: str = "weno5_z",
    artifact_path: str | Path = "theory33_frozen_variants.json",
) -> dict[str, object]:
    """Run a frozen 2D ladder and an analytic finest-grid control."""
    values = tuple(int(value) for value in resolutions)
    if (
        len(values) != 3
        or values[1] != 2 * values[0]
        or values[2] != 2 * values[1]
        or values[0] < 6
    ):
        raise ValueError("resolutions must be a factor-two triplet starting at six")
    if ndim not in (2, 3):
        raise ValueError("multidimensional convergence requires 2D or 3D")
    levels = tuple(
        _run_multidimensional_level(
            points,
            ndim=ndim,
            final_time=final_time,
            model="qualified_frozen",
            reconstruction=reconstruction,
            artifact_path=artifact_path,
            base_steps=base_steps,
        )
        for points in values
    )
    analytic_fine = _run_multidimensional_level(
        values[-1],
        ndim=ndim,
        final_time=final_time,
        model="analytic_m1",
        reconstruction=reconstruction,
        artifact_path=artifact_path,
        base_steps=base_steps,
    )
    cm = _state_difference(
        levels[0]["final_state"],
        levels[1]["final_state"],
        2,
        ndim,
    )
    mf = _state_difference(
        levels[1]["final_state"],
        levels[2]["final_state"],
        2,
        ndim,
    )
    model_difference = _same_grid_difference(
        levels[-1]["final_state"], analytic_fine["final_state"]
    )
    initial_h = _constraint_series(levels, "hamiltonian_l2", "initial")
    initial_m = _constraint_series(levels, "momentum_l2", "initial")
    final_h = _constraint_series(levels, "hamiltonian_l2", "final")
    final_m = _constraint_series(levels, "momentum_l2", "final")
    energy_rates = [
        float(level["energy_change_rate"]) for level in levels
    ]
    energy_differences = [
        abs(energy_rates[index] - energy_rates[index + 1])
        for index in range(2)
    ]
    finest = levels[-1]["final_diagnostics"]
    qualification = {
        "elliptic_constraints_converged": all(
            bool(level["elliptic_report"]["converged"]) for level in levels
        )
        and bool(analytic_fine["elliptic_report"]["converged"]),
        "pointwise_momentum_compatible": all(
            float(level["elliptic_report"]["pointwise_momentum_balance"])
            < 1.0e-14
            for level in levels
        ),
        "genuine_transverse_flow": all(
            float(level["maximum_transverse_carrier_speed"]) > 1.0e-3
            for level in levels
        ),
        "no_recovery_or_entropy_failure": all(
            int(level["final_diagnostics"]["recovery_failures"]) == 0
            and float(level["minimum_step_entropy_change"]) >= -1.0e-10
            for level in levels
        ),
        "currents_conserved": all(
            float(level["relative_balance_errors"][name]) < 1.0e-12
            for level in levels
            for name in ("baryon_mass", "carrier_number")
        ),
        "initial_constraints_converge": (
            float(initial_h["orders"][-1]) > 1.5
            and float(initial_m["orders"][-1]) > 1.5
        ),
        "evolved_constraints_converge": (
            float(final_h["orders"][-1]) > 1.5
            and float(final_m["orders"][-1]) > 1.5
        ),
        "complete_state_converges": (
            np.isfinite(_observed_order(cm, mf))
            and _observed_order(cm, mf) > 1.0
        ),
        "geometric_work_rate_converges": (
            _observed_order(*energy_differences) > 1.0
        ),
        "frozen_physics_gates_hold": (
            float(finest["constitutive_phase_two_fraction"]) > 0.99
            and float(
                finest["constitutive_minimum_switching_margin"]
            )
            > 1.0e-3
            and float(finest["minimum_legendre_eigenvalue"]) > 0.0
            and float(finest["minimum_thermodynamic_eigenvalue"]) > 0.0
            and bool(finest["characteristic_all_strongly_hyperbolic"])
            and bool(finest["characteristic_all_causal"])
        ),
        "neural_response_is_active": model_difference > 1.0e-14,
    }

    def serializable(level: dict[str, object]) -> dict[str, object]:
        return {
            key: value
            for key, value in level.items()
            if key != "final_state"
        }

    return {
        "dimensions": ndim,
        "resolutions": list(values),
        "final_time": final_time,
        "reconstruction": reconstruction,
        "qualified_frozen_levels": [
            serializable(level) for level in levels
        ],
        "analytic_finest_control": serializable(analytic_fine),
        "initial_constraints": {
            "hamiltonian": initial_h,
            "momentum": initial_m,
        },
        "final_constraints": {
            "hamiltonian": final_h,
            "momentum": final_m,
        },
        "complete_state": {
            "coarse_medium_error": cm,
            "medium_fine_error": mf,
            "observed_order": _observed_order(cm, mf),
        },
        "geometric_work_energy_rate": {
            "values": energy_rates,
            "differences": energy_differences,
            "observed_order": _observed_order(*energy_differences),
        },
        "finest_model_difference_l2": model_difference,
        "qualification": qualification,
        "qualified": all(qualification.values()),
    }


def run_three_dimensional_smoke(
    *,
    points: int = 6,
    steps: int = 2,
    dt: float = 1.0e-8,
    artifact_path: str | Path = "theory33_frozen_variants.json",
) -> dict[str, object]:
    """Cross the complete 3D constrained/evolution code path."""
    solver = _solver(
        (points, points, points),
        model="qualified_frozen",
        reconstruction="weno5_z",
        artifact_path=artifact_path,
    )
    state, elliptic = build_constraint_converged_state(solver)
    h, _ = solver.ccz4.physical_geometry(state.geometry)
    _, carrier, momentum_residual = balanced_active_counterflow(solver, h)
    transverse_speed = float(np.max(np.abs(carrier.velocity[1:])))
    minimum_entropy = float("inf")
    for _ in range(steps):
        state = solver.step(state, dt)
        minimum_entropy = min(minimum_entropy, solver.last_step_entropy_change)
    final = _diagnostics(solver, state)
    qualification = {
        "elliptic_constraints_converged": bool(elliptic["converged"]),
        "pointwise_momentum_compatible": momentum_residual < 1.0e-14,
        "three_axis_flow_is_active": transverse_speed > 1.0e-3,
        "evolution_completed": state.time == steps * dt,
        "no_recovery_failure": int(final["recovery_failures"]) == 0,
        "entropy_gate_held": minimum_entropy >= -1.0e-10,
        "frozen_physics_gates_hold": (
            float(final["constitutive_phase_two_fraction"]) > 0.99
            and float(final["constitutive_minimum_switching_margin"]) > 1.0e-3
            and bool(final["characteristic_all_strongly_hyperbolic"])
            and bool(final["characteristic_all_causal"])
        ),
    }
    return {
        "shape": list(solver.grid.shape),
        "cells": int(np.prod(solver.grid.shape)),
        "steps": steps,
        "dt": dt,
        "elliptic_report": elliptic,
        "maximum_transverse_carrier_speed": transverse_speed,
        "minimum_step_entropy_change": minimum_entropy,
        "final_diagnostics": final,
        "qualification": qualification,
        "qualified": all(qualification.values()),
    }


def run_frontier_campaign(
    *,
    artifact_path: str | Path = "theory33_frozen_variants.json",
    horizon_steps: int = 128,
    include_3d: bool = True,
) -> dict[str, object]:
    """Run the complete post-constraint-convergence frontier campaign."""
    reconstruction = reconstruction_accuracy()
    horizon = run_long_horizon(
        steps=horizon_steps,
        artifact_path=artifact_path,
    )
    multidimensional = run_multidimensional_convergence(
        artifact_path=artifact_path
    )
    three_dimensional = (
        run_three_dimensional_smoke(artifact_path=artifact_path)
        if include_3d
        else None
    )
    qualification = {
        "high_order_reconstruction": all(
            reconstruction["qualification"].values()
        ),
        "long_horizon": bool(horizon["qualified"]),
        "two_dimensional_convergence": bool(multidimensional["qualified"]),
        "three_dimensional_path": (
            bool(three_dimensional["qualified"])
            if three_dimensional is not None
            else True
        ),
    }
    return {
        "format": "tesseract.nr.neural-spacetime-frontier.v1",
        "scope": (
            "high-order coupled WENO-Z fluxes, 128-step constrained "
            "analytic/frozen horizons, 2D convergence, and a complete 3D "
            "constraint-compatible neural evolution smoke"
        ),
        "reconstruction_accuracy": reconstruction,
        "long_horizon": horizon,
        "multidimensional_convergence": multidimensional,
        "three_dimensional_smoke": three_dimensional,
        "qualification": qualification,
        "qualified": all(qualification.values()),
    }
