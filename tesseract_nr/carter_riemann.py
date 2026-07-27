"""Special-relativistic Carter Riemann and eigensystem experiments.

The production recovery, master function, full numerical principal symbol,
characteristic WENO flux, positivity limiter, and DLM straight-path
fluctuation are reused directly.  Geometry and vector fields are held fixed
so the experiment isolates the homogeneous two-current Carter subsystem.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .frozen_closure import QualifiedFrozenClosure
from .grid import Array, PeriodicGrid
from .matter import FluidPrimitive
from .production3 import (
    Theory3MatterState,
    Theory3ProductionParameters,
    Theory3ProductionState,
)
from .production33 import Theory33ProductionSolver
from .theory33 import CarrierPrimitive


def _solver(
    points: int,
    *,
    reconstruction: str,
    artifact_path: str | Path,
    model: str = "qualified_frozen",
) -> Theory33ProductionSolver:
    if model not in {"analytic_m1", "qualified_frozen"}:
        raise ValueError(f"unknown Carter constitutive model {model!r}")
    closure = (
        QualifiedFrozenClosure(artifact_path)
        if model == "qualified_frozen"
        else None
    )
    return Theory33ProductionSolver(
        PeriodicGrid((points,), (1.0,), method="fd2"),
        production=Theory3ProductionParameters(
            flux_reconstruction=reconstruction,
            positivity_preserving=True,
            positivity_floor_factor=1.001,
            phase_transition_iterations=2,
        ),
        constitutive_closure=closure,
    )


def _riemann_state(
    solver: Theory33ProductionSolver,
    problem: str,
) -> Theory3ProductionState:
    (coordinate,) = solver.grid.coordinates()
    left = coordinate < 0.5
    if problem == "counterflow_shock":
        n_left, n_right = 0.11, 0.07
        e_left, e_right = 0.17, 0.09
        u_left, u_right = 0.02, 0.0
        d_left, d_right = 0.006, 0.0035
        v_left, v_right = 0.08, -0.02
    elif problem == "colliding_streams":
        n_left, n_right = 0.11, 0.065
        e_left, e_right = 0.17, 0.09
        u_left, u_right = 0.025, -0.02
        d_left, d_right = 0.007, 0.003
        v_left, v_right = 0.06, -0.01
    else:
        raise ValueError(f"unknown Carter Riemann problem {problem!r}")
    fluid_velocity = solver.grid.zeros((3,))
    carrier_velocity = solver.grid.zeros((3,))
    fluid_velocity[0] = np.where(left, u_left, u_right)
    carrier_velocity[0] = np.where(left, v_left, v_right)
    return solver.initialize(
        solver.ccz4.flat_state(),
        FluidPrimitive(
            np.where(left, n_left, n_right),
            np.where(left, e_left, e_right),
            fluid_velocity,
            solver.grid.zeros(),
        ),
        carrier=CarrierPrimitive(
            np.where(left, d_left, d_right),
            carrier_velocity,
        ),
    )


def _principal_rhs(
    solver: Theory33ProductionSolver,
    state: Theory3ProductionState,
) -> tuple[Array, ...]:
    recovery = solver.recover(state)
    h, K = solver.ccz4.physical_geometry(state.geometry)
    matter = solver._conservative_rhs(state, recovery, h, K)
    charge, momentum = solver._target_rhs(state, recovery, h, K)
    return matter + (charge, momentum)


def _state_arrays(state: Theory3ProductionState) -> tuple[Array, ...]:
    return (
        state.matter.D,
        state.matter.momentum,
        state.matter.energy,
        state.matter.entropy,
        state.matter.tracer,
        state.target_charge,
        state.target_current,
    )


def _with_arrays(
    template: Theory3ProductionState,
    arrays: tuple[Array, ...],
    time: float,
) -> Theory3ProductionState:
    return Theory3ProductionState(
        template.geometry,
        Theory3MatterState(
            np.array(arrays[0], copy=True),
            np.array(arrays[1], copy=True),
            np.array(arrays[2], copy=True),
            np.array(arrays[3], copy=True),
            np.array(arrays[4], copy=True),
        ),
        template.a,
        template.pi_A,
        template.b,
        template.pi_B,
        template.longitudinal_A,
        template.longitudinal_B,
        template.cleaning_A,
        template.cleaning_B,
        np.array(arrays[5], copy=True),
        np.array(arrays[6], copy=True),
        time,
    )


def _euler_stage(
    solver: Theory33ProductionSolver,
    state: Theory3ProductionState,
    dt: float,
) -> Theory3ProductionState:
    slope = _principal_rhs(solver, state)
    arrays = tuple(
        value + dt * derivative
        for value, derivative in zip(
            _state_arrays(state), slope, strict=True
        )
    )
    candidate = _with_arrays(state, arrays, state.time + dt)
    return solver._project_to_master_manifold(candidate)


def _combine(
    solver: Theory33ProductionSolver,
    first: Theory3ProductionState,
    first_weight: float,
    second: Theory3ProductionState,
    second_weight: float,
    time: float,
) -> Theory3ProductionState:
    arrays = tuple(
        first_weight * left + second_weight * right
        for left, right in zip(
            _state_arrays(first),
            _state_arrays(second),
            strict=True,
        )
    )
    return solver._project_to_master_manifold(
        _with_arrays(first, arrays, time)
    )


def principal_ssprk3_step(
    solver: Theory33ProductionSolver,
    state: Theory3ProductionState,
    dt: float,
) -> Theory3ProductionState:
    """Advance the homogeneous Carter principal subsystem with SSP-RK3."""
    solver._active_stage_dt = dt
    try:
        first = _euler_stage(solver, state, dt)
        second_euler = _euler_stage(solver, first, dt)
        second = _combine(
            solver,
            state,
            0.75,
            second_euler,
            0.25,
            state.time + 0.5 * dt,
        )
        third_euler = _euler_stage(solver, second, dt)
        result = _combine(
            solver,
            state,
            1.0 / 3.0,
            third_euler,
            2.0 / 3.0,
            state.time + dt,
        )
        solver.recover(result)
        return result
    finally:
        solver._active_stage_dt = None


def evolve_riemann(
    points: int,
    *,
    problem: str,
    final_time: float,
    cfl: float,
    artifact_path: str | Path,
    model: str = "analytic_m1",
) -> tuple[Theory33ProductionSolver, Theory3ProductionState]:
    solver = _solver(
        points,
        reconstruction="full_carter_roe",
        artifact_path=artifact_path,
        model=model,
    )
    state = _riemann_state(solver, problem)
    while state.time < final_time - 1.0e-15:
        dt = min(
            cfl * solver.grid.spacing[0],
            final_time - state.time,
        )
        state = principal_ssprk3_step(solver, state, dt)
    return solver, state


def _primitive_snapshot(
    solver: Theory33ProductionSolver,
    state: Theory3ProductionState,
) -> dict[str, Array]:
    recovery = solver.recover(state)
    pressure = solver.master.thermal_pressure(
        recovery.fluid.baryon_density,
        recovery.fluid.specific_internal_energy,
    )
    return {
        "baryon_density": recovery.fluid.baryon_density,
        "fluid_velocity": recovery.fluid.velocity[0],
        "pressure": pressure,
        "carrier_density": recovery.carrier.number_density,
        "carrier_velocity": recovery.carrier.velocity[0],
        "relative_lorentz_factor": (
            recovery.master.relative_lorentz_factor
        ),
        "phase_two": recovery.master.phase_two.astype(float),
    }


def _restricted_error(
    coarse: dict[str, Array],
    fine: dict[str, Array],
    ratio: int,
) -> tuple[float, dict[str, float]]:
    fields: dict[str, float] = {}
    for name, coarse_value in coarse.items():
        fine_value = fine[name][
            (slice(None, None, ratio),) * fine[name].ndim
        ]
        scale = max(
            float(np.mean(np.abs(fine_value))),
            float(np.max(fine_value) - np.min(fine_value)),
            1.0e-8,
        )
        fields[name] = float(
            np.mean(np.abs(coarse_value - fine_value)) / scale
        )
    physical = [
        value for name, value in fields.items() if name != "phase_two"
    ]
    return float(np.mean(physical)), fields


def _variation(snapshot: dict[str, Array]) -> dict[str, float]:
    return {
        name: float(
            sum(
                np.sum(
                    np.abs(
                        np.roll(value, -1, axis=axis) - value
                    )
                )
                for axis in range(value.ndim)
            )
        )
        for name, value in snapshot.items()
    }


def _oblique_state(
    solver: Theory33ProductionSolver,
) -> Theory3ProductionState:
    x, y = solver.grid.coordinates()
    left = np.mod(x + y, 1.0) < 0.5
    normal = 1.0 / np.sqrt(2.0)
    fluid_velocity = solver.grid.zeros((3,))
    carrier_velocity = solver.grid.zeros((3,))
    fluid_speed = np.where(left, 0.035, -0.015)
    carrier_speed = np.where(left, 0.075, -0.025)
    fluid_velocity[0] = normal * fluid_speed
    fluid_velocity[1] = normal * fluid_speed
    carrier_velocity[0] = normal * carrier_speed
    carrier_velocity[1] = normal * carrier_speed
    return solver.initialize(
        solver.ccz4.flat_state(),
        FluidPrimitive(
            np.where(left, 0.105, 0.07),
            np.where(left, 0.16, 0.09),
            fluid_velocity,
            solver.grid.zeros(),
        ),
        carrier=CarrierPrimitive(
            np.where(left, 0.0058, 0.0035),
            carrier_velocity,
        ),
    )


def oblique_symbol_audit(points: int = 8) -> dict[str, object]:
    """Audit rotational consistency of the full symbol on an oblique state."""
    solver = Theory33ProductionSolver(
        PeriodicGrid((points, points), (1.0, 1.0), method="fd2"),
        production=Theory3ProductionParameters(
            flux_reconstruction="full_carter_roe",
            positivity_preserving=True,
        ),
    )
    state = _oblique_state(solver)
    recovery = solver.recover(state)
    h, _ = solver.ccz4.physical_geometry(state.geometry)
    spectra: list[np.ndarray] = []
    conditions: list[float] = []
    residuals: list[float] = []
    for axis in range(2):
        symbol, _, _, _ = solver._full_carter_symbol(
            recovery,
            h,
            state.geometry.lapse,
            state.geometry.shift,
            axis,
        )
        matrix = symbol[:, :, 0, 0]
        values, vectors = np.linalg.eig(matrix)
        order = np.argsort(values.real)
        values = values[order]
        vectors = vectors[:, order]
        spectra.append(values)
        conditions.append(float(np.linalg.cond(vectors)))
        residuals.append(
            float(
                np.max(
                    np.abs(
                        matrix @ vectors
                        - vectors * values[None, :]
                    )
                )
            )
        )
    difference = float(
        np.max(np.abs(spectra[0].real - spectra[1].real))
    )
    report: dict[str, object] = {
        "x_spectrum": spectra[0].real.tolist(),
        "y_spectrum": spectra[1].real.tolist(),
        "maximum_rotational_spectrum_difference": difference,
        "maximum_imaginary_part": float(
            max(np.max(np.abs(value.imag)) for value in spectra)
        ),
        "maximum_eigenpair_residual": max(residuals),
        "maximum_condition_number": max(conditions),
    }
    report["qualification"] = {
        "x_and_y_symbols_real": (
            report["maximum_imaginary_part"] < 2.0e-7
        ),
        "x_and_y_eigenpairs_close": (
            report["maximum_eigenpair_residual"] < 1.0e-8
        ),
        "oblique_rotation_consistent": difference < 2.0e-7,
        "oblique_symbols_conditioned": (
            report["maximum_condition_number"] < 1.0e8
        ),
    }
    return report


def riemann_resolution_ladder(
    problem: str,
    resolutions: Iterable[int] = (32, 64, 128, 256),
    *,
    final_time: float = 0.03,
    cfl: float = 0.12,
    artifact_path: str | Path = "theory33_frozen_variants.json",
    model: str = "analytic_m1",
) -> dict[str, object]:
    values = tuple(int(value) for value in resolutions)
    if len(values) < 3 or any(value < 16 for value in values):
        raise ValueError("Riemann ladder needs at least three resolutions")
    if any(fine != 2 * coarse for coarse, fine in zip(values, values[1:])):
        raise ValueError("Riemann resolutions must form a factor-two ladder")
    snapshots: list[dict[str, Array]] = []
    diagnostics: list[dict[str, object]] = []
    for points in values:
        solver, state = evolve_riemann(
            points,
            problem=problem,
            final_time=final_time,
            cfl=cfl,
            artifact_path=artifact_path,
            model=model,
        )
        recovery = solver.recover(state)
        snapshot = _primitive_snapshot(solver, state)
        snapshots.append(snapshot)
        diagnostics.append(
            {
                "points": points,
                "recovery_failures": recovery.report.failed_cells,
                "maximum_recovery_residual": (
                    recovery.report.maximum_residual
                ),
                "minimum_legendre_eigenvalue": (
                    recovery.report.minimum_legendre_eigenvalue
                ),
                "minimum_thermodynamic_eigenvalue": (
                    recovery.report.minimum_thermodynamic_eigenvalue
                ),
                "phase_two_fraction": float(
                    np.mean(recovery.master.phase_two)
                ),
                "minimum_density": float(
                    np.min(recovery.fluid.baryon_density)
                ),
                "minimum_carrier_density": float(
                    np.min(recovery.carrier.number_density)
                ),
                "maximum_relative_lorentz_factor": float(
                    np.max(recovery.master.relative_lorentz_factor)
                ),
                "characteristic_maximum_imaginary_part": (
                    solver.last_characteristic_maximum_imaginary_part
                ),
                "characteristic_maximum_speed": (
                    solver.last_characteristic_maximum_speed
                ),
                "characteristic_maximum_physical_speed": (
                    solver.last_characteristic_maximum_physical_speed
                ),
                "characteristic_condition_number": (
                    solver.last_characteristic_condition_number
                ),
                "positivity_limited_faces": (
                    solver.last_positivity_limited_faces
                ),
                "variation": _variation(snapshot),
            }
        )
    errors: list[float] = []
    field_errors: list[dict[str, float]] = []
    for coarse, fine in zip(snapshots, snapshots[1:]):
        error, per_field = _restricted_error(coarse, fine, 2)
        errors.append(error)
        field_errors.append(per_field)
    orders = [
        math.log(errors[index] / errors[index + 1], 2.0)
        for index in range(len(errors) - 1)
        if errors[index] > 0.0 and errors[index + 1] > 0.0
    ]
    report: dict[str, object] = {
        "problem": problem,
        "resolutions": list(values),
        "final_time": final_time,
        "cfl": cfl,
        "constitutive_model": model,
        "successive_l1_errors": errors,
        "successive_field_l1_errors": field_errors,
        "observed_orders": orders,
        "diagnostics": diagnostics,
    }
    report["qualification"] = {
        "all_recoveries_valid": all(
            item["recovery_failures"] == 0 for item in diagnostics
        ),
        "all_symbols_real": all(
            item["characteristic_maximum_imaginary_part"] < 2.0e-6
            for item in diagnostics
        ),
        "all_symbols_causal": all(
            item["characteristic_maximum_physical_speed"]
            <= 1.0 + 2.0e-7
            for item in diagnostics
        ),
        "legendre_margins_positive": all(
            item["minimum_legendre_eigenvalue"] > 0.0
            for item in diagnostics
        ),
        "thermodynamic_margins_positive": all(
            item["minimum_thermodynamic_eigenvalue"] > 0.0
            for item in diagnostics
        ),
        "positive_densities": all(
            item["minimum_density"] > 0.0
            and item["minimum_carrier_density"] > 0.0
            for item in diagnostics
        ),
        "successive_error_decreases": all(
            later < earlier
            for earlier, later in zip(errors, errors[1:])
        ),
        "shock_order_resolved": bool(orders) and orders[-1] > 0.35,
    }
    return report


def eigensystem_audit(
    *,
    points: int = 8,
    artifact_path: str | Path = "theory33_frozen_variants.json",
) -> dict[str, object]:
    solver = _solver(
        points,
        reconstruction="full_carter_weno5_z",
        artifact_path=artifact_path,
    )
    state = _riemann_state(solver, "counterflow_shock")
    recovery = solver.recover(state)
    h, _ = solver.ccz4.physical_geometry(state.geometry)
    symbol, path, _, _ = solver._full_carter_symbol(
        recovery,
        h,
        state.geometry.lapse,
        state.geometry.shift,
        0,
    )
    right, left, _, _ = solver._full_carter_characteristic_basis(
        state, recovery, h, 0
    )
    cell = 0
    matrix = symbol[(slice(None), slice(None), cell)]
    values, vectors = np.linalg.eig(matrix)
    order = np.argsort(values.real)
    values = values[order]
    vectors = vectors[:, order]
    residual = float(
        np.max(
            np.abs(
                matrix @ vectors
                - vectors * values[None, :]
            )
        )
    )
    identity = np.einsum(
        "ab...,bc...->ac...", left, right
    )
    biorthogonality = float(
        np.max(
            np.abs(
                identity
                - np.eye(11).reshape(
                    (11, 11) + (1,) * solver.grid.ndim
                )
            )
        )
    )
    acoustic = np.sort(values.real)[[0, 1, 7, 8]]
    fluid = recovery.fluid
    carrier = recovery.carrier
    legacy = solver.master.characteristic_audit(
        float(fluid.baryon_density[cell]),
        float(fluid.specific_internal_energy[cell]),
        float(fluid.velocity[0, cell]),
        float(carrier.number_density[cell]),
        float(carrier.velocity[0, cell]),
    )
    acoustic_difference = float(
        np.max(np.abs(acoustic - legacy.speeds))
    )
    report: dict[str, object] = {
        "full_eigenvalues": np.sort(values.real).tolist(),
        "maximum_imaginary_part": float(
            np.max(np.abs(values.imag))
        ),
        "maximum_eigenpair_residual": residual,
        "maximum_biorthogonality_residual": biorthogonality,
        "eigenvector_condition_number": float(
            np.linalg.cond(vectors)
        ),
        "path_symbol_norm": float(np.linalg.norm(path[:, :, cell])),
        "longitudinal_acoustic_speeds": acoustic.tolist(),
        "independent_audit_speeds": legacy.speeds.tolist(),
        "maximum_acoustic_speed_difference": acoustic_difference,
    }
    report["qualification"] = {
        "nine_physical_modes_present": len(values) == 9,
        "eigenvalues_real": report["maximum_imaginary_part"] < 2.0e-7,
        "eigenpairs_close": residual < 1.0e-8,
        "left_right_biorthogonal": biorthogonality < 1.0e-10,
        "eigenvectors_conditioned": (
            report["eigenvector_condition_number"] < 1.0e8
        ),
        "path_operator_nonzero": report["path_symbol_norm"] > 1.0e-8,
        "independent_longitudinal_audit_matches": (
            acoustic_difference < 2.0e-7
        ),
        "causal": bool(
            max(abs(value) for value in values.real)
            <= 1.0 + 2.0e-7
        ),
    }
    return report


def run_carter_riemann_frontier(
    resolutions: Iterable[int] = (16, 32, 64, 128),
    *,
    final_time: float = 0.02,
    cfl: float = 0.15,
    artifact_path: str | Path = "theory33_frozen_variants.json",
) -> dict[str, object]:
    """Run the complete eigensystem and discontinuous Riemann campaign."""
    eigensystem = eigensystem_audit(artifact_path=artifact_path)
    oblique = oblique_symbol_audit()
    counterflow = riemann_resolution_ladder(
        "counterflow_shock",
        resolutions,
        final_time=final_time,
        cfl=cfl,
        artifact_path=artifact_path,
        model="analytic_m1",
    )
    colliding = riemann_resolution_ladder(
        "colliding_streams",
        resolutions,
        final_time=final_time,
        cfl=cfl,
        artifact_path=artifact_path,
        model="analytic_m1",
    )
    qualification = {
        **{
            f"eigensystem_{name}": bool(value)
            for name, value in eigensystem["qualification"].items()
        },
        **{
            f"oblique_{name}": bool(value)
            for name, value in oblique["qualification"].items()
        },
        **{
            f"counterflow_{name}": bool(value)
            for name, value in counterflow["qualification"].items()
        },
        **{
            f"colliding_{name}": bool(value)
            for name, value in colliding["qualification"].items()
        },
    }
    return {
        "schema": "tesseract.full-carter-riemann-frontier.v1",
        "principal_system": {
            "physical_fields": 9,
            "extended_reconstruction_fields": 11,
            "physical_order": [
                "baryon_density",
                "carrier_density",
                "momentum_x",
                "momentum_y",
                "momentum_z",
                "energy",
                "carrier_canonical_momentum_x",
                "carrier_canonical_momentum_y",
                "carrier_canonical_momentum_z",
            ],
            "extension": ["entropy", "tracer"],
            "path": "straight-state DLM midpoint fluctuation",
            "riemann_flux": "conditioned full-eigensystem Roe-Rusanov blend",
            "time_integrator": "SSP-RK3 with constitutive projection",
        },
        "eigensystem_audit": eigensystem,
        "oblique_symbol_audit": oblique,
        "counterflow_shock": counterflow,
        "colliding_streams": colliding,
        "qualification": qualification,
        "qualified": all(qualification.values()),
    }
