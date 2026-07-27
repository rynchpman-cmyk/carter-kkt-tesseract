"""Characteristic-shock and hard-phase-boundary qualification.

The experiments in this module deliberately use discontinuous data.  They
separate two claims:

* a conservative forward-Euler flux update preserves configured material and
  carrier density floors where unlimited characteristic WENO does not; and
* a multidimensional frozen constitutive interface can update its hard branch
  between local solves while every Newton solve remains branchwise smooth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .frozen_closure import QualifiedFrozenClosure
from .grid import PeriodicGrid
from .matter import FluidPrimitive
from .production3 import Theory3ProductionParameters
from .production33 import Theory33ProductionSolver
from .theory33 import CarrierPrimitive


def _solver(
    shape: tuple[int, int],
    *,
    positivity: bool,
    positivity_floor_factor: float = 1.001,
    artifact_path: str | Path = "theory33_frozen_variants.json",
) -> Theory33ProductionSolver:
    return Theory33ProductionSolver(
        PeriodicGrid(shape, (1.0, 1.0), method="fd2"),
        production=Theory3ProductionParameters(
            flux_reconstruction="characteristic_weno5_z",
            source_splitting="strang",
            positivity_preserving=positivity,
            positivity_floor_factor=positivity_floor_factor,
            phase_transition_iterations=2,
        ),
        constitutive_closure=QualifiedFrozenClosure(artifact_path),
    )


def _phase_boundary_state(
    solver: Theory33ProductionSolver,
) -> object:
    x, y = solver.grid.coordinates()
    phase_patch = (x < 0.5) ^ (y < 0.5)
    density = np.where(phase_patch, 0.12, 0.045)
    internal_energy = np.where(phase_patch, 0.18, 0.06)
    fluid_velocity = solver.grid.zeros((3,))
    fluid_velocity[0] = np.where(phase_patch, 0.025, -0.02)
    fluid_velocity[1] = np.where(y < 0.5, 0.015, -0.01)
    carrier_density = np.where(phase_patch, 0.008, 0.0025)
    carrier_velocity = solver.grid.zeros((3,))
    # The active quadrants sit just inside phase two.  The discontinuous
    # transport step then moves a small, reproducible set of interface cells
    # across the learned threshold instead of merely preserving a static map.
    carrier_velocity[0] = np.where(phase_patch, 0.13, -0.04)
    carrier_velocity[1] = np.where(y < 0.5, 0.04, -0.03)
    fluid = FluidPrimitive(
        density,
        internal_energy,
        fluid_velocity,
        solver.grid.zeros(),
    )
    carrier = CarrierPrimitive(carrier_density, carrier_velocity)
    return solver.initialize(
        solver.ccz4.flat_state(), fluid, carrier=carrier
    )


def phase_boundary_experiment(
    shape: tuple[int, int] = (8, 8),
    *,
    dt: float = 1.0e-3,
    artifact_path: str | Path = "theory33_frozen_variants.json",
) -> dict[str, object]:
    """Cross a two-dimensional hard interface and evolve one full split step."""
    solver = _solver(
        shape,
        positivity=True,
        artifact_path=artifact_path,
    )
    state = _phase_boundary_state(solver)
    initial = solver.recover(state)
    initial_phase = np.array(initial.master.phase_two, copy=True)

    # Supply an intentionally stale primitive cache from the opposite side of
    # both interfaces.  This emulates cells transported across a hard phase
    # boundary and forces the recovery's outer branch update to do real work.
    solver._primitive_guess = np.roll(
        solver._primitive_guess, shape[0] // 2, axis=1
    )
    crossed = solver.recover(state)
    recovered_phase = np.array(crossed.master.phase_two, copy=True)
    branch_updates = crossed.report.phase_crossings

    baryon_before = float(solver.grid.integrate(state.matter.D))
    carrier_before = float(
        solver.grid.integrate(
            state.target_charge / solver.master_parameters.carrier_charge
        )
    )
    result = solver.step(state, dt)
    final = solver.recover(result)
    baryon_after = float(solver.grid.integrate(result.matter.D))
    carrier_after = float(
        solver.grid.integrate(
            result.target_charge / solver.master_parameters.carrier_charge
        )
    )
    diagnostics = solver.diagnostics(result)
    phase_fraction = float(np.mean(initial_phase))
    report = {
        "shape": list(shape),
        "dt": dt,
        "initial_phase_two_fraction": phase_fraction,
        "outer_branch_updates": branch_updates,
        "branch_recovery_maximum_residual": (
            crossed.report.maximum_residual
        ),
        "phase_map_recovered_exactly": bool(
            np.array_equal(recovered_phase, initial_phase)
        ),
        "phase_changes_during_physical_step": int(
            np.count_nonzero(final.master.phase_two != initial_phase)
        ),
        "baryon_relative_balance_error": abs(
            baryon_after - baryon_before
        )
        / max(abs(baryon_before), 1.0e-300),
        "carrier_relative_balance_error": abs(
            carrier_after - carrier_before
        )
        / max(abs(carrier_before), 1.0e-300),
        "minimum_legendre_eigenvalue": (
            final.report.minimum_legendre_eigenvalue
        ),
        "minimum_thermodynamic_eigenvalue": (
            final.report.minimum_thermodynamic_eigenvalue
        ),
        "maximum_recovery_residual": final.report.maximum_residual,
        "characteristic_condition_number": (
            solver.last_characteristic_condition_number
        ),
        "positivity_limited_faces": solver.last_positivity_limited_faces,
        "minimum_positivity_theta": (
            solver.last_minimum_positivity_theta
        ),
        "stage_positivity_fallbacks": (
            solver.last_stage_positivity_fallbacks
        ),
        "entropy_change": solver.last_step_entropy_change,
        "diagnostic_recovery_failures": diagnostics["recovery_failures"],
    }
    report["qualification"] = {
        "both_hard_phases_present": 0.0 < phase_fraction < 1.0,
        "outer_branch_update_exercised": branch_updates > 0,
        "hard_phase_map_recovered": report["phase_map_recovered_exactly"],
        "physical_phase_crossing_exercised": (
            report["phase_changes_during_physical_step"] > 0
        ),
        "full_step_recovery_valid": (
            report["diagnostic_recovery_failures"] == 0
            and report["maximum_recovery_residual"] < 2.0e-7
        ),
        "legendre_margin_positive": (
            report["minimum_legendre_eigenvalue"] > 0.0
        ),
        "thermodynamic_margin_positive": (
            report["minimum_thermodynamic_eigenvalue"] > 0.0
        ),
        "periodic_number_balances_close": (
            report["baryon_relative_balance_error"] < 1.0e-11
            and report["carrier_relative_balance_error"] < 1.0e-11
        ),
        "characteristic_basis_conditioned": (
            np.isfinite(report["characteristic_condition_number"])
            and report["characteristic_condition_number"] < 1.0e4
        ),
        "entropy_nondecreasing": report["entropy_change"] >= -1.0e-10,
    }
    return report


def _positivity_candidate(
    shape: tuple[int, int],
    *,
    positivity: bool,
    cfl: float,
    positivity_floor_factor: float,
    artifact_path: str | Path,
) -> dict[str, float | int]:
    solver = _solver(
        shape,
        positivity=positivity,
        positivity_floor_factor=positivity_floor_factor,
        artifact_path=artifact_path,
    )
    x, y = solver.grid.coordinates()
    low_patch = (np.abs(x - 0.5) < 0.13) & (
        np.abs(y - 0.5) < 0.13
    )
    material_floor = (
        solver.grhd.parameters.density_floor
        * positivity_floor_factor
    )
    carrier_floor = (
        solver.master_parameters.carrier_floor
        * positivity_floor_factor
    )
    density = np.where(low_patch, 1.01 * material_floor, 0.12)
    carrier_density = np.where(
        low_patch, 1.01 * carrier_floor, 0.006
    )
    internal_energy = np.where(low_patch, 0.03, 0.2)
    fluid_velocity = solver.grid.zeros((3,))
    fluid_velocity[0] = np.where(x < 0.5, -0.25, 0.25)
    fluid_velocity[1] = np.where(y < 0.5, -0.25, 0.25)
    carrier_velocity = 0.8 * fluid_velocity
    state = solver.initialize(
        solver.ccz4.flat_state(),
        FluidPrimitive(
            density,
            internal_energy,
            fluid_velocity,
            solver.grid.zeros(),
        ),
        carrier=CarrierPrimitive(carrier_density, carrier_velocity),
    )
    values = solver._pack(state)
    dt = cfl * min(solver.grid.spacing) / solver.grid.ndim
    solver._active_stage_dt = dt
    slope = solver.rhs(0.0, values)
    solver._active_stage_dt = None
    candidate_D = values[9] + dt * slope[9]
    candidate_carrier = values[22] + dt * slope[22]
    return {
        "minimum_material_floor_ratio": float(
            np.min(candidate_D) / material_floor
        ),
        "minimum_carrier_floor_ratio": float(
            np.min(candidate_carrier)
            / (
                carrier_floor
                * solver.master_parameters.carrier_charge
            )
        ),
        "limited_faces": solver.last_positivity_limited_faces,
        "minimum_theta": solver.last_minimum_positivity_theta,
        "characteristic_condition_number": (
            solver.last_characteristic_condition_number
        ),
    }


def positivity_torture_experiment(
    shape: tuple[int, int] = (16, 16),
    *,
    cfl: float = 0.2,
    positivity_floor_factor: float = 1.0e7,
    artifact_path: str | Path = "theory33_frozen_variants.json",
) -> dict[str, object]:
    """Expose an unlimited WENO undershoot and qualify the conservative fix."""
    control = _positivity_candidate(
        shape,
        positivity=False,
        cfl=cfl,
        positivity_floor_factor=positivity_floor_factor,
        artifact_path=artifact_path,
    )
    protected = _positivity_candidate(
        shape,
        positivity=True,
        cfl=cfl,
        positivity_floor_factor=positivity_floor_factor,
        artifact_path=artifact_path,
    )
    report = {
        "shape": list(shape),
        "cfl": cfl,
        "positivity_floor_factor": positivity_floor_factor,
        "unlimited_control": control,
        "protected": protected,
    }
    report["qualification"] = {
        "unlimited_control_exposes_undershoot": (
            min(
                control["minimum_material_floor_ratio"],
                control["minimum_carrier_floor_ratio"],
            )
            < 1.0
        ),
        "material_floor_preserved": (
            protected["minimum_material_floor_ratio"]
            >= 1.0 - 2.0e-12
        ),
        "carrier_floor_preserved": (
            protected["minimum_carrier_floor_ratio"]
            >= 1.0 - 2.0e-12
        ),
        "local_face_limiter_exercised": protected["limited_faces"] > 0,
        "limiter_is_not_global_first_order": (
            0 < protected["limited_faces"] < 2 * shape[0] * shape[1]
        ),
        "characteristic_basis_conditioned": (
            np.isfinite(protected["characteristic_condition_number"])
            and protected["characteristic_condition_number"] < 1.0e4
        ),
    }
    return report


def run_shock_frontier(
    *,
    phase_shape: tuple[int, int] = (8, 8),
    positivity_shape: tuple[int, int] = (16, 16),
    artifact_path: str | Path = "theory33_frozen_variants.json",
) -> dict[str, object]:
    phase = phase_boundary_experiment(
        phase_shape, artifact_path=artifact_path
    )
    positivity = positivity_torture_experiment(
        positivity_shape, artifact_path=artifact_path
    )
    qualification = {
        **{
            f"phase_{name}": bool(value)
            for name, value in phase["qualification"].items()
        },
        **{
            f"positivity_{name}": bool(value)
            for name, value in positivity["qualification"].items()
        },
    }
    return {
        "schema": "tesseract.characteristic-shock-frontier.v1",
        "reconstruction": {
            "method": "local-LF-split characteristic WENO-Z",
            "material_block": (
                "frozen relativistic acoustic/contact basis over "
                "(D,S_i,E,entropy,tracer)"
            ),
            "carrier_block": (
                "frozen Carter charge/canonical-momentum basis"
            ),
            "hard_decisions": (
                "piecewise constant inside Newton; updated between solves"
            ),
        },
        "phase_boundary": phase,
        "positivity_torture": positivity,
        "qualification": qualification,
        "qualified": all(qualification.values()),
    }
