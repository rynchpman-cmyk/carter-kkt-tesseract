#!/usr/bin/env python3
"""Run a compact coupled Tesseract numerical-relativity PDE evolution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tesseract_nr.grid import PeriodicGrid
from tesseract_nr.initial_constraints import (
    CTTParameters,
    PeriodicCMCInitialDataSolver,
)
from tesseract_nr.matter import FluidPrimitive
from tesseract_nr.production33 import (
    Theory33ProductionSolver,
    save_theory33_production_state,
)
from tesseract_nr.theory33 import CarrierPrimitive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=16)
    parser.add_argument("--length", type=float, default=8.0)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--dt", type=float, default=1.0e-5)
    parser.add_argument(
        "--constraint-iterations",
        type=int,
        default=2,
        help="number of matter/geometry Picard iterations",
    )
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def build_smooth_state(
    solver: Theory33ProductionSolver,
    *,
    constraint_iterations: int,
) -> tuple[object, dict[str, float], dict[str, object]]:
    grid = solver.grid
    (coordinate,) = grid.coordinates()
    phase = 2.0 * np.pi * coordinate / grid.lengths[0]
    fluid_velocity = grid.zeros((3,))
    carrier_velocity = grid.zeros((3,))
    fluid_velocity[0] = 0.005 * np.cos(phase)
    carrier_velocity[0] = 0.01 * np.sin(phase)
    fluid = FluidPrimitive(
        1.0e-3 * (1.0 + 0.01 * np.sin(phase)),
        np.full(grid.shape, 0.1),
        fluid_velocity,
        grid.zeros(),
    )
    carrier = CarrierPrimitive(
        5.0e-5 * (1.0 + 0.02 * np.cos(phase)),
        carrier_velocity,
    )
    state = solver.initialize(
        solver.ccz4.flat_state(),
        fluid,
        carrier=carrier,
    )
    constraints = PeriodicCMCInitialDataSolver(
        grid,
        CTTParameters(
            kappa=solver.ccz4.parameters.kappa_gravity,
            maximum_iterations=4000,
            tolerance=2.0e-7,
            relaxation=0.3,
            report_interval=20,
        ),
        expanding=False,
    )
    conformal_factor = None
    vector_potential = None
    constraint_result = None
    for _ in range(constraint_iterations):
        recovery = solver.recover(state)
        constraint_result = constraints.solve(
            recovery.total_stress.rho,
            recovery.total_stress.momentum,
            conformal_factor,
            vector_potential,
        )
        if not constraint_result.report.converged:
            raise FloatingPointError(
                "periodic CMC initial-data solve did not converge: "
                f"{constraint_result.report}"
            )
        conformal_factor = constraint_result.conformal_factor
        vector_potential = constraint_result.vector_potential
        state = solver.initialize(
            solver.ccz4.from_adm(constraint_result.geometry),
            fluid,
            carrier=carrier,
        )
    if constraint_result is None:
        raise ValueError("--constraint-iterations must be positive")
    diagnostics = solver.diagnostics(state)
    report = constraint_result.report
    return (
        state,
        {
            "baryon_mass": float(diagnostics["baryon_mass"]),
            "carrier_number": float(diagnostics["carrier_number"]),
            "combined_energy": float(diagnostics["combined_energy"]),
            "entropy_integral": float(diagnostics["entropy_integral"]),
            "hamiltonian_l2": float(diagnostics["hamiltonian_l2"]),
            "momentum_l2": float(diagnostics["momentum_l2"]),
        },
        {
            "converged": report.converged,
            "iterations": report.iterations,
            "hamiltonian_residual_l2": report.hamiltonian_residual_l2,
            "momentum_residual_l2": report.momentum_residual_l2,
            "trace_K": report.trace_K,
            "minimum_conformal_factor": report.minimum_conformal_factor,
            "maximum_conformal_factor": report.maximum_conformal_factor,
        },
    )


def relative_error(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1.0e-300)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.points < 4:
        raise ValueError("--points must be at least four")
    if args.length <= 0.0 or args.steps < 0 or args.dt <= 0.0:
        raise ValueError("length and dt must be positive; steps must be nonnegative")
    grid = PeriodicGrid((args.points,), (args.length,), method="fd2")
    solver = Theory33ProductionSolver(grid)
    state, initial, constraint_report = build_smooth_state(
        solver,
        constraint_iterations=args.constraint_iterations,
    )
    for _ in range(args.steps):
        state = solver.step(state, args.dt)
    final = solver.diagnostics(state)
    result: dict[str, object] = {
        "format": "tesseract.nr.smoke.v1",
        "points": args.points,
        "steps": args.steps,
        "dt": args.dt,
        "time": state.time,
        "state_arrays": len(solver._pack(state)),
        "scalar_components_per_cell": 52,
        "initial_constraints": constraint_report,
        "initial": initial,
        "final": final,
        "relative_balance_errors": {
            "baryon_mass": relative_error(
                float(final["baryon_mass"]),
                initial["baryon_mass"],
            ),
            "carrier_number": relative_error(
                float(final["carrier_number"]),
                initial["carrier_number"],
            ),
            "combined_energy": relative_error(
                float(final["combined_energy"]),
                initial["combined_energy"],
            ),
        },
        "qualified": (
            bool(constraint_report["converged"])
            and int(final["recovery_failures"]) == 0
            and float(final["minimum_legendre_eigenvalue"]) > 0.0
            and float(final["minimum_thermodynamic_eigenvalue"]) > 0.0
            and float(final["step_entropy_change"]) >= -1.0e-10
        ),
    }
    if args.checkpoint is not None:
        save_theory33_production_state(
            args.checkpoint,
            state,
            metadata={"runner": "run_tesseract_nr.py"},
        )
        result["checkpoint"] = str(args.checkpoint)
    return result


def main() -> int:
    result = run(parse_args())
    print(json.dumps(result, indent=2))
    return 0 if bool(result["qualified"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
