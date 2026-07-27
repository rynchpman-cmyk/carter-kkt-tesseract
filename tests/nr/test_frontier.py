from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from tesseract_full_stack_visualizer import (
    build_artifact_spacetime_slice,
    load_frontier_results,
)
from tesseract_nr.constrained_convergence import balanced_active_counterflow
from tesseract_nr.frontier import _nested_restrict, reconstruction_accuracy
from tesseract_nr.frozen_closure import QualifiedFrozenClosure
from tesseract_nr.grid import PeriodicGrid
from tesseract_nr.production3 import Theory3ProductionParameters
from tesseract_nr.production33 import Theory33ProductionSolver


class NeuralSpacetimeFrontierTests(unittest.TestCase):
    def test_visualizer_loads_the_qualified_full_stack_artifact(self) -> None:
        root = Path(__file__).resolve().parents[2]
        results = load_frontier_results(
            root / "tesseract_nr_frontier_results.json"
        )
        spacetime = build_artifact_spacetime_slice(results, points=32)
        self.assertEqual(spacetime.hamiltonian.shape, (32, 32))
        self.assertEqual(spacetime.carrier_velocity.shape, (3, 32, 32))
        self.assertTrue(bool(np.all(spacetime.phase_two)))
        self.assertFalse(spacetime.live)

    def test_weno_z_interface_is_fifth_order_on_smooth_advection(self) -> None:
        report = reconstruction_accuracy((16, 32, 64, 128))
        self.assertTrue(all(report["qualification"].values()))
        self.assertGreater(
            report["schemes"]["weno5_z"]["orders"][-1], 4.5
        )

    def test_nested_restriction_covers_every_spatial_axis(self) -> None:
        fine = np.arange(64.0).reshape(8, 8)
        restricted = _nested_restrict(fine, 2, 2)
        expected = fine[::2, ::2]
        np.testing.assert_allclose(restricted, expected)

    def test_two_dimensional_counterflow_is_pointwise_compatible(self) -> None:
        grid = PeriodicGrid((6, 6), (1.0, 1.0), method="fd2")
        solver = Theory33ProductionSolver(
            grid,
            production=Theory3ProductionParameters(
                flux_reconstruction="weno5_z"
            ),
            constitutive_closure=QualifiedFrozenClosure(),
        )
        h, _ = solver.ccz4.physical_geometry(solver.ccz4.flat_state())
        fluid, carrier, residual = balanced_active_counterflow(solver, h)
        self.assertLess(residual, 1.0e-14)
        self.assertGreater(float(np.max(np.abs(carrier.velocity[1]))), 1.0e-3)
        master = solver.master.evaluate(h, fluid, carrier)
        self.assertTrue(bool(np.all(master.phase_two)))

    def test_invalid_reconstruction_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Theory3ProductionParameters(flux_reconstruction="magic")
        with self.assertRaises(ValueError):
            Theory3ProductionParameters(source_splitting="magic")

    def test_weno_strang_step_preserves_the_physics_gates(self) -> None:
        grid = PeriodicGrid((8,), (1.0,), method="fd2")
        solver = Theory33ProductionSolver(
            grid,
            production=Theory3ProductionParameters(
                flux_reconstruction="weno5_z",
                source_splitting="strang",
            ),
            constitutive_closure=QualifiedFrozenClosure(),
        )
        geometry = solver.ccz4.flat_state()
        h, _ = solver.ccz4.physical_geometry(geometry)
        fluid, carrier, _ = balanced_active_counterflow(solver, h)
        state = solver.initialize(geometry, fluid, carrier=carrier)
        result = solver.step(state, 1.0e-8)
        diagnostics = solver.diagnostics(result)
        self.assertEqual(result.time, 1.0e-8)
        self.assertEqual(diagnostics["recovery_failures"], 0)
        self.assertGreater(
            diagnostics["constitutive_minimum_switching_margin"], 1.0e-3
        )
        self.assertGreaterEqual(solver.last_step_entropy_change, -1.0e-10)


if __name__ == "__main__":
    unittest.main()
