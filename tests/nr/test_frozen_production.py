from __future__ import annotations

import unittest

import numpy as np

from tesseract_nr.convergence import (
    build_qualified_counterflow,
    run_constitutive_convergence,
)
from tesseract_nr.frozen_closure import QualifiedFrozenClosure
from tesseract_nr.grid import PeriodicGrid
from tesseract_nr.production33 import Theory33ProductionSolver


class FrozenProductionTests(unittest.TestCase):
    def _solver(self, points: int, frozen: bool) -> Theory33ProductionSolver:
        grid = PeriodicGrid((points,), (1.0,), method="fd2")
        return Theory33ProductionSolver(
            grid,
            constitutive_closure=(
                QualifiedFrozenClosure() if frozen else None
            ),
        )

    def test_frozen_gradient_drives_recovery_and_pde_rhs(self) -> None:
        analytic = self._solver(8, False)
        frozen = self._solver(8, True)
        analytic_state = build_qualified_counterflow(analytic)
        frozen_state = build_qualified_counterflow(frozen)
        recovery = frozen.recover(frozen_state)

        self.assertEqual(recovery.master.constitutive_model, "qualified_frozen")
        self.assertTrue(bool(np.all(recovery.master.phase_two)))
        np.testing.assert_allclose(
            recovery.master.B_N,
            -2.0 * recovery.master.invariant_gradient[..., 0],
        )
        np.testing.assert_allclose(
            recovery.master.B_D,
            -2.0 * recovery.master.invariant_gradient[..., 1],
        )
        np.testing.assert_allclose(
            recovery.master.entrainment,
            -recovery.master.invariant_gradient[..., 2],
        )
        self.assertEqual(recovery.report.failed_cells, 0)

        analytic_rhs = analytic.rhs(
            analytic_state.time,
            analytic._pack(analytic_state),
        )
        frozen_rhs = frozen.rhs(
            frozen_state.time,
            frozen._pack(frozen_state),
        )
        maximum_difference = max(
            float(np.max(np.abs(left - right)))
            for left, right in zip(analytic_rhs, frozen_rhs, strict=True)
        )
        self.assertGreater(maximum_difference, 1.0e-9)
        self.assertAlmostEqual(frozen.grid.integrate(frozen_rhs[9]), 0.0, places=13)
        self.assertAlmostEqual(frozen.grid.integrate(frozen_rhs[11]), 0.0, places=13)
        self.assertAlmostEqual(frozen.grid.integrate(frozen_rhs[22]), 0.0, places=13)

        evolved = frozen.step(frozen_state, 1.0e-6)
        diagnostics = frozen.diagnostics(evolved)
        self.assertEqual(diagnostics["recovery_failures"], 0)
        self.assertEqual(diagnostics["constitutive_model"], "qualified_frozen")
        self.assertGreater(
            float(diagnostics["constitutive_phase_two_fraction"]),
            0.99,
        )
        self.assertGreater(
            float(diagnostics["constitutive_minimum_switching_margin"]),
            1.0e-3,
        )

    def test_small_nested_ladder_passes_constitutive_gates(self) -> None:
        report = run_constitutive_convergence(
            (4, 8, 16),
            final_time=1.0e-6,
        )
        self.assertTrue(report["qualified"])
        self.assertGreater(
            report["finest_model_difference"]["rhs_l2"],
            1.0e-14,
        )
        self.assertGreater(
            report["qualified_frozen"]["evolved"]["observed_order"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
