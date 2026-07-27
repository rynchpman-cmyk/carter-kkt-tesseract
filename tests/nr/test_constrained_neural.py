from __future__ import annotations

import unittest

import numpy as np

from tesseract_nr.constrained_convergence import (
    balanced_active_counterflow,
    run_constrained_convergence,
)
from tesseract_nr.frozen_closure import QualifiedFrozenClosure
from tesseract_nr.grid import PeriodicGrid
from tesseract_nr.production33 import Theory33ProductionSolver


class ConstrainedNeuralSpacetimeTests(unittest.TestCase):
    def test_active_counterflow_is_pointwise_momentum_compatible(self) -> None:
        grid = PeriodicGrid((8,), (1.0,), method="fd2")
        solver = Theory33ProductionSolver(
            grid,
            constitutive_closure=QualifiedFrozenClosure(),
        )
        h, _ = solver.ccz4.physical_geometry(solver.ccz4.flat_state())
        fluid, carrier, residual = balanced_active_counterflow(solver, h)
        master = solver.master.evaluate(h, fluid, carrier)
        self.assertLess(residual, 1.0e-14)
        self.assertTrue(bool(np.all(master.phase_two)))
        self.assertGreater(
            float(
                np.min(
                    master.relative_lorentz_factor
                    - 1.0
                    - master.phase_threshold
                )
            ),
            1.0e-3,
        )

    def test_small_constrained_ladder_passes_joint_gates(self) -> None:
        report = run_constrained_convergence(
            (4, 8, 16),
            final_time=2.0e-7,
        )
        self.assertTrue(report["qualified"])
        self.assertGreater(
            report["qualified_frozen"]["final_constraints"][
                "hamiltonian"
            ]["medium_fine_order"],
            1.8,
        )
        self.assertGreater(
            report["qualified_frozen"]["final_constraints"]["momentum"][
                "medium_fine_order"
            ],
            1.8,
        )
        self.assertGreater(
            report["qualified_frozen"]["evolved_state"]["observed_order"],
            1.8,
        )


if __name__ == "__main__":
    unittest.main()
