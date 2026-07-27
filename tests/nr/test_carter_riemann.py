from __future__ import annotations

import unittest

import numpy as np

from tesseract_nr.carter_riemann import (
    eigensystem_audit,
    oblique_symbol_audit,
    run_carter_riemann_frontier,
)
from tesseract_nr.constrained_convergence import (
    balanced_active_counterflow,
)
from tesseract_nr.frozen_closure import QualifiedFrozenClosure
from tesseract_nr.grid import PeriodicGrid
from tesseract_nr.production3 import Theory3ProductionParameters
from tesseract_nr.production33 import Theory33ProductionSolver


class FullCarterRiemannTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = run_carter_riemann_frontier(
            (16, 32, 64), final_time=0.01, cfl=0.15
        )

    def test_complete_eigensystem_matches_independent_audit(self) -> None:
        audit = eigensystem_audit()
        self.assertTrue(all(audit["qualification"].values()))
        self.assertEqual(len(audit["full_eigenvalues"]), 9)
        self.assertLess(
            audit["maximum_acoustic_speed_difference"], 2.0e-7
        )
        self.assertGreater(audit["path_symbol_norm"], 1.0e-8)

    def test_oblique_symbol_is_rotation_consistent(self) -> None:
        audit = oblique_symbol_audit()
        self.assertTrue(all(audit["qualification"].values()))
        self.assertLess(
            audit["maximum_rotational_spectrum_difference"], 2.0e-7
        )

    def test_both_riemann_ladders_converge(self) -> None:
        self.assertTrue(self.report["qualified"])
        for name in ("counterflow_shock", "colliding_streams"):
            experiment = self.report[name]
            self.assertTrue(all(experiment["qualification"].values()))
            self.assertLess(
                experiment["successive_l1_errors"][-1],
                experiment["successive_l1_errors"][0],
            )
            self.assertGreater(experiment["observed_orders"][-1], 0.35)

    def test_full_carter_weno_runs_complete_projected_step(self) -> None:
        grid = PeriodicGrid((8,), (1.0,), method="fd2")
        solver = Theory33ProductionSolver(
            grid,
            production=Theory3ProductionParameters(
                flux_reconstruction="full_carter_weno5_z",
                positivity_preserving=True,
                source_splitting="strang",
            ),
            constitutive_closure=QualifiedFrozenClosure(),
        )
        geometry = solver.ccz4.flat_state()
        h, _ = solver.ccz4.physical_geometry(geometry)
        fluid, carrier, _ = balanced_active_counterflow(solver, h)
        state = solver.initialize(
            geometry, fluid, carrier=carrier
        )
        result = solver.step(state, 1.0e-8)
        recovery = solver.recover(result)
        self.assertEqual(recovery.report.failed_cells, 0)
        self.assertLess(
            solver.last_characteristic_maximum_imaginary_part,
            2.0e-6,
        )
        self.assertLessEqual(
            solver.last_characteristic_maximum_physical_speed,
            1.0 + 2.0e-7,
        )
        self.assertTrue(
            np.isfinite(solver.last_characteristic_condition_number)
        )


if __name__ == "__main__":
    unittest.main()
