from __future__ import annotations

import unittest

from tesseract_nr.shock_frontier import run_shock_frontier


class CharacteristicShockFrontierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = run_shock_frontier()

    def test_every_shock_qualification_gate_passes(self) -> None:
        self.assertTrue(self.report["qualified"])
        self.assertTrue(all(self.report["qualification"].values()))

    def test_positivity_control_is_falsifiable(self) -> None:
        experiment = self.report["positivity_torture"]
        control = experiment["unlimited_control"]
        protected = experiment["protected"]
        self.assertLess(
            min(
                control["minimum_material_floor_ratio"],
                control["minimum_carrier_floor_ratio"],
            ),
            1.0,
        )
        self.assertGreaterEqual(
            protected["minimum_material_floor_ratio"], 1.0 - 2.0e-12
        )
        self.assertGreaterEqual(
            protected["minimum_carrier_floor_ratio"], 1.0 - 2.0e-12
        )
        self.assertGreater(protected["limited_faces"], 0)
        self.assertGreater(protected["minimum_theta"], 0.0)

    def test_multidimensional_hard_phase_crossing_is_real(self) -> None:
        experiment = self.report["phase_boundary"]
        self.assertEqual(experiment["outer_branch_updates"], 64)
        self.assertGreater(
            experiment["phase_changes_during_physical_step"], 0
        )
        self.assertEqual(experiment["diagnostic_recovery_failures"], 0)
        self.assertLess(experiment["baryon_relative_balance_error"], 1.0e-14)
        self.assertLess(
            experiment["carrier_relative_balance_error"], 1.0e-14
        )


if __name__ == "__main__":
    unittest.main()
