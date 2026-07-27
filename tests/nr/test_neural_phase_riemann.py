from __future__ import annotations

import json
from pathlib import Path
import unittest

from tesseract_nr.carter_riemann import (
    neural_phase_corridor_audit,
    riemann_resolution_ladder,
)
from tesseract_nr.frozen_closure import QualifiedFrozenClosure


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "tesseract_neural_phase_riemann_results.json"


class FrozenNeuralPhaseRiemannTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corridor = neural_phase_corridor_audit(samples=33)
        cls.smoke_ladder = riemann_resolution_ladder(
            "neural_phase_contact",
            (16, 32, 64),
            final_time=0.005,
            cfl=0.12,
            model="qualified_frozen",
        )

    def test_conserved_state_corridor_is_qualified(self) -> None:
        self.assertTrue(
            all(self.corridor["qualification"].values())
        )
        self.assertEqual(self.corridor["phase_transitions"], 1)
        self.assertLessEqual(
            self.corridor["maximum_physical_speed"],
            1.0 + 2.0e-7,
        )
        self.assertGreater(
            self.corridor["path_symbol_minimum_norm"], 1.0e-8
        )

    def test_short_frozen_neural_ladder_is_physically_valid(self) -> None:
        self.assertTrue(
            all(self.smoke_ladder["qualification"].values())
        )
        self.assertLess(
            self.smoke_ladder["successive_l1_errors"][-1],
            self.smoke_ladder["successive_l1_errors"][0],
        )
        for diagnostic in self.smoke_ladder["diagnostics"]:
            self.assertGreaterEqual(
                diagnostic["phase_boundary_faces"], 2
            )
            self.assertLess(
                diagnostic["baryon_relative_balance_error"], 2.0e-12
            )
            self.assertLess(
                diagnostic["carrier_relative_balance_error"], 2.0e-12
            )

    def test_tracked_ladder_contains_dynamic_phase_crossing(self) -> None:
        report = json.loads(RESULTS.read_text(encoding="utf-8"))
        self.assertTrue(report["qualified"])
        self.assertEqual(
            report["variant_digest"],
            QualifiedFrozenClosure().variant_digest,
        )
        ladder = report["resolution_ladder"]
        self.assertEqual(ladder["resolutions"], [16, 32, 64, 128])
        self.assertGreater(
            max(ladder["phase_changes_from_initial_count"]), 0
        )
        self.assertTrue(
            ladder["qualification"][
                "physical_phase_crossing_exercised"
            ]
        )


if __name__ == "__main__":
    unittest.main()
