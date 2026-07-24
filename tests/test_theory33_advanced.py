from __future__ import annotations

import json
from pathlib import Path
import unittest

from export_theory33_frozen_slang import render
from theory33_advanced import (
    load_frozen_variants,
    mobility_covariance_audit,
    replay_frozen_artifact,
    sha256_json_payload,
)
from theory33_kinetic_data import sha256_canonical_text


ROOT = Path(__file__).resolve().parents[1]


class Theory33AdvancedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifact_path = ROOT / "theory33_frozen_variants.json"
        cls.artifact = json.loads(
            cls.artifact_path.read_text(encoding="utf-8")
        )
        cls.results = json.loads(
            (ROOT / "theory33_advanced_results.json").read_text(
                encoding="utf-8"
            )
        )

    def test_independent_data_hashes_match(self) -> None:
        for dataset in self.artifact["datasets"].values():
            path = ROOT / dataset["path"]
            with self.subTest(path=path.name):
                self.assertEqual(
                    sha256_canonical_text(path),
                    dataset["sha256"],
                )
        generator = (ROOT / "theory33_kinetic_data.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("from theory33_hybrid", generator)
        self.assertNotIn("from theory33_experiments", generator)

    def test_frozen_variant_digest_and_results(self) -> None:
        self.assertEqual(
            sha256_json_payload(self.artifact["variants"]),
            self.artifact["variant_digest"],
        )
        self.assertTrue(self.results["passed"])
        self.assertFalse(self.results["quick"])
        self.assertEqual(
            self.results["artifact_digest"],
            self.artifact["variant_digest"],
        )
        self.assertTrue(self.results["replay"]["passed"])
        self.assertTrue(
            self.results["training"]["event_bptt"][
                "jump_aware_best_every_horizon"
            ]
        )

    def test_phase_reports_held_out_uncertainty(self) -> None:
        phase = self.results["training"]["phase_fit"]
        self.assertEqual(phase["trajectory_split"]["test"], 14)
        self.assertGreater(phase["test"]["accuracy"], 0.90)
        interval = phase["uncertainty"]["base"]
        self.assertLess(interval["lower_95"], interval["mean"])
        self.assertGreater(interval["upper_95"], interval["mean"])

    def test_mobility_is_covariant_and_psd_after_loading(self) -> None:
        _, mobility, _, _, _, _ = load_frozen_variants(self.artifact)
        audit = mobility_covariance_audit(mobility)
        self.assertTrue(audit["passed"])
        self.assertGreater(audit["minimum_mobility_eigenvalue"], 0.0)
        self.assertGreaterEqual(audit["minimum_entropy_production"], 0.0)

    def test_qualified_slang_source_is_deterministic(self) -> None:
        generated = render(
            self.artifact,
            (ROOT / "theory33_hybrid.slang").read_text(encoding="utf-8"),
        )
        tracked = (ROOT / "theory33_qualified_frozen.slang").read_text(
            encoding="utf-8"
        )
        self.assertEqual(generated, tracked)
        self.assertIn("qualifiedCovariantMobility", tracked)
        self.assertIn("qualifiedTheory33Master", tracked)

    def test_no_fit_replay_requalifies(self) -> None:
        replay = replay_frozen_artifact(
            self.artifact_path,
            audit_points=33,
            characteristic_points=5,
        )
        self.assertTrue(replay["passed"])
        self.assertFalse(replay["fit_executed"])
        self.assertEqual(max(replay["probe_errors"].values()), 0.0)


if __name__ == "__main__":
    unittest.main()
