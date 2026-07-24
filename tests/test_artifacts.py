from __future__ import annotations

import json
import math
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ArtifactTests(unittest.TestCase):
    def load_model(self, name: str) -> dict:
        return json.loads((ROOT / name).read_text(encoding="utf-8"))

    def test_portable_models_have_expected_shape(self) -> None:
        for name in ("hybrid_model.json", "hybrid_model_conditioned.json"):
            with self.subTest(name=name):
                model = self.load_model(name)
                self.assertEqual(model["format"], "carter-tesseract-hybrid")
                self.assertEqual(model["version"], 1)
                self.assertEqual(model["master_width"], 32)
                self.assertEqual(len(model["master_weights"]), 1287)
                self.assertTrue(
                    all(math.isfinite(value)
                        for value in model["master_weights"])
                )
                self.assertEqual(len(model["target_t"]), 4)
                self.assertTrue(
                    all(len(row) == 4 for row in model["target_t"])
                )

    def test_conditioned_audit_matches_milestone(self) -> None:
        model = self.load_model("hybrid_model_conditioned.json")
        metrics = model["conditioning"]
        self.assertEqual(metrics["steps"], 128)
        self.assertTrue(metrics["kkt_valid"])
        self.assertGreater(metrics["min_kkt_margin"], 0.03)
        self.assertGreater(metrics["min_classifier_margin"], 0.02)
        self.assertGreater(metrics["min_parity_margin"], 0.10)
        self.assertLessEqual(metrics["max_lyapunov_growth"], 0.0)
        self.assertLess(metrics["terminal_lyapunov"], 1.0e-20)

    def test_intrinsic_fixed_point(self) -> None:
        model = self.load_model("hybrid_model_conditioned.json")
        c = float(model["dynamics"]["c"])
        discriminant = 1.0 - 18.0 * c
        self.assertGreater(discriminant, 0.0)
        fixed = (1.0 - 4.5 * c - math.sqrt(discriminant)) / 4.5
        mapped = 2.25 * (fixed + c) ** 2 + c
        gain = abs(4.5 * (fixed + c))
        self.assertAlmostEqual(mapped, fixed, places=14)
        self.assertAlmostEqual(fixed, 0.1019493853295916, places=14)
        self.assertLess(gain, 1.0)

    def test_generated_slang_embeds_model_dynamics(self) -> None:
        pairs = (
            ("hybrid_model.json", "carter_tesseract_full.slang"),
            (
                "hybrid_model_conditioned.json",
                "carter_tesseract_conditioned.slang",
            ),
        )
        for model_name, shader_name in pairs:
            with self.subTest(shader=shader_name):
                model = self.load_model(model_name)
                shader = (ROOT / shader_name).read_text(encoding="utf-8")
                match = re.search(
                    r"static const float PARAM_c = ([^;]+);", shader
                )
                self.assertIsNotNone(match)
                assert match is not None
                self.assertAlmostEqual(
                    float(match.group(1)),
                    float(model["dynamics"]["c"]),
                    places=15,
                )
                self.assertIn("void carterDiagnosticKernel(", shader)
                self.assertIn("void fullPhysicalTapeKernel(", shader)

    def test_documented_entrypoints_exist(self) -> None:
        expected = (
            "run.ps1",
            "run_full_slang.ps1",
            "run_progressive_full.ps1",
            "run_conditioned_literal.ps1",
            "run_train_conditioned.ps1",
            "run_visualization.ps1",
            "FULL_PHYSICS_UNROLL_REPORT.md",
            "LYAPUNOV_CONDITIONING_REPORT.md",
            "tesseract_conditioning_atlas.png",
        )
        for relative in expected:
            with self.subTest(path=relative):
                self.assertTrue((ROOT / relative).is_file())

    def test_relative_markdown_links_resolve(self) -> None:
        markdown_files = [ROOT / "README.md", ROOT / "CONTRIBUTING.md"]
        markdown_files.extend((ROOT / "docs").glob("*.md"))
        markdown_files.extend(
            (
                ROOT / "FULL_PHYSICS_UNROLL_REPORT.md",
                ROOT / "LYAPUNOV_CONDITIONING_REPORT.md",
            )
        )
        pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
        for markdown in markdown_files:
            text = markdown.read_text(encoding="utf-8")
            for destination in pattern.findall(text):
                if (
                    destination.startswith(("http://", "https://", "#"))
                    or "://" in destination
                ):
                    continue
                relative = destination.split("#", 1)[0].strip()
                if not relative:
                    continue
                target = (markdown.parent / relative).resolve()
                with self.subTest(
                    document=markdown.relative_to(ROOT),
                    destination=destination,
                ):
                    self.assertTrue(target.exists(), target)


if __name__ == "__main__":
    unittest.main()
