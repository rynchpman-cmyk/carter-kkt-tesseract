from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from theory33_experiments import (
    ConvexDissipationPotential,
    ConvexDriftResidual,
    MonotoneBranchNetwork,
)


ROOT = Path(__file__).resolve().parents[1]


class Theory33ExperimentTests(unittest.TestCase):
    def test_monotone_lattice_has_certified_slope(self) -> None:
        torch.manual_seed(41)
        branch = MonotoneBranchNetwork()
        coordinate = torch.linspace(-1.0, 1.0, 257).requires_grad_(True)
        value = branch(coordinate)
        derivative = torch.autograd.grad(value.sum(), coordinate)[0]
        self.assertLessEqual(float(derivative.max()), 0.0)
        self.assertLessEqual(
            float(derivative.abs().max()),
            float(branch.slope_bound) + 1.0e-12,
        )

    def test_dissipation_is_normalized_and_locally_convex(self) -> None:
        torch.manual_seed(42)
        potential = ConvexDissipationPotential()
        origin = torch.zeros(2, requires_grad=True)
        value = potential(origin[None, :])[0]
        gradient = torch.autograd.grad(value, origin, create_graph=True)[0]
        hessian = torch.autograd.functional.hessian(
            lambda force: potential(force[None, :])[0], origin
        )
        self.assertAlmostEqual(float(value.detach()), 0.0, places=14)
        torch.testing.assert_close(
            gradient, torch.zeros_like(gradient), atol=1.0e-14, rtol=0.0
        )
        self.assertGreater(float(torch.linalg.eigvalsh(hessian)[0]), 0.0)

    def test_drift_residual_is_zero_anchored_and_convex(self) -> None:
        network = ConvexDriftResidual()
        drift = torch.linspace(0.0, 1.0, 257).requires_grad_(True)
        value = network(drift)
        derivative = torch.autograd.grad(
            value.sum(), drift, create_graph=True
        )[0]
        second_derivative = torch.autograd.grad(
            derivative.sum(), drift
        )[0]
        self.assertAlmostEqual(float(value[0].detach()), 0.0, places=14)
        self.assertAlmostEqual(float(derivative[0].detach()), 0.0, places=14)
        self.assertGreaterEqual(
            float(second_derivative.min().detach()), 0.0
        )

    def test_recorded_full_experiment_ladder_passes(self) -> None:
        document = json.loads(
            (ROOT / "theory33_experiment_results.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(document["format"], "theory33-experiment-ladder")
        self.assertEqual(document["version"], 1)
        self.assertFalse(document["quick"])
        self.assertTrue(document["passed"])
        self.assertEqual(len(document["experiments"]), 5)
        self.assertTrue(
            all(experiment["passed"] for experiment in document["experiments"])
        )


if __name__ == "__main__":
    unittest.main()
