from __future__ import annotations

import unittest

from tesseract_nr.frozen_closure import QualifiedFrozenClosure
from tesseract_nr.theory33 import Theory33MasterFunction
import numpy as np


class NRFrozenBridgeTests(unittest.TestCase):
    def test_frozen_master_and_psd_mobility_reach_nr_contract(self) -> None:
        closure = QualifiedFrozenClosure()
        density = np.array([0.08, 0.10, 0.12])
        carrier = 0.05 * density
        relative_gamma = np.array([1.001, 1.006, 1.012])
        entropy = np.ones_like(density)
        response = closure.evaluate(
            density**2,
            carrier**2,
            density * carrier * relative_gamma,
            entropy,
        )
        self.assertEqual(
            closure.variant_digest,
            "e3a8e89ca635cfe68d5bd963152097f85ec47b7960819830c9513f45b7bcf108",
        )
        self.assertTrue(np.all(np.isfinite(response.lambda_value)))
        self.assertTrue(np.all(np.isfinite(response.lambda_gradient)))
        self.assertTrue(
            np.all(np.linalg.eigvalsh(response.mobility_matrix) > 0.0)
        )
        analytic = Theory33MasterFunction().lambda_from_invariants(
            density**2,
            carrier**2,
            density * carrier * relative_gamma,
            entropy,
        )
        self.assertLess(
            float(np.max(np.abs(response.lambda_value - analytic))),
            1.0e-6,
        )
        self.assertFalse(bool(response.phase_two[0]))
        self.assertTrue(bool(response.phase_two[-1]))

    def test_invariant_gradient_matches_finite_difference(self) -> None:
        closure = QualifiedFrozenClosure()
        point = np.array([0.1**2, 0.005**2, 0.1 * 0.005 * 1.012, 1.0])
        response = closure.evaluate(*point)
        finite = np.empty(4)
        for index in range(4):
            step = 1.0e-6 * max(abs(point[index]), 1.0e-4)
            right = point.copy()
            left = point.copy()
            right[index] += step
            left[index] -= step
            finite[index] = (
                closure.evaluate(*right).lambda_value
                - closure.evaluate(*left).lambda_value
            ) / (2.0 * step)
        np.testing.assert_allclose(
            response.lambda_gradient,
            finite,
            rtol=2.0e-5,
            atol=2.0e-7,
        )

    def test_explicit_branch_evaluation_freezes_the_hard_classifier(self) -> None:
        closure = QualifiedFrozenClosure()
        point = (0.1**2, 0.005**2, 0.1 * 0.005 * 1.012, 1.0)
        classified = closure.evaluate(*point)
        phase_one = closure.evaluate_branch(*point, False)
        phase_two = closure.evaluate_branch(*point, True)
        self.assertTrue(bool(classified.phase_two))
        self.assertFalse(bool(phase_one.phase_two))
        self.assertTrue(bool(phase_two.phase_two))
        self.assertGreater(
            abs(float(phase_two.lambda_value - phase_one.lambda_value)),
            1.0e-12,
        )


if __name__ == "__main__":
    unittest.main()
