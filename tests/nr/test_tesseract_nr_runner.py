from __future__ import annotations

import argparse
import unittest

from run_tesseract_nr import run


class TesseractNRRunnerTests(unittest.TestCase):
    def test_coupled_smoke_run_is_qualified_and_conservative(self) -> None:
        result = run(
            argparse.Namespace(
                points=8,
                length=4.0,
                steps=1,
                dt=1.0e-5,
                constraint_iterations=2,
                checkpoint=None,
            )
        )
        self.assertTrue(result["qualified"], result)
        self.assertEqual(result["state_arrays"], 24)
        self.assertEqual(result["scalar_components_per_cell"], 52)
        self.assertTrue(result["initial_constraints"]["converged"])
        balance = result["relative_balance_errors"]
        self.assertLess(balance["baryon_mass"], 1.0e-12)
        self.assertLess(balance["carrier_number"], 1.0e-12)


if __name__ == "__main__":
    unittest.main()
