import unittest

import numpy as np

from tesseract_nr.adm import ADMParameters, ADMSolver
from tesseract_nr.grid import CartesianGrid, PeriodicGrid
from tesseract_nr.initial_constraints import (
    CTTInitialDataSolver,
    CTTParameters,
    PeriodicCMCInitialDataSolver,
)
from tesseract_nr.initial_data import (
    TwoLobeParameters,
    constraint_solved_rotating_two_lobe_data,
)
from tesseract_nr.production import ProductionSolver


class CTTInitialDataTests(unittest.TestCase):
    def setUp(self):
        self.grid = CartesianGrid((10, 10, 10), (8.0, 8.0, 8.0), method="fd2")

    def test_vacuum_returns_exact_flat_data(self):
        solver = CTTInitialDataSolver(self.grid)
        result = solver.solve(self.grid.zeros(), self.grid.zeros((3,)))
        self.assertTrue(result.report.converged)
        self.assertEqual(result.report.iterations, 1)
        np.testing.assert_allclose(result.conformal_factor, 1.0)
        np.testing.assert_allclose(result.geometry.K, 0.0)

    def test_localized_matter_converges_and_reduces_adm_constraint(self):
        x, y, z = self.grid.coordinates()
        rho = 2.0e-4 * np.exp(-(x * x + y * y + z * z) / 2.0)
        parameters = CTTParameters(maximum_iterations=5000, tolerance=8e-7, report_interval=20)
        result = CTTInitialDataSolver(self.grid, parameters).solve(
            rho, self.grid.zeros((3,))
        )
        self.assertTrue(result.report.converged)
        self.assertGreater(float(np.max(result.conformal_factor)), 1.0)
        self.assertLess(result.report.hamiltonian_residual_l2, parameters.tolerance)
        matter = (rho, self.grid.zeros((3,)), self.grid.zeros((3, 3)))
        hamiltonian, momentum = ADMSolver(
            self.grid, ADMParameters(parameters.kappa)
        ).constraints(result.geometry, matter)
        core = (slice(2, -2),) * 3
        self.assertLess(float(np.sqrt(np.mean(hamiltonian[core] ** 2))), 2e-4)
        self.assertLess(float(np.max(np.abs(momentum))), 1e-12)

    def test_two_lobe_constructor_returns_solved_geometry(self):
        data = constraint_solved_rotating_two_lobe_data(
            self.grid,
            benchmark=TwoLobeParameters(
                peak_density=2e-5, matter_speed=0.05, beta_target=0.02
            ),
            ctt=CTTParameters(maximum_iterations=4000, tolerance=2e-6, report_interval=20),
            picard_iterations=2,
        )
        self.assertTrue(data.gravitational_constraints_solved)
        self.assertIsNotNone(data.constraint_report)
        self.assertTrue(np.all(np.isfinite(data.geometry.h)))

    def test_solved_two_lobe_initializes_full_production_state(self):
        production = ProductionSolver(self.grid)
        state, data = production.initialize_isolated_two_lobe(
            TwoLobeParameters(
                peak_density=2e-5, matter_speed=0.02, beta_target=0.01
            ),
            CTTParameters(
                maximum_iterations=3000, tolerance=3e-6, report_interval=20
            ),
            picard_iterations=2,
        )
        diagnostics = production.diagnostics(state)
        self.assertTrue(data.gravitational_constraints_solved)
        self.assertEqual(diagnostics["recovery_failures"], 0)
        self.assertTrue(np.isfinite(diagnostics["hamiltonian_l2"]))


class PeriodicCMCInitialDataTests(unittest.TestCase):
    def test_one_dimensional_compensated_collision_source_converges(self):
        grid = PeriodicGrid((32,), (8.0,), method="fd2")
        (coordinate,) = grid.coordinates()
        x = coordinate - 4.0
        left = np.exp(-((x + 0.8) / 0.7) ** 2)
        right = np.exp(-((x - 0.8) / 0.7) ** 2)
        rho = 2.0e-3 + 1.0e-2 * (left + right)
        momentum = grid.zeros((3,))
        momentum[0] = 5.0e-4 * (left - right)
        parameters = CTTParameters(
            maximum_iterations=3000,
            tolerance=2.0e-5,
            relaxation=0.3,
            report_interval=20,
        )
        result = PeriodicCMCInitialDataSolver(
            grid, parameters, expanding=False
        ).solve(rho, momentum)
        self.assertTrue(result.report.converged)
        self.assertGreater(result.report.trace_K, 0.0)
        self.assertLess(
            result.report.hamiltonian_residual_l2, parameters.tolerance
        )
        self.assertLess(
            result.report.momentum_residual_l2, parameters.tolerance
        )
        self.assertGreater(result.report.maximum_conformal_factor, 1.0)
        self.assertLess(result.report.minimum_conformal_factor, 1.0)


if __name__ == "__main__":
    unittest.main()
