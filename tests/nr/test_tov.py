import unittest

import numpy as np

from tesseract_nr.tov import solve_tov


class TOVTests(unittest.TestCase):
    def test_standard_gamma_two_star(self):
        solution = solve_tov(1.28e-3, dr=0.01)
        self.assertAlmostEqual(solution.gravitational_mass, 1.400, delta=0.01)
        self.assertAlmostEqual(solution.surface_radius, 9.575, delta=0.03)
        self.assertGreater(solution.baryonic_mass, solution.gravitational_mass)
        self.assertTrue(np.all(np.diff(solution.enclosed_mass) >= 0.0))
        self.assertTrue(np.all(np.diff(solution.pressure) <= 0.0))

    def test_hydrostatic_residual_converges(self):
        solution = solve_tov(1.28e-3, dr=0.01)
        r = solution.radius
        rhs = -(
            (solution.energy_density + solution.pressure)
            * (solution.enclosed_mass + 4.0 * np.pi * r**3 * solution.pressure)
            / (r * (r - 2.0 * solution.enclosed_mass))
        )
        numerical = np.gradient(solution.pressure, r)
        mask = (np.arange(r.size) > 5) & (
            solution.pressure > 1.0e-6 * solution.pressure[0]
        )
        relative_l2 = np.sqrt(np.mean((numerical[mask] - rhs[mask]) ** 2)) / np.sqrt(
            np.mean(rhs[mask] ** 2)
        )
        self.assertLess(relative_l2, 1.0e-4)


if __name__ == "__main__":
    unittest.main()
