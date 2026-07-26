import unittest

import numpy as np

from tesseract_nr.grid import PeriodicGrid


class GridTests(unittest.TestCase):
    def test_spectral_derivative_is_exact_for_resolved_mode(self):
        grid = PeriodicGrid((32,), (2.0 * np.pi,), "spectral")
        (x,) = grid.coordinates()
        field = np.sin(3.0 * x)
        np.testing.assert_allclose(grid.derivative(field, 0), 3.0 * np.cos(3.0 * x), atol=1e-12)

    def test_divergence_and_curl(self):
        grid = PeriodicGrid((24, 20), (2.0 * np.pi, 2.0 * np.pi), "spectral")
        x, y = grid.coordinates()
        vector = grid.zeros((3,))
        vector[0] = np.sin(x)
        vector[1] = np.cos(2.0 * y)
        divergence = np.cos(x) - 2.0 * np.sin(2.0 * y)
        np.testing.assert_allclose(grid.divergence(vector), divergence, atol=1e-11)
        np.testing.assert_allclose(grid.divergence(grid.curl(vector)), 0.0, atol=1e-11)


if __name__ == "__main__":
    unittest.main()
