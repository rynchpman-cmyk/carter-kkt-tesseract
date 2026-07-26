import unittest

import numpy as np

from tesseract_nr.adm import ADMState
from tesseract_nr.ccz4 import CCZ4Parameters, CCZ4Solver
from tesseract_nr.grid import PeriodicGrid


class CCZ4Tests(unittest.TestCase):
    def test_minkowski_is_exact_fixed_point_in_three_dimensions(self):
        grid = PeriodicGrid((4, 4, 4), (1.0, 1.0, 1.0), "fd4")
        solver = CCZ4Solver(grid)
        initial = solver.flat_state()
        evolved = solver.step(initial, 1.0e-3)
        diagnostics = solver.diagnostics(evolved)
        self.assertEqual(diagnostics.hamiltonian_l2, 0.0)
        self.assertEqual(diagnostics.momentum_l2, 0.0)
        np.testing.assert_allclose(evolved.conformal_metric, initial.conformal_metric)

    def test_adm_conversion_round_trip(self):
        grid = PeriodicGrid((16,), (2.0 * np.pi,), "spectral")
        solver = CCZ4Solver(grid)
        (x,) = grid.coordinates()
        h = grid.zeros((3, 3))
        h[0, 0] = 1.0 + 0.1 * np.sin(x)
        h[1, 1] = 1.2
        h[2, 2] = 0.9
        K = grid.zeros((3, 3))
        K[0, 0] = 0.02 * np.cos(x)
        original = ADMState(h, K)
        recovered = solver.to_adm(solver.from_adm(original))
        np.testing.assert_allclose(recovered.h, h, atol=1e-14)
        np.testing.assert_allclose(recovered.K, K, atol=1e-14)

    def test_harmonic_gauge_wave_rhs(self):
        grid = PeriodicGrid((64,), (1.0,), "spectral")
        solver = CCZ4Solver(
            grid,
            CCZ4Parameters(kappa1=0.0, slicing="harmonic", evolve_shift=False),
        )
        (x,) = grid.coordinates()
        amplitude = 0.01
        wave_number = 2.0 * np.pi

        def exact(time):
            H = 1.0 - amplitude * np.sin(wave_number * (x - time))
            h = grid.zeros((3, 3))
            h[0, 0] = H
            h[1, 1] = h[2, 2] = 1.0
            K = grid.zeros((3, 3))
            K[0, 0] = -amplitude * wave_number * np.cos(
                wave_number * (x - time)
            ) / (2.0 * np.sqrt(H))
            return solver.from_adm(ADMState(h, K, time), lapse=np.sqrt(H))

        state = exact(0.0)
        values = (
            state.conformal_metric,
            state.conformal_A,
            state.conformal_factor,
            state.trace_K,
            state.theta,
            state.gamma_hat,
            state.lapse,
            state.shift,
            state.shift_driver,
        )
        rhs = solver.rhs(
            0.0, values, (grid.zeros(), grid.zeros((3,)), grid.zeros((3, 3)))
        )
        epsilon = 1.0e-6
        before, after = exact(-epsilon), exact(epsilon)
        names = (
            "conformal_metric",
            "conformal_A",
            "conformal_factor",
            "trace_K",
            "theta",
            "gamma_hat",
            "lapse",
            "shift",
            "shift_driver",
        )
        errors = []
        for name, derivative in zip(names, rhs):
            numerical = (getattr(after, name) - getattr(before, name)) / (2.0 * epsilon)
            errors.append(np.max(np.abs(derivative - numerical)))
        self.assertLess(max(errors), 3.0e-8)

    def test_projection_enforces_algebraic_constraints(self):
        grid = PeriodicGrid((8,), (1.0,), "fd4")
        solver = CCZ4Solver(grid)
        state = solver.flat_state()
        state.conformal_metric *= 1.1
        state.conformal_A[0, 0] = 0.3
        projected = solver.project_algebraic(state)
        diagnostics = solver.diagnostics(projected)
        self.assertLess(diagnostics.determinant_linf, 1e-14)
        self.assertLess(diagnostics.trace_A_linf, 1e-14)


if __name__ == "__main__":
    unittest.main()
