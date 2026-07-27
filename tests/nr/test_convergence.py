import unittest

import numpy as np

from tesseract_nr.adm import ADMState, flat_metric
from tesseract_nr.ccz4 import CCZ4Parameters, CCZ4Solver
from tesseract_nr.grhd import GRHDParameters, ValenciaGRHD
from tesseract_nr.grid import PeriodicGrid
from tesseract_nr.matter import FluidPrimitive


class ConvergenceTests(unittest.TestCase):
    def _gauge_wave_error(self, points):
        grid = PeriodicGrid((points,), (1.0,), "fd4")
        solver = CCZ4Solver(
            grid,
            CCZ4Parameters(kappa1=0.1, slicing="harmonic", evolve_shift=False),
        )
        (x,) = grid.coordinates()
        amplitude = 0.01
        k = 2.0 * np.pi

        def exact(time):
            H = 1.0 - amplitude * np.sin(k * (x - time))
            h = grid.zeros((3, 3))
            h[0, 0] = H
            h[1, 1] = h[2, 2] = 1.0
            K = grid.zeros((3, 3))
            K[0, 0] = -amplitude * k * np.cos(k * (x - time)) / (2.0 * np.sqrt(H))
            return solver.from_adm(ADMState(h, K, time), lapse=np.sqrt(H))

        final_time = 0.02
        state = exact(0.0)
        dt = 0.1 / points
        steps = int(np.ceil(final_time / dt))
        for _ in range(steps):
            state = solver.step(state, final_time / steps)
        numerical = solver.to_adm(state).h
        reference = solver.to_adm(exact(final_time)).h
        return float(np.sqrt(np.mean((numerical - reference) ** 2)))

    def test_ccz4_gauge_wave_is_fourth_order(self):
        coarse = self._gauge_wave_error(16)
        fine = self._gauge_wave_error(32)
        self.assertGreater(coarse / fine, 10.0)

    def _advection_error(self, points):
        grid = PeriodicGrid((points,), (1.0,), "fd2")
        h = flat_metric(grid)
        solver = ValenciaGRHD(grid, GRHDParameters(density_floor=1e-10))
        (x,) = grid.coordinates()
        velocity = grid.zeros((3,))
        velocity[0] = 0.3
        fluid = FluidPrimitive(
            np.ones(grid.shape),
            np.full(grid.shape, 0.1),
            velocity,
            np.sin(2.0 * np.pi * x),
        )
        state = solver.primitives_to_conserved(fluid, h)
        # A longer interval makes accumulated spatial truncation dominate the
        # startup/time-discretization transient at these modest resolutions.
        final_time = 0.1
        dt = 0.2 * grid.spacing[0]
        steps = int(np.ceil(final_time / dt))
        for _ in range(steps):
            state = solver.step(
                state,
                final_time / steps,
                h,
                grid.zeros((3, 3)),
                np.ones(grid.shape),
                grid.zeros((3,)),
            )
        recovered, _ = solver.conserved_to_primitives(state, h)
        exact = np.sin(2.0 * np.pi * ((x - 0.3 * final_time) % 1.0))
        return float(np.sqrt(np.mean((recovered.sigma - exact) ** 2)))

    def test_grhd_smooth_advection_is_second_order(self):
        coarse = self._advection_error(32)
        fine = self._advection_error(64)
        self.assertGreater(coarse / fine, 3.0)


if __name__ == "__main__":
    unittest.main()
