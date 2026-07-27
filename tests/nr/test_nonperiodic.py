import unittest

import numpy as np

from tesseract_nr.adm import flat_metric
from tesseract_nr.ccz4 import CCZ4Solver
from tesseract_nr.grhd import GRHDParameters, ValenciaGRHD
from tesseract_nr.grid import CartesianGrid, GhostZonePatch
from tesseract_nr.matter import FluidPrimitive
from tesseract_nr.boundary import RadiativeBoundary


class NonperiodicGridTests(unittest.TestCase):
    def test_fd4_one_sided_polynomial_derivative(self):
        grid = CartesianGrid((24,), (2.0,), method="fd4")
        (x,) = grid.coordinates()
        derivative = grid.derivative(x**4 - 2.0 * x**2 + 0.3 * x, 0)
        np.testing.assert_allclose(derivative, 4.0 * x**3 - 4.0 * x + 0.3, atol=2e-12)

    def test_shift_does_not_wrap_opposite_edge(self):
        grid = CartesianGrid((8,), (1.0,), method="fd2")
        field = np.arange(8.0)
        self.assertEqual(grid.shift(field, 1, 0)[0], 0.0)
        self.assertEqual(grid.shift(field, -1, 0)[-1], 7.0)

    def test_explicit_ghost_zones_constant_extrapolate(self):
        grid = CartesianGrid((8, 8), (1.0, 1.0), method="fd2")
        patch = GhostZonePatch(grid, components=(3,), ghost_width=2)
        values = np.arange(3 * 8 * 8, dtype=float).reshape(3, 8, 8)
        patch.set_interior(values)
        patch.fill_outflow()
        np.testing.assert_allclose(patch.face(0, -1, True), patch.face(0, -1))
        np.testing.assert_allclose(patch.face(1, 1, True), patch.face(1, 1))


class NonperiodicEvolutionTests(unittest.TestCase):
    def test_minkowski_is_fixed_point_on_cartesian_patch(self):
        grid = CartesianGrid((8, 8, 8), (2.0, 2.0, 2.0), method="fd4")
        solver = CCZ4Solver(grid)
        state = solver.flat_state()
        evolved = solver.step(state, 1.0e-4)
        for name in (
            "conformal_metric", "conformal_A", "conformal_factor", "trace_K",
            "theta", "gamma_hat", "lapse", "shift", "shift_driver",
        ):
            np.testing.assert_allclose(getattr(evolved, name), getattr(state, name), atol=1e-14)

    def test_fluid_outflow_loses_mass_without_opposite_edge_inflow(self):
        grid = CartesianGrid((48,), (1.0,), origin=(0.0,), method="fd2")
        solver = ValenciaGRHD(
            grid, GRHDParameters(density_floor=1e-10, internal_energy_floor=1e-9)
        )
        h = flat_metric(grid)
        (x,) = grid.coordinates()
        velocity = grid.zeros((3,))
        velocity[0] = 0.4
        primitive = FluidPrimitive(
            1e-10 + np.exp(-((x - 0.88) / 0.06) ** 2),
            np.full(grid.shape, 0.1), velocity, np.zeros(grid.shape),
        )
        state = solver.primitives_to_conserved(primitive, h)
        mass_rhs = solver.rhs(
            state, h, grid.zeros((3, 3)), np.ones(grid.shape), grid.zeros((3,))
        )[0]
        self.assertLess(grid.integrate(mass_rhs), 0.0)
        self.assertLess(abs(mass_rhs[0]), 1e-8)

    def test_z4_boundary_damps_incoming_constraint_characteristic(self):
        grid = CartesianGrid((8, 8, 8), (2.0, 2.0, 2.0), method="fd2")
        state = CCZ4Solver(grid).flat_state()
        state.theta[:] = 0.2
        theta_rhs, gamma_rhs = RadiativeBoundary(grid).z4_constraint_rhs(
            state, grid.zeros(), grid.zeros((3,)), damping=2.0
        )
        boundary = RadiativeBoundary(grid)
        # Z_n and its original derivative vanish, so preserving C_out and
        # setting dC_in=-2 C_in gives dTheta=-0.2 on the boundary.
        np.testing.assert_allclose(theta_rhs[boundary.mask], -0.2)
        np.testing.assert_allclose(theta_rhs[~boundary.mask], 0.0)
        self.assertTrue(np.all(np.isfinite(gamma_rhs)))


if __name__ == "__main__":
    unittest.main()
