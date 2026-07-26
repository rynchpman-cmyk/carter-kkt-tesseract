import unittest

import numpy as np

from tesseract_nr.adm import flat_metric
from tesseract_nr.grhd import GRHDParameters, ValenciaGRHD
from tesseract_nr.grid import PeriodicGrid
from tesseract_nr.matter import FluidPrimitive


class GRHDTests(unittest.TestCase):
    def setUp(self):
        self.grid = PeriodicGrid((64,), (1.0,), "fd2")
        self.h = flat_metric(self.grid)
        self.solver = ValenciaGRHD(
            self.grid,
            GRHDParameters(density_floor=1e-10, internal_energy_floor=1e-9),
        )

    def test_primitive_recovery_round_trip(self):
        (x,) = self.grid.coordinates()
        velocity = self.grid.zeros((3,))
        velocity[0] = 0.3 + 0.05 * np.sin(2.0 * np.pi * x)
        primitive = FluidPrimitive(
            1.0 + 0.2 * np.sin(2.0 * np.pi * x),
            0.3 + 0.05 * np.cos(2.0 * np.pi * x),
            velocity,
            np.cos(4.0 * np.pi * x),
        )
        conserved = self.solver.primitives_to_conserved(primitive, self.h)
        recovered, report = self.solver.conserved_to_primitives(conserved, self.h)
        self.assertEqual(report.failed_cells, 0)
        np.testing.assert_allclose(recovered.baryon_density, primitive.baryon_density, atol=1e-13)
        np.testing.assert_allclose(recovered.specific_internal_energy, primitive.specific_internal_energy, atol=1e-13)
        np.testing.assert_allclose(recovered.velocity, primitive.velocity, atol=1e-13)

    def test_uniform_flow_is_stationary_and_conservative(self):
        velocity = self.grid.zeros((3,))
        velocity[0] = 0.4
        primitive = FluidPrimitive(
            np.ones(self.grid.shape),
            np.full(self.grid.shape, 0.2),
            velocity,
            np.zeros(self.grid.shape),
        )
        state = self.solver.primitives_to_conserved(primitive, self.h)
        rhs = self.solver.rhs(
            state,
            self.h,
            self.grid.zeros((3, 3)),
            np.ones(self.grid.shape),
            self.grid.zeros((3,)),
        )
        for derivative in rhs:
            np.testing.assert_allclose(derivative, 0.0, atol=1e-14)

    def test_periodic_update_conserves_integrated_variables(self):
        (x,) = self.grid.coordinates()
        velocity = self.grid.zeros((3,))
        velocity[0] = 0.2
        primitive = FluidPrimitive(
            1.0 + 0.1 * np.sin(2.0 * np.pi * x),
            np.full(self.grid.shape, 0.1),
            velocity,
            np.sin(2.0 * np.pi * x),
        )
        state = self.solver.primitives_to_conserved(primitive, self.h)
        initial = self.solver.diagnostics(state, self.h)
        state = self.solver.step(
            state,
            1e-3,
            self.h,
            self.grid.zeros((3, 3)),
            np.ones(self.grid.shape),
            self.grid.zeros((3,)),
        )
        final = self.solver.diagnostics(state, self.h)
        self.assertAlmostEqual(final.baryon_mass, initial.baryon_mass, places=13)
        self.assertAlmostEqual(final.energy_integral, initial.energy_integral, places=13)
        self.assertEqual(final.recovery_failures, 0)

    def test_relativistic_shock_tube_remains_physical(self):
        (x,) = self.grid.coordinates()
        primitive = FluidPrimitive(
            np.where(x < 0.5, 1.0, 0.125),
            np.where(x < 0.5, 1.5, 1.2),
            self.grid.zeros((3,)),
            np.zeros(self.grid.shape),
        )
        state = self.solver.primitives_to_conserved(primitive, self.h)
        for _ in range(5):
            state = self.solver.step(
                state,
                5.0e-4,
                self.h,
                self.grid.zeros((3, 3)),
                np.ones(self.grid.shape),
                self.grid.zeros((3,)),
            )
        diagnostics = self.solver.diagnostics(state, self.h)
        self.assertGreaterEqual(diagnostics.minimum_density, self.solver.parameters.density_floor)
        self.assertLess(diagnostics.maximum_speed, 1.0)
        self.assertEqual(diagnostics.recovery_failures, 0)


if __name__ == "__main__":
    unittest.main()
