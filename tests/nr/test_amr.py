import unittest

import numpy as np

from tesseract_nr.amr import (
    BergerColellaStepper,
    FluxRegister,
    Theory3AMRFields,
    Theory3FluxRegisters,
    conservative_prolong,
    conservative_restrict,
    prolong_theory3_fields,
    restrict_theory3_fields,
)
from tesseract_nr.adm import flat_metric
from tesseract_nr.grid import PeriodicGrid
from tesseract_nr.theory3 import Theory3System


class AMRTests(unittest.TestCase):
    def test_prolongation_is_parent_conservative_in_three_dimensions(self):
        rng = np.random.default_rng(7)
        coarse = rng.normal(size=(2, 4, 5, 6))
        fine = conservative_prolong(coarse, 3)
        self.assertEqual(fine.shape, (2, 8, 10, 12))
        np.testing.assert_allclose(conservative_restrict(fine, 3), coarse, atol=1e-15)

    def test_flux_register_restores_global_interface_conservation(self):
        register = FluxRegister((8,), (0.5,), ((2, 6),))
        coarse = np.zeros(8)
        # Fine flux exceeds coarse flux by 3 at both patch faces. Corrections
        # enter with opposite signs and therefore conserve the global sum.
        register.accumulate(0, -1, np.array(2.0), np.array(5.0), 0.1)
        register.accumulate(0, +1, np.array(2.0), np.array(5.0), 0.1)
        corrected = register.reflux(coarse)
        self.assertAlmostEqual(float(np.sum(corrected)), 0.0)
        self.assertAlmostEqual(corrected[1], -0.6)
        self.assertAlmostEqual(corrected[6], 0.6)

    def test_subcycling_reaches_same_time_and_synchronizes(self):
        calls = []
        stepper = BergerColellaStepper(2)
        def step(value, time, dt):
            calls.append((time, dt))
            return value + dt
        coarse, fine = stepper.advance(0.0, 0.0, 1.0, 0.4, step, step, lambda c, f: (c, f))
        self.assertAlmostEqual(coarse, 0.4)
        self.assertAlmostEqual(fine, 0.4)
        self.assertEqual(calls, [(1.0, 0.4), (1.0, 0.2), (1.2, 0.2)])

    def test_theory3_transfer_is_conservative_and_gauss_compatible(self):
        coarse_grid = PeriodicGrid((16,), (1.0,), "fd2")
        fine_grid = PeriodicGrid((32,), (1.0,), "fd2")
        coarse_system = Theory3System(coarse_grid)
        fine_system = Theory3System(fine_grid)
        (x,) = coarse_grid.coordinates()
        pi_A = coarse_grid.zeros((3,))
        pi_B = coarse_grid.zeros((3,))
        pi_A[0] = 0.1 * np.sin(2.0 * np.pi * x)
        pi_B[0] = -0.05 * np.cos(4.0 * np.pi * x)
        long_A, long_B = coarse_system.initial_longitudinal_charges(
            flat_metric(coarse_grid), pi_A, pi_B
        )
        fields = Theory3AMRFields(
            pi_A,
            pi_B,
            long_A,
            long_B,
            0.2 + 0.01 * np.sin(2.0 * np.pi * x),
            np.stack((0.03 * np.cos(2.0 * np.pi * x), np.zeros_like(x), np.zeros_like(x))),
        )
        fine = prolong_theory3_fields(
            fields, fine_grid, flat_metric(fine_grid)
        )
        gauss = fine_system.gauss_constraints(
            flat_metric(fine_grid),
            fine.pi_A,
            fine.pi_B,
            fine.longitudinal_A,
            fine.longitudinal_B,
        )
        np.testing.assert_allclose(gauss[0], 0.0, atol=1.0e-13)
        np.testing.assert_allclose(gauss[1], 0.0, atol=1.0e-13)
        restricted = restrict_theory3_fields(
            fine, coarse_grid, flat_metric(coarse_grid)
        )
        np.testing.assert_allclose(restricted.target_charge, fields.target_charge, atol=1e-15)
        np.testing.assert_allclose(restricted.target_current, fields.target_current, atol=1e-15)

    def test_theory3_grouped_reflux_conserves_each_charge(self):
        grid = PeriodicGrid((8,), (4.0,), "fd2")
        zeros = grid.zeros()
        fields = Theory3AMRFields(
            grid.zeros((3,)), grid.zeros((3,)), zeros.copy(), zeros.copy(),
            zeros.copy(), grid.zeros((3,)),
        )
        registers = Theory3FluxRegisters(
            grid.shape, grid.spacing, ((2, 6),)
        )
        for name, coarse_flux, fine_flux in (
            ("longitudinal_A", np.array(1.0), np.array(3.0)),
            ("longitudinal_B", np.array(-2.0), np.array(1.0)),
            ("target_charge", np.array(0.0), np.array(4.0)),
            ("target_current", np.zeros(3), np.array([1.0, 2.0, 3.0])),
        ):
            registers.accumulate(name, 0, -1, coarse_flux, fine_flux, 0.1)
            registers.accumulate(name, 0, 1, coarse_flux, fine_flux, 0.1)
        corrected = registers.reflux(fields)
        for value in (
            corrected.longitudinal_A,
            corrected.longitudinal_B,
            corrected.target_charge,
            corrected.target_current,
        ):
            np.testing.assert_allclose(np.sum(value, axis=-1), 0.0, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
