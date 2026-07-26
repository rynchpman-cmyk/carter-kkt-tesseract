import unittest

import numpy as np

from tesseract_nr.adm import flat_metric
from tesseract_nr.boundary import (
    Theory3CharacteristicBoundary,
    zero_incoming_linear_flux,
)
from tesseract_nr.grid import CartesianGrid
from tesseract_nr.matter import FluidPrimitive
from tesseract_nr.production3 import Theory3ProductionSolver


class Theory3BoundaryTests(unittest.TestCase):
    def test_linear_characteristic_flux_removes_only_incoming_mode(self):
        matrix = np.diag([-1.0, 1.0])
        state = np.array([2.0, 3.0])
        np.testing.assert_allclose(
            zero_incoming_linear_flux(matrix, state, 1), [0.0, 3.0]
        )
        np.testing.assert_allclose(
            zero_incoming_linear_flux(matrix, state, -1), [-2.0, 0.0]
        )

    def test_radiative_mixed_proca_rhs_remains_gauss_compatible(self):
        grid = CartesianGrid((48,), (1.0,), origin=(0.0,), method="fd2")
        solver = Theory3ProductionSolver(grid)
        (x,) = grid.coordinates()
        fluid = FluidPrimitive(
            np.full(grid.shape, 0.1),
            np.full(grid.shape, 0.5),
            grid.zeros((3,)),
            grid.zeros(),
        )
        a = grid.zeros((3,))
        b = grid.zeros((3,))
        pi_A = grid.zeros((3,))
        pi_B = grid.zeros((3,))
        a[1] = 0.02 * np.exp(-((x - 0.75) / 0.1) ** 2)
        b[2] = -0.01 * np.exp(-((x - 0.7) / 0.12) ** 2)
        pi_A[0] = 0.03 * np.sin(2.0 * np.pi * x)
        pi_B[0] = -0.02 * np.cos(2.0 * np.pi * x)
        state = solver.initialize(
            solver.ccz4.flat_state(),
            fluid=fluid,
            a=a,
            b=b,
            pi_A=pi_A,
            pi_B=pi_B,
            reservoir_energy=0.2,
        )
        recovery = solver.recover(state)
        h, _ = solver.ccz4.physical_geometry(state.geometry)
        raw = solver.system.rhs_fields(
            h,
            state.a,
            state.pi_A,
            state.b,
            state.pi_B,
            state.longitudinal_A,
            state.longitudinal_B,
            state.cleaning_A,
            state.cleaning_B,
            recovery.target_charge_eulerian,
            recovery.target_current_up,
            state.geometry.lapse,
            state.geometry.shift,
            solver.production_parameters.cleaning_damping,
        )
        projected = Theory3CharacteristicBoundary(grid).mixed_proca_rhs(
            solver.system, h, state, raw
        )
        dconstraint_A = grid.divergence(
            -projected[1] / solver.theory_parameters.Z_A
        ) + projected[4]
        dconstraint_B = grid.divergence(
            -projected[3] / solver.theory_parameters.Z_B
        ) + projected[5]
        np.testing.assert_allclose(dconstraint_A, 0.0, atol=3.0e-12)
        np.testing.assert_allclose(dconstraint_B, 0.0, atol=3.0e-12)

    def test_outgoing_target_pulse_loses_charge_without_lower_inflow(self):
        grid = CartesianGrid((64,), (1.0,), origin=(0.0,), method="fd2")
        solver = Theory3ProductionSolver(grid)
        (x,) = grid.coordinates()
        fluid = FluidPrimitive(
            np.full(grid.shape, 0.1),
            np.full(grid.shape, 0.5),
            grid.zeros((3,)),
            grid.zeros(),
        )
        charge = 1.0e-5 * np.exp(-((x - 0.92) / 0.05) ** 2)
        relative = grid.zeros((3,))
        relative[0] = solver.theory_parameters.target_speed * charge
        state = solver.initialize(
            solver.ccz4.flat_state(),
            fluid=fluid,
            target_charge_eulerian=charge,
            target_relative_current_up=relative,
            reservoir_energy=0.2,
        )
        recovery = solver.recover(state)
        h, K = solver.ccz4.physical_geometry(state.geometry)
        dcharge, _ = solver._target_rhs(state, recovery, h, K)
        self.assertLess(grid.integrate(dcharge), 0.0)
        self.assertLess(abs(float(dcharge[0])), 1.0e-12)
        initial_charge = grid.integrate(state.target_charge)
        for _ in range(3):
            state = solver.step(state, 1.0e-5)
        self.assertLess(grid.integrate(state.target_charge), initial_charge)
        diagnostics = solver.diagnostics(state)
        self.assertLess(diagnostics["gauss_B_l2"], 1.0e-15)
        self.assertEqual(diagnostics["recovery_failures"], 0)

    def test_finite_grid_vacuum_is_exact_fixed_point(self):
        grid = CartesianGrid((16,), (1.0,), method="fd2")
        solver = Theory3ProductionSolver(grid)
        state = solver.vacuum_state()
        evolved = solver.step(state, 1.0e-5)
        for before, after in zip(solver._pack(state), solver._pack(evolved)):
            np.testing.assert_allclose(after, before, atol=2.0e-14)


if __name__ == "__main__":
    unittest.main()
