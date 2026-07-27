import tempfile
import unittest
from pathlib import Path

import numpy as np

from tesseract_nr.grid import PeriodicGrid
from tesseract_nr.matter import FluidPrimitive
from tesseract_nr.production3 import (
    Theory3ProductionSolver,
    load_theory3_production_state,
    save_theory3_production_state,
)
from tesseract_nr.theory3 import Theory3Parameters, Theory3System


class Theory3FormulationTests(unittest.TestCase):
    def setUp(self):
        self.grid = PeriodicGrid((32,), (1.0,), "fd2")
        self.h = np.zeros((3, 3) + self.grid.shape)
        for i in range(3):
            self.h[i, i] = 1.0

    def test_parameter_domain_enforces_eos_speed_gap(self):
        with self.assertRaises(ValueError):
            Theory3Parameters(target_speed=0.8, sound_speed_margin=0.05)
        parameters = Theory3Parameters(target_speed=0.95)
        self.assertGreater(parameters.gauss_determinant, 0.0)

    def test_two_longitudinal_charges_reconstruct_both_gauss_constraints(self):
        system = Theory3System(self.grid)
        (x,) = self.grid.coordinates()
        pi_A = self.grid.zeros((3,))
        pi_B = self.grid.zeros((3,))
        pi_A[0] = 0.1 * np.sin(2.0 * np.pi * x)
        pi_B[0] = -0.2 * np.cos(4.0 * np.pi * x)
        long_A, long_B = system.initial_longitudinal_charges(
            self.h, pi_A, pi_B
        )
        constraint_A, constraint_B = system.gauss_constraints(
            self.h, pi_A, pi_B, long_A, long_B
        )
        np.testing.assert_allclose(constraint_A, 0.0, atol=1.0e-14)
        np.testing.assert_allclose(constraint_B, 0.0, atol=1.0e-14)
        target_charge = 0.3 + 0.1 * np.sin(2.0 * np.pi * x)
        phi_A, phi_B = system.reconstruct_phi(
            self.h, long_A, long_B, target_charge
        )
        rhs = system.parameters.gauss_matrix @ np.stack((phi_A, phi_B))
        np.testing.assert_allclose(rhs[0], long_A, atol=1.0e-13)
        np.testing.assert_allclose(
            rhs[1], long_B + target_charge / system.parameters.Z_B,
            atol=1.0e-13,
        )

    def test_lorentz_force_sign_pushes_positive_charge_along_electric_field(self):
        system = Theory3System(self.grid)
        b = self.grid.zeros((3,))
        pi_B = self.grid.zeros((3,))
        pi_B[0] = -2.0  # E_B^x=+2 with the locked canonical convention.
        rho = np.full(self.grid.shape, 3.0)
        current = self.grid.zeros((3,))
        power, force = system.drive_exchange(
            self.h, b, pi_B, rho, current
        )
        np.testing.assert_allclose(power, 0.0, atol=0.0)
        np.testing.assert_allclose(force[0], 6.0, atol=0.0)
        np.testing.assert_allclose(force[1:], 0.0, atol=0.0)

    def test_target_current_uses_exact_comoving_decomposition(self):
        system = Theory3System(self.grid)
        velocity = self.grid.zeros((3,))
        velocity[0] = 0.3
        fluid = FluidPrimitive(
            np.ones(self.grid.shape),
            np.ones(self.grid.shape),
            velocity,
            self.grid.zeros(),
        )
        relative = self.grid.zeros((3,))
        relative[0] = 0.2
        W = 1.0 / np.sqrt(1.0 - 0.3**2)
        comoving_charge = 0.4
        rho = np.full(self.grid.shape, comoving_charge * W + 0.3 * 0.2)
        recovered_q, recovered_r, residual = system.target_comoving_decomposition(
            self.h, rho, relative, fluid
        )
        _, current = system.target_current(self.h, rho, relative, fluid)
        np.testing.assert_allclose(recovered_q, comoving_charge, atol=2e-15)
        np.testing.assert_allclose(recovered_r, relative, atol=0.0)
        np.testing.assert_allclose(residual, 0.0, atol=0.0)
        np.testing.assert_allclose(
            current[0], comoving_charge * W * 0.3 + 0.2, atol=2e-15
        )

    def test_field_and_charge_rhs_preserves_satisfied_gauss_data(self):
        parameters = Theory3Parameters()
        system = Theory3System(self.grid, parameters)
        (x,) = self.grid.coordinates()
        a = self.grid.zeros((3,))
        b = self.grid.zeros((3,))
        a[1] = 0.2 * np.sin(2.0 * np.pi * x)
        b[2] = 0.1 * np.cos(2.0 * np.pi * x)
        pi_A = self.grid.zeros((3,))
        pi_B = self.grid.zeros((3,))
        pi_A[0] = 0.03 * np.sin(4.0 * np.pi * x)
        pi_B[0] = -0.04 * np.cos(2.0 * np.pi * x)
        long_A, long_B = system.initial_longitudinal_charges(
            self.h, pi_A, pi_B
        )
        rho = 0.01 * (1.0 + 0.1 * np.cos(2.0 * np.pi * x))
        current = self.grid.zeros((3,))
        current[0] = 0.02 * np.sin(2.0 * np.pi * x)
        rhs = system.rhs_fields(
            self.h,
            a,
            pi_A,
            b,
            pi_B,
            long_A,
            long_B,
            self.grid.zeros(),
            self.grid.zeros(),
            rho,
            current,
            np.ones(self.grid.shape),
            self.grid.zeros((3,)),
            1.0,
        )
        dpi_A, dpi_B = rhs[1], rhs[3]
        dlong_A, dlong_B = rhs[4], rhs[5]
        dconstraint_A = self.grid.divergence(-dpi_A / parameters.Z_A) + dlong_A
        dconstraint_B = self.grid.divergence(-dpi_B / parameters.Z_B) + dlong_B
        np.testing.assert_allclose(dconstraint_A, 0.0, atol=2.0e-13)
        np.testing.assert_allclose(dconstraint_B, 0.0, atol=2.0e-13)


class Theory3ProductionTests(unittest.TestCase):
    def _fluid(self, grid, density=0.1, internal=1.0):
        return FluidPrimitive(
            np.full(grid.shape, density),
            np.full(grid.shape, internal),
            grid.zeros((3,)),
            grid.zeros(),
        )

    def test_initial_state_recovers_fluid_and_reservoir(self):
        grid = PeriodicGrid((16,), (1.0,), "fd2")
        solver = Theory3ProductionSolver(grid)
        fluid = self._fluid(grid)
        fluid.velocity[0] = 0.2
        state = solver.initialize(
            solver.ccz4.flat_state(), fluid=fluid, reservoir_energy=0.25
        )
        recovery = solver.recover(state)
        np.testing.assert_allclose(
            recovery.fluid.baryon_density, fluid.baryon_density, atol=2.0e-13
        )
        np.testing.assert_allclose(
            recovery.fluid.specific_internal_energy,
            fluid.specific_internal_energy,
            atol=2.0e-12,
        )
        np.testing.assert_allclose(recovery.fluid.velocity, fluid.velocity, atol=2e-13)
        np.testing.assert_allclose(recovery.reservoir_energy, 0.25, atol=2e-12)

    def test_target_flux_has_relativistic_velocity_added_speeds(self):
        grid = PeriodicGrid((8,), (1.0,), "fd2")
        solver = Theory3ProductionSolver(grid)
        fluid = self._fluid(grid)
        fluid.velocity[0] = 0.4
        h = np.zeros((3, 3) + grid.shape)
        for i in range(3):
            h[i, i] = 1.0
        a, b, c, d = solver._target_longitudinal_flux_coefficients(
            fluid,
            h,
            np.ones(grid.shape),
            grid.zeros((3,)),
            0,
        )
        matrix = np.array([[a[0], b[0]], [c[0], d[0]]])
        measured = np.sort(np.linalg.eigvals(matrix).real)
        speed = solver.theory_parameters.target_speed
        expected = np.sort(
            np.array([(0.4 - speed) / (1.0 - 0.4 * speed),
                      (0.4 + speed) / (1.0 + 0.4 * speed)])
        )
        np.testing.assert_allclose(measured, expected, atol=2.0e-14)
        self.assertLess(float(np.max(np.abs(measured))), 1.0)

    def test_projected_target_geometric_source_contains_extrinsic_curvature(self):
        grid = PeriodicGrid((8,), (1.0,), "fd2")
        solver = Theory3ProductionSolver(grid)
        relative = grid.zeros((3,))
        relative[0] = 0.02
        state = solver.initialize(
            solver.ccz4.flat_state(),
            fluid=self._fluid(grid),
            target_relative_current_up=relative,
            reservoir_energy=0.2,
        )
        recovery = solver.recover(state)
        h, _ = solver.ccz4.physical_geometry(state.geometry)
        K = grid.zeros((3, 3))
        K[0, 0] = 0.1
        source = solver._target_geometric_source(state, recovery, h, K)
        np.testing.assert_allclose(source[0], 0.002, atol=2e-15)
        np.testing.assert_allclose(source[1:], 0.0, atol=2e-15)

    def test_exact_damping_preserves_gauss_and_combined_four_momentum(self):
        grid = PeriodicGrid((32,), (1.0,), "fd2")
        parameters = Theory3Parameters(gamma_W=2.0, gamma_B=0.5)
        solver = Theory3ProductionSolver(grid, parameters)
        pi_A = grid.zeros((3,))
        pi_B = grid.zeros((3,))
        pi_A[0] = -0.2
        pi_B[0] = 0.1
        state = solver.initialize(
            solver.ccz4.flat_state(),
            fluid=self._fluid(grid),
            pi_A=pi_A,
            pi_B=pi_B,
            reservoir_energy=0.2,
        )
        recovery_before = solver.recover(state)
        energy_before = state.matter.energy.copy()
        momentum_before = state.matter.momentum.copy()
        entropy_before = state.matter.entropy.copy()
        damped = solver._apply_damping(state, 0.1)
        recovery_after = solver.recover(damped)
        np.testing.assert_array_equal(damped.matter.energy, energy_before)
        np.testing.assert_array_equal(damped.matter.momentum, momentum_before)
        self.assertGreater(float(np.min(damped.matter.entropy - entropy_before)), 0.0)
        self.assertLess(solver.last_damping_report.vector_energy_change, 0.0)
        self.assertGreater(solver.last_damping_report.irreversible_heat, 0.0)
        self.assertLess(solver.last_damping_report.maximum_gauss_change, 1.0e-13)
        np.testing.assert_allclose(
            recovery_after.reservoir_energy,
            recovery_before.reservoir_energy,
            atol=2.0e-12,
        )

    def test_nonuniform_damping_deposits_complete_vector_stress_change(self):
        grid = PeriodicGrid((64,), (1.0,), "fd2")
        solver = Theory3ProductionSolver(
            grid, Theory3Parameters(gamma_W=1.5, gamma_B=0.4)
        )
        (x,) = grid.coordinates()
        pi_A = grid.zeros((3,))
        pi_B = grid.zeros((3,))
        pi_A[0] = -0.2 * np.sin(2.0 * np.pi * x)
        pi_B[0] = 0.05 * np.cos(4.0 * np.pi * x)
        state = solver.initialize(
            solver.ccz4.flat_state(),
            fluid=self._fluid(grid),
            pi_A=pi_A,
            pi_B=pi_B,
            reservoir_energy=1.0,
        )
        before = solver.recover(state)
        damped = solver._apply_damping(state, 0.08)
        after = solver.recover(damped)
        np.testing.assert_allclose(
            (after.material_stress.rho - before.material_stress.rho)
            + (after.vector_stress.rho - before.vector_stress.rho),
            0.0,
            atol=3.0e-13,
        )
        np.testing.assert_allclose(
            (after.material_stress.momentum - before.material_stress.momentum)
            + (after.vector_stress.momentum - before.vector_stress.momentum),
            0.0,
            atol=3.0e-13,
        )
        h, _ = solver.ccz4.physical_geometry(damped.geometry)
        gauss = solver.system.gauss_constraints(
            h,
            damped.pi_A,
            damped.pi_B,
            damped.longitudinal_A,
            damped.longitudinal_B,
        )
        np.testing.assert_allclose(gauss[0], 0.0, atol=3.0e-13)
        np.testing.assert_allclose(gauss[1], 0.0, atol=3.0e-13)
        # The Phi/mass correction is real: full vector loss need not equal the
        # positive electric heat that is assigned to entropy.
        self.assertGreater(
            abs(
                solver.last_damping_report.vector_energy_change
                + solver.last_damping_report.irreversible_heat
            ),
            1.0e-7,
        )

    def test_periodic_conservative_rhs_preserves_integrated_charges(self):
        grid = PeriodicGrid((32,), (1.0,), "fd2")
        solver = Theory3ProductionSolver(grid)
        (x,) = grid.coordinates()
        fluid = FluidPrimitive(
            0.1 * (1.0 + 0.01 * np.sin(2.0 * np.pi * x)),
            np.full(grid.shape, 0.5),
            grid.zeros((3,)),
            grid.zeros(),
        )
        fluid.velocity[0] = 0.05
        rho_D = 0.01 * (1.0 + 0.1 * np.cos(2.0 * np.pi * x))
        state = solver.initialize(
            solver.ccz4.flat_state(),
            fluid=fluid,
            target_charge_eulerian=rho_D,
            reservoir_energy=0.1,
        )
        recovery = solver.recover(state)
        h, K = solver.ccz4.physical_geometry(state.geometry)
        matter_rhs = solver._conservative_rhs(state, recovery, h, K)
        target_rhs = solver._target_rhs(state, recovery, h, K)
        for derivative in matter_rhs:
            self.assertLess(abs(float(np.sum(derivative))), 2.0e-11)
        self.assertLess(abs(float(np.sum(target_rhs[0]))), 2.0e-11)

    def test_nontrivial_step_keeps_both_gauss_constraints(self):
        grid = PeriodicGrid((32,), (1.0,), "fd2")
        solver = Theory3ProductionSolver(
            grid,
            Theory3Parameters(
                target_strength=0.01,
                gamma_W=0.2,
                gamma_B=0.1,
            ),
        )
        (x,) = grid.coordinates()
        fluid = self._fluid(grid, internal=0.5)
        a = grid.zeros((3,))
        b = grid.zeros((3,))
        pi_A = grid.zeros((3,))
        pi_B = grid.zeros((3,))
        a[1] = 0.02 * np.sin(2.0 * np.pi * x)
        b[1] = 0.01 * np.cos(2.0 * np.pi * x)
        pi_A[1] = -0.01 * np.cos(2.0 * np.pi * x)
        pi_B[1] = 0.005 * np.sin(2.0 * np.pi * x)
        state = solver.initialize(
            solver.ccz4.flat_state(),
            fluid=fluid,
            a=a,
            pi_A=pi_A,
            b=b,
            pi_B=pi_B,
            target_charge_eulerian=np.full(grid.shape, 0.001),
            reservoir_energy=0.1,
        )
        state = solver.step(state, 1.0e-5)
        diagnostics = solver.diagnostics(state)
        self.assertLess(diagnostics["gauss_A_l2"], 1.0e-12)
        self.assertLess(diagnostics["gauss_B_l2"], 1.0e-12)
        self.assertEqual(diagnostics["recovery_failures"], 0)

    def test_checkpoint_round_trip(self):
        grid = PeriodicGrid((8,), (1.0,), "fd2")
        solver = Theory3ProductionSolver(grid)
        state = solver.vacuum_state()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "theory3.npz"
            save_theory3_production_state(path, state, metadata={"case": "vacuum"})
            loaded, metadata = load_theory3_production_state(path)
        self.assertEqual(metadata["case"], "vacuum")
        for original, restored in zip(solver._pack(state), solver._pack(loaded)):
            np.testing.assert_array_equal(original, restored)


if __name__ == "__main__":
    unittest.main()
