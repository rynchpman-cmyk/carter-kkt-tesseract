import tempfile
import unittest
from pathlib import Path

import numpy as np

from tesseract_nr.grid import CartesianGrid, PeriodicGrid
from tesseract_nr.matter import FluidPrimitive
from tesseract_nr.production33 import (
    Theory33ProductionSolver,
    load_theory33_production_state,
    save_theory33_production_state,
)
from tesseract_nr.theory3 import Theory3Parameters
from tesseract_nr.theory33 import (
    CarrierPrimitive,
    Theory33MasterFunction,
    Theory33MasterParameters,
)


class Theory33ConstitutiveTests(unittest.TestCase):
    def setUp(self):
        self.parameters = Theory33MasterParameters()
        self.master = Theory33MasterFunction(self.parameters)
        self.h = np.eye(3).reshape(3, 3, 1)

    def test_uncoupled_carrier_power_law_has_locked_sound_speed(self):
        p = self.parameters
        density = 0.013
        rho = p.carrier_K * density ** (1.0 + p.carrier_sound_speed**2)
        derivative = (
            p.carrier_K
            * (1.0 + p.carrier_sound_speed**2)
            * density ** p.carrier_sound_speed**2
        )
        pressure = density * derivative - rho
        self.assertAlmostEqual(pressure / rho, p.carrier_sound_speed**2, places=13)

    def test_source_current_is_timelike_with_fixed_charge_sign(self):
        fluid = FluidPrimitive(
            np.array([0.1]), np.array([1.0]), np.zeros((3, 1)), np.zeros(1)
        )
        velocity = np.array([[0.3], [0.1], [0.0]])
        carrier = CarrierPrimitive(np.array([0.01]), velocity)
        state = self.master.evaluate(self.h, fluid, carrier)
        invariant = state.source_charge_eulerian**2 - np.einsum(
            "ij...,i...,j...->...",
            self.h,
            state.source_current_up,
            state.source_current_up,
        )
        np.testing.assert_allclose(
            invariant,
            (self.parameters.carrier_charge * carrier.number_density) ** 2,
            rtol=2e-13,
        )
        self.assertGreater(state.source_charge_eulerian[0], 0.0)

    def test_master_stress_is_symmetric_and_single_fluid_limit_is_exact(self):
        parameters = Theory33MasterParameters(
            target_fraction=0.0,
            chemical_susceptibility=1.0e30,
            carrier_K=1.0e-30,
        )
        master = Theory33MasterFunction(parameters)
        density = np.array([0.1])
        internal = np.array([0.7])
        velocity = np.array([[0.2], [0.0], [0.0]])
        fluid = FluidPrimitive(density, internal, velocity, np.zeros(1))
        carrier = CarrierPrimitive(np.array([1.0e-10]), velocity.copy())
        state = master.evaluate(self.h, fluid, carrier)
        np.testing.assert_allclose(state.stress.stress, state.stress.stress.swapaxes(0, 1))
        gamma_ad = parameters.gamma_ad
        pressure = (gamma_ad - 1.0) * density * internal
        energy = density * (1.0 + internal)
        enthalpy = energy + pressure
        W = 1.0 / np.sqrt(1.0 - velocity[0] ** 2)
        expected = enthalpy * W**2 - pressure
        np.testing.assert_allclose(state.stress.rho, expected, rtol=2e-9, atol=2e-12)

    def test_drag_is_orthogonal_and_entropy_positive(self):
        fluid = FluidPrimitive(
            np.array([0.1]),
            np.array([1.0]),
            np.array([[0.15], [0.0], [0.0]]),
            np.zeros(1),
        )
        carrier = CarrierPrimitive(
            np.array([0.005]),
            np.array([[-0.2], [0.0], [0.0]]),
        )
        state = self.master.evaluate(self.h, fluid, carrier)
        force, heat = self.master.drag_force_down(self.h, fluid, carrier, state)
        W_D = 1.0 / np.sqrt(1.0 - carrier.velocity[0] ** 2)
        spatial_contraction = W_D * carrier.velocity[0] * force[0]
        resistance = state.drag_inertia / self.parameters.relaxation_time
        force_normal = resistance * (
            -state.relative_lorentz_factor + state.relative_lorentz_factor * 1.0
        )
        # Directly check the analytic invariant used by the closure.
        np.testing.assert_allclose(
            heat,
            resistance * (state.relative_lorentz_factor**2 - 1.0),
            rtol=1e-13,
        )
        self.assertGreater(heat[0], 0.0)
        self.assertTrue(np.all(np.isfinite(spatial_contraction + force_normal)))

    def test_characteristic_gate_covers_comoving_and_counterflow_states(self):
        for baryon_velocity, carrier_velocity in ((0.0, 0.0), (0.2, -0.2), (0.4, -0.4)):
            audit = self.master.characteristic_audit(
                0.1,
                1.0,
                baryon_velocity,
                0.005,
                carrier_velocity,
            )
            self.assertTrue(audit.strongly_hyperbolic)
            self.assertTrue(audit.causal)
            self.assertEqual(len(audit.speeds), 4)

    def test_carrier_loading_scan_stops_at_the_causal_boundary(self):
        for fraction in (0.5, 1.0, 2.0):
            master = Theory33MasterFunction(
                Theory33MasterParameters(target_fraction=fraction)
            )
            audit = master.characteristic_audit(
                0.05, 0.25, 0.18, fraction * 0.05, 0.46
            )
            self.assertTrue(audit.strongly_hyperbolic)
            self.assertTrue(audit.causal)
        supercritical = Theory33MasterFunction(
            Theory33MasterParameters(target_fraction=4.0)
        ).characteristic_audit(0.05, 0.25, 0.18, 0.2, 0.46)
        self.assertTrue(supercritical.strongly_hyperbolic)
        self.assertFalse(supercritical.causal)
        self.assertGreater(supercritical.maximum_absolute_speed, 1.0)

    def test_eos_falsification_states_are_causal_and_ordered(self):
        speeds = []
        for sound_speed in (0.5, 0.7, 0.9):
            master = Theory33MasterFunction(
                Theory33MasterParameters(
                    target_fraction=2.0,
                    carrier_sound_speed=sound_speed,
                )
            )
            audit = master.characteristic_audit(
                0.05, 0.25, 0.18, 0.1, 0.46
            )
            self.assertTrue(audit.strongly_hyperbolic)
            self.assertTrue(audit.causal)
            speeds.append(audit.maximum_absolute_speed)
        self.assertLess(speeds[0], speeds[1])
        self.assertLess(speeds[1], speeds[2])

    def test_kd_attack_grid_is_causal_and_exposes_low_stiffness_boundary(self):
        for carrier_K in (0.05, 0.10, 0.20, 0.40):
            audit = Theory33MasterFunction(
                Theory33MasterParameters(
                    target_fraction=2.0,
                    carrier_sound_speed=0.7,
                    carrier_K=carrier_K,
                )
            ).characteristic_audit(0.05, 0.25, 0.18, 0.1, 0.46)
            self.assertTrue(audit.strongly_hyperbolic)
            self.assertTrue(audit.causal)
        low_stiffness = Theory33MasterFunction(
            Theory33MasterParameters(
                target_fraction=2.0,
                carrier_sound_speed=0.7,
                carrier_K=0.025,
            )
        ).characteristic_audit(0.05, 0.25, 0.18, 0.1, 0.46)
        self.assertTrue(low_stiffness.strongly_hyperbolic)
        self.assertFalse(low_stiffness.causal)
        self.assertGreater(low_stiffness.maximum_absolute_speed, 1.0)

    def test_spherical_collision_widened_counterflow_domain_is_causal(self):
        master = Theory33MasterFunction(
            Theory33MasterParameters(
                target_fraction=2.0,
                carrier_K=0.4,
                carrier_sound_speed=0.7,
                maximum_relative_lorentz_factor=4.0,
            )
        )
        relative_gamma = 4.0
        rapidity = np.arccosh(relative_gamma)
        speed = np.tanh(0.5 * rapidity)
        for density in (1.0e-7, 0.005, 0.01):
            for ratio in (0.5, 1.0, 2.0, 3.0, 4.0):
                audit = master.characteristic_audit(
                    density,
                    0.25,
                    -speed,
                    ratio * density,
                    speed,
                )
                self.assertTrue(audit.strongly_hyperbolic)
                self.assertTrue(audit.causal)
                self.assertLess(audit.maximum_absolute_speed, 1.0)


class Theory33ProductionTests(unittest.TestCase):
    def _solver(self, points=8, damping=False):
        grid = PeriodicGrid((points,), (2.0,), method="fd2")
        master = Theory33MasterParameters()
        vector = Theory3Parameters(
            target_speed=0.9,
            target_relaxation_time=master.relaxation_time,
            gamma_ad=master.gamma_ad,
            gamma_W=0.2 if damping else 0.0,
            gamma_B=0.1 if damping else 0.0,
        )
        return grid, Theory33ProductionSolver(grid, master, vector)

    def _fluid_and_carrier(self, grid):
        (x,) = grid.coordinates()
        fluid = FluidPrimitive(
            1.0e-3 * (1.0 + 0.01 * np.sin(np.pi * x)),
            np.full(grid.shape, 0.1),
            grid.zeros((3,)),
            grid.zeros(),
        )
        carrier = CarrierPrimitive(
            5.0e-5 * (1.0 + 0.02 * np.cos(np.pi * x)),
            grid.zeros((3,)),
        )
        carrier.velocity[0] = 0.01 * np.sin(np.pi * x)
        return fluid, carrier

    def test_multifluid_recovery_round_trip_has_no_reservoir(self):
        grid, solver = self._solver()
        fluid, carrier = self._fluid_and_carrier(grid)
        state = solver.initialize(solver.ccz4.flat_state(), fluid, carrier=carrier)
        recovery = solver.recover(state)
        np.testing.assert_allclose(recovery.fluid.baryon_density, fluid.baryon_density, rtol=2e-10)
        np.testing.assert_allclose(recovery.carrier.number_density, carrier.number_density, rtol=2e-10)
        np.testing.assert_allclose(recovery.carrier.velocity, carrier.velocity, atol=2e-10)
        self.assertEqual(recovery.report.failed_cells, 0)
        self.assertFalse(hasattr(recovery, "reservoir_energy"))

    def test_periodic_rhs_conserves_both_number_currents_and_total_energy(self):
        grid, solver = self._solver()
        fluid, carrier = self._fluid_and_carrier(grid)
        state = solver.initialize(solver.ccz4.flat_state(), fluid, carrier=carrier)
        rhs = solver.rhs(state.time, solver._pack(state))
        self.assertAlmostEqual(grid.integrate(rhs[9]), 0.0, places=13)
        self.assertAlmostEqual(grid.integrate(rhs[11]), 0.0, places=13)
        self.assertAlmostEqual(grid.integrate(rhs[22]), 0.0, places=13)

    def test_nonuniform_coupled_step_remains_admissible(self):
        grid, solver = self._solver(points=12)
        fluid, carrier = self._fluid_and_carrier(grid)
        state = solver.initialize(solver.ccz4.flat_state(), fluid, carrier=carrier)
        baryons = grid.integrate(state.matter.D)
        carriers = grid.integrate(state.target_charge) / solver.master_parameters.carrier_charge
        state = solver.step(state, 1.0e-5)
        recovery = solver.recover(state)
        self.assertEqual(recovery.report.failed_cells, 0)
        self.assertAlmostEqual(grid.integrate(state.matter.D), baryons, places=13)
        self.assertAlmostEqual(
            grid.integrate(state.target_charge) / solver.master_parameters.carrier_charge,
            carriers,
            places=13,
        )

    def test_stage_projection_supports_practical_gradient_cfl(self):
        grid = PeriodicGrid((16,), (8.0,), method="fd2")
        (coordinate,) = grid.coordinates()
        x = coordinate - 4.0
        left = np.exp(-((x + 0.7) / 0.75) ** 2)
        right = np.exp(-((x - 0.7) / 0.75) ** 2)
        profile = left + right
        localization = profile / (profile + 0.01)
        direction = (left - right) / np.maximum(profile, 1.0e-300)
        master = Theory33MasterParameters(relaxation_time=0.1)
        vector = Theory3Parameters(
            target_speed=0.9,
            target_relaxation_time=master.relaxation_time,
            gamma_ad=master.gamma_ad,
        )
        solver = Theory33ProductionSolver(grid, master, vector)
        velocity = grid.zeros((3,))
        velocity[0] = 0.15 * direction * localization
        fluid = FluidPrimitive(
            2.0e-4 + 0.02 * profile,
            np.full(grid.shape, 0.25),
            velocity,
            grid.zeros(),
        )
        carrier_velocity = velocity.copy()
        carrier_velocity[0] += 0.15 * direction * localization
        carrier = CarrierPrimitive(
            0.05 * fluid.baryon_density, carrier_velocity
        )
        state = solver.initialize(
            solver.ccz4.flat_state(), fluid, carrier=carrier
        )
        baryons = grid.integrate(state.matter.D)
        carriers = grid.integrate(state.target_charge)
        energy = grid.integrate(state.matter.energy)
        state = solver.step(state, 1.0e-3)
        report = solver.recover(state).report
        self.assertEqual(report.failed_cells, 0)
        self.assertAlmostEqual(grid.integrate(state.matter.D), baryons, places=13)
        self.assertAlmostEqual(grid.integrate(state.target_charge), carriers, places=13)
        # Matter energy exchanges with the evolving geometry; the projection
        # itself introduces no appreciable extra balance error.
        self.assertLess(
            abs(grid.integrate(state.matter.energy) - energy) / energy,
            1.0e-6,
        )
        self.assertGreaterEqual(solver.last_projection_entropy_change, -1.0e-12)

    def test_exact_vector_damping_closes_without_reservoir(self):
        grid, solver = self._solver(damping=True)
        fluid, carrier = self._fluid_and_carrier(grid)
        pi_A = grid.zeros((3,))
        pi_B = grid.zeros((3,))
        pi_A[0] = -0.03
        pi_B[0] = 0.02
        state = solver.initialize(
            solver.ccz4.flat_state(), fluid, carrier=carrier, pi_A=pi_A, pi_B=pi_B
        )
        energy_before = grid.integrate(state.matter.energy)
        entropy_before = grid.integrate(state.matter.entropy)
        state = solver._apply_damping(state, 0.01)
        self.assertAlmostEqual(grid.integrate(state.matter.energy), energy_before, places=14)
        self.assertGreater(grid.integrate(state.matter.entropy), entropy_before)
        self.assertGreater(solver.last_damping_report.irreversible_heat, 0.0)
        self.assertLess(solver.last_damping_report.maximum_gauss_change, 1.0e-12)
        self.assertEqual(solver.recover(state).report.failed_cells, 0)

    def test_nonuniform_damping_absorbs_reversible_vector_stress_in_master_state(self):
        grid = PeriodicGrid((16,), (4.0,), method="fd2")
        (x,) = grid.coordinates()
        master = Theory33MasterParameters()
        vector = Theory3Parameters(
            target_speed=0.9,
            target_relaxation_time=master.relaxation_time,
            gamma_ad=master.gamma_ad,
            gamma_W=0.3,
            gamma_B=0.2,
        )
        solver = Theory33ProductionSolver(grid, master, vector)
        fluid = FluidPrimitive(
            np.full(grid.shape, 0.1),
            np.full(grid.shape, 1.0),
            grid.zeros((3,)),
            grid.zeros(),
        )
        carrier = CarrierPrimitive(
            0.005 * (1.0 + 0.01 * np.sin(0.5 * np.pi * x)),
            grid.zeros((3,)),
        )
        a = grid.zeros((3,))
        b = grid.zeros((3,))
        pi_A = grid.zeros((3,))
        pi_B = grid.zeros((3,))
        a[0] = 0.01 * np.cos(0.5 * np.pi * x)
        b[0] = -0.005 * np.sin(0.5 * np.pi * x)
        pi_A[0] = -0.02 * (1.0 + 0.2 * np.sin(0.5 * np.pi * x))
        pi_B[0] = 0.01 * (1.0 + 0.3 * np.cos(0.5 * np.pi * x))
        state = solver.initialize(
            solver.ccz4.flat_state(),
            fluid,
            carrier=carrier,
            a=a,
            b=b,
            pi_A=pi_A,
            pi_B=pi_B,
        )
        combined_energy = state.matter.energy.copy()
        state = solver._apply_damping(state, 0.01)
        recovery = solver.recover(state)
        np.testing.assert_allclose(state.matter.energy, combined_energy, atol=0.0)
        self.assertEqual(recovery.report.failed_cells, 0)
        self.assertGreater(solver.last_damping_report.irreversible_heat, 0.0)
        self.assertNotAlmostEqual(
            solver.last_damping_report.vector_energy_change,
            -solver.last_damping_report.irreversible_heat,
            places=10,
        )
        self.assertLess(solver.last_damping_report.maximum_gauss_change, 1.0e-12)

    def test_stiff_drag_map_is_unconditionally_decaying_and_conservative(self):
        grid = PeriodicGrid((4,), (1.0,), method="fd2")
        master = Theory33MasterParameters(
            relaxation_time=0.01,
            maximum_relative_lorentz_factor=2.0,
        )
        vector = Theory3Parameters(
            target_speed=0.9,
            target_relaxation_time=master.relaxation_time,
            gamma_ad=master.gamma_ad,
        )
        solver = Theory33ProductionSolver(grid, master, vector)
        fluid = FluidPrimitive(
            np.full(grid.shape, 0.1),
            np.full(grid.shape, 1.0),
            grid.zeros((3,)),
            grid.zeros(),
        )
        carrier = CarrierPrimitive(np.full(grid.shape, 0.005), grid.zeros((3,)))
        carrier.velocity[0] = 0.4
        state = solver.initialize(solver.ccz4.flat_state(), fluid, carrier=carrier)
        before = solver.recover(state)
        energy = state.matter.energy.copy()
        momentum = state.matter.momentum.copy()
        baryons = state.matter.D.copy()
        carriers = state.target_charge.copy()
        # Ten relaxation times would be far outside an explicit source CFL.
        state = solver._apply_carrier_drag(state, 0.1)
        after = solver.recover(state)
        self.assertLess(
            float(np.max(after.master.relative_lorentz_factor - 1.0)),
            float(np.max(before.master.relative_lorentz_factor - 1.0)) * 1.0e-4,
        )
        self.assertGreater(solver.last_drag_entropy_change, 0.0)
        np.testing.assert_allclose(state.matter.energy, energy, atol=0.0)
        np.testing.assert_allclose(state.matter.momentum, momentum, atol=0.0)
        np.testing.assert_allclose(state.matter.D, baryons, atol=0.0)
        np.testing.assert_allclose(state.target_charge, carriers, atol=0.0)

    def test_checkpoint_round_trip(self):
        grid, solver = self._solver()
        fluid, carrier = self._fluid_and_carrier(grid)
        state = solver.initialize(solver.ccz4.flat_state(), fluid, carrier=carrier)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "theory33.npz"
            save_theory33_production_state(path, state, metadata={"test": True})
            loaded, metadata = load_theory33_production_state(path)
        self.assertTrue(metadata["test"])
        np.testing.assert_allclose(loaded.target_charge, state.target_charge)
        np.testing.assert_allclose(loaded.target_current, state.target_current)
        np.testing.assert_allclose(loaded.matter.energy, state.matter.energy)

    def test_checkpoint_restart_matches_uninterrupted_next_step(self):
        grid, solver = self._solver()
        fluid, carrier = self._fluid_and_carrier(grid)
        state = solver.initialize(solver.ccz4.flat_state(), fluid, carrier=carrier)
        dt = 1.0e-5
        join_state = solver.step(state, dt)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "join.npz"
            save_theory33_production_state(path, join_state)
            restarted_state, _ = load_theory33_production_state(path)
        _, restarted_solver = self._solver()
        restarted_solver.recover(restarted_state)
        uninterrupted = solver.step(join_state, dt)
        restarted = restarted_solver.step(restarted_state, dt)
        self.assertAlmostEqual(restarted.time, uninterrupted.time, places=15)
        for restarted_field, uninterrupted_field in zip(
            restarted_solver._pack(restarted), solver._pack(uninterrupted)
        ):
            np.testing.assert_allclose(
                restarted_field,
                uninterrupted_field,
                rtol=2.0e-11,
                atol=2.0e-13,
            )

    def test_finite_boundary_allows_outgoing_carrier_without_wraparound(self):
        grid = CartesianGrid((8,), (2.0,), method="fd2")
        master = Theory33MasterParameters()
        vector = Theory3Parameters(
            target_speed=0.9,
            target_relaxation_time=master.relaxation_time,
            gamma_ad=master.gamma_ad,
        )
        solver = Theory33ProductionSolver(grid, master, vector)
        fluid = FluidPrimitive(
            np.full(grid.shape, 0.1),
            np.full(grid.shape, 1.0),
            grid.zeros((3,)),
            grid.zeros(),
        )
        carrier = CarrierPrimitive(np.full(grid.shape, 0.005), grid.zeros((3,)))
        carrier.velocity[0] = 0.2
        state = solver.initialize(solver.ccz4.flat_state(), fluid, carrier=carrier)
        rhs = solver.rhs(state.time, solver._pack(state))
        self.assertLess(grid.integrate(rhs[22]), 0.0)
        outward_rate = solver.carrier_boundary_number_outflow_rate(state)
        self.assertGreater(outward_rate, 0.0)
        self.assertAlmostEqual(
            grid.integrate(rhs[22]) / master.carrier_charge,
            -outward_rate,
            places=14,
        )

    def test_rk4_carrier_change_closes_against_boundary_flux(self):
        grid = CartesianGrid((8,), (2.0,), method="fd2")
        master = Theory33MasterParameters()
        solver = Theory33ProductionSolver(grid, master)
        fluid = FluidPrimitive(
            np.full(grid.shape, 0.1),
            np.full(grid.shape, 1.0),
            grid.zeros((3,)),
            grid.zeros(),
        )
        carrier = CarrierPrimitive(np.full(grid.shape, 0.005), grid.zeros((3,)))
        carrier.velocity[0] = 0.2
        state = solver.initialize(solver.ccz4.flat_state(), fluid, carrier=carrier)
        updated = solver.step(state, 1.0e-4)
        measured_change = (
            grid.integrate(updated.target_charge - state.target_charge)
            / master.carrier_charge
        )
        self.assertAlmostEqual(
            solver.last_carrier_number_change, measured_change, places=15
        )
        self.assertAlmostEqual(
            measured_change,
            -solver.last_carrier_boundary_number_outflow,
            places=14,
        )
        self.assertLess(abs(solver.last_carrier_number_balance_residual), 1.0e-15)


if __name__ == "__main__":
    unittest.main()
