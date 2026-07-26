import unittest

import numpy as np

from tesseract_nr.adm import ADMState, flat_metric
from tesseract_nr.ccz4 import CCZ4Solver
from tesseract_nr.grid import CartesianGrid
from tesseract_nr.nr_diagnostics import (
    ApparentHorizonFinder,
    HawkingMassDiagnostic,
    Psi4Extractor,
    sphere_quadrature,
)
from tesseract_nr.kkt_horizon import (
    CCZ4PhysicalNormalMOTSTracker,
    DirectBrillLindquistMOTSFinder,
    MOTSBranchDefinition,
    MOTSBranchObservation,
    MOTSBranchTracker,
    NormalMOTSStabilityOperator,
    ParametricAxisymmetricMOTSFinder,
    ParametricMOTSFinder,
    SpectralMOTSFinder,
    SpectralMOTSParameters,
    real_spherical_harmonic,
)


class NRDiagnosticsTests(unittest.TestCase):
    @staticmethod
    def isotropic_schwarzschild(grid, mass=1.0, time=0.0):
        x, y, z = grid.coordinates()
        radius = np.sqrt(x * x + y * y + z * z)
        psi = 1.0 + mass / (2.0 * radius)
        h = grid.zeros((3, 3))
        for i in range(3):
            h[i, i] = psi**4
        return ADMState(h, grid.zeros((3, 3)), time)

    def test_minkowski_has_zero_psi4(self):
        grid = CartesianGrid((8, 8, 8), (6.0, 6.0, 6.0), method="fd2")
        state = ADMState(flat_metric(grid), grid.zeros((3, 3)))
        extraction = Psi4Extractor(grid, 6, 12).extract(state, 1.5)
        np.testing.assert_allclose(extraction.psi4, 0.0, atol=1e-14)
        for mode in extraction.modes_l2.values():
            self.assertAlmostEqual(abs(mode), 0.0, places=13)

    def test_isotropic_schwarzschild_horizon_is_near_m_over_two(self):
        grid = CartesianGrid((32, 32, 32), (6.0, 6.0, 6.0), method="fd4")
        state = self.isotropic_schwarzschild(grid)
        result = ApparentHorizonFinder(grid, 10, 20).find(state, 0.3, 0.8, tolerance=2e-3)
        self.assertTrue(result.found)
        self.assertAlmostEqual(result.coordinate_radius, 0.5, delta=0.08)
        self.assertAlmostEqual(result.irreducible_mass, 1.0, delta=0.15)

    def test_hawking_mass_vanishes_in_minkowski_and_recovers_schwarzschild(self):
        grid = CartesianGrid((32, 32, 32), (8.0, 8.0, 8.0), method="fd4")
        diagnostic = HawkingMassDiagnostic(grid, 12, 24)
        flat = ADMState(flat_metric(grid), grid.zeros((3, 3)))
        flat_mass = diagnostic.evaluate(flat, 2.0)
        self.assertAlmostEqual(flat_mass.areal_radius, 2.0, delta=2.0e-3)
        self.assertAlmostEqual(flat_mass.mass, 0.0, delta=2.0e-3)

        x, y, z = grid.coordinates()
        coordinate_radius = np.sqrt(x * x + y * y + z * z)
        expected_mass = 0.5
        psi = 1.0 + expected_mass / (2.0 * coordinate_radius)
        h = grid.zeros((3, 3))
        for i in range(3):
            h[i, i] = psi**4
        schwarzschild = diagnostic.evaluate(
            ADMState(h, grid.zeros((3, 3))), 2.0
        )
        self.assertAlmostEqual(schwarzschild.mass, expected_mass, delta=0.03)

    def test_real_spherical_harmonics_are_orthonormal(self):
        sphere = sphere_quadrature(1.0, n_theta=14, n_phi=28)
        y20 = real_spherical_harmonic((2, 0), sphere.theta, sphere.phi)
        y22 = real_spherical_harmonic((2, 2), sphere.theta, sphere.phi)
        self.assertAlmostEqual(float(np.sum(sphere.weights * y20 * y20)), 1.0, places=12)
        self.assertAlmostEqual(float(np.sum(sphere.weights * y22 * y22)), 1.0, places=12)
        self.assertAlmostEqual(float(np.sum(sphere.weights * y20 * y22)), 0.0, places=12)

    def test_spectral_kkt_finder_rejects_minkowski_at_active_radius_bound(self):
        grid = CartesianGrid((16, 16, 16), (8.0, 8.0, 8.0), method="fd4")
        state = ADMState(flat_metric(grid), grid.zeros((3, 3)))
        parameters = SpectralMOTSParameters(
            maximum_degree=0,
            n_theta=8,
            n_phi=16,
            minimum_radius=0.5,
            maximum_radius=2.5,
            expansion_tolerance=1.0e-3,
            maximum_iterations=12,
        )
        result = SpectralMOTSFinder(grid, parameters).find(state, 1.0)
        self.assertFalse(result.found)
        self.assertEqual(result.reason, "KKT stationary point without a MOTS")
        self.assertEqual(result.active_upper, (0,))
        self.assertGreater(result.upper_multipliers[0], 0.0)
        self.assertAlmostEqual(result.coordinate_radius_mean, 2.5, places=12)
        self.assertGreater(result.spectral_residual_norm, 0.5)

    def test_parametric_cartesian_observer_recovers_flat_surface_geometry(self):
        grid = CartesianGrid((16, 16, 16), (6.0, 6.0, 6.0), method="fd4")
        state = ADMState(flat_metric(grid), grid.zeros((3, 3)))
        parameters = SpectralMOTSParameters(
            modes=((0, 0), (2, 0), (4, 0)),
            n_theta=16,
            n_phi=12,
            minimum_radius=0.25,
            maximum_radius=2.0,
        )
        finder = ParametricAxisymmetricMOTSFinder(grid, parameters)
        coefficients = finder.initial_coefficients(1.0)
        evaluation = finder.evaluator(state).evaluate(coefficients)
        self.assertTrue(evaluation.valid)
        np.testing.assert_allclose(evaluation.expansion, 2.0, atol=2.0e-13)
        self.assertAlmostEqual(float(np.sum(evaluation.area_weights)), 4.0 * np.pi, places=12)
        np.testing.assert_allclose(evaluation.normal_covector_norm, 1.0, atol=2.0e-13)

    def test_ccz4_physical_normal_tracker_accepts_restartable_snapshots(self):
        grid = CartesianGrid((16, 16, 16), (6.0, 6.0, 6.0), method="fd4")
        ccz4 = CCZ4Solver(grid)
        parameters = SpectralMOTSParameters(
            modes=((0, 0),),
            n_theta=8,
            n_phi=8,
            minimum_radius=0.5,
            maximum_radius=1.5,
            maximum_iterations=2,
        )
        finder = ParametricAxisymmetricMOTSFinder(grid, parameters)
        tracker = CCZ4PhysicalNormalMOTSTracker(
            ccz4,
            finder,
            [MOTSBranchDefinition("common", (0.0, 0.0, 0.0), 1.0)],
        )
        observations = tracker.observe(ccz4.flat_state())
        self.assertIn("common", observations)
        self.assertFalse(observations["common"].branch.result.found)
        self.assertIsNone(observations["common"].normal_stability)
        self.assertFalse(
            observations["common"].qualification.resolution_gate_passed
        )
        self.assertFalse(
            observations["common"].qualification.continuum_zero_qualified
        )

        checkpoint = tracker.checkpoint()
        tracker.update_center("common", (0.0, 0.0, 0.1))
        self.assertEqual(
            tracker.branches.checkpoint()["branches"]["common"]["center"],
            [0.0, 0.0, 0.1],
        )
        restarted = CCZ4PhysicalNormalMOTSTracker(
            ccz4,
            finder,
            [MOTSBranchDefinition("common", (0.0, 0.0, 0.0), 1.0)],
        )
        restarted.restore(checkpoint)
        self.assertEqual(restarted.checkpoint(), checkpoint)

    def test_ccz4_tracker_applies_calibrated_resolution_and_mode_floor(self):
        grid = CartesianGrid((32, 32, 32), (1.0, 1.0, 1.0), method="fd4")
        parameters = SpectralMOTSParameters(
            modes=((0, 0),),
            n_theta=12,
            n_phi=8,
            minimum_radius=0.1,
            maximum_radius=1.0,
            maximum_mode_amplitude=2.0,
            expansion_tolerance=1.0e-8,
            jacobian_step=1.0e-5,
            maximum_iterations=30,
        )
        direct = DirectBrillLindquistMOTSFinder(parameters)
        surface = direct.find(0.0, 0.5)
        stability = direct.normal_stability(0.0, surface)
        branch = MOTSBranchObservation(
            "common_outer", 0.0, surface, True, False, 1, 0, 0.0
        )
        tracker = CCZ4PhysicalNormalMOTSTracker(
            CCZ4Solver(grid),
            ParametricAxisymmetricMOTSFinder(grid, parameters),
            [MOTSBranchDefinition("common_outer", (0.0, 0.0, 0.0), 0.5)],
        )
        qualification = tracker.qualify(branch, stability)
        self.assertAlmostEqual(qualification.cells_per_mean_radius, 16.0)
        self.assertTrue(qualification.resolution_gate_passed)
        self.assertTrue(qualification.principal_mode_sign_gate_passed)
        self.assertFalse(qualification.continuum_zero_qualified)

        strict = CCZ4PhysicalNormalMOTSTracker(
            CCZ4Solver(grid),
            ParametricAxisymmetricMOTSFinder(grid, parameters),
            [MOTSBranchDefinition("common_outer", (0.0, 0.0, 0.0), 0.5)],
            minimum_resolved_mode_magnitude=0.3,
        )
        self.assertFalse(
            strict.qualify(branch, stability).principal_mode_sign_gate_passed
        )

    def test_spectral_kkt_finder_removes_distortion_on_schwarzschild(self):
        grid = CartesianGrid((32, 32, 32), (6.0, 6.0, 6.0), method="fd4")
        state = self.isotropic_schwarzschild(grid)
        parameters = SpectralMOTSParameters(
            modes=((0, 0), (2, 0)),
            n_theta=10,
            n_phi=20,
            minimum_radius=0.25,
            maximum_radius=1.0,
            maximum_mode_amplitude=0.4,
            expansion_tolerance=3.0e-3,
            jacobian_step=5.0e-4,
            maximum_iterations=15,
        )
        finder = SpectralMOTSFinder(grid, parameters)
        seed = finder.initial_coefficients(0.65)
        seed[1] = 0.12
        result = finder.find(state, 0.65, initial_coefficients=seed)
        self.assertTrue(result.found)
        self.assertAlmostEqual(result.coordinate_radius_mean, 0.5, delta=0.08)
        self.assertAlmostEqual(result.irreducible_mass, 1.0, delta=0.15)
        self.assertLess(abs(result.coefficients[1]), 0.02)
        self.assertLess(result.spectral_residual_norm, parameters.expansion_tolerance)
        self.assertTrue(np.all(np.isfinite(result.response_eigenvalues)))

        stability = NormalMOTSStabilityOperator(finder).analyze(state, result)
        self.assertTrue(stability.valid)
        self.assertGreater(stability.principal_eigenvalue.real, 0.0)
        self.assertAlmostEqual(stability.principal_eigenvalue.imag, 0.0, places=12)
        self.assertLess(stability.normalizer_roundtrip_error, 1.0e-12)
        self.assertLess(stability.finite_difference_relative_error, 1.0e-5)
        self.assertLess(stability.normalizer_condition_number, 2.0)

        tracker = MOTSBranchTracker(
            finder,
            [MOTSBranchDefinition("outer", (0.0, 0.0, 0.0), 0.65, seed)],
        )
        first = tracker.observe(self.isotropic_schwarzschild(grid, time=0.25))["outer"]
        second = tracker.observe(self.isotropic_schwarzschild(grid, time=0.5))["outer"]
        self.assertTrue(first.newly_found)
        self.assertFalse(second.newly_found)
        self.assertEqual(second.successful_observations, 2)
        self.assertEqual(second.first_found_time, 0.25)

        restarted = MOTSBranchTracker(
            finder,
            [MOTSBranchDefinition("outer", (0.0, 0.0, 0.0), 0.65, seed)],
        )
        restarted.restore(tracker.checkpoint())
        third = restarted.observe(self.isotropic_schwarzschild(grid, time=0.75))["outer"]
        self.assertFalse(third.newly_found)
        self.assertEqual(third.successful_observations, 3)
        self.assertEqual(third.first_found_time, 0.25)

    def test_direct_conformal_surface_recovers_exact_schwarzschild_spectrum(self):
        parameters = SpectralMOTSParameters(
            modes=((0, 0), (2, 0), (4, 0)),
            n_theta=20,
            n_phi=8,
            minimum_radius=0.1,
            maximum_radius=2.0,
            maximum_mode_amplitude=2.0,
            expansion_tolerance=1.0e-7,
            jacobian_step=1.0e-5,
            maximum_iterations=30,
        )
        finder = DirectBrillLindquistMOTSFinder(parameters, total_bare_mass=1.0)
        seed = finder.initial_coefficients(0.6)
        seed[1] = 0.05
        result = finder.find(0.0, 0.6, seed)
        self.assertTrue(result.found)
        self.assertAlmostEqual(result.coordinate_radius_mean, 0.5, places=6)
        self.assertAlmostEqual(result.area, 16.0 * np.pi, places=8)
        self.assertAlmostEqual(result.irreducible_mass, 1.0, places=10)
        self.assertLess(result.rms_expansion, 1.0e-7)

        stability = finder.normal_stability(0.0, result)
        self.assertTrue(stability.valid)
        np.testing.assert_allclose(
            np.sort(stability.eigenvalues.real),
            [0.25, 1.75, 5.25],
            rtol=2.0e-7,
            atol=2.0e-8,
        )
        self.assertLess(stability.normalizer_roundtrip_error, 1.0e-12)

    def test_parametric_observer_supports_nonaxisymmetric_real_modes(self):
        grid = CartesianGrid((24, 24, 24), (8.0, 8.0, 8.0), method="fd4")
        state = ADMState(flat_metric(grid), grid.zeros((3, 3)))
        parameters = SpectralMOTSParameters(
            maximum_degree=2,
            modes=((0, 0), (2, 2), (2, -2)),
            n_theta=12,
            n_phi=32,
            minimum_radius=0.5,
            maximum_radius=2.0,
        )
        evaluator = ParametricMOTSFinder(grid, parameters).evaluator(state)
        radius = 1.2
        sphere = evaluator.evaluate(np.array([np.log(radius), 0.0, 0.0]))
        self.assertTrue(sphere.valid, sphere.reason)
        np.testing.assert_allclose(sphere.expansion, 2.0 / radius, atol=2.0e-7)
        self.assertAlmostEqual(
            float(np.sum(sphere.area_weights)),
            4.0 * np.pi * radius * radius,
            delta=2.0e-6,
        )

        cosine = evaluator.evaluate(np.array([np.log(radius), 0.08, 0.0]))
        sine = evaluator.evaluate(np.array([np.log(radius), 0.0, 0.08]))
        self.assertTrue(cosine.valid, cosine.reason)
        self.assertTrue(sine.valid, sine.reason)
        self.assertAlmostEqual(
            float(np.sum(cosine.area_weights)),
            float(np.sum(sine.area_weights)),
            delta=2.0e-7,
        )
        cosine_rms = np.sqrt(
            np.sum(cosine.area_weights * cosine.expansion**2)
            / np.sum(cosine.area_weights)
        )
        sine_rms = np.sqrt(
            np.sum(sine.area_weights * sine.expansion**2)
            / np.sum(sine.area_weights)
        )
        self.assertAlmostEqual(float(cosine_rms), float(sine_rms), delta=2.0e-7)

if __name__ == "__main__":
    unittest.main()
