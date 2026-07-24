from __future__ import annotations

import math
import unittest

import torch

from run_theory33_hybrid import validate_broad_basin
from theory33_hybrid import (
    Theory33HybridTesseract,
    Theory33MasterFunction,
    Theory33Parameters,
)


class Theory33HybridTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = Theory33HybridTesseract(
            learn_dynamics=False,
            boundary_certificates=False,
        )
        self.initial = torch.tensor([-0.8, -0.2, 0.2, 0.8])

    def state(self):
        z = self.initial.clone().requires_grad_(True)
        sigma = torch.where(
            z.detach() >= 0.0, torch.ones_like(z), -torch.ones_like(z)
        )
        return z, self.model.evaluate(z, sigma)

    def test_kkt_map_constructs_future_timelike_currents(self) -> None:
        _, state = self.state()
        torch.testing.assert_close(
            state.x_plus,
            state.number_density.square(),
            rtol=2.0e-13,
            atol=2.0e-15,
        )
        torch.testing.assert_close(
            state.x_minus,
            state.carrier_density.square(),
            rtol=2.0e-13,
            atol=2.0e-15,
        )
        expected_cross = (
            state.number_density
            * state.carrier_density
            * torch.cosh(
                state.baryon_rapidity - state.carrier_rapidity
            )
        )
        torch.testing.assert_close(
            state.x_cross,
            expected_cross,
            rtol=2.0e-13,
            atol=2.0e-15,
        )
        self.assertTrue(bool(state.physical_valid.all()))
        self.assertTrue(bool((state.relative_lorentz_factor >= 1.0).all()))

    def test_analytic_m1_entrainment_derivative(self) -> None:
        parameters = Theory33Parameters(
            entrainment_linear=0.03,
            entrainment_quadratic=0.02,
        )
        model = Theory33HybridTesseract(
            parameters=parameters,
            learn_dynamics=False,
            boundary_certificates=False,
        )
        z = self.initial.clone().requires_grad_(True)
        sigma = torch.where(
            z.detach() >= 0.0, torch.ones_like(z), -torch.ones_like(z)
        )
        state = model.evaluate(z, sigma)
        expected = (
            parameters.entrainment_linear
            + parameters.entrainment_quadratic
            * state.relative_invariant
            / parameters.entrainment_scale_fourth
        )
        torch.testing.assert_close(
            state.entrainment, expected, rtol=2.0e-12, atol=2.0e-14
        )

    def test_hilbert_stress_is_symmetric(self) -> None:
        _, state = self.state()
        contravariant = torch.einsum(
            "bij,bjk->bik", state.stress, torch.linalg.inv(state.metric)
        )
        torch.testing.assert_close(
            contravariant,
            contravariant.transpose(-1, -2),
            rtol=2.0e-12,
            atol=2.0e-14,
        )

    def test_full_characteristic_symbol_is_causal(self) -> None:
        _, state = self.state()
        for audit in self.model.characteristic_audits(state):
            self.assertTrue(audit.strongly_hyperbolic)
            self.assertTrue(audit.causal)
            self.assertEqual(audit.speeds.numel(), 4)
            self.assertLessEqual(audit.maximum_absolute_speed, 1.0 + 1.0e-8)

    def test_characteristic_gate_rejects_superluminal_domain(self) -> None:
        parameters = Theory33Parameters(target_fraction=4.0)
        master = Theory33MasterFunction(parameters)
        density = 0.05
        internal = 0.25
        pressure = (
            (parameters.gamma_ad - 1.0) * density * internal
        )
        entropy = (
            math.log(pressure / density**parameters.gamma_ad)
            / (parameters.gamma_ad - 1.0)
        )
        audit = master.characteristic_audit(
            density,
            entropy,
            math.atanh(0.18),
            parameters.target_fraction * density,
            math.atanh(0.46),
        )
        self.assertTrue(audit.strongly_hyperbolic)
        self.assertFalse(audit.causal)
        self.assertGreater(audit.maximum_absolute_speed, 1.0)

    def test_step_is_differentiable_through_physical_closure(self) -> None:
        z = self.initial.clone().requires_grad_(True)
        next_z, diagnostics = self.model.step(
            z, measure_local_gain=True, audit_characteristics=True
        )
        loss = next_z.square().sum()
        loss.backward()
        self.assertIsNotNone(z.grad)
        assert z.grad is not None
        self.assertTrue(bool(torch.isfinite(z.grad).all()))
        self.assertTrue(bool(diagnostics.kkt_valid.all()))
        self.assertTrue(bool(diagnostics.physical_valid.all()))
        self.assertTrue(bool(diagnostics.characteristic_valid.all()))
        self.assertLess(
            float(diagnostics.current_normalization_error.max()), 1.0e-14
        )

    def test_selected_region_gradient_matches_finite_difference(self) -> None:
        z = torch.tensor([0.8], requires_grad=True)
        next_z, diagnostics = self.model.step(z)
        derivative = torch.autograd.grad(next_z.sum(), z)[0]
        epsilon = 1.0e-4
        samples = []
        for direction in (-1.0, 1.0):
            probe = torch.tensor(
                [0.8 + direction * epsilon], requires_grad=True
            )
            value, probe_diagnostics = self.model.step(probe)
            self.assertEqual(
                int(probe_diagnostics.active_mask[0]),
                int(diagnostics.active_mask[0]),
            )
            self.assertEqual(
                int(probe_diagnostics.classifier_mask[0]),
                int(diagnostics.classifier_mask[0]),
            )
            samples.append(value.detach())
        finite_difference = (samples[1] - samples[0]) / (2.0 * epsilon)
        torch.testing.assert_close(
            derivative,
            finite_difference,
            rtol=3.0e-4,
            atol=2.0e-3,
        )

    def test_literal_128_step_rollout_reaches_intrinsic_attractor(self) -> None:
        target = 0.1019493853295916
        z = self.initial.clone()
        previous = (z - target).square()
        maximum_growth = -math.inf
        for _ in range(128):
            probe = z.detach().clone().requires_grad_(True)
            z, diagnostics = self.model.step(probe)
            z = z.detach()
            current = (z - target).square()
            maximum_growth = max(
                maximum_growth, float((current - previous).max())
            )
            previous = current
            self.assertTrue(bool(diagnostics.kkt_valid.all()))
            self.assertTrue(bool(diagnostics.physical_valid.all()))
        torch.testing.assert_close(
            z,
            torch.full_like(z, target),
            rtol=0.0,
            atol=2.0e-15,
        )
        self.assertLessEqual(maximum_growth, 0.0)

    def test_dense_broad_basin_is_bounded_and_physical(self) -> None:
        metrics = validate_broad_basin(
            self.model,
            steps=128,
            target=0.1019493853295916,
            points=257,
            characteristic_points=17,
        )
        self.assertTrue(metrics["kkt_valid"])
        self.assertTrue(metrics["physical_valid"])
        self.assertTrue(metrics["characteristic_valid"])
        self.assertLess(float(metrics["max_abs_state"]), 1.051)
        self.assertGreater(float(metrics["min_kkt_margin"]), 0.0)
        self.assertGreater(float(metrics["min_physical_margin"]), 0.0)
        self.assertLessEqual(
            float(metrics["max_characteristic_speed"]), 1.0
        )
        self.assertEqual(float(metrics["converged_fraction"]), 1.0)
        self.assertLess(float(metrics["terminal_max_error"]), 1.0e-14)


if __name__ == "__main__":
    unittest.main()
