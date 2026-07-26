"""Pure-NumPy bridge from the qualified frozen artifact to NR local states."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .theory33 import Theory33MasterFunction, Theory33MasterParameters


@dataclass(frozen=True)
class FrozenInvariantResponse:
    """Frozen master and mobility outputs for broadcast invariant arrays."""

    lambda_value: np.ndarray
    lambda_gradient: np.ndarray
    phase_two: np.ndarray
    phase_threshold: np.ndarray
    mobility_matrix: np.ndarray


def _softplus(value: np.ndarray | float) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    return np.maximum(array, 0.0) + np.log1p(np.exp(-np.abs(array)))


def _sigmoid(value: np.ndarray | float) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    positive = array >= 0.0
    result = np.empty_like(array)
    result[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exponential = np.exp(array[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


class QualifiedFrozenClosure:
    """Evaluate the qualified master and PSD mobility without fitting or Torch.

    The implementation directly consumes the serialized weights. This avoids
    loading a second OpenMP runtime into the NumPy NR process and supplies
    analytic branchwise derivatives of the frozen two-phase master.
    """

    def __init__(
        self,
        artifact_path: str | Path = "theory33_frozen_variants.json",
    ) -> None:
        self.artifact_path = Path(artifact_path).resolve()
        self.artifact = json.loads(
            self.artifact_path.read_text(encoding="utf-8")
        )
        if self.artifact.get("format") != (
            "theory33-frozen-constitutive-variants"
        ):
            raise ValueError("unexpected frozen Theory 3.3 artifact format")
        variants = self.artifact["variants"]
        canonical = json.dumps(
            variants,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()
        if digest != self.artifact["variant_digest"]:
            raise ValueError("frozen Theory 3.3 variant digest mismatch")
        self.variant_digest = digest
        self.parameters = Theory33MasterParameters()
        self.analytic = Theory33MasterFunction(self.parameters)
        self.data_config = dict(self.artifact["data_config"])
        self.phase = dict(variants["phase"])
        self.drift = dict(variants["drift_master"])
        self.mobility = {
            name: np.asarray(value, dtype=float)
            for name, value in variants["mobility"]["state"].items()
        }

    def _normalized_context(
        self,
        density: np.ndarray,
        temperature: np.ndarray,
        fraction: np.ndarray,
    ) -> np.ndarray:
        config = self.data_config
        density_coordinate = (
            2.0
            * (density - config["density_minimum"])
            / (config["density_maximum"] - config["density_minimum"])
            - 1.0
        )
        log_temperature = np.log(np.maximum(temperature, 1.0e-12))
        log_minimum = math.log(config["temperature_minimum"])
        log_maximum = math.log(config["temperature_maximum"])
        temperature_coordinate = (
            2.0
            * (log_temperature - log_minimum)
            / (log_maximum - log_minimum)
            - 1.0
        )
        fraction_coordinate = (
            2.0
            * (fraction - config["fraction_minimum"])
            / (config["fraction_maximum"] - config["fraction_minimum"])
            - 1.0
        )
        return np.clip(
            np.stack(
                [
                    density_coordinate,
                    temperature_coordinate,
                    fraction_coordinate,
                ],
                axis=-1,
            ),
            -1.5,
            1.5,
        )

    def _mobility_matrix(self, context: np.ndarray) -> np.ndarray:
        state = self.mobility
        hidden = np.tanh(
            context @ state["input_layer.weight"].T
            + state["input_layer.bias"]
        )
        raw = (
            hidden @ state["output_layer.weight"].T
            + state["output_layer.bias"]
        )
        diagonal0 = _softplus(raw[..., 0]) + 1.0e-6
        diagonal1 = _softplus(raw[..., 2]) + 1.0e-6
        factor = np.zeros(raw.shape[:-1] + (2, 2), dtype=float)
        factor[..., 0, 0] = diagonal0
        factor[..., 1, 0] = raw[..., 1]
        factor[..., 1, 1] = diagonal1
        return factor @ np.swapaxes(factor, -1, -2)

    def _drift_value_and_derivative(
        self,
        coordinate: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        state = self.drift["state"]
        knots = np.asarray(state["knots"], dtype=float)
        sharpness = float(state["sharpness"])
        raw_weights = np.asarray(state["raw_weights"], dtype=float)
        quadratic = float(_softplus(state["raw_quadratic"]))
        weights = _softplus(raw_weights)
        shifted = sharpness * (
            coordinate[..., None] - knots
        )
        shifted_origin = -sharpness * knots
        origin_value = _softplus(shifted_origin)
        origin_derivative = sharpness * _sigmoid(shifted_origin)
        bregman = (
            _softplus(shifted)
            - origin_value
            - origin_derivative * coordinate[..., None]
        )
        value = (
            0.5 * quadratic * coordinate**2
            + np.einsum("...i,i->...", bregman, weights)
        )
        derivative = (
            quadratic * coordinate
            + np.einsum(
                "...i,i->...",
                sharpness * _sigmoid(shifted) - origin_derivative,
                weights,
            )
        )
        return value, derivative

    def evaluate(
        self,
        n_squared: np.ndarray,
        d_squared: np.ndarray,
        x_squared: np.ndarray,
        entropy: np.ndarray,
    ) -> FrozenInvariantResponse:
        n2, d2, cross, sigma = np.broadcast_arrays(
            np.asarray(n_squared, dtype=float),
            np.asarray(d_squared, dtype=float),
            np.asarray(x_squared, dtype=float),
            np.asarray(entropy, dtype=float),
        )
        n = np.sqrt(np.maximum(n2, 1.0e-300))
        d = np.sqrt(np.maximum(d2, 1.0e-300))
        fraction = d / np.maximum(n, 1.0e-300)
        p = self.parameters
        pressure = np.exp((p.gamma_ad - 1.0) * sigma) * n**p.gamma_ad
        temperature = pressure / np.maximum(n, 1.0e-300)
        base = self.analytic.lambda_from_invariants(n2, d2, cross, sigma)

        relative = cross / np.maximum(n * d, 1.0e-300) - 1.0
        relative = np.maximum(relative, 0.0)
        drift_scale = float(self.drift["drift_scale"])
        coordinate = relative / drift_scale
        context = self._normalized_context(n, temperature, fraction)
        phase_threshold = (
            self.phase["base"]
            + self.phase["temperature_coefficient"] * context[..., 1]
            + self.phase["density_coefficient"] * context[..., 0]
            + self.phase["fraction_coefficient"] * context[..., 2]
        )
        threshold_coordinate = phase_threshold / drift_scale
        phase_two = relative >= phase_threshold
        drift_value, drift_derivative = (
            self._drift_value_and_derivative(coordinate)
        )
        threshold_value, threshold_derivative = (
            self._drift_value_and_derivative(threshold_coordinate)
        )
        energy_scale = float(self.drift["energy_scale"])
        phase_energy = energy_scale * (drift_value - threshold_value)
        lambda_value = base - np.where(phase_two, phase_energy, 0.0)

        anchor = d - p.target_fraction * n
        rho_n = (
            1.0
            + p.gamma_ad
            * pressure
            / np.maximum((p.gamma_ad - 1.0) * n, 1.0e-300)
            - p.target_fraction * anchor / p.chemical_susceptibility
        )
        rho_d = (
            p.carrier_K
            * (1.0 + p.carrier_sound_speed**2)
            * d ** (p.carrier_sound_speed**2)
            + anchor / p.chemical_susceptibility
        )
        invariant_relative = cross - n * d
        entrainment = p.entrainment_linear + (
            p.entrainment_quadratic
            * invariant_relative
            / p.entrainment_scale_fourth
        )
        base_gradient = np.stack(
            [
                (-rho_n + entrainment * d)
                / np.maximum(2.0 * n, 1.0e-300),
                (-rho_d + entrainment * n)
                / np.maximum(2.0 * d, 1.0e-300),
                -entrainment,
                -pressure,
            ],
            axis=-1,
        )

        log_temperature_span = math.log(
            self.data_config["temperature_maximum"]
        ) - math.log(self.data_config["temperature_minimum"])
        fraction_span = (
            self.data_config["fraction_maximum"]
            - self.data_config["fraction_minimum"]
        )
        threshold_n = (
            self.phase["density_coefficient"]
            * 2.0
            / (
                self.data_config["density_maximum"]
                - self.data_config["density_minimum"]
            )
            + self.phase["temperature_coefficient"]
            * 2.0
            * (p.gamma_ad - 1.0)
            / (np.maximum(n, 1.0e-300) * log_temperature_span)
            - self.phase["fraction_coefficient"]
            * 2.0
            * fraction
            / (np.maximum(n, 1.0e-300) * fraction_span)
        )
        threshold_d = (
            self.phase["fraction_coefficient"]
            * 2.0
            / (np.maximum(n, 1.0e-300) * fraction_span)
        )
        threshold_sigma = (
            self.phase["temperature_coefficient"]
            * 2.0
            * (p.gamma_ad - 1.0)
            / log_temperature_span
        )
        gamma_relative = relative + 1.0
        drift_gradient = np.stack(
            [
                -gamma_relative
                / np.maximum(2.0 * n**2 * drift_scale, 1.0e-300),
                -gamma_relative
                / np.maximum(2.0 * d**2 * drift_scale, 1.0e-300),
                1.0 / np.maximum(n * d * drift_scale, 1.0e-300),
                np.zeros_like(n),
            ],
            axis=-1,
        )
        threshold_gradient = np.stack(
            [
                threshold_n
                / np.maximum(2.0 * n * drift_scale, 1.0e-300),
                threshold_d
                / np.maximum(2.0 * d * drift_scale, 1.0e-300),
                np.zeros_like(n),
                np.broadcast_to(
                    threshold_sigma / drift_scale,
                    n.shape,
                ),
            ],
            axis=-1,
        )
        residual_gradient = energy_scale * (
            drift_derivative[..., None] * drift_gradient
            - threshold_derivative[..., None] * threshold_gradient
        )
        lambda_gradient = base_gradient - np.where(
            phase_two[..., None],
            residual_gradient,
            0.0,
        )
        return FrozenInvariantResponse(
            lambda_value=lambda_value,
            lambda_gradient=lambda_gradient,
            phase_two=phase_two,
            phase_threshold=phase_threshold,
            mobility_matrix=self._mobility_matrix(context),
        )
