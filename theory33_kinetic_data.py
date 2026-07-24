"""Independent relativistic kinetic-theory benchmark data for Theory 3.3.

This module intentionally imports neither the Tesseract simulator nor any
neural constitutive model. Targets are produced by direct Gauss-Legendre
quadrature of a dimensionless Maxwell-Juttner distribution and by a separate
seeded stochastic trajectory generator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class KineticDataConfig:
    seed: int = 3304
    quadrature_points: int = 128
    density_points: int = 7
    temperature_points: int = 9
    fraction_points: int = 4
    drift_points: int = 11
    trajectory_count: int = 72
    trajectory_steps: int = 48
    density_minimum: float = 0.06
    density_maximum: float = 0.18
    temperature_minimum: float = 0.08
    temperature_maximum: float = 0.80
    fraction_minimum: float = 0.025
    fraction_maximum: float = 0.090
    drift_maximum: float = 0.016
    phase_threshold: float = 0.006
    phase_width: float = 2.5e-4
    relaxation_time: float = 0.35


TRANSPORT_FIELDS = (
    "density",
    "temperature",
    "carrier_fraction",
    "relative_drift",
    "pressure",
    "energy_density",
    "enthalpy_per_particle",
    "mobility_00",
    "mobility_01",
    "mobility_11",
    "drift_energy",
)

TRAJECTORY_FIELDS = (
    "trajectory_id",
    "step",
    "density",
    "temperature",
    "carrier_fraction",
    "relative_drift",
    "latent_threshold",
    "phase_label",
    "transport_response",
    "response_noise_std",
)


def _quadrature(config: KineticDataConfig) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(
        config.quadrature_points
    )
    maximum_momentum = 24.0
    momentum = 0.5 * maximum_momentum * (nodes + 1.0)
    scaled_weights = 0.5 * maximum_momentum * weights
    return momentum, scaled_weights


def kinetic_moments(
    temperature: float,
    *,
    config: KineticDataConfig,
) -> dict[str, Any]:
    """Integrate equilibrium EOS and a two-force transport Gram matrix."""
    momentum, quadrature_weights = _quadrature(config)
    energy = np.sqrt(1.0 + momentum * momentum)
    # Removing the rest-mass exponential prevents harmless underflow at the
    # cold edge; it cancels in every normalized moment.
    weight = (
        quadrature_weights
        * momentum**2
        * np.exp(-(energy - 1.0) / temperature)
    )
    normalization = float(weight.sum())
    probability = weight / normalization
    mean_energy = float(np.sum(probability * energy))
    pressure_per_particle = float(
        np.sum(probability * momentum**2 / (3.0 * energy))
    )
    velocity_squared = momentum**2 / energy**2
    centered_energy = (energy - mean_energy) / temperature
    basis = np.stack(
        [np.ones_like(momentum), centered_energy],
        axis=-1,
    )
    gram = np.einsum(
        "p,pi,pj->ij",
        probability * velocity_squared / 3.0,
        basis,
        basis,
    )
    gram += np.eye(2) * 1.0e-8
    return {
        "mean_energy": mean_energy,
        "pressure_per_particle": pressure_per_particle,
        "enthalpy_per_particle": mean_energy + pressure_per_particle,
        "transport_gram": gram,
    }


def generate_transport_records(
    config: KineticDataConfig = KineticDataConfig(),
) -> list[dict[str, float]]:
    """Generate an EOS/transport table without using the learned model."""
    densities = np.linspace(
        config.density_minimum,
        config.density_maximum,
        config.density_points,
    )
    temperatures = np.geomspace(
        config.temperature_minimum,
        config.temperature_maximum,
        config.temperature_points,
    )
    fractions = np.linspace(
        config.fraction_minimum,
        config.fraction_maximum,
        config.fraction_points,
    )
    drifts = np.linspace(0.0, config.drift_maximum, config.drift_points)
    records: list[dict[str, float]] = []
    for temperature in temperatures:
        moments = kinetic_moments(float(temperature), config=config)
        for density in densities:
            scale = (
                config.relaxation_time
                * density
                / max(float(temperature), 1.0e-12)
            )
            mobility = scale * moments["transport_gram"]
            for fraction in fractions:
                mixture_scale = 1.0 + 0.5 * float(fraction)
                mixed_mobility = mixture_scale * mobility
                curvature = float(mixed_mobility[1, 1])
                for drift in drifts:
                    normalized_drift = drift / config.drift_maximum
                    drift_energy = (
                        0.5 * curvature * normalized_drift**2
                        + 0.08 * curvature * normalized_drift**4
                    )
                    records.append(
                        {
                            "density": float(density),
                            "temperature": float(temperature),
                            "carrier_fraction": float(fraction),
                            "relative_drift": float(drift),
                            "pressure": (
                                float(density)
                                * moments["pressure_per_particle"]
                            ),
                            "energy_density": (
                                float(density) * moments["mean_energy"]
                            ),
                            "enthalpy_per_particle": moments[
                                "enthalpy_per_particle"
                            ],
                            "mobility_00": float(mixed_mobility[0, 0]),
                            "mobility_01": float(mixed_mobility[0, 1]),
                            "mobility_11": float(mixed_mobility[1, 1]),
                            "drift_energy": float(drift_energy),
                        }
                    )
    return records


def _interpolate_mobility_11(
    temperature: float,
    density: float,
    fraction: float,
    *,
    config: KineticDataConfig,
) -> float:
    moments = kinetic_moments(temperature, config=config)
    scale = (
        config.relaxation_time
        * density
        / max(temperature, 1.0e-12)
    )
    return float(
        scale
        * (1.0 + 0.5 * fraction)
        * moments["transport_gram"][1, 1]
    )


def generate_phase_trajectories(
    config: KineticDataConfig = KineticDataConfig(),
) -> list[dict[str, float | int]]:
    """Generate grouped noisy phase trajectories from the kinetic table."""
    rng = np.random.default_rng(config.seed)
    records: list[dict[str, float | int]] = []
    for trajectory_id in range(config.trajectory_count):
        density_center = rng.uniform(
            config.density_minimum, config.density_maximum
        )
        temperature_center = rng.uniform(
            config.temperature_minimum, config.temperature_maximum
        )
        fraction = rng.uniform(
            config.fraction_minimum, config.fraction_maximum
        )
        drift = rng.uniform(0.002, 0.012)
        phase_offset = rng.normal(0.0, 1.5e-4)
        phase_state = 0
        for step in range(config.trajectory_steps):
            angle = (
                2.0
                * math.pi
                * (step / config.trajectory_steps)
                + 0.19 * trajectory_id
            )
            temperature = float(
                np.clip(
                    temperature_center
                    * (1.0 + 0.18 * math.sin(angle))
                    + rng.normal(0.0, 0.006),
                    config.temperature_minimum,
                    config.temperature_maximum,
                )
            )
            density = float(
                np.clip(
                    density_center
                    * (1.0 + 0.12 * math.cos(0.7 * angle))
                    + rng.normal(0.0, 0.0015),
                    config.density_minimum,
                    config.density_maximum,
                )
            )
            forcing = (
                0.0015 * math.sin(1.3 * angle)
                + 0.0008 * math.cos(0.4 * angle)
            )
            drift = float(
                np.clip(
                    0.82 * drift
                    + 0.18 * config.phase_threshold
                    + forcing
                    + rng.normal(0.0, 3.0e-4),
                    0.0,
                    config.drift_maximum,
                )
            )
            latent_threshold = (
                config.phase_threshold
                + phase_offset
                + 3.5e-4
                * (
                    temperature
                    - 0.5
                    * (
                        config.temperature_minimum
                        + config.temperature_maximum
                    )
                )
                / (
                    config.temperature_maximum
                    - config.temperature_minimum
                )
                - 2.0e-4
                * (
                    density
                    - 0.5
                    * (
                        config.density_minimum
                        + config.density_maximum
                    )
                )
                / (
                    config.density_maximum
                    - config.density_minimum
                )
            )
            transition_probability = 1.0 / (
                1.0
                + math.exp(
                    -(drift - latent_threshold) / config.phase_width
                )
            )
            phase_state = int(rng.random() < transition_probability)
            mobility = _interpolate_mobility_11(
                temperature,
                density,
                fraction,
                config=config,
            )
            phase_multiplier = 1.0 + 0.40 * phase_state
            noiseless_response = (
                phase_multiplier
                * mobility
                * drift
                / config.drift_maximum
            )
            noise_std = 0.025 * max(abs(noiseless_response), 1.0e-5)
            response = noiseless_response + rng.normal(0.0, noise_std)
            records.append(
                {
                    "trajectory_id": trajectory_id,
                    "step": step,
                    "density": density,
                    "temperature": temperature,
                    "carrier_fraction": float(fraction),
                    "relative_drift": drift,
                    "latent_threshold": float(latent_threshold),
                    "phase_label": phase_state,
                    "transport_response": float(response),
                    "response_noise_std": float(noise_std),
                }
            )
    return records


def _write_csv(
    path: Path,
    records: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_kinetic_dataset(
    output_directory: Path,
    config: KineticDataConfig = KineticDataConfig(),
) -> dict[str, Any]:
    transport_path = output_directory / "theory33_kinetic_transport.csv"
    phase_path = output_directory / "theory33_noisy_phase_trajectories.csv"
    metadata_path = output_directory / "theory33_kinetic_metadata.json"
    transport = generate_transport_records(config)
    phase = generate_phase_trajectories(config)
    _write_csv(transport_path, transport, TRANSPORT_FIELDS)
    _write_csv(phase_path, phase, TRAJECTORY_FIELDS)
    metadata = {
        "format": "theory33-independent-kinetic-data",
        "version": 1,
        "generator": "Maxwell-Juttner Gauss-Legendre moment quadrature",
        "config": asdict(config),
        "transport_rows": len(transport),
        "phase_rows": len(phase),
        "transport_sha256": sha256_file(transport_path),
        "phase_sha256": sha256_file(phase_path),
        "independence_boundary": (
            "generator imports no Tesseract or neural constitutive module"
        ),
        "provenance": {
            "eos": "relativistic Maxwell-Boltzmann equilibrium moments",
            "transport": (
                "positive transport Gram matrix under fixed "
                "relaxation-time quadrature"
            ),
            "phase": (
                "seeded grouped trajectories with context-dependent latent "
                "edge and observation noise"
            ),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata
