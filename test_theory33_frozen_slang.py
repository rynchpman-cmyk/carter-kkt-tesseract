#!/usr/bin/env python3
"""Validate the qualified frozen Theory 3.3 module on Vulkan."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import slangpy as spy

from export_theory33_frozen_slang import (
    effective_lattice,
    render,
    sigmoid,
    softplus,
)


HERE = Path(__file__).resolve().parent


def lattice(values: list[float], coordinate: np.ndarray) -> np.ndarray:
    scaled = (np.clip(coordinate, -1.0, 1.0) + 1.0) * (
        len(values) - 1
    ) / 2.0
    index = np.clip(np.floor(scaled).astype(np.int64), 0, len(values) - 2)
    fraction = scaled - index
    values_array = np.asarray(values)
    return (
        values_array[index]
        + fraction * (values_array[index + 1] - values_array[index])
    )


def controller_reference(
    artifact: dict,
    z: np.ndarray,
) -> np.ndarray:
    state = artifact["variants"]["controller"]["state"]
    negative = effective_lattice(state, "negative")
    positive = effective_lattice(state, "positive")
    negative_coordinate = np.clip(
        2.0 * (z + 1.0) / ((-2.0 / 3.0 - 0.05) + 1.0) - 1.0,
        -1.0,
        1.0,
    )
    positive_coordinate = np.clip(
        2.0
        * (z - (2.0 / 3.0 - 0.05))
        / (1.05 - (2.0 / 3.0 - 0.05))
        - 1.0,
        -1.0,
        1.0,
    )
    return np.where(
        z < 0.0,
        lattice(negative, negative_coordinate),
        lattice(positive, positive_coordinate),
    )


def mobility_reference(
    artifact: dict,
    context: np.ndarray,
    force: np.ndarray,
) -> np.ndarray:
    state = artifact["variants"]["mobility"]["state"]
    input_weight = np.asarray(state["input_layer.weight"])
    input_bias = np.asarray(state["input_layer.bias"])
    output_weight = np.asarray(state["output_layer.weight"])
    output_bias = np.asarray(state["output_layer.bias"])
    hidden = np.tanh(context @ input_weight.T + input_bias)
    raw = hidden @ output_weight.T + output_bias
    diagonal0 = np.vectorize(softplus)(raw[:, 0]) + 1.0e-6
    diagonal1 = np.vectorize(softplus)(raw[:, 2]) + 1.0e-6
    l00 = diagonal0**2
    l01 = diagonal0 * raw[:, 1]
    l11 = raw[:, 1] ** 2 + diagonal1**2
    response0 = l00 * force[:, 0] + l01 * force[:, 1]
    response1 = l01 * force[:, 0] + l11 * force[:, 1]
    entropy = force[:, 0] * response0 + force[:, 1] * response1
    return np.stack(
        [l00, l01, l11, response0, response1, entropy], axis=-1
    )


def covariant_mobility_reference(
    mobility: np.ndarray,
    metric: np.ndarray,
    baryon_velocity: np.ndarray,
    carrier_velocity: np.ndarray,
    force: np.ndarray,
) -> np.ndarray:
    output = []
    for index in range(force.shape[0]):
        g = metric[index]
        u = baryon_velocity[index]
        carrier_u = carrier_velocity[index]
        gamma = -u @ g @ carrier_u
        spatial = carrier_u - gamma * u
        norm = math.sqrt(max(float(spatial @ g @ spatial), 0.0))
        if norm <= 1.0e-9:
            axis = np.asarray([0.0, 1.0, 0.0, 0.0])
            projector = np.linalg.inv(g) + np.outer(u, u)
            spatial = projector @ (g @ axis)
            norm = math.sqrt(max(float(spatial @ g @ spatial), 1.0e-18))
        spatial = spatial / norm
        response0 = mobility[index, 3]
        response1 = mobility[index, 4]
        flux0 = -response0 * spatial
        flux1 = -response1 * spatial
        output.append(
            [
                *flux0,
                *flux1,
                mobility[index, 5],
                mobility[index, 0],
                mobility[index, 1],
                mobility[index, 2],
            ]
        )
    return np.asarray(output)


def drift_network(artifact: dict, y: np.ndarray) -> np.ndarray:
    state = artifact["variants"]["drift_master"]["state"]
    result = (
        0.5
        * softplus(float(state["raw_quadratic"]))
        * y**2
    )
    sharpness = float(state["sharpness"])
    for knot, raw_weight in zip(
        state["knots"], state["raw_weights"], strict=True
    ):
        origin = -sharpness * float(knot)
        result += softplus(float(raw_weight)) * (
            np.vectorize(softplus)(sharpness * (y - float(knot)))
            - softplus(origin)
            - sigmoid(origin) * sharpness * y
        )
    return result


def master_reference(
    artifact: dict,
    features: np.ndarray,
) -> np.ndarray:
    n2, d2, cross, entropy = features.T
    density = np.sqrt(n2)
    carrier = np.sqrt(d2)
    pressure = np.exp((5.0 / 3.0 - 1.0) * entropy) * density ** (5.0 / 3.0)
    thermal = density + pressure / (5.0 / 3.0 - 1.0)
    anchor = carrier - 0.05 * density
    base = -(
        thermal
        + 0.2 * carrier ** (1.0 + 0.9**2)
        + 0.5 * anchor**2 / 10.0
    )
    config = artifact["data_config"]
    temperature = pressure / density
    density_coordinate = np.clip(
        2.0
        * (density - config["density_minimum"])
        / (config["density_maximum"] - config["density_minimum"])
        - 1.0,
        -1.5,
        1.5,
    )
    temperature_coordinate = np.clip(
        2.0
        * (
            np.log(temperature)
            - math.log(config["temperature_minimum"])
        )
        / (
            math.log(config["temperature_maximum"])
            - math.log(config["temperature_minimum"])
        )
        - 1.0,
        -1.5,
        1.5,
    )
    fraction_coordinate = np.clip(
        2.0
        * (
            carrier / density
            - config["fraction_minimum"]
        )
        / (config["fraction_maximum"] - config["fraction_minimum"])
        - 1.0,
        -1.5,
        1.5,
    )
    phase = artifact["variants"]["phase"]
    threshold = (
        phase["base"]
        + phase["temperature_coefficient"] * temperature_coordinate
        + phase["density_coefficient"] * density_coordinate
        + phase["fraction_coefficient"] * fraction_coordinate
    )
    drift = np.maximum(cross / (density * carrier) - 1.0, 0.0)
    scale = artifact["variants"]["drift_master"]["drift_scale"]
    correction = (
        artifact["variants"]["drift_master"]["energy_scale"]
        * (
            drift_network(artifact, drift / scale)
            - drift_network(artifact, threshold / scale)
        )
    )
    return base - np.where(drift >= threshold, correction, 0.0)


def finite_difference_gradient(
    artifact: dict,
    features: np.ndarray,
) -> np.ndarray:
    gradient = np.zeros_like(features, dtype=np.float64)
    for column in range(features.shape[1]):
        step = 2.0e-5 * np.maximum(np.abs(features[:, column]), 1.0e-3)
        plus = features.copy()
        minus = features.copy()
        plus[:, column] += step
        minus[:, column] -= step
        gradient[:, column] = (
            master_reference(artifact, plus)
            - master_reference(artifact, minus)
        ) / (2.0 * step)
    return gradient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=HERE / "theory33_frozen_variants.json",
    )
    parser.add_argument(
        "--module",
        type=Path,
        default=HERE / "theory33_qualified_frozen.slang",
    )
    args = parser.parse_args()
    artifact = json.loads(
        args.artifact.read_text(encoding="utf-8")
    )
    generated = render(
        artifact,
        (HERE / "theory33_hybrid.slang").read_text(encoding="utf-8"),
    )
    tracked = args.module.read_text(encoding="utf-8")
    if generated != tracked:
        raise RuntimeError("tracked frozen Slang module is stale")

    density = np.asarray([0.08, 0.10, 0.13, 0.17], dtype=np.float64)
    carrier = np.asarray([0.003, 0.005, 0.008, 0.014], dtype=np.float64)
    drift = np.asarray([0.002, 0.007, 0.011, 0.015], dtype=np.float64)
    entropy = np.asarray([0.8, 1.0, 1.1, 1.2], dtype=np.float64)
    features_np = np.stack(
        [
            density**2,
            carrier**2,
            density * carrier * (1.0 + drift),
            entropy,
        ],
        axis=-1,
    ).astype(np.float32)
    expected_master = master_reference(
        artifact, features_np.astype(np.float64)
    )
    expected_gradient = finite_difference_gradient(
        artifact, features_np.astype(np.float64)
    )
    z_np = np.asarray([-0.95, -0.80, 0.70, 0.90], dtype=np.float32)
    expected_controller = controller_reference(artifact, z_np)
    context_np = np.asarray(
        [
            [-1.0, -1.0, -1.0],
            [-0.2, 0.1, 0.4],
            [0.5, -0.4, 0.8],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    force_np = np.asarray(
        [[0.02, -0.1], [-0.03, 0.2], [0.01, 0.3], [0.04, -0.2]],
        dtype=np.float32,
    )
    expected_mobility = mobility_reference(
        artifact, context_np, force_np
    )
    rapidity = np.asarray([-0.4, 0.1, 0.55, -0.2], dtype=np.float32)
    carrier_rapidity = rapidity + np.asarray(
        [0.15, -0.2, 0.25, 0.12], dtype=np.float32
    )
    baryon_velocity_np = np.stack(
        [
            np.cosh(rapidity),
            np.sinh(rapidity),
            np.zeros_like(rapidity),
            np.zeros_like(rapidity),
        ],
        axis=-1,
    ).astype(np.float32)
    carrier_velocity_np = np.stack(
        [
            np.cosh(carrier_rapidity),
            np.sinh(carrier_rapidity),
            np.zeros_like(rapidity),
            np.zeros_like(rapidity),
        ],
        axis=-1,
    ).astype(np.float32)
    metric_np = np.tile(
        np.diag([-1.0, 1.0, 1.0, 1.0])[None, :, :],
        (features_np.shape[0], 1, 1),
    ).astype(np.float32)
    expected_covariant = covariant_mobility_reference(
        expected_mobility,
        metric_np,
        baryon_velocity_np,
        carrier_velocity_np,
        force_np,
    )

    device = spy.create_device(
        type=spy.DeviceType.vulkan,
        include_paths=[HERE],
    )
    module = spy.Module.load_from_file(
        device, str(args.module)
    )
    count = features_np.shape[0]
    grid = spy.grid(shape=(count,))
    features = spy.Tensor.from_numpy(
        device, features_np
    ).with_grads(zero=True)
    master_output = spy.Tensor.empty(
        device, shape=(count,), dtype=module.float
    ).with_grads()
    module.qualifiedTheory33Master(
        sample=grid, features=features, output=master_output
    )
    master_output.grad.copy_from_numpy(
        np.ones(count, dtype=np.float32)
    )
    module.qualifiedTheory33Master.bwds(
        sample=grid, features=features, output=master_output
    )

    z = spy.Tensor.from_numpy(device, z_np)
    controller_output = spy.Tensor.empty(
        device, shape=(count,), dtype=module.float
    )
    module.qualifiedKappa(
        sample=grid, z=z, output=controller_output
    )
    context = spy.Tensor.from_numpy(device, context_np)
    force = spy.Tensor.from_numpy(device, force_np)
    mobility_output = spy.Tensor.empty(
        device, shape=(count, 6), dtype=module.float
    )
    module.qualifiedMobility(
        sample=grid,
        inputContext=context,
        force=force,
        output=mobility_output,
    )
    metric = spy.Tensor.from_numpy(
        device, metric_np.reshape(count, 16)
    )
    baryon_velocity = spy.Tensor.from_numpy(
        device, baryon_velocity_np
    )
    carrier_velocity = spy.Tensor.from_numpy(
        device, carrier_velocity_np
    )
    covariant_output = spy.Tensor.empty(
        device, shape=(count, 12), dtype=module.float
    )
    module.qualifiedCovariantMobility(
        sample=grid,
        inputContext=context,
        metric=metric,
        baryonVelocity=baryon_velocity,
        carrierVelocity=carrier_velocity,
        force=force,
        output=covariant_output,
    )

    master_error = float(
        np.max(np.abs(master_output.to_numpy() - expected_master))
    )
    reverse_error = float(
        np.max(np.abs(features.grad.to_numpy() - expected_gradient))
    )
    controller_error = float(
        np.max(
            np.abs(controller_output.to_numpy() - expected_controller)
        )
    )
    mobility_error = float(
        np.max(
            np.abs(mobility_output.to_numpy() - expected_mobility)
        )
    )
    covariant_error = float(
        np.max(
            np.abs(covariant_output.to_numpy() - expected_covariant)
        )
    )
    minimum_entropy = float(mobility_output.to_numpy()[:, 5].min())
    if (
        master_error > 3.0e-5
        or reverse_error > 3.0e-3
        or controller_error > 3.0e-6
        or mobility_error > 3.0e-6
        or covariant_error > 5.0e-6
        or minimum_entropy < -1.0e-7
    ):
        raise RuntimeError(
            "frozen Slang mismatch: "
            f"master={master_error:.3g}, reverse={reverse_error:.3g}, "
            f"controller={controller_error:.3g}, "
            f"mobility={mobility_error:.3g}, "
            f"covariant={covariant_error:.3g}, "
            f"entropy={minimum_entropy:.3g}"
        )
    print("backend: Vulkan / qualified frozen Theory 3.3")
    print(f"master error:     {master_error:.3g}")
    print(f"reverse error:    {reverse_error:.3g}")
    print(f"controller error: {controller_error:.3g}")
    print(f"mobility error:   {mobility_error:.3g}")
    print(f"covariant error:  {covariant_error:.3g}")
    print(f"minimum entropy:  {minimum_entropy:.3g}")
    print("qualified frozen Slang validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
