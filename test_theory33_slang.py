#!/usr/bin/env python3
"""Validate the Theory 3.3 M1 constitutive kernel and native reverse mode."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import slangpy as spy


HERE = Path(__file__).resolve().parent
GAMMA_AD = np.float32(5.0 / 3.0)
CARRIER_K = np.float32(0.2)
CARRIER_SOUND_SPEED = np.float32(0.9)
CHEMICAL_SUSCEPTIBILITY = np.float32(10.0)
TARGET_FRACTION = np.float32(0.05)


def reference(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n2, d2, x2, entropy = features.T
    n = np.sqrt(n2)
    d = np.sqrt(d2)
    sound2 = CARRIER_SOUND_SPEED**2
    pressure = np.exp((GAMMA_AD - 1.0) * entropy) * n**GAMMA_AD
    thermal = n + pressure / (GAMMA_AD - 1.0)
    anchor = d - TARGET_FRACTION * n
    carrier = CARRIER_K * d ** (1.0 + sound2)
    chemical = 0.5 * anchor**2 / CHEMICAL_SUSCEPTIBILITY
    lambda_value = -(thermal + carrier + chemical)
    chemical_n = (
        1.0
        + GAMMA_AD
        * pressure
        / ((GAMMA_AD - 1.0) * n)
        - TARGET_FRACTION * anchor / CHEMICAL_SUSCEPTIBILITY
    )
    chemical_d = (
        CARRIER_K * (1.0 + sound2) * d**sound2
        + anchor / CHEMICAL_SUSCEPTIBILITY
    )
    b_n = chemical_n / n
    b_d = chemical_d / d
    entrainment = np.zeros_like(n)
    psi = lambda_value + b_n * n2 + b_d * d2
    h_nn = (
        GAMMA_AD * pressure / n2
        + TARGET_FRACTION**2 / CHEMICAL_SUSCEPTIBILITY
    )
    h_nd = np.full_like(n, -TARGET_FRACTION / CHEMICAL_SUSCEPTIBILITY)
    h_dd = (
        CARRIER_K
        * (1.0 + sound2)
        * sound2
        * d ** (sound2 - 1.0)
        + 1.0 / CHEMICAL_SUSCEPTIBILITY
    )

    def minimum_eigenvalue(a, b, c):
        return 0.5 * (a + c - np.sqrt((a - c) ** 2 + 4.0 * b**2))

    constitutive = np.stack(
        [
            lambda_value,
            b_n,
            b_d,
            entrainment,
            psi,
            x2 / (n * d),
            minimum_eigenvalue(b_n, entrainment, b_d),
            minimum_eigenvalue(h_nn, h_nd, h_dd),
        ],
        axis=-1,
    )
    gradient = np.stack(
        [
            -0.5 * b_n,
            -0.5 * b_d,
            -entrainment,
            -pressure,
        ],
        axis=-1,
    )
    return constitutive.astype(np.float32), gradient.astype(np.float32)


def main() -> int:
    density = np.asarray([0.08, 0.10, 0.12, 0.15], dtype=np.float32)
    carrier = np.asarray([0.003, 0.005, 0.008, 0.012], dtype=np.float32)
    relative_gamma = np.asarray([1.0, 1.01, 1.04, 1.08], dtype=np.float32)
    entropy = np.asarray([0.8, 1.0, 1.1, 1.2], dtype=np.float32)
    features_np = np.stack(
        [
            density**2,
            carrier**2,
            density * carrier * relative_gamma,
            entropy,
        ],
        axis=-1,
    )
    upstream_np = np.asarray([1.0, -0.5, 0.25, 1.5], dtype=np.float32)
    expected, expected_gradient = reference(features_np)

    device = spy.create_device(
        type=spy.DeviceType.vulkan,
        include_paths=[HERE],
    )
    module = spy.Module.load_from_file(
        device, str(HERE / "theory33_hybrid.slang")
    )
    features = spy.Tensor.from_numpy(device, features_np).with_grads(zero=True)
    master_output = spy.Tensor.empty(
        device, shape=(features_np.shape[0],), dtype=module.float
    ).with_grads()
    constitutive_output = spy.Tensor.empty(
        device, shape=(features_np.shape[0], 8), dtype=module.float
    )
    grid = spy.grid(shape=(features_np.shape[0],))
    module.theory33Master(
        sample=grid,
        features=features,
        output=master_output,
    )
    module.theory33Constitutive(
        sample=grid,
        features=features,
        output=constitutive_output,
    )
    master_output.grad.copy_from_numpy(upstream_np)
    module.theory33Master.bwds(
        sample=grid,
        features=features,
        output=master_output,
    )

    constitutive_error = float(
        np.max(np.abs(constitutive_output.to_numpy() - expected))
    )
    expected_weighted_gradient = expected_gradient * upstream_np[:, None]
    reverse_error = float(
        np.max(
            np.abs(
                features.grad.to_numpy() - expected_weighted_gradient
            )
        )
    )
    if constitutive_error > 2.0e-4 or reverse_error > 2.0e-4:
        raise RuntimeError(
            "Theory 3.3 Slang mismatch: "
            f"constitutive={constitutive_error:.3g}, "
            f"reverse={reverse_error:.3g}"
        )
    print("backend: Vulkan / Slang reverse mode")
    print(f"max constitutive error: {constitutive_error:.3g}")
    print(f"max reverse error:      {reverse_error:.3g}")
    print("Theory 3.3 Slang constitutive validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
