#!/usr/bin/env python3
"""Validate Vulkan reverse-through-time and shared neural-weight gradients."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import slangpy as spy


HERE = Path(__file__).resolve().parent


def main() -> int:
    document = json.loads((HERE / "hybrid_model.json").read_text(encoding="utf-8"))
    base_weights = np.asarray(document["master_weights"], dtype=np.float32)
    initial_np = np.asarray([-0.7, -0.15, 0.25, 0.8], dtype=np.float32)
    upstream = np.asarray([0.5, -0.75, 1.25, 0.4], dtype=np.float32)
    steps = 2

    device = spy.create_device(
        type=spy.DeviceType.vulkan,
        include_paths=[HERE],
    )
    module = spy.Module.load_from_file(
        device, str(HERE / "carter_master_ad.slang")
    )
    grid = spy.grid(shape=initial_np.shape)

    def forward(weights_np: np.ndarray, initial_values: np.ndarray) -> np.ndarray:
        weights = spy.Tensor.from_numpy(device, weights_np)
        initial = spy.Tensor.from_numpy(device, initial_values)
        output = spy.Tensor.empty(
            device, shape=initial_np.shape, dtype=module.float
        )
        module.carterBpttProbe(
            sample=grid,
            initialZ=initial,
            weights=weights,
            output=output,
        )
        return output.to_numpy()

    weights = spy.Tensor.from_numpy(device, base_weights).with_grads(zero=True)
    initial = spy.Tensor.from_numpy(device, initial_np).with_grads(zero=True)
    output = spy.Tensor.empty(
        device, shape=initial_np.shape, dtype=module.float
    ).with_grads()
    module.carterBpttProbe(
        sample=grid,
        initialZ=initial,
        weights=weights,
        output=output,
    )
    output.grad.copy_from_numpy(upstream)
    module.carterBpttProbe.bwds(
        sample=grid,
        initialZ=initial,
        weights=weights,
        output=output,
    )
    weight_grad = weights.grad.to_numpy()
    initial_grad = initial.grad.to_numpy()
    if not np.isfinite(weight_grad).all() or not np.isfinite(initial_grad).all():
        raise RuntimeError("The Vulkan BPTT kernel produced non-finite gradients.")

    epsilon = np.float32(2.0e-3)
    selected = [0, 6, 198, 700, 1254, 1286]
    errors = []
    for index in selected:
        plus = base_weights.copy()
        minus = base_weights.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        numeric = (
            np.dot(forward(plus, initial_np), upstream)
            - np.dot(forward(minus, initial_np), upstream)
        ) / (2.0 * epsilon)
        errors.append(abs(float(weight_grad[index] - numeric)))

    initial_plus = initial_np.copy()
    initial_minus = initial_np.copy()
    initial_plus[2] += epsilon
    initial_minus[2] -= epsilon
    numeric_initial = (
        np.dot(forward(base_weights, initial_plus), upstream)
        - np.dot(forward(base_weights, initial_minus), upstream)
    ) / (2.0 * epsilon)
    initial_error = abs(float(initial_grad[2] - numeric_initial))
    max_error = max(errors + [initial_error])
    if max_error > 4.0e-4:
        raise RuntimeError(f"Vulkan BPTT finite-difference error {max_error}.")

    print("backend: Vulkan / Slang reverse-through-time")
    print(f"unroll steps: {steps}")
    print(f"shared weights differentiated: {weight_grad.size}")
    print(f"nonzero weight gradients: {np.count_nonzero(weight_grad)}")
    print(f"max sampled finite-difference error: {max_error:.3g}")
    print("native recurrent BPTT probe passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
