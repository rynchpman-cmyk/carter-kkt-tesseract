#!/usr/bin/env python3
"""Validate native Vulkan reverse mode for all Carter master weights."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import slangpy as spy


HERE = Path(__file__).resolve().parent


def reference_master_backward(
    features: np.ndarray, weights: np.ndarray, upstream: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    skip = features @ weights[0:5] + weights[5]
    l1_weight = weights[6:166].reshape(32, 5)
    l1_bias = weights[166:198]
    l2_weight = weights[198:1222].reshape(32, 32)
    l2_bias = weights[1222:1254]
    l3_weight = weights[1254:1286]
    l3_bias = weights[1286]
    hidden1 = np.tanh(features @ l1_weight.T + l1_bias)
    hidden2 = np.tanh(hidden1 @ l2_weight.T + l2_bias)
    output = skip + hidden2 @ l3_weight + l3_bias

    grad_weights = np.zeros_like(weights)
    grad_weights[0:5] = features.T @ upstream
    grad_weights[5] = upstream.sum()
    grad_weights[1254:1286] = hidden2.T @ upstream
    grad_weights[1286] = upstream.sum()
    grad_hidden2 = upstream[:, None] * l3_weight[None, :]
    grad_pre2 = grad_hidden2 * (1.0 - hidden2 * hidden2)
    grad_weights[198:1222] = (grad_pre2.T @ hidden1).reshape(-1)
    grad_weights[1222:1254] = grad_pre2.sum(axis=0)
    grad_hidden1 = grad_pre2 @ l2_weight
    grad_pre1 = grad_hidden1 * (1.0 - hidden1 * hidden1)
    grad_weights[6:166] = (grad_pre1.T @ features).reshape(-1)
    grad_weights[166:198] = grad_pre1.sum(axis=0)
    grad_features = (
        upstream[:, None] * weights[0:5][None, :]
        + grad_pre1 @ l1_weight
    )
    return output, grad_weights, grad_features


def main() -> int:
    document = json.loads((HERE / "hybrid_model.json").read_text(encoding="utf-8"))
    weights_np = np.asarray(document["master_weights"], dtype=np.float32)
    features_np = np.asarray(
        [
            [0.7, 0.5, -0.2, 1.1, 0.1],
            [1.3, 0.8, 0.4, 0.9, -0.3],
            [0.2, 1.1, -0.7, 1.0, 0.25],
            [1.8, 0.3, 0.9, 1.2, -0.1],
        ],
        dtype=np.float32,
    )
    upstream_np = np.asarray([1.0, -0.5, 0.25, 1.5], dtype=np.float32)

    device = spy.create_device(
        type=spy.DeviceType.vulkan,
        include_paths=[HERE],
    )
    module = spy.Module.load_from_file(
        device, str(HERE / "carter_master_ad.slang")
    )
    features = spy.Tensor.from_numpy(device, features_np).with_grads(zero=True)
    weights = spy.Tensor.from_numpy(device, weights_np).with_grads(zero=True)
    output = spy.Tensor.empty(
        device, shape=(features_np.shape[0],), dtype=module.float
    ).with_grads()
    grid = spy.grid(shape=(features_np.shape[0],))
    module.carterMaster(
        sample=grid, features=features, weights=weights, output=output
    )
    output.grad.copy_from_numpy(upstream_np)
    module.carterMaster.bwds(
        sample=grid, features=features, weights=weights, output=output
    )

    reference_output, reference_weight_grad, reference_feature_grad = (
        reference_master_backward(features_np, weights_np, upstream_np)
    )
    output_error = np.max(np.abs(output.to_numpy() - reference_output))
    slang_weight_grad = weights.grad.to_numpy()
    slang_feature_grad = features.grad.to_numpy()
    weight_delta = np.abs(slang_weight_grad - reference_weight_grad)
    feature_delta = np.abs(slang_feature_grad - reference_feature_grad)
    weight_error = np.max(weight_delta)
    feature_error = np.max(feature_delta)
    if output_error > 2.0e-5 or weight_error > 2.0e-5 or feature_error > 2.0e-5:
        weight_index = int(np.argmax(weight_delta))
        feature_index = np.unravel_index(np.argmax(feature_delta), feature_delta.shape)
        raise RuntimeError(
            "Slang/reference AD mismatch: "
            f"output={output_error}, weights={weight_error} at {weight_index} "
            f"({slang_weight_grad[weight_index]} vs "
            f"{reference_weight_grad[weight_index]}), "
            f"features={feature_error} at {feature_index} "
            f"({slang_feature_grad[feature_index]} vs "
            f"{reference_feature_grad[feature_index]})"
        )
    print("backend: Vulkan / Slang reverse mode")
    print(f"weights differentiated: {weights_np.size}")
    print(f"max output error:  {output_error:.3g}")
    print(f"max weight error:  {weight_error:.3g}")
    print(f"max feature error: {feature_error:.3g}")
    print("native Carter master reverse-mode test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
