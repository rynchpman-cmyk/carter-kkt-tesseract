#!/usr/bin/env python3
"""End-to-end primal and reverse test for the complete physical recurrence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import slangpy as spy


HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--physical-only",
        action="store_true",
        help="Stop after validating the compact physical-closure reverse.",
    )
    args = parser.parse_args()
    document = json.loads((HERE / "hybrid_model.json").read_text(encoding="utf-8"))
    weights_np = np.asarray(document["master_weights"], dtype=np.float32)
    z_np = np.asarray([-0.8, -0.2, 0.2, 0.8], dtype=np.float32)

    device = spy.create_device(type=spy.DeviceType.vulkan, include_paths=[HERE])
    module = spy.Module.load_from_file(
        device, str(HERE / "carter_tesseract_full.slang")
    )
    weights = spy.Tensor.from_numpy(device, weights_np).with_grads(zero=True)
    z = spy.Tensor.from_numpy(device, z_np)
    full_result = spy.Tensor.empty(
        device, shape=z_np.shape, dtype=module.float4
    )
    tape = spy.Tensor.empty(device, shape=(z_np.size, 32), dtype=module.float)
    grid = spy.grid(shape=z_np.shape)
    module.fullPhysicalStep(z=z, weights=weights, _result=full_result)
    module.fullPhysicalTapeKernel(
        sample=grid, zValues=z, weights=weights, hardTape=tape
    )

    tape_np = tape.to_numpy()
    closure = spy.Tensor.empty(
        device, shape=(z_np.size, 2, 8), dtype=module.float
    ).with_grads(zero=True)
    result = spy.Tensor.empty(
        device, shape=z_np.shape, dtype=module.float
    ).with_grads()
    closure_grid = spy.grid(shape=(z_np.size, 2))
    module.neuralClosureKernel(
        index=closure_grid,
        zValues=z,
        hardTape=tape,
        weights=weights,
        closure=closure,
    )
    module.physicalClosureKernel(
        sample=grid,
        zValues=z,
        hardTape=tape,
        closure=closure,
        output=result,
    )
    base_continuous_primal = result.to_numpy()
    upstream_np = np.asarray([0.5, -0.75, 1.25, 0.4], dtype=np.float32)
    closure_base_np = closure.to_numpy()
    closure_grad_np = np.zeros_like(closure_base_np)
    closure_epsilon = np.float32(2.0e-2)
    print("running piecewise physical closure VJP...", flush=True)
    for side in range(2):
        for component in range(8):
            outputs = []
            for direction in (-1.0, 1.0):
                perturbed = closure_base_np.copy()
                perturbed[:, side, component] += (
                    np.float32(direction) * closure_epsilon
                )
                closure.copy_from_numpy(perturbed)
                module.physicalClosureKernel(
                    sample=grid,
                    zValues=z,
                    hardTape=tape,
                    closure=closure,
                    output=result,
                )
                outputs.append(result.to_numpy())
            derivative = (outputs[1] - outputs[0]) / (
                np.float32(2.0) * closure_epsilon
            )
            closure_grad_np[:, side, component] = upstream_np * derivative
    closure.copy_from_numpy(closure_base_np)
    print("physical closure VJP succeeded", flush=True)
    if not np.count_nonzero(closure_grad_np):
        raise RuntimeError("Physical closure reverse produced zero adjoints.")
    print(
        "nonzero closure adjoints:",
        int(np.count_nonzero(closure_grad_np)),
        "max:",
        float(np.max(np.abs(closure_grad_np))),
        flush=True,
    )
    if args.physical_only:
        print("physical-only reverse validation succeeded", flush=True)
        return 0
    print("compiling/running exact neural closure VJP...", flush=True)
    closure_adjoint = spy.Tensor.from_numpy(
        device, closure_grad_np.astype(np.float32)
    )
    sample_gradient = spy.Tensor.empty(
        device, shape=(z_np.size, 2, weights_np.size), dtype=module.float
    )
    module.neuralClosureBackwardKernel(
        index=closure_grid,
        zValues=z,
        hardTape=tape,
        weights=weights,
        sampleGradient=sample_gradient,
        closureAdjoint=closure_adjoint,
    )
    sample_gradient_np = sample_gradient.to_numpy()
    print(
        "nonzero per-sample weight VJP:",
        int(np.count_nonzero(sample_gradient_np)),
        "max:",
        float(np.max(np.abs(sample_gradient_np))),
        flush=True,
    )
    module.reduceWeightGradientKernel(
        weightIndex=spy.grid(shape=(weights_np.size,)),
        sampleGradient=sample_gradient,
        weightGradient=weights.grad,
    )
    print("neural closure VJP succeeded", flush=True)

    primal = full_result.to_numpy()
    continuous_primal = base_continuous_primal
    weight_grad = weights.grad.to_numpy()
    largest = np.argsort(np.abs(weight_grad))[-6:]
    epsilon = np.float32(2.0e-2)
    finite_difference = np.empty(largest.size, dtype=np.float32)
    for position, weight_index in enumerate(largest):
        objectives = []
        for direction in (-1.0, 1.0):
            perturbed = weights_np.copy()
            perturbed[weight_index] += direction * epsilon
            weights.copy_from_numpy(perturbed)
            module.neuralClosureKernel(
                index=closure_grid,
                zValues=z,
                hardTape=tape,
                weights=weights,
                closure=closure,
            )
            objectives.append(
                float(
                    np.sum(
                        closure.to_numpy().astype(np.float64)
                        * closure_grad_np.astype(np.float64)
                    )
                )
            )
        finite_difference[position] = (
            objectives[1] - objectives[0]
        ) / (2.0 * float(epsilon))
    weights.copy_from_numpy(weights_np)
    fd_error = np.abs(weight_grad[largest] - finite_difference)
    max_fd_error = float(fd_error.max())
    max_fd_relative = float(
        np.max(
            fd_error
            / np.maximum(np.abs(finite_difference), np.float32(1.0e-6))
        )
    )
    if not np.isfinite(primal).all():
        raise RuntimeError("Complete Slang recurrence produced non-finite output.")
    if not np.isfinite(weight_grad).all():
        raise RuntimeError("Complete Slang recurrence produced non-finite gradients.")
    if not np.count_nonzero(weight_grad):
        raise RuntimeError("Complete Slang recurrence produced zero weight gradients.")
    if not np.all(primal[:, 3] == 1.0):
        raise RuntimeError("Complete Slang recurrence produced an invalid KKT state.")
    if not np.all(tape_np[:, 20] == 1.0):
        raise RuntimeError("Hard tape contains an invalid KKT certificate.")
    if not np.all(tape_np[:, 22] == 1.0):
        raise RuntimeError("Opposite branch contains an invalid KKT certificate.")
    max_primal_error = float(
        np.max(np.abs(continuous_primal - primal[:, 0]))
    )
    if max_primal_error > 2.0e-5:
        raise RuntimeError(
            "Compact differentiable recurrence disagrees with the exact full "
            f"primal by {max_primal_error:.9g}."
        )
    if max_fd_error > 2.0e-5 and max_fd_relative > 0.08:
        raise RuntimeError(
            "Manual neural VJP disagrees with central finite differences: "
            f"absolute={max_fd_error:.9g}, relative={max_fd_relative:.9g}; "
            f"indices={largest.tolist()}, "
            f"VJP={weight_grad[largest].tolist()}, "
            f"FD={finite_difference.tolist()}."
        )
    print("backend: Vulkan / complete Carter-KKT Slang recurrence")
    print("z_next:", " ".join(f"{value:.9g}" for value in primal[:, 0]))
    print("active masks:", primal[:, 1].astype(np.int32).tolist())
    print("classifier masks:", primal[:, 2].astype(np.int32).tolist())
    print("boundary iota:", tape_np[:, 0].astype(np.int32).tolist())
    print(f"compact/full primal max error: {max_primal_error:.9g}")
    print(f"weights differentiated: {weight_grad.size}")
    print(f"nonzero weight gradients: {np.count_nonzero(weight_grad)}")
    print(
        "VJP finite-difference indices:",
        largest.astype(np.int32).tolist(),
    )
    print(f"VJP max finite-difference error: {max_fd_error:.9g}")
    print(f"VJP max finite-difference relative error: {max_fd_relative:.9g}")
    print("complete physical reverse pass succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
