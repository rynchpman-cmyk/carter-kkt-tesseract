#!/usr/bin/env python3
"""Progressively integrate and backpropagate through the full Slang recurrence.

Every forward step evaluates the literal Carter/KKT shader and records its
piecewise-constant hard tape. The reverse sweep uses the validated continuous
physical reverse and exact neural jet VJP. The scalar temporal Jacobian is
measured with a centered perturbation of the exact Slang step while requiring
the KKT/branch/classifier/parity signature to remain unchanged. An adaptive
residual-flow scale keeps the numerical continuation below a requested local
gain without clamping the physical state.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import slangpy as spy


HERE = Path(__file__).resolve().parent
TAPE_WIDTH = 32
DECISION_COLUMNS = np.asarray(
    [0, 1, 18, 19, 21, 29, 30], dtype=np.int32
)


class StepFailure(RuntimeError):
    """A forward step left the valid piecewise physical regime."""


@dataclass
class StepRecord:
    index: int
    z: np.ndarray
    raw_next: np.ndarray
    z_next: np.ndarray
    integration_scale: float
    raw_temporal_jacobian: np.ndarray | None
    full_result: np.ndarray
    tape_np: np.ndarray
    z_tensor: Any
    tape_tensor: Any
    closure: Any
    output: Any

    @property
    def signature(self) -> np.ndarray:
        return self.tape_np[:, DECISION_COLUMNS].astype(np.int32)


@dataclass
class HorizonResult:
    horizon: int
    completed: bool
    records: list[StepRecord]
    failure: str | None = None
    loss: float | None = None
    max_temporal_jacobian: float | None = None
    max_post_capture_jacobian: float | None = None
    initial_adjoint_norm: float | None = None
    weight_gradient_norm: float | None = None
    nonzero_weight_gradients: int | None = None
    finite_difference_error: float | None = None
    finite_difference_relative: float | None = None
    min_kkt_margin: float | None = None
    min_classifier_margin: float | None = None
    min_parity_margin: float | None = None
    max_kkt_condition: float | None = None
    min_integration_scale: float | None = None
    max_lyapunov_growth: float | None = None
    max_lyapunov_ratio: float | None = None
    terminal_lyapunov: float | None = None


class FullPhysicsRuntime:
    def __init__(
        self,
        initial: np.ndarray,
        target: float,
        *,
        temporal_epsilon: float,
        closure_epsilon: float,
        max_state: float,
        integration_scale: float,
        max_local_gain: float,
        min_switch_margin: float,
        model_path: Path,
        module_path: Path,
    ) -> None:
        document = json.loads(
            model_path.read_text(encoding="utf-8")
        )
        self.base_weights = np.asarray(
            document["master_weights"], dtype=np.float32
        )
        self.initial = np.asarray(initial, dtype=np.float32)
        self.target = np.float32(target)
        self.temporal_epsilon = np.float32(temporal_epsilon)
        self.closure_epsilon = np.float32(closure_epsilon)
        self.max_state = float(max_state)
        self.integration_scale = float(integration_scale)
        self.max_local_gain = float(max_local_gain)
        self.min_switch_margin = float(min_switch_margin)
        self.batch = self.initial.size

        self.device = spy.create_device(
            type=spy.DeviceType.vulkan,
            include_paths=[HERE],
        )
        self.module = spy.Module.load_from_file(
            self.device, str(module_path)
        )
        self.weights = spy.Tensor.from_numpy(
            self.device, self.base_weights
        ).with_grads(zero=True)
        self.sample_grid = spy.grid(shape=(self.batch,))
        self.closure_grid = spy.grid(shape=(self.batch, 2))
        self.weight_grid = spy.grid(shape=(self.base_weights.size,))

    def _hard_tape(
        self, z_np: np.ndarray
    ) -> tuple[Any, np.ndarray, np.ndarray]:
        z_tensor = spy.Tensor.from_numpy(
            self.device, np.asarray(z_np, dtype=np.float32)
        )
        full_result = spy.Tensor.empty(
            self.device, shape=(self.batch,), dtype=self.module.float4
        )
        tape = spy.Tensor.empty(
            self.device,
            shape=(self.batch, TAPE_WIDTH),
            dtype=self.module.float,
        )
        self.module.fullPhysicalStep(
            z=z_tensor, weights=self.weights, _result=full_result
        )
        self.module.fullPhysicalTapeKernel(
            sample=self.sample_grid,
            zValues=z_tensor,
            weights=self.weights,
            hardTape=tape,
        )
        return z_tensor, full_result.to_numpy(), tape.to_numpy()

    def _continuous_forward(
        self,
        z_tensor: Any,
        tape_tensor: Any,
        *,
        differentiable: bool,
    ) -> tuple[Any, Any]:
        closure = spy.Tensor.empty(
            self.device,
            shape=(self.batch, 2, 8),
            dtype=self.module.float,
        )
        output = spy.Tensor.empty(
            self.device,
            shape=(self.batch,),
            dtype=self.module.float,
        )
        if differentiable:
            closure = closure.with_grads(zero=True)
            output = output.with_grads()
        self.module.neuralClosureKernel(
            index=self.closure_grid,
            zValues=z_tensor,
            hardTape=tape_tensor,
            weights=self.weights,
            closure=closure,
        )
        self.module.physicalClosureKernel(
            sample=self.sample_grid,
            zValues=z_tensor,
            hardTape=tape_tensor,
            closure=closure,
            output=output,
        )
        return closure, output

    @staticmethod
    def _valid_tape(tape_np: np.ndarray) -> bool:
        return bool(
            np.all(tape_np[:, 20] == 1.0)
            and np.all(tape_np[:, 22] == 1.0)
        )

    def forward_step(self, z_np: np.ndarray, index: int) -> StepRecord:
        z_tensor, full_result, tape_np = self._hard_tape(z_np)
        if not np.isfinite(full_result).all() or not np.isfinite(tape_np).all():
            raise StepFailure(f"step {index}: non-finite primal or hard tape")
        if not self._valid_tape(tape_np):
            invalid = np.flatnonzero(
                (tape_np[:, 20] != 1.0) | (tape_np[:, 22] != 1.0)
            ).tolist()
            raise StepFailure(
                f"step {index}: invalid KKT certificate in lanes {invalid}"
            )

        tape_tensor = spy.Tensor.from_numpy(
            self.device, tape_np.astype(np.float32)
        )
        closure, output = self._continuous_forward(
            z_tensor, tape_tensor, differentiable=True
        )
        raw_next = output.to_numpy()
        primal_error = float(np.max(np.abs(raw_next - full_result[:, 0])))
        if primal_error > 2.0e-5:
            raise StepFailure(
                f"step {index}: compact/full disagreement {primal_error:.3g}"
            )
        if not np.isfinite(raw_next).all():
            raise StepFailure(f"step {index}: non-finite raw next state")

        record = StepRecord(
            index=index,
            z=np.asarray(z_np, dtype=np.float32).copy(),
            raw_next=raw_next.copy(),
            z_next=raw_next.copy(),
            integration_scale=1.0,
            raw_temporal_jacobian=None,
            full_result=full_result.copy(),
            tape_np=tape_np.copy(),
            z_tensor=z_tensor,
            tape_tensor=tape_tensor,
            closure=closure,
            output=output,
        )
        raw_jacobian = self.raw_temporal_jacobian(record)
        scale = self.integration_scale
        if self.max_local_gain > 0.0:
            while scale > 1.0e-6:
                relaxed = 1.0 + scale * (raw_jacobian - 1.0)
                if float(np.max(np.abs(relaxed))) <= self.max_local_gain:
                    break
                scale *= 0.5
            if scale <= 1.0e-6:
                raise StepFailure(
                    f"step {index}: no integration scale satisfies "
                    f"local gain <= {self.max_local_gain:g}"
                )

        z_next = np.asarray(
            z_np + np.float32(scale) * (raw_next - z_np),
            dtype=np.float32,
        )
        if float(np.max(np.abs(z_next))) > self.max_state:
            raise StepFailure(
                f"step {index}: |z| exceeded {self.max_state:g} "
                f"(max={np.max(np.abs(z_next)):.6g})"
            )
        record.z_next = z_next.copy()
        record.integration_scale = scale
        record.raw_temporal_jacobian = raw_jacobian
        return record

    def forward_trajectory(
        self, horizon: int, *, retain_records: bool = True
    ) -> HorizonResult:
        z = self.initial.copy()
        records: list[StepRecord] = []
        for index in range(1, horizon + 1):
            try:
                record = self.forward_step(z, index)
            except StepFailure as exc:
                return HorizonResult(
                    horizon=horizon,
                    completed=False,
                    records=records,
                    failure=str(exc),
                )
            records.append(record)
            z = record.z_next
        if not retain_records:
            records = records[-1:]
        states = np.stack(
            [self.initial]
            + [record.z_next for record in records],
            axis=0,
        ).astype(np.float64)
        lyapunov = (states - float(self.target)) ** 2
        lyapunov_growth = lyapunov[1:] - lyapunov[:-1]
        denominator = np.maximum(lyapunov[:-1], 1.0e-20)
        return HorizonResult(
            horizon=horizon,
            completed=True,
            records=records,
            loss=self.loss(z),
            min_kkt_margin=min(
                float(np.min(record.tape_np[:, 23:25]))
                for record in records
            ),
            min_classifier_margin=min(
                float(np.min(record.tape_np[:, 25:27]))
                for record in records
            ),
            min_parity_margin=min(
                float(np.min(record.tape_np[:, 31]))
                for record in records
            ),
            max_kkt_condition=max(
                float(np.max(record.tape_np[:, 27:29]))
                for record in records
            ),
            min_integration_scale=min(
                record.integration_scale for record in records
            ),
            max_lyapunov_growth=float(np.max(lyapunov_growth)),
            max_lyapunov_ratio=float(
                np.max(lyapunov[1:] / denominator)
            ),
            terminal_lyapunov=float(np.max(lyapunov[-1])),
        )

    def loss(self, z: np.ndarray) -> float:
        residual = np.asarray(z, dtype=np.float64) - float(self.target)
        return 0.5 * float(np.mean(residual * residual))

    def terminal_adjoint(self, z: np.ndarray) -> np.ndarray:
        return (
            (np.asarray(z, dtype=np.float32) - self.target)
            / np.float32(self.batch)
        )

    def step_weight_vjp(
        self, record: StepRecord, output_adjoint: np.ndarray
    ) -> np.ndarray:
        closure_base = record.closure.to_numpy()
        closure_adjoint_np = np.zeros_like(closure_base)
        seed = (
            np.asarray(output_adjoint, dtype=np.float32)
            * np.float32(record.integration_scale)
        )
        # recurrenceValue multiplies every learned-closure contribution by
        # parity/normalization.  In an even parity cell the exact weight VJP is
        # therefore zero; enforce that identity instead of amplifying float32
        # cancellation noise from the centered physical probe.
        odd_parity = (
            record.tape_np[:, 30].astype(np.int32) & 1
        ).astype(np.float32)
        seed = seed * odd_parity
        for side in range(2):
            for component in range(8):
                outputs: list[np.ndarray] = []
                for direction in (-1.0, 1.0):
                    perturbed = closure_base.copy()
                    perturbed[:, side, component] += (
                        np.float32(direction) * self.closure_epsilon
                    )
                    record.closure.copy_from_numpy(perturbed)
                    self.module.physicalClosureKernel(
                        sample=self.sample_grid,
                        zValues=record.z_tensor,
                        hardTape=record.tape_tensor,
                        closure=record.closure,
                        output=record.output,
                    )
                    outputs.append(record.output.to_numpy())
                derivative = (outputs[1] - outputs[0]) / (
                    np.float32(2.0) * self.closure_epsilon
                )
                closure_adjoint_np[:, side, component] = seed * derivative
        record.closure.copy_from_numpy(closure_base)
        closure_adjoint = spy.Tensor.from_numpy(
            self.device, closure_adjoint_np.astype(np.float32)
        )
        sample_gradient = spy.Tensor.empty(
            self.device,
            shape=(self.batch, 2, self.base_weights.size),
            dtype=self.module.float,
        )
        self.module.neuralClosureBackwardKernel(
            index=self.closure_grid,
            zValues=record.z_tensor,
            hardTape=record.tape_tensor,
            weights=self.weights,
            sampleGradient=sample_gradient,
            closureAdjoint=closure_adjoint,
        )
        step_gradient = spy.Tensor.empty(
            self.device,
            shape=(self.base_weights.size,),
            dtype=self.module.float,
        )
        self.module.reduceWeightGradientKernel(
            weightIndex=self.weight_grid,
            sampleGradient=sample_gradient,
            weightGradient=step_gradient,
        )
        return step_gradient.to_numpy()

    def _continuous_at_perturbed_state(
        self,
        z_np: np.ndarray,
        base_tape: np.ndarray,
        expected_signature: np.ndarray,
    ) -> np.ndarray:
        z_tensor, _full_result, tape_np = self._hard_tape(z_np)
        if not self._valid_tape(tape_np):
            raise StepFailure("perturbed temporal probe has invalid KKT tape")
        signature = tape_np[:, DECISION_COLUMNS].astype(np.int32)
        if not np.array_equal(signature, expected_signature):
            raise StepFailure("perturbed temporal probe crossed a hard decision")

        # Re-evaluate continuous KKT values/tangents at the perturbed z, while
        # freezing the literal hard boundary decision from the primal step.
        tape_np[:, 0] = base_tape[:, 0]
        tape_np[:, 1] = base_tape[:, 1]
        tape_tensor = spy.Tensor.from_numpy(
            self.device, tape_np.astype(np.float32)
        )
        _closure, output = self._continuous_forward(
            z_tensor, tape_tensor, differentiable=False
        )
        return output.to_numpy()

    def raw_temporal_jacobian(self, record: StepRecord) -> np.ndarray:
        scale = np.float32(1.0) + np.abs(record.z)
        epsilon = self.temporal_epsilon * scale
        expected = record.signature
        last_error: Exception | None = None
        for _attempt in range(8):
            try:
                plus = self._continuous_at_perturbed_state(
                    record.z + epsilon,
                    record.tape_np,
                    expected,
                )
                minus = self._continuous_at_perturbed_state(
                    record.z - epsilon,
                    record.tape_np,
                    expected,
                )
                jacobian = (plus - minus) / (2.0 * epsilon)
                if not np.isfinite(jacobian).all():
                    raise StepFailure("non-finite temporal Jacobian")
                return jacobian.astype(np.float32)
            except StepFailure as exc:
                last_error = exc
                epsilon *= np.float32(0.5)
        raise StepFailure(
            f"could not remain inside the hard region: {last_error}"
        )

    def temporal_jacobian(self, record: StepRecord) -> np.ndarray:
        raw = record.raw_temporal_jacobian
        if raw is None:
            raw = self.raw_temporal_jacobian(record)
        scale = np.float32(record.integration_scale)
        return np.float32(1.0) + scale * (raw - np.float32(1.0))

    def backward(self, result: HorizonResult) -> np.ndarray:
        if not result.completed or not result.records:
            raise StepFailure("cannot backpropagate an incomplete horizon")
        adjoint = self.terminal_adjoint(result.records[-1].z_next)
        weight_gradient = np.zeros_like(self.base_weights)
        max_jacobian = 0.0
        max_post_capture_jacobian = 0.0

        for record in reversed(result.records):
            weight_gradient += self.step_weight_vjp(record, adjoint)
            jacobian = self.temporal_jacobian(record)
            max_jacobian = max(
                max_jacobian, float(np.max(np.abs(jacobian)))
            )
            if record.index > 1:
                max_post_capture_jacobian = max(
                    max_post_capture_jacobian,
                    float(np.max(np.abs(jacobian))),
                )
            adjoint = adjoint * jacobian

        result.max_temporal_jacobian = max_jacobian
        result.max_post_capture_jacobian = max_post_capture_jacobian
        result.initial_adjoint_norm = float(np.linalg.norm(adjoint))
        result.weight_gradient_norm = float(np.linalg.norm(weight_gradient))
        result.nonzero_weight_gradients = int(
            np.count_nonzero(weight_gradient)
        )
        return weight_gradient

    def validate_weight_gradient(
        self,
        result: HorizonResult,
        weight_gradient: np.ndarray,
        *,
        count: int,
        epsilon: float,
    ) -> tuple[float, float]:
        if count <= 0:
            return 0.0, 0.0
        indices = np.argsort(np.abs(weight_gradient))[-count:]
        base_signatures = [record.signature for record in result.records]
        base_scales = [record.integration_scale for record in result.records]
        errors: list[float] = []
        relatives: list[float] = []
        original = self.base_weights.copy()

        for weight_index in indices:
            objectives: list[float] = []
            for direction in (-1.0, 1.0):
                perturbed = original.copy()
                perturbed[weight_index] += np.float32(direction * epsilon)
                self.weights.copy_from_numpy(perturbed)
                trial = self.forward_trajectory(result.horizon)
                if not trial.completed:
                    raise StepFailure(
                        "weight finite difference left the valid trajectory: "
                        f"{trial.failure}"
                    )
                signatures = [record.signature for record in trial.records]
                if any(
                    not np.array_equal(actual, expected)
                    for actual, expected in zip(signatures, base_signatures)
                ):
                    raise StepFailure(
                        "weight finite difference crossed a hard decision"
                    )
                if any(
                    actual.integration_scale != expected
                    for actual, expected in zip(trial.records, base_scales)
                ):
                    raise StepFailure(
                        "weight finite difference changed an adaptive "
                        "integration decision"
                    )
                objectives.append(float(trial.loss))
            numeric = (objectives[1] - objectives[0]) / (2.0 * epsilon)
            error = abs(float(weight_gradient[weight_index]) - numeric)
            errors.append(error)
            relatives.append(error / max(abs(numeric), 1.0e-8))

        self.weights.copy_from_numpy(original)
        return max(errors, default=0.0), max(relatives, default=0.0)


def parse_horizons(value: str) -> list[int]:
    horizons = sorted({int(item) for item in value.split(",") if item.strip()})
    if not horizons or horizons[0] < 1:
        raise argparse.ArgumentTypeError("horizons must be positive integers")
    return horizons


def format_vector(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(value):.6g}" for value in values) + "]"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--horizons",
        type=parse_horizons,
        default=parse_horizons("1,2,3,4,8"),
    )
    parser.add_argument(
        "--initial",
        type=float,
        nargs="+",
        default=[-0.8, -0.2, 0.2, 0.8],
    )
    parser.add_argument("--target", type=float, default=0.2)
    parser.add_argument("--temporal-epsilon", type=float, default=2.0e-3)
    parser.add_argument("--closure-epsilon", type=float, default=2.0e-2)
    parser.add_argument("--weight-epsilon", type=float, default=3.0e-2)
    parser.add_argument("--fd-weights", type=int, default=4)
    parser.add_argument("--max-state", type=float, default=1.0e6)
    parser.add_argument(
        "--integration-scale",
        type=float,
        default=0.05,
        help="residual-flow step: z_next = z + scale*(F(z)-z)",
    )
    parser.add_argument(
        "--max-local-gain",
        type=float,
        default=1.1,
        help="halve integration scale until the local gain is below this",
    )
    parser.add_argument(
        "--min-switch-margin",
        type=float,
        default=1.0e-4,
        help="report trajectories closer than this to a hard switch",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=HERE / "hybrid_model.json",
    )
    parser.add_argument(
        "--module",
        type=Path,
        default=HERE / "carter_tesseract_full.slang",
    )
    args = parser.parse_args()
    if not 0.0 < args.integration_scale <= 1.0:
        parser.error("--integration-scale must be in (0, 1]")

    runtime = FullPhysicsRuntime(
        np.asarray(args.initial, dtype=np.float32),
        args.target,
        temporal_epsilon=args.temporal_epsilon,
        closure_epsilon=args.closure_epsilon,
        max_state=args.max_state,
        integration_scale=args.integration_scale,
        max_local_gain=args.max_local_gain,
        min_switch_margin=args.min_switch_margin,
        model_path=args.model.resolve(),
        module_path=args.module.resolve(),
    )

    results: list[HorizonResult] = []
    for horizon in args.horizons:
        result = runtime.forward_trajectory(horizon)
        if result.completed:
            gradient = runtime.backward(result)
            try:
                (
                    result.finite_difference_error,
                    result.finite_difference_relative,
                ) = runtime.validate_weight_gradient(
                    result,
                    gradient,
                    count=args.fd_weights,
                    epsilon=args.weight_epsilon,
                )
            finally:
                runtime.weights.copy_from_numpy(runtime.base_weights)
        results.append(result)

    print("backend: Vulkan / progressive full Carter-KKT recurrence")
    print("initial:", format_vector(runtime.initial))
    print(f"target: {float(runtime.target):.6g}")
    print(
        "integrator: "
        f"z_next=z+scale*(F(z)-z), base scale={runtime.integration_scale:g}, "
        f"max local gain={runtime.max_local_gain:g}"
    )
    for result in results:
        print(f"\nhorizon {result.horizon}")
        if not result.completed:
            terminal = (
                result.records[-1].z_next
                if result.records
                else runtime.initial
            )
            print(f"  status: STOPPED ({result.failure})")
            print(f"  valid steps: {len(result.records)}")
            print(f"  last state: {format_vector(terminal)}")
            continue
        terminal = result.records[-1].z_next
        print("  status: valid")
        print(f"  terminal: {format_vector(terminal)}")
        print(f"  raw terminal map: {format_vector(result.records[-1].raw_next)}")
        print(f"  max |z|: {np.max(np.abs(terminal)):.6g}")
        print(f"  loss: {result.loss:.9g}")
        print(
            "  max |dz_next/dz|: "
            f"{result.max_temporal_jacobian:.9g}"
        )
        print(
            "  max post-capture |dz_next/dz|: "
            f"{result.max_post_capture_jacobian:.9g}"
        )
        print(
            "  initial adjoint norm: "
            f"{result.initial_adjoint_norm:.9g}"
        )
        print(
            "  weight gradient norm: "
            f"{result.weight_gradient_norm:.9g}"
        )
        print(
            "  nonzero weight gradients: "
            f"{result.nonzero_weight_gradients}/{runtime.base_weights.size}"
        )
        print(
            "  sampled BPTT FD error: "
            f"{result.finite_difference_error:.9g}"
        )
        print(
            "  sampled BPTT FD relative: "
            f"{result.finite_difference_relative:.9g}"
        )
        print(f"  min KKT margin: {result.min_kkt_margin:.9g}")
        print(
            "  min classifier margin: "
            f"{result.min_classifier_margin:.9g}"
        )
        print(f"  min parity margin: {result.min_parity_margin:.9g}")
        print(f"  max KKT condition: {result.max_kkt_condition:.9g}")
        print(
            "  min integration scale: "
            f"{result.min_integration_scale:.9g}"
        )
        print(
            "  max Lyapunov growth: "
            f"{result.max_lyapunov_growth:.9g}"
        )
        print(
            "  max Lyapunov ratio: "
            f"{result.max_lyapunov_ratio:.9g}"
        )
        print(
            "  terminal Lyapunov: "
            f"{result.terminal_lyapunov:.9g}"
        )
        if min(
            result.min_kkt_margin,
            result.min_classifier_margin,
            result.min_parity_margin,
        ) < runtime.min_switch_margin:
            print(
                "  gradient confidence: LOW "
                "(trajectory approaches a hard switching surface)"
            )
        else:
            print("  gradient confidence: nominal")

    completed = [result for result in results if result.completed]
    if not completed:
        raise RuntimeError("No requested horizon completed successfully.")
    print("\nprogressive full-physics unroll completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
