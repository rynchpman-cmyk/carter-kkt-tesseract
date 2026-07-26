#!/usr/bin/env python3
"""Run the standalone physical Theory33HybridTesseract validation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from theory33_hybrid import Theory33HybridTesseract


DEFAULT_INITIAL = (-0.8, -0.2, 0.2, 0.8)
DEFAULT_TARGET = 0.1019493853295916


def validate_global_safety(
    model: Theory33HybridTesseract,
    *,
    steps: int,
    target: float,
    exponent_points: int,
) -> dict[str, object]:
    """Exercise the constructed absorbing map up to float64 scale."""
    if exponent_points < 2:
        raise ValueError("global validation needs at least two exponents")
    magnitudes = torch.logspace(
        1.0e-6, 300.0, exponent_points, dtype=torch.float64
    )
    initial = torch.cat([-magnitudes.flip(0), magnitudes])
    probe = initial.clone().requires_grad_(True)

    boundary_certificates = model.boundary_certificates
    model.boundary_certificates = False
    try:
        z, first_diagnostics = model.step(
            probe,
            detach_diagnostics=False,
            measure_local_gain=True,
        )
        outer_gradient = torch.autograd.grad(z.sum(), probe)[0]
        z = z.detach()
        kkt_valid = bool(first_diagnostics.kkt_valid.all())
        physical_valid = bool(first_diagnostics.physical_valid.all())
        finite = bool(torch.isfinite(z).all())
        maximum_abs_after_projection = float(z.abs().max())
        all_outer_gates_fired = bool(
            first_diagnostics.outer_safety_gate.all()
        )
        maximum_projected_input = float(
            first_diagnostics.projected_input.detach().abs().max()
        )
        maximum_outer_gradient = float(outer_gradient.abs().max())

        for _ in range(1, steps):
            step_probe = z.detach().clone().requires_grad_(True)
            z, diagnostics = model.step(step_probe)
            z = z.detach()
            finite &= bool(torch.isfinite(z).all())
            kkt_valid &= bool(diagnostics.kkt_valid.all())
            physical_valid &= bool(diagnostics.physical_valid.all())
            maximum_abs_after_projection = max(
                maximum_abs_after_projection, float(z.abs().max())
            )
    finally:
        model.boundary_certificates = boundary_certificates

    return {
        "domain": "all finite float64 inputs",
        "samples": int(initial.numel()),
        "maximum_input_magnitude": float(initial.abs().max()),
        "steps": steps,
        "all_outer_gates_fired": all_outer_gates_fired,
        "finite": finite,
        "kkt_valid": kkt_valid,
        "physical_valid": physical_valid,
        "maximum_projected_input": maximum_projected_input,
        "maximum_abs_after_projection": maximum_abs_after_projection,
        "maximum_outer_gradient": maximum_outer_gradient,
        "terminal_max_error": float((z - target).abs().max()),
    }


def validate_broad_basin(
    model: Theory33HybridTesseract,
    *,
    steps: int,
    target: float,
    points: int,
    characteristic_points: int,
) -> dict[str, object]:
    """Qualify the compact basin ``z_0 in [-1, 1]``."""
    if points < 2:
        raise ValueError("broad-basin validation needs at least two points")
    initial = torch.linspace(-1.0, 1.0, points)
    z = initial.clone()
    previous_lyapunov = (z - target).square()
    maximum_abs_state = float(z.abs().max())
    maximum_lyapunov_growth = -math.inf
    minimum_kkt_margin = math.inf
    minimum_physical_margin = math.inf
    kkt_valid = True
    physical_valid = True

    # omega_delta is exactly zero in the qualified model, so boundary-crossing
    # certificates cannot alter the recurrence value. Disabling their
    # bisection here makes the dense audit tractable without changing the map.
    if float(model.omega_delta.detach()) != 0.0:
        raise ValueError("dense basin audit requires omega_delta == 0")
    boundary_certificates = model.boundary_certificates
    model.boundary_certificates = False
    try:
        for _ in range(steps):
            probe = z.detach().clone().requires_grad_(True)
            z, diagnostics = model.step(probe)
            z = z.detach()
            lyapunov = (z - target).square()
            maximum_abs_state = max(
                maximum_abs_state, float(z.abs().max())
            )
            maximum_lyapunov_growth = max(
                maximum_lyapunov_growth,
                float((lyapunov - previous_lyapunov).max()),
            )
            minimum_kkt_margin = min(
                minimum_kkt_margin,
                float(diagnostics.kkt_feasibility_margin.min()),
            )
            minimum_physical_margin = min(
                minimum_physical_margin,
                float(
                    torch.stack(
                        [
                            diagnostics.legendre_margin,
                            diagnostics.thermodynamic_margin,
                            diagnostics.causal_screen_margin,
                            diagnostics.timelike_margin,
                            diagnostics.metric_signature_margin,
                        ],
                        dim=-1,
                    ).min()
                ),
            )
            kkt_valid &= bool(diagnostics.kkt_valid.all())
            physical_valid &= bool(diagnostics.physical_valid.all())
            previous_lyapunov = lyapunov
    finally:
        model.boundary_certificates = boundary_certificates

    characteristic_valid = True
    maximum_characteristic_speed = 0.0
    if characteristic_points > 0:
        characteristic_grid = torch.linspace(
            -1.0, max(1.05, maximum_abs_state), characteristic_points
        )
        for sigma_value in (-1.0, 1.0):
            probe = characteristic_grid.clone().requires_grad_(True)
            sigma = torch.full_like(probe, sigma_value)
            state = model.evaluate(probe, sigma)
            for audit in model.characteristic_audits(state):
                characteristic_valid &= audit.causal
                maximum_characteristic_speed = max(
                    maximum_characteristic_speed,
                    audit.maximum_absolute_speed,
                )

    terminal_error = (z - target).abs()
    return {
        "interval": [-1.0, 1.0],
        "points": points,
        "steps": steps,
        "kkt_valid": kkt_valid,
        "physical_valid": physical_valid,
        "characteristic_valid": characteristic_valid,
        "max_abs_state": maximum_abs_state,
        "max_lyapunov_growth": maximum_lyapunov_growth,
        "min_kkt_margin": minimum_kkt_margin,
        "min_physical_margin": minimum_physical_margin,
        "max_characteristic_speed": maximum_characteristic_speed,
        "terminal_max_error": float(terminal_error.max()),
        "converged_fraction": float(
            (terminal_error < 1.0e-10).to(torch.float64).mean()
        ),
    }


def parse_initial(value: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in value.split(","))
    if not values:
        raise argparse.ArgumentTypeError("initial state list cannot be empty")
    if not all(math.isfinite(item) for item in values):
        raise argparse.ArgumentTypeError("initial states must be finite")
    return values


def validate(
    *,
    steps: int,
    initial: tuple[float, ...],
    target: float,
    characteristic_stride: int,
    basin_points: int,
    basin_characteristic_points: int,
    global_exponent_points: int,
) -> dict[str, object]:
    model = Theory33HybridTesseract(
        learn_dynamics=False,
        boundary_certificates=True,
    )
    model.eval()
    z = torch.tensor(initial)
    lyapunov = (z - target).square()
    metrics: dict[str, object] = {
        "mode": "theory33-hybrid",
        "steps": steps,
        "initial_z": list(initial),
        "target_z": target,
        "kkt_valid": True,
        "physical_valid": True,
        "characteristic_valid": True,
        "max_abs_state": float(z.abs().max()),
        "max_lyapunov_growth": -math.inf,
        "min_kkt_margin": math.inf,
        "min_classifier_margin": math.inf,
        "min_parity_margin": math.inf,
        "min_legendre_margin": math.inf,
        "min_thermodynamic_margin": math.inf,
        "min_causal_screen_margin": math.inf,
        "min_timelike_margin": math.inf,
        "min_metric_signature_margin": math.inf,
        "max_relative_lorentz_factor": 1.0,
        "max_current_normalization_error": 0.0,
        "max_characteristic_speed": 0.0,
        "max_local_gain": 0.0,
    }
    for index in range(steps):
        audit = (
            characteristic_stride > 0
            and (
                index % characteristic_stride == 0
                or index == steps - 1
            )
        )
        probe = z.detach().clone().requires_grad_(True)
        z, diagnostics = model.step(
            probe,
            measure_local_gain=True,
            audit_characteristics=audit,
        )
        z = z.detach()
        next_lyapunov = (z - target).square()
        metrics["kkt_valid"] = bool(metrics["kkt_valid"]) and bool(
            diagnostics.kkt_valid.all()
        )
        metrics["physical_valid"] = bool(metrics["physical_valid"]) and bool(
            diagnostics.physical_valid.all()
        )
        metrics["max_abs_state"] = max(
            float(metrics["max_abs_state"]), float(z.abs().max())
        )
        metrics["max_lyapunov_growth"] = max(
            float(metrics["max_lyapunov_growth"]),
            float((next_lyapunov - lyapunov).max()),
        )
        metrics["min_kkt_margin"] = min(
            float(metrics["min_kkt_margin"]),
            float(diagnostics.kkt_feasibility_margin.min()),
        )
        metrics["min_classifier_margin"] = min(
            float(metrics["min_classifier_margin"]),
            float(diagnostics.classifier_margin.min()),
        )
        metrics["min_parity_margin"] = min(
            float(metrics["min_parity_margin"]),
            float(diagnostics.parity_margin.min()),
        )
        metrics["min_legendre_margin"] = min(
            float(metrics["min_legendre_margin"]),
            float(diagnostics.legendre_margin.min()),
        )
        metrics["min_thermodynamic_margin"] = min(
            float(metrics["min_thermodynamic_margin"]),
            float(diagnostics.thermodynamic_margin.min()),
        )
        metrics["min_causal_screen_margin"] = min(
            float(metrics["min_causal_screen_margin"]),
            float(diagnostics.causal_screen_margin.min()),
        )
        metrics["min_timelike_margin"] = min(
            float(metrics["min_timelike_margin"]),
            float(diagnostics.timelike_margin.min()),
        )
        metrics["min_metric_signature_margin"] = min(
            float(metrics["min_metric_signature_margin"]),
            float(diagnostics.metric_signature_margin.min()),
        )
        metrics["max_relative_lorentz_factor"] = max(
            float(metrics["max_relative_lorentz_factor"]),
            float(diagnostics.maximum_relative_lorentz_factor.max()),
        )
        metrics["max_current_normalization_error"] = max(
            float(metrics["max_current_normalization_error"]),
            float(diagnostics.current_normalization_error.max()),
        )
        if diagnostics.local_gain is not None:
            metrics["max_local_gain"] = max(
                float(metrics["max_local_gain"]),
                float(diagnostics.local_gain.abs().max()),
            )
        if diagnostics.characteristic_valid is not None:
            metrics["characteristic_valid"] = bool(
                metrics["characteristic_valid"]
            ) and bool(diagnostics.characteristic_valid.all())
            assert diagnostics.maximum_characteristic_speed is not None
            metrics["max_characteristic_speed"] = max(
                float(metrics["max_characteristic_speed"]),
                float(diagnostics.maximum_characteristic_speed.max()),
            )
        lyapunov = next_lyapunov
    metrics["terminal_z"] = [float(value) for value in z]
    metrics["terminal_lyapunov"] = float(lyapunov.max())
    if basin_points > 0:
        metrics["broad_basin"] = validate_broad_basin(
            model,
            steps=steps,
            target=target,
            points=basin_points,
            characteristic_points=basin_characteristic_points,
        )
    if global_exponent_points > 0:
        metrics["global_safety"] = validate_global_safety(
            model,
            steps=steps,
            target=target,
            exponent_points=global_exponent_points,
        )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument(
        "--initial",
        type=parse_initial,
        default=DEFAULT_INITIAL,
        help="comma-separated scalar initial states",
    )
    parser.add_argument("--target", type=float, default=DEFAULT_TARGET)
    parser.add_argument(
        "--characteristic-stride",
        type=int,
        default=16,
        help="run the full principal-symbol audit every N steps; 0 disables",
    )
    parser.add_argument(
        "--basin-points",
        type=int,
        default=1025,
        help="dense [-1,1] initialization grid; 0 disables",
    )
    parser.add_argument(
        "--basin-characteristic-points",
        type=int,
        default=33,
        help="states per branch in the broad-domain characteristic audit",
    )
    parser.add_argument(
        "--global-exponent-points",
        type=int,
        default=151,
        help="log-spaced positive magnitudes (plus negatives); 0 disables",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.characteristic_stride < 0:
        parser.error("--characteristic-stride cannot be negative")
    if args.basin_points != 0 and args.basin_points < 2:
        parser.error("--basin-points must be zero or at least two")
    if args.basin_characteristic_points < 0:
        parser.error("--basin-characteristic-points cannot be negative")
    if args.global_exponent_points != 0 and args.global_exponent_points < 2:
        parser.error("--global-exponent-points must be zero or at least two")
    return args


def main() -> int:
    args = parse_args()
    metrics = validate(
        steps=args.steps,
        initial=args.initial,
        target=args.target,
        characteristic_stride=args.characteristic_stride,
        basin_points=args.basin_points,
        basin_characteristic_points=args.basin_characteristic_points,
        global_exponent_points=args.global_exponent_points,
    )
    rendered = json.dumps(metrics, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not bool(metrics["kkt_valid"]):
        raise RuntimeError("Theory 3.3 rollout lost KKT validity")
    if not bool(metrics["physical_valid"]):
        raise RuntimeError("Theory 3.3 rollout left its physical domain")
    if not bool(metrics["characteristic_valid"]):
        raise RuntimeError("Theory 3.3 characteristic audit failed")
    basin = metrics.get("broad_basin")
    if isinstance(basin, dict):
        if not bool(basin["kkt_valid"]):
            raise RuntimeError("broad-basin rollout lost KKT validity")
        if not bool(basin["physical_valid"]):
            raise RuntimeError("broad-basin rollout left the physical domain")
        if not bool(basin["characteristic_valid"]):
            raise RuntimeError("broad-basin characteristic audit failed")
        if float(basin["max_abs_state"]) > 1.051:
            raise RuntimeError("broad-basin rollout escaped its enclosure")
        if float(basin["converged_fraction"]) < 0.999:
            raise RuntimeError("broad-basin convergence target was missed")
    global_safety = metrics.get("global_safety")
    if isinstance(global_safety, dict):
        if not bool(global_safety["all_outer_gates_fired"]):
            raise RuntimeError("an outer safety gate failed to activate")
        if not bool(global_safety["finite"]):
            raise RuntimeError("global safety rollout became non-finite")
        if not bool(global_safety["kkt_valid"]):
            raise RuntimeError("global safety rollout lost KKT validity")
        if not bool(global_safety["physical_valid"]):
            raise RuntimeError("global safety rollout became non-physical")
        if float(global_safety["maximum_projected_input"]) > 1.0:
            raise RuntimeError("global projection exceeded its inner radius")
        if float(global_safety["maximum_abs_after_projection"]) > 1.051:
            raise RuntimeError("global rollout escaped the absorbing enclosure")
        if float(global_safety["maximum_outer_gradient"]) != 0.0:
            raise RuntimeError("outer projection sensitivity is nonzero")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
