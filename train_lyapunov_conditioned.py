#!/usr/bin/env python3
"""Condition the literal Carter/KKT map around an intrinsic stable attractor.

The scale-1 recurrence is used throughout.  A fixed c=0.05 gives the
parity-zero core a stable nonzero fixed point.  Training learns only the
parity-one capture dynamics and Carter master so the requested initial states
enter that basin while preserving hard KKT and switching decisions.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch import Tensor

from export_hybrid import export_checkpoint
from hybrid_tesseract import HybridTesseract


HERE = Path(__file__).resolve().parent


def stable_fixed_point(c: float) -> float:
    discriminant = 1.0 - 18.0 * c
    if discriminant <= 0.0:
        raise ValueError(f"c={c} has no real parity-zero fixed point")
    return (1.0 - 4.5 * c - math.sqrt(discriminant)) / 4.5


def lyapunov_objective(
    trajectory: Tensor,
    diagnostics,
    attractor: float,
    *,
    contraction: float,
    margin_floor: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    error = trajectory - attractor
    value = error.square()
    scale2 = 0.25**2
    regime = torch.log1p(value[1:] / scale2).mean()
    growth = torch.relu(
        value[1:] - contraction * value[:-1]
    ).mean() / scale2
    margins = torch.stack(
        [item.switch_margin for item in diagnostics], dim=0
    )
    margin_loss = (
        torch.relu(margin_floor - margins).square().mean()
        / (margin_floor * margin_floor)
    )
    total = regime + 2.0 * growth + 0.15 * margin_loss
    return total, {
        "regime": regime,
        "growth": growth,
        "margin": margin_loss,
    }


def local_gain_penalty(
    model: HybridTesseract,
    states: Tensor,
    *,
    gain_limit: float,
) -> tuple[Tensor, Tensor]:
    probe = states.detach().clone().requires_grad_(True)
    _next_state, diagnostics = model.step(
        probe,
        detach_diagnostics=False,
        measure_local_gain=True,
    )
    assert diagnostics.local_gain is not None
    gain = diagnostics.local_gain.abs()
    penalty = torch.relu(gain - gain_limit).square().mean()
    return penalty, gain


def clamp_dynamics(model: HybridTesseract) -> None:
    with torch.no_grad():
        model.kappa.clamp_(-2.0, 2.0)
        model.alpha.clamp_(-2.0, 2.0)
        model.zeta.clamp_(-2.0, 2.0)
        model.omega_delta.clamp_(-2.0, 2.0)


def optimize_stage(
    model: HybridTesseract,
    optimizer: torch.optim.Optimizer,
    initial: Tensor,
    *,
    horizon: int,
    epochs: int,
    attractor: float,
    contraction: float,
    margin_floor: float,
    gain_limit: float,
    label: str,
) -> None:
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        trajectory, diagnostics = model.rollout(
            initial,
            horizon,
            detach_diagnostics=False,
        )
        loss, terms = lyapunov_objective(
            trajectory,
            diagnostics,
            attractor,
            contraction=contraction,
            margin_floor=margin_floor,
        )
        gain_states = torch.cat(
            [initial, trajectory[-1].detach()], dim=0
        )
        gain_loss, gain = local_gain_penalty(
            model, gain_states, gain_limit=gain_limit
        )
        loss = loss + 0.05 * gain_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"{label} produced a non-finite loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters()
             if parameter.requires_grad],
            max_norm=5.0,
        )
        optimizer.step()
        clamp_dynamics(model)

        report_every = max(1, epochs // 5)
        if epoch == 1 or epoch == epochs or epoch % report_every == 0:
            minimum_margin = min(
                float(item.switch_margin.detach().min())
                for item in diagnostics
            )
            print(
                f"{label:>10s} {epoch:4d}/{epochs}"
                f" loss={float(loss.detach()):.7g}"
                f" regime={float(terms['regime'].detach()):.4g}"
                f" growth={float(terms['growth'].detach()):.4g}"
                f" margin={minimum_margin:.4g}"
                f" gain={float(gain.detach().max()):.4g}"
                f" grad={float(gradient_norm):.4g}"
            )


def validate_literal(
    model: HybridTesseract,
    initial: Tensor,
    *,
    steps: int,
    attractor: float,
) -> dict[str, float | bool | list[float]]:
    z = initial.detach().clone()
    values = [(z - attractor).square()]
    minimum_kkt_margin = math.inf
    minimum_classifier_margin = math.inf
    minimum_parity_margin = math.inf
    maximum_condition = 0.0
    maximum_gain = 0.0
    maximum_abs_state = float(z.abs().max())
    all_valid = True

    for _ in range(steps):
        z = z.detach().requires_grad_(True)
        z_next, diagnostics = model.step(
            z,
            measure_local_gain=True,
        )
        assert diagnostics.local_gain is not None
        all_valid &= bool(diagnostics.kkt_valid.all())
        minimum_kkt_margin = min(
            minimum_kkt_margin,
            float(diagnostics.kkt_feasibility_margin.min()),
        )
        minimum_classifier_margin = min(
            minimum_classifier_margin,
            float(diagnostics.classifier_margin.min()),
        )
        minimum_parity_margin = min(
            minimum_parity_margin,
            float(diagnostics.parity_margin.min()),
        )
        maximum_condition = max(
            maximum_condition,
            float(diagnostics.kkt_condition_number.max()),
        )
        maximum_gain = max(
            maximum_gain, float(diagnostics.local_gain.abs().max())
        )
        z = z_next.detach()
        maximum_abs_state = max(maximum_abs_state, float(z.abs().max()))
        values.append((z - attractor).square())

    value = torch.stack(values)
    growth = value[1:] - value[:-1]
    return {
        "steps": steps,
        "kkt_valid": all_valid,
        "terminal": [float(item) for item in z],
        "max_abs_state": maximum_abs_state,
        "max_local_gain": maximum_gain,
        "min_kkt_margin": minimum_kkt_margin,
        "min_classifier_margin": minimum_classifier_margin,
        "min_parity_margin": minimum_parity_margin,
        "max_kkt_condition": maximum_condition,
        "max_lyapunov_growth": float(growth.max()),
        "terminal_lyapunov": float(value[-1].max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        default=HERE / "hybrid_tesseract.pt",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=HERE / "hybrid_tesseract_conditioned.pt",
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=HERE / "hybrid_model_conditioned.json",
    )
    parser.add_argument("--capture-epochs", type=int, default=400)
    parser.add_argument("--curriculum-epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--refine-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--validation-steps", type=int, default=128)
    parser.add_argument("--margin-floor", type=float, default=0.02)
    parser.add_argument("--gain-limit", type=float, default=1.0)
    parser.add_argument("--contraction", type=float, default=0.9)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = torch.load(
        args.base, map_location="cpu", weights_only=True
    )
    model = HybridTesseract(
        master_width=32,
        learn_dynamics=True,
        boundary_certificates=True,
    )
    model.load_state_dict(checkpoint["model_state"])

    fixed_c = 0.05
    with torch.no_grad():
        model.c.fill_(fixed_c)
    model.c.requires_grad_(False)
    attractor = stable_fixed_point(fixed_c)
    local_attractor_gain = abs(4.5 * (attractor + fixed_c))
    print(
        f"intrinsic attractor={attractor:.12g}"
        f" core_gain={local_attractor_gain:.9g}"
    )

    trainable = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    capture_optimizer = torch.optim.Adam(
        trainable, lr=args.learning_rate
    )
    capture = torch.tensor(
        [-0.95, -0.85, -0.80, -0.75, 0.65, 0.75, 0.80, 0.90, 0.95]
    )
    optimize_stage(
        model,
        capture_optimizer,
        capture,
        horizon=2,
        epochs=args.capture_epochs,
        attractor=attractor,
        contraction=args.contraction,
        margin_floor=args.margin_floor,
        gain_limit=args.gain_limit,
        label="capture",
    )

    refine_optimizer = torch.optim.Adam(
        trainable, lr=args.refine_learning_rate
    )
    robust_initial = torch.tensor(
        [
            -0.82, -0.80, -0.78,
            -0.22, -0.20, -0.18,
             0.18,  0.20,  0.22,
             0.78,  0.80,  0.82,
        ]
    )
    for horizon in (4, 8, 16, 32):
        optimize_stage(
            model,
            refine_optimizer,
            robust_initial,
            horizon=horizon,
            epochs=args.curriculum_epochs,
            attractor=attractor,
            contraction=args.contraction,
            margin_floor=args.margin_floor,
            gain_limit=args.gain_limit,
            label=f"h={horizon}",
        )

    validation_initial = torch.tensor([-0.8, -0.2, 0.2, 0.8])
    metrics = validate_literal(
        model,
        validation_initial,
        steps=args.validation_steps,
        attractor=attractor,
    )
    if not metrics["kkt_valid"]:
        raise RuntimeError("conditioned literal rollout lost KKT validity")
    if not math.isfinite(float(metrics["terminal_lyapunov"])):
        raise RuntimeError("conditioned literal rollout is non-finite")

    payload = {
        "model_state": model.state_dict(),
        "initial_z": validation_initial,
        "steps": args.validation_steps,
        "target_z": attractor,
        "initial_loss": None,
        "final_loss": float(metrics["terminal_lyapunov"]),
        "conditioning": metrics,
    }
    torch.save(payload, args.save)
    export_checkpoint(args.save, args.export)
    print("literal validation:", metrics)
    print(
        "learned dynamics:"
        f" c={float(model.c.detach()):.7g}"
        f" kappa={float(model.kappa.detach()):.7g}"
        f" alpha={float(model.alpha.detach()):.7g}"
        f" zeta={float(model.zeta.detach()):.7g}"
        f" omega_delta={float(model.omega_delta.detach()):.7g}"
    )
    print(f"saved: {args.save}")
    print(f"exported: {args.export}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
