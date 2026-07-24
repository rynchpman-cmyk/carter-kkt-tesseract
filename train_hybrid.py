#!/usr/bin/env python3
"""Train the learned Carter master function through an unrolled recurrence."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from export_hybrid import export_checkpoint
from hybrid_tesseract import HybridTesseract


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--target-z", type=float, default=0.35)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument(
        "--learn-target",
        action="store_true",
        help="also optimize the target stress tensor",
    )
    parser.add_argument(
        "--no-boundary-certificates",
        action="store_true",
        help="disable the hard iota boundary test for faster experiments",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=HERE / "hybrid_tesseract.pt",
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=HERE / "hybrid_model.json",
        help="portable model written for the Vulkan runner",
    )
    return parser.parse_args()


def objective(trajectory: torch.Tensor, target_z: float) -> torch.Tensor:
    # Train on the latter half of the horizon while mildly discouraging
    # violent transitions. This remains a genuine long-horizon BPTT loss.
    start = max(1, trajectory.shape[0] // 2)
    regime_loss = torch.log1p((trajectory[start:] - target_z).square()).mean()
    transition_loss = torch.log1p(
        (trajectory[1:] - trajectory[:-1]).square()
    ).mean()
    magnitude_loss = torch.log1p(trajectory.square()).mean()
    return regime_loss + 2.0e-3 * transition_loss + 1.0e-4 * magnitude_loss


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.steps < 1:
        raise SystemExit("--epochs and --steps must be positive")

    torch.manual_seed(7)
    model = HybridTesseract(
        master_width=args.width,
        learn_dynamics=True,
        learn_target=args.learn_target,
        boundary_certificates=not args.no_boundary_certificates,
    )
    initial_z = torch.tensor([-1.0, -0.45, -0.10, 0.10, 0.45, 1.0])
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)

    initial_loss = None
    last_loss = None
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        trajectory, diagnostics = model.rollout(initial_z, args.steps)
        loss = objective(trajectory, args.target_z)
        if not torch.isfinite(loss):
            raise RuntimeError("Training produced a non-finite loss.")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=10.0)
        optimizer.step()

        if initial_loss is None:
            initial_loss = float(loss.detach())
        last_loss = float(loss.detach())
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 5) == 0:
            valid = torch.stack([item.kkt_valid for item in diagnostics]).all()
            violation = max(
                float(item.kkt_max_violation.max()) for item in diagnostics
            )
            print(
                f"epoch {epoch:4d}  loss={last_loss:.8f}"
                f"  grad={float(gradient_norm):.4g}"
                f"  kkt_valid={bool(valid)}  max_violation={violation:.3g}"
            )

    assert initial_loss is not None and last_loss is not None
    final_trajectory, final_diagnostics = model.rollout(initial_z, args.steps)
    if not torch.isfinite(final_trajectory).all():
        raise RuntimeError("Final trajectory contains non-finite values.")
    if not all(bool(item.kkt_valid.all()) for item in final_diagnostics):
        raise RuntimeError("At least one hard KKT certificate is invalid.")

    payload = {
        "model_state": model.state_dict(),
        "initial_z": initial_z,
        "steps": args.steps,
        "target_z": args.target_z,
        "initial_loss": initial_loss,
        "final_loss": last_loss,
    }
    torch.save(payload, args.save)
    exported = export_checkpoint(args.save, args.export)
    print(f"loss change: {initial_loss:.8f} -> {last_loss:.8f}")
    print(
        "learned dynamics:"
        f" c={float(model.c.detach()):.6f}"
        f" kappa={float(model.kappa.detach()):.6f}"
        f" alpha={float(model.alpha.detach()):.6f}"
        f" zeta={float(model.zeta.detach()):.6f}"
        f" omega_delta={float(model.omega_delta.detach()):.6f}"
    )
    print("terminal z:", " ".join(f"{value:.5f}" for value in final_trajectory[-1]))
    print(f"saved: {args.save}")
    print(
        f"exported: {args.export} "
        f"({len(exported['master_weights'])} neural weights)"
    )

    trainable_with_grad = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not any(name.startswith("master.") for name in trainable_with_grad):
        raise RuntimeError("The learned master function received no gradient.")
    if not math.isfinite(last_loss):
        raise RuntimeError("Final loss is not finite.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
