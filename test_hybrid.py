"""Numerical smoke tests for the differentiable hybrid simulator."""

from __future__ import annotations

import torch

from hybrid_tesseract import HybridTesseract
from train_hybrid import objective


def main() -> int:
    torch.manual_seed(11)
    model = HybridTesseract(boundary_certificates=False)
    initial = torch.tensor([-0.8, -0.2, 0.2, 0.8])
    trajectory, diagnostics = model.rollout(initial, steps=3)
    loss = objective(trajectory, target_z=0.35)
    loss.backward()

    assert torch.isfinite(trajectory).all()
    assert torch.isfinite(loss)
    assert all(bool(item.kkt_valid.all()) for item in diagnostics)
    assert max(float(item.kkt_max_violation.max()) for item in diagnostics) < 2.0e-5
    assert model.c.grad is not None and torch.isfinite(model.c.grad)
    master_gradients = [
        parameter.grad
        for parameter in model.master.parameters()
        if parameter.grad is not None
    ]
    assert master_gradients
    assert all(torch.isfinite(gradient).all() for gradient in master_gradients)
    assert sum(float(gradient.norm()) for gradient in master_gradients) > 0.0

    # Check one local recurrence derivative against a central difference while
    # staying away from a known hard switch. Parameters are held fixed.
    probe_value = 0.60
    probe = torch.tensor([probe_value], requires_grad=True)
    output, _ = model.step(probe)
    analytic = torch.autograd.grad(output.sum(), probe)[0]
    epsilon = 1.0e-5
    plus, _ = model.step(torch.tensor([probe_value + epsilon]))
    minus, _ = model.step(torch.tensor([probe_value - epsilon]))
    finite_difference = (plus.detach() - minus.detach()) / (2.0 * epsilon)
    error = float((analytic - finite_difference).abs())
    assert error < 2.0e-4, (analytic, finite_difference, error)

    print(f"loss={float(loss.detach()):.8f}")
    print(
        f"max_kkt_violation="
        f"{max(float(item.kkt_max_violation.max()) for item in diagnostics):.3g}"
    )
    print(
        f"dz_next/dz analytic={float(analytic):.8f}"
        f" finite_difference={float(finite_difference):.8f}"
        f" error={error:.3g}"
    )
    print("hybrid differentiability smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
