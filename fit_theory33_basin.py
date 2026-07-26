#!/usr/bin/env python3
"""Regenerate and verify the Theory 3.3 broad-basin kappa controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor

from theory33_hybrid import (
    BroadBasinKappaController,
    Theory33HybridTesseract,
)


TARGET = 0.1019493853295916


def chebyshev_basis(value: Tensor, degree: int) -> Tensor:
    terms = [torch.ones_like(value), value]
    for _ in range(2, degree + 1):
        terms.append(2.0 * value * terms[-1] - terms[-2])
    return torch.stack(terms[: degree + 1], dim=-1)


def fit_interval(
    model: Theory33HybridTesseract,
    interval: tuple[float, float],
    *,
    points: int,
    degree: int,
) -> tuple[Tensor, float]:
    lower, upper = interval
    z = torch.linspace(lower, upper, points).requires_grad_(True)
    sigma = torch.where(
        z.detach() >= 0.0, torch.ones_like(z), -torch.ones_like(z)
    )
    state = model.evaluate(z, sigma)
    d_source = torch.autograd.grad(
        state.j_limited.sum(), z, retain_graph=True
    )[0]
    d_free_energy = torch.autograd.grad(state.free_energy.sum(), z)[0]
    u = 1.5 * (z + model.c)
    normalization = torch.sqrt(
        z.square() + model.c.square() + model.epsilon_n
    )
    base = u.square() + model.c
    required_kappa = (
        normalization * (TARGET - base)
        + model.alpha * d_source
        - model.chi * state.residual
        + model.zeta * d_free_energy
    ) / u
    required_residual = (required_kappa - model.kappa).detach()
    coordinate = 2.0 * (z.detach() - lower) / (upper - lower) - 1.0
    basis = chebyshev_basis(coordinate, degree)
    coefficients = torch.linalg.lstsq(
        basis, required_residual
    ).solution
    fitted_kappa = model.kappa.detach() + basis @ coefficients
    mapped = (
        base.detach()
        + (1.0 / normalization.detach())
        * (
            fitted_kappa * u.detach()
            - model.alpha.detach() * d_source.detach()
            + model.chi * state.residual.detach()
            - model.zeta.detach() * d_free_energy.detach()
        )
    )
    return coefficients, float((mapped - TARGET).abs().max())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=4097)
    parser.add_argument("--degree", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.points < args.degree + 1:
        parser.error("--points must exceed the number of coefficients")
    if args.degree != 5:
        parser.error("the shipped controller currently has degree five")
    return args


def main() -> int:
    args = parse_args()
    model = Theory33HybridTesseract(
        learn_dynamics=False,
        broad_basin_conditioning=False,
        boundary_certificates=False,
    )
    controller = model.kappa_controller
    negative, negative_error = fit_interval(
        model,
        BroadBasinKappaController.negative_interval,
        points=args.points,
        degree=args.degree,
    )
    positive, positive_error = fit_interval(
        model,
        BroadBasinKappaController.positive_interval,
        points=args.points,
        degree=args.degree,
    )
    coefficient_error = max(
        float(
            (
                negative - controller.negative_coefficients.detach()
            ).abs().max()
        ),
        float(
            (
                positive - controller.positive_coefficients.detach()
            ).abs().max()
        ),
    )
    result = {
        "format": "theory33-broad-basin-kappa",
        "version": 1,
        "degree": args.degree,
        "points_per_interval": args.points,
        "target": TARGET,
        "negative_interval": list(
            BroadBasinKappaController.negative_interval
        ),
        "positive_interval": list(
            BroadBasinKappaController.positive_interval
        ),
        "negative_residual_coefficients": [
            float(value) for value in negative
        ],
        "positive_residual_coefficients": [
            float(value) for value in positive
        ],
        "max_coefficient_error_from_shipped": coefficient_error,
        "max_one_step_map_error": max(negative_error, positive_error),
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if coefficient_error > 1.0e-10:
        raise RuntimeError("regenerated coefficients differ from the source")
    if max(negative_error, positive_error) > 2.0e-5:
        raise RuntimeError("controller projection error is too large")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
