#!/usr/bin/env python3
"""Export a HybridTesseract checkpoint to the portable Vulkan model format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


FORMAT_NAME = "carter-tesseract-hybrid"
FORMAT_VERSION = 1


def ordered_master_weights(state: dict[str, torch.Tensor]) -> tuple[int, list[float]]:
    """Flatten weights in the exact order consumed by the GLSL kernel."""
    width = int(state["master.residual.0.bias"].numel())
    if width != 32:
        raise ValueError(
            f"The Vulkan shader currently requires master width 32, got {width}."
        )
    tensors = [
        state["master.linear.weight"],
        state["master.linear.bias"],
        state["master.residual.0.weight"],
        state["master.residual.0.bias"],
        state["master.residual.2.weight"],
        state["master.residual.2.bias"],
        state["master.residual.4.weight"],
        state["master.residual.4.bias"],
    ]
    flattened = torch.cat([tensor.detach().cpu().reshape(-1) for tensor in tensors])
    expected = 1287
    if flattened.numel() != expected:
        raise ValueError(f"Expected {expected} master weights, got {flattened.numel()}.")
    return width, [float(value) for value in flattened]


def export_checkpoint(checkpoint_path: Path, output_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["model_state"]
    width, weights = ordered_master_weights(state)
    document = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "master_width": width,
        "master_weights": weights,
        "dynamics": {
            name: float(state[name])
            for name in ("c", "kappa", "alpha", "zeta", "omega_delta")
        },
        "target_t": [
            [float(value) for value in row] for row in state["target_t"]
        ],
        "training": {
            key: checkpoint.get(key)
            for key in ("steps", "target_z", "initial_loss", "final_loss")
        },
    }
    if checkpoint.get("conditioning") is not None:
        document["conditioning"] = checkpoint["conditioning"]
    output_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent / "hybrid_tesseract.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "hybrid_model.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    document = export_checkpoint(args.checkpoint, args.output)
    print(
        f"exported {len(document['master_weights'])} neural weights "
        f"to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
