#!/usr/bin/env python3
"""Verify learned-model parity between PyTorch and the Vulkan shader."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import torch

from hybrid_tesseract import HybridTesseract


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=HERE / "hybrid_tesseract.pt")
    parser.add_argument("--model", type=Path, default=HERE / "hybrid_model.json")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument(
        "values",
        type=float,
        nargs="*",
        default=[-0.8, -0.2, 0.2, 0.8],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint["model_state"]
    width = int(state["master.residual.0.bias"].numel())
    model = HybridTesseract(
        master_width=width,
        learn_target=True,
        boundary_certificates=True,
    )
    model.load_state_dict(state)
    inputs = torch.tensor(args.values)
    trajectory, torch_diagnostics = model.rollout(inputs, args.steps)
    torch_outputs = trajectory[-1].detach()

    validator = HERE / ".tools" / "glslang" / "bin" / "glslangValidator.exe"
    venv_python = HERE / ".venv" / "Scripts" / "python.exe"
    if not validator.exists() or not venv_python.exists():
        raise SystemExit("Run .\\run.ps1 once to install the Vulkan runner tools.")
    subprocess.run(
        [
            str(validator),
            "-V",
            "--target-env",
            "vulkan1.2",
            "-o",
            str(HERE / "carter_tesseract_kkt.spv"),
            str(HERE / "carter_tesseract_kkt.comp"),
        ],
        check=True,
    )
    with tempfile.TemporaryDirectory(prefix="tesseract-parity-") as temporary:
        result_path = Path(temporary) / "result.json"
        subprocess.run(
            [
                str(venv_python),
                str(HERE / "run_tesseract.py"),
                "--model",
                str(args.model),
                "--steps",
                str(args.steps),
                "--json-output",
                str(result_path),
                *[str(value) for value in args.values],
            ],
            check=True,
        )
        vulkan_result = json.loads(result_path.read_text(encoding="utf-8"))

    vulkan_outputs = torch.tensor(vulkan_result["outputs"])
    absolute_error = (torch_outputs - vulkan_outputs).abs()
    relative_error = absolute_error / torch_outputs.abs().clamp_min(1.0)
    max_absolute = float(absolute_error.max())
    max_relative = float(relative_error.max())
    vulkan_debug = vulkan_result["debug"]
    final_torch_diagnostics = torch_diagnostics[-1]
    vulkan_active = torch.tensor([record["masks"][0] for record in vulkan_debug])
    vulkan_classifier = torch.tensor(
        [record["masks"][1] for record in vulkan_debug]
    )
    if not torch.equal(vulkan_active, final_torch_diagnostics.active_mask):
        raise RuntimeError("PyTorch and Vulkan selected different KKT active sets.")
    if not torch.equal(vulkan_classifier, final_torch_diagnostics.classifier_mask):
        raise RuntimeError("PyTorch and Vulkan selected different classifier masks.")
    if max_absolute > 2.0e-4 and max_relative > 2.0e-5:
        raise RuntimeError(
            f"Backend output mismatch: abs={max_absolute}, rel={max_relative}"
        )

    print("PyTorch:", " ".join(f"{value:.9g}" for value in torch_outputs))
    print("Vulkan: ", " ".join(f"{value:.9g}" for value in vulkan_outputs))
    print(f"max absolute error: {max_absolute:.3g}")
    print(f"max relative error: {max_relative:.3g}")
    print("active-set and classifier masks match")
    print("backend parity check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
