#!/usr/bin/env python3
"""Run constraint-converged active-neural Tesseract spacetime evolution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tesseract_nr.constrained_convergence import run_constrained_convergence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs=3,
        default=(8, 16, 32),
        metavar=("COARSE", "MEDIUM", "FINE"),
    )
    parser.add_argument("--final-time", type=float, default=1.0e-6)
    parser.add_argument("--picard-iterations", type=int, default=3)
    parser.add_argument("--elliptic-tolerance", type=float, default=2.0e-8)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("theory33_frozen_variants.json"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_constrained_convergence(
        args.resolutions,
        final_time=args.final_time,
        artifact_path=args.artifact,
        picard_iterations=args.picard_iterations,
        elliptic_tolerance=args.elliptic_tolerance,
    )
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if bool(result["qualified"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
