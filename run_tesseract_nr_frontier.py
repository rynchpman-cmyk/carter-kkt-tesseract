#!/usr/bin/env python3
"""Run Tesseract's long-horizon, high-order, multidimensional NR campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tesseract_nr.frontier import run_frontier_campaign


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("theory33_frozen_variants.json"),
    )
    parser.add_argument("--horizon-steps", type=int, default=128)
    parser.add_argument("--skip-3d", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tesseract_nr_frontier_results.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_frontier_campaign(
        artifact_path=args.artifact,
        horizon_steps=args.horizon_steps,
        include_3d=not args.skip_3d,
    )
    rendered = json.dumps(result, indent=2)
    print(rendered)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if bool(result["qualified"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
