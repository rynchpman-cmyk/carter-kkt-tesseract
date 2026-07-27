"""Run the characteristic-shock and phase-boundary qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tesseract_nr.shock_frontier import run_shock_frontier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tesseract_nr_shock_results.json"),
    )
    parser.add_argument("--phase-points", type=int, default=8)
    parser.add_argument("--positivity-points", type=int, default=16)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("theory33_frozen_variants.json"),
    )
    args = parser.parse_args()
    report = run_shock_frontier(
        phase_shape=(args.phase_points, args.phase_points),
        positivity_shape=(
            args.positivity_points,
            args.positivity_points,
        ),
        artifact_path=args.artifact,
    )
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["qualification"], indent=2))
    print(f"qualified={report['qualified']}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
