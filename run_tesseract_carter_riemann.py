"""Run the full Carter eigensystem and Riemann convergence frontier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tesseract_nr.carter_riemann import run_carter_riemann_frontier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resolutions", default="16,32,64,128"
    )
    parser.add_argument("--final-time", type=float, default=0.02)
    parser.add_argument("--cfl", type=float, default=0.15)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("theory33_frozen_variants.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tesseract_carter_riemann_results.json"),
    )
    args = parser.parse_args()
    resolutions = tuple(
        int(value) for value in args.resolutions.split(",")
    )
    report = run_carter_riemann_frontier(
        resolutions,
        final_time=args.final_time,
        cfl=args.cfl,
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
