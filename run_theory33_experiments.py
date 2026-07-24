#!/usr/bin/env python3
"""Run the ordered Theory 3.3 nonsmooth/constitutive experiment ladder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from theory33_experiments import run_all_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use shorter fits while retaining all acceptance audits",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = run_all_experiments(quick=args.quick)
    document = {
        "format": "theory33-experiment-ladder",
        "version": 1,
        "quick": args.quick,
        "passed": all(result.passed for result in results),
        "experiments": [result.to_dict() for result in results],
    }
    rendered = json.dumps(document, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not document["passed"]:
        failed = [
            result.name for result in results if not result.passed
        ]
        raise RuntimeError(f"failed experiments: {', '.join(failed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
