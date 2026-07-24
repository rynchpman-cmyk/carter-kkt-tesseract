#!/usr/bin/env python3
"""Generate independent kinetic EOS, transport, and phase benchmark data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from theory33_kinetic_data import KineticDataConfig, write_kinetic_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data"),
    )
    args = parser.parse_args()
    metadata = write_kinetic_dataset(
        args.output_directory,
        KineticDataConfig(),
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
