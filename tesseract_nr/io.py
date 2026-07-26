"""Portable NPZ checkpoints with JSON metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_checkpoint(path: str | Path, *, metadata: dict[str, Any] | None = None, **arrays: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: np.asarray(value) for name, value in arrays.items()}
    payload["metadata_json"] = np.asarray(json.dumps(metadata or {}, sort_keys=True))
    np.savez_compressed(destination, **payload)
    return destination


def load_checkpoint(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(Path(path), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files if name != "metadata_json"}
        metadata = json.loads(str(archive["metadata_json"])) if "metadata_json" in archive else {}
    return arrays, metadata
