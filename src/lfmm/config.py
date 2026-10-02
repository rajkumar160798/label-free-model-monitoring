"""Locations of benchmark data on disk."""

from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "LFMM_DATA_DIR"


def data_dir(override: str | os.PathLike | None = None) -> Path:
    """Root data directory: explicit override, else $LFMM_DATA_DIR."""
    value = override or os.environ.get(ENV_VAR)
    if not value:
        raise RuntimeError(
            f"Data directory not set. Set the {ENV_VAR} environment variable "
            "(e.g. C:\\data\\lfmm) or pass --data-dir."
        )
    path = Path(value).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_dir(override: str | os.PathLike | None = None) -> Path:
    path = data_dir(override) / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def processed_dir(override: str | os.PathLike | None = None) -> Path:
    path = data_dir(override) / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path
