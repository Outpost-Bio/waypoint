"""Resolve file paths against bundled package resources.

Lets paths like ``examples/finetune_classification.parquet`` (and the
``configs/...`` paths used elsewhere) work from any working directory after a
``pip install`` — without cloning the repo — by falling back to the same
relative path inside the installed ``waypoint_bio`` package.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

_PKG = files("waypoint_bio")


def resolve_packaged_path(value: str | Path) -> str:
    """Return ``value`` if it exists in the cwd, else the same relative path
    bundled inside the package, else ``value`` unchanged (so the caller's own
    "file not found" error surfaces normally)."""
    value = str(value)
    if Path(value).exists():
        return value
    bundled = Path(str(_PKG)) / value
    if bundled.exists():
        return str(bundled)
    stripped = value.lstrip("/")
    if stripped != value:
        bundled = Path(str(_PKG)) / stripped
        if bundled.exists():
            return str(bundled)
    return value
