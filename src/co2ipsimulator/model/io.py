from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from .properties import GridMetadata


def load_metadata(metadata_path: Path) -> GridMetadata:
    arr = np.load(Path(metadata_path), allow_pickle=True)
    raw = arr.item() if arr.ndim == 0 else arr
    if isinstance(raw, str):
        raw = ast.literal_eval(raw)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"unexpected metadata format: {raw!r}")
    first = next(iter(raw.values()))
    if isinstance(first, str):
        first = ast.literal_eval(first)
    return GridMetadata(
        nx=int(first["nx"]),
        ny=int(first["ny"]),
        xmin=float(first["xmin"]),
        xmax=float(first["xmax"]),
        ymin=float(first["ymin"]),
        ymax=float(first["ymax"]),
    )


def load_rms_polygon(path: Path | str) -> np.ndarray:
    """Read an RMS-style ASCII polygon file. Returns an ``(N, 2)`` float64 array of (x, y) UTM coordinates."""
    path = Path(path)
    pts: list[tuple[float, float]] = []
    in_header = False
    saw_data = False
    with path.open() as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("!"):
                continue
            # Z-MAP+ block: start at '@FREE...' and end at a bare '@'.
            if stripped.startswith("@"):
                if stripped == "@":
                    in_header = False
                else:
                    in_header = True
                continue
            if in_header:
                continue
            parts = stripped.split()
            try:
                x = float(parts[0])
                y = float(parts[1])
            except (ValueError, IndexError):
                continue
            pts.append((x, y))
            saw_data = True
    if not saw_data:
        raise ValueError(f"no coordinates found in {path}")
    return np.asarray(pts, dtype=np.float64)
