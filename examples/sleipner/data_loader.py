from __future__ import annotations

from pathlib import Path

import numpy as np

from co2ipsimulator.model import GridMetadata, load_metadata


def load_sleipner_surfaces(
    directory: Path | str,
) -> tuple[dict[str, np.ndarray], GridMetadata]:
    directory = Path(directory)
    raw = load_metadata(directory / "_metadata.npy")
    metadata = GridMetadata(
        nx=raw.ny,
        ny=raw.nx,
        xmin=raw.xmin,
        xmax=raw.xmax,
        ymin=raw.ymin,
        ymax=raw.ymax,
    )
    disk_layout = (raw.ny, raw.nx)
    canonical = (metadata.ny, metadata.nx)
    surfaces: dict[str, np.ndarray] = {}
    for path in sorted(directory.glob("*.npy")):
        if path.name.startswith("_"):
            continue
        arr = np.load(path).astype(np.float64)
        if arr.shape != disk_layout:
            raise ValueError(
                f"surface {path.name} shape {arr.shape} != expected {disk_layout}"
            )
        reoriented = np.ascontiguousarray(np.flipud(arr.T))
        if reoriented.shape != canonical:
            raise ValueError(
                f"surface {path.name} reoriented to {reoriented.shape} != {canonical}"
            )
        surfaces[path.stem] = reoriented
    return surfaces, metadata
