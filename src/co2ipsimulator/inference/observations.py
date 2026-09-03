from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from co2ipsimulator.model import GridMetadata, load_rms_polygon, polygon_column_mask


@dataclass(frozen=True)
class PlumeOutline:
    label: str
    layer_name: str
    points: np.ndarray


@dataclass(frozen=True)
class SimulatedSnapshot:
    year: int
    mass_per_layer_kg: dict[str, float]
    footprints: dict[str, np.ndarray]
    metadata: GridMetadata


@dataclass(frozen=True)
class SnapshotObservation:
    year: int
    layer_names: tuple[str, ...]
    footprints: np.ndarray
    mass_per_layer_kg: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )


def load_plume_outlines(
    directory: Path | str, layer_names: tuple[str, ...]
) -> tuple[PlumeOutline, ...]:
    paths = tuple(
        path
        for path in sorted(Path(directory).iterdir())
        if path.is_file() and not path.name.startswith(".")
    )
    return tuple(
        PlumeOutline(path.stem, layer_name, load_rms_polygon(path))
        for layer_name in layer_names
        for path in paths
        if re.fullmatch(
            rf"{re.escape(layer_name)}[A-Za-z]*", path.stem, flags=re.IGNORECASE
        )
    )


def rasterize_plume_outlines(
    outlines: tuple[PlumeOutline, ...],
    metadata: GridMetadata,
    layer_names: tuple[str, ...],
) -> np.ndarray:
    masks = np.zeros((len(layer_names), metadata.nx, metadata.ny), dtype=bool)
    layer_index = {name: position for position, name in enumerate(layer_names)}
    for outline in outlines:
        masks[layer_index[outline.layer_name]] |= polygon_column_mask(
            metadata, outline.points
        )
    return masks


def make_snapshot_observation(
    year: int,
    layer_names: tuple[str, ...],
    footprints: Mapping[str, np.ndarray],
    mass_per_layer_kg: np.ndarray,
) -> SnapshotObservation:
    shape = next(iter(footprints.values())).shape
    masks = np.stack(
        [footprints.get(name, np.zeros(shape, dtype=bool)) for name in layer_names]
    )
    return SnapshotObservation(int(year), layer_names, masks, mass_per_layer_kg)
