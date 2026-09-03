from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from co2ipsimulator.model import GridMetadata, LayerKind, LayerProps


@dataclass(frozen=True)
class LayeredDome:
    surfaces: dict[str, np.ndarray]
    stack: tuple[LayerProps, ...]
    metadata: GridMetadata
    source_xy: tuple[float, float]


@pytest.fixture(scope="session")
def layered_dome() -> LayeredDome:
    """Small deterministic geometry shared by simulator/inference tests."""
    n = 11
    metadata = GridMetadata(
        nx=n,
        ny=n,
        xmin=0.0,
        xmax=1000.0,
        ymin=0.0,
        ymax=1000.0,
    )
    x, y = np.meshgrid(metadata.x(), metadata.y(), indexing="xy")
    dome = 30.0 * np.exp(-((x - 500.0) ** 2 + (y - 500.0) ** 2) / (2.0 * 300.0**2))

    def surface(base: float) -> np.ndarray:
        return np.ascontiguousarray(base - dome)

    surfaces = {
        "top": surface(800.0),
        "mid1": surface(840.0),
        "mid2": surface(846.0),
        "bot": surface(886.0),
    }
    stack = (
        LayerProps("upper_sand", LayerKind.SAND, "top", "mid1", 700.0, 100.0, 0.35),
        LayerProps("shale", LayerKind.SHALE, "mid1", "mid2", 700.0, 4.0e4, 0.0),
        LayerProps("lower_sand", LayerKind.SAND, "mid2", "bot", 700.0, 100.0, 0.35),
    )
    return LayeredDome(surfaces, stack, metadata, (500.0, 500.0))
