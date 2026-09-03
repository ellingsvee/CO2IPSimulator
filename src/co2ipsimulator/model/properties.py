from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class LayerKind(Enum):
    SAND = "sand"
    SHALE = "shale"


@dataclass(frozen=True)
class GridMetadata:
    nx: int
    ny: int
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    @property
    def dx(self) -> float:
        return (self.xmax - self.xmin) / (self.nx - 1)

    @property
    def dy(self) -> float:
        return (self.ymax - self.ymin) / (self.ny - 1)

    def x(self) -> np.ndarray:
        return np.linspace(self.xmin, self.xmax, self.nx)

    def y(self) -> np.ndarray:
        return np.linspace(self.ymin, self.ymax, self.ny)


@dataclass(frozen=True)
class LayerProps:
    name: str
    kind: LayerKind
    top_surface: str
    base_surface: str
    density_co2: float
    pth_pa: float
    porosity: float

    @property
    def is_shale(self) -> bool:
        return self.kind is LayerKind.SHALE

    @property
    def is_sand(self) -> bool:
        return not self.is_shale
