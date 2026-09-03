from __future__ import annotations

from typing import Sequence
from dataclasses import dataclass

import numpy as np

from .properties import LayerProps, GridMetadata


@dataclass
class GridArrays:
    dx: float
    dy: float
    dz: np.ndarray
    z_top: np.ndarray  # m, depth of top face (positive down)
    pth: np.ndarray  # Pa
    density_co2: np.ndarray  # kg/m^3
    porosity: np.ndarray  # fraction
    layer_id: np.ndarray  # int32, layer index into layer_stack
    active_mask: np.ndarray  # bool
    connate_water_saturation: float
    layer_stack: tuple[LayerProps, ...]
    metadata: GridMetadata

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.pth.shape)

    @property
    def is_shale(self) -> np.ndarray:
        kinds = np.array([layer.is_shale for layer in self.layer_stack], dtype=bool)
        return kinds[self.layer_id]

    @property
    def is_sand(self) -> np.ndarray:
        return ~self.is_shale

    @property
    def layer_kind(self) -> np.ndarray:
        return self.is_shale.astype(np.uint8)

    @property
    def z_bot(self) -> np.ndarray:
        return self.z_top + self.dz

    def cell_volume(self) -> np.ndarray:
        return self.dx * self.dy * self.dz

    def pore_volume(self) -> np.ndarray:
        return self.cell_volume() * self.porosity


def _dilate_max(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    out[:-1, :] = np.maximum(out[:-1, :], a[1:, :])
    out[1:, :] = np.maximum(out[1:, :], a[:-1, :])
    out[:, :-1] = np.maximum(out[:, :-1], a[:, 1:])
    out[:, 1:] = np.maximum(out[:, 1:], a[:, :-1])
    return out


def build_stratigraphic_grid(
    depth_surfaces: dict[str, np.ndarray],
    layer_stack: Sequence[LayerProps],
    metadata: GridMetadata,
    *,
    dz: float = 1.0,
    connate_water_saturation: float = 0.30,
) -> GridArrays:
    """Discretize depth surfaces into an ``(x, y, z)`` grid.

    Depths and ``dz`` are in metres. Surface arrays use ``(y, x)`` ordering. The returned fields use the simulator's ``(x, y, z)`` ordering.
    """
    nx, ny = metadata.nx, metadata.ny
    layer_stack = tuple(layer_stack)
    for layer in layer_stack:
        for surface in (layer.top_surface, layer.base_surface):
            if surface not in depth_surfaces:
                raise KeyError(f"surface {surface!r} not in depth_surfaces")

    tops = [
        np.asarray(depth_surfaces[layer.top_surface], dtype=np.float64).T
        for layer in layer_stack
    ]
    bases = [
        np.asarray(depth_surfaces[layer.base_surface], dtype=np.float64).T
        for layer in layer_stack
    ]

    z0 = float(np.nanmin(tops[0]))
    z1 = float(np.nanmax(bases[-1]))
    nz = max(1, int(np.ceil((z1 - z0) / dz)))
    z_level = z0 + np.arange(nz) * dz
    z_center = z_level + 0.5 * dz
    ks = np.arange(nz)

    layer_id = np.full((nx, ny, nz), -1, dtype=np.int32)
    for li in range(len(layer_stack)):
        top = tops[li][:, :, None]
        base = bases[li][:, :, None]
        valid = np.isfinite(top) & np.isfinite(base) & (base > top)
        inside = valid & (top <= z_center) & (z_center < base)
        layer_id = np.where(inside, np.int32(li), layer_id)

    for li, layer in enumerate(layer_stack):
        if not layer.is_shale:
            continue
        top, base = tops[li], bases[li]
        exists = np.isfinite(top) & np.isfinite(base) & (base > top)
        lo = np.floor((top - z0) / dz)
        hi = np.maximum(np.floor((base - z0) / dz - 1.0e-9), lo)
        hi = np.maximum(hi, _dilate_max(np.where(exists, lo, -np.inf)))
        lo = np.clip(np.where(exists, lo, np.inf), 0, nz - 1)
        hi = np.clip(hi, 0, nz - 1)
        band = exists[:, :, None] & (ks >= lo[:, :, None]) & (ks <= hi[:, :, None])
        layer_id = np.where(band, np.int32(li), layer_id)

    active_mask = layer_id >= 0
    li_idx = np.clip(layer_id, 0, len(layer_stack) - 1)
    pth_by = np.array([layer.pth_pa for layer in layer_stack], dtype=np.float64)
    dens_by = np.array([layer.density_co2 for layer in layer_stack], dtype=np.float64)
    poro_by = np.array([layer.porosity for layer in layer_stack], dtype=np.float64)

    z_top = np.broadcast_to(z_level, (nx, ny, nz)).astype(np.float64)
    dz_arr = np.where(active_mask, dz, 0.0)
    pth = np.where(active_mask, pth_by[li_idx], 0.0)
    density_co2 = np.where(active_mask, dens_by[li_idx], 0.0)
    porosity = np.where(active_mask, poro_by[li_idx], 0.0)

    return GridArrays(
        dx=metadata.dx,
        dy=metadata.dy,
        dz=dz_arr,
        z_top=z_top,
        pth=pth,
        density_co2=density_co2,
        porosity=porosity,
        layer_id=layer_id,
        active_mask=active_mask,
        connate_water_saturation=float(connate_water_saturation),
        layer_stack=layer_stack,
        metadata=metadata,
    )


def cell_index_for_xy(metadata: GridMetadata, x: float, y: float) -> tuple[int, int]:
    i = int(round((x - metadata.xmin) / metadata.dx))
    j = int(round((y - metadata.ymin) / metadata.dy))
    i = max(0, min(metadata.nx - 1, i))
    j = max(0, min(metadata.ny - 1, j))
    return i, j


def polygon_column_mask(metadata: GridMetadata, polygon: np.ndarray) -> np.ndarray:
    from matplotlib.path import Path

    xs = metadata.xmin + np.arange(metadata.nx) * metadata.dx
    ys = metadata.ymin + np.arange(metadata.ny) * metadata.dy
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    inside = Path(np.asarray(polygon, dtype=np.float64)).contains_points(pts)
    return inside.reshape(metadata.nx, metadata.ny)
