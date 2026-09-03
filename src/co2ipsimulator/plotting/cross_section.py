from __future__ import annotations

from pathlib import Path

from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
import numpy as np

from ..model.grid import GridArrays
from ..model.properties import LayerKind
from ._helpers import (
    ACCUMULATION_COLOR,
    PATHWAY_COLOR,
    SAND_COLOR,
    SHALE_COLOR,
    SHALE_TRANSIT_COLOR,
)

SourceCell = tuple[int, int, int]


def cross_section(
    occupation: np.ndarray,
    grid: GridArrays,
    *,
    saturation: np.ndarray | None = None,
    accumulation: np.ndarray | None = None,
    direction: str = "x",
    section_index: int | None = None,
    source_ijk: SourceCell | None = None,
    title: str | None = None,
    output: Path | str | None = None,
    figsize: tuple[float, float] = (8, 5),
    xlabel: str | None = None,
) -> plt.Figure:
    if direction not in ("x", "y"):
        raise ValueError(f"direction must be 'x' or 'y', got {direction!r}")

    nx, ny, _nk = grid.shape
    if direction == "x":
        if section_index is None:
            section_index = int(np.argmax(occupation.sum(axis=(0, 2))))
        section_index = max(0, min(ny - 1, section_index))
        slc = (slice(None), section_index, slice(None))
        coords = grid.metadata.x() / 1000.0
        if xlabel is None:
            xlabel = "UTM easting (km)"
        section_label = f"y = {grid.metadata.y()[section_index] / 1000.0:.2f} km"
    else:
        if section_index is None:
            section_index = int(np.argmax(occupation.sum(axis=(1, 2))))
        section_index = max(0, min(nx - 1, section_index))
        slc = (section_index, slice(None), slice(None))
        coords = grid.metadata.y() / 1000.0
        if xlabel is None:
            xlabel = "UTM northing (km)"
        section_label = f"x = {grid.metadata.x()[section_index] / 1000.0:.2f} km"

    z_top = grid.z_top[slc]
    dz = grid.dz[slc]
    occ = occupation[slc]
    sat = saturation[slc] if saturation is not None else None
    acc = (
        accumulation[slc]
        if accumulation is not None
        else np.zeros_like(occ, dtype=bool)
    )
    layer_kind = grid.layer_kind[slc]
    active = grid.active_mask[slc]

    n_lat = z_top.shape[0]
    if n_lat < 2:
        raise ValueError("cross-section requires at least 2 lateral cells")

    half = (coords[1] - coords[0]) / 2.0
    x0 = np.broadcast_to(coords[:, None] - half, z_top.shape)
    x1 = np.broadcast_to(coords[:, None] + half, z_top.shape)
    y0 = z_top
    y1 = z_top + dz

    active_flat = active.ravel()
    if not active_flat.any():
        raise ValueError("cross-section has no active cells")

    colors = np.full(occ.shape, SAND_COLOR, dtype=object)
    colors[layer_kind == 1] = SHALE_COLOR
    if sat is None:
        colors[occ & (layer_kind != 1)] = ACCUMULATION_COLOR
    else:
        colors[occ & (layer_kind == 1)] = SHALE_TRANSIT_COLOR
        colors[occ & (layer_kind != 1) & (sat > 0) & ~acc] = PATHWAY_COLOR
        colors[occ & acc] = ACCUMULATION_COLOR

    vertices = np.stack(
        [
            np.stack([x0, y0], axis=-1),
            np.stack([x1, y0], axis=-1),
            np.stack([x1, y1], axis=-1),
            np.stack([x0, y1], axis=-1),
        ],
        axis=2,
    ).reshape(-1, 4, 2)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    cells = PolyCollection(
        vertices[active_flat],
        facecolors=colors.ravel()[active_flat],
        edgecolors="none",
        linewidths=0,
    )
    ax.add_collection(cells)

    source_marker: tuple[float, float] | None = None
    if source_ijk is not None:
        i, j, k = (int(v) for v in source_ijk)
        if 0 <= i < nx and 0 <= j < ny and 0 <= k < grid.shape[2]:
            source_coord = (
                float(grid.metadata.x()[i]) / 1000.0
                if direction == "x"
                else float(grid.metadata.y()[j]) / 1000.0
            )
            source_depth = float(grid.z_top[i, j, k] + 0.5 * grid.dz[i, j, k])
            if np.isfinite(source_depth):
                source_marker = (source_coord, source_depth)

    depth_min = float(y0[active].min())
    depth_max = float(y1[active].max())
    if source_marker is not None:
        depth_min = min(depth_min, source_marker[1])
        depth_max = max(depth_max, source_marker[1])
    pad = max(8.0, 0.06 * (depth_max - depth_min))
    depth_min -= pad
    depth_max += pad

    top_active = np.where(active, y0, np.inf)
    col_top = top_active.min(axis=1)
    col_top = np.where(np.isfinite(col_top), col_top, depth_min)
    edges = np.concatenate([coords - half, [coords[-1] + half]])
    ax.fill_between(
        edges,
        depth_min,
        np.concatenate([col_top, [col_top[-1]]]),
        step="post",
        color=SHALE_COLOR,
        linewidth=0,
        zorder=0,
    )

    ax.invert_yaxis()
    ax.set_xlim(float(coords[0] - half), float(coords[-1] + half))
    ax.set_ylim(depth_max, depth_min)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Depth (m)")
    if source_marker is not None:
        ax.plot(
            source_marker[0],
            source_marker[1],
            marker="x",
            color="white",
            markersize=21,
            markeredgewidth=6.0,
            linestyle="none",
            zorder=19,
        )
        ax.plot(
            source_marker[0],
            source_marker[1],
            marker="x",
            color="black",
            markersize=19,
            markeredgewidth=3.8,
            linestyle="none",
            zorder=20,
        )
    if title:
        ax.set_title(f"{title}  ({section_label})")
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(output), dpi=180, transparent=True)
    return fig


def cross_section_continuous(
    grid: GridArrays,
    surfaces: dict[str, np.ndarray],
    *,
    columns: dict[str, np.ndarray] | None = None,
    direction: str = "x",
    section_index: int | None = None,
    source_ijk: SourceCell | None = None,
    title: str | None = None,
    output: Path | str | None = None,
    figsize: tuple[float, float] = (8, 5),
    xlabel: str | None = None,
) -> plt.Figure:
    if direction not in ("x", "y"):
        raise ValueError(f"direction must be 'x' or 'y', got {direction!r}")

    nx, ny, _nk = grid.shape
    if direction == "x":
        if section_index is None:
            section_index = _best_column_section(columns, ny, axis=0)
        section_index = max(0, min(ny - 1, section_index))
        coords = grid.metadata.x() / 1000.0
        if xlabel is None:
            xlabel = "UTM easting (km)"
        section_label = f"y = {grid.metadata.y()[section_index] / 1000.0:.2f} km"

        def surface_profile(name: str) -> np.ndarray:
            return np.asarray(surfaces[name], dtype=np.float64)[section_index, :]

        def column_profile(field: np.ndarray) -> np.ndarray:
            return np.asarray(field, dtype=np.float64)[:, section_index]

    else:
        if section_index is None:
            section_index = _best_column_section(columns, nx, axis=1)
        section_index = max(0, min(nx - 1, section_index))
        coords = grid.metadata.y() / 1000.0
        if xlabel is None:
            xlabel = "UTM northing (km)"
        section_label = f"x = {grid.metadata.x()[section_index] / 1000.0:.2f} km"

        def surface_profile(name: str) -> np.ndarray:
            return np.asarray(surfaces[name], dtype=np.float64)[:, section_index]

        def column_profile(field: np.ndarray) -> np.ndarray:
            return np.asarray(field, dtype=np.float64)[section_index, :]

    stack = grid.layer_stack
    if not stack:
        raise ValueError("grid has no layers to plot")

    layer_tops = [surface_profile(layer.top_surface) for layer in stack]
    bottom_base = surface_profile(stack[-1].base_surface)
    all_vals = np.concatenate(layer_tops + [bottom_base])
    finite = all_vals[np.isfinite(all_vals)]
    if finite.size == 0:
        raise ValueError("cross-section has no finite surface depths")
    depth_min = float(finite.min())
    depth_max = float(finite.max())

    source_marker: tuple[float, float] | None = None
    if source_ijk is not None:
        i, j, k = (int(v) for v in source_ijk)
        if 0 <= i < nx and 0 <= j < ny and 0 <= k < grid.shape[2]:
            source_coord = (
                float(grid.metadata.x()[i]) / 1000.0
                if direction == "x"
                else float(grid.metadata.y()[j]) / 1000.0
            )
            source_depth = float(grid.z_top[i, j, k] + 0.5 * grid.dz[i, j, k])
            if np.isfinite(source_depth):
                source_marker = (source_coord, source_depth)
                depth_min = min(depth_min, source_depth)
                depth_max = max(depth_max, source_depth)

    pad = max(8.0, 0.06 * (depth_max - depth_min))
    depth_min -= pad
    depth_max += pad

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.fill_between(
        coords, depth_min, layer_tops[0], color=SHALE_COLOR, linewidth=0, zorder=0
    )

    for layer, top in zip(stack, layer_tops):
        base = surface_profile(layer.base_surface)
        color = SHALE_COLOR if layer.kind is LayerKind.SHALE else SAND_COLOR
        ax.fill_between(coords, top, base, color=color, linewidth=0, zorder=1)

    if columns is not None:
        for layer in stack:
            if layer.kind is LayerKind.SHALE or layer.name not in columns:
                continue
            top = surface_profile(layer.top_surface)
            base = surface_profile(layer.base_surface)
            height = np.clip(column_profile(columns[layer.name]), 0.0, None)
            contact = np.minimum(base, top + height)
            ax.fill_between(
                coords,
                top,
                contact,
                where=height > 1.0e-9,
                color=ACCUMULATION_COLOR,
                linewidth=0,
                zorder=4,
            )

    for surf in layer_tops + [bottom_base]:
        ax.plot(coords, surf, color="0.15", linewidth=0.6, zorder=5)

    ax.invert_yaxis()
    ax.set_xlim(float(coords[0]), float(coords[-1]))
    ax.set_ylim(depth_max, depth_min)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Depth (m)")
    if source_marker is not None:
        ax.plot(
            source_marker[0],
            source_marker[1],
            marker="x",
            color="white",
            markersize=21,
            markeredgewidth=6.0,
            linestyle="none",
            zorder=19,
        )
        ax.plot(
            source_marker[0],
            source_marker[1],
            marker="x",
            color="black",
            markersize=19,
            markeredgewidth=3.8,
            linestyle="none",
            zorder=20,
        )
    if title:
        ax.set_title(f"{title}  ({section_label})")
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(output), dpi=180, transparent=True)
    return fig


def _best_column_section(
    columns: dict[str, np.ndarray] | None, fallback_size: int, *, axis: int
) -> int:
    if not columns:
        return fallback_size // 2
    total = None
    for field in columns.values():
        arr = np.asarray(field, dtype=np.float64)
        total = arr.copy() if total is None else total + arr
    if total is None or not np.isfinite(total).any() or float(np.nanmax(total)) <= 0.0:
        return fallback_size // 2
    profile = np.nan_to_num(total).sum(axis=axis)
    return int(np.argmax(profile))
