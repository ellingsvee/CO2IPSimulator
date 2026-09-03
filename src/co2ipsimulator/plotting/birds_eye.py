from __future__ import annotations

from pathlib import Path
from typing import Literal, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..model.grid import GridArrays
from ..model.properties import GridMetadata, LayerKind
from .axes import set_spatial_ticks
from ._helpers import (
    PLUME_FALLBACK,
    draw_topography,
    layer_sort_key,
    overlay_layer_polygons,
    pretty_layer_name,
)

SourceCell = tuple[int, int, int]


def _sand_layer_names(grid: GridArrays) -> list[str]:
    return [layer.name for layer in grid.layer_stack if layer.kind is LayerKind.SAND]


def plume_thickness_per_layer(
    occupation: np.ndarray,
    grid: GridArrays,
    *,
    residual_saturation: float,
    saturation: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    if occupation.shape != grid.shape:
        raise ValueError(
            f"occupation shape {occupation.shape} != grid shape {grid.shape}"
        )
    if saturation is None:
        sat_factor = occupation.astype(np.float64)
    else:
        accum_sat = max(1.0e-12, 1.0 - float(residual_saturation))
        sat_factor = np.asarray(saturation, dtype=np.float64) / accum_sat
    thicknesses: dict[str, np.ndarray] = {}
    for li, layer in enumerate(grid.layer_stack):
        if layer.kind is not LayerKind.SAND:
            continue
        layer_mask = (grid.layer_id == li) & grid.active_mask
        contrib = np.where(layer_mask & occupation, grid.dz * sat_factor, 0.0)
        thicknesses[layer.name] = contrib.sum(axis=2)
    return thicknesses


def birds_eye_panels(
    fields: Mapping[str, np.ndarray],
    *,
    grid: GridArrays | None = None,
    metadata: GridMetadata | None = None,
    layer_order: Sequence[str] | None = None,
    depth_surfaces: Mapping[str, np.ndarray] | None = None,
    observed_polygons: Mapping[str, np.ndarray] | None = None,
    overlay_outlines: Mapping[str, np.ndarray] | None = None,
    overlay_outline_color: str = "black",
    overlay_outline_linewidth: float = 2.4,
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float | None = None,
    cbar_label: str = "",
    panel_titles: Mapping[str, str] | None = None,
    ncols: int = 3,
    figsize: tuple[float, float] = (14.0, 13.0),
    mask_nonpositive: bool = True,
    field_style: Literal["image", "outline", "filled_mask"] = "image",
    outline_color: str = PLUME_FALLBACK,
    outline_linewidth: float = 2.2,
    mask_alpha: float = 0.85,
    show_colorbar: bool = True,
    source_ijk: SourceCell | None = None,
    title: str | None = None,
    output: Path | str | None = None,
    xlabel: str = "UTM easting (km)",
    ylabel: str = "UTM northing (km)",
) -> plt.Figure:
    if grid is None and metadata is None:
        raise TypeError("birds_eye_panels requires either grid or metadata")
    meta = metadata if metadata is not None else grid.metadata

    if layer_order is None:
        if grid is None:
            raise TypeError("layer_order is required when only metadata is given")
        names = [n for n in _sand_layer_names(grid) if n in fields]
    else:
        names = list(layer_order)
        missing = [n for n in names if n not in fields]
        if missing:
            raise KeyError(f"fields missing keys for layers: {missing}")
    names = sorted(names, key=layer_sort_key)
    n = len(names)
    if n == 0:
        raise ValueError("no layers to plot")

    if depth_surfaces is not None and grid is None:
        raise TypeError("depth_surfaces overlay requires grid")

    if vmax is None:
        vmax = max(
            (float(np.asarray(fields[name]).max()) for name in names), default=0.0
        )
        if vmax <= vmin:
            vmax = vmin + 1.0

    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    fig.set_constrained_layout_pads(wspace=0.02, hspace=0.03)

    xmin = meta.xmin / 1000.0
    xmax = meta.xmax / 1000.0
    ymin = meta.ymin / 1000.0
    ymax = meta.ymax / 1000.0
    extent = [xmin, xmax, ymin, ymax]
    xs_km = meta.x() / 1000.0
    ys_km = meta.y() / 1000.0
    layer_to_idx = (
        {layer.name: li for li, layer in enumerate(grid.layer_stack)}
        if grid is not None
        else {}
    )
    source_layer_name: str | None = None
    source_xy_km: tuple[float, float] | None = None
    if grid is not None and source_ijk is not None:
        i, j, k = (int(v) for v in source_ijk)
        nx, ny, nk = grid.shape
        if 0 <= i < nx and 0 <= j < ny and 0 <= k < nk:
            li = int(grid.layer_id[i, j, k])
            if 0 <= li < len(grid.layer_stack):
                source_layer_name = grid.layer_stack[li].name
                source_xy_km = (
                    float(grid.metadata.x()[i]) / 1000.0,
                    float(grid.metadata.y()[j]) / 1000.0,
                )
    last_image = None

    for panel_idx, name in enumerate(names):
        r = panel_idx // ncols
        c = panel_idx % ncols
        ax = axes[r, c]

        if depth_surfaces is not None and name in layer_to_idx:
            top = depth_surfaces.get(grid.layer_stack[layer_to_idx[name]].top_surface)
            if top is not None:
                draw_topography(ax, xs_km, ys_km, np.asarray(top, dtype=np.float64))

        field = np.asarray(fields[name], dtype=np.float64).T
        if field_style == "image":
            if mask_nonpositive:
                field = np.ma.masked_where(field <= 0.0, field)
            last_image = ax.imshow(
                field,
                origin="lower",
                extent=extent,
                interpolation="nearest",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                zorder=2,
            )
        elif field_style == "outline":
            if np.nanmax(field) > 0.0 and np.nanmin(field) < 1.0:
                ax.contour(
                    xs_km,
                    ys_km,
                    field,
                    levels=[0.5],
                    colors=[outline_color],
                    linewidths=outline_linewidth,
                    zorder=3,
                )
        elif field_style == "filled_mask":
            if np.nanmax(field) > 0.0:
                ax.contourf(
                    xs_km,
                    ys_km,
                    field,
                    levels=[0.5, 1.5],
                    colors=[outline_color],
                    alpha=mask_alpha,
                    zorder=3,
                )
        else:
            raise ValueError(f"unknown field_style {field_style!r}")

        if overlay_outlines is not None and name in overlay_outlines:
            overlay = np.asarray(overlay_outlines[name], dtype=np.float64).T
            if np.nanmax(overlay) > 0.5 and np.nanmin(overlay) < 0.5:
                ax.contour(
                    xs_km,
                    ys_km,
                    overlay,
                    levels=[0.5],
                    colors=[overlay_outline_color],
                    linewidths=overlay_outline_linewidth,
                    zorder=10,
                )

        overlay_layer_polygons(ax, name, observed_polygons)
        if source_xy_km is not None and name == source_layer_name:
            ax.plot(
                source_xy_km[0],
                source_xy_km[1],
                marker="x",
                markersize=20,
                markeredgewidth=6.0,
                color="white",
                linestyle="none",
                zorder=19,
            )
            ax.plot(
                source_xy_km[0],
                source_xy_km[1],
                marker="x",
                markersize=18,
                color="black",
                markeredgewidth=3.6,
                linestyle="none",
                zorder=20,
            )

        set_spatial_ticks(ax, xlim=(xmin, xmax), ylim=(ymin, ymax))
        ax.set_aspect("equal")
        ax.ticklabel_format(style="plain", useOffset=False)
        title_text = (panel_titles or {}).get(name, name)
        ax.set_title(title_text)
        if r == nrows - 1:
            ax.set_xlabel(xlabel)
        if c == 0:
            ax.set_ylabel(ylabel)

    for panel_idx in range(n, nrows * ncols):
        axes[panel_idx // ncols, panel_idx % ncols].set_visible(False)

    if show_colorbar and last_image is not None:
        fig.colorbar(
            last_image,
            ax=axes.ravel().tolist(),
            # aspect=55,
            pad=0.015,
            label=cbar_label,
        )

    if title:
        fig.suptitle(title, y=1.02)
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(output), dpi=180, bbox_inches="tight", transparent=True)
    return fig


def plot_plume_thickness(
    occupation: np.ndarray,
    grid: GridArrays,
    *,
    residual_saturation: float,
    saturation: np.ndarray | None = None,
    depth_surfaces: Mapping[str, np.ndarray] | None = None,
    observed_polygons: Mapping[str, np.ndarray] | None = None,
    title: str | None = None,
    output: Path | str | None = None,
    ncols: int = 3,
    figsize: tuple[float, float] = (14.0, 13.0),
    mode: Literal["mask", "thickness"] = "mask",
    source_ijk: SourceCell | None = None,
    xlabel: str = "UTM easting (km)",
    ylabel: str = "UTM northing (km)",
) -> plt.Figure:
    thick = plume_thickness_per_layer(
        occupation, grid, residual_saturation=residual_saturation, saturation=saturation
    )
    panel_titles = {name: pretty_layer_name(name) for name in thick}
    if mode == "mask":
        fields = {name: (t > 0.0).astype(np.float64) for name, t in thick.items()}
        return birds_eye_panels(
            fields,
            grid=grid,
            depth_surfaces=depth_surfaces,
            observed_polygons=observed_polygons,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            cbar_label="",
            panel_titles=panel_titles,
            ncols=ncols,
            figsize=figsize,
            field_style="filled_mask",
            outline_color=PLUME_FALLBACK,
            show_colorbar=False,
            source_ijk=source_ijk,
            title=title,
            output=output,
            xlabel=xlabel,
            ylabel=ylabel,
        )
    if mode != "thickness":
        raise ValueError(f"mode must be 'mask' or 'thickness', got {mode!r}")
    return birds_eye_panels(
        thick,
        grid=grid,
        depth_surfaces=depth_surfaces,
        observed_polygons=observed_polygons,
        cmap="viridis",
        vmin=0.0,
        vmax=None,
        cbar_label="CO2 equivalent column height (m)",
        panel_titles=panel_titles,
        ncols=ncols,
        figsize=figsize,
        source_ijk=source_ijk,
        title=title,
        output=output,
        xlabel=xlabel,
        ylabel=ylabel,
    )
