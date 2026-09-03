from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.patches import Patch

from co2ipsimulator.plotting import birds_eye_panels, extent_colors
from co2ipsimulator.plotting import pretty_layer_name as _pretty_name
from co2ipsimulator.plotting._helpers import layer_sort_key

from ..plot import _save
from .operator import OperatorResponse


def plot_detection_extents(
    response: OperatorResponse,
    grid,
    depth_surfaces: dict[str, np.ndarray],
    outline_thresholds: tuple[float, ...],
    *,
    source_ijk=None,
    figsize: tuple[float, float] = (12.0, 4.0),
    output: Path | str,
) -> None:
    """Detected extent of the truth at each threshold, nested per sand unit.

    Laid out like the scenario year-extent figure. Extents shrink as the
    threshold rises, so they are drawn from the loosest threshold down and the
    tightest one ends on top.
    """
    reference = response.masks[float(response.thresholds[0])]
    names = [name for name in response.layer_names if reference[name].any()] or list(
        response.layer_names
    )
    names = sorted(names, key=layer_sort_key)
    empty = {name: np.zeros_like(reference[name], dtype=np.float64) for name in names}
    fig = birds_eye_panels(
        empty,
        grid=grid,
        layer_order=names,
        depth_surfaces=depth_surfaces,
        panel_titles={name: _pretty_name(name) for name in names},
        ncols=len(names),
        figsize=figsize,
        field_style="filled_mask",
        show_colorbar=False,
        source_ijk=source_ijk,
        xlabel="x (km)",
        ylabel="y (km)",
    )
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.set(rect=(0.0, 0.18, 1.0, 1.0))

    xs_km = response.metadata.x() / 1000.0
    ys_km = response.metadata.y() / 1000.0
    # Reversed, so the largest extent - the one a perfect survey would see - is
    # the brightest and the shrinking ones darken inward.
    colors = extent_colors(len(outline_thresholds))[::-1]
    axes = [ax for ax in fig.axes if ax.get_visible()]
    for threshold, color in zip(outline_thresholds, colors):
        for ax, name in zip(axes, names):
            field = response.masks[threshold][name].astype(np.float64).T
            if np.nanmax(field) <= 0.0 or np.nanmin(field) >= 1.0:
                continue
            ax.contourf(
                xs_km,
                ys_km,
                field,
                levels=[0.5, 1.5],
                colors=[color],
                antialiased=False,
                zorder=4,
            )

    handles = [
        Patch(
            facecolor=color,
            edgecolor="none",
            label=rf"$h_\mathrm{{det}} = {threshold:g}$ m",
        )
        for threshold, color in zip(outline_thresholds, colors)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(handles),
        frameon=False,
    )
    _save(fig, output)
