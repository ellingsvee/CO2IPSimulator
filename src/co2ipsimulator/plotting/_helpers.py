from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

SHALE_COLOR = "#444444"
SAND_COLOR = "#dddddd"
ACCUMULATION_COLOR = "#d62728"
PATHWAY_COLOR = "#ffbb78"
SHALE_TRANSIT_COLOR = "#d28c3a"
PLUME_FALLBACK = ACCUMULATION_COLOR
TOPO_CMAP = "Greys_r"
CONTOUR_KW = dict(colors="0.30", linewidths=0.4)
POLYGON_OUTER = dict(color="white", linewidth=1.5)
POLYGON_INNER = dict(color="black", linewidth=1.0)


def layer_sort_key(name: str) -> tuple[int, str]:
    if len(name) > 1 and name[1:].isdigit():
        return (int(name[1:]), name)
    return (10**6, name)


def pretty_layer_name(name: str) -> str:
    if name.startswith("Shale_"):
        return "Shale " + name[len("Shale_") :]
    if len(name) > 1 and name[0] in {"L", "S"} and name[1:].isdigit():
        return rf"$\mathcal{{S}}_{{{name[1:]}}}$"
    return name


def draw_topography(
    ax: plt.Axes,
    xs_km: np.ndarray,
    ys_km: np.ndarray,
    surface: np.ndarray | None,
    *,
    n_levels: int = 20,
) -> None:
    if surface is None or not np.isfinite(surface).any():
        return
    zmin = float(np.nanmin(surface))
    zmax = float(np.nanmax(surface))
    if zmax <= zmin:
        return
    levels = np.linspace(zmin, zmax, n_levels)
    ax.contourf(xs_km, ys_km, surface, levels=levels, cmap=TOPO_CMAP, alpha=0.65)
    ax.contour(xs_km, ys_km, surface, levels=levels, **CONTOUR_KW)


def overlay_polygon(ax: plt.Axes, poly: np.ndarray) -> None:
    xs = poly[:, 0] / 1000.0
    ys = poly[:, 1] / 1000.0
    ax.plot(xs, ys, **POLYGON_OUTER)
    ax.plot(xs, ys, **POLYGON_INNER)


def overlay_layer_polygons(
    ax: plt.Axes,
    layer_name: str,
    observed_polygons: Mapping[str, np.ndarray] | None,
) -> None:
    if not observed_polygons:
        return
    needle = layer_name.upper()
    for label, poly in observed_polygons.items():
        upper = label.upper()
        if not upper.startswith(needle):
            continue
        rest = upper[len(needle) :]
        if rest and rest[0].isdigit():
            continue
        overlay_polygon(ax, poly)
