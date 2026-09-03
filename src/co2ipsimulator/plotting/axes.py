from __future__ import annotations

from collections.abc import Sequence

import matplotlib.axes
import numpy as np
from matplotlib.ticker import MaxNLocator


def set_year_ticks(
    ax: matplotlib.axes.Axes,
    years: Sequence[int | float] | np.ndarray,
    *,
    start: int | float | None = None,
    max_ticks: int = 5,
) -> None:
    """Show sampled years while always retaining both visible endpoints."""
    values = np.unique(np.asarray(years, dtype=float))
    if values.size == 0:
        raise ValueError("years must contain at least one value")

    first = float(values[0] if start is None else start)
    last = float(values[-1])
    visible = values[(values >= first) & (values <= last)]
    visible = np.unique(np.concatenate(([first], visible, [last])))

    count = min(max_ticks, visible.size)
    indices = np.rint(np.linspace(0, visible.size - 1, count)).astype(int)
    ax.set_xticks(np.unique(visible[indices]))
    ax.set_xlim(first, last)


def set_spatial_ticks(
    ax: matplotlib.axes.Axes,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    intervals: int = 2,
) -> None:
    """Set map ticks deterministically, independent of subplot dimensions."""
    locator = MaxNLocator(nbins=intervals)

    def ticks_within(limits: tuple[float, float]) -> np.ndarray:
        lower, upper = limits
        tolerance = max(abs(lower), abs(upper), 1.0) * 1.0e-10
        ticks = locator.tick_values(lower, upper)
        return ticks[(ticks >= lower - tolerance) & (ticks <= upper + tolerance)]

    ax.set_xticks(ticks_within(xlim))
    ax.set_yticks(ticks_within(ylim))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
