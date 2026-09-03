from __future__ import annotations

import numpy as np
from matplotlib import colormaps

# Change this name to try another built-in Matplotlib qualitative colour scheme.
COMPARISON_COLOR_SCHEME = "Set1"

# Set1 blue and orange form a high-contrast, colourblind-friendly pair.
MODEL_COLOR_INDEX = 1
REFERENCE_COLOR_INDEX = 4

_comparison_colors = colormaps[COMPARISON_COLOR_SCHEME]
MODEL_COLOR = _comparison_colors(MODEL_COLOR_INDEX)
REFERENCE_COLOR = _comparison_colors(REFERENCE_COLOR_INDEX)

# Ordered colours for the plume-extent snapshot years.
EXTENT_COLOR_SCHEME = "viridis"
EXTENT_COLOR_MIN = 0.08
EXTENT_COLOR_MAX = 0.90


def extent_colors(count: int) -> tuple[tuple[float, float, float, float], ...]:
    cmap = colormaps[EXTENT_COLOR_SCHEME]
    positions = np.linspace(EXTENT_COLOR_MIN, EXTENT_COLOR_MAX, count)
    return tuple(cmap(position) for position in positions)
