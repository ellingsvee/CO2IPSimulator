from __future__ import annotations

from ._helpers import pretty_layer_name
from .axes import set_spatial_ticks, set_year_ticks
from .birds_eye import (
    birds_eye_panels,
    plot_plume_thickness,
    plume_thickness_per_layer,
)
from .cross_section import cross_section, cross_section_continuous
from .mass_bars import plot_mass_per_layer
from .palette import (
    COMPARISON_COLOR_SCHEME,
    MODEL_COLOR,
    REFERENCE_COLOR,
    extent_colors,
)

__all__ = [
    "birds_eye_panels",
    "COMPARISON_COLOR_SCHEME",
    "cross_section",
    "cross_section_continuous",
    "extent_colors",
    "MODEL_COLOR",
    "plot_mass_per_layer",
    "plot_plume_thickness",
    "plume_thickness_per_layer",
    "pretty_layer_name",
    "REFERENCE_COLOR",
    "set_spatial_ticks",
    "set_year_ticks",
]
