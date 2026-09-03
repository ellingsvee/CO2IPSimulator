from __future__ import annotations

from .grid import (
    GridArrays,
    build_stratigraphic_grid,
    cell_index_for_xy,
    polygon_column_mask,
)
from .io import load_metadata, load_rms_polygon
from .properties import GridMetadata, LayerKind, LayerProps
from .trapfill import (
    SealFields,
    StepDiagnostics,
    TrapFill,
    TrapFillResult,
    build_trapfill,
    describe_convergence,
    seal_fields,
)

__all__ = [
    "GridArrays",
    "GridMetadata",
    "LayerKind",
    "LayerProps",
    "StepDiagnostics",
    "TrapFill",
    "TrapFillResult",
    "build_stratigraphic_grid",
    "build_trapfill",
    "describe_convergence",
    "SealFields",
    "seal_fields",
    "cell_index_for_xy",
    "polygon_column_mask",
    "load_metadata",
    "load_rms_polygon",
]
