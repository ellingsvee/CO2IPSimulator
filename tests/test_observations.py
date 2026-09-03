from __future__ import annotations

import numpy as np

from co2ipsimulator.inference import load_plume_outlines, rasterize_plume_outlines
from co2ipsimulator.model import GridMetadata


def _write_polygon(path, points):
    path.write_text("\n".join(f"{x} {y}" for x, y in points))


def test_outline_loader_groups_parts_by_exact_layer_name(tmp_path):
    square_a = [(1, 1), (4, 1), (4, 4), (1, 4), (1, 1)]
    square_b = [(6, 6), (9, 6), (9, 9), (6, 9), (6, 6)]
    _write_polygon(tmp_path / "L1a", square_a)
    _write_polygon(tmp_path / "L1b", square_b)
    _write_polygon(tmp_path / "L10", square_b)

    outlines = load_plume_outlines(tmp_path, ("L1", "L10"))
    metadata = GridMetadata(nx=11, ny=11, xmin=0.0, xmax=10.0, ymin=0.0, ymax=10.0)
    masks = rasterize_plume_outlines(outlines, metadata, ("L1", "L10"))

    assert [outline.layer_name for outline in outlines] == ["L1", "L1", "L10"]
    assert masks.shape == (2, 11, 11)
    assert masks[0].sum() > masks[1].sum()
    assert not np.array_equal(masks[0], masks[1])
