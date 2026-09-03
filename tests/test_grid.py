from __future__ import annotations

import numpy as np

from co2ipsimulator.model import (
    GridMetadata,
    LayerKind,
    LayerProps,
    build_stratigraphic_grid,
)


def _cross_layer_sand_contacts(g) -> int:
    li = g.layer_id
    sand = g.active_mask & g.is_sand

    def viol(a, b, la, lb):
        return int((a & b & (la != lb)).sum())

    return (
        viol(sand[:, :, :-1], sand[:, :, 1:], li[:, :, :-1], li[:, :, 1:])
        + viol(sand[:-1], sand[1:], li[:-1], li[1:])
        + viol(sand[:, :-1], sand[:, 1:], li[:, :-1], li[:, 1:])
    )


def _build_small():
    metadata = GridMetadata(nx=4, ny=3, xmin=0.0, xmax=300.0, ymin=0.0, ymax=200.0)
    surfaces = {
        "top": np.zeros((3, 4)),
        "mid1": np.full((3, 4), 10.0),
        "mid2": np.full((3, 4), 13.0),
        "bot": np.full((3, 4), 25.0),
    }
    stack = [
        LayerProps("sand_a", LayerKind.SAND, "top", "mid1", 500.0, 100.0, 0.36),
        LayerProps("shale", LayerKind.SHALE, "mid1", "mid2", 500.0, 42000.0, 0.0),
        LayerProps("sand_b", LayerKind.SAND, "mid2", "bot", 500.0, 100.0, 0.36),
    ]
    return build_stratigraphic_grid(surfaces, stack, metadata, dz=1.0)


def test_small_grid_is_regular():
    g = _build_small()
    assert g.shape == (4, 3, 25)
    assert np.all(g.dz[g.active_mask] == 1.0)
    assert float(np.ptp(g.z_top, axis=(0, 1)).max()) == 0.0
    assert np.allclose(g.z_bot, g.z_top + g.dz)
    for k in range(25):
        assert g.z_top[0, 0, k] == float(k)


def test_small_grid_layering_and_sealing():
    g = _build_small()
    assert np.all(g.layer_id[..., 0:10] == 0)
    assert np.all(g.layer_id[..., 10:13] == 1)
    assert np.all(g.layer_id[..., 13:25] == 2)
    assert np.all(g.is_shale[..., 10:13])
    assert np.array_equal(g.layer_kind.astype(bool), g.is_shale)
    assert np.all(g.pth[g.layer_id == 1] == 42000.0)
    assert np.all(g.pth[g.is_sand & g.active_mask] == 100.0)
    assert _cross_layer_sand_contacts(g) == 0


def test_steep_thin_shale_stays_sealed():
    metadata = GridMetadata(nx=5, ny=2, xmin=0.0, xmax=200.0, ymin=0.0, ymax=50.0)
    i = np.arange(5, dtype=np.float64)
    ramp = np.broadcast_to(10.0 + 3.0 * i, (2, 5))
    surfaces = {
        "top": np.zeros((2, 5)),
        "s1": ramp.copy(),
        "s2": ramp + 1.0,
        "bot": np.full((2, 5), 40.0),
    }
    stack = [
        LayerProps("sand_a", LayerKind.SAND, "top", "s1", 500.0, 100.0, 0.36),
        LayerProps("shale", LayerKind.SHALE, "s1", "s2", 500.0, 42000.0, 0.0),
        LayerProps("sand_b", LayerKind.SAND, "s2", "bot", 500.0, 100.0, 0.36),
    ]
    g = build_stratigraphic_grid(surfaces, stack, metadata, dz=1.0)
    assert _cross_layer_sand_contacts(g) == 0
    assert g.is_shale[g.active_mask].any()
