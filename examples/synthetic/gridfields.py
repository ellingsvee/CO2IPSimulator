from __future__ import annotations

import numpy as np

from co2ipsimulator.model import cell_index_for_xy


def to_grid_fields(grid, depth_surfaces, columns):
    nx, ny, nz = grid.shape
    occupation = np.zeros((nx, ny, nz), dtype=bool)
    saturation = np.zeros((nx, ny, nz), dtype=np.float64)
    accumulation = np.zeros((nx, ny, nz), dtype=bool)
    accumulation_sat = 1.0 - grid.connate_water_saturation
    z_top = grid.z_top

    for li, layer in enumerate(grid.layer_stack):
        if layer.is_shale or layer.name not in columns:
            continue
        in_layer = grid.layer_id == li
        top = np.asarray(depth_surfaces[layer.top_surface], dtype=np.float64).T
        column = columns[layer.name]
        contact = (top + column)[:, :, None]
        is_acc = in_layer & (z_top < contact) & (column[:, :, None] > 1e-9)
        occupation |= is_acc
        saturation = np.where(is_acc, accumulation_sat, saturation)
        accumulation |= is_acc

    return occupation, saturation, accumulation


def source_ijk(grid, metadata, well_xy):
    i, j = cell_index_for_xy(metadata, *well_xy)
    l1 = next(idx for idx, layer in enumerate(grid.layer_stack) if layer.name == "L1")
    column = np.where((grid.layer_id[i, j, :] == l1) & grid.active_mask[i, j, :])[0]
    return (i, j, int(column.max())) if column.size else None
