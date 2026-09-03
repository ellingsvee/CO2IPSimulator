from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from co2ipsimulator.model import (
    build_stratigraphic_grid,
    build_trapfill,
    describe_convergence,
    cell_index_for_xy,
    load_rms_polygon,
)
from co2ipsimulator.plotting import (
    cross_section_continuous,
    plot_mass_per_layer,
    plot_plume_thickness,
)

from examples.synthetic.gridfields import to_grid_fields

from .config import ANNUAL_RATES_MT, DEPTH_SURFACES, INJECTION_SITE_FILE
from .data_loader import load_sleipner_surfaces
from .layer_stack import sleipner_layer_stack

DATA_DIR = DEPTH_SURFACES
OUTPUT_DIR = Path("examples/sleipner/output")
CAPROCK_PTH_PA = 1.0e6


def render(seal_log10_mobility: float | None = None) -> Path:
    surfaces, metadata = load_sleipner_surfaces(DATA_DIR)
    stack = sleipner_layer_stack()
    grid = build_stratigraphic_grid(surfaces, stack, metadata)

    site = load_rms_polygon(INJECTION_SITE_FILE)
    cx, cy = (float(v) for v in site.mean(axis=0))

    tf = build_trapfill(
        surfaces,
        stack,
        metadata,
        source_xy=(cx, cy),
        top_seal_pth_pa=CAPROCK_PTH_PA,
        seal_log10_mobility=seal_log10_mobility,
    )
    annual = [rate * 1e9 for rate in ANNUAL_RATES_MT]
    if seal_log10_mobility is None:
        result, convergence = tf.fill(sum(annual)), None
        columns = tf.column_heights(sum(annual))
    else:
        result, convergence = tf.run_schedule(annual)
        columns = tf.state_column_heights()

    occupation, saturation, _accumulation = to_grid_fields(grid, surfaces, columns)

    i, j = cell_index_for_xy(metadata, cx, cy)
    l1 = next(idx for idx, layer in enumerate(grid.layer_stack) if layer.name == "L1")
    col = np.where((grid.layer_id[i, j, :] == l1) & grid.active_mask[i, j, :])[0]
    source = (i, j, int(col.max())) if col.size else None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cross_section_continuous(
        grid,
        surfaces,
        direction="y",
        source_ijk=source,
        section_index=i,
        output=OUTPUT_DIR / "cross_section.pdf",
        figsize=(9, 5),
    )
    plot_plume_thickness(
        occupation,
        grid,
        residual_saturation=grid.connate_water_saturation,
        saturation=saturation,
        depth_surfaces=surfaces,
        source_ijk=source,
        mode="mask",
        output=OUTPUT_DIR / "birds_eye.pdf",
    )
    plot_mass_per_layer(result.mass_per_layer, output=OUTPUT_DIR / "mass_bars.pdf")
    plt.close("all")

    print(
        f"sleipner: stored {result.stored_kg / 1e9:.1f} Mt, "
        f"escaped {result.escaped_kg / 1e9:.2f} Mt"
        f"{describe_convergence(convergence, result)} -> {OUTPUT_DIR}"
    )
    return OUTPUT_DIR


def main() -> None:
    render()


if __name__ == "__main__":
    main()
