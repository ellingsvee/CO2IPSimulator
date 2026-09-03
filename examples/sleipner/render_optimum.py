from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from co2ipsimulator.inference import (
    load_plume_outlines,
    load_run_config,
    mass_fractions,
    seal_log10_mobility,
    with_threshold_pressures_kpa,
)
from co2ipsimulator.inference.configs import MassMeasure
from co2ipsimulator.model import build_stratigraphic_grid, build_trapfill
from co2ipsimulator.plotting import plot_plume_thickness

from examples.synthetic.gridfields import source_ijk, to_grid_fields
from examples.synthetic.plot import (
    OBSERVED_COLOR,
    POSTERIOR_COLOR,
    BarSeries,
    plot_mass_fraction_bars,
)

from .config import ANNUAL_RATES_MT
from .experiment import (
    CAPROCK_PTH_PA,
    POLYGON_DIRS,
    START_YEAR,
    build_forward_config,
    build_observed,
    load_model,
    sand_layer_names,
    snapshot_years,
)
from .fit import output_dir


def run(run_config) -> Path:
    outdir = output_dir(run_config)
    theta = np.load(outdir / "optimum_parameters.npy")
    n_pth = len(run_config.inference.pth_priors)
    stack_pth = theta[:n_pth]

    surfaces, metadata, stack, source_xy = load_model()
    stack = with_threshold_pressures_kpa(
        stack, run_config.inference.pth_layer_names, stack_pth
    )
    config = build_forward_config(run_config)
    grid = build_stratigraphic_grid(surfaces, stack, metadata)
    tf = build_trapfill(
        surfaces,
        stack,
        metadata,
        source_xy=source_xy,
        top_seal_pth_pa=CAPROCK_PTH_PA,
        seal_log10_mobility=seal_log10_mobility(config, theta),
    )

    years = snapshot_years(run_config)
    annual = [rate * 1e9 for rate in ANNUAL_RATES_MT]
    tf.reset()
    states: dict[int, tuple] = {}
    for k in range(len(annual)):
        tf.step(annual[k])
        year = START_YEAR + k
        if year in years:
            states[year] = (
                tf.state_column_heights(),
                tf._mass_per_layer(),
            )

    _, _, _, observations = build_observed(config, run_config)
    observed = {obs.year: obs for obs in observations}
    layers = sand_layer_names()
    source = source_ijk(grid, metadata, source_xy)
    area_measure = run_config.summary.mass_measure is MassMeasure.FOOTPRINT_AREA
    ylabel = "Footprint-area fraction" if area_measure else "Mass fraction"

    for year in years:
        columns, mass = states[year]
        occupation, saturation, _ = to_grid_fields(grid, surfaces, columns)
        outlines = load_plume_outlines(POLYGON_DIRS[year], layers)
        observed_polygons = {outline.label: outline.points for outline in outlines}
        plot_plume_thickness(
            occupation,
            grid,
            residual_saturation=grid.connate_water_saturation,
            saturation=saturation,
            observed_polygons=observed_polygons,
            source_ijk=source,
            mode="mask",
            output=outdir / f"optimum_birdseye_{year}.pdf",
        )
        if area_measure:
            model_values = np.array(
                [float((columns[name] > 1.0e-9).sum()) for name in layers]
            )
        else:
            model_values = np.array([mass[name] for name in layers])
        optimum_fraction = mass_fractions(model_values)
        observed_fraction = mass_fractions(observed[year].mass_per_layer_kg)
        plot_mass_fraction_bars(
            layers,
            [
                BarSeries("Observed", observed_fraction, OBSERVED_COLOR),
                BarSeries("Optimized", optimum_fraction, POSTERIOR_COLOR),
            ],
            ylabel=ylabel,
            output=outdir / f"optimum_mass_fraction_{year}.pdf",
        )
        print(f"{year}: birds-eye + mass bars -> {outdir}")
    return outdir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a saved optimum for a Sleipner configuration."
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Experiment TOML file."
    )
    args = parser.parse_args()
    run(load_run_config(args.config))


if __name__ == "__main__":
    main()
