from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from co2ipsimulator.inference import RunConfig, load_run_config
from co2ipsimulator.model import build_stratigraphic_grid, build_trapfill
from co2ipsimulator.plotting import (
    birds_eye_panels,
    cross_section_continuous,
    extent_colors,
    plot_mass_per_layer,
    plot_plume_thickness,
    plume_thickness_per_layer,
    pretty_layer_name,
)

from .experiment import (
    KG_PER_MT,
    calibration_years,
    prior_distributions_and_log,
    sand_layer_names,
    truth_parameters,
)
from .gridfields import source_ijk, to_grid_fields
from .plot import plot_priors
from .scenarios import get_scenario

OUTPUT_ROOT = Path("examples/synthetic/output")


def _layer_key(name: str) -> tuple[int, str]:
    if len(name) > 1 and name[1:].isdigit():
        return (int(name[1:]), name)
    return (10**6, name)


def _year_colors(
    years: tuple[int, ...],
) -> dict[int, tuple[float, float, float, float]]:
    return dict(zip(years, extent_colors(len(years))))


def _capture(scenario, years):
    surfaces = scenario.depth_surfaces()
    stack = scenario.layer_stack()
    metadata = scenario.metadata()
    grid = build_stratigraphic_grid(surfaces, stack, metadata)
    tf = build_trapfill(
        surfaces,
        stack,
        metadata,
        source_xy=scenario.well_xy,
        seal_log10_mobility=scenario.seal_log10_mobility,
    )
    tf.reset()
    annual = scenario.annual_rate_mt * KG_PER_MT
    wanted = set(years)
    states: dict[int, dict] = {}
    for year in range(max(years) + 1):
        tf.step(annual)
        if year in wanted:
            states[year] = {
                "columns": tf.state_column_heights(),
                "mass": tf._mass_per_layer(),
            }
    return grid, surfaces, metadata, states


def _render_year(grid, surfaces, metadata, state, well_xy, outdir, tag):
    occupation, saturation, _accumulation = to_grid_fields(
        grid, surfaces, state["columns"]
    )
    source = source_ijk(grid, metadata, well_xy)
    cross_section_continuous(
        grid,
        surfaces,
        columns=state["columns"],
        direction="x",
        section_index=source[1] if source is not None else None,
        source_ijk=source,
        xlabel="x (km)",
        output=outdir / f"truth_cross_section_{tag}.pdf",
    )
    plot_plume_thickness(
        occupation,
        grid,
        residual_saturation=grid.connate_water_saturation,
        saturation=saturation,
        depth_surfaces=surfaces,
        source_ijk=source,
        mode="mask",
        output=outdir / f"truth_birdseye_{tag}.pdf",
    )
    plt.close("all")


def _render_year_extents(grid, surfaces, metadata, states, years, well_xy, outdir):
    source = source_ijk(grid, metadata, well_xy)
    year_masks = {}
    for year in years:
        occupation, saturation, _accumulation = to_grid_fields(
            grid, surfaces, states[year]["columns"]
        )
        thick = plume_thickness_per_layer(
            occupation,
            grid,
            residual_saturation=grid.connate_water_saturation,
            saturation=saturation,
        )
        year_masks[year] = {
            name: (field > 0.0).astype(np.float64) for name, field in thick.items()
        }

    first_masks = next(iter(year_masks.values()))
    layer_names = sorted(first_masks, key=_layer_key)
    empty_fields = {name: np.zeros_like(first_masks[name]) for name in layer_names}
    panel_titles = {name: pretty_layer_name(name) for name in layer_names}
    # map_aspect = (metadata.xmax - metadata.xmin) / (metadata.ymax - metadata.ymin)
    # panel_height = 3.7
    # figsize = (panel_height * map_aspect * len(layer_names), panel_height)

    figsize = (12, 4)
    fig = birds_eye_panels(
        empty_fields,
        grid=grid,
        layer_order=layer_names,
        depth_surfaces=surfaces,
        panel_titles=panel_titles,
        ncols=len(layer_names),
        figsize=figsize,
        field_style="filled_mask",
        show_colorbar=False,
        source_ijk=source,
        xlabel="x (km)",
        ylabel="y (km)",
    )
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.0, 0.18, 1.0, 1.0))

    xs_km = metadata.x() / 1000.0
    ys_km = metadata.y() / 1000.0
    colors = _year_colors(years)
    axes = [ax for ax in fig.axes if ax.get_visible()]
    for year in reversed(years):
        color = colors[year]
        for ax, layer_name in zip(axes, layer_names):
            field = year_masks[year][layer_name].T
            if np.nanmax(field) <= 0.0 or np.nanmin(field) >= 1.0:
                continue
            ax.contourf(
                xs_km,
                ys_km,
                field,
                levels=[0.5, 1.5],
                colors=[color],
                antialiased=False,
                zorder=4,
            )

    handles = [
        Patch(facecolor=colors[year], edgecolor="none", label=str(year))
        for year in years
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(handles),
        frameon=False,
    )
    output = outdir / "truth_birdseye_extents.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", transparent=True)
    plt.close(fig)


def run(scenario, run_config: RunConfig | None) -> Path:
    calib = (
        scenario.calibration_years
        if run_config is None
        else calibration_years(scenario, run_config)
    )
    years = tuple(sorted(set(calib) | {int(scenario.forecast_year)}))

    grid, surfaces, metadata, states = _capture(scenario, years)
    outdir = OUTPUT_ROOT / scenario.name / "scenario"
    outdir.mkdir(parents=True, exist_ok=True)

    sands = sand_layer_names(scenario)
    print(
        f"\n{scenario.name}: true Pth(Sh3,Sh2,Sh1)={scenario.true_pth_kpa} kPa, "
        f"log10lambda={scenario.seal_log10_mobility}"
    )
    header = "  year  " + "".join(f"{name:>9}" for name in sands) + "     total"
    print(header)
    for year in years:
        mass = states[year]["mass"]
        row = "".join(f"{mass[name] / KG_PER_MT:9.2f}" for name in sands)
        print(f"  {year:4d}  {row}  {sum(mass.values()) / KG_PER_MT:8.2f}")

    empty = {"columns": {}}
    _render_year(grid, surfaces, metadata, empty, scenario.well_xy, outdir, "initial")
    _render_year_extents(
        grid,
        surfaces,
        metadata,
        states,
        years,
        scenario.well_xy,
        outdir,
    )
    for year in years:
        _render_year(
            grid,
            surfaces,
            metadata,
            states[year],
            scenario.well_xy,
            outdir,
            str(year),
        )
        plot_mass_per_layer(
            states[year]["mass"], output=outdir / f"truth_mass_bars_{year}.pdf"
        )

    if run_config is not None:
        truth = truth_parameters(scenario, run_config)
        names = run_config.inference.parameter_names
        distributions, is_log = prior_distributions_and_log(run_config)
        plot_priors(
            names,
            distributions,
            is_log,
            truth=truth,
            output=outdir / "scenario_priors.pdf",
        )
    print(f"  -> {outdir}")
    return outdir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the geometry and truth for a synthetic scenario."
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="Optional experiment TOML file."
    )
    parser.add_argument(
        "--scenario", type=str, default=None, help="Scenario name: dome or grf."
    )
    args = parser.parse_args()

    run_config = load_run_config(args.config) if args.config is not None else None
    name = args.scenario
    if name is None and run_config is not None:
        name = run_config.extras.get("scenario")
    if name is None:
        raise SystemExit(
            "provide --scenario NAME or --config PATH (with extras.scenario)"
        )
    run(get_scenario(name), run_config)


if __name__ == "__main__":
    main()
