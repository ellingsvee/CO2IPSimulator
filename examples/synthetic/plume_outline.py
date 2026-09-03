"""Generate the three single-layer panels used to explain a plume outline.

Edit the variables in ``USER CONFIGURATION`` and run

    uv run python -m examples.synthetic.plume_outline

The requested section coordinates are snapped to the nearest grid coordinate;
the exact coordinates used are printed when the script finishes.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from co2ipsimulator.model import build_stratigraphic_grid, build_trapfill
from co2ipsimulator.plotting import birds_eye_panels, pretty_layer_name

from .experiment import KG_PER_MT
from .scenarios.grf import GRF, GrfTopography


# GrfTopography uses this seed for the first structural unit and consecutive
# seeds for the remaining units.
# GRF_SEED = 36
GRF_SEED = 16
YEAR = 30
LAYER = "L2"

# Coordinates are specified in metres and snapped to the nearest grid node.
# CROSS_SECTION_X_M = 2500.0
CROSS_SECTION_X_M = 3100.0
# CROSS_SECTION_Y_M = 3700.0
CROSS_SECTION_Y_M = 3500.0

OUTPUT_DIR = Path("examples/synthetic/output/grf/plume_outline")

# RGB(103, 170, 207), matching LaTeX's ``layerblue``.
PLUME_COLOR = "#67aacf"
# Choose "filled", "outline", or "outline_with_fill".
BIRDSEYE_PLUME_STYLE = "outline_with_fill"
BIRDSEYE_FILL_ALPHA = 0.50

BIRDSEYE_OUTLINE_LINE_WIDTH = 6.0
# SECTION_LINE_COLOR = "#d62728"
SECTION_LINE_COLOR = "black"
SECTION_LINE_WIDTH = 4.0
# SECTION_LINE_STYLE = "--"
# SECTION_LINE_STYLE = (0, (1.0, 4.0))
SECTION_LINE_STYLE = (4.5, (3, 2.0))


BIRDSEYE_FIGSIZE = (5.8, 5.2)
CROSS_SECTION_FIGSIZE = (7.2, 4.5)
DEPTH_PADDING_FRACTION = 0.08
MINIMUM_DEPTH_PADDING_M = 5.0


SURROUNDING_COLOR = "#444444"
SURFACE_LINE_COLOR = "0.15"
PLUME_HEIGHT_EPSILON_M = 1.0e-9


def _scenario_with_seed(seed: int):
    topography = GRF.topography
    if not isinstance(topography, GrfTopography):
        raise TypeError("the GRF scenario does not use GrfTopography")
    return replace(GRF, topography=replace(topography, warp_seed=seed))


def _simulate_columns(scenario, year: int):
    """Run the GRF truth through ``year``, matching synthetic.scenario."""
    if year < scenario.start_year:
        raise ValueError(
            f"YEAR={year} is before scenario start year {scenario.start_year}"
        )

    surfaces = scenario.depth_surfaces()
    stack = scenario.layer_stack()
    metadata = scenario.metadata()
    grid = build_stratigraphic_grid(surfaces, stack, metadata)
    trapfill = build_trapfill(
        surfaces,
        stack,
        metadata,
        source_xy=scenario.well_xy,
        seal_log10_mobility=scenario.seal_log10_mobility,
    )
    trapfill.reset()
    annual_mass_kg = scenario.annual_rate_mt * KG_PER_MT
    for _simulation_year in range(scenario.start_year, year + 1):
        trapfill.step(annual_mass_kg)

    return grid, surfaces, trapfill.state_column_heights()


def _nearest_index(values: np.ndarray, requested: float, label: str) -> int:
    lower = float(values[0])
    upper = float(values[-1])
    if not lower <= requested <= upper:
        raise ValueError(
            f"{label}={requested:g} m is outside the domain [{lower:g}, {upper:g}] m"
        )
    return int(np.argmin(np.abs(values - requested)))


def _layer_topography(grid, surfaces, layer_name: str) -> np.ndarray:
    layers = {layer.name: layer for layer in grid.layer_stack if not layer.is_shale}
    if layer_name not in layers:
        choices = ", ".join(layers)
        raise ValueError(f"unknown sand LAYER={layer_name!r}; choose one of {choices}")
    layer = layers[layer_name]
    return np.asarray(surfaces[layer.top_surface], dtype=np.float64)


def _save(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", transparent=True)
    plt.close(fig)


def _plot_birdseye(
    grid,
    surfaces,
    plume_height: np.ndarray,
    *,
    x_index: int,
    y_index: int,
    output: Path,
) -> None:
    plume_mask = (plume_height > PLUME_HEIGHT_EPSILON_M).astype(np.float64)
    style = _validate_birdseye_style(BIRDSEYE_PLUME_STYLE)
    draw_outline = style in ("outline", "outline_with_fill")
    draw_fill = style in ("filled", "outline_with_fill")
    fig = birds_eye_panels(
        {LAYER: plume_mask},
        grid=grid,
        layer_order=(LAYER,),
        depth_surfaces=surfaces,
        panel_titles={LAYER: pretty_layer_name(LAYER)},
        ncols=1,
        figsize=BIRDSEYE_FIGSIZE,
        field_style="filled_mask" if draw_fill else "outline",
        outline_color=PLUME_COLOR,
        outline_linewidth=BIRDSEYE_OUTLINE_LINE_WIDTH,
        mask_alpha=BIRDSEYE_FILL_ALPHA if draw_outline else 1.0,
        overlay_outlines={LAYER: plume_mask} if draw_fill and draw_outline else None,
        overlay_outline_color=PLUME_COLOR,
        overlay_outline_linewidth=BIRDSEYE_OUTLINE_LINE_WIDTH,
        show_colorbar=False,
        xlabel="x (km)",
        ylabel="y (km)",
    )
    ax = fig.axes[0]
    ax.set_title("")
    if draw_outline:
        _plot_domain_boundary_outline(ax, plume_mask.astype(bool), grid.metadata)
    x_km = float(grid.metadata.x()[x_index]) / 1000.0
    y_km = float(grid.metadata.y()[y_index]) / 1000.0
    line_style = {
        "color": SECTION_LINE_COLOR,
        "linestyle": SECTION_LINE_STYLE,
        "linewidth": SECTION_LINE_WIDTH,
        "zorder": 30,
    }
    ax.axhline(y_km, **line_style)
    ax.axvline(x_km, **line_style)
    _save(fig, output)


def _validate_birdseye_style(style: str) -> str:
    choices = ("filled", "outline", "outline_with_fill")
    if style not in choices:
        formatted = ", ".join(repr(choice) for choice in choices)
        raise ValueError(
            f"BIRDSEYE_PLUME_STYLE must be one of {formatted}, got {style!r}"
        )
    if not 0.0 <= BIRDSEYE_FILL_ALPHA <= 1.0:
        raise ValueError(
            f"BIRDSEYE_FILL_ALPHA must be between 0 and 1, got {BIRDSEYE_FILL_ALPHA}"
        )
    return style


def _plot_domain_boundary_outline(
    ax: plt.Axes,
    plume_mask: np.ndarray,
    metadata,
) -> None:
    """Close contour segments wherever the plume reaches a domain edge."""
    expected_shape = (metadata.nx, metadata.ny)
    if plume_mask.shape != expected_shape:
        raise ValueError(
            f"plume mask has shape {plume_mask.shape}, expected {expected_shape}"
        )

    xs_km = metadata.x() / 1000.0
    ys_km = metadata.y() / 1000.0
    xmin, xmax = float(xs_km[0]), float(xs_km[-1])
    ymin, ymax = float(ys_km[0]), float(ys_km[-1])

    _plot_boundary_runs(ax, ys_km, plume_mask[0, :], fixed=xmin, vertical=True)
    _plot_boundary_runs(ax, ys_km, plume_mask[-1, :], fixed=xmax, vertical=True)
    _plot_boundary_runs(ax, xs_km, plume_mask[:, 0], fixed=ymin, vertical=False)
    _plot_boundary_runs(ax, xs_km, plume_mask[:, -1], fixed=ymax, vertical=False)


def _plot_boundary_runs(
    ax: plt.Axes,
    coordinates: np.ndarray,
    occupied: np.ndarray,
    *,
    fixed: float,
    vertical: bool,
) -> None:
    transitions = np.diff(np.pad(np.asarray(occupied, dtype=np.int8), (1, 1)))
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1) - 1

    for start, stop in zip(starts, stops):
        begin = (
            float(coordinates[0])
            if start == 0
            else float(0.5 * (coordinates[start - 1] + coordinates[start]))
        )
        end = (
            float(coordinates[-1])
            if stop == len(coordinates) - 1
            else float(0.5 * (coordinates[stop] + coordinates[stop + 1]))
        )
        varying = [begin, end]
        constant = [fixed, fixed]
        ax.plot(
            constant if vertical else varying,
            varying if vertical else constant,
            color=PLUME_COLOR,
            # The axes clips the half centred outside the domain. Doubling the
            # boundary stroke retains the same visible width as the interior
            # outline while keeping it fully inside the map box.
            linewidth=2.0 * BIRDSEYE_OUTLINE_LINE_WIDTH,
            alpha=1.0,
            solid_capstyle="butt",
            clip_on=True,
            zorder=15,
        )


def _profiles(
    top: np.ndarray,
    plume_height: np.ndarray,
    *,
    direction: str,
    section_index: int,
    grid,
):
    if direction == "x":
        coords_km = grid.metadata.x() / 1000.0
        return (
            coords_km,
            top[section_index, :],
            plume_height[:, section_index],
        )
    if direction == "y":
        coords_km = grid.metadata.y() / 1000.0
        return (
            coords_km,
            top[:, section_index],
            plume_height[section_index, :],
        )
    raise ValueError(f"unknown direction {direction!r}")


def _shared_depth_axis(*profiles) -> tuple[tuple[float, float], np.ndarray]:
    values = np.concatenate(
        [np.asarray(profile, dtype=np.float64) for profile in profiles]
    )
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("cross-sections contain no finite depth values")
    depth_min = float(finite.min())
    depth_max = float(finite.max())
    padding = max(
        MINIMUM_DEPTH_PADDING_M,
        DEPTH_PADDING_FRACTION * (depth_max - depth_min),
    )
    limits = (depth_min - padding, depth_max + padding)
    locator = MaxNLocator(nbins=5)
    ticks = locator.tick_values(*limits)
    ticks = ticks[(ticks >= limits[0]) & (ticks <= limits[1])]
    return limits, ticks


def _plot_cross_section(
    coords_km: np.ndarray,
    top: np.ndarray,
    plume_height: np.ndarray,
    *,
    xlabel: str,
    depth_limits: tuple[float, float],
    depth_ticks: np.ndarray,
    output: Path,
) -> None:
    depth_min, depth_max = depth_limits
    height = np.clip(np.asarray(plume_height, dtype=np.float64), 0.0, None)

    fig, ax = plt.subplots(figsize=CROSS_SECTION_FIGSIZE, constrained_layout=True)
    ax.fill_between(
        coords_km,
        depth_min,
        top,
        color=SURROUNDING_COLOR,
        linewidth=0,
        zorder=0,
    )
    _fill_occupied_cells(ax, coords_km, top, height)
    ax.plot(coords_km, top, color=SURFACE_LINE_COLOR, linewidth=0.8, zorder=5)

    ax.set_xlim(float(coords_km[0]), float(coords_km[-1]))
    ax.set_ylim(depth_max, depth_min)
    ax.set_yticks(depth_ticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Depth (m)")
    _save(fig, output)


def _fill_occupied_cells(
    ax: plt.Axes,
    coords_km: np.ndarray,
    top: np.ndarray,
    height: np.ndarray,
) -> None:
    """Fill each plume through the topography/contact intersections.

    Matplotlib's masked ``fill_between`` requires both ends of an interval to
    be occupied, so it stops at the outermost occupied node. The physical CO2
    contact instead continues to the point between nodes where it intersects
    the topographic surface. Add those intersection points explicitly.
    """
    occupied = height > PLUME_HEIGHT_EPSILON_M
    transitions = np.diff(np.pad(occupied.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1) - 1

    for start, stop in zip(starts, stops):
        contact = top[start : stop + 1] + height[start : stop + 1]

        if start == 0:
            left = float(coords_km[0])
            left_top = float(top[0])
            left_contact = float(contact[0])
        else:
            left_contact = float(contact[0])
            left = _surface_level_intersection(
                float(coords_km[start - 1]),
                float(top[start - 1]),
                float(coords_km[start]),
                float(top[start]),
                left_contact,
            )
            left_top = left_contact

        if stop == len(coords_km) - 1:
            right = float(coords_km[-1])
            right_top = float(top[-1])
            right_contact = float(contact[-1])
        else:
            right_contact = float(contact[-1])
            right = _surface_level_intersection(
                float(coords_km[stop]),
                float(top[stop]),
                float(coords_km[stop + 1]),
                float(top[stop + 1]),
                right_contact,
            )
            right_top = right_contact

        section_coords = np.concatenate(([left], coords_km[start : stop + 1], [right]))
        section_top = np.concatenate(([left_top], top[start : stop + 1], [right_top]))
        section_contact = np.concatenate(([left_contact], contact, [right_contact]))
        ax.fill_between(
            section_coords,
            section_top,
            section_contact,
            color=PLUME_COLOR,
            linewidth=0,
            zorder=4,
        )


def _surface_level_intersection(
    x0: float,
    surface0: float,
    x1: float,
    surface1: float,
    level: float,
) -> float:
    delta = surface1 - surface0
    if abs(delta) < 1.0e-12:
        return 0.5 * (x0 + x1)
    fraction = np.clip((level - surface0) / delta, 0.0, 1.0)
    return float(x0 + fraction * (x1 - x0))


def main() -> None:
    scenario = _scenario_with_seed(GRF_SEED)
    grid, surfaces, columns = _simulate_columns(scenario, YEAR)
    if LAYER not in columns:
        available = ", ".join(sorted(columns)) or "none"
        raise ValueError(
            f"LAYER={LAYER!r} has no CO2 column at YEAR={YEAR}; "
            f"available layers: {available}"
        )

    plume_height = np.asarray(columns[LAYER], dtype=np.float64)
    x_index = _nearest_index(grid.metadata.x(), CROSS_SECTION_X_M, "x")
    y_index = _nearest_index(grid.metadata.y(), CROSS_SECTION_Y_M, "y")
    x_m = float(grid.metadata.x()[x_index])
    y_m = float(grid.metadata.y()[y_index])

    top = _layer_topography(grid, surfaces, LAYER)
    x_profile = _profiles(
        top,
        plume_height,
        direction="x",
        section_index=y_index,
        grid=grid,
    )
    y_profile = _profiles(
        top,
        plume_height,
        direction="y",
        section_index=x_index,
        grid=grid,
    )
    depth_limits, depth_ticks = _shared_depth_axis(
        x_profile[1],
        x_profile[1] + x_profile[2],
        y_profile[1],
        y_profile[1] + y_profile[2],
    )

    outputs = {
        "bird's-eye": OUTPUT_DIR / "plume_outline_birdseye.pdf",
        "at y": OUTPUT_DIR / "plume_outline_cross_section_at_y.pdf",
        "at x": OUTPUT_DIR / "plume_outline_cross_section_at_x.pdf",
    }
    _plot_birdseye(
        grid,
        surfaces,
        plume_height,
        x_index=x_index,
        y_index=y_index,
        output=outputs["bird's-eye"],
    )
    _plot_cross_section(
        *x_profile,
        xlabel="x (km)",
        depth_limits=depth_limits,
        depth_ticks=depth_ticks,
        output=outputs["at y"],
    )
    _plot_cross_section(
        *y_profile,
        xlabel="y (km)",
        depth_limits=depth_limits,
        depth_ticks=depth_ticks,
        output=outputs["at x"],
    )

    print(
        f"Generated {pretty_layer_name(LAYER)} at year {YEAR} with GRF seed "
        f"{GRF_SEED}; sections use x={x_m:g} m and y={y_m:g} m."
    )
    for label, output in outputs.items():
        print(f"  {label:10s} -> {output}")


if __name__ == "__main__":
    main()
