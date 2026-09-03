from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from co2ipsimulator.inference import RunConfig, load_run_config
from co2ipsimulator.model import build_stratigraphic_grid

from ..comparison import (
    OUTPUT_ROOT,
    experiment_root,
    load_variant_results,
    plot_forecast_comparison,
    plot_posterior_comparison,
    print_summary,
    run_variants,
)
from ..experiment import build_forward_config, calibration_years, sand_layer_names
from ..gridfields import source_ijk
from ..scenarios import Scenario, get_scenario
from .experiment import EXPERIMENT_ROOT, cases_from_config, truth_threshold
from .experiment import variant_configs
from .operator import DEFAULT_THRESHOLDS, operator_response
from .plot import plot_detection_extents


def run_operator(scenario: Scenario, run_config: RunConfig, outdir: Path) -> None:
    """Truth-side diagnostics: what the threshold removes, before any inference."""
    scanned = tuple(
        float(value)
        for value in run_config.extras.get("operator_thresholds", DEFAULT_THRESHOLDS)
    )
    outlines = tuple(
        float(value)
        for value in run_config.extras.get("outline_thresholds", (0.0, 4.0, 8.0, 12.0))
    )
    thresholds = tuple(sorted(set(scanned) | set(outlines)))
    response = operator_response(scenario, run_config, thresholds)

    config = build_forward_config(scenario, run_config, response.year)
    grid = build_stratigraphic_grid(
        config.depth_surfaces, config.layer_stack, config.metadata
    )
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "operator_thresholds.npy", response.thresholds)
    np.save(outdir / "operator_areas.npy", response.areas)
    plot_detection_extents(
        response,
        grid,
        config.depth_surfaces,
        outlines,
        source_ijk=source_ijk(grid, config.metadata, scenario.well_xy),
        output=outdir / "detection_extents.pdf",
    )
    print(f"\n{scenario.name} detection operator (year {response.year}):")
    print(f"  {'h_det':>7}  " + "  ".join(f"{n:>8}" for n in response.layer_names))
    reference = response.areas[0]
    for index, threshold in enumerate(response.thresholds):
        shares = "  ".join(
            f"{(a / r if r > 0 else 0.0):>8.2f}"
            for a, r in zip(response.areas[index], reference)
        )
        print(f"  {threshold:>7.1f}  {shares}")


def run(
    scenario: Scenario,
    run_config: RunConfig,
    *,
    plot_only: bool = False,
    operator_only: bool = False,
) -> Path:
    outdir = OUTPUT_ROOT / scenario.name / experiment_root(run_config, EXPERIMENT_ROOT)
    if run_config.extras.get("run_operator", True):
        run_operator(scenario, run_config, outdir)
    if operator_only:
        return outdir

    cases = cases_from_config(run_config)
    configs = variant_configs(run_config, cases)
    if not plot_only:
        run_variants(scenario, configs)
    results = load_variant_results(scenario, configs)

    sands = sand_layer_names(scenario)
    primary = int(scenario.forecast_year)
    last_survey = max(calibration_years(scenario, run_config))

    plot_posterior_comparison(results, output=outdir / "posterior_comparison.pdf")
    plot_forecast_comparison(
        results, sands, last_survey, output=outdir / "forecast_comparison.pdf"
    )

    print(
        f"\nSummary [plume detection] ({scenario.name}, truth "
        f"h_det = {truth_threshold(run_config):g} m):"
    )
    print_summary(results, primary)
    print(f"  -> {outdir}")
    return outdir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the synthetic plume-detection comparison."
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Experiment TOML file."
    )
    parser.add_argument(
        "--plot-only", action="store_true", help="Plot existing completed fits."
    )
    parser.add_argument(
        "--operator-only",
        action="store_true",
        help="Generate only the detection-operator diagnostics.",
    )
    args = parser.parse_args()
    run_config = load_run_config(args.config)
    scenario = get_scenario(run_config.extras["scenario"])
    run(
        scenario,
        run_config,
        plot_only=args.plot_only,
        operator_only=args.operator_only,
    )


if __name__ == "__main__":
    main()
