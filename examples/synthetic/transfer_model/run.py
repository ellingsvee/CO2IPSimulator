from __future__ import annotations

import argparse
from pathlib import Path

from co2ipsimulator.inference import load_run_config

from ..comparison import (
    OUTPUT_ROOT,
    experiment_root,
    load_variant_results,
    parameter_limits,
    plot_forecast_comparison,
    plot_posterior_comparison,
    print_summary,
    run_variants,
)
from ..experiment import calibration_years, sand_layer_names
from ..scenarios import get_scenario
from .experiment import EXPERIMENT_ROOT, TRANSFER_KEY, TRANSFER_LABEL, TRANSFERS
from .experiment import variant_configs


def run(scenario, run_config, *, plot_only: bool = False) -> Path:
    sands = sand_layer_names(scenario)
    primary = int(scenario.forecast_year)
    last_survey = max(calibration_years(scenario, run_config))
    root = OUTPUT_ROOT / scenario.name / experiment_root(run_config, EXPERIMENT_ROOT)

    results_by_transfer = {}
    for transfer in TRANSFERS:
        configs = variant_configs(run_config, transfer)
        if not plot_only:
            print(f"\n########## truth: {TRANSFER_LABEL[transfer]} ##########")
            run_variants(scenario, configs)
        results_by_transfer[transfer] = load_variant_results(scenario, configs)

    # The four cells share one seal-mobility prior, so the parameter has to sit
    # on one axis in both figures; per-figure limits would zoom each to its own
    # posterior and show the same prior as two different ones.
    rate_limit = run_config.inference.rate_limit_prior
    fixed_limits = {}
    if rate_limit is not None:
        span = parameter_limits(list(results_by_transfer.values()), rate_limit.name)
        if span is not None:
            fixed_limits[rate_limit.name] = span

    for transfer in TRANSFERS:
        results = results_by_transfer[transfer]
        outdir = root / TRANSFER_KEY[transfer]
        plot_posterior_comparison(
            results,
            fixed_limits=fixed_limits,
            output=outdir / "posterior_comparison.pdf",
        )
        plot_forecast_comparison(
            results, sands, last_survey, output=outdir / "forecast_comparison.pdf"
        )

        print(f"\nSummary [{TRANSFER_LABEL[transfer]}] ({scenario.name}):")
        print_summary(results, primary)
        print(f"  -> {outdir}")

    print(f"\n  -> {root}")
    return root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the synthetic transfer-model comparison."
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Experiment TOML file."
    )
    parser.add_argument(
        "--plot-only", action="store_true", help="Plot existing completed fits."
    )
    args = parser.parse_args()
    run_config = load_run_config(args.config)
    scenario = get_scenario(run_config.extras["scenario"])
    run(scenario, run_config, plot_only=args.plot_only)


if __name__ == "__main__":
    main()
