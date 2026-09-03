from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from co2ipsimulator.inference import forward_snapshots, load_run_config
from co2ipsimulator.model import build_stratigraphic_grid

from .experiment import (
    build_forward_config,
    calibration_years,
    forecast_years,
    sand_layer_names,
    truth_forward,
)
from .fit import output_dir
from .plot import (
    OBSERVED_COLOR,
    POSTERIOR_COLOR,
    BarSeries,
    plot_forecast_timeseries,
    plot_mass_fraction_bars,
    plot_probability_maps,
)
from .scenarios import get_scenario


def _fractions(mass: np.ndarray) -> np.ndarray:
    total = mass.sum(axis=-1, keepdims=True)
    return np.divide(mass, total, out=np.zeros_like(mass), where=total > 0.0)


def run(scenario, run_config) -> Path:
    outdir = output_dir(scenario, run_config)
    samples = np.load(outdir / "abc_posterior_parameter_samples.npy")

    years = forecast_years(scenario, run_config)
    primary = int(scenario.forecast_year)
    sands = sand_layer_names(scenario)
    metadata = scenario.metadata()
    config = build_forward_config(scenario, run_config, max(years))
    grid = build_stratigraphic_grid(
        config.depth_surfaces, config.layer_stack, config.metadata
    )

    ensemble = int(run_config.extras.get("posterior_ensemble_samples", 100))
    rng = np.random.default_rng(int(run_config.pymc.seed))
    n_used = min(ensemble, samples.shape[0])
    indices = rng.choice(samples.shape[0], n_used, replace=False)

    shape = (metadata.nx, metadata.ny)
    probability = {name: np.zeros(shape) for name in sands}
    mass = np.zeros((n_used, len(years), len(sands)))
    for position, index in enumerate(indices):
        by_year = {
            snapshot.year: snapshot
            for snapshot in forward_snapshots(config, samples[index], years)[0]
        }
        for year_index, year in enumerate(years):
            snapshot = by_year[year]
            for layer_index, name in enumerate(sands):
                mass[position, year_index, layer_index] = snapshot.mass_per_layer_kg[
                    name
                ]
                if year == primary:
                    probability[name] += snapshot.footprints[name].astype(float)
    for name in sands:
        probability[name] /= n_used

    truth_config, truth = truth_forward(scenario, run_config, config)
    truth_by_year = {
        snapshot.year: snapshot
        for snapshot in forward_snapshots(truth_config, truth, years)[0]
    }
    truth_footprints = {
        name: truth_by_year[primary].footprints[name].astype(float) for name in sands
    }
    truth_mass = np.array(
        [[truth_by_year[y].mass_per_layer_kg[name] for name in sands] for y in years]
    )
    truth_fraction = _fractions(truth_mass)

    fractions = _fractions(mass)
    median = np.quantile(fractions, 0.5, axis=0)
    lower = np.quantile(fractions, 0.05, axis=0)
    upper = np.quantile(fractions, 0.95, axis=0)
    primary_index = years.index(primary)
    last_survey = max(calibration_years(scenario, run_config))

    np.save(outdir / "forecast_years.npy", np.array(years))
    np.save(outdir / "forecast_posterior_mass_fraction.npy", fractions)
    np.save(outdir / "forecast_truth_mass_fraction.npy", truth_fraction)
    np.save(
        outdir / f"forecast_posterior_probability_{primary}.npy",
        np.stack([probability[name] for name in sands]),
    )

    plot_probability_maps(
        probability,
        metadata,
        grid=grid,
        depth_surfaces=config.depth_surfaces,
        reference_footprints=truth_footprints,
        layer_order=sands,
        xlabel="x (km)",
        ylabel="y (km)",
        output_mean=outdir / f"forecast_probability_mean_{primary}.pdf",
        output_std=outdir / f"forecast_probability_std_{primary}.pdf",
    )
    plot_mass_fraction_bars(
        sands,
        [
            BarSeries("Truth", truth_fraction[primary_index], OBSERVED_COLOR),
            BarSeries(
                "Posterior",
                median[primary_index],
                POSTERIOR_COLOR,
                lower=lower[primary_index],
                upper=upper[primary_index],
            ),
        ],
        output=outdir / f"forecast_mass_fraction_{primary}.pdf",
        figsize=(7, 4),
    )
    plot_forecast_timeseries(
        np.array(years),
        sands,
        truth_fraction,
        median,
        lower,
        upper,
        last_survey_year=last_survey,
        output=outdir / "forecast_mass_fraction_timeseries.pdf",
    )

    print(f"\n{scenario.name} forecast (ensemble {n_used}, years {years}):")
    for layer_index, name in enumerate(sands):
        print(
            f"  {name}: year {primary} truth frac={truth_fraction[primary_index, layer_index]:.2f}"
            f"  posterior median={median[primary_index, layer_index]:.2f}"
            f"  90% [{lower[primary_index, layer_index]:.2f}, {upper[primary_index, layer_index]:.2f}]"
        )
    print(f"  -> {outdir}")
    return outdir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Forecast from a completed synthetic ABC-SMC fit."
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Experiment TOML file."
    )
    args = parser.parse_args()
    run_config = load_run_config(args.config)
    scenario = get_scenario(run_config.extras["scenario"])
    run(scenario, run_config)


if __name__ == "__main__":
    main()
