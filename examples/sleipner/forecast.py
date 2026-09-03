from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from co2ipsimulator.inference import (
    forward_snapshots,
    load_run_config,
    mass_fractions,
)
from co2ipsimulator.inference.configs import MassMeasure
from co2ipsimulator.model import build_stratigraphic_grid

from examples.synthetic.plot import (
    OBSERVED_COLOR,
    POSTERIOR_COLOR,
    BarSeries,
    _sample_prior,
    plot_mass_fraction_bars,
    plot_probability_maps,
)

from .experiment import build_forward_config, build_observed, sand_layer_names
from .fit import output_dir


def _prior_parameter_sets(run_config, n: int, rng: np.random.Generator) -> np.ndarray:
    columns = [
        _sample_prior(prior.distribution, n, rng)
        for prior in run_config.inference.pth_priors
    ]
    if run_config.inference.rate_limit_prior is not None:
        columns.append(
            _sample_prior(run_config.inference.rate_limit_prior.distribution, n, rng)
        )
    return np.column_stack(columns)


def _ensemble(config, parameter_sets, year: int, layers: tuple[str, ...], area_measure):
    shape = (config.metadata.nx, config.metadata.ny)
    probability = {name: np.zeros(shape) for name in layers}
    fractions = np.zeros((len(parameter_sets), len(layers)))
    for position, parameters in enumerate(parameter_sets):
        snapshots, _ = forward_snapshots(config, parameters, (year,))
        snapshot = snapshots[0]
        if area_measure:
            values = np.array(
                [float(snapshot.footprints[name].sum()) for name in layers]
            )
        else:
            values = np.array([snapshot.mass_per_layer_kg[name] for name in layers])
        fractions[position] = mass_fractions(values)
        for name in layers:
            probability[name] += snapshot.footprints[name].astype(float)
    for name in layers:
        probability[name] /= len(parameter_sets)
    return probability, fractions


def run(run_config) -> Path:
    config = build_forward_config(run_config)
    grid = build_stratigraphic_grid(
        config.depth_surfaces, config.layer_stack, config.metadata
    )
    layers = sand_layer_names()
    years = tuple(int(y) for y in run_config.extras.get("snapshot_years", (2010, 2023)))
    outdir = output_dir(run_config)
    samples = np.load(outdir / "abc_posterior_parameter_samples.npy")

    area_measure = run_config.summary.mass_measure is MassMeasure.FOOTPRINT_AREA
    ylabel = "Footprint-area fraction" if area_measure else "Mass fraction"

    _, _, _, observations = build_observed(config, run_config)
    observed = {obs.year: obs for obs in observations}

    ensemble = int(run_config.extras.get("posterior_ensemble_samples", 100))
    rng = np.random.default_rng(int(run_config.pymc.seed))
    n_used = min(ensemble, samples.shape[0])
    indices = rng.choice(samples.shape[0], n_used, replace=False)
    posterior_parameter_sets = samples[indices]
    prior_parameter_sets = _prior_parameter_sets(run_config, n_used, rng)

    def stats(fraction):
        return (
            np.quantile(fraction, 0.5, axis=0),
            np.quantile(fraction, 0.05, axis=0),
            np.quantile(fraction, 0.95, axis=0),
        )

    print(f"\nsleipner forecast (ensemble {n_used}, years {years}):")
    for year in years:
        observation = observed[year]
        observed_fraction = mass_fractions(observation.mass_per_layer_kg)
        observed_footprints = {
            name: observation.footprints[index].astype(float)
            for index, name in enumerate(observation.layer_names)
        }

        probability, posterior_fraction = _ensemble(
            config, posterior_parameter_sets, year, layers, area_measure
        )
        _, prior_fraction = _ensemble(
            config, prior_parameter_sets, year, layers, area_measure
        )

        post_med, post_lo, post_hi = stats(posterior_fraction)
        prior_med, prior_lo, prior_hi = stats(prior_fraction)

        np.save(
            outdir / f"forecast_posterior_probability_{year}.npy",
            np.stack([probability[name] for name in layers]),
        )
        np.save(
            outdir / f"forecast_posterior_mass_fraction_{year}.npy",
            posterior_fraction,
        )

        plot_probability_maps(
            probability,
            config.metadata,
            grid=grid,
            depth_surfaces=config.depth_surfaces,
            reference_footprints=observed_footprints,
            layer_order=layers,
            ncols=3,
            xlabel="UTM easting (km)",
            ylabel="UTM northing (km)",
            output_mean=outdir / f"posterior_plume_probability_mean_{year}.pdf",
            output_std=outdir / f"posterior_plume_probability_std_{year}.pdf",
        )
        plot_mass_fraction_bars(
            layers,
            [
                BarSeries("Observed", observed_fraction, OBSERVED_COLOR),
                # BarSeries("Prior", prior_med, PRIOR_COLOR, lower=prior_lo, upper=prior_hi),
                BarSeries(
                    "Posterior",
                    post_med,
                    POSTERIOR_COLOR,
                    lower=post_lo,
                    upper=post_hi,
                ),
            ],
            ylabel=ylabel,
            output=outdir / f"posterior_mass_fraction_{year}.pdf",
        )

        print(f"  {year}:")
        for index, name in enumerate(layers):
            print(
                f"    {name}: observed={observed_fraction[index]:.2f}  "
                f"posterior median={post_med[index]:.2f}  "
                f"90% [{post_lo[index]:.2f}, {post_hi[index]:.2f}]"
            )
    print(f"  -> {outdir}")
    return outdir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Forecast from a completed Sleipner ABC-SMC fit."
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Experiment TOML file."
    )
    args = parser.parse_args()
    run(load_run_config(args.config))


if __name__ == "__main__":
    main()
