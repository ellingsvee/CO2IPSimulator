from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from co2ipsimulator.inference import load_run_config, mass_fractions
from co2ipsimulator.model import build_stratigraphic_grid

from examples.synthetic.experiment import prior_distributions_and_log
from examples.synthetic.plot import (
    OBSERVED_COLOR,
    POSTERIOR_COLOR,
    BarSeries,
    plot_mass_fraction_bars,
    plot_probability_maps,
    plot_recovery,
)

from .experiment import build_forward_config, build_observed, sand_layer_names
from .posterior_correlation import write_posterior_correlation

DEFAULT_CONFIG = Path("examples/sleipner/configs/abc.toml")


def _saved_forecast_years(output_dir: Path) -> tuple[int, ...]:
    years = []
    for path in output_dir.glob("forecast_posterior_probability_*.npy"):
        try:
            years.append(int(path.stem.rsplit("_", 1)[-1]))
        except ValueError:
            continue
    return tuple(sorted(set(years)))


def _load_names(output_dir: Path) -> tuple[str, ...]:
    return tuple((output_dir / "abc_parameter_names.txt").read_text().splitlines())


def replot(output_dir: Path, *, config_path: Path = DEFAULT_CONFIG) -> None:
    output_dir = Path(output_dir)
    years = _saved_forecast_years(output_dir)
    base_config = load_run_config(config_path)
    run_config = replace(
        base_config,
        extras={
            **base_config.extras,
            "output_dir": str(output_dir),
            "snapshot_years": years or base_config.extras.get("snapshot_years", ()),
        },
    )

    samples = np.load(output_dir / "abc_posterior_parameter_samples.npy")
    distributions, is_log = prior_distributions_and_log(run_config)
    plot_recovery(
        _load_names(output_dir),
        samples,
        distributions,
        is_log,
        ncols=3,
        output=output_dir / "abc_posterior_hist.pdf",
        threshold_xlim=(
            (0.0, 200.0)
            if run_config.inference.rate_limit_prior is None
            else (0.0, 150.0)
        ),
    )
    write_posterior_correlation(
        samples,
        _load_names(output_dir),
        is_log,
        output_dir=output_dir,
    )

    if not years:
        print(
            "[replot] wrote posterior plots only; no forecast probability arrays in "
            f"{output_dir}"
        )
        return

    forward_config = build_forward_config(run_config)
    grid = build_stratigraphic_grid(
        forward_config.depth_surfaces,
        forward_config.layer_stack,
        forward_config.metadata,
    )
    layers = sand_layer_names()
    _, _, _, observations = build_observed(forward_config, run_config)
    observed = {obs.year: obs for obs in observations}
    ylabel = "Area fraction"

    for year in years:
        probability_stack = np.load(
            output_dir / f"forecast_posterior_probability_{year}.npy"
        )
        posterior_fraction = np.load(
            output_dir / f"forecast_posterior_mass_fraction_{year}.npy"
        )
        probability = {
            name: probability_stack[index] for index, name in enumerate(layers)
        }
        post_med = np.quantile(posterior_fraction, 0.5, axis=0)
        post_lo = np.quantile(posterior_fraction, 0.05, axis=0)
        post_hi = np.quantile(posterior_fraction, 0.95, axis=0)
        observation = observed[year]
        observed_fraction = mass_fractions(observation.mass_per_layer_kg)
        observed_footprints = {
            name: observation.footprints[index].astype(float)
            for index, name in enumerate(observation.layer_names)
        }

        plot_probability_maps(
            probability,
            forward_config.metadata,
            grid=grid,
            depth_surfaces=forward_config.depth_surfaces,
            reference_footprints=observed_footprints,
            layer_order=layers,
            ncols=3,
            xlabel="UTM easting (km)",
            ylabel="UTM northing (km)",
            output_mean=output_dir / f"posterior_plume_probability_mean_{year}.pdf",
            output_std=output_dir / f"posterior_plume_probability_std_{year}.pdf",
            figsize=(8, 12),
        )

        figsize = (7, 3.5)
        plot_mass_fraction_bars(
            layers,
            [
                BarSeries("Observed", observed_fraction, OBSERVED_COLOR),
                BarSeries(
                    "Posterior",
                    post_med,
                    POSTERIOR_COLOR,
                    lower=post_lo,
                    upper=post_hi,
                ),
            ],
            ylabel=ylabel,
            figsize=figsize,
            output=output_dir / f"posterior_mass_fraction_{year}.pdf",
        )
    print(
        f"[replot] regenerated posterior correlation, histogram, and forecast plots in {output_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate Sleipner posterior and forecast plots from saved arrays."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/sleipner/output/abc"),
        help="Directory containing saved abc/forecast .npy products.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Run config used for priors, labels, observations, and topography.",
    )
    args = parser.parse_args()
    replot(args.output_dir, config_path=args.config)


if __name__ == "__main__":
    main()
