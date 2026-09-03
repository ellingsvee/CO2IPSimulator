from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from co2ipsimulator.inference import RunConfig, forward_snapshots, load_run_config
from co2ipsimulator.model import build_stratigraphic_grid

from .experiment import (
    build_forward_config,
    calibration_years,
    prior_distributions_and_log,
    sand_layer_names,
    truth_forward,
    truth_parameters,
)
from .plot import (
    OBSERVED_COLOR,
    POSTERIOR_COLOR,
    BarSeries,
    plot_forecast_timeseries,
    plot_mass_fraction_bars,
    plot_probability_maps,
    plot_recovery,
)
from .scenarios import get_scenario

CONFIG_ROOT = Path("examples/synthetic/configs")


def _infer_config(output_dir: Path) -> Path:
    for name in (output_dir.name, output_dir.parent.name):
        candidate = CONFIG_ROOT / f"{name}.toml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "could not infer config from output directory; pass --config explicitly"
    )


def _load_names(output_dir: Path) -> tuple[str, ...]:
    return tuple((output_dir / "abc_parameter_names.txt").read_text().splitlines())


def _primary_year(output_dir: Path, fallback: int) -> int:
    for path in sorted(output_dir.glob("forecast_posterior_probability_*.npy")):
        try:
            return int(path.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
    return int(fallback)


def replot_saved(output_dir: Path, run_config: RunConfig) -> None:
    """Regenerate plots in one saved run using its fully resolved config."""
    output_dir = Path(output_dir)
    scenario = get_scenario(run_config.extras["scenario"])
    layers = sand_layer_names(scenario)
    truth = truth_parameters(scenario, run_config)

    samples = np.load(output_dir / "abc_posterior_parameter_samples.npy")
    distributions, is_log = prior_distributions_and_log(run_config)
    plot_recovery(
        _load_names(output_dir),
        samples,
        distributions,
        is_log,
        truth=truth,
        output=output_dir / "abc_recovery_hist.pdf",
        panel_size=(3.5, 3.5),
    )

    years_path = output_dir / "forecast_years.npy"
    posterior_path = output_dir / "forecast_posterior_mass_fraction.npy"
    truth_path = output_dir / "forecast_truth_mass_fraction.npy"
    if not (years_path.exists() and posterior_path.exists() and truth_path.exists()):
        print(f"[replot] wrote histogram only; no forecast arrays in {output_dir}")
        return

    years = tuple(int(y) for y in np.load(years_path))
    primary = _primary_year(output_dir, int(scenario.forecast_year))
    primary_index = years.index(primary)
    posterior_fraction = np.load(posterior_path)
    truth_fraction = np.load(truth_path)
    median = np.quantile(posterior_fraction, 0.5, axis=0)
    lower = np.quantile(posterior_fraction, 0.05, axis=0)
    upper = np.quantile(posterior_fraction, 0.95, axis=0)

    config = build_forward_config(scenario, run_config, max(years))
    grid = build_stratigraphic_grid(
        config.depth_surfaces,
        config.layer_stack,
        config.metadata,
    )
    probability_stack = np.load(
        output_dir / f"forecast_posterior_probability_{primary}.npy"
    )
    probability = {name: probability_stack[index] for index, name in enumerate(layers)}
    truth_config, truth_dgp = truth_forward(scenario, run_config, config)
    truth_snapshot = forward_snapshots(truth_config, truth_dgp, (primary,))[0][0]
    truth_footprints = {
        name: truth_snapshot.footprints[name].astype(float) for name in layers
    }

    plot_probability_maps(
        probability,
        config.metadata,
        grid=grid,
        depth_surfaces=config.depth_surfaces,
        reference_footprints=truth_footprints,
        layer_order=layers,
        xlabel="x (km)",
        ylabel="y (km)",
        output_mean=output_dir / f"forecast_probability_mean_{primary}.pdf",
        output_std=output_dir / f"forecast_probability_std_{primary}.pdf",
        figsize=(12, 3),
    )
    plot_mass_fraction_bars(
        layers,
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
        output=output_dir / f"forecast_mass_fraction_{primary}.pdf",
        figsize=(7, 4),
    )
    plot_forecast_timeseries(
        np.asarray(years),
        layers,
        truth_fraction,
        median,
        lower,
        upper,
        last_survey_year=max(calibration_years(scenario, run_config)),
        output=output_dir / "forecast_mass_fraction_timeseries.pdf",
    )
    print(f"[replot] regenerated histogram and forecast plots in {output_dir}")


def replot(output_dir: Path, *, config_path: Path | None = None) -> None:
    output_dir = Path(output_dir)
    if config_path is None:
        config_path = _infer_config(output_dir)
    replot_saved(output_dir, load_run_config(config_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate synthetic posterior and forecast plots from saved arrays."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/synthetic/output/dome/run"),
        help="Directory containing saved abc/forecast .npy products.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Run config used for priors, labels, truth, and topography.",
    )
    args = parser.parse_args()
    replot(args.output_dir, config_path=args.config)


if __name__ == "__main__":
    main()
