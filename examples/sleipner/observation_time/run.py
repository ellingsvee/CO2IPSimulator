from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from co2ipsimulator.inference import (
    RunConfig,
    forward_snapshots,
    load_run_config,
    mass_fractions,
)
from co2ipsimulator.plotting import pretty_layer_name, set_year_ticks

from examples.synthetic.experiment import prior_distributions_and_log
from examples.synthetic.plot import (
    PRIOR_COLOR,
    _grid_axes,
    _parameter_axis_label,
    _parameter_title,
    _sample_prior,
    _save,
)

from ..config import ANNUAL_RATES_MT
from ..experiment import START_YEAR, build_forward_config, sand_layer_names
from ..fit import run as fit_run
from .experiment import (
    ObservationTimeModel,
    experiment_root,
    model_configs,
)

KG_PER_MT = 1.0e9
COLORS = ("#377eb8", "#e41a1c")


@dataclass(frozen=True)
class ModelResult:
    model: ObservationTimeModel
    parameter_names: tuple[str, ...]
    parameter_is_log: tuple[bool, ...]
    samples: np.ndarray
    forecast_years: np.ndarray
    forecast_fractions: np.ndarray


def forecast_years(run_config: RunConfig) -> tuple[int, ...]:
    end = int(run_config.extras.get("forecast_end_year", 2030))
    if end < START_YEAR:
        raise ValueError(f"forecast_end_year must be at least {START_YEAR}")
    return tuple(range(START_YEAR, end + 1))


def forecast_annual_rates_mt(run_config: RunConfig) -> tuple[float, ...]:
    """Injection history extended with a declared constant future rate."""
    years = forecast_years(run_config)
    required = len(years)
    historical = tuple(float(value) for value in ANNUAL_RATES_MT)
    if required <= len(historical):
        return historical[:required]
    future_rate = float(run_config.extras.get("future_annual_rate_mt", historical[-1]))
    return historical + (future_rate,) * (required - len(historical))


def _forecast_ensemble(
    run_config: RunConfig, samples: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    years = forecast_years(run_config)
    annual = tuple(rate * KG_PER_MT for rate in forecast_annual_rates_mt(run_config))
    config = replace(build_forward_config(run_config), annual_masses_kg=annual)
    layers = sand_layer_names()
    requested = int(run_config.extras.get("posterior_ensemble_samples", 100))
    n_used = min(requested, samples.shape[0])
    rng = np.random.default_rng(int(run_config.pymc.seed))
    indices = rng.choice(samples.shape[0], n_used, replace=False)
    fractions = np.zeros((n_used, len(years), len(layers)), dtype=float)

    for ensemble_index, sample_index in enumerate(indices):
        snapshots, _ = forward_snapshots(config, samples[sample_index], years)
        by_year = {snapshot.year: snapshot for snapshot in snapshots}
        for year_index, year in enumerate(years):
            mass = np.array(
                [by_year[year].mass_per_layer_kg[name] for name in layers],
                dtype=float,
            )
            fractions[ensemble_index, year_index] = mass_fractions(mass)
    return np.asarray(years), fractions


def _load_and_forecast(
    model: ObservationTimeModel, run_config: RunConfig
) -> ModelResult:
    outdir = Path(run_config.extras["output_dir"])
    samples = np.load(outdir / "abc_posterior_parameter_samples.npy")
    years, fractions = _forecast_ensemble(run_config, samples)
    np.save(outdir / "forecast_years.npy", years)
    np.save(outdir / "forecast_posterior_mass_fraction.npy", fractions)
    _, is_log = prior_distributions_and_log(run_config)
    return ModelResult(
        model=model,
        parameter_names=run_config.inference.parameter_names,
        parameter_is_log=is_log,
        samples=samples,
        forecast_years=years,
        forecast_fractions=fractions,
    )


def _plot_forecasts(results: tuple[ModelResult, ...], output: Path) -> None:
    layers = sand_layer_names()
    fig, axes = _grid_axes(len(layers), 3, (3.6, 3.4))
    for ax, layer_index in zip(axes, range(len(layers))):
        for result, color in zip(results, COLORS, strict=True):
            values = result.forecast_fractions[:, :, layer_index]
            lower, median, upper = np.quantile(values, (0.05, 0.5, 0.95), axis=0)
            # ax.fill_between(
            #     result.forecast_years,
            #     lower,
            #     upper,
            #     color=color,
            #     alpha=0.2,
            #     linewidth=0,
            #     zorder=1,
            # )
            ax.plot(
                result.forecast_years,
                median,
                color=color,
                linewidth=3.0,
                label=result.model.label,
                zorder=3,
            )
        ax.axvline(2010, color="0.35", linestyle=":", linewidth=1.5, zorder=0)
        ax.axvline(2023, color="0.35", linestyle="--", linewidth=1.5, zorder=0)
        ax.set_title(pretty_layer_name(layers[layer_index]))
        ax.set_xlabel("Year")
        ax.set_ylim(0.0, 1.0)
        set_year_ticks(ax, results[0].forecast_years, start=START_YEAR)
    axes[0].set_ylabel("Mass fraction")
    handles = [
        Line2D([], [], color=color, linewidth=3.0, label=result.model.label)
        for result, color in zip(results, COLORS, strict=True)
    ]
    handles.extend(
        (
            Line2D([], [], color="0.35", linestyle=":", label="2010 observation"),
            Line2D([], [], color="0.35", linestyle="--", label="2023 observation"),
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
    )
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.set(rect=(0.0, 0.11, 1.0, 1.0))
    _save(fig, output)


def _plot_forecasts_individual_axes(
    results: tuple[ModelResult, ...], output: Path
) -> None:
    """Forecast comparison with a useful range for each sand-unit panel."""
    layers = sand_layer_names()
    # fig, axes = _grid_axes(len(layers), 3, (4.0, 3.4))
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(12.0, 10.0),
        constrained_layout=True,
        squeeze=True,
    )
    axes = axes.ravel()

    start_year = START_YEAR
    years = results[0].forecast_years
    shown = years >= start_year

    for ax, layer_index in zip(axes, range(len(layers))):
        largest_upper = 0.0
        for result, color in zip(results, COLORS, strict=True):
            values = result.forecast_fractions[:, :, layer_index]
            lower, median, upper = np.quantile(values, (0.05, 0.5, 0.95), axis=0)
            largest_upper = max(largest_upper, float(upper[shown].max()))
            plot_years = result.forecast_years[shown]
            ax.fill_between(
                plot_years,
                lower[shown],
                upper[shown],
                color=color,
                alpha=0.2,
                linewidth=0,
                zorder=1,
            )
            ax.plot(
                plot_years,
                lower[shown],
                color=color,
                linestyle="--",
                linewidth=3.5,
                zorder=2,
            )
            ax.plot(
                plot_years,
                upper[shown],
                color=color,
                linestyle="--",
                linewidth=3.5,
                zorder=2,
            )
            ax.plot(
                plot_years,
                median[shown],
                color=color,
                linewidth=3.5,
                label=result.model.label,
                zorder=3,
            )
        ax.set_title(pretty_layer_name(layers[layer_index]))
        ax.set_xlabel("Year")
        upper_limit = max(0.05, np.ceil(largest_upper / 0.05) * 0.05)
        ax.set_ylim(0.0, upper_limit)
        set_year_ticks(ax, years, start=start_year)
    axes[0].set_ylabel("Mass fraction")
    handles = [
        Line2D(
            [], [], color="0.35", linestyle="--", linewidth=3.5, label=r"90\% interval"
        )
    ]
    handles.extend(
        Line2D([], [], color=color, linewidth=3.5, label=result.model.label)
        for result, color in zip(results, COLORS, strict=True)
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.12),
        ncol=len(handles),
        frameon=False,
    )
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.set(rect=(0.0, 0.11, 1.0, 1.0))
    _save(fig, output)


def _interval_width(fractions: np.ndarray) -> np.ndarray:
    lower, upper = np.quantile(fractions, (0.05, 0.95), axis=0)
    return upper - lower


def _plot_forecast_uncertainty(results: tuple[ModelResult, ...], output: Path) -> None:
    """Show the part of the comparison hidden by overlapping forecast bands."""
    layers = sand_layer_names()
    fig, axes = _grid_axes(len(layers), 3, (3.6, 3.4))
    widths = tuple(_interval_width(result.forecast_fractions) for result in results)
    years = results[0].forecast_years
    comparison = years >= 2010
    largest = max(float(width[comparison].max()) for width in widths)
    shared_upper = max(0.05, np.ceil(largest / 0.05) * 0.05)

    for ax, layer_index in zip(axes, range(len(layers))):
        for result, color, width in zip(results, COLORS, widths, strict=True):
            ax.plot(
                result.forecast_years,
                width[:, layer_index],
                color=color,
                linewidth=3.0,
                label=result.model.label,
                zorder=3,
            )
        ax.axvline(2010, color="0.35", linestyle=":", linewidth=1.5, zorder=0)
        ax.axvline(2023, color="0.35", linestyle="--", linewidth=1.5, zorder=0)
        ax.set_title(pretty_layer_name(layers[layer_index]))
        ax.set_xlabel("Year")
        ax.set_ylim(0.0, shared_upper)
        set_year_ticks(ax, years, start=2010)
    axes[0].set_ylabel(r"Width of 90\% interval")
    handles = [
        Line2D([], [], color=color, linewidth=3.0, label=result.model.label)
        for result, color in zip(results, COLORS, strict=True)
    ]
    handles.extend(
        (
            Line2D([], [], color="0.35", linestyle=":", label="2010 observation"),
            Line2D([], [], color="0.35", linestyle="--", label="2023 observation"),
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
    )
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.set(rect=(0.0, 0.11, 1.0, 1.0))
    _save(fig, output)


def _plot_posteriors(
    results: tuple[ModelResult, ...], run_config: RunConfig, output: Path
) -> None:
    distributions, is_log = prior_distributions_and_log(run_config)
    names = results[0].parameter_names
    rng = np.random.default_rng(0)
    fig, axes = _grid_axes(len(names), 3, (3.5, 3.5))
    for index, (ax, name, distribution, log_scale) in enumerate(
        zip(axes, names, distributions, is_log, strict=True)
    ):
        prior = _sample_prior(distribution, 40_000, rng)
        posterior = [result.samples[:, index] for result in results]
        if log_scale:
            prior = np.log10(prior)
            posterior = [np.log10(values) for values in posterior]
        values = np.concatenate([prior, *posterior])
        lo, hi = np.quantile(values, (0.001, 0.999))
        if name.startswith("Shale_"):
            lo, hi = 0.0, 150.0
        bins = np.linspace(lo, hi, 40)
        ax.hist(
            prior,
            bins=bins,
            density=True,
            histtype="step",
            color=PRIOR_COLOR,
            linestyle="--",
            linewidth=2.5,
            label="Prior",
        )
        for result, color, values_for_model in zip(
            results, COLORS, posterior, strict=True
        ):
            ax.hist(
                values_for_model,
                bins=bins,
                density=True,
                histtype="step",
                color=color,
                linewidth=2.2,
                label=result.model.label,
            )
        ax.set_xlim(lo, hi)
        ax.set_yticks([])
        ax.set_xlabel(_parameter_axis_label(name, log_scale))
        ax.set_title(_parameter_title(name))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(handles),
        frameon=False,
    )
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.set(rect=(0.0, 0.10, 1.0, 1.0))
    _save(fig, output)


def _write_estimates(results: tuple[ModelResult, ...], output: Path) -> None:
    rows = []
    for parameter_index, name in enumerate(results[0].parameter_names):
        quantiles = [
            np.quantile(result.samples[:, parameter_index], (0.05, 0.5, 0.95))
            for result in results
        ]
        rows.append(
            {
                "parameter": name,
                "both_lower": quantiles[0][0],
                "both_median": quantiles[0][1],
                "both_upper": quantiles[0][2],
                "2010_only_lower": quantiles[1][0],
                "2010_only_median": quantiles[1][1],
                "2010_only_upper": quantiles[1][2],
                "median_change_2010_only_minus_both": quantiles[1][1] - quantiles[0][1],
            }
        )
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(
    base: RunConfig, *, reuse_existing: bool = False, plot_only: bool = False
) -> Path:
    configs = model_configs(base)
    if not plot_only:
        for model, run_config in configs:
            sample_file = (
                Path(run_config.extras["output_dir"])
                / "abc_posterior_parameter_samples.npy"
            )
            if reuse_existing and sample_file.exists():
                print(f"\n=== {model.label}: reusing {sample_file} ===")
            else:
                print(f"\n=== Fitting {model.label} ===")
                fit_run(run_config)

    results = tuple(
        _load_and_forecast(model, run_config) for model, run_config in configs
    )
    root = experiment_root(base)
    root.mkdir(parents=True, exist_ok=True)
    _plot_forecasts(results, root / "forecast_mass_fraction_timeseries.pdf")
    _plot_forecasts_individual_axes(
        results, root / "forecast_mass_fraction_timeseries_individual_axes.pdf"
    )
    _plot_forecast_uncertainty(
        results, root / "forecast_mass_fraction_uncertainty_timeseries.pdf"
    )
    _plot_posteriors(results, base, root / "posterior_parameter_comparison.pdf")
    _write_estimates(results, root / "parameter_estimate_comparison.csv")
    metadata = {
        "forecast_start_year": START_YEAR,
        "forecast_end_year": int(results[0].forecast_years[-1]),
        "historical_injection_end_year": START_YEAR + len(ANNUAL_RATES_MT) - 1,
        "future_annual_rate_mt": float(
            base.extras.get("future_annual_rate_mt", ANNUAL_RATES_MT[-1])
        ),
        "uncertainty_interval": [0.05, 0.95],
        "posterior_ensemble_samples": int(results[0].forecast_fractions.shape[0]),
        "models": [
            {"key": result.model.key, "snapshot_years": result.model.snapshot_years}
            for result in results
        ],
    }
    (root / "forecast_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nObservation-time comparison -> {root}")
    return root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Sleipner fits using different observation times."
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Experiment TOML file."
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse completed fits and run only missing work.",
    )
    parser.add_argument(
        "--plot-only", action="store_true", help="Plot existing completed fits."
    )
    args = parser.parse_args()
    run(
        load_run_config(args.config),
        reuse_existing=args.reuse_existing,
        plot_only=args.plot_only,
    )


if __name__ == "__main__":
    main()
