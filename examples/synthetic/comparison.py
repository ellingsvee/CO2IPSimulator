from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

from co2ipsimulator.inference import PriorDistribution, RunConfig
from co2ipsimulator.plotting import pretty_layer_name as _pretty_name
from co2ipsimulator.plotting import set_year_ticks

from .experiment import prior_distributions_and_log, truth_parameters
from .fit import output_dir
from .fit import run as fit_run
from .forecast import run as forecast_run
from .plot import (
    PRIOR_COLOR,
    TRUTH_COLOR,
    _grid_axes,
    _layer_key,
    _parameter_axis_label,
    _parameter_title,
    _sample_prior,
    _save,
)
from .scenarios import Scenario

OUTPUT_ROOT = Path("examples/synthetic/output")

# Set1 without its orange, which the truth already owns, and without its red.
_VARIANT_COLOR_INDICES = (1, 2, 3, 6, 7, 0)
_OUTLINE_STYLES = ("-", "--", "-.", ":")


def line_styles(count: int) -> list[str]:
    """Distinguish a handful of arms that may lie exactly on top of each other.

    A sweep with many arms is ordered by colour instead, where a cycling dash
    pattern would only add noise.
    """
    if count > len(_OUTLINE_STYLES):
        return ["-"] * count
    return list(_OUTLINE_STYLES[:count])


def variant_colors(count: int) -> list:
    scheme = colormaps["Set1"]
    return [scheme(_VARIANT_COLOR_INDICES[i % 6]) for i in range(count)]


def experiment_root(base: RunConfig, default: str) -> str:
    """Where this run writes, so a smoke config cannot overwrite a production one."""
    return str(base.extras.get("experiment_root", default))


@dataclass(frozen=True)
class Variant:
    key: str
    label: str


@dataclass(frozen=True)
class VariantResult:
    variant: Variant
    parameter_names: tuple[str, ...]
    parameter_is_log: tuple[bool, ...]
    prior_distributions: tuple[PriorDistribution, ...]
    samples: np.ndarray
    truth: np.ndarray
    forecast_years: np.ndarray
    forecast_fraction: np.ndarray
    truth_fraction: np.ndarray

    @property
    def label(self) -> str:
        return self.variant.label

    def index_of(self, name: str) -> int | None:
        if name not in self.parameter_names:
            return None
        return self.parameter_names.index(name)


def run_variants(
    scenario: Scenario,
    variant_configs: dict[str, tuple[Variant, RunConfig]],
) -> None:
    for variant, run_config in variant_configs.values():
        print(f"\n=== {variant.label} ===")
        fit_run(scenario, run_config)
        forecast_run(scenario, run_config)


def load_variant_result(
    scenario: Scenario, variant: Variant, run_config: RunConfig
) -> VariantResult | None:
    outdir = output_dir(scenario, run_config)
    samples = outdir / "abc_posterior_parameter_samples.npy"
    forecast = outdir / "forecast_posterior_mass_fraction.npy"
    if not (samples.exists() and forecast.exists()):
        return None
    distributions, is_log = prior_distributions_and_log(run_config)
    return VariantResult(
        variant=variant,
        parameter_names=run_config.inference.parameter_names,
        parameter_is_log=is_log,
        prior_distributions=distributions,
        samples=np.load(samples),
        truth=truth_parameters(scenario, run_config),
        forecast_years=np.load(outdir / "forecast_years.npy"),
        forecast_fraction=np.load(forecast),
        truth_fraction=np.load(outdir / "forecast_truth_mass_fraction.npy"),
    )


def load_variant_results(
    scenario: Scenario, variant_configs: dict[str, tuple[Variant, RunConfig]]
) -> list[VariantResult]:
    results = [
        result
        for variant, run_config in variant_configs.values()
        if (result := load_variant_result(scenario, variant, run_config)) is not None
    ]
    if not results:
        raise SystemExit("no variant outputs found; run without --plot-only first")
    return results


def union_parameters(
    results: list[VariantResult],
) -> tuple[tuple[str, ...], tuple[bool, ...]]:
    names: list[str] = []
    is_log: list[bool] = []
    for result in results:
        for name, log in zip(result.parameter_names, result.parameter_is_log):
            if name not in names:
                names.append(name)
                is_log.append(log)
    return tuple(names), tuple(is_log)


def _reference(results: list[VariantResult], name: str):
    for result in results:
        index = result.index_of(name)
        if index is not None:
            return result.prior_distributions[index], float(result.truth[index])
    raise KeyError(name)


def _panel_limits(
    values: list[np.ndarray],
    prior: np.ndarray,
    truth: float | None,
    minimum_span: float,
) -> tuple[float, float]:
    """Enough room to show the posteriors, the truth and some prior context."""
    lo = min(float(v.min()) for v in values)
    hi = max(float(v.max()) for v in values)
    if truth is not None:
        lo, hi = min(lo, truth), max(hi, truth)
    pad = max(0.35 * (hi - lo), 0.5 * (minimum_span - (hi - lo)), 1.0e-9)
    return (
        max(lo - pad, float(np.quantile(prior, 0.001))),
        min(hi + pad, float(np.quantile(prior, 0.999))),
    )


def parameter_limits(
    result_sets: list[list[VariantResult]], name: str
) -> tuple[float, float] | None:
    """One x-range wide enough for every arm of every set.

    Two figures drawn from the same prior are only comparable if the parameter
    is on the same axis in both; per-figure limits zoom each one to its own
    posterior, which makes one prior look like two different ones.
    """
    rng = np.random.default_rng(0)
    spans = []
    for results in result_sets:
        names, is_log = union_parameters(results)
        if name not in names:
            continue
        log = is_log[names.index(name)]
        distribution, truth_value = _reference(results, name)
        prior = _sample_prior(distribution, 40000, rng)
        present = []
        for result in results:
            index = result.index_of(name)
            if index is not None:
                present.append(result.samples[:, index].astype(float))
        if not present:
            continue
        if log:
            prior = np.log10(prior)
            present = [np.log10(values) for values in present]
            truth_value = np.log10(truth_value)
        truth = float(truth_value) if np.isfinite(truth_value) else None
        spans.append(_panel_limits(present, prior, truth, 2.0 if log else 0.0))
    if not spans:
        return None
    return min(span[0] for span in spans), max(span[1] for span in spans)


def plot_posterior_comparison(
    results: list[VariantResult],
    *,
    colors: list | None = None,
    ncols: int | None = None,
    fixed_limits: dict[str, tuple[float, float]] | None = None,
    output: Path | str,
    panel_size: tuple[float, float] = (3.5, 3.5),
) -> None:
    """Posterior of every variant on one axis per parameter.

    Laid out like ``abc_recovery_hist.pdf``: prior as a dashed black line, truth
    as a dashed reference line, and one histogram per arm on top. ``fixed_limits``
    pins a parameter's x-range, so the same panel can be compared across figures.
    """
    rng = np.random.default_rng(0)
    names, is_log = union_parameters(results)
    colors = variant_colors(len(results)) if colors is None else colors

    fig, axes = _grid_axes(len(names), ncols, panel_size)
    fig.set_constrained_layout_pads(h_pad=0.10, hspace=0.04)
    for position, (name, ax) in enumerate(zip(names, axes)):
        distribution, truth_value = _reference(results, name)
        prior = _sample_prior(distribution, 40000, rng)
        posteriors = []
        for result in results:
            index = result.index_of(name)
            posteriors.append(
                None if index is None else result.samples[:, index].astype(float)
            )
        if is_log[position]:
            prior = np.log10(prior)
            posteriors = [None if p is None else np.log10(p) for p in posteriors]
            truth_value = np.log10(truth_value)
        truth_value = None if not np.isfinite(truth_value) else float(truth_value)

        present = [p for p in posteriors if p is not None]
        pinned = None if fixed_limits is None else fixed_limits.get(name)
        if pinned is not None:
            lo, hi = pinned
        elif name.startswith("Shale_"):
            lo, hi = 0.0, 200.0
        else:
            lo, hi = _panel_limits(
                present, prior, truth_value, 2.0 if is_log[position] else 0.0
            )
        grid = np.linspace(lo, hi, 400)

        ax.plot(
            grid,
            gaussian_kde(prior)(grid),
            color=PRIOR_COLOR,
            linestyle="--",
            linewidth=4,
            label="Prior",
            zorder=2,
        )
        bins = np.linspace(lo, hi, 40)
        # Every fill first, then every outline, so a bar drawn later cannot bury
        # an earlier arm: its outline is redrawn on top at full opacity. The fills
        # are close to solid, as in ``abc_recovery_hist.pdf``.
        for values, color in zip(posteriors, colors):
            if values is None:
                continue
            ax.hist(
                values,
                bins=bins,
                density=True,
                histtype="stepfilled",
                color=color,
                alpha=0.8,
                edgecolor="none",
                zorder=3,
            )
        for values, color in zip(posteriors, colors):
            if values is None:
                continue
            ax.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                color=color,
                linewidth=2.2,
                label=None,
                zorder=4,
            )
        if truth_value is not None:
            ax.axvline(
                truth_value,
                color=TRUTH_COLOR,
                linestyle="--",
                linewidth=4,
                label="Truth",
                zorder=5,
            )
        if name.startswith("Shale_"):
            ax.set_title(_parameter_title(name), pad=10)
        ax.set_xlabel(_parameter_axis_label(name, is_log[position]), labelpad=8)
        ax.set_yticks([])
        ax.set_xlim(lo, hi)
        ax.set_ylim(bottom=0.0)

    handles = [
        Line2D([], [], color=PRIOR_COLOR, linestyle="--", linewidth=4, label="Prior")
    ]
    handles += [
        Patch(
            facecolor=to_rgba(color, 0.8),
            edgecolor=color,
            linewidth=2.2,
            label=result.label,
        )
        for result, color in zip(results, colors)
    ]
    handles.append(
        Line2D([], [], color=TRUTH_COLOR, linestyle="--", linewidth=4, label="Truth")
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(handles),
        frameon=False,
    )
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.set(rect=(0.0, 0.14, 1.0, 1.0))
    _save(fig, output)


def _layer_order(layer_names: tuple[str, ...]) -> list[int]:
    return sorted(range(len(layer_names)), key=lambda i: _layer_key(layer_names[i]))


def plot_forecast_comparison(
    results: list[VariantResult],
    layer_names: tuple[str, ...],
    last_survey_year: int,
    *,
    colors: list | None = None,
    show_interval: bool = True,
    ncols: int | None = None,
    output: Path | str,
    panel_size: tuple[float, float] = (3.6, 3.4),
) -> None:
    """Out-of-sample forecast of every variant, one panel per sand unit."""
    colors = variant_colors(len(results)) if colors is None else colors
    styles = line_styles(len(results))
    order = _layer_order(layer_names)
    fig, axes = _grid_axes(len(order), ncols, panel_size)
    years = results[0].forecast_years
    for ax, i in zip(axes, order):
        for result, color, style in zip(results, colors, styles):
            fraction = result.forecast_fraction[:, :, i]
            median = np.quantile(fraction, 0.5, axis=0)
            if show_interval:
                ax.fill_between(
                    result.forecast_years,
                    np.quantile(fraction, 0.05, axis=0),
                    np.quantile(fraction, 0.95, axis=0),
                    color=color,
                    alpha=0.22,
                    linewidth=0,
                    zorder=1,
                )
            ax.plot(
                result.forecast_years,
                median,
                color=color,
                linestyle=style,
                linewidth=3.5,
                label=result.label,
                zorder=3,
            )
        ax.plot(
            years,
            results[0].truth_fraction[:, i],
            color="black",
            linestyle="--",
            linewidth=3.5,
            label="Truth",
            zorder=4,
        )
        ax.set_title(_pretty_name(layer_names[i]))
        ax.set_xlabel("Year")
        set_year_ticks(ax, years, start=last_survey_year)
    axes[0].set_ylabel("Mass fraction")
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
        engine.set(rect=(0.0, 0.16, 1.0, 1.0))
    _save(fig, output)


def year_index(years: np.ndarray, year: int) -> int:
    return int(np.argmin(np.abs(np.asarray(years) - year)))


def forecast_skill(result: VariantResult, year: int) -> tuple[float, float, float]:
    """Accuracy, precision and coverage of one variant at one forecast year."""
    index = year_index(result.forecast_years, year)
    fraction = result.forecast_fraction[:, index, :]
    truth = result.truth_fraction[index]
    median = np.quantile(fraction, 0.5, axis=0)
    lower = np.quantile(fraction, 0.05, axis=0)
    upper = np.quantile(fraction, 0.95, axis=0)
    return (
        float(np.sqrt(np.mean((median - truth) ** 2))),
        float(np.mean(upper - lower)),
        float(np.mean((truth >= lower) & (truth <= upper))),
    )


def print_summary(results: list[VariantResult], year: int) -> None:
    names, is_log = union_parameters(results)
    header = f"  {'variant':>26}  " + "  ".join(f"{n:>12}" for n in names)
    print(header + f"  {'RMSE':>8}{'width':>8}{'cover':>7}")
    for result in results:
        cells = []
        for name, log in zip(names, is_log):
            index = result.index_of(name)
            if index is None:
                cells.append(f"{'-':>12}")
                continue
            column = result.samples[:, index]
            value = np.median(np.log10(column) if log else column)
            cells.append(f"{value:>12.3f}")
        rmse, width, coverage = forecast_skill(result, year)
        print(
            f"  {result.label:>26}  "
            + "  ".join(cells)
            + f"  {rmse:8.4f}{width:8.4f}{coverage:7.2f}"
        )
