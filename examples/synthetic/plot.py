from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from co2ipsimulator.inference import PriorDistribution
from co2ipsimulator.plotting import (
    MODEL_COLOR,
    REFERENCE_COLOR,
    birds_eye_panels,
    pretty_layer_name as _pretty_name,
    set_year_ticks,
)

POSTERIOR_COLOR = MODEL_COLOR
PRIOR_COLOR = "black"
TRUTH_COLOR = REFERENCE_COLOR
OBSERVED_COLOR = TRUTH_COLOR


def _save(fig: plt.Figure, output: Path | str) -> None:
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(output), dpi=180, bbox_inches="tight", transparent=True)
    plt.close(fig)


def _layer_key(name: str) -> int:
    digits = "".join(c for c in name if c.isdigit())
    return int(digits) if digits else 0


def _sample_prior(
    distribution: PriorDistribution, n: int, rng: np.random.Generator
) -> np.ndarray:
    return distribution.sample(rng, n)


def _grid_axes(n: int, ncols: int | None, panel: tuple[float, float]):
    columns = n if ncols is None else min(ncols, n)
    rows = (n + columns - 1) // columns
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(panel[0] * columns, panel[1] * rows),
        constrained_layout=True,
        squeeze=True,
    )
    flat = axes.ravel()
    for extra in flat[n:]:
        extra.set_visible(False)
    return fig, flat[:n]


_PARAMETER_SYMBOL = {
    "lambda_s": r"\lambda",
}


def _parameter_axis_label(name: str, is_log: bool) -> str:
    if is_log:
        return rf"$\log_{{10}}\,{_PARAMETER_SYMBOL.get(name, name)}$"
    if name == "h_det":
        return r"$h_\mathrm{det}$ [m]"
    return r"$P_\mathrm{th}$"


def _parameter_title(name: str) -> str:
    if name == "Shale_thick":
        return "Shale 8"
    if name == "h_det":
        return "Detection threshold"
    return _pretty_name(name)


def _is_threshold_parameter(name: str, is_log: bool) -> bool:
    return not is_log and (name.startswith("Shale_") or name == "Shale_thick")


def plot_recovery(
    parameter_names: tuple[str, ...],
    samples: np.ndarray,
    prior_distributions: tuple[PriorDistribution, ...],
    parameter_is_log: tuple[bool, ...],
    *,
    truth: np.ndarray | None = None,
    ncols: int | None = None,
    output: Path | str,
    panel_size: tuple[float, float] | None = None,
    threshold_xlim: tuple[float, float] = (0.0, 200.0),
) -> None:
    rng = np.random.default_rng(0)

    if panel_size is None:
        panel_size = (3.5, 3.5)

    fig, axes = _grid_axes(len(parameter_names), ncols, panel_size)
    fig.set_constrained_layout_pads(h_pad=0.10, hspace=0.04)

    for idx, (name, ax) in enumerate(zip(parameter_names, axes)):
        post = samples[:, idx]
        prior = _sample_prior(prior_distributions[idx], 40000, rng)
        truth_value = None if truth is None else float(truth[idx])
        if truth_value is not None and not np.isfinite(truth_value):
            truth_value = None
        if parameter_is_log[idx]:
            post = np.log10(post)
            prior = np.log10(prior)
            truth_value = None if truth_value is None else np.log10(truth_value)

        is_threshold_param = _is_threshold_parameter(name, parameter_is_log[idx])
        if is_threshold_param:
            lo, hi = threshold_xlim
        else:
            candidates = [
                post.min(),
                post.max(),
                np.quantile(prior, 0.005),
                np.quantile(prior, 0.995),
            ]
            if truth_value is not None:
                candidates.append(truth_value)
            lo, hi = min(candidates), max(candidates)
        grid = np.linspace(lo, hi, 200)
        ax.plot(
            grid,
            gaussian_kde(prior)(grid),
            color=PRIOR_COLOR,
            linestyle="--",
            label="Prior",
            linewidth=4,
        )
        ax.hist(
            post,
            bins=np.linspace(lo, hi, 40),
            density=True,
            color=POSTERIOR_COLOR,
            alpha=1.0,
            label="Posterior",
        )
        ax.axvline(np.median(post), color="black", label="Median", linewidth=3.5)
        if truth_value is not None:
            ax.axvline(
                truth_value,
                color=TRUTH_COLOR,
                linestyle="--",
                label="Truth",
                linewidth=4,
            )
        if is_threshold_param:
            ax.set_title(_parameter_title(name), pad=10)
        ax.set_xlabel(_parameter_axis_label(name, parameter_is_log[idx]), labelpad=8)
        ax.set_yticks([])
        ax.set_xlim(lo, hi)
        if idx == 0:
            ax.legend(
                frameon=True,
                facecolor="white",
                edgecolor="0.8",
                framealpha=1.0,
            )
    _save(fig, output)


def plot_priors(
    parameter_names: tuple[str, ...],
    prior_distributions: tuple[PriorDistribution, ...],
    parameter_is_log: tuple[bool, ...],
    *,
    truth: np.ndarray | None = None,
    ncols: int | None = None,
    output: Path | str,
) -> None:
    rng = np.random.default_rng(0)
    fig, axes = _grid_axes(len(parameter_names), ncols, (3.6, 3.4))
    for idx, (name, ax) in enumerate(zip(parameter_names, axes)):
        prior = _sample_prior(prior_distributions[idx], 40000, rng)
        truth_value = None if truth is None else float(truth[idx])
        if truth_value is not None and not np.isfinite(truth_value):
            truth_value = None
        if parameter_is_log[idx]:
            prior = np.log10(prior)
            truth_value = None if truth_value is None else np.log10(truth_value)
        grid = np.linspace(np.quantile(prior, 0.001), np.quantile(prior, 0.999), 200)
        ax.plot(grid, gaussian_kde(prior)(grid), color=PRIOR_COLOR, label="Prior")
        if truth_value is not None:
            ax.axvline(truth_value, color=TRUTH_COLOR, linestyle="--", label="Truth")
        ax.set_title(_pretty_name(name))
        ax.set_xlabel(_parameter_axis_label(name, parameter_is_log[idx]))
        ax.set_yticks([])
        if idx == 0:
            ax.legend(frameon=False)
    _save(fig, output)


def plot_probability_maps(
    probability: dict[str, np.ndarray],
    metadata,
    *,
    grid=None,
    depth_surfaces: dict[str, np.ndarray] | None = None,
    reference_footprints: dict[str, np.ndarray] | None = None,
    layer_order: tuple[str, ...] | None = None,
    ncols: int | None = None,
    xlabel: str = "x (km)",
    ylabel: str = "y (km)",
    output_mean: Path | str,
    output_std: Path | str,
    figsize: tuple[float, float] | None = None,
) -> None:
    names = sorted(probability, key=_layer_key)
    columns = len(names) if ncols is None else min(ncols, len(names))
    rows = (len(names) + columns - 1) // columns
    map_aspect = (metadata.xmax - metadata.xmin) / (metadata.ymax - metadata.ymin)

    panel_height = 3.45
    panel_width = max(2.35, panel_height * map_aspect)
    if figsize is None:
        figsize = (panel_width * columns + 0.7, panel_height * rows)
    panel_titles = {name: _pretty_name(name) for name in names}

    for field_kind, out in (("mean", output_mean), ("std", output_std)):
        fields = {}
        for name in names:
            prob = probability[name]
            if field_kind == "mean":
                fields[name], cmap, vmax = prob, "viridis", 1.0
            else:
                fields[name] = np.sqrt(np.clip(prob * (1.0 - prob), 0.0, 0.25))
                cmap, vmax = "magma", 0.5
        # label = "Probability" if field_kind == "mean" else "Std. dev."
        label = "Recurrence" if field_kind == "mean" else "Std. dev."
        fig = birds_eye_panels(
            fields,
            grid=grid,
            metadata=metadata,
            layer_order=tuple(names) if layer_order is None else layer_order,
            depth_surfaces=depth_surfaces,
            overlay_outlines=reference_footprints,
            overlay_outline_color="black",
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
            cbar_label=label,
            panel_titles=panel_titles,
            ncols=columns,
            figsize=figsize,
            mask_nonpositive=True,
            xlabel=xlabel,
            ylabel=ylabel,
            output=out,
        )
        plt.close(fig)


@dataclass(frozen=True)
class BarSeries:
    label: str
    values: np.ndarray
    color: str | tuple[float, float, float, float]
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None


def plot_mass_fraction_bars(
    layer_names: tuple[str, ...],
    series: list[BarSeries],
    *,
    ylabel: str = "Mass fraction",
    figsize: tuple[float, float] | None = None,
    output: Path | str,
) -> None:
    order = sorted(range(len(layer_names)), key=lambda i: _layer_key(layer_names[i]))
    names = [_pretty_name(layer_names[i]) for i in order]
    x = np.arange(len(names))
    width = 0.8 / len(series)

    if figsize is None:
        figsize = (1.0 * len(names), 4.0)

    fig, ax = plt.subplots(figsize=figsize)
    for k, s in enumerate(series):
        offset = (k - (len(series) - 1) / 2.0) * width
        yerr = None
        if s.lower is not None and s.upper is not None:
            values = s.values[order]
            yerr = np.vstack([values - s.lower[order], s.upper[order] - values])
        ax.bar(
            x + offset,
            s.values[order],
            width,
            color=s.color,
            edgecolor="black",
            linewidth=0.6,
            yerr=yerr,
            capsize=4,
            label=s.label,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=0, ha="center")
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle=":", alpha=1)
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="0.8",
        framealpha=1.0,
    )
    fig.tight_layout()
    _save(fig, output)


def plot_forecast_timeseries(
    years: np.ndarray,
    layer_names: tuple[str, ...],
    truth: np.ndarray,
    median: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    last_survey_year: int,
    ncols: int | None = None,
    output: Path | str,
    panel_size: tuple[float, float] | None = None,
) -> None:
    order = sorted(range(len(layer_names)), key=lambda i: _layer_key(layer_names[i]))

    if panel_size is None:
        panel_size = (3.6, 3.4)

    fig, axes = _grid_axes(len(order), ncols, panel_size)
    for ax, i in zip(axes, order):
        ax.fill_between(
            years,
            lower[:, i],
            upper[:, i],
            color=POSTERIOR_COLOR,
            alpha=0.35,
            linewidth=0,
            # linewidth=3.5,
            # label=r"90\% interval",
            zorder=1,
        )
        ax.plot(
            years,
            lower[:, i],
            color=POSTERIOR_COLOR,
            linestyle="--",
            linewidth=3.5,
            label=r"90\% interval",
            zorder=2,
        )
        ax.plot(
            years,
            upper[:, i],
            color=POSTERIOR_COLOR,
            linestyle="--",
            linewidth=3.5,
            zorder=2,
        )
        ax.plot(
            years,
            median[:, i],
            color=POSTERIOR_COLOR,
            linewidth=3.5,
            label="Posterior",
            zorder=3,
        )
        ax.plot(
            years,
            truth[:, i],
            color="black",
            linestyle="--",
            linewidth=3.5,
            label="Truth",
            zorder=3,
        )
        ax.set_title(_pretty_name(layer_names[i]))
        ax.set_xlabel("Year")
        set_year_ticks(ax, years, start=last_survey_year)
    axes[0].set_ylabel("Mass fraction")
    axes[0].legend(frameon=False)
    _save(fig, output)
