from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from co2ipsimulator.inference import load_run_config

from examples.synthetic.experiment import prior_distributions_and_log


def posterior_correlation(
    samples: np.ndarray,
    parameter_names: tuple[str, ...],
    parameter_is_log: tuple[bool, ...],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return Pearson correlations in the coordinates used by posterior plots."""
    samples = np.asarray(samples, dtype=float)
    n_parameters = len(parameter_names)
    if samples.ndim != 2 or samples.shape[1] != n_parameters:
        raise ValueError(
            "samples must have shape (n_samples, n_parameters); "
            f"got {samples.shape} for {n_parameters} parameter names"
        )
    if len(parameter_is_log) != n_parameters:
        raise ValueError("parameter_is_log must match parameter_names")
    if samples.shape[0] < 2:
        raise ValueError("at least two posterior samples are required")
    if not np.isfinite(samples).all():
        raise ValueError("posterior samples must all be finite")

    transformed = samples.copy()
    coordinate_names = list(parameter_names)
    for index, is_log in enumerate(parameter_is_log):
        if not is_log:
            continue
        if np.any(transformed[:, index] <= 0.0):
            raise ValueError(
                f"log-scaled parameter {parameter_names[index]!r} has non-positive samples"
            )
        transformed[:, index] = np.log10(transformed[:, index])
        coordinate_names[index] = f"log10({parameter_names[index]})"

    correlation = np.corrcoef(transformed, rowvar=False)
    if not np.isfinite(correlation).all():
        raise ValueError("correlations are undefined for a constant parameter")
    return correlation, tuple(coordinate_names)


def _plot_labels(coordinate_names: tuple[str, ...]) -> list[str]:
    labels = []
    for name in coordinate_names:
        if name == "Shale_thick":
            labels.append(r"$P_{\mathrm{th},8}$")
        elif name.startswith("Shale_"):
            index = name.removeprefix("Shale_")
            labels.append(rf"$P_{{\mathrm{{th}},{index}}}$")
        elif name == "log10(lambda_s)":
            labels.append(r"$\log_{10}\lambda$")
        else:
            labels.append(name)
    return labels


def write_posterior_correlation(
    samples: np.ndarray,
    parameter_names: tuple[str, ...],
    parameter_is_log: tuple[bool, ...],
    *,
    output_dir: Path,
) -> tuple[Path, Path]:
    correlation, coordinate_names = posterior_correlation(
        samples, parameter_names, parameter_is_log
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "abc_posterior_correlation.csv"
    pdf_path = output_dir / "abc_posterior_correlation.pdf"

    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", *coordinate_names])
        for name, row in zip(coordinate_names, correlation):
            writer.writerow([name, *(f"{value:.8f}" for value in row)])

    labels = _plot_labels(coordinate_names)
    size = max(7.0, 0.78 * len(labels) + 1.8)
    fig, ax = plt.subplots(figsize=(size, size - 0.4), constrained_layout=True)
    upper_triangle = np.triu(np.ones_like(correlation, dtype=bool), k=1)
    lower_triangle = np.ma.masked_where(upper_triangle, correlation)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(alpha=0.0)
    image = ax.imshow(lower_triangle, cmap=cmap, vmin=-1.0, vmax=1.0)
    ax.set_xticks(
        np.arange(len(labels)), labels=labels, rotation=45, ha="right", fontsize=20
    )
    ax.set_yticks(np.arange(len(labels)), labels=labels, fontsize=20)

    for row in range(len(labels)):
        for column in range(row + 1):
            value = correlation[row, column]
            color = "white" if abs(value) >= 0.55 else "black"
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=16,
            )

    divider = make_axes_locatable(ax)
    colorbar_axis = divider.append_axes("right", size="5%", pad=0.22)
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.ax.tick_params(labelsize=18)
    colorbar.set_label("Pearson r", fontsize=20)
    fig.savefig(pdf_path, dpi=180, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return csv_path, pdf_path


def generate(output_dir: Path, *, config_path: Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    samples = np.load(output_dir / "abc_posterior_parameter_samples.npy")
    names = tuple((output_dir / "abc_parameter_names.txt").read_text().splitlines())
    run_config = load_run_config(config_path)
    _, parameter_is_log = prior_distributions_and_log(run_config)
    return write_posterior_correlation(
        samples, names, parameter_is_log, output_dir=output_dir
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a Sleipner posterior parameter correlation matrix."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/sleipner/output/abc"),
        help="Directory containing the saved posterior samples and parameter names.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/sleipner/configs/abc.toml"),
        help="Run config used to identify log-scaled parameters.",
    )
    args = parser.parse_args()
    csv_path, pdf_path = generate(args.output_dir, config_path=args.config)
    print(f"[posterior-correlation] wrote {csv_path} and {pdf_path}")


if __name__ == "__main__":
    main()
