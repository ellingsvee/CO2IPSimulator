from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ._helpers import pretty_layer_name
from .palette import MODEL_COLOR

KG_PER_MT = 1.0e9


def plot_mass_per_layer(
    mass_per_layer: dict[str, float],
    *,
    title: str | None = None,
    output: Path | str | None = None,
    figsize: tuple[float, float] = (9, 4.5),
) -> plt.Figure:
    layers = list(mass_per_layer)
    mass_mt = np.array([mass_per_layer[name] for name in layers]) / KG_PER_MT
    x = np.arange(len(layers))

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.bar(
        x,
        mass_mt,
        width=0.7,
        color=MODEL_COLOR,
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [pretty_layer_name(name) for name in layers], rotation=45, ha="right"
    )
    ax.set_ylabel("Stored mass (Mt)")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    if title:
        ax.set_title(title)
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(output), dpi=180, transparent=True)
    return fig
