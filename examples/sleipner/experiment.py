from __future__ import annotations

import numpy as np

from co2ipsimulator.inference import (
    ForwardModelConfig,
    RunConfig,
    SnapshotObservation,
    SummaryStatistics,
    build_observed_summary,
    build_summary_statistics,
)
from co2ipsimulator.inference.observations import (
    load_plume_outlines,
    rasterize_plume_outlines,
)
from co2ipsimulator.model import load_rms_polygon

from .config import (
    ANNUAL_RATES_MT,
    DEPTH_SURFACES,
    INJECTION_SITE_FILE,
    PAPER_LAYER_NAMES,
)
from .data_loader import load_sleipner_surfaces
from .layer_stack import sleipner_layer_stack

CAPROCK_PTH_PA = 1.0e6
START_YEAR = 1996
KG_PER_MT = 1.0e9

POLYGON_DIRS = {
    2010: DEPTH_SURFACES.parent / "Sleipner_Plumes_Boundaries" / "data",
    2023: DEPTH_SURFACES.parent / "sleipner_2023_polygons",
}


def sand_layer_names() -> tuple[str, ...]:
    return tuple(PAPER_LAYER_NAMES)


def cumulative_mt(year: int) -> float:
    return float(sum(ANNUAL_RATES_MT[: year - START_YEAR + 1]))


def load_model():
    surfaces, metadata = load_sleipner_surfaces(DEPTH_SURFACES)
    stack = tuple(sleipner_layer_stack())
    site = load_rms_polygon(INJECTION_SITE_FILE)
    source_xy = (float(site[:, 0].mean()), float(site[:, 1].mean()))
    return surfaces, metadata, stack, source_xy


def snapshot_years(run_config: RunConfig) -> tuple[int, ...]:
    return tuple(int(y) for y in run_config.extras.get("snapshot_years", (2010, 2023)))


def build_forward_config(run_config: RunConfig) -> ForwardModelConfig:
    surfaces, metadata, stack, source_xy = load_model()
    annual = tuple(rate * KG_PER_MT for rate in ANNUAL_RATES_MT)
    return ForwardModelConfig(
        depth_surfaces=surfaces,
        layer_stack=stack,
        metadata=metadata,
        inference=run_config.inference,
        source_xy=source_xy,
        annual_masses_kg=annual,
        start_year=START_YEAR,
        top_seal_pth_pa=CAPROCK_PTH_PA,
        time_rtol=float(run_config.extras.get("time_rtol", 1.0e-4)),
        max_substeps=int(run_config.extras.get("max_substeps", 16_384)),
    )


def observed_snapshot(year: int, metadata) -> SnapshotObservation:
    layers = sand_layer_names()
    outlines = load_plume_outlines(POLYGON_DIRS[year], layers)
    masks = rasterize_plume_outlines(outlines, metadata, layers)
    cells = masks.reshape(len(layers), -1).sum(axis=1).astype(np.float64)
    total = cells.sum()
    mass_kg = (
        cells / total * cumulative_mt(year) * KG_PER_MT
        if total > 0
        else np.zeros(len(layers))
    )
    return SnapshotObservation(int(year), layers, masks, mass_kg)


def build_observed(
    config: ForwardModelConfig, run_config: RunConfig
) -> tuple[SummaryStatistics, np.ndarray, np.ndarray, tuple[SnapshotObservation, ...]]:
    years = snapshot_years(run_config)
    observations = tuple(observed_snapshot(year, config.metadata) for year in years)
    statistics = build_summary_statistics(
        run_config.summary,
        years,
        sand_layer_names(),
        config.metadata,
        observations,
    )
    observed, epsilon = build_observed_summary(
        statistics, observations, config.metadata
    )
    return statistics, observed, epsilon, observations
