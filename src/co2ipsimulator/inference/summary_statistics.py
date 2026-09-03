from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from co2ipsimulator.model import GridMetadata

from .configs import SummaryConfig, SummaryMode
from .observations import SimulatedSnapshot, SnapshotObservation
from .summary_components import (
    FootprintSummary,
    MassSummary,
    SummaryComponent,
    TransportSummary,
    area_shares,
    mask_points,
    unit_directions,
)

TRANSPORT_DIRECTIONS = 8


def _concatenate(parts: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate(parts).astype(np.float64, copy=False)


@dataclass(frozen=True)
class SummaryStatistics:
    snapshot_years: tuple[int, ...]
    components: tuple[SummaryComponent, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(
            f"{year}::{name}"
            for year in self.snapshot_years
            for component in self.components
            for name in component.names
        )

    def simulated(self, snapshots: Sequence[SimulatedSnapshot]) -> np.ndarray:
        by_year = {snapshot.year: snapshot for snapshot in snapshots}
        return _concatenate(
            [
                component.simulated(by_year[year])
                for year in self.snapshot_years
                for component in self.components
            ]
        )

    def observed(
        self,
        observations: Sequence[SnapshotObservation],
        metadata: GridMetadata,
    ) -> np.ndarray:
        by_year = {observation.year: observation for observation in observations}
        return _concatenate(
            [
                component.observed(by_year[year], metadata)
                for year in self.snapshot_years
                for component in self.components
            ]
        )

    def epsilon(self, observed: np.ndarray) -> np.ndarray:
        parts: list[np.ndarray] = []
        start = 0
        for _ in self.snapshot_years:
            for component in self.components:
                stop = start + len(component.names)
                parts.append(component.epsilon(observed[start:stop]))
                start = stop
        return _concatenate(parts)


def _transport_summary(
    config: SummaryConfig,
    layer_names: tuple[str, ...],
    metadata: GridMetadata,
    observations: Sequence[SnapshotObservation],
) -> TransportSummary:
    by_year = {}
    for observation in observations:
        order = [observation.layer_names.index(name) for name in layer_names]
        masks = [observation.footprints[position] for position in order]
        by_year[observation.year] = (
            tuple(mask_points(mask, metadata) for mask in masks),
            area_shares(masks),
        )
    half_diagonal = 0.5 * float(
        np.hypot(metadata.xmax - metadata.xmin, metadata.ymax - metadata.ymin)
    )
    return TransportSummary(
        layer_names=layer_names,
        observed_points={year: points for year, (points, _) in by_year.items()},
        observed_shares={year: shares for year, (_, shares) in by_year.items()},
        metadata=metadata,
        directions=unit_directions(TRANSPORT_DIRECTIONS),
        unmatched_cost=half_diagonal,
        relative_epsilon=config.transport_relative_epsilon,
    )


def build_summary_statistics(
    config: SummaryConfig,
    snapshot_years: tuple[int, ...],
    layer_names: tuple[str, ...],
    metadata: GridMetadata,
    observations: Sequence[SnapshotObservation],
) -> SummaryStatistics:
    if config.mode is SummaryMode.TRANSPORT:
        components: tuple[SummaryComponent, ...] = (
            _transport_summary(config, layer_names, metadata, observations),
        )
        return SummaryStatistics(snapshot_years, components)

    mass = MassSummary(
        layer_names=layer_names,
        mode=config.mass_mode,
        measure=config.mass_measure,
        relative_epsilon=config.mass_relative_epsilon,
        epsilon_floor=config.mass_epsilon_floor,
    )
    if config.mode is SummaryMode.MASS:
        return SummaryStatistics(snapshot_years, (mass,))
    footprint = FootprintSummary(
        layer_names=layer_names,
        relative_epsilon=config.moment_relative_epsilon,
        epsilon_floor=config.moment_epsilon_floor,
    )
    return SummaryStatistics(snapshot_years, (footprint, mass))


def build_observed_summary(
    statistics: SummaryStatistics,
    observations: Sequence[SnapshotObservation],
    metadata: GridMetadata,
) -> tuple[np.ndarray, np.ndarray]:
    observed = statistics.observed(observations, metadata)
    return observed, statistics.epsilon(observed)
