from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from co2ipsimulator.model import GridMetadata

from .configs import MassMeasure, MassMode
from .observations import SimulatedSnapshot, SnapshotObservation


class FootprintFeature(Enum):
    """Where a layer's plume sits.

    The first spatial moment of the footprint, normalised by the total footprint
    over all layers, so it equals the layer's area fraction times its centroid.
    A plain centroid cannot be used: it is undefined for a layer that holds
    nothing, and any stand-in value scores as a match, which rewards exactly the
    parameter draws that produce no plume at all. The moment vanishes instead,
    which is the correct mismatch against a layer that was observed.
    """

    MOMENT_X_FRACTION = "moment_x_fraction"
    MOMENT_Y_FRACTION = "moment_y_fraction"


FOOTPRINT_FEATURES = tuple(FootprintFeature)


def _layer_positions(requested: Sequence[str], available: Sequence[str]) -> np.ndarray:
    positions = {name: index for index, name in enumerate(available)}
    return np.fromiter(
        (positions[name] for name in requested), dtype=np.intp, count=len(requested)
    )


def footprint_summary(
    footprints: np.ndarray,
    metadata: GridMetadata,
) -> np.ndarray:
    x = np.arange(metadata.nx, dtype=np.float64) / max(metadata.nx - 1, 1)
    y = np.arange(metadata.ny, dtype=np.float64) / max(metadata.ny - 1, 1)
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
    total = max(float(footprints.sum()), 1.0)
    return np.concatenate(
        [
            np.array(
                [(mask * x_grid).sum() / total, (mask * y_grid).sum() / total],
                dtype=np.float64,
            )
            for mask in footprints
        ]
    )


def mass_fractions(mass: np.ndarray) -> np.ndarray:
    return np.divide(
        mass,
        mass.sum(),
        out=np.zeros_like(mass),
        where=mass.sum() > 0.0,
    )


def mass_megatonnes(mass: np.ndarray) -> np.ndarray:
    return mass / 1.0e9


_MASS_TRANSFORMS: dict[MassMode, Callable[[np.ndarray], np.ndarray]] = {
    MassMode.FRACTION: mass_fractions,
    MassMode.MT: mass_megatonnes,
}

_MASS_NAMES = {
    MassMode.FRACTION: "co2_mass_fraction",
    MassMode.MT: "co2_mass_mt",
}


@dataclass(frozen=True)
class FootprintSummary:
    layer_names: tuple[str, ...]
    relative_epsilon: float = 0.8
    epsilon_floor: float = 0.01

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(
            f"{layer_name}.{feature.value}"
            for layer_name in self.layer_names
            for feature in FOOTPRINT_FEATURES
        )

    def simulated(self, snapshot: SimulatedSnapshot) -> np.ndarray:
        footprints = np.stack([snapshot.footprints[name] for name in self.layer_names])
        return footprint_summary(footprints, snapshot.metadata)

    def observed(
        self, observation: SnapshotObservation, metadata: GridMetadata
    ) -> np.ndarray:
        positions = _layer_positions(self.layer_names, observation.layer_names)
        return footprint_summary(observation.footprints[positions], metadata)

    def epsilon(self, observed: np.ndarray) -> np.ndarray:
        return np.maximum(np.abs(observed) * self.relative_epsilon, self.epsilon_floor)


@dataclass(frozen=True)
class MassSummary:
    layer_names: tuple[str, ...]
    mode: MassMode = MassMode.FRACTION
    measure: MassMeasure = MassMeasure.STORED_MASS
    relative_epsilon: float = 0.25
    epsilon_floor: float = 0.03

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(
            f"{layer_name}.{_MASS_NAMES[self.mode]}" for layer_name in self.layer_names
        )

    def _summarize(self, mass_kg: np.ndarray) -> np.ndarray:
        return _MASS_TRANSFORMS[self.mode](mass_kg)

    def simulated(self, snapshot: SimulatedSnapshot) -> np.ndarray:
        if self.measure is MassMeasure.FOOTPRINT_AREA:
            raw = np.array(
                [float(snapshot.footprints[name].sum()) for name in self.layer_names],
                dtype=np.float64,
            )
        else:
            raw = np.array(
                [snapshot.mass_per_layer_kg[name] for name in self.layer_names],
                dtype=np.float64,
            )
        return self._summarize(raw)

    def observed(
        self, observation: SnapshotObservation, metadata: GridMetadata
    ) -> np.ndarray:
        positions = _layer_positions(self.layer_names, observation.layer_names)
        if self.measure is MassMeasure.FOOTPRINT_AREA:
            raw = (
                observation.footprints[positions]
                .reshape(len(positions), -1)
                .sum(axis=1)
                .astype(np.float64)
            )
        else:
            raw = observation.mass_per_layer_kg[positions]
        return self._summarize(raw)

    def epsilon(self, observed: np.ndarray) -> np.ndarray:
        return np.maximum(np.abs(observed) * self.relative_epsilon, self.epsilon_floor)


def unit_directions(count: int) -> np.ndarray:
    angles = np.pi * np.arange(count, dtype=np.float64) / count
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def _wasserstein_1d(left: np.ndarray, right: np.ndarray) -> float:
    values = np.concatenate([left, right])
    weights = np.concatenate(
        [np.full(left.size, 1.0 / left.size), np.full(right.size, -1.0 / right.size)]
    )
    order = np.argsort(values, kind="stable")
    residual = np.cumsum(weights[order])[:-1]
    return float(np.abs(residual) @ np.diff(values[order]))


def sliced_wasserstein(
    left: np.ndarray, right: np.ndarray, directions: np.ndarray
) -> float:
    """Sliced 1-Wasserstein distance between two uniform clouds of points.

    Scaled by ``pi/2``, the reciprocal of the mean of ``|cos|`` over directions,
    so that a rigidly translated cloud reports the length of the translation and
    the result reads as a distance in metres. Averaging over a finite number of
    directions makes that exact only on average over orientations; with eight
    directions the residual anisotropy is about one percent.
    """
    projections = (
        _wasserstein_1d(left @ direction, right @ direction) for direction in directions
    )
    return 0.5 * np.pi * float(sum(projections) / len(directions))


def mask_points(mask: np.ndarray, metadata: GridMetadata) -> np.ndarray:
    rows, columns = np.nonzero(mask)
    return np.stack(
        [
            metadata.xmin + rows * metadata.dx,
            metadata.ymin + columns * metadata.dy,
        ],
        axis=1,
    )


def area_shares(footprints: Sequence[np.ndarray]) -> np.ndarray:
    areas = np.array([float(mask.sum()) for mask in footprints], dtype=np.float64)
    total = areas.sum()
    return areas / total if total > 0.0 else areas


def plume_transport_distance(
    simulated: Sequence[np.ndarray],
    observed_points: Sequence[np.ndarray],
    observed_shares: np.ndarray,
    metadata: GridMetadata,
    directions: np.ndarray,
    unmatched_cost: float,
) -> float:
    """How far the simulated plume has to be moved to reproduce the observed one.

    Both plumes are read as a distribution over (layer, x, y) carrying unit total
    mass. Within a layer, the share that both plumes agree on is transported
    laterally, at the sliced Wasserstein cost in metres. The share that has to
    change layer cannot be moved laterally at all, and is charged ``unmatched_cost``
    instead. The result is zero only on an exact match, and it is bounded by
    ``unmatched_cost``, which is what a draw with no plume at all scores.
    """
    shares = area_shares(simulated)
    lateral = 0.0
    matched = 0.0
    for mask, points, share, observed_share in zip(
        simulated, observed_points, shares, observed_shares, strict=True
    ):
        common = min(float(share), float(observed_share))
        if common <= 0.0:
            continue
        lateral += common * sliced_wasserstein(
            mask_points(mask, metadata), points, directions
        )
        matched += common
    return lateral + unmatched_cost * (1.0 - matched)


@dataclass(frozen=True)
class TransportSummary:
    """One scalar per survey: the transport distance to the observed plume.

    Unlike the other components this is a discrepancy rather than a statistic, so
    its observed value is zero by construction and the observed plume is carried
    here. It replaces a per-layer moment vector with a single number that responds
    to the shape of the footprint and not only to its centroid, which keeps the
    summary dimension independent of the number of layers.
    """

    layer_names: tuple[str, ...]
    observed_points: dict[int, tuple[np.ndarray, ...]]
    observed_shares: dict[int, np.ndarray]
    metadata: GridMetadata
    directions: np.ndarray
    unmatched_cost: float
    relative_epsilon: float = 0.05

    @property
    def names(self) -> tuple[str, ...]:
        return ("plume_transport_m",)

    def simulated(self, snapshot: SimulatedSnapshot) -> np.ndarray:
        return np.array(
            [
                plume_transport_distance(
                    [snapshot.footprints[name] for name in self.layer_names],
                    self.observed_points[snapshot.year],
                    self.observed_shares[snapshot.year],
                    self.metadata,
                    self.directions,
                    self.unmatched_cost,
                )
            ]
        )

    def observed(
        self, observation: SnapshotObservation, metadata: GridMetadata
    ) -> np.ndarray:
        return np.zeros(1)

    def epsilon(self, observed: np.ndarray) -> np.ndarray:
        return np.full(1, self.relative_epsilon * self.unmatched_cost)


type SummaryComponent = FootprintSummary | MassSummary | TransportSummary
