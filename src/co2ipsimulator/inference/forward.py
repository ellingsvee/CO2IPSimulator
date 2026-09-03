from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, Sequence

import numpy as np

from co2ipsimulator.model import (
    GridMetadata,
    LayerProps,
    StepDiagnostics,
    TrapFill,
    TrapFillResult,
    build_trapfill,
    seal_fields,
)

from .configs import InferenceConfig
from .observations import SimulatedSnapshot
from .summary_statistics import SummaryStatistics

FOOTPRINT_THRESHOLD_M = 1.0e-9


class ForwardModel(Protocol):
    def __call__(
        self,
        rng: np.random.Generator,
        parameters: np.ndarray,
        size: object = None,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class ForwardModelConfig:
    depth_surfaces: dict[str, np.ndarray]
    layer_stack: tuple[LayerProps, ...]
    metadata: GridMetadata
    inference: InferenceConfig
    source_xy: tuple[float, float] | tuple[tuple[float, float], ...]
    annual_masses_kg: tuple[float, ...]
    start_year: int = 0
    connate_water_saturation: float = 0.30
    top_seal_pth_pa: float | None = None
    time_rtol: float = 1.0e-4
    max_substeps: int = 16_384
    # Minimum detectable column height of the seismic detection operator. The
    # default keeps every cell holding CO2, which is the h_det = 0 limit. Ignored
    # when ``inference.detection_prior`` is set, which infers it per draw instead.
    detection_threshold_m: float = FOOTPRINT_THRESHOLD_M


def with_threshold_pressures_kpa(
    layer_stack: Sequence[LayerProps],
    layer_names: tuple[str, ...],
    threshold_pressures_kpa: np.ndarray,
) -> tuple[LayerProps, ...]:
    index = {layer.name: position for position, layer in enumerate(layer_stack)}
    stack = list(layer_stack)
    for name, value_kpa in zip(layer_names, threshold_pressures_kpa, strict=True):
        position = index[name]
        stack[position] = replace(stack[position], pth_pa=float(value_kpa) * 1.0e3)
    return tuple(stack)


@dataclass(frozen=True)
class ForwardDiagnostics:
    """What a forward run could not account for.

    ``converged`` is false when some injection interval exhausted its substep
    budget before meeting ``time_rtol``; ``worst_relative_error`` is then the
    error the run attained. ``escaped_fraction`` is the share of the injected
    mass that left the model: the lateral edge is a no-flow wall, so this can
    only be non-zero once a unit is completely full or once the top seal is
    breached. ``stalled_fraction`` is the part of that the spill-graph cascade
    guard dropped, and is zero for a well-formed graph.
    """

    converged: bool
    worst_relative_error: float
    escaped_fraction: float
    stalled_fraction: float

    @property
    def is_clean(self) -> bool:
        return (
            self.converged
            and self.escaped_fraction <= 0.0
            and self.stalled_fraction <= 0.0
        )


_CLEAN = ForwardDiagnostics(
    converged=True,
    worst_relative_error=0.0,
    escaped_fraction=0.0,
    stalled_fraction=0.0,
)


def _diagnostics(
    result: TrapFillResult,
    convergence: StepDiagnostics | None,
    injected_kg: float,
) -> ForwardDiagnostics:
    scale = max(injected_kg, 1.0)
    return ForwardDiagnostics(
        converged=True if convergence is None else convergence.converged,
        worst_relative_error=(
            0.0 if convergence is None else convergence.estimated_relative_error
        ),
        escaped_fraction=result.escaped_kg / scale,
        stalled_fraction=result.stalled_kg / scale,
    )


def _footprints(
    columns: dict[str, np.ndarray], threshold_m: float
) -> dict[str, np.ndarray]:
    return {name: column > threshold_m for name, column in columns.items()}


def _capture_stateful(
    trapfill,
    annual_masses: Sequence[float],
    wanted_years: tuple[int, ...],
    metadata: GridMetadata,
    start_year: int,
    threshold_m: float,
) -> tuple[list[SimulatedSnapshot], ForwardDiagnostics]:
    trapfill.reset()
    wanted = set(wanted_years)
    snapshots: list[SimulatedSnapshot] = []
    convergence: list[StepDiagnostics] = []
    for offset, mass in enumerate(annual_masses):
        convergence.append(trapfill.step(float(mass)))
        year = offset + start_year
        if year in wanted:
            snapshots.append(
                SimulatedSnapshot(
                    year=year,
                    mass_per_layer_kg=trapfill._mass_per_layer(),
                    footprints=_footprints(
                        trapfill.state_column_heights(), threshold_m
                    ),
                    metadata=metadata,
                )
            )
    diagnostics = _diagnostics(
        trapfill.state_result(),
        StepDiagnostics.worst(convergence) if convergence else None,
        float(sum(annual_masses)),
    )
    return snapshots, diagnostics


def _capture_stateless(
    trapfill,
    annual_masses: Sequence[float],
    wanted_years: tuple[int, ...],
    metadata: GridMetadata,
    start_year: int,
    threshold_m: float,
) -> tuple[list[SimulatedSnapshot], ForwardDiagnostics]:
    cumulative = np.cumsum(np.asarray(annual_masses, dtype=np.float64))
    snapshots: list[SimulatedSnapshot] = []
    results: list[tuple[TrapFillResult, float]] = []
    for year in sorted(set(wanted_years)):
        mass = float(cumulative[year - start_year])
        result = trapfill.fill(mass)
        results.append((result, mass))
        snapshots.append(
            SimulatedSnapshot(
                year=year,
                mass_per_layer_kg=result.mass_per_layer,
                footprints=_footprints(trapfill.column_heights(mass), threshold_m),
                metadata=metadata,
            )
        )
    # Every snapshot is an independent fill, so the worst leak over them is what
    # describes the run.
    worst = max(
        (_diagnostics(result, None, mass) for result, mass in results),
        key=lambda d: (d.stalled_fraction, d.escaped_fraction),
        default=_CLEAN,
    )
    return snapshots, worst


_GEOMETRY_CACHE: dict[int, tuple[ForwardModelConfig, TrapFill]] = {}


def seal_log10_mobility(
    config: ForwardModelConfig, parameters: np.ndarray
) -> float | None:
    if config.inference.rate_limit_prior is None:
        return None
    return float(np.log10(parameters[len(config.inference.pth_priors)]))


def detection_threshold(config: ForwardModelConfig, parameters: np.ndarray) -> float:
    """The detection threshold this draw is observed through.

    Fixed by the configuration unless the detection prior is set, in which case
    it is the last entry of the parameter vector and varies over the inference.
    """
    if config.inference.detection_prior is None:
        return config.detection_threshold_m
    return float(parameters[-1])


def _trapfill(config: ForwardModelConfig, parameters: np.ndarray) -> TrapFill:
    """The simulator for this draw.

    The trap graph depends only on the geometry, so it is built once per forward
    configuration and re-pointed at the new seal parameters afterwards. The cache
    holds the configuration as well, so its ``id`` cannot be reused by another
    object while the entry is alive.
    """
    layer_stack = with_threshold_pressures_kpa(
        config.layer_stack,
        config.inference.pth_layer_names,
        parameters[: len(config.inference.pth_priors)],
    )
    log10_mobility = seal_log10_mobility(config, parameters)

    cached = _GEOMETRY_CACHE.get(id(config))
    if cached is None:
        trapfill = build_trapfill(
            config.depth_surfaces,
            layer_stack,
            config.metadata,
            source_xy=config.source_xy,
            connate_water_saturation=config.connate_water_saturation,
            seal_log10_mobility=log10_mobility,
            top_seal_pth_pa=config.top_seal_pth_pa,
            time_rtol=config.time_rtol,
            max_substeps=config.max_substeps,
        )
        _GEOMETRY_CACHE[id(config)] = (config, trapfill)
        return trapfill

    trapfill = cached[1]
    trapfill.update_seals(
        seal_fields(
            layer_stack,
            seal_log10_mobility=log10_mobility,
            top_seal_pth_pa=config.top_seal_pth_pa,
        )
    )
    return trapfill


def forward_snapshots(
    config: ForwardModelConfig,
    parameters: np.ndarray,
    snapshot_years: tuple[int, ...],
) -> tuple[list[SimulatedSnapshot], ForwardDiagnostics]:
    """Run one parameter set and capture the requested calendar-year states."""
    trapfill = _trapfill(config, parameters)
    quasi_static = seal_log10_mobility(config, parameters) is None
    capture = _capture_stateless if quasi_static else _capture_stateful
    return capture(
        trapfill,
        config.annual_masses_kg,
        tuple(snapshot_years),
        config.metadata,
        config.start_year,
        detection_threshold(config, parameters),
    )


def run_forward_model(
    config: ForwardModelConfig,
    statistics: SummaryStatistics,
    parameters: np.ndarray,
) -> np.ndarray:
    snapshots, _ = forward_snapshots(config, parameters, statistics.snapshot_years)
    return statistics.simulated(snapshots)


@dataclass(frozen=True)
class _SimulatorForward:
    config: ForwardModelConfig
    statistics: SummaryStatistics
    parameter_size: int

    def __call__(
        self,
        rng: np.random.Generator,
        parameters: np.ndarray,
        size: object = None,
    ) -> np.ndarray:
        summary = run_forward_model(
            self.config, self.statistics, parameters.reshape(self.parameter_size)
        )
        normalized_size = () if size is None else size
        sample_shape = tuple(
            int(value)
            for value in np.asarray(normalized_size, dtype=np.int64).reshape(-1)
        )
        return np.broadcast_to(summary, sample_shape + summary.shape).copy()


def build_forward_model(
    config: ForwardModelConfig, statistics: SummaryStatistics
) -> ForwardModel:
    return _SimulatorForward(config, statistics, len(config.inference.parameter_names))
