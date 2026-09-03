from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from co2ipsimulator.inference import RunConfig, forward_snapshots
from co2ipsimulator.model import GridMetadata

from ..experiment import (
    build_forward_config,
    calibration_years,
    sand_layer_names,
    truth_forward,
)
from ..scenarios import Scenario

DEFAULT_THRESHOLDS = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0)


@dataclass(frozen=True)
class OperatorResponse:
    """What the detection threshold removes from the truth, before any inference.

    ``areas`` counts detected cells per threshold and layer, so a column of it
    read against ``thresholds`` is the complementary distribution of column
    height over that layer's plume.
    """

    thresholds: np.ndarray
    layer_names: tuple[str, ...]
    year: int
    metadata: GridMetadata
    areas: np.ndarray
    masks: dict[float, dict[str, np.ndarray]]


def operator_response(
    scenario: Scenario,
    run_config: RunConfig,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> OperatorResponse:
    inference = replace(run_config.inference, detection_prior=None)
    run_config = replace(run_config, inference=inference)
    year = max(calibration_years(scenario, run_config))
    layers = sand_layer_names(scenario)
    base = build_forward_config(scenario, run_config, year)
    truth_config, truth = truth_forward(scenario, run_config, base)

    areas = np.empty((len(thresholds), len(layers)))
    masks: dict[float, dict[str, np.ndarray]] = {}
    for index, threshold in enumerate(thresholds):
        config = replace(truth_config, detection_threshold_m=threshold)
        snapshots, _ = forward_snapshots(config, truth, (year,))
        footprints = snapshots[-1].footprints
        areas[index] = [float(footprints[name].sum()) for name in layers]
        masks[threshold] = {name: footprints[name].copy() for name in layers}

    return OperatorResponse(
        thresholds=np.asarray(thresholds, dtype=np.float64),
        layer_names=layers,
        year=year,
        metadata=base.metadata,
        areas=areas,
        masks=masks,
    )
