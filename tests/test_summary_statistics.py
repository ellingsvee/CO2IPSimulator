from __future__ import annotations

import numpy as np
import pytest

from co2ipsimulator.inference import (
    MassMode,
    SimulatedSnapshot,
    SummaryConfig,
    SummaryMode,
    build_observed_summary,
    build_summary_statistics,
    make_snapshot_observation,
)
from co2ipsimulator.model import GridMetadata


def test_mass_and_footprint_summaries_align_observed_values_and_tolerances():
    metadata = GridMetadata(nx=7, ny=7, xmin=0.0, xmax=600.0, ymin=0.0, ymax=600.0)
    layers = ("upper", "lower")
    footprints = {
        "upper": np.eye(7, dtype=bool),
        "lower": np.fliplr(np.eye(7, dtype=bool)),
    }
    observation = make_snapshot_observation(
        2007, layers, footprints, mass_per_layer_kg=np.array([3.0e9, 1.0e9])
    )
    statistics = build_summary_statistics(
        SummaryConfig(
            mode=SummaryMode.MASS_AND_FOOTPRINTS,
            mass_relative_epsilon=0.20,
            mass_epsilon_floor=0.04,
            moment_relative_epsilon=0.10,
            moment_epsilon_floor=0.006,
        ),
        (2007,),
        layers,
        metadata,
        (observation,),
    )

    observed, epsilon = build_observed_summary(statistics, (observation,), metadata)

    assert observed.shape == epsilon.shape == (6,)
    assert statistics.names[-1] == "2007::lower.co2_mass_fraction"
    assert np.allclose(observed[-2:], [0.75, 0.25])
    assert np.allclose(epsilon[:4], np.maximum(np.abs(observed[:4]) * 0.10, 0.006))
    assert np.allclose(epsilon[-2:], [0.15, 0.05])


def test_mass_summary_supports_absolute_mass():
    statistics = build_summary_statistics(
        SummaryConfig(
            mode=SummaryMode.MASS,
            mass_mode=MassMode.MT,
            mass_relative_epsilon=0.1,
            mass_epsilon_floor=0.2,
        ),
        snapshot_years=(2, 4),
        layer_names=("L1", "L2"),
        metadata=GridMetadata(nx=3, ny=3, xmin=0.0, xmax=100.0, ymin=0.0, ymax=100.0),
        observations=(),
    )

    assert statistics.names == (
        "2::L1.co2_mass_mt",
        "2::L2.co2_mass_mt",
        "4::L1.co2_mass_mt",
        "4::L2.co2_mass_mt",
    )
    assert np.allclose(
        statistics.epsilon(np.array([1.0, 3.0, 4.0, 1.0])), [0.2, 0.3, 0.4, 0.2]
    )


def test_transport_summary_measures_spatial_and_cross_layer_mismatch():
    metadata = GridMetadata(nx=21, ny=21, xmin=0.0, xmax=2000.0, ymin=0.0, ymax=2000.0)
    layers = ("upper", "lower")
    truth = {name: np.zeros((21, 21), dtype=bool) for name in layers}
    truth["upper"][2:6, 2:6] = True
    truth["lower"][2:6, 2:6] = True
    observation = make_snapshot_observation(3, layers, truth, np.array([1.0, 1.0]))
    statistics = build_summary_statistics(
        SummaryConfig(mode=SummaryMode.TRANSPORT),
        (3,),
        layers,
        metadata,
        (observation,),
    )

    def snapshot(footprints):
        return SimulatedSnapshot(
            year=3,
            mass_per_layer_kg={name: 1.0 for name in layers},
            footprints=footprints,
            metadata=metadata,
        )

    shifted = {name: np.roll(mask, 4, axis=0) for name, mask in truth.items()}
    relayered = {"upper": truth["lower"], "lower": np.zeros((21, 21), dtype=bool)}
    unmatched_cost = statistics.components[0].unmatched_cost

    assert statistics.simulated([snapshot(truth)])[0] == pytest.approx(0.0)
    assert statistics.simulated([snapshot(shifted)])[0] == pytest.approx(
        400.0, rel=0.02
    )
    assert statistics.simulated([snapshot(relayered)])[0] == pytest.approx(
        0.5 * unmatched_cost
    )


def test_transport_tolerance_scales_with_domain_size():
    metadata = GridMetadata(nx=11, ny=11, xmin=0.0, xmax=1000.0, ymin=0.0, ymax=1000.0)
    footprint = np.zeros((11, 11), dtype=bool)
    footprint[4:7, 4:7] = True
    observation = make_snapshot_observation(
        1, ("only",), {"only": footprint}, np.array([1.0])
    )
    statistics = build_summary_statistics(
        SummaryConfig(mode=SummaryMode.TRANSPORT, transport_relative_epsilon=0.2),
        (1,),
        ("only",),
        metadata,
        (observation,),
    )

    observed, epsilon = build_observed_summary(statistics, (observation,), metadata)

    assert observed[0] == 0.0
    assert epsilon[0] == pytest.approx(0.2 * 0.5 * np.hypot(1000.0, 1000.0))
