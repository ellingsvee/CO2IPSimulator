from __future__ import annotations

import numpy as np

from co2ipsimulator.inference import (
    FootprintSummary,
    ForwardModelConfig,
    InferenceConfig,
    LogNormalPrior,
    MassMode,
    MassSummary,
    RateLimitPrior,
    SummaryStatistics,
    ThresholdPressurePrior,
    build_forward_model,
    run_forward_model,
    with_threshold_pressures_kpa,
)


def _config(layered_dome, inference):
    return ForwardModelConfig(
        depth_surfaces=layered_dome.surfaces,
        layer_stack=layered_dome.stack,
        metadata=layered_dome.metadata,
        inference=inference,
        source_xy=layered_dome.source_xy,
        annual_masses_kg=(2.0e8,) * 8,
        start_year=2000,
    )


def test_threshold_pressure_parameters_update_only_named_seals(layered_dome):
    updated = with_threshold_pressures_kpa(
        layered_dome.stack, ("shale",), np.array([75.0])
    )
    before = {layer.name: layer for layer in layered_dome.stack}
    after = {layer.name: layer for layer in updated}

    assert after["shale"].pth_pa == 75.0e3
    assert before["shale"].pth_pa == 4.0e4
    assert after["upper_sand"] == before["upper_sand"]
    assert after["lower_sand"] == before["lower_sand"]


def test_forward_model_runs_requested_calendar_year(layered_dome):
    inference = InferenceConfig((ThresholdPressurePrior("shale"),))
    statistics = SummaryStatistics(
        (2007,),
        (
            FootprintSummary(("lower_sand",)),
            MassSummary(("lower_sand",), mode=MassMode.MT),
        ),
    )
    config = _config(layered_dome, inference)

    direct = run_forward_model(config, statistics, np.array([50.0]))
    wrapped = build_forward_model(config, statistics)(
        np.random.default_rng(0), np.array([50.0])
    )

    assert direct.shape == (3,)
    assert np.array_equal(direct, wrapped)
    assert np.all(np.isfinite(direct))
    assert direct[-1] > 0.0


def test_lower_seal_mobility_retains_more_mass_below(layered_dome):
    inference = InferenceConfig(
        (ThresholdPressurePrior("shale"),),
        RateLimitPrior("lambda_s", LogNormalPrior(mu=-25.0, sigma=1.0)),
    )
    statistics = SummaryStatistics(
        (2007,),
        (MassSummary(("upper_sand", "lower_sand"), mode=MassMode.FRACTION),),
    )
    config = _config(layered_dome, inference)

    throttled = run_forward_model(config, statistics, np.array([40.0, 1.0e-15]))
    relaxed = run_forward_model(config, statistics, np.array([40.0, 1.0e-10]))

    assert throttled[1] > relaxed[1]
