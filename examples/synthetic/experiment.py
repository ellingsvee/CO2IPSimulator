from __future__ import annotations

from dataclasses import replace

import numpy as np

from co2ipsimulator.inference import (
    FOOTPRINT_THRESHOLD_M,
    ForwardModelConfig,
    RateLimitPrior,
    RunConfig,
    SummaryStatistics,
    build_observed_summary,
    build_summary_statistics,
    forward_snapshots,
    make_snapshot_observation,
)

from .scenarios import Scenario

KG_PER_MT = 1.0e9

FINITE_RATE = "finite-rate"
QUASI_STATIC = "quasi-static"


def sand_layer_names(scenario: Scenario) -> tuple[str, ...]:
    return tuple(layer.name for layer in scenario.layer_stack() if layer.is_sand)


def calibration_years(scenario: Scenario, run_config: RunConfig) -> tuple[int, ...]:
    years = run_config.extras.get("calibration_years", scenario.calibration_years)
    return tuple(int(year) for year in years)


def forecast_years(scenario: Scenario, run_config: RunConfig) -> tuple[int, ...]:
    stop = run_config.extras.get("forecast_years_stop")
    step = int(run_config.extras.get("forecast_years_step", 5))
    forecast = int(scenario.forecast_year)
    if stop is None:
        return (forecast,)
    # A sweep over the monitoring period needs one common forecast grid, or the
    # arms cannot be drawn against the same truth curve.
    start = int(
        run_config.extras.get(
            "forecast_years_start", max(calibration_years(scenario, run_config))
        )
    )
    years = set(range(start, int(stop) + 1, step)) | {forecast}
    return tuple(sorted(years))


def truth_detection_threshold_m(run_config: RunConfig) -> float:
    """The threshold the synthetic observations are generated through."""
    return float(
        run_config.extras.get("truth_detection_threshold_m", FOOTPRINT_THRESHOLD_M)
    )


def truth_transfer(run_config: RunConfig) -> str:
    """Which transfer law the synthetic observation is generated through.

    Defaults to the fitted one, so a config that says nothing keeps the truth and
    the model on the same physics. Set to ``"quasi-static"`` or ``"finite-rate"``
    it decouples them, which is what a model-comparison experiment needs.
    """
    fitted = (
        QUASI_STATIC if run_config.inference.rate_limit_prior is None else FINITE_RATE
    )
    return str(run_config.extras.get("truth_transfer", fitted))


def truth_run_config(run_config: RunConfig) -> RunConfig:
    """The run configuration the synthetic observation is generated from."""
    wants_rate = truth_transfer(run_config) == FINITE_RATE
    if wants_rate == (run_config.inference.rate_limit_prior is not None):
        return run_config
    inference = replace(
        run_config.inference,
        rate_limit_prior=RateLimitPrior() if wants_rate else None,
    )
    return replace(run_config, inference=inference)


def truth_parameters(scenario: Scenario, run_config: RunConfig) -> np.ndarray:
    """The truth in the parameter space of ``run_config``.

    An entry the data-generating model leaves undefined - the seal mobility of a
    quasi-static truth, which is the ``lambda -> inf`` limit - is NaN, so the
    plots can drop it rather than draw a reference line that does not exist.
    """
    pth_by_shale = dict(zip(scenario.shale_layer_names, scenario.true_pth_kpa))
    values = [pth_by_shale[name] for name in run_config.inference.pth_layer_names]
    if run_config.inference.rate_limit_prior is not None:
        values.append(
            10.0 ** float(scenario.seal_log10_mobility)
            if truth_transfer(run_config) == FINITE_RATE
            else np.nan
        )
    if run_config.inference.detection_prior is not None:
        values.append(truth_detection_threshold_m(run_config))
    return np.array(values, dtype=np.float64)


def build_forward_config(
    scenario: Scenario, run_config: RunConfig, end_year: int
) -> ForwardModelConfig:
    annual = tuple(rate * KG_PER_MT for rate in scenario.annual_rates_mt(end_year))
    return ForwardModelConfig(
        depth_surfaces=scenario.depth_surfaces(),
        layer_stack=tuple(scenario.layer_stack()),
        metadata=scenario.metadata(),
        inference=run_config.inference,
        source_xy=scenario.well_xy,
        annual_masses_kg=annual,
        start_year=scenario.start_year,
        time_rtol=float(run_config.extras.get("time_rtol", 1.0e-4)),
        max_substeps=int(run_config.extras.get("max_substeps", 16_384)),
        detection_threshold_m=float(
            run_config.extras.get("detection_threshold_m", FOOTPRINT_THRESHOLD_M)
        ),
    )


def truth_forward_config(
    config: ForwardModelConfig, run_config: RunConfig
) -> ForwardModelConfig:
    """The configuration the synthetic observations are generated through.

    Differs from the fitted one in the detection threshold, which is a property
    of the data rather than of the model, and in the transfer law when the
    experiment deliberately generates the truth off the fitted physics. When the
    threshold is inferred it travels in the parameter vector instead and no
    override is needed.
    """
    if run_config.inference.detection_prior is None:
        threshold = truth_detection_threshold_m(run_config)
        if threshold != config.detection_threshold_m:
            config = replace(config, detection_threshold_m=threshold)
    truth_config = truth_run_config(run_config)
    if truth_config is run_config:
        return config
    return replace(config, inference=truth_config.inference)


def truth_forward(
    scenario: Scenario, run_config: RunConfig, config: ForwardModelConfig
) -> tuple[ForwardModelConfig, np.ndarray]:
    """The configuration and parameters that produce the synthetic truth."""
    truth_config = truth_run_config(run_config)
    return (
        truth_forward_config(config, run_config),
        truth_parameters(scenario, truth_config),
    )


def build_observed(
    config: ForwardModelConfig,
    scenario: Scenario,
    run_config: RunConfig,
) -> tuple[SummaryStatistics, np.ndarray, np.ndarray]:
    """Synthetic truth: the observation is a forward run at the true parameters."""
    layers = sand_layer_names(scenario)
    years = calibration_years(scenario, run_config)
    truth_config, truth = truth_forward(scenario, run_config, config)
    observations = tuple(
        make_snapshot_observation(
            snapshot.year,
            layers,
            snapshot.footprints,
            np.array([snapshot.mass_per_layer_kg[name] for name in layers]),
        )
        for snapshot in forward_snapshots(truth_config, truth, years)[0]
    )
    statistics = build_summary_statistics(
        run_config.summary, years, layers, config.metadata, observations
    )
    observed, epsilon = build_observed_summary(
        statistics, observations, config.metadata
    )
    return statistics, observed, epsilon


def prior_distributions_and_log(run_config: RunConfig):
    distributions = tuple(
        prior.distribution for prior in run_config.inference.pth_priors
    )
    is_log = (False,) * len(distributions)
    rate_limit = run_config.inference.rate_limit_prior
    if rate_limit is not None:
        distributions = distributions + (rate_limit.distribution,)
        is_log = is_log + (True,)
    detection = run_config.inference.detection_prior
    if detection is not None:
        distributions = distributions + (detection.distribution,)
        is_log = is_log + (False,)
    return distributions, is_log
