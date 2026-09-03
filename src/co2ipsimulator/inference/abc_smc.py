from __future__ import annotations

import numpy as np
import pymc
import pytensor.tensor as pt

from .configs import InferenceConfig, InferenceVariable, PyMCConfig
from .forward import ForwardModel


def construct_priors(inference_config: InferenceConfig) -> tuple:
    pth_values = tuple(
        prior.distribution.build(prior.variable_name)
        for prior in inference_config.pth_priors
    )
    rate_limit_values = (
        ()
        if inference_config.rate_limit_prior is None
        else (
            inference_config.rate_limit_prior.distribution.build(
                inference_config.rate_limit_prior.name
            ),
        )
    )
    detection_values = (
        ()
        if inference_config.detection_prior is None
        else (
            inference_config.detection_prior.distribution.build(
                inference_config.detection_prior.name
            ),
        )
    )
    pymc.Deterministic(
        InferenceVariable.THRESHOLD_PRESSURE.value,
        pt.stack(pth_values),
    )
    parameters = pymc.Deterministic(
        InferenceVariable.PARAMETERS.value,
        pt.stack(pth_values + rate_limit_values + detection_values),
    )
    return (parameters,)


def build_abc_smc_model(
    inference_config: InferenceConfig,
    forward_model: ForwardModel,
    epsilons: np.ndarray,
    observed: np.ndarray,
) -> pymc.Model:
    parameter_size = len(inference_config.parameter_names)
    output_size = observed.size
    with pymc.Model() as model:
        parameters = construct_priors(inference_config)
        pymc.Simulator(
            InferenceVariable.PLUME_SUMMARY.value,
            forward_model,
            params=parameters,
            epsilon=epsilons,
            observed=observed,
            signature=f"({parameter_size})->({output_size})",
        )
    return model


def run_abc_smc(
    inference_config: InferenceConfig,
    pymc_config: PyMCConfig,
    forward_model: ForwardModel,
    epsilons: np.ndarray,
    observed: np.ndarray,
):
    """Sample the ABC posterior using the priors and seed in the run config."""
    model = build_abc_smc_model(
        inference_config=inference_config,
        forward_model=forward_model,
        epsilons=epsilons,
        observed=observed,
    )
    with model:
        return pymc.sample_smc(
            draws=pymc_config.draws,
            chains=pymc_config.chains,
            cores=pymc_config.cores,
            random_seed=pymc_config.seed,
            progressbar=pymc_config.progressbar,
            threshold=pymc_config.threshold,
            correlation_threshold=pymc_config.correlation_threshold,
        )
