from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pymc

from co2ipsimulator.inference import (
    InferenceConfig,
    InferenceVariable,
    LogNormalPrior,
    PyMCConfig,
    RateLimitPrior,
    ThresholdPressurePrior,
    UniformPrior,
    build_abc_smc_model,
    evaluate_posterior_predictive,
    posterior_samples,
    run_abc_smc,
    summarize_posterior,
)


def _forward(rng, parameters, size=None):
    summary = parameters / 100.0
    if size is None or size == ():
        return summary
    sample_shape = tuple(np.atleast_1d(size).astype(int))
    return np.broadcast_to(summary, sample_shape + summary.shape).copy()


def _inference_config():
    return InferenceConfig(
        (
            ThresholdPressurePrior("a", LogNormalPrior(mu=4.0, sigma=0.47)),
            ThresholdPressurePrior("b", UniformPrior(lower=20.0, upper=80.0)),
        ),
        RateLimitPrior("lambda_s", UniformPrior(lower=0.1, upper=0.2)),
    )


def test_abc_model_preserves_independent_prior_families():
    inference = _inference_config()
    model = build_abc_smc_model(
        inference,
        _forward,
        epsilons=np.full(3, 0.1),
        observed=np.array([0.4, 0.6, 1.0e-3]),
    )

    pth_name = InferenceVariable.THRESHOLD_PRESSURE.value
    parameter_name = InferenceVariable.PARAMETERS.value
    with model:
        pth = pymc.draw(model[pth_name], draws=4000, random_seed=1)
        parameters = pymc.draw(model[parameter_name], draws=10, random_seed=2)

    assert np.isclose(np.median(pth[:, 0]), np.exp(4.0), rtol=0.04)
    assert np.isclose(np.std(np.log(pth[:, 0])), 0.47, rtol=0.04)
    assert 47.0 < np.mean(pth[:, 1]) < 53.0
    assert parameters.shape == (10, 3)


def test_abc_smc_returns_named_joint_posterior_samples():
    inference = _inference_config()
    idata = run_abc_smc(
        inference,
        PyMCConfig(draws=30, chains=1, cores=1, seed=4, progressbar=False),
        _forward,
        epsilons=np.array([0.08, 0.08, 0.001]),
        observed=np.array([0.4, 0.6, 0.0015]),
    )

    samples = posterior_samples(idata)
    summary = summarize_posterior(idata, inference.parameter_names)
    assert samples.shape == (30, 3)
    assert np.all(np.isfinite(samples))
    assert summary.sample_count == 30
    assert [estimate.name for estimate in summary.estimates] == ["a", "b", "lambda_s"]


def test_posterior_predictive_preserves_joint_draws():
    values = np.array([[[20.0, 80.0], [40.0, 60.0]], [[60.0, 40.0], [80.0, 20.0]]])
    variable = InferenceVariable.PARAMETERS.value
    idata = SimpleNamespace(posterior={variable: SimpleNamespace(values=values)})

    prediction = evaluate_posterior_predictive(
        idata,
        _forward,
        ("first", "second"),
        credible_mass=0.5,
    )

    assert prediction.samples.shape == (4, 2)
    assert np.allclose(prediction.mean, [0.5, 0.5])
    assert np.allclose(prediction.samples.sum(axis=1), 1.0)
