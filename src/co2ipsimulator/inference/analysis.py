from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .configs import InferenceVariable
from .forward import ForwardModel


@dataclass(frozen=True)
class ParameterEstimate:
    name: str
    mean: float
    standard_deviation: float
    median: float
    lower: float
    upper: float


@dataclass(frozen=True)
class PosteriorSummary:
    sample_count: int
    credible_mass: float
    estimates: tuple[ParameterEstimate, ...]


@dataclass(frozen=True)
class PosteriorPrediction:
    names: tuple[str, ...]
    samples: np.ndarray
    mean: np.ndarray
    median: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def posterior_samples(idata) -> np.ndarray:
    values = idata.posterior[InferenceVariable.PARAMETERS.value].values
    sample_count = values.shape[0] * values.shape[1]
    parameter_count = int(np.prod(values.shape[2:]))
    return values.reshape(sample_count, parameter_count)


def summarize_posterior(
    idata,
    parameter_names: tuple[str, ...],
    *,
    credible_mass: float = 0.90,
) -> PosteriorSummary:
    samples = posterior_samples(idata)
    tail = (1.0 - credible_mass) / 2.0
    quantiles = np.quantile(samples, [tail, 0.5, 1.0 - tail], axis=0).T
    estimates = tuple(
        ParameterEstimate(
            name=name,
            mean=float(values.mean()),
            standard_deviation=float(values.std()),
            median=float(bounds[1]),
            lower=float(bounds[0]),
            upper=float(bounds[2]),
        )
        for name, values, bounds in zip(
            parameter_names, samples.T, quantiles, strict=True
        )
    )
    return PosteriorSummary(samples.shape[0], credible_mass, estimates)


def evaluate_posterior_predictive(
    idata,
    forward_model: ForwardModel,
    summary_names: tuple[str, ...],
    *,
    draws: int = 100,
    credible_mass: float = 0.90,
    seed: int = 0,
) -> PosteriorPrediction:
    parameters = posterior_samples(idata)
    rng = np.random.default_rng(seed)
    indices = rng.choice(
        parameters.shape[0], min(draws, parameters.shape[0]), replace=False
    )
    predictions = np.stack(
        [forward_model(rng, parameters[index], ()) for index in indices]
    ).astype(np.float64, copy=False)
    tail = (1.0 - credible_mass) / 2.0
    quantiles = np.quantile(predictions, [tail, 0.5, 1.0 - tail], axis=0)
    return PosteriorPrediction(
        names=tuple(summary_names),
        samples=predictions,
        mean=predictions.mean(axis=0),
        median=quantiles[1],
        lower=quantiles[0],
        upper=quantiles[2],
    )
