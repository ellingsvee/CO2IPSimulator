from __future__ import annotations

import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .configs import (
    ConfigSection,
    DetectionPrior,
    InferenceConfig,
    MassMeasure,
    MassMode,
    PyMCConfig,
    RateLimitPrior,
    RunConfig,
    SummaryConfig,
    SummaryMode,
    ThresholdPressurePrior,
)
from .priors import (
    LogNormalPrior,
    NormalPrior,
    PriorDistribution,
    PriorFamily,
    UniformPrior,
)

_DEFAULT_PYMC = PyMCConfig()
_DEFAULT_SUMMARY = SummaryConfig()
_DEFAULT_LOGNORMAL = LogNormalPrior()
_DEFAULT_NORMAL = NormalPrior()


def _lognormal(values: dict[str, Any]) -> PriorDistribution:
    return LogNormalPrior(
        mu=float(values.get("mu", _DEFAULT_LOGNORMAL.mu)),
        sigma=float(values.get("sigma", _DEFAULT_LOGNORMAL.sigma)),
    )


def _normal(values: dict[str, Any]) -> PriorDistribution:
    return NormalPrior(
        mu=float(values.get("mu", _DEFAULT_NORMAL.mu)),
        sigma=float(values.get("sigma", _DEFAULT_NORMAL.sigma)),
    )


def _uniform(values: dict[str, Any]) -> PriorDistribution:
    return UniformPrior(lower=float(values["lower"]), upper=float(values["upper"]))


_PRIOR_LOADERS: dict[PriorFamily, Callable[[dict[str, Any]], PriorDistribution]] = {
    PriorFamily.LOG_NORMAL: _lognormal,
    PriorFamily.NORMAL: _normal,
    PriorFamily.UNIFORM: _uniform,
}

_PRIOR_FIELDS: dict[PriorFamily, tuple[str, ...]] = {
    PriorFamily.LOG_NORMAL: ("mu", "sigma"),
    PriorFamily.NORMAL: ("mu", "sigma"),
    PriorFamily.UNIFORM: ("lower", "upper"),
}


def _reject_unknown(section: str, values: dict[str, Any], known: set[str]) -> None:
    """Fail on a key the loader would otherwise ignore.

    A silently dropped key reads as a setting that is in force when it is not,
    so a stale name survives a refactor and the default it masks is never seen.
    ``extras`` is deliberately open and is not checked.
    """
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(
            f"unknown key(s) in [{section}]: {', '.join(unknown)}; "
            f"accepted keys are {', '.join(sorted(known))}"
        )


def _prior_keys(family: PriorFamily, *extra: str) -> set[str]:
    return {"distribution", *extra} | set(_PRIOR_FIELDS[family])


def _load_prior(values: dict[str, Any]) -> PriorDistribution:
    family = PriorFamily(values.get("distribution", PriorFamily.LOG_NORMAL))
    return _PRIOR_LOADERS[family](values)


def _load_pth_priors(values: dict[str, Any]) -> tuple[ThresholdPressurePrior, ...]:
    family = PriorFamily(values.get("distribution", PriorFamily.LOG_NORMAL))
    _reject_unknown(
        "inference.threshold_pressure", values, _prior_keys(family, "layer_names")
    )
    fields = _PRIOR_FIELDS[family]
    rows = zip(
        values["layer_names"],
        *(values[field] for field in fields),
        strict=True,
    )
    return tuple(
        ThresholdPressurePrior(
            layer_name=str(row[0]),
            distribution=_load_prior(
                {
                    "distribution": family,
                    **dict(zip(fields, row[1:], strict=True)),
                }
            ),
        )
        for row in rows
    )


def _load_named_prior(section: str, values: dict[str, Any]) -> PriorDistribution:
    family = PriorFamily(values.get("distribution", PriorFamily.LOG_NORMAL))
    _reject_unknown(section, values, _prior_keys(family, "name"))
    return _load_prior(values)


def _load_inference(values: dict[str, Any]) -> InferenceConfig:
    _reject_unknown(
        "inference", values, {"threshold_pressure", "rate_limit", "detection"}
    )
    pth_priors = _load_pth_priors(values["threshold_pressure"])
    rate_limit_values = values.get("rate_limit")
    rate_limit_prior = (
        None
        if rate_limit_values is None
        else RateLimitPrior(
            name=str(rate_limit_values.get("name", "lambda_s")),
            distribution=_load_named_prior("inference.rate_limit", rate_limit_values),
        )
    )
    detection_values = values.get("detection")
    detection_prior = (
        None
        if detection_values is None
        else DetectionPrior(
            name=str(detection_values.get("name", "h_det")),
            distribution=_load_named_prior("inference.detection", detection_values),
        )
    )
    return InferenceConfig(pth_priors, rate_limit_prior, detection_prior)


_SUMMARY_KEYS = {
    "mode",
    "mass_mode",
    "mass_measure",
    "mass_relative_epsilon",
    "mass_epsilon_floor",
    "moment_relative_epsilon",
    "moment_epsilon_floor",
    "transport_relative_epsilon",
}

_PYMC_KEYS = {
    "draws",
    "chains",
    "cores",
    "seed",
    "progressbar",
    "threshold",
    "correlation_threshold",
}


def _load_summary(values: dict[str, Any]) -> SummaryConfig:
    _reject_unknown("summary", values, _SUMMARY_KEYS)
    return SummaryConfig(
        mode=SummaryMode(values.get("mode", _DEFAULT_SUMMARY.mode)),
        mass_mode=MassMode(values.get("mass_mode", _DEFAULT_SUMMARY.mass_mode)),
        mass_measure=MassMeasure(
            values.get("mass_measure", _DEFAULT_SUMMARY.mass_measure)
        ),
        mass_relative_epsilon=float(
            values.get("mass_relative_epsilon", _DEFAULT_SUMMARY.mass_relative_epsilon)
        ),
        mass_epsilon_floor=float(
            values.get("mass_epsilon_floor", _DEFAULT_SUMMARY.mass_epsilon_floor)
        ),
        moment_relative_epsilon=float(
            values.get(
                "moment_relative_epsilon", _DEFAULT_SUMMARY.moment_relative_epsilon
            )
        ),
        moment_epsilon_floor=float(
            values.get("moment_epsilon_floor", _DEFAULT_SUMMARY.moment_epsilon_floor)
        ),
        transport_relative_epsilon=float(
            values.get(
                "transport_relative_epsilon",
                _DEFAULT_SUMMARY.transport_relative_epsilon,
            )
        ),
    )


def _load_pymc(values: dict[str, Any]) -> PyMCConfig:
    _reject_unknown("pymc", values, _PYMC_KEYS)
    return PyMCConfig(
        draws=int(values.get("draws", _DEFAULT_PYMC.draws)),
        chains=int(values.get("chains", _DEFAULT_PYMC.chains)),
        cores=int(values.get("cores", _DEFAULT_PYMC.cores)),
        seed=int(values.get("seed", _DEFAULT_PYMC.seed)),
        progressbar=bool(values.get("progressbar", _DEFAULT_PYMC.progressbar)),
        threshold=float(values.get("threshold", _DEFAULT_PYMC.threshold)),
        correlation_threshold=float(
            values.get("correlation_threshold", _DEFAULT_PYMC.correlation_threshold)
        ),
    )


def load_run_config(path: Path | str) -> RunConfig:
    """Load and validate a TOML experiment configuration."""
    with Path(path).open("rb") as config_file:
        values = tomllib.load(config_file)
    _reject_unknown(str(path), values, {str(section) for section in ConfigSection})
    inference = _load_inference(values[ConfigSection.INFERENCE])
    summary = _load_summary(values.get(ConfigSection.SUMMARY, {}))
    pymc = _load_pymc(values.get(ConfigSection.PYMC, {}))
    extras = dict(values.get(ConfigSection.EXTRAS, {}))
    return RunConfig(inference, summary, pymc, extras)
