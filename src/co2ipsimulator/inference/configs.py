from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .priors import LogNormalPrior, PriorDistribution


class InferenceVariable(StrEnum):
    PARAMETERS = "parameters"
    THRESHOLD_PRESSURE = "pth_kpa"
    PLUME_SUMMARY = "plume_summary"


class ConfigSection(StrEnum):
    PYMC = "pymc"
    INFERENCE = "inference"
    SUMMARY = "summary"
    EXTRAS = "extras"


class SummaryMode(StrEnum):
    MASS = "mass"
    MASS_AND_FOOTPRINTS = "mass-and-footprints"
    TRANSPORT = "transport"


class MassMode(StrEnum):
    FRACTION = "fraction"
    MT = "mt"


class MassMeasure(StrEnum):
    STORED_MASS = "stored_mass"
    FOOTPRINT_AREA = "footprint_area"


@dataclass(frozen=True)
class ThresholdPressurePrior:
    layer_name: str
    distribution: PriorDistribution = LogNormalPrior()

    @property
    def variable_name(self) -> str:
        return f"{self.layer_name}_{InferenceVariable.THRESHOLD_PRESSURE.value}"


@dataclass(frozen=True)
class RateLimitPrior:
    name: str = "lambda_s"
    distribution: PriorDistribution = LogNormalPrior()


@dataclass(frozen=True)
class DetectionPrior:
    """Treat the minimum detectable column height as an unknown of the inference.

    Left unset the detection threshold is a fixed property of the forward
    configuration. Set, it becomes the last parameter of the vector, so the
    observation operator is marginalised over rather than assumed.
    """

    name: str = "h_det"
    distribution: PriorDistribution = LogNormalPrior()


@dataclass(frozen=True)
class InferenceConfig:
    pth_priors: tuple[ThresholdPressurePrior, ...]
    rate_limit_prior: RateLimitPrior | None = None
    detection_prior: DetectionPrior | None = None

    @property
    def pth_layer_names(self) -> tuple[str, ...]:
        return tuple(prior.layer_name for prior in self.pth_priors)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        rate_limit_names = (
            () if self.rate_limit_prior is None else (self.rate_limit_prior.name,)
        )
        detection_names = (
            () if self.detection_prior is None else (self.detection_prior.name,)
        )
        return self.pth_layer_names + rate_limit_names + detection_names


@dataclass(frozen=True)
class SummaryConfig:
    mode: SummaryMode = SummaryMode.MASS
    mass_mode: MassMode = MassMode.FRACTION
    mass_measure: MassMeasure = MassMeasure.STORED_MASS
    mass_relative_epsilon: float = 0.25
    mass_epsilon_floor: float = 0.03
    moment_relative_epsilon: float = 0.9
    moment_epsilon_floor: float = 0.01
    # Tolerance of the transport summary, as a fraction of the cost charged to a
    # plume that misses the observed one entirely (half the domain diagonal). It
    # sets how close the simulated outlines are expected to get, so it should be
    # of the order of the mismatch the model can actually reach.
    transport_relative_epsilon: float = 0.12


@dataclass(frozen=True)
class PyMCConfig:
    draws: int = 200
    chains: int = 2
    cores: int = 1
    seed: int = 0
    progressbar: bool = True
    threshold: float = 0.5
    correlation_threshold: float = 0.01


@dataclass(frozen=True)
class RunConfig:
    inference: InferenceConfig
    summary: SummaryConfig
    pymc: PyMCConfig
    extras: dict[str, Any]
