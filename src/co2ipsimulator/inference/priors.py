from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np
import pymc
from pytensor.tensor.variable import TensorVariable


class PriorFamily(StrEnum):
    LOG_NORMAL = "lognormal"
    NORMAL = "normal"
    UNIFORM = "uniform"


class PriorDistribution(Protocol):
    def build(self, name: str) -> TensorVariable: ...

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray: ...


@dataclass(frozen=True)
class LogNormalPrior:
    mu: float = 4.0
    sigma: float = 0.47

    def build(self, name: str) -> TensorVariable:
        return pymc.LogNormal(name, mu=self.mu, sigma=self.sigma)

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.lognormal(self.mu, self.sigma, size)


@dataclass(frozen=True)
class NormalPrior:
    mu: float = 0.0
    sigma: float = 1.0

    def build(self, name: str) -> TensorVariable:
        return pymc.Normal(name, mu=self.mu, sigma=self.sigma)

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.normal(self.mu, self.sigma, size)


@dataclass(frozen=True)
class UniformPrior:
    lower: float
    upper: float

    def build(self, name: str) -> TensorVariable:
        return pymc.Uniform(name, lower=self.lower, upper=self.upper)

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.uniform(self.lower, self.upper, size)
