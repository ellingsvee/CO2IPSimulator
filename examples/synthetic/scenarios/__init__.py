from __future__ import annotations

from .base import Scenario, Topography
from .dome import DOME, DOME_NO_RATE_LIMIT, DomeTopography
from .grf import GRF, GRF_NO_RATE_LIMIT, GrfTopography

SCENARIOS = {
    scenario.name: scenario
    for scenario in (DOME, DOME_NO_RATE_LIMIT, GRF, GRF_NO_RATE_LIMIT)
}
SCENARIO_NAMES = tuple(SCENARIOS)


def get_scenario(name: str) -> Scenario:
    try:
        return SCENARIOS[name]
    except KeyError:
        raise ValueError(
            f"unknown scenario {name!r}; choose from {SCENARIO_NAMES}"
        ) from None


__all__ = [
    "DOME",
    "DOME_NO_RATE_LIMIT",
    "GRF",
    "GRF_NO_RATE_LIMIT",
    "SCENARIOS",
    "SCENARIO_NAMES",
    "Scenario",
    "Topography",
    "DomeTopography",
    "GrfTopography",
    "get_scenario",
]
