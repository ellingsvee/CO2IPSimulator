from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from co2ipsimulator.model import GridMetadata

from .base import Scenario, _dome


@dataclass(frozen=True)
class DomeTopography:
    amplitude_m: float = 50.0
    sigma_m: float = 1500.0
    center_xy: tuple[float, float] = (2500.0, 2500.0)

    def unit_shapes(
        self, meta: GridMetadata, units: tuple[str, ...]
    ) -> dict[str, np.ndarray]:
        dome = _dome(
            meta, self.center_xy[0], self.center_xy[1], self.amplitude_m, self.sigma_m
        )
        return {unit: dome for unit in units}


DOME = Scenario(
    name="dome",
    topography=DomeTopography(amplitude_m=90.0, sigma_m=1100.0),
    true_pth_kpa=(60.0, 50.0, 40.0),
    seal_log10_mobility=-11.0,
    annual_rate_mt=2.75,
    calibration_years=(2, 4, 6, 8, 10, 12, 14, 16),
    forecast_year=30,
)

DOME_NO_RATE_LIMIT = Scenario(
    name="dome_no_rate_limit",
    topography=DomeTopography(),
    true_pth_kpa=(55.0, 45.0, 35.0),
    seal_log10_mobility=None,
)
