from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from co2ipsimulator.field import GRFKernel, sample_grf_2d
from co2ipsimulator.model import GridMetadata

from .base import Scenario, _dome


@dataclass(frozen=True)
class GrfTopography:
    regional_amplitude_m: float = 30.0
    regional_sigma_m: float = 1080.0
    warp_sigma_m: float = 10.0
    warp_correlation_m: float = 1000.0
    warp_seed: int = 36
    kernel: str = "matern"
    nu: float = 1.5
    trap_centers: tuple[tuple[str, float, float], ...] = (
        ("L4", 2540.0, 2960.0),
        ("S3", 2290.0, 2875.0),
        ("S2", 2670.0, 2710.0),
        ("S1", 2500.0, 2500.0),
    )

    def unit_shapes(
        self, meta: GridMetadata, units: tuple[str, ...]
    ) -> dict[str, np.ndarray]:
        kernel = GRFKernel(
            sigma=self.warp_sigma_m,
            correlation_length_m=self.warp_correlation_m,
            kernel="matern" if self.kernel == "matern" else "gaussian",
            nu=self.nu,
        )
        centers = {unit: (cx, cy) for unit, cx, cy in self.trap_centers}
        shapes: dict[str, np.ndarray] = {}
        for i, unit in enumerate(units):
            warp = sample_grf_2d(
                meta.nx,
                meta.ny,
                dx=meta.dx,
                dy=meta.dy,
                kernel=kernel,
                rng=self.warp_seed + i,
            )
            shapes[unit] = (
                _dome(
                    meta,
                    *centers[unit],
                    self.regional_amplitude_m,
                    self.regional_sigma_m,
                )
                + warp
            )
        return shapes


GRF = Scenario(
    name="grf",
    topography=GrfTopography(),
    true_pth_kpa=(66.0, 52.0, 44.0),
    seal_log10_mobility=-10.0,
    annual_rate_mt=2.5,
    calibration_years=(2, 4, 6, 8, 10, 12, 14, 16),
    forecast_year=30,
)

GRF_NO_RATE_LIMIT = Scenario(
    name="grf_no_rate_limit",
    topography=GrfTopography(),
    true_pth_kpa=(60.0, 50.0, 45.0),
    seal_log10_mobility=None,
)
