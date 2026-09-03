from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from co2ipsimulator.model import GridMetadata, LayerKind, LayerProps

KPA_TO_PA = 1.0e3

ORDERED_SURFACE_NAMES = (
    "S_top",
    "Sh3_top",
    "Sh3_base",
    "Sh2_top",
    "Sh2_base",
    "Sh1_top",
    "Sh1_base",
    "S_bottom",
)

SURFACE_UNIT = {
    "S_top": "L4",
    "Sh3_top": "S3",
    "Sh3_base": "S3",
    "Sh2_top": "S2",
    "Sh2_base": "S2",
    "Sh1_top": "S1",
    "Sh1_base": "S1",
    "S_bottom": "S1",
}

STRUCTURAL_UNITS = tuple(
    dict.fromkeys(SURFACE_UNIT[name] for name in ORDERED_SURFACE_NAMES)
)

_LAYER_LAYOUT = (
    ("sand", "L4", "S_top", "Sh3_top"),
    ("shale", "Shale_3", "Sh3_top", "Sh3_base"),
    ("sand", "L3", "Sh3_base", "Sh2_top"),
    ("shale", "Shale_2", "Sh2_top", "Sh2_base"),
    ("sand", "L2", "Sh2_base", "Sh1_top"),
    ("shale", "Shale_1", "Sh1_top", "Sh1_base"),
    ("sand", "L1", "Sh1_base", "S_bottom"),
)


def _dome(
    meta: GridMetadata, cx: float, cy: float, amp: float, sig: float
) -> np.ndarray:
    X, Y = np.meshgrid(meta.x(), meta.y(), indexing="xy")
    return amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sig**2))


class Topography(Protocol):
    def unit_shapes(
        self, meta: GridMetadata, units: tuple[str, ...]
    ) -> dict[str, np.ndarray]: ...


@dataclass(frozen=True)
class Scenario:
    name: str
    topography: Topography

    nx: int = 51
    ny: int = 51
    xmin: float = 0.0
    xmax: float = 5000.0
    ymin: float = 0.0
    ymax: float = 5000.0

    top_depth_m: float = 800.0
    sand_thickness_m: float = 50.0
    shale_thickness_m: float = 6.0
    min_surface_gap_m: float = 1.0

    sand_porosity: float = 0.36
    shale_porosity: float = 0.0
    sand_pth_pa: float = 100.0
    co2_density_kg_m3: float = 700.0

    layer_order: tuple[str, ...] = ("L4", "L3", "L2", "L1")
    shale_layer_names: tuple[str, ...] = ("Shale_3", "Shale_2", "Shale_1")

    start_year: int = 0
    annual_rate_mt: float = 1.5
    # calibration_years: tuple[int, ...] = (5, 8, 10)
    calibration_years: tuple[int, ...] = (2, 4, 6, 8, 10)
    forecast_year: int = 25

    true_pth_kpa: tuple[float, ...] = (50.0, 60.0, 40.0)
    prior_center_pth_kpa: tuple[float, ...] = (50.0, 50.0, 50.0)

    well_xy: tuple[float, float] = (2500.0, 2500.0)

    # log10 of the effective seal mobility [lambda = k_eff/mu_g in m^2/(Pa*s)]
    # for this scenario's forward model. ``None`` = pure invasion percolation; a
    # float enables the finite-rate seal at ``lambda = 10**value`` (smaller =
    # stronger throttle = more lower-layer filling). Log-scaled because lambda
    # spans orders of magnitude. Used by ``render``.
    seal_log10_mobility: float | None = None

    ordered_surface_names: tuple[str, ...] = ORDERED_SURFACE_NAMES

    def metadata(self) -> GridMetadata:
        return GridMetadata(
            nx=self.nx,
            ny=self.ny,
            xmin=self.xmin,
            xmax=self.xmax,
            ymin=self.ymin,
            ymax=self.ymax,
        )

    def surface_offsets(self) -> dict[str, float]:
        sand, shale = self.sand_thickness_m, self.shale_thickness_m
        return {
            "S_top": 0.0,
            "Sh3_top": sand,
            "Sh3_base": sand + shale,
            "Sh2_top": 2 * sand + shale,
            "Sh2_base": 2 * sand + 2 * shale,
            "Sh1_top": 3 * sand + 2 * shale,
            "Sh1_base": 3 * sand + 3 * shale,
            "S_bottom": 4 * sand + 3 * shale,
        }

    def depth_surfaces(self) -> dict[str, np.ndarray]:
        meta = self.metadata()
        shapes = self.topography.unit_shapes(meta, STRUCTURAL_UNITS)
        offsets = self.surface_offsets()
        surfaces: dict[str, np.ndarray] = {}
        prev: np.ndarray | None = None
        for name in ORDERED_SURFACE_NAMES:
            depth = self.top_depth_m + offsets[name] - shapes[SURFACE_UNIT[name]]
            if prev is not None:
                depth = np.maximum(depth, prev + self.min_surface_gap_m)
            surfaces[name] = np.ascontiguousarray(depth)
            prev = surfaces[name]
        return surfaces

    def layer_stack(self) -> list[LayerProps]:
        pth_by_shale = dict(zip(self.shale_layer_names, self.true_pth_kpa))

        def make(kind: str, name: str, top: str, base: str) -> LayerProps:
            if kind == "sand":
                return LayerProps(
                    name=name,
                    kind=LayerKind.SAND,
                    top_surface=top,
                    base_surface=base,
                    density_co2=self.co2_density_kg_m3,
                    pth_pa=self.sand_pth_pa,
                    porosity=self.sand_porosity,
                )
            return LayerProps(
                name=name,
                kind=LayerKind.SHALE,
                top_surface=top,
                base_surface=base,
                density_co2=self.co2_density_kg_m3,
                pth_pa=pth_by_shale[name] * KPA_TO_PA,
                porosity=self.shale_porosity,
            )

        return [make(*entry) for entry in _LAYER_LAYOUT]

    def true_pth_kpa_array(self) -> np.ndarray:
        return np.asarray(self.true_pth_kpa, dtype=np.float64)

    def prior_center_pth_kpa_array(self) -> np.ndarray:
        return np.asarray(self.prior_center_pth_kpa, dtype=np.float64)

    def annual_rates_mt(self, end_year: int) -> tuple[float, ...]:
        n = end_year - self.start_year + 1
        if n <= 0:
            raise ValueError(f"end_year {end_year} before start {self.start_year}")
        return (self.annual_rate_mt,) * n
