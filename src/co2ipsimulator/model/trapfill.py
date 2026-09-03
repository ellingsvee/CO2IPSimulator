from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from co2ipsimulator import rust

from .properties import GridMetadata, LayerProps

SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0
TOP_SEAL_THICKNESS_M = 50.0


@dataclass(frozen=True)
class TrapFillResult:
    mass_per_layer: dict[str, float]
    escaped_kg: float
    stalled_kg: float = 0.0

    @property
    def stored_kg(self) -> float:
        return float(sum(self.mass_per_layer.values()))


@dataclass(frozen=True)
class StepDiagnostics:
    """One injection interval.

    ``converged`` is false when the interval exhausted the substep budget before reaching ``time_rtol``, so ``estimated_relative_error`` is the error the run actually attained rather than the one that was asked for.
    """

    accepted_substeps: int
    estimated_relative_error: float
    converged: bool

    @staticmethod
    def worst(diagnostics: Sequence[StepDiagnostics]) -> StepDiagnostics:
        return max(
            diagnostics,
            key=lambda d: (not d.converged, d.estimated_relative_error),
        )


def describe_convergence(
    convergence: StepDiagnostics | None, result: TrapFillResult
) -> str:
    """The part of a run's report that is only worth printing when it is bad."""
    notes = []
    if convergence is not None and not convergence.converged:
        notes.append(
            f"NOT CONVERGED: attained {convergence.estimated_relative_error:.2e} "
            f"at the {convergence.accepted_substeps}-substep cap"
        )
    if result.stalled_kg > 0.0:
        notes.append(f"{result.stalled_kg / 1e9:.3f} Mt stalled in the spill graph")
    return "".join(f", {note}" for note in notes)


@dataclass(frozen=True)
class TrapFill:
    _sim: object
    layer_names: tuple[str, ...]
    time_rtol: float = 1.0e-4

    def fill(self, mass_kg: float) -> TrapFillResult:
        mass_per_layer, escaped, stalled = self._sim.fill(float(mass_kg))
        return TrapFillResult(
            mass_per_layer=dict(
                zip(self.layer_names, (float(m) for m in mass_per_layer))
            ),
            escaped_kg=float(escaped),
            stalled_kg=float(stalled),
        )

    def column_heights(self, mass_kg: float) -> dict[str, np.ndarray]:
        columns = self._sim.column_heights(float(mass_kg))
        return dict(zip(self.layer_names, columns))

    def reset(self) -> None:
        self._sim.reset()

    def update_seals(self, seals: SealFields) -> None:
        self._sim.update_seals(seals.critical_height, seals.mobility)

    def step(self, mass_kg: float) -> StepDiagnostics:
        accepted_substeps, error = self._sim.step(float(mass_kg))
        converged = error <= self.time_rtol
        if not converged:
            warnings.warn(
                "time integration reached the substep cap before meeting time_rtol; "
                "the attained error is in StepDiagnostics.estimated_relative_error",
                RuntimeWarning,
                stacklevel=2,
            )
        return StepDiagnostics(int(accepted_substeps), float(error), converged)

    def _mass_per_layer(self) -> dict[str, float]:
        return dict(
            zip(self.layer_names, (float(m) for m in self._sim.mass_per_layer()))
        )

    def state_result(self) -> TrapFillResult:
        return TrapFillResult(
            mass_per_layer=self._mass_per_layer(),
            escaped_kg=float(self._sim.escaped_kg()),
            stalled_kg=float(self._sim.stalled_kg()),
        )

    def state_column_heights(self) -> dict[str, np.ndarray]:
        return dict(zip(self.layer_names, self._sim.state_column_heights()))

    def run_schedule(
        self, annual_masses: Sequence[float]
    ) -> tuple[TrapFillResult, StepDiagnostics]:
        """Reset the simulator and inject one mass, in kilograms, per time step."""
        self._sim.reset()
        diagnostics = [self.step(float(mass)) for mass in annual_masses]
        return self.state_result(), StepDiagnostics.worst(diagnostics)


def _sand_layers(stack):
    sands = []
    for i, layer in enumerate(stack):
        if layer.is_shale:
            continue
        seal = stack[i - 1] if i > 0 and stack[i - 1].is_shale else None
        sands.append((layer, seal))
    sands.reverse()
    return sands


@dataclass(frozen=True)
class SealFields:
    """Seal properties of every sand unit, bottom-up.

    A seal is uniform over a unit, so it needs one critical column height, one thickness and one mobility per unit.
    """

    critical_height: np.ndarray
    thickness: np.ndarray
    mobility: np.ndarray | None


def seal_fields(
    layer_stack: Sequence[LayerProps],
    depth_surfaces: dict[str, np.ndarray] | None = None,
    *,
    density_brine: float = 1000.0,
    gravity: float = 9.81,
    seal_log10_mobility: float | None = None,
    top_seal_pth_pa: float | None = None,
) -> SealFields:
    sands = _sand_layers(tuple(layer_stack))
    critical_height = [
        np.inf
        if seal is None
        else seal.pth_pa / ((density_brine - layer.density_co2) * gravity)
        for layer, seal in sands
    ]
    if top_seal_pth_pa is not None:
        top_layer = sands[-1][0]
        critical_height[-1] = top_seal_pth_pa / (
            (density_brine - top_layer.density_co2) * gravity
        )

    if depth_surfaces is None:
        thickness = [np.inf for _ in sands]
    else:
        thickness = [
            TOP_SEAL_THICKNESS_M
            if seal is None
            else float(
                np.mean(
                    depth_surfaces[seal.base_surface] - depth_surfaces[seal.top_surface]
                )
            )
            for _, seal in sands
        ]

    mobility = (
        None
        if seal_log10_mobility is None
        else np.full(len(sands), 10.0 ** float(seal_log10_mobility))
    )
    return SealFields(
        critical_height=np.asarray(critical_height, dtype=np.float64),
        thickness=np.asarray(thickness, dtype=np.float64),
        mobility=mobility,
    )


def build_trapfill(
    depth_surfaces: dict[str, np.ndarray],
    layer_stack: Sequence[LayerProps],
    metadata: GridMetadata,
    *,
    source_xy: tuple[float, float] | Sequence[tuple[float, float]],
    density_brine: float = 1000.0,
    gravity: float = 9.81,
    connate_water_saturation: float = 0.30,
    usediags: bool = True,
    seal_log10_mobility: float | None = None,
    top_seal_pth_pa: float | None = None,
    step_seconds: float = SECONDS_PER_YEAR,
    time_rtol: float = 1.0e-4,
    max_substeps: int = 16_384,
) -> TrapFill:
    """Build a trap-filling simulator for one or more injection locations.

    Depths and coordinates are in metres, pressures in pascals, densities in kilograms per cubic metre, injected masses in kilograms, and time in seconds. Setting ``seal_log10_mobility`` enables finite-rate seal transfer. Otherwise each requested mass is filled quasi-statically.
    """
    stack = tuple(layer_stack)
    sands = _sand_layers(stack)

    tops = np.stack(
        [
            np.ascontiguousarray(depth_surfaces[layer.top_surface].T, dtype=np.float64)
            for layer, _ in sands
        ]
    )
    bases = np.stack(
        [
            np.ascontiguousarray(depth_surfaces[layer.base_surface].T, dtype=np.float64)
            for layer, _ in sands
        ]
    )
    seals = seal_fields(
        stack,
        depth_surfaces,
        density_brine=density_brine,
        gravity=gravity,
        seal_log10_mobility=seal_log10_mobility,
        top_seal_pth_pa=top_seal_pth_pa,
    )
    density = np.array([layer.density_co2 for layer, _ in sands], dtype=np.float64)
    porosity = np.array([layer.porosity for layer, _ in sands], dtype=np.float64)
    delta_rho_g = np.array(
        [(density_brine - layer.density_co2) * gravity for layer, _ in sands],
        dtype=np.float64,
    )

    first = source_xy[0]
    xy_list = [source_xy] if np.isscalar(first) else list(source_xy)
    sources = [
        (
            max(0, min(metadata.nx - 1, int(round((x - metadata.xmin) / metadata.dx)))),
            max(0, min(metadata.ny - 1, int(round((y - metadata.ymin) / metadata.dy)))),
        )
        for x, y in xy_list
    ]

    sim = rust.TrapFill(
        tops,
        bases,
        seals.critical_height,
        seals.thickness,
        density,
        porosity,
        delta_rho_g,
        metadata.dx,
        metadata.dy,
        sources,
        float(connate_water_saturation),
        usediags,
        seals.mobility,
        float(step_seconds),
        float(time_rtol),
        int(max_substeps),
    )
    return TrapFill(
        _sim=sim,
        layer_names=tuple(layer.name for layer, _ in sands),
        time_rtol=float(time_rtol),
    )
