from pathlib import Path

import numpy as np
import pytest

from co2ipsimulator.inference import (
    ForwardModelConfig,
    InferenceConfig,
    MassMode,
    MassSummary,
    PyMCConfig,
    SummaryStatistics,
    ThresholdPressurePrior,
    UniformPrior,
    build_forward_model,
    posterior_samples,
    run_abc_smc,
    run_forward_model,
)
from co2ipsimulator.model import (
    build_stratigraphic_grid,
    build_trapfill,
    load_rms_polygon,
)


def _cross_layer_sand_contacts(grid) -> int:
    layer_id = grid.layer_id
    sand = grid.active_mask & grid.is_sand

    def count(first, second, first_layer, second_layer):
        return int((first & second & (first_layer != second_layer)).sum())

    return (
        count(sand[:, :, :-1], sand[:, :, 1:], layer_id[:, :, :-1], layer_id[:, :, 1:])
        + count(sand[:-1], sand[1:], layer_id[:-1], layer_id[1:])
        + count(sand[:, :-1], sand[:, 1:], layer_id[:, :-1], layer_id[:, 1:])
    )


def test_simulator_output_can_drive_abc_smc(layered_dome):
    """Minimal end-to-end contract for the simulator/inference coupling."""
    inference = InferenceConfig(
        (ThresholdPressurePrior("shale", UniformPrior(lower=30.0, upper=70.0)),)
    )
    statistics = SummaryStatistics(
        (2007,),
        (MassSummary(("upper_sand", "lower_sand"), mode=MassMode.FRACTION),),
    )
    config = ForwardModelConfig(
        depth_surfaces=layered_dome.surfaces,
        layer_stack=layered_dome.stack,
        metadata=layered_dome.metadata,
        inference=inference,
        source_xy=layered_dome.source_xy,
        annual_masses_kg=(2.0e8,) * 8,
        start_year=2000,
    )
    forward = build_forward_model(config, statistics)
    observed = run_forward_model(config, statistics, np.array([50.0]))

    idata = run_abc_smc(
        inference,
        PyMCConfig(draws=12, chains=1, cores=1, seed=8, progressbar=False),
        forward,
        epsilons=np.full(observed.shape, 0.15),
        observed=observed,
    )

    samples = posterior_samples(idata)
    assert samples.shape == (12, 1)
    assert np.all(np.isfinite(samples))
    assert np.all((30.0 <= samples) & (samples <= 70.0))


def test_sleipner_geometry_and_trapfill_conserve_mass():
    """Validate the published field geometry when its separately licensed data exist."""
    from examples.sleipner.config import (
        ANNUAL_RATES_MT,
        DEPTH_SURFACES,
        INJECTION_SITE_FILE,
    )
    from examples.sleipner.data_loader import load_sleipner_surfaces
    from examples.sleipner.layer_stack import sleipner_layer_stack
    from examples.sleipner.render import CAPROCK_PTH_PA

    if not Path(DEPTH_SURFACES).exists():
        pytest.skip("Sleipner depth surfaces not available")

    surfaces, metadata = load_sleipner_surfaces(DEPTH_SURFACES)
    stack = sleipner_layer_stack()
    grid = build_stratigraphic_grid(surfaces, stack, metadata, dz=1.0)
    site = load_rms_polygon(INJECTION_SITE_FILE)
    source_xy = tuple(float(value) for value in site.mean(axis=0))
    trapfill = build_trapfill(
        surfaces,
        stack,
        metadata,
        source_xy=source_xy,
        top_seal_pth_pa=CAPROCK_PTH_PA,
    )
    injected = sum(ANNUAL_RATES_MT) * 1e9
    result = trapfill.fill(injected)

    assert (metadata.nx, metadata.ny) == (65, 119)
    assert grid.shape == (65, 119, 324)
    assert np.allclose(grid.z_bot, grid.z_top + grid.dz)
    assert _cross_layer_sand_contacts(grid) == 0
    assert float(grid.pth[grid.is_shale & grid.active_mask].min()) > float(
        grid.pth[grid.is_sand & grid.active_mask].max()
    )
    assert result.escaped_kg <= 1e-6 * injected
    assert abs(result.stored_kg + result.escaped_kg - injected) <= 1e-6 * injected
