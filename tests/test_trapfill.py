from __future__ import annotations

from dataclasses import replace

import pytest

from co2ipsimulator.model import build_trapfill

from examples.synthetic.scenarios import DOME_NO_RATE_LIMIT, GRF_NO_RATE_LIMIT

LAYERS = ("L1", "L2", "L3")


def _trap_graph(scenario):
    injected = sum(scenario.annual_rates_mt(10)) * 1e9
    tf = build_trapfill(
        scenario.depth_surfaces(),
        scenario.layer_stack(),
        scenario.metadata(),
        source_xy=scenario.well_xy,
    )
    return tf.fill(injected), injected


@pytest.mark.parametrize("scenario", [DOME_NO_RATE_LIMIT, GRF_NO_RATE_LIMIT])
def test_trapfill_conserves_mass_in_closed_domain(scenario):
    state, injected = _trap_graph(scenario)
    assert abs(state.stored_kg + state.escaped_kg - injected) <= 1e-6 * injected
    assert state.escaped_kg <= 1e-6 * injected


@pytest.mark.parametrize("scenario", [DOME_NO_RATE_LIMIT, GRF_NO_RATE_LIMIT])
def test_trapfill_column_heights_reconstruct_layer_mass(scenario):
    injected = sum(scenario.annual_rates_mt(10)) * 1e9
    tf = build_trapfill(
        scenario.depth_surfaces(),
        scenario.layer_stack(),
        scenario.metadata(),
        source_xy=scenario.well_xy,
    )
    state = tf.fill(injected)
    columns = tf.column_heights(injected)
    meta = scenario.metadata()
    area = meta.dx * meta.dy
    density = {layer.name: layer.density_co2 for layer in scenario.layer_stack()}
    porosity = {layer.name: layer.porosity for layer in scenario.layer_stack()}
    for name in LAYERS:
        kg_per_geom = density[name] * porosity[name] * (1.0 - 0.30)
        integral = float(columns[name].sum()) * area * kg_per_geom
        assert abs(integral - state.mass_per_layer[name]) <= 1e-6 * max(
            1.0, state.mass_per_layer[name]
        )


def _annual(scenario):
    return [r * 1e9 for r in scenario.annual_rates_mt(10)]


def _rate_limited_run(scenario, log10_mobility):
    tf = build_trapfill(
        scenario.depth_surfaces(),
        scenario.layer_stack(),
        scenario.metadata(),
        source_xy=scenario.well_xy,
        seal_log10_mobility=log10_mobility,
    )
    return tf.run_schedule(_annual(scenario))[0]


def test_trapfill_rate_limited_run_conserves_mass():
    scenario = DOME_NO_RATE_LIMIT
    injected = sum(_annual(scenario))
    res = _rate_limited_run(scenario, -11.0)
    assert abs(res.stored_kg + res.escaped_kg - injected) <= 1e-6 * injected


def test_trapfill_finite_rate_holds_more_below():
    scenario = DOME_NO_RATE_LIMIT
    off, injected = _trap_graph(scenario)
    on = _rate_limited_run(scenario, -11.0)
    assert abs(on.stored_kg + on.escaped_kg - injected) < 1e-6 * injected
    assert on.mass_per_layer["L1"] > 1.5 * off.mass_per_layer["L1"]


def test_trapfill_retention_monotone_in_mobility():
    scenario = DOME_NO_RATE_LIMIT

    def l1_fraction(log10_mobility):
        res = _rate_limited_run(scenario, log10_mobility)
        return res.mass_per_layer["L1"] / res.stored_kg

    weak = l1_fraction(-10.0)
    mid = l1_fraction(-11.5)
    strong = l1_fraction(-13.0)
    assert weak < mid < strong
    assert weak < 0.5 < strong


def test_trapfill_fast_seal_recovers_the_quasi_static_limit():
    scenario = DOME_NO_RATE_LIMIT
    stateless, injected = _trap_graph(scenario)
    fast = _rate_limited_run(scenario, -4.0)
    for name, mass in stateless.mass_per_layer.items():
        assert abs(fast.mass_per_layer[name] - mass) < 1.0e-3 * injected


def test_trapfill_two_wells_conserve_mass():
    scenario = GRF_NO_RATE_LIMIT
    injected = sum(scenario.annual_rates_mt(10)) * 1e9
    wells = [scenario.well_xy, (1500.0, 3500.0)]

    def build(**kw):
        return build_trapfill(
            scenario.depth_surfaces(),
            scenario.layer_stack(),
            scenario.metadata(),
            source_xy=wells,
            **kw,
        )

    off = build().fill(injected)
    assert off.escaped_kg <= 1e-6 * injected
    assert abs(off.stored_kg + off.escaped_kg - injected) <= 1e-6 * injected

    on, _ = build(seal_log10_mobility=-11.0).run_schedule(_annual(scenario))
    inj = sum(_annual(scenario))
    assert abs(on.stored_kg + on.escaped_kg - inj) <= 1e-6 * inj


def test_trapfill_two_wells_in_one_basin_merge_to_single_plume():
    scenario = DOME_NO_RATE_LIMIT
    injected = sum(scenario.annual_rates_mt(10)) * 1e9
    surfaces, stack, meta = (
        scenario.depth_surfaces(),
        scenario.layer_stack(),
        scenario.metadata(),
    )
    one = build_trapfill(surfaces, stack, meta, source_xy=scenario.well_xy).fill(
        injected
    )
    two = build_trapfill(
        surfaces, stack, meta, source_xy=[(2500.0, 2500.0), (2400.0, 2400.0)]
    ).fill(injected)
    for name in LAYERS:
        assert (
            abs(one.mass_per_layer[name] - two.mass_per_layer[name]) <= 1e-6 * injected
        )


def test_trapfill_top_caprock_traps_what_reaches_the_shallowest_unit():
    scenario = replace(DOME_NO_RATE_LIMIT, true_pth_kpa=(1.0, 1.0, 1.0))
    injected = sum(_annual(scenario))
    res = build_trapfill(
        scenario.depth_surfaces(),
        scenario.layer_stack(),
        scenario.metadata(),
        source_xy=scenario.well_xy,
        top_seal_pth_pa=1.0e6,
    ).fill(injected)
    assert res.escaped_kg <= 1e-6 * injected
    assert res.mass_per_layer["L4"] > 0.5 * injected
