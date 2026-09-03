from pathlib import Path

import numpy as np
import pytest

from co2ipsimulator import rust

GOLDEN = Path(__file__).parent / "fixtures" / "spill_golden"
CASES = (
    ("mini_diag", True),
    ("mini_nodiag", False),
    ("synsurf_diag", True),
    ("synsurf_nodiag", False),
)


@pytest.mark.parametrize("name, use_diagonals", CASES)
def test_spill_graph_matches_reference_implementation(name, use_diagonals):
    """Exercise the complete spill analysis and its public intermediate results."""
    with np.load(GOLDEN / f"{name}.npz") as reference:
        topography = np.ascontiguousarray(reference["topography"], dtype=np.float64)
        direction, slope = rust.spillfield(topography, use_diagonals)
        regions = rust.spillregions(direction, use_diagonals)
        downstream_region, current_cell, downstream_cell, elevation = rust.spillpoints(
            topography, regions, use_diagonals
        )
        analysis = rust.spillanalysis(topography, use_diagonals)

        assert np.array_equal(direction, reference["spillfield"].astype(np.int8))
        finite = np.isfinite(reference["slope"]) & np.isfinite(slope)
        assert np.allclose(
            slope[finite], reference["slope"][finite], atol=1e-12, rtol=0
        )
        assert np.array_equal(regions, reference["regions"].astype(np.int64))
        assert np.array_equal(downstream_region, reference["raw_sp_downstream_region"])
        assert np.array_equal(current_cell, reference["raw_sp_current_cell"])
        assert np.array_equal(downstream_cell, reference["raw_sp_downstream_cell"])
        assert np.allclose(elevation, reference["raw_sp_elevation"], atol=1e-12, rtol=0)

        assert int(analysis["num_spoints"]) == int(reference["num_spoints"][0])
        assert int(analysis["agglom_ne"]) == int(reference["agglom_ne"][0])
        assert int(analysis["num_regions"]) == int(reference["num_regions"][0])
        assert np.array_equal(
            np.asarray(analysis["regions"]), reference["ts_regions"].astype(np.int64)
        )
        assert np.array_equal(
            np.asarray(analysis["spillfield"]),
            reference["ts_spillfield"].astype(np.int8),
        )

        for actual_name, tolerance in (
            ("trapvolumes", 1e-6),
            ("subvolumes", 1e-6),
            ("sp_elevation", 1e-9),
        ):
            assert np.allclose(
                np.sort(np.asarray(analysis[actual_name])),
                np.sort(reference[actual_name]),
                atol=tolerance,
            )
        assert np.array_equal(
            np.sort(np.asarray(analysis["footprint_lengths"])),
            np.sort(reference["footprint_lengths"]),
        )
