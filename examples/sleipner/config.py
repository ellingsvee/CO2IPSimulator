from __future__ import annotations

from pathlib import Path

EXAMPLE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = EXAMPLE_ROOT / "data"
DEPTH_SURFACES = DATA_ROOT / "depth_surfaces"
# Outline of the imaged feeder chimney; its centroid is the injection point.
INJECTION_SITE_FILE = DATA_ROOT / "feeders" / "data" / "Main_feeder_chimney"

KPA_TO_PA = 1.0e3

ANNUAL_RATES_MT = [
    0.07,
    0.67,
    0.85,
    0.94,
    0.94,
    1.02,
    0.96,
    0.92,
    0.76,
    0.87,
    0.83,
    0.93,
    0.82,
    0.86,
    0.76,
    0.85,
    0.85,
    0.90,
    0.90,
    0.85,
    0.85,
    0.85,
    0.80,
    0.80,
    0.75,
    0.75,
    0.70,
    0.70,
]

PAPER_LAYER_NAMES = [f"L{i}" for i in range(1, 10)]

PAPER_DENSITY_KG_M3 = [
    570.0,
    542.5,
    515.0,
    487.5,
    460.0,
    432.5,
    405.0,
    377.5,
    350.0,
]

PAPER_PTH_KPA = [42.0, 49.0, 38.0, 40.0, 56.0, 46.0, 45.0, 68.0, 54.0]
