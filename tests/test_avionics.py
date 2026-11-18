"""Avionics stack choice (Cube Orange+ / ARK Jetson PAB), the optional Jetson, and the avionics assumptions."""

import json
from pathlib import Path
from typing import Final

import pytest

from src.tools.sizing.mission_solver import size_aircraft
from src.ui.examples import EXAMPLE_MISSIONS

BASELINE_PATH: Final[Path] = Path(__file__).parent / "data" / "avionics_baseline.json"
BASELINE_FIELDS: Final[tuple[str, ...]] = (
    "mtow_kg", "empty_mass_kg", "battery_mass_kg", "fuel_mass_kg", "cruise_power_w", "hover_power_w",
)
EXACT: Final[float] = 1e-9


# Recorded from the code before the avionics change: Cube Orange+ without a Jetson at default assumptions
# must size every reference design exactly as before.
@pytest.mark.parametrize("req", list(EXAMPLE_MISSIONS.values()), ids=lambda r: r.mission_name)
def test_reference_designs_unchanged_without_jetson(req) -> None:
    expected: dict[str, float] = json.loads(BASELINE_PATH.read_text())[req.mission_name]
    sizing = size_aircraft(req)
    for field in BASELINE_FIELDS:
        assert getattr(sizing, field) == pytest.approx(expected[field], rel=EXACT, abs=EXACT), field
