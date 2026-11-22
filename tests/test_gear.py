"""Gear mass lookup: immutable fractions and a clear error for an unknown topology."""
import pytest

from src.tools.gear import GEAR_FRACTION, gear_mass_kg


def test_unknown_topology_raises_a_named_value_error() -> None:
    with pytest.raises(ValueError, match="flying_saucer"):
        gear_mass_kg("flying_saucer", 10.0)


def test_gear_fractions_are_immutable() -> None:
    with pytest.raises(TypeError):
        GEAR_FRACTION["fixed_wing"] = 0.5  # type: ignore[index]
