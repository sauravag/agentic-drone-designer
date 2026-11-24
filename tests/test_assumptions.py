"""DesignAssumptions: the single source of technology constants."""

import pytest
from pydantic import ValidationError

from src.schemas.assumptions import DEFAULT_ASSUMPTIONS, DesignAssumptions


def test_defaults_match_todays_constants() -> None:
    a = DEFAULT_ASSUMPTIONS
    assert (a.liion_specific_energy_wh_kg, a.liion_max_c_rate, a.lipo_specific_energy_wh_kg) == (200, 3, 155)
    assert (a.battery_reserve_fraction, a.fuel_reserve_fraction, a.bsfc_kg_per_kwh) == (0.20, 0.10, 0.50)
    assert a.pusher_efficiency_factor == 0.95
    assert a.non_default() == {}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("liion_specific_energy_wh_kg", 99),
        ("liion_specific_energy_wh_kg", 501),
        ("liion_max_c_rate", 0.5),
        ("pusher_efficiency_factor", 0.5),
        ("battery_reserve_fraction", 0.05),
    ],
)
def test_out_of_bounds_values_are_rejected(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        DesignAssumptions(**{field: value})


def test_reserve_floor_and_helpers() -> None:
    a = DesignAssumptions(liion_specific_energy_wh_kg=300).with_reserve_floor(0.15)
    assert a.reserve_for("series_hybrid") == 0.15 and a.reserve_for("electric") == 0.20
    assert a.non_default()["liion_specific_energy_wh_kg"] == (200, 300)
    assert a.specific_energy_wh_kg("li_ion") == 300 and a.max_c_rate("lipo") == 25


def test_extra_fields_forbidden_and_model_frozen() -> None:
    with pytest.raises(ValidationError):
        DesignAssumptions(bogus=1)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        DEFAULT_ASSUMPTIONS.bsfc_kg_per_kwh = 0.4  # type: ignore[misc]


def test_catalog_pack_constants_come_from_the_defaults() -> None:
    from src.tools.database import seed_data

    assert seed_data.PACK_SPECIFIC_ENERGY_WH_KG == {"li_ion": 200.0, "lipo": 155.0}
    assert seed_data.PACK_ENERGY_DENSITY_WH_L == {"li_ion": 250.0, "lipo": 280.0}


LEGACY_EFFICIENCY_KEYS = {"drivetrain_efficiency": 0.7, "hover_figure_of_merit": 0.75, "lift_motor_efficiency": 0.9}


def test_efficiency_assumptions_are_gone() -> None:
    assert not set(LEGACY_EFFICIENCY_KEYS) & set(DesignAssumptions.model_fields)


def test_legacy_assumptions_with_efficiency_keys_load(caplog: pytest.LogCaptureFixture) -> None:
    legacy = {"liion_specific_energy_wh_kg": 250.0, **LEGACY_EFFICIENCY_KEYS}

    with caplog.at_level("INFO", logger="src.schemas.assumptions"):
        loaded = DesignAssumptions.model_validate(legacy)

    assert loaded == DesignAssumptions(liion_specific_energy_wh_kg=250.0)
    assert "drivetrain_efficiency" in caplog.text and "propulsion matching" in caplog.text


def test_legacy_assumption_changes_with_efficiency_keys_load() -> None:
    from src.schemas.changes import AssumptionChanges

    changes = AssumptionChanges.model_validate({"bsfc_kg_per_kwh": 0.6, **LEGACY_EFFICIENCY_KEYS})

    assert changes == AssumptionChanges(bsfc_kg_per_kwh=0.6)


def test_unknown_keys_are_still_rejected_next_to_legacy_ones() -> None:
    with pytest.raises(ValidationError):
        DesignAssumptions.model_validate({"drivetrain_efficiency": 0.7, "bogus": 1})
