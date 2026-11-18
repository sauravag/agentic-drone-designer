"""Milestone 1 acceptance tests: COTS catalog seeding, SI-unit sanity, and matcher logic."""

import sqlite3
from pathlib import Path

import pytest

from src.schemas.assumptions import DEFAULT_ASSUMPTIONS, DesignAssumptions
from src.schemas.components import ComponentNeeds
from src.tools.database.matcher import (
    SAFETY_MARGIN,
    ComponentNotFoundError,
    check_mtow_convergence,
    load_catalog,
    match_battery,
    match_cruise_motor,
    match_ice_generator,
    match_payload,
    match_propeller,
    match_vtol_motor,
    rescale_pack,
    select_bom,
)
from src.tools.database.seed_data import CATALOG_TABLES, seed_database
from src.units import WH_TO_J

MIN_ROWS_PER_TABLE: int = 15


@pytest.fixture(scope="module")
def db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path: Path = tmp_path_factory.mktemp("db") / "cots_catalog.db"
    seed_database(path)
    return path


# ---------------------------------------------------------------- seeding ---


@pytest.mark.parametrize("table", list(CATALOG_TABLES))
def test_each_table_has_at_least_15_rows(db_path: Path, table: str) -> None:
    with sqlite3.connect(db_path) as conn:
        count: int = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    assert count >= MIN_ROWS_PER_TABLE


def test_reseeding_is_idempotent(tmp_path: Path) -> None:
    path: Path = tmp_path / "catalog.db"
    seed_database(path)
    first: int = len(load_catalog(path, "batteries"))
    seed_database(path)
    assert len(load_catalog(path, "batteries")) == first  # would double if seeding appended


def test_load_catalog_rejects_unknown_table(db_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown catalog table"):
        load_catalog(db_path, "users; DROP TABLE batteries")


def test_load_catalog_rejects_missing_db(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_catalog(tmp_path / "missing.db", "batteries")


# ----------------------------------------------------------- SI sanity ---


def test_all_masses_and_lengths_are_si(db_path: Path) -> None:
    """Lengths must be metres (no mm/inch leakage) and masses kilograms."""
    for table in CATALOG_TABLES:
        for item in load_catalog(db_path, table):
            assert 0 < item.mass_kg <= 200.0, f"{table}:{item.name}"
            for field, value in item.model_dump().items():
                if field.endswith("_m"):
                    assert 0 < value < 3.0, f"{table}:{item.name}.{field}={value}"


def test_battery_specific_energy_is_physical(db_path: Path) -> None:
    for battery in load_catalog(db_path, "batteries"):
        assert 100.0 <= battery.specific_energy_wh_kg <= 300.0, battery.name
        assert battery.capacity_j == pytest.approx(battery.capacity_wh * 3600.0)


def test_ice_bsfc_is_physical(db_path: Path) -> None:
    for gen in load_catalog(db_path, "ice_generators"):
        assert 0.3 <= gen.bsfc_kg_kwh <= 1.0, gen.name
        assert gen.bsfc_kg_per_j == pytest.approx(gen.bsfc_kg_kwh / 3.6e6)


def test_propeller_diameters_are_metres(db_path: Path) -> None:
    for prop in load_catalog(db_path, "propellers"):
        assert 0.15 <= prop.diameter_m <= 2.0, prop.name


def test_catalog_spans_group1_to_group3_scales(db_path: Path) -> None:
    cruise = load_catalog(db_path, "cruise_motors")
    vtol = load_catalog(db_path, "vtol_motors")
    gens = load_catalog(db_path, "ice_generators")
    batts = load_catalog(db_path, "batteries")
    assert min(m.continuous_power_w for m in cruise) <= 200.0
    assert max(m.continuous_power_w for m in cruise) >= 40_000.0
    # 500 kg quadplane at T/W 1.45 needs ~1780 N per rotor
    assert max(m.max_thrust_n for m in vtol) >= 1_800.0
    assert max(g.continuous_power_w for g in gens) >= 50_000.0
    assert max(b.capacity_wh for b in batts) >= 10_000.0


@pytest.mark.parametrize(("motor_table", "application"), [("cruise_motors", "cruise"), ("vtol_motors", "vtol")])
def test_every_motor_has_a_compatible_propeller(db_path: Path, motor_table: str, application: str) -> None:
    props = [p for p in load_catalog(db_path, "propellers") if p.application == application]
    for motor in load_catalog(db_path, motor_table):
        assert any(p.max_power_w >= motor.max_power_w for p in props), motor.name


# ------------------------------------------------------------- matching ---


@pytest.mark.parametrize("required_w", [300.0, 2_500.0, 25_000.0])  # Group 1, 2, 3
def test_cruise_motor_is_lightest_meeting_margin(db_path: Path, required_w: float) -> None:
    motor = match_cruise_motor(db_path, required_w)
    assert motor.continuous_power_w >= required_w * SAFETY_MARGIN
    qualifying = [m for m in load_catalog(db_path, "cruise_motors") if m.continuous_power_w >= required_w * SAFETY_MARGIN]
    assert motor.mass_kg == min(m.mass_kg for m in qualifying)


@pytest.mark.parametrize("required_n", [15.0, 150.0, 1_500.0])
def test_vtol_motor_meets_thrust_margin(db_path: Path, required_n: float) -> None:
    motor = match_vtol_motor(db_path, required_n)
    assert motor.max_thrust_n >= required_n * SAFETY_MARGIN


@pytest.mark.parametrize("required_w", [800.0, 5_000.0, 40_000.0])
def test_ice_generator_meets_power_margin(db_path: Path, required_w: float) -> None:
    gen = match_ice_generator(db_path, required_w)
    assert gen.continuous_power_w >= required_w * SAFETY_MARGIN


def test_battery_meets_energy_and_peak_power(db_path: Path) -> None:
    energy_j: float = 500.0 * 3600.0
    peak_w: float = 6_000.0
    battery = match_battery(db_path, energy_j, peak_w)
    assert battery.capacity_j >= energy_j * SAFETY_MARGIN
    assert battery.max_power_w >= peak_w * SAFETY_MARGIN


def test_battery_chemistry_filter(db_path: Path) -> None:
    battery = match_battery(db_path, 100.0 * 3600.0, 1_000.0, chemistry="lipo")
    assert battery.chemistry == "lipo"


def test_propeller_absorbs_full_motor_power(db_path: Path) -> None:
    prop = match_propeller(db_path, 4_000.0, "vtol")
    assert prop.application == "vtol"
    assert prop.max_power_w >= 4_000.0


def test_payload_match_picks_closest_mass(db_path: Path) -> None:
    payload = match_payload(db_path, 3.5, sensor_type="eo_ir")
    assert payload.sensor_type == "eo_ir"
    assert payload.mass_kg == pytest.approx(3.5)


def test_payload_match_never_undershoots_requested_mass(db_path: Path) -> None:
    # Catalog has eo_ir parts at 1.4 kg and 3.5 kg: a 3.0 kg need must not get the 1.4 kg unit.
    payload = match_payload(db_path, 3.0, sensor_type="eo_ir")
    assert payload.mass_kg >= 3.0
    assert payload.mass_kg == pytest.approx(3.5)


def test_payload_heavier_than_catalog_raises(db_path: Path) -> None:
    with pytest.raises(ComponentNotFoundError, match="payloads"):
        match_payload(db_path, 100.0)


def test_payload_match_unknown_sensor_raises(db_path: Path) -> None:
    with pytest.raises(ComponentNotFoundError, match="payloads"):
        match_payload(db_path, 3.5, sensor_type="eo_ir", max_power_w=0.1)


def test_impossible_requirement_raises(db_path: Path) -> None:
    with pytest.raises(ComponentNotFoundError, match="cruise_motors"):
        match_cruise_motor(db_path, 10_000_000.0)


@pytest.mark.parametrize("bad", [0.0, -5.0, float("nan"), float("inf")])
def test_non_physical_requirement_rejected(db_path: Path, bad: float) -> None:
    with pytest.raises(ValueError):
        match_cruise_motor(db_path, bad)


# ------------------------------------------------------------ BOM & MTOW ---


def test_electric_fixed_wing_bom(db_path: Path) -> None:
    needs = ComponentNeeds(
        cruise_power_w=400.0,
        battery_energy_j=300.0 * 3600.0,
        battery_peak_power_w=900.0,
        payload_mass_kg=1.0,
    )
    bom = select_bom(db_path, needs)
    assert bom.vtol_motor is None and bom.ice_generator is None
    assert bom.flight_controller is not None and bom.companion_computer is None  # default: Cube, no Jetson
    expected = (bom.cruise_motor.mass_kg + bom.cruise_propeller.mass_kg + bom.battery.mass_kg + bom.payload.mass_kg
                + bom.flight_controller.mass_kg)
    assert bom.total_hardware_mass_kg == pytest.approx(expected)


def test_hybrid_quadplane_bom_includes_generator_and_four_rotors(db_path: Path) -> None:
    needs = ComponentNeeds(
        cruise_power_w=4_000.0,
        battery_energy_j=800.0 * 3600.0,
        battery_peak_power_w=20_000.0,
        payload_mass_kg=20.0,
        vtol_thrust_per_motor_n=300.0,
        generator_power_w=5_000.0,
    )
    bom = select_bom(db_path, needs)
    assert bom.ice_generator is not None and bom.vtol_motor is not None and bom.vtol_propeller is not None
    assert bom.vtol_motor_count == 4
    rotor_mass = 4 * (bom.vtol_motor.mass_kg + bom.vtol_propeller.mass_kg)
    assert bom.total_hardware_mass_kg > rotor_mass + bom.ice_generator.mass_kg


@pytest.mark.parametrize(
    ("sized", "actual", "converged"),
    [(10.0, 10.29, True), (10.0, 9.71, True), (10.0, 10.31, False), (10.0, 9.69, False)],
)
def test_mtow_convergence_band(sized: float, actual: float, converged: bool) -> None:
    assert check_mtow_convergence(sized, actual) is converged


def test_mtow_convergence_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        check_mtow_convergence(0.0, 5.0)


# ------------------------------------------------- assumption-driven packs ---


def test_default_assumptions_leave_catalog_packs_unchanged(db_path: Path) -> None:
    for b in load_catalog(db_path, "batteries"):
        assert rescale_pack(b, DEFAULT_ASSUMPTIONS) == b


def test_higher_specific_energy_makes_the_same_pack_lighter(db_path: Path) -> None:
    b = next(b for b in load_catalog(db_path, "batteries") if b.chemistry == "li_ion")
    light = rescale_pack(b, DesignAssumptions(liion_specific_energy_wh_kg=300))
    assert light.mass_kg == pytest.approx(b.capacity_wh / 300, rel=1e-3)
    assert light.capacity_wh == b.capacity_wh


def test_c_rate_assumption_changes_which_packs_fit(db_path: Path) -> None:
    # 500 Wh / 60 kW: the largest seeded Li-ion pack (15 kWh at 3C = 45 kW) cannot supply 69 kW.
    with pytest.raises(ComponentNotFoundError):
        match_battery(db_path, 500 * WH_TO_J, 60_000, "li_ion")
    fast = match_battery(db_path, 500 * WH_TO_J, 60_000, "li_ion", DesignAssumptions(liion_max_c_rate=30))
    assert fast.chemistry == "li_ion"
    assert fast.max_c_rate == 30
