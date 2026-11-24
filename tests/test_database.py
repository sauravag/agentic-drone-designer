"""Milestone 1 acceptance tests: COTS catalog seeding, SI-unit sanity, and matcher logic."""

import sqlite3
from pathlib import Path

import pytest

from src.schemas.assumptions import DesignAssumptions
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
    select_bom,
)
from src.tools.database.packs import bus_window_v
from src.tools.database.seed_data import CATALOG_TABLES, CATALOG_VERSION, catalog_rows, seed_database
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
    first: int = len(load_catalog(path, "cells"))
    seed_database(path)
    assert len(load_catalog(path, "cells")) == first  # would double if seeding appended


def test_load_catalog_rejects_unknown_table(db_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown catalog table"):
        load_catalog(db_path, "users; DROP TABLE batteries")


def test_load_catalog_rejects_missing_db(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_catalog(tmp_path / "missing.db", "cells")


# ----------------------------------------------------------- SI sanity ---


def test_all_masses_and_lengths_are_si(db_path: Path) -> None:
    """Lengths must be metres (no mm/inch leakage) and masses kilograms."""
    for table in CATALOG_TABLES:
        for item in load_catalog(db_path, table):
            assert 0 < item.mass_kg <= 200.0, f"{table}:{item.name}"
            for field, value in item.model_dump().items():
                if field.endswith("_m"):
                    assert 0 < value < 3.0, f"{table}:{item.name}.{field}={value}"


def test_fixed_battery_table_is_gone(db_path: Path) -> None:
    assert "batteries" not in CATALOG_TABLES and CATALOG_VERSION == 5
    with pytest.raises(ValueError, match="Unknown catalog table"):
        load_catalog(db_path, "batteries")


def test_cell_built_packs_are_physical(db_path: Path) -> None:
    for cell in load_catalog(db_path, "cells"):
        pack = match_battery(db_path, cell.name, 6, 300.0 * WH_TO_J, 900.0)
        assert 100.0 <= pack.specific_energy_wh_kg <= 300.0, cell.name
        assert pack.capacity_j == pytest.approx(pack.capacity_wh * 3600.0)


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
    assert min(m.continuous_power_w for m in cruise) <= 200.0
    assert max(m.continuous_power_w for m in cruise) >= 40_000.0
    # 500 kg quadplane at T/W 1.45 needs ~1780 N per rotor
    assert max(m.max_thrust_n for m in vtol) >= 1_800.0
    assert max(g.continuous_power_w for g in gens) >= 50_000.0
    assert {c.chemistry for c in load_catalog(db_path, "cells")} == {"li_ion", "lipo"}


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


def test_battery_is_built_from_the_matched_cell_and_series(db_path: Path) -> None:
    energy_j: float = 500.0 * 3600.0
    peak_w: float = 6_000.0
    battery = match_battery(db_path, "Li-ion 21700 power 4.2Ah", 12, energy_j, peak_w)
    assert (battery.cell_name, battery.series) == ("Li-ion 21700 power 4.2Ah", 12)
    assert battery.capacity_j >= energy_j
    assert battery.max_power_w >= peak_w
    fewer = match_battery(db_path, "Li-ion 21700 power 4.2Ah", 12, energy_j / 2, peak_w / 2)
    assert fewer.parallel is not None and battery.parallel is not None and fewer.parallel < battery.parallel


def test_battery_names_an_unknown_cell(db_path: Path) -> None:
    with pytest.raises(ComponentNotFoundError, match="cells"):
        match_battery(db_path, "no such cell", 4, 100.0 * 3600.0, 1_000.0)


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


def _part(table: str, application: str | None = None):
    return next(row for row in catalog_rows(table) if application is None or row.application == application)


def _matched_needs(**changes: object) -> ComponentNeeds:
    base: dict[str, object] = dict(
        cruise_power_w=400.0, battery_energy_j=300.0 * 3600.0, battery_peak_power_w=900.0, payload_mass_kg=1.0,
        cruise_motor_name=catalog_rows("cruise_motors")[0].name,
        cruise_propeller_name=_part("propellers", "cruise").name, cruise_esc_name=catalog_rows("escs")[0].name,
        cell_name="Li-ion 21700 energy 5.0Ah", battery_series=4,
    )
    return ComponentNeeds(**{**base, **changes})


def test_electric_fixed_wing_bom(db_path: Path) -> None:
    needs = _matched_needs()
    bom = select_bom(db_path, needs)
    assert bom.vtol_motor is None and bom.ice_generator is None and bom.vtol_esc is None
    assert bom.flight_controller is not None and bom.companion_computer is None  # default: Cube, no Jetson
    assert (bom.cruise_motor.name, bom.cruise_propeller.name) == (needs.cruise_motor_name,
                                                                   needs.cruise_propeller_name)
    assert bom.cruise_esc is not None and bom.cruise_esc.name == needs.cruise_esc_name
    assert (bom.battery.cell_name, bom.battery.series) == (needs.cell_name, 4)
    expected = (bom.cruise_motor.mass_kg + bom.cruise_propeller.mass_kg + bom.cruise_esc.mass_kg
                + bom.battery.mass_kg + bom.payload.mass_kg + bom.flight_controller.mass_kg)
    assert bom.total_hardware_mass_kg == pytest.approx(expected)


def test_bom_needs_the_matched_propulsion_set(db_path: Path) -> None:
    with pytest.raises(ValueError, match="cruise_motor_name.*propulsion matching"):
        select_bom(db_path, ComponentNeeds(cruise_power_w=400.0, battery_energy_j=3e5, battery_peak_power_w=900.0,
                                           payload_mass_kg=1.0))
    with pytest.raises(ValueError, match="vtol_motor_name"):
        select_bom(db_path, _matched_needs(vtol_thrust_per_motor_n=30.0))


def test_hybrid_quadplane_bom_includes_generator_and_four_rotors(db_path: Path) -> None:
    lift_motor = next(m for m in catalog_rows("vtol_motors") if m.kv is not None)
    needs = _matched_needs(
        cruise_power_w=4_000.0, battery_energy_j=800.0 * 3600.0, battery_peak_power_w=20_000.0,
        payload_mass_kg=20.0, vtol_thrust_per_motor_n=300.0, generator_power_w=5_000.0,
        battery_chemistry="lipo", cell_name="LiPo pouch 22Ah 25C", battery_series=12,
        vtol_motor_name=lift_motor.name, vtol_propeller_name=_part("propellers", "vtol").name,
        vtol_esc_name=catalog_rows("escs")[1].name,
    )
    bom = select_bom(db_path, needs)
    assert bom.ice_generator is not None and bom.vtol_motor is not None and bom.vtol_propeller is not None
    assert bom.vtol_esc is not None and bom.vtol_motor_count == 4
    low, high = bus_window_v(next(c for c in catalog_rows("cells") if c.name == needs.cell_name), 12)
    assert bom.ice_generator.output_voltage_v is not None and low <= bom.ice_generator.output_voltage_v <= high
    rotor_mass = 4 * (bom.vtol_motor.mass_kg + bom.vtol_propeller.mass_kg + bom.vtol_esc.mass_kg)
    assert bom.total_hardware_mass_kg > rotor_mass + bom.ice_generator.mass_kg


def test_hybrid_bom_rejects_a_cell_of_the_wrong_chemistry(db_path: Path) -> None:
    with pytest.raises(ValueError, match="lipo"):
        select_bom(db_path, _matched_needs(battery_chemistry="lipo", generator_power_w=500.0))


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


def test_higher_specific_energy_makes_the_same_pack_lighter(db_path: Path) -> None:
    base = match_battery(db_path, "Li-ion 21700 energy 5.0Ah", 6, 300 * WH_TO_J, 900.0)
    light = match_battery(db_path, "Li-ion 21700 energy 5.0Ah", 6, 300 * WH_TO_J, 900.0,
                          DesignAssumptions(liion_specific_energy_wh_kg=300))
    assert light.mass_kg == pytest.approx(base.capacity_wh / 300, rel=1e-3)
    assert light.capacity_wh == base.capacity_wh


def test_peak_power_sets_the_parallel_count(db_path: Path) -> None:
    # 500 Wh at 12S of 10 A cells: energy alone needs a few strings, 60 kW needs far more.
    by_energy = match_battery(db_path, "Li-ion 21700 energy 5.0Ah", 12, 500 * WH_TO_J, 1_000.0)
    by_power = match_battery(db_path, "Li-ion 21700 energy 5.0Ah", 12, 500 * WH_TO_J, 60_000.0)
    assert by_power.parallel is not None and by_energy.parallel is not None
    assert by_power.parallel > by_energy.parallel
    assert by_power.max_power_w >= 60_000.0  # the cells' continuous current covers the peak


# ---------------------------------------------------- generator voltages ---

GENSET_POWERS_W: tuple[float, ...] = (300.0, 800.0, 1_800.0, 5_000.0, 12_000.0, 40_000.0, 60_000.0)


def test_generator_voltage_variants_follow_their_base_rows(db_path: Path) -> None:
    gens = load_catalog(db_path, "ice_generators")
    bases = [g for g in gens if g.output_voltage_v is None]
    variants = [g for g in gens if g.output_voltage_v is not None]
    assert gens[:len(bases)] == bases  # every legacy (unregulated) row first, in its original order
    labels = {g.name.rsplit(" ", 1)[-1] for g in variants}
    assert {"6S", "8S", "10S", "12S", "14S", "24S", "48S", "96S"} == labels
    assert max(g.output_voltage_v for g in variants) >= 300.0  # HV classes for the big gensets
    by_name = {g.name: g for g in bases}
    for variant in variants:
        base = by_name[variant.name.rsplit(" ", 1)[0]]
        assert (variant.mass_kg, variant.continuous_power_w) == (base.mass_kg, base.continuous_power_w)


@pytest.mark.parametrize("required_w", GENSET_POWERS_W)
def test_generator_pick_without_window_is_the_base_row(db_path: Path, required_w: float) -> None:
    gen = match_ice_generator(db_path, required_w)
    assert gen.output_voltage_v is None
    bases = [g for g in load_catalog(db_path, "ice_generators") if g.output_voltage_v is None]
    expected = min((g for g in bases if g.continuous_power_w >= required_w * SAFETY_MARGIN), key=lambda g: g.mass_kg)
    assert gen == expected


def test_generator_inside_bus_window_is_matched(db_path: Path) -> None:
    gen = match_ice_generator(db_path, 2_500.0, bus_window_v=(39.6, 50.4))  # a 12S Li-ion bus
    assert 39.6 <= gen.output_voltage_v <= 50.4
    assert gen.continuous_power_w >= 2_500.0 * SAFETY_MARGIN
    assert gen.mass_kg == match_ice_generator(db_path, 2_500.0).mass_kg


def test_generator_outside_bus_window_not_matched(db_path: Path) -> None:
    with pytest.raises(ComponentNotFoundError, match=r"1\.0.*2\.0 V"):
        match_ice_generator(db_path, 1_000.0, bus_window_v=(1.0, 2.0))


@pytest.mark.parametrize("window", [(50.0, 40.0), (0.0, 40.0), (float("nan"), 40.0)])
def test_generator_window_must_be_valid(db_path: Path, window: tuple[float, float]) -> None:
    with pytest.raises(ValueError):
        match_ice_generator(db_path, 1_000.0, bus_window_v=window)
