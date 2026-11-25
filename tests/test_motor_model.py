"""Motor equivalent-circuit model and the winding / ESC / propeller-pitch catalogue."""

import math
import sqlite3
from pathlib import Path

import pytest

from src.schemas.components import CruiseMotor, Esc, VtolMotor
from src.tools.database.matcher import ComponentNotFoundError, part_by_name
from src.tools.database.seed_data import CRUISE_PROPS_IN, VTOL_PROPS_IN, catalog_rows, seed_database
from src.tools.propulsion.motor import motor_point

ESC_CURRENT_MARGIN = 1.2


def test_motor_point_no_load() -> None:
    point = motor_point(kv_rpm_per_v=900.0, rm_ohm=0.05, i0_a=1.5, rpm=9000.0, torque_nm=0.0)
    assert point.current_a == pytest.approx(1.5)
    assert point.voltage_v == pytest.approx(9000.0 / 900.0 + 1.5 * 0.05)
    assert point.shaft_power_w == pytest.approx(0.0)


def test_motor_efficiency_peaks_between_no_load_and_stall() -> None:
    kv, rm, i0, voltage = 900.0, 0.05, 1.5, 14.8
    kt = 60.0 / (2.0 * math.pi * kv)
    stall_torque = (voltage / rm - i0) * kt
    effs = []
    for frac in (0.01, 0.1, 0.3, 0.6, 0.99):
        torque = frac * stall_torque
        # solve rpm so the supply voltage is fixed: V = I*Rm + rpm/KV
        current = torque / kt + i0
        rpm = (voltage - current * rm) * kv
        effs.append(motor_point(kv, rm, i0, rpm, torque).efficiency)
    peak = max(effs)
    assert effs.index(peak) not in (0, len(effs) - 1)
    assert 0.5 < peak < 1.0


@pytest.mark.parametrize(
    "kv, rm, i0, rpm, torque",
    [
        (0.0, 0.05, 1.0, 1000.0, 0.1),
        (-900.0, 0.05, 1.0, 1000.0, 0.1),
        (float("nan"), 0.05, 1.0, 1000.0, 0.1),
        (900.0, 0.0, 1.0, 1000.0, 0.1),
        (900.0, float("inf"), 1.0, 1000.0, 0.1),
        (900.0, 0.05, -1.0, 1000.0, 0.1),
        (900.0, 0.05, 1.0, -1.0, 0.1),
        (900.0, 0.05, 1.0, 1000.0, -0.1),
        (900.0, 0.05, 1.0, float("nan"), 0.1),
    ],
)
def test_motor_point_rejects_bad_inputs(kv: float, rm: float, i0: float, rpm: float, torque: float) -> None:
    with pytest.raises(ValueError):
        motor_point(kv, rm, i0, rpm, torque)


def test_motor_point_at_rest_with_no_current_has_zero_efficiency() -> None:
    # No input and no output power: efficiency is defined as 0, not a division error.
    point = motor_point(900.0, 0.05, 0.0, 0.0, 0.0)
    assert point.current_a == 0.0 and point.efficiency == 0.0


def _motors() -> list[CruiseMotor | VtolMotor]:
    return [*catalog_rows("cruise_motors"), *catalog_rows("vtol_motors")]


def test_every_motor_size_has_at_least_two_windings() -> None:
    for table in ("cruise_motors", "vtol_motors"):
        by_size: dict[tuple, set[str]] = {}
        for m in catalog_rows(table):
            by_size.setdefault((m.mass_kg, m.max_power_w, m.diameter_m, m.length_m), set()).add(m.name)
        assert all(len(names) >= 2 for names in by_size.values()), table


def test_windings_scale_constants_with_kv() -> None:
    rows = {m.name: m for m in catalog_rows("cruise_motors")}
    base = next(m for m in catalog_rows("cruise_motors") if m.name.startswith("Outrunner 2814-"))  # base winding
    other = next(m for n, m in rows.items() if n.startswith("Outrunner 2814-") and m is not base and m.kv > base.kv)
    ratio = other.kv / base.kv
    assert other.rm_ohm == pytest.approx(base.rm_ohm / ratio**2, rel=1e-3)
    assert other.i0_a == pytest.approx(base.i0_a * ratio, rel=1e-3)
    assert other.max_current_a == pytest.approx(base.max_current_a * ratio, rel=0.02)
    assert other.voltage_v < base.voltage_v
    assert base.voltage_v == pytest.approx(16.8)


def test_every_cruise_diameter_has_several_pitches_and_a_folding_variant() -> None:
    props = [p for p in catalog_rows("propellers") if p.application == "cruise"]
    by_diameter: dict[float, list] = {}
    for p in props:
        by_diameter.setdefault(p.diameter_m, []).append(p)
    for diameter, group in by_diameter.items():
        solid_pitches = {p.pitch_m for p in group if not p.is_folding}
        assert len(solid_pitches) >= 4, diameter
        assert any(p.is_folding for p in group), diameter


def test_lift_props_have_low_pitch_variants() -> None:
    props = [p for p in catalog_rows("propellers") if p.application == "vtol"]
    ratios = {round(p.pitch_m / p.diameter_m, 1) for p in props}
    assert {0.3, 0.4, 0.5, 0.6} <= ratios
    assert not any(p.is_folding for p in props)


def test_folding_prop_is_heavier_than_its_solid_twin() -> None:
    props = {p.name: p for p in catalog_rows("propellers")}
    for name, p in props.items():
        if p.is_folding:
            twin = props[name.removesuffix(" folding")]
            assert p.mass_kg == pytest.approx(twin.mass_kg * 1.3)
            assert p.max_power_w == twin.max_power_w


def test_escs_cover_every_motor_current_and_voltage() -> None:
    escs: list[Esc] = catalog_rows("escs")
    for m in _motors():
        assert m.max_current_a is not None and m.rm_ohm is not None and m.i0_a is not None
        assert any(
            e.continuous_current_a >= ESC_CURRENT_MARGIN * m.max_current_a and e.max_voltage_v >= m.voltage_v
            for e in escs
        ), m.name


def test_esc_ratings_are_consistent() -> None:
    for e in catalog_rows("escs"):
        assert e.burst_current_a == pytest.approx(1.5 * e.continuous_current_a)
        assert 0.95 <= e.efficiency <= 0.97


def test_catalog_rows_match_seeded_database(tmp_path: Path) -> None:
    db = seed_database(tmp_path / "c.db")
    with sqlite3.connect(db) as conn:
        for table in ("cruise_motors", "vtol_motors", "propellers", "escs"):
            (count,) = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            assert count == len(catalog_rows(table)), table
    assert part_by_name(db, "escs", catalog_rows("escs")[0].name, Esc) == catalog_rows("escs")[0].model_copy(
        update={"id": 1}
    )


def test_catalog_rows_unknown_table_raises() -> None:
    with pytest.raises(ValueError):
        catalog_rows("nope")


def test_legacy_motor_row_without_constants_loads() -> None:
    motor = CruiseMotor(
        name="legacy", mass_kg=0.1, max_power_w=500, continuous_power_w=300, kv=900.0,
        voltage_v=14.8, diameter_m=0.03, length_m=0.03,
    )
    assert motor.rm_ohm is None and motor.i0_a is None and motor.max_current_a is None


def test_part_by_name_missing_raises(tmp_path: Path) -> None:
    db = seed_database(tmp_path / "c.db")
    with pytest.raises(ComponentNotFoundError, match="escs.*no-such-esc"):
        part_by_name(db, "escs", "no-such-esc", Esc)


def test_part_by_name_returns_row(tmp_path: Path) -> None:
    db = seed_database(tmp_path / "c.db")
    row: CruiseMotor = part_by_name(db, "cruise_motors", "Outrunner 2814-1009KV", CruiseMotor)
    assert row.kv == 1009.0


def test_part_by_name_refuses_the_wrong_model(tmp_path: Path) -> None:
    db = seed_database(tmp_path / "c.db")
    with pytest.raises(TypeError, match="cruise_motors rows are CruiseMotor, not Esc"):
        part_by_name(db, "cruise_motors", "Outrunner 2814-1009KV", Esc)


def test_every_motor_has_an_esc_in_its_own_lowest_adequate_voltage_class() -> None:
    escs: list[Esc] = catalog_rows("escs")
    classes = sorted({e.max_voltage_v for e in escs})
    for m in _motors():
        own = min(v for v in classes if v >= m.voltage_v - 1e-6)
        assert any(
            e.max_voltage_v == own and e.continuous_current_a >= ESC_CURRENT_MARGIN * m.max_current_a for e in escs
        ), (m.name, own)


def test_base_propeller_names_keep_legacy_format() -> None:
    names = {p.name for p in catalog_rows("propellers")}
    for app, table in (("cruise", CRUISE_PROPS_IN), ("vtol", VTOL_PROPS_IN)):
        for dia, pitch, *_ in table:
            assert f"Carbon {app} prop {dia}x{pitch}in" in names
