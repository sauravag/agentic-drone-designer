"""Propulsor: propeller table + motor equivalent circuit + ESC at a bus voltage (spec 6.3, 6.4)."""

import math
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Final

import pytest

from src.schemas.components import Battery, CruiseMotor, Esc, Propeller, VtolMotor
from src.schemas.propulsion import OperatingPoint
from src.tools.database.seed_data import catalog_rows
from src.tools.propulsion import prop_tables
from src.tools.propulsion.prop_tables import PropellerTable, propeller_table
from src.tools.propulsion.propulsor import Propulsor, propulsor_from_parts

SMALL_PROP: Final[tuple[float, float]] = (0.254, 0.1143)  # 10 x 4.5 inch
MOTOR_NAME: Final[str] = "Outrunner 2814-900KV"
RHO, MU, SOUND = 1.225, 1.81e-5, 340.3
AIR: Final[tuple[float, float, float]] = (RHO, MU, SOUND)
BUS_V: Final[float] = 14.8
ROUND_TRIP_TOL: Final[float] = 0.01


@pytest.fixture(scope="module")
def table() -> PropellerTable:
    return prop_tables._build_table(*SMALL_PROP)


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("PROP_TABLE_CACHE_DIR", str(tmp_path))
    propeller_table.cache_clear()
    yield tmp_path
    propeller_table.cache_clear()


def _motor() -> CruiseMotor:
    return next(m for m in catalog_rows("cruise_motors") if m.name == MOTOR_NAME)


@pytest.fixture
def propulsor(table: PropellerTable) -> Propulsor:
    motor = _motor()
    return Propulsor(table=table, kv=motor.kv, rm_ohm=motor.rm_ohm, i0_a=motor.i0_a,
                     max_current_a=motor.max_current_a, esc_efficiency=0.95, bus_voltage_v=BUS_V)


@pytest.mark.parametrize("thrust_n, airspeed_ms", [(5.0, 12.0), (3.0, 18.0), (8.0, 0.0)])
def test_point_for_thrust_round_trips_thrust_n(propulsor: Propulsor, thrust_n: float, airspeed_ms: float) -> None:
    point = propulsor.point_for_thrust("cruise", thrust_n, airspeed_ms, *AIR)
    assert isinstance(point, OperatingPoint)
    assert 0.0 < point.throttle <= 1.0
    back = propulsor.thrust_n(point.throttle, airspeed_ms, *AIR)
    assert back == pytest.approx(thrust_n, rel=ROUND_TRIP_TOL)


def test_operating_point_fields_are_consistent(propulsor: Propulsor) -> None:
    point = propulsor.point_for_thrust("loiter", 5.0, 12.0, *AIR)
    assert point is not None
    assert (point.flight_point, point.propulsor, point.sagged) == ("loiter", "cruise", False)
    assert point.tip_mach == pytest.approx(math.pi * SMALL_PROP[0] * point.rpm / 60.0 / SOUND)
    assert point.electrical_power_w == pytest.approx(point.throttle * BUS_V * point.current_a / 0.95)
    assert 0.3 < point.prop_efficiency < 0.9
    assert 0.5 < point.motor_efficiency < 1.0
    shaft_w = 5.0 * 12.0 / point.prop_efficiency
    assert point.electrical_power_w > shaft_w


def test_hover_point_reports_figure_of_merit(propulsor: Propulsor) -> None:
    point = propulsor.point_for_thrust("hover", 8.0, 0.0, *AIR)
    assert point is not None
    assert 0.3 < point.prop_efficiency < 0.85  # figure of merit, not T V / P (which is 0 in hover)


# The generic blade still pushes at its geometric pitch speed (cambered sections: zero thrust near J 0.58 for
# P/D 0.45), so "above pitch speed" means past the zero-thrust speed at the table's top rpm: 1.35 x lands inside
# the table's J axis (CT <= 0 there), 2 x beyond it.
@pytest.mark.parametrize("pitch_speed_factor", [1.35, 2.0])
def test_point_for_thrust_above_pitch_speed_is_none(propulsor: Propulsor, pitch_speed_factor: float) -> None:
    roomy = replace(propulsor, bus_voltage_v=200.0, max_current_a=500.0)  # only the propeller can say no
    top_rpm = 0.80 * SOUND / (math.pi * SMALL_PROP[0]) * 60.0
    airspeed = pitch_speed_factor * roomy.table.pitch_speed_ms(top_rpm)
    assert roomy.point_for_thrust("dash", 1.0, airspeed, *AIR) is None


def test_thrust_beyond_the_tip_mach_limit_is_none(propulsor: Propulsor) -> None:
    roomy = replace(propulsor, bus_voltage_v=200.0, max_current_a=500.0)
    assert roomy.point_for_thrust("hover", 500.0, 0.0, *AIR) is None


def test_point_needing_more_than_the_bus_voltage_is_none(propulsor: Propulsor) -> None:
    point = propulsor.point_for_thrust("climb", 8.0, 0.0, *AIR)
    assert point is not None
    needed_v = point.throttle * BUS_V
    assert propulsor.point_for_thrust("climb", 8.0, 0.0, *AIR, bus_voltage_v=0.95 * needed_v) is None


def test_sagged_bus_raises_throttle_and_is_echoed(propulsor: Propulsor) -> None:
    nominal = propulsor.point_for_thrust("climb", 5.0, 12.0, *AIR)
    sagged = propulsor.point_for_thrust("climb", 5.0, 12.0, *AIR, bus_voltage_v=13.2, sagged=True)
    assert nominal is not None and sagged is not None
    assert sagged.sagged and not nominal.sagged
    assert sagged.throttle == pytest.approx(nominal.throttle * BUS_V / 13.2)
    assert sagged.rpm == pytest.approx(nominal.rpm)


def test_current_limit_caps_thrust(propulsor: Propulsor) -> None:
    limited = replace(propulsor, max_current_a=8.0)
    assert limited.thrust_n(1.0, 0.0, *AIR) < propulsor.thrust_n(1.0, 0.0, *AIR)
    full = propulsor.point_for_thrust("hover", 10.0, 0.0, *AIR)
    assert full is not None and full.current_a > 8.0
    assert limited.point_for_thrust("hover", 10.0, 0.0, *AIR) is None


@pytest.mark.parametrize("airspeed_ms", [0.0, 12.0])
def test_thrust_n_is_monotonic_in_throttle(propulsor: Propulsor, airspeed_ms: float) -> None:
    thrusts = [propulsor.thrust_n(t / 10.0, airspeed_ms, *AIR) for t in range(0, 11)]
    assert thrusts[0] == 0.0
    assert all(b >= a for a, b in zip(thrusts, thrusts[1:]))
    assert thrusts[-1] > thrusts[5] > 0.0


def test_installation_factor_scales_prop_efficiency_only(propulsor: Propulsor) -> None:
    clean = propulsor.point_for_thrust("cruise", 5.0, 12.0, *AIR)
    pusher = replace(propulsor, installation_factor=0.9).point_for_thrust("cruise", 5.0, 12.0, *AIR)
    assert clean is not None and pusher is not None
    assert pusher.prop_efficiency == pytest.approx(0.9 * clean.prop_efficiency)
    assert (pusher.rpm, pusher.current_a, pusher.throttle) == (clean.rpm, clean.current_a, clean.throttle)


@pytest.mark.parametrize("throttle", [-0.1, 1.1, math.nan])
def test_thrust_n_rejects_throttle_outside_unit_range(propulsor: Propulsor, throttle: float) -> None:
    with pytest.raises(ValueError):
        propulsor.thrust_n(throttle, 0.0, *AIR)


@pytest.mark.parametrize("bad", [{"kv": 0.0}, {"rm_ohm": -1.0}, {"esc_efficiency": 1.5}, {"bus_voltage_v": 0.0},
                                 {"installation_factor": 0.0}, {"max_current_a": math.inf}, {"i0_a": -0.1}])
def test_propulsor_rejects_bad_constants(propulsor: Propulsor, bad: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        replace(propulsor, **bad)


def test_point_for_thrust_rejects_bad_thrust(propulsor: Propulsor) -> None:
    with pytest.raises(ValueError):
        propulsor.point_for_thrust("cruise", -1.0, 10.0, *AIR)


def _parts(application: str) -> tuple[Propeller, Esc, Battery]:
    prop = Propeller(name="test prop", diameter_m=SMALL_PROP[0], pitch_m=SMALL_PROP[1], mass_kg=0.02,
                     max_power_w=600.0, application=application)
    esc = next(e for e in catalog_rows("escs") if e.max_voltage_v >= 16.8)
    battery = Battery(name="4S", chemistry="lipo", capacity_wh=74.0, max_c_rate=25.0, voltage_v=BUS_V,
                      mass_kg=0.5, length_m=0.15, width_m=0.05, height_m=0.04)
    return prop, esc, battery


def test_propulsor_from_parts_wires_catalogue_values(table: PropellerTable, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prop_tables, "_build_table", lambda d, p: table)
    prop, esc, battery = _parts("cruise")
    built = propulsor_from_parts(_motor(), prop, esc, battery, installation_factor=0.92)
    motor = _motor()
    assert (built.kv, built.rm_ohm, built.i0_a, built.max_current_a) == (motor.kv, motor.rm_ohm, motor.i0_a,
                                                                          motor.max_current_a)
    assert (built.esc_efficiency, built.bus_voltage_v, built.installation_factor) == (esc.efficiency, BUS_V, 0.92)
    assert built.table is table and built.role == "cruise"


def test_propulsor_from_parts_lift_role(table: PropellerTable, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prop_tables, "_build_table", lambda d, p: table)
    prop, esc, battery = _parts("vtol")
    lift_motor: VtolMotor = next(m for m in catalog_rows("vtol_motors") if m.kv is not None)
    built = propulsor_from_parts(lift_motor, prop, esc, battery)
    assert built.role == "lift" and built.kv == lift_motor.kv
    point = built.point_for_thrust("hover", 3.0, 0.0, *AIR)
    assert point is None or point.propulsor == "lift"


def test_propulsor_from_parts_rejects_legacy_motor(table: PropellerTable, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prop_tables, "_build_table", lambda d, p: table)
    prop, esc, battery = _parts("cruise")
    legacy = _motor().model_copy(update={"rm_ohm": None})
    with pytest.raises(ValueError, match="rm_ohm"):
        propulsor_from_parts(legacy, prop, esc, battery)
