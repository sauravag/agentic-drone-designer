"""Fix ruling L1: every generic motor frame reaches its own rating on its catalogue-paired propeller at
MAX_MATCH_THROTTLE on its nominal voltage, within its current and power limits (sea level, static)."""

import math

import pytest

from src.schemas.components import CruiseMotor, VtolMotor
from src.tools.database.seed_data import CRUISE_MOTORS, DERIVED_MOTOR_RATINGS, VTOL_MOTORS, base_propellers, catalog_rows
from src.tools.propulsion import motor_ratings as mr
from src.tools.propulsion.match_types import MAX_MATCH_THROTTLE
from src.tools.propulsion.motor import motor_point
from src.tools.propulsion.prop_tables import propeller_table

LIMIT_RTOL = 1e-3  # rounding of the stored KV / current (whole rpm/V, 0.1 A)
UNDERIVABLE = {"Axial-flux AX-90"}


def _frames(table: str) -> list:
    rows = catalog_rows(table)
    count = len(CRUISE_MOTORS if table == "cruise_motors" else VTOL_MOTORS)
    return rows[:count]  # base windings first (ordering contract)


def _check(motor: CruiseMotor | VtolMotor, rpm: float, torque: float) -> None:
    nominal = motor.voltage_v / 4.2 * 3.7
    point = motor_point(motor.kv, motor.rm_ohm, motor.i0_a, rpm, torque)
    assert point.voltage_v <= MAX_MATCH_THROTTLE * nominal * (1 + LIMIT_RTOL), motor.name
    assert point.current_a <= motor.max_current_a * (1 + LIMIT_RTOL), motor.name
    assert point.shaft_power_w <= motor.max_power_w * (1 + LIMIT_RTOL), motor.name


@pytest.mark.parametrize("motor", _frames("vtol_motors"), ids=lambda m: m.name)
def test_every_lift_frame_reaches_its_rated_thrust_on_its_paired_rotor(motor: VtolMotor) -> None:
    rotor = mr.lift_rotor_for(motor.max_thrust_n, base_propellers("vtol"))
    point = mr.static_point_for_thrust(propeller_table(rotor.diameter_m, rotor.pitch_m), motor.max_thrust_n)
    assert point is not None, f"{motor.name} cannot make {motor.max_thrust_n} N on {rotor.name}"
    _check(motor, *point)


@pytest.mark.parametrize("motor", [m for m in _frames("cruise_motors") if m.name not in UNDERIVABLE],
                         ids=lambda m: m.name)
def test_every_cruise_frame_reaches_its_max_power_on_its_paired_propeller(motor: CruiseMotor) -> None:
    prop = mr.cruise_propeller_for(motor.max_power_w, base_propellers("cruise"))
    _check(motor, *mr.static_point_for_power(propeller_table(prop.diameter_m, prop.pitch_m), motor.max_power_w))


def test_the_stored_ratings_are_what_the_model_derives() -> None:
    from src.tools.database.catalog_variants import VTOL_CONTINUOUS_FRACTION, motor_constants

    name, _mass, thrust, max_w, volts, *_ = next(r for r in VTOL_MOTORS if r[0].startswith("Lift 12030"))
    c = motor_constants(max_w * VTOL_CONTINUOUS_FRACTION, max_w, volts)
    rating = mr.lift_frame_rating(thrust, max_w, c["rm_ohm"], c["i0_a"], mr.nominal_voltage_v(volts),
                                  base_propellers("vtol"))
    kv, imax, rated = DERIVED_MOTOR_RATINGS[mr.frame_key(name)]
    assert (kv, imax, rated) == (round(rating.kv), round(rating.max_current_a, 1), round(rating.max_thrust_n))


def test_ax90_keeps_its_listed_values_because_no_propeller_absorbs_it() -> None:
    assert "Axial-flux AX-90" not in DERIVED_MOTOR_RATINGS
    with pytest.raises(mr.RatingError, match="90000"):
        mr.cruise_propeller_for(90_000.0, base_propellers("cruise"))


def test_kv_solution_sits_on_the_low_current_branch() -> None:
    kv = mr.derive_kv(3000.0, 2.0, 0.05, 1.0, 44.4)
    point = motor_point(kv, 0.05, 1.0, 3000.0, 2.0)
    assert point.voltage_v == pytest.approx(MAX_MATCH_THROTTLE * 44.4, rel=1e-6)
    assert kv < math.sqrt(60.0 * 3000.0 / (2.0 * math.pi * 2.0 * 0.05))
    with pytest.raises(mr.RatingError):
        mr.derive_kv(3000.0, 2.0, 50.0, 1.0, 44.4)  # copper loss too big for any winding
