"""Self-consistent generic motor frames (fix ruling L1): each frame's KV and max current are DERIVED from the
propulsor model so the frame reaches its own rating on its catalogue-paired propeller at MAX_MATCH_THROTTLE on its
nominal voltage, at sea level, static.

Pairing (the catalogue's own sizing rules):
- lift frame: the rotor the sizing would give a quadplane whose peak per-rotor thrust (T/W VTOL_THRUST_TO_WEIGHT,
  VTOL rotors) is the frame's max_thrust_n - the smallest catalogue lift diameter at or above the disk-loading
  diameter at that MTOW (base pitch);
- cruise frame: the lightest base cruise propeller rated for the frame's max power (the former BOM pairing).

Rating point: lift - max_thrust_n; cruise - max_power_w of shaft power. KV is solved so the motor voltage there is
MAX_MATCH_THROTTLE x the nominal voltage; max current is the larger of max power / nominal voltage and the current
at the rating point. Masses and power ratings are kept. Where a lift frame's listed thrust needs more shaft power
than its max power on its paired rotor (no KV can fix a power deficit), the thrust rating becomes what max power
delivers there. Everything is an ESTIMATE. Run `python -m src.tools.propulsion.motor_ratings` to regenerate the table
seed_data holds (DERIVED_MOTOR_RATINGS); tests/test_motor_ratings.py checks every row against this model.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Optional

from src.schemas.components import Propeller
from src.tools.database.catalog_variants import NOMINAL_CELL_V
from src.tools.database.seed_data import frame_key  # noqa: F401  (re-exported: one definition)
from src.tools.propulsion.match_types import MAX_MATCH_THROTTLE
from src.tools.propulsion.motor import MotorPoint, motor_point
from src.tools.propulsion.prop_tables import PropellerTable, propeller_table
from src.tools.propulsion.propulsor import rpm_window, shaft_for_thrust
from src.tools.sizing.aero_model import air_properties, speed_of_sound_ms
from src.tools.sizing.mission_profile import VTOL_THRUST_TO_WEIGHT, disk_loading_n_m2
from src.units import G0

RATING_RHO, RATING_MU = air_properties(0.0)
RATING_SPEED_OF_SOUND: Final[float] = speed_of_sound_ms(0.0)
RATING_ROTOR_COUNT: Final[int] = 4  # quadplane lift rotors (mission_solver.VTOL_ROTOR_COUNT)
KV_BRACKET: Final[tuple[float, float]] = (1.0, 20_000.0)
BISECT_STEPS: Final[int] = 80
CURRENT_STEPS: Final[float] = 10.0  # max current stored to 0.1 A
THRUST_RTOL: Final[float] = 1e-6


class RatingError(ValueError):
    """The frame cannot be made consistent on its paired propeller (the message says why)."""


@dataclass(frozen=True)
class FrameRating:
    kv: float
    max_current_a: float
    max_thrust_n: Optional[float]  # lift frames only (possibly reduced to what max power delivers)
    propeller: str
    rpm: float
    shaft_power_w: float


def nominal_voltage_v(listed_v: float) -> float:
    """The frame's nominal voltage: its rated cell count (listed / 3.7 V, as the catalogue rounds it) x 3.7 V."""
    return max(1, round(listed_v / NOMINAL_CELL_V)) * NOMINAL_CELL_V


def lift_rotor_for(max_thrust_n: float, rotors: Sequence[Propeller]) -> Propeller:
    """The smallest catalogue lift rotor at or above the disk-loading diameter of the quadplane this frame lifts."""
    mtow_kg: float = RATING_ROTOR_COUNT * max_thrust_n / (VTOL_THRUST_TO_WEIGHT * G0)
    area_m2: float = mtow_kg * G0 / disk_loading_n_m2(mtow_kg) / RATING_ROTOR_COUNT
    diameter_m: float = math.sqrt(4.0 * area_m2 / math.pi)
    big_enough: list[Propeller] = [p for p in rotors if p.diameter_m >= diameter_m]
    return min(big_enough, key=lambda p: p.diameter_m) if big_enough else max(rotors, key=lambda p: p.diameter_m)


def _absorbs(prop: Propeller, shaft_w: float) -> bool:
    table: PropellerTable = propeller_table(prop.diameter_m, prop.pitch_m)
    window = rpm_window(table, 0.0, RATING_RHO, RATING_MU, RATING_SPEED_OF_SOUND)
    return window is not None and _shaft_w(table, window[1]) >= shaft_w


def cruise_propeller_for(max_power_w: float, props: Sequence[Propeller]) -> Propeller:
    """The lightest cruise propeller rated for the frame's max power that can also absorb it static below tip
    Mach 0.80 (first row on a tie)."""
    rated: list[Propeller] = sorted((p for p in props if p.max_power_w >= max_power_w), key=lambda p: p.mass_kg)
    usable: Optional[Propeller] = next((p for p in rated if _absorbs(p, max_power_w)), None)
    if usable is None:
        raise RatingError(f"no cruise propeller is rated for and absorbs {max_power_w:g} W")
    return usable


def _shaft_w(table: PropellerTable, rpm: float) -> float:
    return table.torque_nm(rpm, 0.0, RATING_RHO, RATING_MU, RATING_SPEED_OF_SOUND) * rpm * 2.0 * math.pi / 60.0


def _bisect(f: Callable[[float], float], lo: float, hi: float) -> float:
    """Root of an increasing f on [lo, hi] (f(lo) < 0 < f(hi) checked by the caller)."""
    for _ in range(BISECT_STEPS):
        mid: float = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < 0.0 else (lo, mid)
    return 0.5 * (lo + hi)


def static_point_for_power(table: PropellerTable, shaft_w: float) -> tuple[float, float]:
    """(rpm, torque) at which the propeller absorbs `shaft_w` static at sea level."""
    window = rpm_window(table, 0.0, RATING_RHO, RATING_MU, RATING_SPEED_OF_SOUND)
    if window is None or _shaft_w(table, window[1]) < shaft_w:
        raise RatingError(f"the {table.diameter_m:.3f} m propeller cannot absorb {shaft_w:g} W below tip Mach 0.80")
    rpm: float = _bisect(lambda r: _shaft_w(table, r) - shaft_w, *window)
    return rpm, shaft_w / (rpm * 2.0 * math.pi / 60.0)


def static_point_for_thrust(table: PropellerTable, thrust_n: float) -> Optional[tuple[float, float]]:
    return shaft_for_thrust(table, thrust_n, 0.0, RATING_RHO, RATING_MU, RATING_SPEED_OF_SOUND)


def derive_kv(rpm: float, torque_nm: float, rm_ohm: float, i0_a: float, nominal_v: float) -> float:
    """KV at which the motor needs exactly MAX_MATCH_THROTTLE x nominal voltage at (rpm, torque).

    V(KV) = rpm / KV + (torque 2 pi KV / 60 + I0) Rm falls, then rises (copper loss); its minimum sits at
    KV* = sqrt(60 rpm / (2 pi torque Rm)). The root is taken on the falling branch (the lower-current winding)."""
    target: float = MAX_MATCH_THROTTLE * nominal_v

    def surplus(kv: float) -> float:
        return target - motor_point(kv, rm_ohm, i0_a, rpm, torque_nm).voltage_v

    lo: float = KV_BRACKET[0]
    best: float = min(math.sqrt(60.0 * rpm / (2.0 * math.pi * max(torque_nm, 1e-12) * rm_ohm)), KV_BRACKET[1])
    if surplus(lo) >= 0.0 or surplus(best) <= 0.0:
        raise RatingError(f"no KV meets {target:.1f} V at {rpm:.0f} rpm, {torque_nm:.3g} N m (Rm {rm_ohm:.3g})")
    return _bisect(surplus, lo, best)


def rate_frame(rpm: float, torque_nm: float, rm_ohm: float, i0_a: float, nominal_v: float,
               max_power_w: float) -> tuple[float, float, MotorPoint]:
    """(KV, max current, the motor point) that make the rating point reachable at MAX_MATCH_THROTTLE."""
    # Stored as a whole KV rounded up (a little lower voltage) and the current rounded up to 0.1 A at that KV, so
    # the stored frame still meets the rating.
    kv: float = float(math.ceil(derive_kv(rpm, torque_nm, rm_ohm, i0_a, nominal_v)))
    point: MotorPoint = motor_point(kv, rm_ohm, i0_a, rpm, torque_nm)
    return kv, math.ceil(max(max_power_w / nominal_v, point.current_a) * CURRENT_STEPS) / CURRENT_STEPS, point


def lift_rating_point(max_thrust_n: float, max_power_w: float, table: PropellerTable) -> tuple[float, float, float]:
    """(rated thrust, rpm, torque): the listed thrust, or what max power delivers when the thrust needs more."""
    point = static_point_for_thrust(table, max_thrust_n)
    if point is not None and point[0] * point[1] * 2.0 * math.pi / 60.0 <= max_power_w:
        return max_thrust_n, point[0], point[1]
    rpm, torque = static_point_for_power(table, max_power_w)
    thrust: float = table.thrust_n(rpm, 0.0, RATING_RHO, RATING_MU, RATING_SPEED_OF_SOUND)
    return thrust, rpm, torque


def lift_frame_rating(thrust_n: float, max_w: float, rm_ohm: float, i0_a: float, nominal_v: float,
                      rotors: Sequence[Propeller]) -> FrameRating:
    """The rating on the paired rotor. When max power cuts the thrust, the lower thrust may pair with a smaller
    rotor, so the pairing is repeated until the rotor holds (at most once per catalogue rotor)."""
    rated: float = thrust_n
    for _ in range(len(rotors)):
        rotor: Propeller = lift_rotor_for(rated, rotors)
        new_rated, rpm, torque = lift_rating_point(rated, max_w, propeller_table(rotor.diameter_m, rotor.pitch_m))
        new_rated = math.floor(new_rated)  # stored as whole newtons, never above what the rotor gives
        if lift_rotor_for(new_rated, rotors).name == rotor.name:
            if new_rated < rated:
                rpm, torque = _thrust_point(rotor, new_rated)
            kv, imax, point = rate_frame(rpm, torque, rm_ohm, i0_a, nominal_v, max_w)
            return FrameRating(kv, imax, new_rated, rotor.name, rpm, point.shaft_power_w)
        rated = new_rated
    raise RatingError(f"lift frame pairing did not settle for {thrust_n:g} N")


def _thrust_point(rotor: Propeller, thrust_n: float) -> tuple[float, float]:
    point = static_point_for_thrust(propeller_table(rotor.diameter_m, rotor.pitch_m), thrust_n)
    if point is None:
        raise RatingError(f"{rotor.name} cannot make {thrust_n:g} N static")
    return point


def cruise_frame_rating(max_w: float, rm_ohm: float, i0_a: float, nominal_v: float,
                        props: Sequence[Propeller]) -> FrameRating:
    prop: Propeller = cruise_propeller_for(max_w, props)
    rpm, torque = static_point_for_power(propeller_table(prop.diameter_m, prop.pitch_m), max_w)
    kv, imax, point = rate_frame(rpm, torque, rm_ohm, i0_a, nominal_v, max_w)
    return FrameRating(kv, imax, None, prop.name, rpm, point.shaft_power_w)


def _print_table() -> None:
    from src.tools.database.catalog_variants import VTOL_CONTINUOUS_FRACTION, motor_constants
    from src.tools.database.seed_data import CRUISE_MOTORS, VTOL_MOTORS, base_propellers

    print("[Tool] motor_ratings: frame -> (kv, max_current_a, max_thrust_n or None)")
    rotors, props = base_propellers("vtol"), base_propellers("cruise")
    for name, _mass, thrust, max_w, volts, *_ in VTOL_MOTORS:
        c = motor_constants(max_w * VTOL_CONTINUOUS_FRACTION, max_w, volts)
        r = lift_frame_rating(thrust, max_w, c["rm_ohm"], c["i0_a"], nominal_voltage_v(volts), rotors)
        print(f"    {frame_key(name)!r}: ({r.kv:g}, {r.max_current_a:.1f}, {r.max_thrust_n:g}),"
              f"  # {r.propeller} {r.rpm:.0f} rpm {r.shaft_power_w:.0f} W")
    for name, _mass, max_w, cont_w, _kv, volts, *_ in CRUISE_MOTORS:
        c = motor_constants(cont_w, max_w, volts)
        try:
            r = cruise_frame_rating(max_w, c["rm_ohm"], c["i0_a"], nominal_voltage_v(volts), props)
        except RatingError as exc:
            print(f"    # {frame_key(name)!r}: listed values kept ({exc})")
            continue
        print(f"    {frame_key(name)!r}: ({r.kv:g}, {r.max_current_a:.1f}, None),"
              f"  # {r.propeller} {r.rpm:.0f} rpm {r.shaft_power_w:.0f} W")


if __name__ == "__main__":
    _print_table()
