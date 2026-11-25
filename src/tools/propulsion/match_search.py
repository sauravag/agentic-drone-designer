"""Candidate evaluation for the propulsion matching search (spec 6.4).

The search is staged so the expensive part (a propeller's table, ~2 s cold) only runs for propellers that survive
cheap pruning, and the rpm solve runs once per (propeller, flight point), not per motor or series count:

1. Propeller: diameter, folding and power-rating limits against the momentum-theory ideal power (a lower bound on
   shaft power), then the table: rpm and torque per flight point, tip Mach and the propeller's power rating.
2. Drive (propeller + motor winding): the motor's current and power at each point. Current does not depend on the
   bus voltage, so one drive serves every series count.
3. Option (drive + series count + ESC): throttle at the nominal and sagged bus, the motor's voltage rating, the
   lightest ESC, the generator's power (hybrid), and the objective (hardware + energy-storage mass).

Every rule failure is returned as a `Rejection` (rule, flight point) so a failed search can name its limit.
"""

import logging
import math
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Final, Optional, Union

from src.schemas.assumptions import Chemistry
from src.schemas.components import CruiseMotor, Esc, Propeller, VtolMotor
from src.tools.database.catalog_variants import VOLTAGE_EPS_V
from src.tools.database.packs import LOADED_MIN_V_CELL
from src.tools.propulsion.match_types import (
    ESC_CURRENT_MARGIN, MAX_MATCH_THROTTLE, MAX_SAGGED_THROTTLE, PROP_TIP_MACH_MAX, FlightPoint,
)
from src.tools.propulsion.bemt import PropModelError
from src.tools.propulsion.motor import MotorPoint, motor_point
from src.tools.propulsion.prop_tables import PropellerTable, equivalent_density, propeller_table
from src.tools.propulsion.propulsor import shaft_for_thrust

logger: logging.Logger = logging.getLogger(__name__)
Motor = Union[CruiseMotor, VtolMotor]
SECONDS_PER_MINUTE: Final[float] = 60.0
# Pruning before any table is built (builder's call, spec 8): a propeller or motor rated more than this many times
# the ideal (momentum-theory) power of its hardest point is skipped as oversized. A real set runs at 1.2-2x ideal.
PROP_OVERSIZE_FACTOR: Final[float] = 6.0
MOTOR_OVERSIZE_FACTOR: Final[float] = 6.0
RULE_MODEL: Final[str] = "propeller model failed (no table)"
RULE_OVERSIZE: Final[str] = "oversize prune"  # not a limit: a candidate skipped as rated above 6x ideal power
RULE_TABLE: Final[str] = "thrust out of propeller range (tip Mach 0.80 or pitch speed)"
RULE_AIR: Final[str] = "air outside the propeller tables"
# A point this short (e.g. Hover_Peak, 0 s) is a transient the ESC's burst rating covers; longer points need the
# continuous rating with ESC_CURRENT_MARGIN. 10 s is the usual ESC burst rating window (ESTIMATE).
ESC_BURST_DURATION_S: Final[float] = 10.0


@dataclass(frozen=True)
class Rejection:
    rule: str
    point: str  # the flight point that failed ("" for a set-level rule)


@dataclass(frozen=True)
class Bus:
    """A pack voltage class (one chemistry and cell voltages) at one series count."""

    chemistry: Chemistry
    nominal_cell_v: float
    max_cell_v: float
    series: int
    storage_kg_per_j: float  # battery (electric) or fuel + fuel system (hybrid) mass per joule of mission energy
    generator_cap_w: float = math.inf  # hybrid: the most a generator variant inside the bus window supplies
    buffer_kg_per_j: Optional[float] = None  # hybrid: buffer-battery mass per joule above the generator share

    @property
    def key(self) -> tuple[str, float, float, int]:
        return self.chemistry, self.nominal_cell_v, self.max_cell_v, self.series

    @property
    def nominal_v(self) -> float:
        return self.series * self.nominal_cell_v

    @property
    def charged_v(self) -> float:
        return self.series * self.max_cell_v

    @property
    def sagged_v(self) -> float:
        return self.series * LOADED_MIN_V_CELL[self.chemistry]


@dataclass(frozen=True)
class Shaft:
    """The propeller alone at one flight point."""

    point: FlightPoint
    rpm: float
    torque_nm: float

    @property
    def shaft_w(self) -> float:
        return self.torque_nm * self.rpm * 2.0 * math.pi / SECONDS_PER_MINUTE


@dataclass(frozen=True)
class Drive:
    """A propeller and a motor winding, solved at every flight point of its role."""

    propeller: Propeller
    motor: Motor
    shafts: tuple[Shaft, ...]
    motor_points: tuple[MotorPoint, ...]

    @cached_property
    def peak_current_a(self) -> float:
        return max(m.current_a for m in self.motor_points)

    @cached_property
    def sustained_current_a(self) -> float:
        """Highest current at the points the ESC must carry continuously (0 without any)."""
        return max((m.current_a for s, m in zip(self.shafts, self.motor_points) if not is_transient(s.point)),
                   default=0.0)

    @cached_property
    def transient_current_a(self) -> float:
        """Highest current at the transient points (within the ESC burst rating; 0 without any)."""
        return max((m.current_a for s, m in zip(self.shafts, self.motor_points) if is_transient(s.point)),
                   default=0.0)

    @cached_property
    def need_nominal(self) -> tuple[float, str]:
        """(highest motor terminal voltage, its point) over every point."""
        return max((m.voltage_v, s.point.name) for s, m in zip(self.shafts, self.motor_points))

    @cached_property
    def need_sagged(self) -> tuple[float, str]:
        """(highest motor terminal voltage, its point) over the check_sagged points; (0, "") without any."""
        return max(((m.voltage_v, s.point.name) for s, m in zip(self.shafts, self.motor_points)
                    if s.point.check_sagged), default=(0.0, ""))


@dataclass(frozen=True)
class Option:
    """A drive on one bus with its ESC: a feasible choice for one role (cruise, or the lift rotor set)."""

    drive: Drive
    esc: Esc
    bus: Bus
    count: int  # propulsors of this role
    hardware_kg: float
    energy_j: float
    storage_kg: float = 0.0  # the energy priced as sizing books it (`priced_storage_kg`)

    @property
    def objective_kg(self) -> float:
        return self.hardware_kg + self.storage_kg

    @property
    def names(self) -> tuple[str, str, str]:
        return self.drive.motor.name, self.drive.propeller.name, self.esc.name

    def electrical_w(self, shaft_index: int) -> float:
        """Bus power of ALL propulsors of this role at one point, ESC loss included."""
        m: MotorPoint = self.drive.motor_points[shaft_index]
        return self.count * m.voltage_v * m.current_a / self.esc.efficiency


def is_transient(point: FlightPoint) -> bool:
    return point.duration_s <= ESC_BURST_DURATION_S


def priced_storage_kg(powers_w: tuple[float, ...], durations_s: tuple[float, ...], generator_share_w: float,
                      fuel_kg_per_j: float, buffer_kg_per_j: float) -> float:
    """Energy-storage mass the way sizing books it: per point, the power up to the generator's share at the fuel
    rate, the power above it (climb, hover, peaks) at the buffer-battery rate. Electric: an infinite share."""
    total: float = 0.0
    for power, seconds in zip(powers_w, durations_s):
        from_generator: float = min(power, generator_share_w)
        total += (from_generator * fuel_kg_per_j + (power - from_generator) * buffer_kg_per_j) * seconds
    return total


def ideal_power_w(thrust_n: float, airspeed_ms: float, rho: float, diameter_m: float) -> float:
    """Momentum-theory power T (V + v_i) of an actuator disk: a lower bound on any propeller's shaft power."""
    disk_area: float = math.pi * diameter_m ** 2 / 4.0
    half_v: float = airspeed_ms / 2.0
    induced: float = -half_v + math.sqrt(half_v ** 2 + thrust_n / (2.0 * rho * disk_area))
    return thrust_n * (airspeed_ms + induced)


def hardest_ideal_w(propeller: Propeller, points: tuple[FlightPoint, ...], installation_factor: float) -> tuple[
        float, str]:
    """(highest ideal power, its point) of a propeller over the points (thrust grossed up by the installation)."""
    return max((ideal_power_w(p.thrust_n / installation_factor, p.airspeed_ms, p.rho, propeller.diameter_m), p.name)
               for p in points)


@lru_cache(maxsize=None)
def _table_failure(diameter_m: float, pitch_m: float) -> Optional[str]:
    """None when the propeller's table builds (it is then cached by `propeller_table`), else the model's reason.
    A failure is remembered, so a propeller that cannot be modelled costs one build attempt per process."""
    try:
        propeller_table(diameter_m, pitch_m)
    except PropModelError as exc:  # rejected and logged, never replaced by a guessed table
        logger.warning("Propeller %.4f x %.4f m skipped by matching: its table cannot be built (%s)",
                       diameter_m, pitch_m, exc)
        return str(exc)
    return None


def _air_off_axis(table: PropellerTable, point: FlightPoint) -> Optional[str]:
    """The off-axis rule text when the point's air (its equivalent density) lies outside the table's density axis,
    else None. Every table shares the axis, so the text is the same for every propeller."""
    rho_eq: float = equivalent_density(point.rho, point.mu, point.speed_of_sound_ms)
    low, high = table.rho_eq[0], table.rho_eq[-1]
    if low <= rho_eq <= high:
        return None
    return f"{RULE_AIR} (equivalent density {rho_eq:.3g} kg/m³ not in [{low:.3g}, {high:.3g}])"


def propeller_shafts(propeller: Propeller, points: tuple[FlightPoint, ...],
                     installation_factor: float) -> Union[tuple[Shaft, ...], Rejection]:
    """The propeller's rpm and torque at each point from its table, or the first rule it breaks."""
    if _table_failure(propeller.diameter_m, propeller.pitch_m) is not None:
        return Rejection(f"{RULE_MODEL}: {propeller.name}", "")
    table = propeller_table(propeller.diameter_m, propeller.pitch_m)
    shafts: list[Shaft] = []
    for p in points:
        off_axis: Optional[str] = _air_off_axis(table, p)
        if off_axis is not None:
            return Rejection(off_axis, p.name)
        solved: Optional[tuple[float, float]] = shaft_for_thrust(
            table, p.thrust_n / installation_factor, p.airspeed_ms, p.rho, p.mu, p.speed_of_sound_ms)
        if solved is None:
            return Rejection(RULE_TABLE, p.name)
        shaft = Shaft(p, *solved)
        tip_mach: float = math.pi * propeller.diameter_m * shaft.rpm / SECONDS_PER_MINUTE / p.speed_of_sound_ms
        if tip_mach > PROP_TIP_MACH_MAX:
            return Rejection("tip Mach", p.name)
        if shaft.shaft_w > propeller.max_power_w:
            return Rejection("power (propeller rating)", p.name)
        shafts.append(shaft)
    return tuple(shafts)


def _power_limit_w(motor: Motor, point: FlightPoint) -> float:
    """Peak points (climb, hover) may use the max rating; sustained points the continuous rating where given."""
    if point.check_sagged or isinstance(motor, VtolMotor):
        return motor.max_power_w
    return motor.continuous_power_w


def drive_for(propeller: Propeller, motor: Motor, shafts: tuple[Shaft, ...]) -> Union[Drive, Rejection]:
    """The motor at each of the propeller's points, or the first current or power rule it breaks."""
    points: list[MotorPoint] = []
    for s in shafts:
        m: MotorPoint = motor_point(motor.kv, motor.rm_ohm, motor.i0_a, s.rpm, s.torque_nm)
        if m.current_a > motor.max_current_a:
            return Rejection("current", s.point.name)
        if m.shaft_power_w > _power_limit_w(motor, s.point):
            return Rejection("power (motor rating)", s.point.name)
        points.append(m)
    return Drive(propeller, motor, shafts, tuple(points))


def lightest_esc(escs: tuple[Esc, ...], sustained_a: float, transient_a: float, charged_v: float) -> Optional[Esc]:
    """Lightest ESC (escs sorted by mass) with continuous current >= margin x the sustained current, burst current
    >= the transient current, and voltage >= the charged pack."""
    need_a: float = ESC_CURRENT_MARGIN * sustained_a
    return next((e for e in escs if e.continuous_current_a >= need_a and e.burst_current_a >= transient_a
                 and e.max_voltage_v >= charged_v - VOLTAGE_EPS_V), None)


def _bus_rejection(drive: Drive, bus: Bus) -> Optional[Rejection]:
    """Throttle at the nominal and the sagged bus, and the winding's voltage rating against the charged pack."""
    if bus.charged_v > drive.motor.voltage_v + VOLTAGE_EPS_V:
        return Rejection("motor voltage rating below the charged pack", "")
    need_v, point = drive.need_nominal
    if need_v > MAX_MATCH_THROTTLE * bus.nominal_v:
        return Rejection("throttle", point)
    need_sag_v, sag_point = drive.need_sagged
    if need_sag_v > MAX_SAGGED_THROTTLE * bus.sagged_v:
        return Rejection("sagged throttle", sag_point)
    return None


def option_at(drive: Drive, bus: Bus, escs: tuple[Esc, ...], count: int, sustained_cap_w: float,
              generator_share_w: float = math.inf) -> Union[Option, Rejection]:
    """The drive on this bus with its ESC, or the first bus, ESC or generator rule it breaks."""
    rejected: Optional[Rejection] = _bus_rejection(drive, bus)
    if rejected is not None:
        return rejected
    esc: Optional[Esc] = lightest_esc(escs, drive.sustained_current_a, drive.transient_current_a, bus.charged_v)
    if esc is None:
        return Rejection("no ESC", "")
    hardware: float = count * (drive.motor.mass_kg + drive.propeller.mass_kg + esc.mass_kg)
    option = Option(drive, esc, bus, count, hardware, energy_j=0.0)
    powers: list[float] = [option.electrical_w(i) for i in range(len(drive.shafts))]
    for s, power in zip(drive.shafts, powers):
        if not s.point.check_sagged and power > sustained_cap_w:
            return Rejection("generator power", s.point.name)
    durations: tuple[float, ...] = tuple(s.point.duration_s for s in drive.shafts)
    energy: float = sum(power * seconds for power, seconds in zip(powers, durations))
    buffer: float = bus.storage_kg_per_j if bus.buffer_kg_per_j is None else bus.buffer_kg_per_j
    storage: float = priced_storage_kg(tuple(powers), durations, generator_share_w, bus.storage_kg_per_j, buffer)
    return Option(drive, esc, bus, count, hardware, energy, storage)
