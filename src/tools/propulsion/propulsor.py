"""A propulsor: one propeller table, one motor winding and its ESC on a bus voltage (spec 6.3, 6.4).

`point_for_thrust` solves the rpm at which the table gives the requested thrust (a bounded 1-D root solve on rpm,
tables only, no BEMT), then the motor's voltage and current at that rpm and torque. `thrust_n` is the inverse for
the simulator: the rpm at which the motor torque at throttle x bus voltage equals the propeller torque.
"""

import math
from dataclasses import dataclass
from typing import Callable, Final, Optional

from src.schemas.components import Battery, CruiseMotor, Esc, Propeller, VtolMotor
from src.schemas.propulsion import OperatingPoint, PropulsorRole
from src.tools.propulsion.bemt import PropModelError
from src.tools.propulsion.motor import MotorPoint, motor_point
from src.tools.propulsion.prop_tables import PropellerTable, equivalent_density, propeller_table

SECONDS_PER_MINUTE: Final[float] = 60.0
AXIS_INSET: Final[float] = 1e-9  # relative: keep rpm bounds strictly inside the table axes
ROOT_RTOL: Final[float] = 1e-9  # relative rpm bracket at which the root solve stops
ROOT_THRUST_RTOL: Final[float] = 1e-7  # relative thrust residual at which point_for_thrust's solve stops
ROOT_MAX_ITER: Final[int] = 100
HOVER_AIRSPEED_MS: Final[float] = 1e-6  # at or below this, efficiency is the figure of merit
MOTOR_PARAMS: Final[tuple[str, ...]] = ("kv", "rm_ohm", "i0_a", "max_current_a")


def _bracketed_root(f: Callable[[float], float], lo: float, hi: float, f_lo: float, f_hi: float,
                    f_tol: float = 0.0) -> float:
    """Root of f in [lo, hi] (f_lo and f_hi of opposite sign) by the Illinois false-position method."""
    side: int = 0
    x: float = lo
    for _ in range(ROOT_MAX_ITER):
        x = (f_lo * hi - f_hi * lo) / (f_lo - f_hi)
        fx: float = f(x)
        if abs(fx) <= f_tol or fx == 0.0:
            return x
        if (fx > 0.0) == (f_hi > 0.0):
            hi, f_hi = x, fx
            f_lo = f_lo / 2.0 if side == -1 else f_lo
            side = -1
        else:
            lo, f_lo = x, fx
            f_hi = f_hi / 2.0 if side == 1 else f_hi
            side = 1
        if hi - lo <= ROOT_RTOL * x:
            return x
    return x


def _require(value: float, name: str, low: float, high: float = math.inf, low_open: bool = True) -> None:
    ok_low: bool = value > low if low_open else value >= low
    if not (math.isfinite(value) and ok_low and value <= high):
        raise ValueError(f"{name} must be finite and in {'(' if low_open else '['}{low}, {high}], got {value!r}")


def _require_air(airspeed_ms: float, rho: float, mu: float, speed_of_sound_ms: float) -> None:
    _require(airspeed_ms, "airspeed_ms", 0.0, low_open=False)
    for value, name in ((rho, "rho"), (mu, "mu"), (speed_of_sound_ms, "speed_of_sound_ms")):
        _require(value, name, 0.0)


def rpm_window(table: PropellerTable, airspeed_ms: float, rho: float, mu: float,
               speed_of_sound_ms: float) -> Optional[tuple[float, float]]:
    """rpm range inside the table's tip Mach and J axes for this air and airspeed (None if empty or if the air's
    equivalent density lies outside the table)."""
    rho_eq: float = equivalent_density(rho, mu, speed_of_sound_ms)
    if not table.rho_eq[0] <= rho_eq <= table.rho_eq[-1]:
        return None
    rpm_per_mach: float = speed_of_sound_ms * SECONDS_PER_MINUTE / (math.pi * table.diameter_m)
    lo: float = max(table.tip_mach[0] * rpm_per_mach,
                    airspeed_ms * SECONDS_PER_MINUTE / (table.diameter_m * table.j[-1]))
    hi: float = table.tip_mach[-1] * rpm_per_mach
    lo, hi = lo * (1.0 + AXIS_INSET), hi * (1.0 - AXIS_INSET)
    return (lo, hi) if lo < hi else None


def _rpm_for_thrust(table: PropellerTable, thrust_n: float, airspeed_ms: float, rho: float, mu: float,
                    speed_of_sound_ms: float) -> Optional[float]:
    window: Optional[tuple[float, float]] = rpm_window(table, airspeed_ms, rho, mu, speed_of_sound_ms)
    if window is None:
        return None
    lo, hi = window

    def excess(rpm: float) -> float:
        return table.thrust_n(rpm, airspeed_ms, rho, mu, speed_of_sound_ms) - thrust_n

    f_lo, f_hi = excess(lo), excess(hi)
    if f_hi < 0.0 or f_lo > 0.0:  # beyond tip Mach 0.80 / above pitch speed, or below the table floor
        return None
    if f_hi == 0.0:
        return hi
    return _bracketed_root(excess, lo, hi, f_lo, f_hi, ROOT_THRUST_RTOL * thrust_n)


def shaft_for_thrust(table: PropellerTable, prop_thrust_n: float, airspeed_ms: float, rho: float, mu: float,
                     speed_of_sound_ms: float) -> Optional[tuple[float, float]]:
    """(rpm, shaft torque N m) at which the propeller itself makes `prop_thrust_n`, or None when the table cannot
    (tip Mach 0.80, above pitch speed, below the table floor, a negative or out-of-range torque). Motor-free, so
    the matching search solves it once per propeller and flight point."""
    try:
        rpm: Optional[float] = _rpm_for_thrust(table, prop_thrust_n, airspeed_ms, rho, mu, speed_of_sound_ms)
        if rpm is None:
            return None
        torque: float = table.torque_nm(rpm, airspeed_ms, rho, mu, speed_of_sound_ms)
    except (ValueError, PropModelError):  # a table limit (range, past the computed J): not achievable
        return None
    return (rpm, torque) if torque >= 0.0 else None


@dataclass(frozen=True)
class Propulsor:
    table: PropellerTable
    kv: float
    rm_ohm: float
    i0_a: float
    max_current_a: float
    esc_efficiency: float
    bus_voltage_v: float  # nominal bus; point_for_thrust may solve at another (sagged) voltage
    # Delivered / propeller thrust (pusher installation loss): the propeller makes T / factor to deliver T, so the
    # loss costs rpm, current and power, and the efficiencies are quoted on the delivered thrust.
    installation_factor: float = 1.0
    role: PropulsorRole = "cruise"

    def __post_init__(self) -> None:
        for name in ("kv", "rm_ohm", "max_current_a", "bus_voltage_v"):
            _require(getattr(self, name), name, 0.0)
        _require(self.i0_a, "i0_a", 0.0, low_open=False)
        _require(self.esc_efficiency, "esc_efficiency", 0.0, 1.0)
        _require(self.installation_factor, "installation_factor", 0.0, 1.0)

    def _rpm_window(self, airspeed_ms: float, rho: float, mu: float,
                    speed_of_sound_ms: float) -> Optional[tuple[float, float]]:
        return rpm_window(self.table, airspeed_ms, rho, mu, speed_of_sound_ms)

    def _efficiency(self, delivered_n: float, airspeed_ms: float, rho: float, shaft_w: float) -> float:
        """T V / P_shaft forward, the figure of merit in hover, both on the delivered thrust."""
        if airspeed_ms <= HOVER_AIRSPEED_MS:
            disk_area: float = math.pi * self.table.diameter_m ** 2 / 4.0
            ideal_w: float = delivered_n ** 1.5 / math.sqrt(2.0 * rho * disk_area)  # momentum-theory hover power
            return ideal_w / shaft_w
        return delivered_n * airspeed_ms / shaft_w

    def point_for_thrust(self, name: str, thrust_n: float, airspeed_ms: float, rho: float, mu: float,
                         speed_of_sound_ms: float, bus_voltage_v: Optional[float] = None,
                         sagged: bool = False) -> Optional[OperatingPoint]:
        """The operating point delivering `thrust_n` (the propeller makes thrust_n / installation_factor), or None
        when it is not achievable: outside the table (tip Mach 0.80, above pitch speed, below the lowest tabulated
        rpm), more voltage than the bus, or more current than the motor's max."""
        _require(thrust_n, "thrust_n", 0.0)
        _require_air(airspeed_ms, rho, mu, speed_of_sound_ms)
        bus: float = self.bus_voltage_v if bus_voltage_v is None else bus_voltage_v
        _require(bus, "bus_voltage_v", 0.0)
        shaft: Optional[tuple[float, float]] = shaft_for_thrust(
            self.table, thrust_n / self.installation_factor, airspeed_ms, rho, mu, speed_of_sound_ms)
        if shaft is None:
            return None
        rpm: float = shaft[0]
        motor: MotorPoint = motor_point(self.kv, self.rm_ohm, self.i0_a, rpm, shaft[1])
        if motor.current_a > self.max_current_a or motor.voltage_v > bus or motor.shaft_power_w <= 0.0:
            return None
        return OperatingPoint(
            flight_point=name, propulsor=self.role, rpm=rpm, current_a=motor.current_a,
            throttle=motor.voltage_v / bus,
            prop_efficiency=self._efficiency(thrust_n, airspeed_ms, rho, motor.shaft_power_w),
            motor_efficiency=motor.efficiency,
            tip_mach=math.pi * self.table.diameter_m * rpm / SECONDS_PER_MINUTE / speed_of_sound_ms,
            electrical_power_w=motor.voltage_v * motor.current_a / self.esc_efficiency, sagged=sagged,
        )

    def _motor_torque_nm(self, motor_voltage_v: float, rpm: float) -> float:
        """Shaft torque at a terminal voltage and rpm, with the current clamped at max_current_a."""
        current: float = min((motor_voltage_v - rpm / self.kv) / self.rm_ohm, self.max_current_a)
        return SECONDS_PER_MINUTE / (2.0 * math.pi * self.kv) * (current - self.i0_a)

    def _prop_torque_nm(self, rpm: float, airspeed_ms: float, rho: float, mu: float,
                        speed_of_sound_ms: float) -> float:
        if self.table.past_computed_j(rpm, airspeed_ms, rho, mu, speed_of_sound_ms):
            return 0.0  # past zero thrust (windmilling): the propeller loads the motor with nothing
        return self.table.torque_nm(rpm, airspeed_ms, rho, mu, speed_of_sound_ms)

    def thrust_n(self, throttle: float, airspeed_ms: float, rho: float, mu: float, speed_of_sound_ms: float,
                 bus_voltage_v: Optional[float] = None) -> float:
        """Delivered thrust (N, >= 0: propeller thrust x installation_factor) at `throttle` x bus voltage (default
        the nominal bus; pass a sagged value to model sag). Zero when the motor cannot turn the propeller up to the
        table's lowest rpm; held at the table's top rpm (tip Mach 0.80) when the motor could spin it faster."""
        _require(throttle, "throttle", 0.0, 1.0, low_open=False)
        _require_air(airspeed_ms, rho, mu, speed_of_sound_ms)
        bus: float = self.bus_voltage_v if bus_voltage_v is None else bus_voltage_v
        _require(bus, "bus_voltage_v", 0.0)
        volts: float = throttle * bus
        window: Optional[tuple[float, float]] = self._rpm_window(airspeed_ms, rho, mu, speed_of_sound_ms)
        if window is None:
            return 0.0
        lo, hi = window[0], min(window[1], self.kv * (volts - self.i0_a * self.rm_ohm))  # motor torque 0 at hi

        def surplus(rpm: float) -> float:
            return self._motor_torque_nm(volts, rpm) - self._prop_torque_nm(rpm, airspeed_ms, rho, mu,
                                                                           speed_of_sound_ms)

        if hi <= lo or (f_lo := surplus(lo)) <= 0.0:
            return 0.0
        f_hi: float = surplus(hi)
        rpm: float = hi if f_hi >= 0.0 else _bracketed_root(surplus, lo, hi, f_lo, f_hi)
        return self.installation_factor * self.table.thrust_n(rpm, airspeed_ms, rho, mu, speed_of_sound_ms)


def propulsor_from_parts(motor: CruiseMotor | VtolMotor, propeller: Propeller, esc: Esc, battery: Battery,
                         installation_factor: float = 1.0) -> Propulsor:
    """A propulsor from catalogue parts: the motor winding's constants, the propeller's cached table, the ESC's
    efficiency and the pack's nominal voltage. Legacy motor rows without constants raise ValueError."""
    missing: list[str] = [name for name in MOTOR_PARAMS if getattr(motor, name) is None]
    if missing:
        raise ValueError(f"motor {motor.name!r} has no {', '.join(missing)}: it cannot be matched")
    return Propulsor(
        table=propeller_table(propeller.diameter_m, propeller.pitch_m), kv=motor.kv, rm_ohm=motor.rm_ohm,
        i0_a=motor.i0_a, max_current_a=motor.max_current_a, esc_efficiency=esc.efficiency,
        bus_voltage_v=battery.voltage_v, installation_factor=installation_factor,
        role="lift" if propeller.application == "vtol" else "cruise",
    )
