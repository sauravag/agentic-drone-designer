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
from src.tools.propulsion.prop_tables import PropellerTable, propeller_table, reynolds75_at

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


@dataclass(frozen=True)
class Propulsor:
    table: PropellerTable
    kv: float
    rm_ohm: float
    i0_a: float
    max_current_a: float
    esc_efficiency: float
    bus_voltage_v: float  # nominal bus; point_for_thrust may solve at another (sagged) voltage
    installation_factor: float = 1.0  # multiplies the propeller efficiency (pusher installation loss)
    role: PropulsorRole = "cruise"

    def __post_init__(self) -> None:
        for name in ("kv", "rm_ohm", "max_current_a", "bus_voltage_v"):
            _require(getattr(self, name), name, 0.0)
        _require(self.i0_a, "i0_a", 0.0, low_open=False)
        _require(self.esc_efficiency, "esc_efficiency", 0.0, 1.0)
        _require(self.installation_factor, "installation_factor", 0.0, 1.0)

    def _rpm_window(self, airspeed_ms: float, rho: float, mu: float,
                    speed_of_sound_ms: float) -> Optional[tuple[float, float]]:
        """rpm range inside the table's tip Mach, Re75 and J axes for this air and airspeed (None if empty)."""
        t: PropellerTable = self.table
        rpm_per_mach: float = speed_of_sound_ms * SECONDS_PER_MINUTE / (math.pi * t.diameter_m)
        re_per_rpm: float = reynolds75_at(t.diameter_m, t.pitch_m, 1.0, rho, mu)
        lo: float = max(t.tip_mach[0] * rpm_per_mach, t.reynolds75[0] / re_per_rpm,
                        airspeed_ms * SECONDS_PER_MINUTE / (t.diameter_m * t.j[-1]))
        hi: float = min(t.tip_mach[-1] * rpm_per_mach, t.reynolds75[-1] / re_per_rpm)
        lo, hi = lo * (1.0 + AXIS_INSET), hi * (1.0 - AXIS_INSET)
        return (lo, hi) if lo < hi else None

    def _rpm_for_thrust(self, thrust_n: float, airspeed_ms: float, rho: float, mu: float,
                        speed_of_sound_ms: float) -> Optional[float]:
        window: Optional[tuple[float, float]] = self._rpm_window(airspeed_ms, rho, mu, speed_of_sound_ms)
        if window is None:
            return None
        lo, hi = window

        def excess(rpm: float) -> float:
            return self.table.thrust_n(rpm, airspeed_ms, rho, mu, speed_of_sound_ms) - thrust_n

        f_lo, f_hi = excess(lo), excess(hi)
        if f_hi < 0.0 or f_lo > 0.0:  # beyond tip Mach 0.80 / above pitch speed, or below the table floor
            return None
        if f_hi == 0.0:
            return hi
        return _bracketed_root(excess, lo, hi, f_lo, f_hi, ROOT_THRUST_RTOL * thrust_n)

    def _efficiency(self, thrust_n: float, airspeed_ms: float, rho: float, shaft_w: float) -> float:
        if airspeed_ms <= HOVER_AIRSPEED_MS:
            disk_area: float = math.pi * self.table.diameter_m ** 2 / 4.0
            ideal_w: float = thrust_n ** 1.5 / math.sqrt(2.0 * rho * disk_area)  # momentum-theory hover power
            return self.installation_factor * ideal_w / shaft_w
        return self.installation_factor * thrust_n * airspeed_ms / shaft_w

    def point_for_thrust(self, name: str, thrust_n: float, airspeed_ms: float, rho: float, mu: float,
                         speed_of_sound_ms: float, bus_voltage_v: Optional[float] = None,
                         sagged: bool = False) -> Optional[OperatingPoint]:
        """The operating point giving `thrust_n`, or None when it is not achievable: outside the table (tip Mach
        0.80, above pitch speed), more voltage than the bus, or more current than the motor's max."""
        _require(thrust_n, "thrust_n", 0.0)
        _require_air(airspeed_ms, rho, mu, speed_of_sound_ms)
        bus: float = self.bus_voltage_v if bus_voltage_v is None else bus_voltage_v
        _require(bus, "bus_voltage_v", 0.0)
        try:
            rpm: Optional[float] = self._rpm_for_thrust(thrust_n, airspeed_ms, rho, mu, speed_of_sound_ms)
            if rpm is None:
                return None
            torque: float = self.table.torque_nm(rpm, airspeed_ms, rho, mu, speed_of_sound_ms)
            motor: MotorPoint = motor_point(self.kv, self.rm_ohm, self.i0_a, rpm, torque)  # torque < 0 raises
        except (ValueError, PropModelError):  # a table limit (range, past the computed J, CP < 0): not achievable
            return None
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
        try:
            return self.table.torque_nm(rpm, airspeed_ms, rho, mu, speed_of_sound_ms)
        except ValueError:  # inside the rpm window this is only "past the computed J": windmilling, no load
            return 0.0

    def thrust_n(self, throttle: float, airspeed_ms: float, rho: float, mu: float,
                 speed_of_sound_ms: float) -> float:
        """Thrust (N, >= 0) at `throttle` x bus voltage. Zero when the motor cannot turn the propeller up to the
        table's lowest rpm; held at the table's top rpm (tip Mach 0.80) when the motor could spin it faster."""
        _require(throttle, "throttle", 0.0, 1.0, low_open=False)
        _require_air(airspeed_ms, rho, mu, speed_of_sound_ms)
        volts: float = throttle * self.bus_voltage_v
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
        return self.table.thrust_n(rpm, airspeed_ms, rho, mu, speed_of_sound_ms)


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
