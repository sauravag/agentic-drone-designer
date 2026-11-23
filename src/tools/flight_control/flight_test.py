"""pymavlink pilot that flies the SITL acceptance sequence (SRS 4.4 Module D.2).

Sequence: arm -> take off (FBWA runway take-off, or QHOVER climb for a quadplane) -> FBWB
(quadplane: this is the transition, its altitude loss is recorded) -> trim at cruise -> FBWA with the
trim throttle -> +15 deg roll step and back -> +10 deg pitch step and back.

Steps are injected as RC stick overrides in FBWA, where ArduPilot maps stick deflection linearly to a
demanded angle (roll = stick * ROLL_LIMIT_DEG, pitch = stick * PTCH_LIM_MAX/MIN_DEG; STAB_PITCH_DOWN is
zeroed in SITL so the map is exact); the demanded angle it used is read back from NAV_CONTROLLER_OUTPUT.
Like a test pilot, the sequence starts each step from the trimmed pitch measured in FBWB, and adds the
climb power the flight model predicts for the +10 deg step so airspeed (and hence trim) stays put: the
test then measures the attitude loops, not the airframe's speed stability. All timing uses simulation time
(ATTITUDE.time_boot_ms), so results do not depend on the SITL speed-up.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Final, Literal, Mapping, Optional

from pymavlink import mavutil

from src.tools.sizing.topology import HAND_LAUNCH_HEIGHT_M

logger = logging.getLogger(__name__)

ROLL_STEP_DEG: Final[float] = 15.0  # SRS
PITCH_STEP_DEG: Final[float] = 10.0  # SRS
STEP_HOLD_S: Final[float] = 6.0
RECOVER_S: Final[float] = 5.0
PRE_STEP_HOLD_S: Final[float] = 4.0
CRUISE_SETTLE_S: Final[float] = 12.0
THROTTLE_AVERAGE_S: Final[float] = 3.0
QUAD_CLIMB_ALT_M: Final[float] = 40.0
QUAD_CLIMB_STICK_PWM: Final[int] = 1800
TAKEOFF_ALT_FRACTION: Final[float] = 0.9
TAKEOFF_PITCH_DEG: Final[float] = 10.0  # climb attitude after rotation
TRANSITION_TIMEOUT_S: Final[float] = 60.0
TRANSITION_DONE_TEXT: Final[str] = "transition done"
POST_TRANSITION_S: Final[float] = 3.0
TAKEOFF_TIMEOUT_S: Final[float] = 150.0
GROUND_CONTACT_MARGIN_M: Final[float] = 0.1  # altitude above the ground at which a hand-launched flight has failed
ARM_TIMEOUT_S: Final[float] = 120.0
ARM_RETRY_S: Final[float] = 3.0
MODE_TIMEOUT_S: Final[float] = 10.0
HEARTBEAT_TIMEOUT_S: Final[float] = 60.0
STALL_TIMEOUT_S: Final[float] = 30.0  # wall seconds without simulation time advancing
RECV_BLOCK_S: Final[float] = 0.05
RC_SEND_PERIOD_S: Final[float] = 0.02
PWM_TRIM: Final[int] = 1500
PWM_MIN: Final[int] = 1000
PWM_SPAN: Final[int] = 1000
PITCH_UP_PWM_SIGN: Final[int] = 1  # ArduPlane: RC2 above trim demands nose up (RC2_REVERSED = 0)
ARMED_FLAG: Final[int] = mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
# 25 Hz: ArduPlane's default 50 Hz scheduler rejects faster streams
MESSAGE_INTERVALS_US: Final[dict[int, int]] = {
    mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE: 40_000,
    mavutil.mavlink.MAVLINK_MSG_ID_NAV_CONTROLLER_OUTPUT: 40_000,
    mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT: 50_000,
    mavutil.mavlink.MAVLINK_MSG_ID_VFR_HUD: 100_000,
}


class SitlFlightError(RuntimeError):
    """The SITL flight could not complete a phase (timeout, arming refused, connection lost)."""


@dataclass(frozen=True)
class Sample:
    t_s: float
    roll_deg: float
    pitch_deg: float
    nav_roll_deg: float
    nav_pitch_deg: float
    alt_m: float
    airspeed_ms: float
    throttle_pct: float


@dataclass
class FlightLog:
    samples: list[Sample] = field(default_factory=list)
    texts: list[tuple[float, str]] = field(default_factory=list)
    roll_step_t_s: Optional[float] = None
    pitch_step_t_s: Optional[float] = None
    transition_start_t_s: Optional[float] = None
    transition_end_t_s: Optional[float] = None
    transition_start_alt_m: Optional[float] = None
    trim_throttle_pct: Optional[float] = None
    trim_pitch_deg: Optional[float] = None
    pid: list[tuple[float, int, float, float, float, float, float, float]] = field(default_factory=list)
    # (t, axis 1=roll 2=pitch, desired deg/s, achieved deg/s, FF, P, I, D) from PID_TUNING


class SitlPilot:
    """Thin stateful wrapper: pumps telemetry, keeps RC overrides alive, waits on sim-time conditions."""

    def __init__(self, conn: Any, health: Optional[Callable[[], Optional[str]]] = None) -> None:
        self._conn, self._health = conn, health
        self.log = FlightLog()
        self.t_s = 0.0
        self.nav_roll = self.nav_pitch = 0.0
        self.alt_m = self.airspeed = self.throttle = 0.0
        self.armed = False
        self.mode: Optional[int] = None
        self.rc: list[int] = [PWM_TRIM, PWM_TRIM, PWM_MIN, PWM_TRIM]
        self._last_rc_send = 0.0

    def connect(self) -> None:
        if self._conn.wait_heartbeat(timeout=HEARTBEAT_TIMEOUT_S) is None:
            raise SitlFlightError("no heartbeat from arduplane")
        for msg_id, interval in MESSAGE_INTERVALS_US.items():
            self._command(mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, msg_id, interval)

    def _command(self, command: int, *params: float) -> None:
        values: list[float] = list(params) + [0.0] * (7 - len(params))
        self._conn.mav.command_long_send(self._conn.target_system, self._conn.target_component, command, 0, *values)

    def _handle(self, msg: Any) -> None:
        kind: str = msg.get_type()
        if kind == "ATTITUDE":
            self.t_s = msg.time_boot_ms / 1000.0
            self.log.samples.append(Sample(self.t_s, math.degrees(msg.roll), math.degrees(msg.pitch), self.nav_roll,
                                           self.nav_pitch, self.alt_m, self.airspeed, self.throttle))
        elif kind == "NAV_CONTROLLER_OUTPUT":
            self.nav_roll, self.nav_pitch = msg.nav_roll, msg.nav_pitch
        elif kind == "GLOBAL_POSITION_INT":
            self.alt_m = msg.relative_alt / 1000.0
        elif kind == "VFR_HUD":
            self.airspeed, self.throttle = msg.airspeed, msg.throttle
        elif kind == "HEARTBEAT" and msg.type != mavutil.mavlink.MAV_TYPE_GCS:
            self.armed, self.mode = bool(msg.base_mode & ARMED_FLAG), msg.custom_mode
        elif kind == "PID_TUNING":
            self.log.pid.append((self.t_s, msg.axis, msg.desired, msg.achieved, msg.FF, msg.P, msg.I, msg.D))
        elif kind == "STATUSTEXT":
            self.log.texts.append((self.t_s, msg.text))
            logger.info("[SITL %.1fs] %s", self.t_s, msg.text)

    def _send_rc(self) -> None:
        self._conn.mav.rc_channels_override_send(self._conn.target_system, self._conn.target_component,
                                                 *self.rc, 0, 0, 0, 0)
        self._last_rc_send = time.monotonic()

    def set_rc(self, channel: int, pwm: int) -> float:
        """Change one stick (0-based channel), send it now, and return the sim time of the first attitude
        sample after the send - the moment the command reached the autopilot."""
        self.rc[channel] = pwm
        self._send_rc()
        sent_at: float = self.t_s
        self.wait(STEP_HOLD_S, lambda: self.t_s > sent_at, "telemetry after an RC change")
        return self.t_s

    def pump(self) -> None:
        if self._health is not None and (problem := self._health()) is not None:
            raise SitlFlightError(problem)
        msg = self._conn.recv_match(blocking=True, timeout=RECV_BLOCK_S)
        while msg is not None:
            self._handle(msg)
            msg = self._conn.recv_match(blocking=False)
        if time.monotonic() - self._last_rc_send >= RC_SEND_PERIOD_S:
            self._send_rc()

    def wait(self, sim_seconds: float, until: Optional[Callable[[], bool]] = None, what: str = "") -> bool:
        """Pump for sim_seconds or until `until()` holds; returns whether the condition was met. Raises if
        simulation time stops advancing for STALL_TIMEOUT_S wall seconds (a slow host is fine)."""
        start_sim: float = self.t_s
        last_t, last_progress = self.t_s, time.monotonic()
        while self.t_s - start_sim < sim_seconds:
            if until is not None and until():
                return True
            if self.t_s > last_t:
                last_t, last_progress = self.t_s, time.monotonic()
            elif time.monotonic() - last_progress > STALL_TIMEOUT_S:
                raise SitlFlightError(f"SITL stalled (no telemetry for {STALL_TIMEOUT_S:.0f} s) while waiting for "
                                      f"{what or 'telemetry'}")
            self.pump()
        return until is None

    def require(self, sim_seconds: float, until: Callable[[], bool], what: str) -> None:
        if not self.wait(sim_seconds, until, what):
            raise SitlFlightError(f"timed out after {sim_seconds:.0f} s (sim) waiting for {what}")

    def set_mode(self, name: str) -> None:
        mode_id: Optional[int] = (self._conn.mode_mapping() or {}).get(name)
        if mode_id is None:
            raise SitlFlightError(f"unknown flight mode {name}")
        self._conn.set_mode(mode_id)
        self.require(MODE_TIMEOUT_S, lambda: self.mode == mode_id, f"mode {name}")

    def arm(self, mode: str) -> None:
        """Arm in `mode`. The mode is re-asserted before every attempt because ArduPilot applies the RC
        flight-mode switch position once RC input becomes valid during boot."""
        deadline: float = self.t_s + ARM_TIMEOUT_S
        while not self.armed:
            if self.t_s > deadline:
                prearm: list[str] = [text for _, text in self.log.texts if "prearm" in text.lower()][-3:]
                raise SitlFlightError(f"arming refused: {prearm or 'no PreArm message'}")
            self.set_mode(mode)
            self._command(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1)
            self.wait(ARM_RETRY_S, lambda: self.armed, "arming")
        if self.mode != (self._conn.mode_mapping() or {}).get(mode):
            raise SitlFlightError(f"armed, but not in {mode} (mode {self.mode})")


def stick_pwm(fraction: float) -> int:
    """Centred stick (-1..1) -> PWM."""
    return int(round(PWM_TRIM + max(-1.0, min(1.0, fraction)) * (PWM_SPAN / 2)))


def throttle_pwm(fraction: float) -> int:
    return int(round(PWM_MIN + max(0.0, min(1.0, fraction)) * PWM_SPAN))


def cruise_stick_pwm(params: Mapping[str, float]) -> int:
    """FBWB maps the throttle stick linearly onto [AIRSPEED_MIN, AIRSPEED_MAX]."""
    lo, hi = params["AIRSPEED_MIN"], params["AIRSPEED_MAX"]
    return throttle_pwm((params["AIRSPEED_CRUISE"] - lo) / (hi - lo))


def _take_off_fixed_wing(pilot: SitlPilot, params: Mapping[str, float]) -> None:
    """Runway take-off in FBWA (SRS: "switch to AUTO / FBWA, execute takeoff"): full throttle, wings
    level, rotate to TAKEOFF_PITCH_DEG at TKOFF_ROTATE_SPD, climb to the SITL take-off altitude."""
    pilot.arm("FBWA")
    pilot.set_rc(2, throttle_pwm(1.0))
    rotate: float = params["TKOFF_ROTATE_SPD"]
    pilot.require(TAKEOFF_TIMEOUT_S, lambda: pilot.airspeed >= rotate, f"take-off roll to {rotate} m/s")
    pilot.set_rc(1, stick_pwm(PITCH_UP_PWM_SIGN * TAKEOFF_PITCH_DEG / params["PTCH_LIM_MAX_DEG"]))
    target: float = TAKEOFF_ALT_FRACTION * params["TKOFF_ALT"]
    pilot.require(TAKEOFF_TIMEOUT_S, lambda: pilot.alt_m >= target, f"climb to {target:.0f} m")
    pilot.set_rc(1, PWM_TRIM)


def _clear_of_the_ground(pilot: SitlPilot, target_alt_m: float) -> Callable[[], bool]:
    """Condition for the hand-launch climb: reaches the target altitude; raises if the aircraft comes down to the
    ground first. Altitude is relative to the release point, which is HAND_LAUNCH_HEIGHT_M above the ground."""
    def reached() -> bool:
        if pilot.alt_m <= GROUND_CONTACT_MARGIN_M - HAND_LAUNCH_HEIGHT_M:
            raise SitlFlightError(f"the hand-launched aircraft came down to the ground ({pilot.alt_m:.1f} m "
                                  f"relative to the release point) before reaching {target_alt_m:.0f} m")
        return pilot.alt_m >= target_alt_m
    return reached


def _take_off_hand_launched(pilot: SitlPilot, params: Mapping[str, float]) -> None:
    """Hand launch: the bridge holds the aircraft at the release pose until the throttle opens. The pilot arms in
    FBWA, holds the take-off pitch stick, then opens full throttle (the throw), and climbs without touching the
    ground to the SITL take-off altitude."""
    pilot.arm("FBWA")
    pilot.rc[1] = stick_pwm(PITCH_UP_PWM_SIGN * TAKEOFF_PITCH_DEG / params["PTCH_LIM_MAX_DEG"])
    pilot.set_rc(2, throttle_pwm(1.0))
    target: float = TAKEOFF_ALT_FRACTION * params["TKOFF_ALT"]
    pilot.require(TAKEOFF_TIMEOUT_S, _clear_of_the_ground(pilot, target), f"climb to {target:.0f} m after the throw")
    pilot.set_rc(1, PWM_TRIM)


def _take_off_quadplane(pilot: SitlPilot) -> None:
    pilot.arm("QHOVER")
    pilot.set_rc(2, QUAD_CLIMB_STICK_PWM)
    pilot.require(TAKEOFF_TIMEOUT_S, lambda: pilot.alt_m >= QUAD_CLIMB_ALT_M, f"VTOL climb to {QUAD_CLIMB_ALT_M} m")
    pilot.set_rc(2, PWM_TRIM)  # mid stick: hold altitude
    pilot.wait(PRE_STEP_HOLD_S)


def _transition_done(log: FlightLog) -> bool:
    return any(TRANSITION_DONE_TEXT in text.lower() for _, text in log.texts)


def _enter_cruise(pilot: SitlPilot, params: Mapping[str, float], is_quadplane: bool) -> None:
    pilot.set_rc(2, cruise_stick_pwm(params))
    log = pilot.log
    if is_quadplane:
        log.transition_start_t_s, log.transition_start_alt_m = pilot.t_s, pilot.alt_m
    pilot.set_mode("FBWB")
    if is_quadplane:
        pilot.require(TRANSITION_TIMEOUT_S, lambda: _transition_done(log), "the VTOL -> fixed-wing transition")
        pilot.wait(POST_TRANSITION_S)
        log.transition_end_t_s = pilot.t_s
    pilot.wait(CRUISE_SETTLE_S)


def pitch_stick_pwm(pitch_deg: float, params: Mapping[str, float]) -> int:
    """FBWA pitch demand -> RC2 PWM (positive stick scales PTCH_LIM_MAX_DEG, negative PTCH_LIM_MIN_DEG)."""
    limit: float = params["PTCH_LIM_MAX_DEG"] if pitch_deg >= 0 else -params["PTCH_LIM_MIN_DEG"]
    return stick_pwm(PITCH_UP_PWM_SIGN * pitch_deg / limit)


def _recent_mean(pilot: SitlPilot, attr: str) -> float:
    recent: list[float] = [getattr(s, attr) for s in pilot.log.samples if s.t_s >= pilot.t_s - THROTTLE_AVERAGE_S]
    return sum(recent) / len(recent) if recent else 0.0


def _hold_trim_in_fbwa(pilot: SitlPilot, params: Mapping[str, float]) -> None:
    pilot.log.trim_throttle_pct = _recent_mean(pilot, "throttle_pct")
    pilot.log.trim_pitch_deg = _recent_mean(pilot, "pitch_deg")
    pilot.rc[1] = pitch_stick_pwm(pilot.log.trim_pitch_deg, params)
    pilot.set_rc(2, throttle_pwm(pilot.log.trim_throttle_pct / 100.0))
    pilot.set_mode("FBWA")
    pilot.wait(PRE_STEP_HOLD_S)


def _doublets(pilot: SitlPilot, params: Mapping[str, float], climb_throttle_pct: float) -> None:
    log = pilot.log
    assert log.trim_pitch_deg is not None and log.trim_throttle_pct is not None
    log.roll_step_t_s = pilot.set_rc(0, stick_pwm(ROLL_STEP_DEG / params["ROLL_LIMIT_DEG"]))
    pilot.wait(STEP_HOLD_S)
    pilot.set_rc(0, PWM_TRIM)
    pilot.wait(RECOVER_S)
    pilot.rc[2] = throttle_pwm((log.trim_throttle_pct + climb_throttle_pct) / 100.0)
    log.pitch_step_t_s = pilot.set_rc(1, pitch_stick_pwm(log.trim_pitch_deg + PITCH_STEP_DEG, params))
    pilot.wait(STEP_HOLD_S)
    pilot.rc[2] = throttle_pwm(log.trim_throttle_pct / 100.0)
    pilot.set_rc(1, pitch_stick_pwm(log.trim_pitch_deg, params))
    pilot.wait(RECOVER_S)


def fly_acceptance_test(pilot: SitlPilot, params: Mapping[str, float], is_quadplane: bool,
                        climb_throttle_pct: float,
                        launch: Literal["runway", "hand", "vertical"] = "runway") -> FlightLog:
    """Fly the full sequence. `params` is the merged vehicle + SITL parameter set; climb_throttle_pct is
    the extra throttle (percent) the flight model needs to climb at +PITCH_STEP_DEG at cruise speed; `launch`
    picks the take-off (runway roll or hand launch; a quadplane always climbs vertically)."""
    pilot.connect()
    if is_quadplane:
        _take_off_quadplane(pilot)
    elif launch == "hand":
        _take_off_hand_launched(pilot, params)
    else:
        _take_off_fixed_wing(pilot, params)
    _enter_cruise(pilot, params, is_quadplane)
    _hold_trim_in_fbwa(pilot, params)
    _doublets(pilot, params, climb_throttle_pct)
    return pilot.log
