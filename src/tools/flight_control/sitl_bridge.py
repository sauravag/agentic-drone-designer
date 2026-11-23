"""Lock-step bridge between `arduplane --model JSON` and the exported JSBSim aircraft.

ArduPilot's built-in JSBSim backend only drives aileron/elevator/rudder/one throttle, so it cannot fly a
quadplane. The JSON SITL interface instead sends every servo output each physics frame and waits for
the vehicle state, so this bridge can route lift motors too. Each received frame advances JSBSim by
exactly 1/frame_rate, which makes runs reproducible regardless of host speed.

Wire format (ArduPilot libraries/SITL/SIM_JSON): servo packet little-endian
<uint16 magic=18458, uint16 frame_rate, uint32 frame_count, uint16 pwm[16]> (magic 29569 for 32
channels); reply: one JSON line with timestamp, imu.gyro (body rad/s), imu.accel_body (specific force,
m/s^2), position (NED m from home), attitude (rad), velocity (NED m/s) and airspeed (m/s).

Servo mapping (ArduPilot outputs, PWM > trim = roll right / pitch up / yaw right):
SERVO1 aileron, SERVO2 elevator, SERVO3 cruise throttle, SERVO4 rudder + nose-wheel steering,
SERVO5-8 lift motors 1-4 (a layout with no rudder steers the nose wheel from the first free output).
"""

import json
import logging
import math
import socket
import struct
import threading
from dataclasses import dataclass, field
from typing import Final, MutableMapping, Optional, Sequence

import jsbsim

from src.tools.flight_control.control_channels import THROTTLE_SERVO, ControlChannel, jsbsim_stem
from src.tools.flight_control.jsbsim_exporter import JSBSimModel

logger = logging.getLogger(__name__)

MAGIC_16: Final[int] = 18458
MAGIC_32: Final[int] = 29569
HEADER: Final[struct.Struct] = struct.Struct("<HHI")
PWM_16: Final[struct.Struct] = struct.Struct("<HHI16H")
PWM_32: Final[struct.Struct] = struct.Struct("<HHI32H")
PWM_TRIM: Final[float] = 1500.0
PWM_HALF_RANGE: Final[float] = 500.0
PWM_MIN: Final[float] = 1000.0
PWM_RANGE: Final[float] = 1000.0
FT_TO_M: Final[float] = 0.3048
EARTH_RADIUS_M: Final[float] = 6_378_137.0
RECV_TIMEOUT_S: Final[float] = 0.5
INITIAL_GROUND_GAP_M: Final[float] = 0.01
LIFT_MOTOR_FIRST_CHANNEL: Final[int] = 4  # zero-based index of SERVO5
MAX_LOGGED_ERRORS: Final[int] = 20
TRUTH_RATE_HZ: Final[float] = 25.0  # ground-truth log rate (JSBSim state, for diagnosis and the report)
TRUTH_PROPS: Final[tuple[tuple[str, str, float], ...]] = (  # (key, JSBSim property, scale to SI)
    ("airspeed_ms", "velocities/vt-fps", FT_TO_M), ("alpha_deg", "aero/alpha-deg", 1.0),
    ("pitch_deg", "attitude/theta-deg", 1.0), ("roll_deg", "attitude/phi-deg", 1.0),
    ("agl_m", "position/h-agl-ft", FT_TO_M), ("elevator_rad", "fcs/elevator-pos-rad", 1.0),
    ("aileron_rad", "fcs/aileron-pos-rad", 1.0), ("throttle", "fcs/cruise-throttle", 1.0),
    ("heading_deg", "attitude/psi-deg", 1.0),
)
TRUTH_POSITION_KEYS: Final[tuple[str, str]] = ("north_m", "east_m")  # metres from the start point


def north_east_m(lat_rad: float, lon_rad: float, origin_lat_rad: float, origin_lon_rad: float) -> tuple[float, float]:
    """Flat-earth north/east offset (m) of a position from the origin."""
    north: float = (lat_rad - origin_lat_rad) * EARTH_RADIUS_M
    east: float = (lon_rad - origin_lon_rad) * EARTH_RADIUS_M * math.cos(origin_lat_rad)
    return north, east


@dataclass(frozen=True)
class Home:
    lat_deg: float = -35.36  # ArduPilot autotest site (CMAC)
    lon_deg: float = 149.16
    alt_m: float = 0.0
    heading_deg: float = 0.0

    def arg(self) -> str:
        return f"{self.lat_deg},{self.lon_deg},{self.alt_m},{self.heading_deg}"


@dataclass
class BridgeStats:
    frames: int = 0
    frame_rate: int = 0
    resets: int = 0
    errors: list[str] = field(default_factory=list)
    fatal: Optional[str] = None  # set when the physics loop stops for good
    truth: list[dict[str, float]] = field(default_factory=list)


def _centred(pwm: int) -> float:
    return min(max((pwm - PWM_TRIM) / PWM_HALF_RANGE, -1.0), 1.0) if pwm else 0.0


def _apply_channels(fdm: MutableMapping[str, float], channels: Sequence[ControlChannel],
                    pwm: Sequence[float]) -> None:
    """Each servo drives its own surface; ArduPilot has already mixed, so nothing is un-mixed here."""
    for c in channels:
        fdm[f"fcs/{jsbsim_stem(c.surface)}-cmd-norm"] = c.bridge_sign * _centred(pwm[c.servo - 1])
        if c.steers_nose_wheel:  # the nose wheel follows the rudder servo, or the dedicated steering servo
            fdm["fcs/steer-cmd-norm"] = _centred(pwm[c.servo - 1])


def _unit(pwm: int) -> float:
    return min(max((pwm - PWM_MIN) / PWM_RANGE, 0.0), 1.0) if pwm else 0.0


def decode_servo_packet(data: bytes) -> tuple[int, int, tuple[int, ...]]:
    """(frame_rate, frame_count, pwm) from an ArduPilot JSON servo packet; ValueError if malformed."""
    if len(data) < HEADER.size:
        raise ValueError(f"short servo packet ({len(data)} bytes)")
    magic: int = HEADER.unpack_from(data)[0]
    layout: Optional[struct.Struct] = {MAGIC_16: PWM_16, MAGIC_32: PWM_32}.get(magic)
    if layout is None or len(data) < layout.size:
        raise ValueError(f"bad servo packet: magic={magic} length={len(data)}")
    _, rate, count, *pwm = layout.unpack_from(data)
    if rate <= 0:
        raise ValueError("servo packet with zero frame rate")
    return rate, count, tuple(pwm)


class JsonPhysicsBridge:
    """Runs JSBSim as ArduPilot's physics backend on a background thread."""

    def __init__(self, model: JSBSimModel, home: Home, port: int, host: str = "127.0.0.1") -> None:
        self._model, self._home = model, home
        self._address = (host, port)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None
        self._fdm: Optional[jsbsim.FGFDMExec] = None
        self._origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._last_count: int = -1
        self._last_reply: bytes = b""
        self.stats = BridgeStats()

    def __enter__(self) -> "JsonPhysicsBridge":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def start(self) -> None:
        self._fdm = self._load_fdm()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind(self._address)  # OSError if another run holds the port
        except OSError:
            sock.close()
            raise
        sock.settimeout(RECV_TIMEOUT_S)
        self._sock = sock
        self._thread = threading.Thread(target=self._serve, name="jsbsim-bridge", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._sock is not None:
            self._sock.close()

    def _load_fdm(self) -> jsbsim.FGFDMExec:
        fdm = jsbsim.FGFDMExec(str(self._model.root_dir), None)
        fdm.set_debug_level(0)
        if not fdm.load_model(self._model.name):
            raise RuntimeError(f"JSBSim could not load {self._model.xml_path}")
        self._initialise(fdm)
        return fdm

    def _initialise(self, fdm: jsbsim.FGFDMExec) -> None:
        h = self._home
        fdm["ic/lat-geod-deg"], fdm["ic/long-gc-deg"] = h.lat_deg, h.lon_deg
        fdm["ic/terrain-elevation-ft"] = h.alt_m / FT_TO_M
        fdm["ic/h-agl-ft"] = (self._model.rest_height_m + INITIAL_GROUND_GAP_M) / FT_TO_M
        fdm["ic/psi-true-deg"] = h.heading_deg
        for axis in "uvw":
            fdm[f"ic/{axis}-fps"] = 0.0
        if not fdm.run_ic():
            raise RuntimeError("JSBSim rejected the initial conditions")
        self._origin = (fdm["position/lat-geod-rad"], fdm["position/long-gc-rad"], fdm["position/h-sl-meters"])

    def _serve(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                data, sender = self._sock.recvfrom(256)
                reply: bytes = self._handle(data)
                self._sock.sendto(reply, sender)
            except socket.timeout:
                continue
            except ValueError as exc:  # one malformed packet: skip it (log a bounded number)
                if len(self.stats.errors) < MAX_LOGGED_ERRORS:
                    self.stats.errors.append(str(exc))
                    logger.warning("JSON bridge: %s", exc)
            except Exception as exc:  # JSBSim/socket failure: stop and let the pilot report it
                self.stats.fatal = f"{type(exc).__name__}: {exc}"
                logger.error("JSON bridge stopped: %s", self.stats.fatal)
                return

    def _handle(self, data: bytes) -> bytes:
        rate, count, pwm = decode_servo_packet(data)
        fdm = self._fdm
        assert fdm is not None
        if count == self._last_count and self._last_reply:
            return self._last_reply  # ArduPilot resent the frame: do not advance physics
        if count < self._last_count:  # ArduPilot restarted: start the flight over
            self.stats.resets += 1
            self._initialise(fdm)
        if rate != self.stats.frame_rate:
            fdm.set_dt(1.0 / rate)
            self.stats.frame_rate = rate
        self._apply(pwm)
        fdm.run()
        self._last_count, self.stats.frames = count, self.stats.frames + 1
        if self.stats.frames % max(1, round(rate / TRUTH_RATE_HZ)) == 0:
            self._record_truth(fdm)
        self._last_reply = ("\n" + json.dumps(self._state()) + "\n").encode()
        return self._last_reply

    def _record_truth(self, fdm: jsbsim.FGFDMExec) -> None:
        sample: dict[str, float] = {"t_s": fdm.get_sim_time()}
        sample |= {key: fdm[prop] * scale for key, prop, scale in TRUTH_PROPS}
        lat0, lon0, _ = self._origin
        sample |= dict(zip(TRUTH_POSITION_KEYS, north_east_m(
            fdm["position/lat-geod-rad"], fdm["position/long-gc-rad"], lat0, lon0)))
        self.stats.truth.append(sample)

    def _apply(self, pwm: tuple[int, ...]) -> None:
        fdm = self._fdm
        assert fdm is not None
        _apply_channels(fdm, self._model.channels, pwm)
        fdm["fcs/cruise-throttle-cmd"] = _unit(pwm[THROTTLE_SERVO - 1])
        for motor in self._model.lift_motor_numbers:
            fdm[f"fcs/lift{motor}-cmd"] = _unit(pwm[LIFT_MOTOR_FIRST_CHANNEL + motor - 1])

    def _state(self) -> dict[str, object]:
        fdm = self._fdm
        assert fdm is not None
        lat0, lon0, h0 = self._origin
        mass_slug: float = fdm["inertia/mass-slugs"]
        north, east = north_east_m(fdm["position/lat-geod-rad"], fdm["position/long-gc-rad"], lat0, lon0)
        return {
            "timestamp": fdm.get_sim_time(),
            "imu": {"gyro": [fdm["velocities/p-rad_sec"], fdm["velocities/q-rad_sec"], fdm["velocities/r-rad_sec"]],
                    "accel_body": [fdm[f"forces/fb{a}-total-lbs"] / mass_slug * FT_TO_M for a in "xyz"]},
            "position": [north, east, h0 - fdm["position/h-sl-meters"]],
            "attitude": [fdm["attitude/phi-rad"], fdm["attitude/theta-rad"], fdm["attitude/psi-rad"]],
            "velocity": [fdm[f"velocities/v-{axis}-fps"] * FT_TO_M for axis in ("north", "east", "down")],
            "airspeed": fdm["velocities/vt-fps"] * FT_TO_M,
        }
