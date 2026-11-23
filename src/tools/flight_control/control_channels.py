"""One control channel per physical surface: the contract between ArduPilot, JSBSim and the SITL bridge.

ArduPilot does all the mixing (SERVOn_FUNCTION per surface); JSBSim models each surface's own deflection; the
bridge maps each servo straight to its surface. Nothing downstream ever un-mixes an output with an assumed
mix, so a wrong servo reversal cannot cancel against the plant and pass SITL.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Optional, Sequence

from src.schemas.state import ControlSurfaceDerivatives

if TYPE_CHECKING:
    from src.tools.flight_control.flight_model import AeroCoefficients

AILERON_MAX_RAD: Final[float] = math.radians(20.0)
ELEVATOR_MAX_RAD: Final[float] = math.radians(20.0)
RUDDER_MAX_RAD: Final[float] = math.radians(25.0)
RUDDERVATOR_MAX_RAD: Final[float] = math.radians(20.0)
THROTTLE_SERVO: Final[int] = 3
THROTTLE_SERVO_FUNCTION: Final[int] = 70
LIFT_MOTOR_SERVOS: Final[tuple[int, ...]] = (5, 6, 7, 8)  # outputs reserved for lift motors 1-4
STEERING_SURFACE: Final[str] = "steering"
STEERING_SERVO_FUNCTION: Final[int] = 26  # ground steering
RUDDER_SURFACE: Final[str] = "rudder"
ELEVATOR_SURFACE: Final[str] = "elevator"
RUDDERVATOR_SURFACES: Final[tuple[str, str]] = ("ruddervator_left", "ruddervator_right")
MAX_SERVO_OUTPUT: Final[int] = 16


@dataclass(frozen=True)
class SurfaceDerivatives:
    """Aerodynamic derivatives per rad of this surface's own deflection."""

    cl: float = 0.0  # lift
    cl_roll: float = 0.0  # rolling moment
    cm: float = 0.0
    cn: float = 0.0


@dataclass(frozen=True)
class ControlChannel:
    surface: str  # JSBSim name stem: "aileron", "elevator", "rudder", "ruddervator_left", ...
    servo: int  # ArduPilot output number (1-based)
    servo_function: int  # SERVOn_FUNCTION
    bridge_sign: float  # +1 / -1: PWM above trim -> positive deflection
    max_deflection_rad: float
    derivatives: SurfaceDerivatives
    steers_nose_wheel: bool = False  # the bridge drives the nose wheel from this channel's servo


@dataclass(frozen=True)
class _SurfaceSpec:
    servo: int
    servo_function: int
    bridge_sign: float
    max_deflection_rad: float


# Bridge signs: a positive ArduPilot output (nose-up elevator, nose-right rudder; a V-tail panel's mixed share of
# them) puts a non-reversed servo's PWM above trim, which should move the trailing edge up (elevator, ruddervators) or
# right (rudder). JSBSim's positive deflections are trailing edge down / left, so those bridge signs are -1. The
# V-tail's SERVOn_REVERSED is then chosen per panel from its measured pitch sign (param_generator.ruddervator_params).
SURFACE_SPECS: Final[dict[str, _SurfaceSpec]] = {
    "aileron": _SurfaceSpec(1, 4, 1.0, AILERON_MAX_RAD),
    ELEVATOR_SURFACE: _SurfaceSpec(2, 19, -1.0, ELEVATOR_MAX_RAD),
    RUDDER_SURFACE: _SurfaceSpec(4, 21, -1.0, RUDDER_MAX_RAD),
    RUDDERVATOR_SURFACES[0]: _SurfaceSpec(2, 79, -1.0, RUDDERVATOR_MAX_RAD),
    RUDDERVATOR_SURFACES[1]: _SurfaceSpec(4, 80, -1.0, RUDDERVATOR_MAX_RAD),
}


def _channel(record: ControlSurfaceDerivatives) -> ControlChannel:
    spec: Optional[_SurfaceSpec] = SURFACE_SPECS.get(record.name)
    if spec is None:
        raise ValueError(f"unknown control surface {record.name!r}; known: {sorted(SURFACE_SPECS)}")
    return ControlChannel(record.name, spec.servo, spec.servo_function, spec.bridge_sign, spec.max_deflection_rad,
                          SurfaceDerivatives(cl=record.cl_per_rad, cl_roll=record.cl_roll_per_rad,
                                             cm=record.cm_per_rad, cn=record.cn_per_rad),
                          steers_nose_wheel=record.name == RUDDER_SURFACE)


def _tractor_records(aero: AeroCoefficients) -> tuple[ControlSurfaceDerivatives, ...]:
    """The tractor triple from the aggregated derivatives (the path for records saved before per-surface data)."""
    return (
        ControlSurfaceDerivatives(name="aileron", cl_per_rad=0.0, cl_roll_per_rad=aero.cl_delta_a, cm_per_rad=0.0,
                                  cn_per_rad=aero.cn_delta_a),
        ControlSurfaceDerivatives(name="elevator", cl_per_rad=aero.cl_delta_e, cl_roll_per_rad=0.0,
                                  cm_per_rad=aero.cm_delta_e, cn_per_rad=0.0),
        ControlSurfaceDerivatives(name=RUDDER_SURFACE, cl_per_rad=0.0, cl_roll_per_rad=0.0, cm_per_rad=0.0,
                                  cn_per_rad=aero.cn_delta_r),
    )


def jsbsim_stem(surface: str) -> str:
    """JSBSim property-name stem of a surface: underscores become hyphens (ruddervator_left -> ruddervator-left).
    The exporter and the SITL bridge both build `fcs/<stem>-...` names from this so they always agree."""
    return surface.replace("_", "-")


def control_channels(aero: AeroCoefficients,
                     surfaces: Sequence[ControlSurfaceDerivatives] = ()) -> tuple[ControlChannel, ...]:
    """One channel per servo-driven surface, in order: from `surfaces` when given, else the tractor triple derived
    from the aggregated derivatives."""
    return tuple(_channel(r) for r in (surfaces or _tractor_records(aero)))


def steering_channel(channels: Sequence[ControlChannel], quad_motors: bool) -> ControlChannel:
    """The channel that steers the nose wheel: the rudder when there is one, otherwise a dedicated ground-steering
    channel on the first output not used by a surface, the throttle or (with lift motors) outputs 5-8. ValueError
    when every output is taken."""
    rudder = next((c for c in channels if c.surface == RUDDER_SURFACE), None)
    if rudder is not None:
        return replace(rudder, steers_nose_wheel=True)
    taken: set[int] = {c.servo for c in channels} | {THROTTLE_SERVO}
    if quad_motors:
        taken |= set(LIFT_MOTOR_SERVOS)
    servo = next((n for n in range(1, MAX_SERVO_OUTPUT + 1) if n not in taken), None)
    if servo is None:
        raise ValueError(f"no free servo output for ground steering (outputs 1-{MAX_SERVO_OUTPUT} are all in use)")
    return ControlChannel(STEERING_SURFACE, servo, STEERING_SERVO_FUNCTION, 1.0, 0.0, SurfaceDerivatives(),
                          steers_nose_wheel=True)


def channels_with_steering(channels: tuple[ControlChannel, ...], quad_motors: bool) -> tuple[ControlChannel, ...]:
    """`channels` plus the dedicated steering channel when no surface steers the nose wheel."""
    if any(c.steers_nose_wheel for c in channels):
        return channels
    return (*channels, steering_channel(channels, quad_motors))
