"""One control channel per physical surface: the contract between ArduPilot, JSBSim and the SITL bridge.

ArduPilot does all the mixing (SERVOn_FUNCTION per surface); JSBSim models each surface's own deflection; the
bridge maps each servo straight to its surface. Nothing downstream ever un-mixes an output with an assumed
mix, so a wrong servo reversal cannot cancel against the plant and pass SITL.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from src.tools.flight_control.flight_model import AeroCoefficients

AILERON_MAX_RAD: Final[float] = math.radians(20.0)
ELEVATOR_MAX_RAD: Final[float] = math.radians(20.0)
RUDDER_MAX_RAD: Final[float] = math.radians(25.0)
THROTTLE_SERVO: Final[int] = 3
THROTTLE_SERVO_FUNCTION: Final[int] = 70


@dataclass(frozen=True)
class SurfaceDerivatives:
    """Aerodynamic derivatives per rad of this surface's own deflection."""

    cl: float = 0.0  # lift
    cl_roll: float = 0.0  # rolling moment
    cm: float = 0.0
    cn: float = 0.0


@dataclass(frozen=True)
class ControlChannel:
    surface: str  # JSBSim name stem: "aileron", "elevator", "rudder"
    servo: int  # ArduPilot output number (1-based)
    servo_function: int  # SERVOn_FUNCTION
    bridge_sign: float  # +1 / -1: PWM above trim -> positive deflection
    max_deflection_rad: float
    derivatives: SurfaceDerivatives


def control_channels(aero: AeroCoefficients) -> tuple[ControlChannel, ...]:
    """The tractor airframe's surfaces: aileron, elevator, rudder."""
    return (
        ControlChannel("aileron", 1, 4, 1.0, AILERON_MAX_RAD,
                       SurfaceDerivatives(cl_roll=aero.cl_delta_a, cn=aero.cn_delta_a)),
        ControlChannel("elevator", 2, 19, -1.0, ELEVATOR_MAX_RAD,
                       SurfaceDerivatives(cl=aero.cl_delta_e, cm=aero.cm_delta_e)),
        ControlChannel("rudder", 4, 21, -1.0, RUDDER_MAX_RAD, SurfaceDerivatives(cn=aero.cn_delta_r)),
    )
