"""Propulsion operating points produced by the propulsor model (spec 6.4)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PropulsorRole = Literal["cruise", "lift"]


class OperatingPoint(BaseModel):
    """One flight point solved for one propulsor. `prop_efficiency` is T V / P_shaft in forward flight and the
    figure of merit in hover (airspeed 0), times the installation factor in both. `sagged` echoes whether the
    bus voltage the point was solved at is the end-of-reserve loaded voltage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    flight_point: str = Field(..., min_length=1, max_length=60)
    propulsor: PropulsorRole
    rpm: float = Field(..., gt=0)
    current_a: float = Field(..., gt=0)
    throttle: float = Field(..., gt=0, le=1.0)  # motor voltage / bus voltage (ESC duty)
    prop_efficiency: float = Field(..., gt=0)
    motor_efficiency: float = Field(..., gt=0, le=1.0)
    tip_mach: float = Field(..., gt=0)
    electrical_power_w: float = Field(..., gt=0)  # drawn from the bus, ESC loss included
    sagged: bool
