"""Propulsion operating points produced by the propulsor model (spec 6.4)."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

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


SetKey = tuple[int, str, str, Optional[str], Optional[str]]


class PropulsionMatch(BaseModel):
    """The matched propulsion set (spec 6.4) with its operating points. `efficiencies` per flight point name:
    forward phases T V / P_electrical, hover phases ideal induced power / P_electrical (what sizing consumes).
    `propulsion_mass_kg` is the hardware only: cruise motor + propeller + ESC, plus lift motor + rotor + ESC per
    lift rotor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cell_name: str = Field(..., min_length=1, max_length=120)
    series: int = Field(..., ge=1, le=1_000)
    cruise_motor: str = Field(..., min_length=1, max_length=120)
    cruise_propeller: str = Field(..., min_length=1, max_length=120)
    cruise_esc: str = Field(..., min_length=1, max_length=120)
    lift_motor: Optional[str] = Field(None, min_length=1, max_length=120)
    lift_rotor: Optional[str] = Field(None, min_length=1, max_length=120)
    lift_esc: Optional[str] = Field(None, min_length=1, max_length=120)
    operating_points: tuple[OperatingPoint, ...] = Field(..., min_length=1)
    efficiencies: Mapping[str, float]  # read-only view (a frozen model must not hand out a mutable dict)
    propulsion_mass_kg: float = Field(..., gt=0)
    notes: tuple[str, ...] = ()

    @field_validator("efficiencies", mode="after")
    @classmethod
    def _read_only(cls, value: Mapping[str, float]) -> Mapping[str, float]:
        return MappingProxyType(dict(value))

    @field_serializer("efficiencies")
    def _plain(self, value: Mapping[str, float]) -> dict[str, float]:
        return dict(value)

    @property
    def set_key(self) -> SetKey:
        """What "the set" means for the outer sizing loop's convergence: (S, cruise motor winding, cruise
        propeller, lift motor winding, lift rotor)."""
        return self.series, self.cruise_motor, self.cruise_propeller, self.lift_motor, self.lift_rotor
