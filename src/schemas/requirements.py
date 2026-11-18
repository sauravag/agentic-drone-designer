"""Operational requirements: the validated contract between the LLM parser and the physics tools."""

from typing import Any, Final, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.units import HOUR_TO_S, KM_TO_M

FlightControllerId = Literal["cube_orange_plus", "ark_jetson_pab"]
CUBE_ORANGE_PLUS: Final[str] = "cube_orange_plus"
ARK_JETSON_PAB: Final[str] = "ark_jetson_pab"  # carries the Jetson Orin NX on the same board stack


class OperationalRequirements(BaseModel):
    """User mission requirements. Field units follow the SRS (hours, km) at this I/O boundary;
    use the `*_s` / `*_m` properties inside physics code."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mission_name: str = Field(..., min_length=1, max_length=200)
    group_constraint: Optional[Literal[1, 2, 3]] = None  # auto-assigned if None
    propulsion_type: Literal["electric", "series_hybrid", "auto"] = "auto"
    airframe_topology: Literal["fixed_wing", "quadplane_vtol", "auto"] = "auto"
    payload_mass_kg: float = Field(..., gt=0, le=200.0)
    payload_power_w: float = Field(..., ge=0, le=5000.0)  # continuous hotel load
    endurance_hours: float = Field(..., gt=0.1, le=48.0)
    combat_radius_km: float = Field(..., gt=1.0, le=1000.0)
    cruise_speed_ms: float = Field(..., gt=8.0, le=120.0)
    service_ceiling_m: float = Field(default=3000.0, gt=100.0, le=10000.0)
    max_stall_speed_ms: float = Field(default=16.0, gt=5.0, le=45.0)
    runway_available: bool = False
    # Defaults (Cube, no Jetson) keep requirements written before the avionics choice existed unchanged.
    flight_controller: FlightControllerId = CUBE_ORANGE_PLUS
    companion_computer: bool = False  # separate Jetson on its own carrier; only meaningful with the Cube

    @model_validator(mode="before")
    @classmethod
    def _ark_carries_the_jetson(cls, data: Any) -> Any:
        """The ARK stack already holds the Jetson, so a separate companion is never also counted."""
        if isinstance(data, dict) and data.get("flight_controller") == ARK_JETSON_PAB:
            return {**data, "companion_computer": False}
        return data

    @property
    def jetson_fitted(self) -> bool:
        return self.flight_controller == ARK_JETSON_PAB or self.companion_computer

    @property
    def endurance_s(self) -> float:
        return self.endurance_hours * HOUR_TO_S

    @property
    def combat_radius_m(self) -> float:
        return self.combat_radius_km * KM_TO_M
