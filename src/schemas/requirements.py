"""Operational requirements: the validated contract between the LLM parser and the physics tools."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.units import HOUR_TO_S, KM_TO_M


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

    @property
    def endurance_s(self) -> float:
        return self.endurance_hours * HOUR_TO_S

    @property
    def combat_radius_m(self) -> float:
        return self.combat_radius_km * KM_TO_M
