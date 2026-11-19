"""Operational requirements: the validated contract between the LLM parser and the physics tools."""

from typing import Any, Final, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.units import HOUR_TO_S, KM_TO_M

FlightControllerId = Literal["cube_orange_plus", "ark_jetson_pab"]
CUBE_ORANGE_PLUS: Final[str] = "cube_orange_plus"
ARK_JETSON_PAB: Final[str] = "ark_jetson_pab"  # carries the Jetson Orin NX on the same board stack
# Defined here, not in payload_catalog, because components -> requirements; payload_catalog re-exports it.
CUSTOM_PAYLOAD: Final[str] = "custom"
PAYLOAD_EDIT_TOLERANCE: Final[float] = 1e-6  # kg and W: typed back to the datasheet value counts as unedited


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
    # A catalogue payload name, or Custom. payload_mass_kg / payload_power_w stay the values sized with: a model
    # pre-fills them from its datasheet, and the designer may edit either (payload_edited).
    payload_model: str = CUSTOM_PAYLOAD

    @field_validator("payload_model")
    @classmethod
    def _known_payload(cls, value: str) -> str:
        from src.schemas.payload_catalog import payload_spec  # deferred: payload_catalog -> components -> here

        if value != CUSTOM_PAYLOAD and payload_spec(value) is None:
            raise ValueError(f"unknown payload model {value!r}: choose a catalogue payload or {CUSTOM_PAYLOAD!r}")
        return value

    @property
    def payload_edited(self) -> bool:
        """A catalogue model whose mass or power was changed from its datasheet values."""
        from src.schemas.payload_catalog import payload_edited  # deferred: payload_catalog -> components -> here

        return payload_edited(self.payload_model, self.payload_mass_kg, self.payload_power_w)

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

    def to_context_json(self) -> dict[str, Any]:
        """JSON for LLM context (report facts, design chat): adds `jetson_fitted`, since `companion_computer` alone
        reads as "no Jetson" on an ARK design. Not for re-validation (`extra="forbid"` rejects the added key)."""
        return {**self.model_dump(mode="json"), "jetson_fitted": self.jetson_fitted}

    @property
    def endurance_s(self) -> float:
        return self.endurance_hours * HOUR_TO_S

    @property
    def combat_radius_m(self) -> float:
        return self.combat_radius_km * KM_TO_M
