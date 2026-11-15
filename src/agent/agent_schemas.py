"""Schemas the LLM fills (structured outputs). Zero-LLM-math by construction:

* the parser returns each quantity with the unit the user used ({value: 3.5, unit: "kg"}); conversion
  to SI happens in deterministic code (src/agent/parsing.py);
* the critic picks *which* sizing knob to move and in *which direction* from an enumerated set; the
  step sizes and bounds are deterministic (src/agent/critic.py);
* the report writer only produces prose; every number in the report comes from the tools.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from src.schemas.components import SensorType


class Mass(BaseModel):
    value: float = Field(gt=0)
    unit: Literal["kg", "g", "lb", "oz"]


class Duration(BaseModel):
    value: float = Field(gt=0)
    unit: Literal["h", "min", "s", "day"]


class Distance(BaseModel):
    value: float = Field(gt=0)
    unit: Literal["km", "m", "mi", "nmi"]


class Speed(BaseModel):
    value: float = Field(gt=0)
    unit: Literal["m/s", "km/h", "kt", "mph"]


class Altitude(BaseModel):
    value: float = Field(gt=0)
    unit: Literal["m", "km", "ft"]


class Power(BaseModel):
    value: float = Field(ge=0)
    unit: Literal["W", "kW"]


class RequirementsDraft(BaseModel):
    """What the user asked for, exactly as stated. Leave a field null when the user did not state it."""

    mission_name: str = Field(description="Short descriptive name for the mission (3-8 words)")
    payload_mass: Optional[Mass] = Field(default=None, description="Mass of the sensor payload")
    payload_power: Optional[Power] = Field(default=None, description="Continuous electrical power of the payload")
    payload_sensor: Optional[SensorType] = Field(default=None, description="Sensor class, e.g. eo_ir for an EO/IR gimbal")
    endurance: Optional[Duration] = Field(default=None, description="Time on station / loiter endurance")
    radius: Optional[Distance] = Field(
        default=None, description="One-way distance to the operating area (combat, mission or datalink radius)")
    total_range: Optional[Distance] = Field(
        default=None, description="Only if the user gave a round-trip/total range instead of a radius")
    cruise_speed: Optional[Speed] = None
    service_ceiling: Optional[Altitude] = None
    max_stall_speed: Optional[Speed] = None
    runway_available: Optional[bool] = Field(
        default=None, description="false for runway-independent, VTOL or ship/field operations")
    vtol_required: Optional[bool] = None
    propulsion: Literal["electric", "series_hybrid", "auto"] = "auto"
    dod_group: Optional[Literal[1, 2, 3]] = Field(default=None, description="Only if the user named a DoD UAS group")
    ambiguities: list[str] = Field(default_factory=list, description="Anything unclear or contradictory")


Knob = Literal["aspect_ratio", "tail_volume_h", "tail_volume_v", "fuselage_length_factor", "reserve_fraction"]


class Adjustment(BaseModel):
    knob: Knob
    direction: Literal["increase", "decrease"]
    reason: str = Field(description="Which failed gate this addresses and why it helps")


class CriticDecision(BaseModel):
    """Sizing knobs to move for the next design iteration (empty if nothing can help)."""

    adjustments: list[Adjustment] = Field(default_factory=list)
    summary: str = Field(description="One or two sentences for the design log")


class ReportNarrative(BaseModel):
    executive_summary: str = Field(description="One paragraph: what was designed and whether it meets the mission")
    trade_offs: str = Field(description="Key design trade-offs and their consequences, as prose")
    risks_and_next_steps: str = Field(description="Open issues and recommended next steps, as prose")
