"""Schemas the LLM fills (structured outputs). Zero-LLM-math by construction:

* the parser returns each quantity with the unit the user used ({value: 3.5, unit: "kg"}); conversion
  to SI happens in deterministic code (src/agent/parsing.py);
* the critic picks *which* sizing knob to move and in *which direction* from an enumerated set; the
  step sizes and bounds are deterministic (src/agent/critic.py);
* the report writer only produces prose; every number in the report comes from the tools;
* the design chat proposes field changes (a flat allow-list) for the solver to check; it never states a
  result itself.
"""

from typing import Final, Literal, Optional

from pydantic import BaseModel, Field

from src.schemas.changes import AssumptionChanges, KnobChanges, RequirementChanges
from src.schemas.components import SensorType
from src.schemas.requirements import FlightControllerId

MAX_WHAT_IFS: Final[int] = 4  # per design-chat round; enforced after parsing (see DesignerTurn)
# every field a designer change set may touch, generated from the change models so the two cannot drift
CHANGE_FIELDS: Final[tuple[str, ...]] = (*RequirementChanges.model_fields, *AssumptionChanges.model_fields,
                                         *KnobChanges.model_fields)
ChangeField = Literal[CHANGE_FIELDS]  # type: ignore[valid-type]  # Literal["propulsion_type", ...]


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
    payload_model: Optional[str] = Field(
        default=None, description="The payload's model name if the user names a specific product "
                                  "(e.g. 'SIYI ZT30', 'NextVision Raptor')")
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
    flight_controller: Optional[FlightControllerId] = Field(
        default=None, description="ark_jetson_pab if the user names ARK / PAB / a Jetson carrier with autopilot; "
                                  "cube_orange_plus if they name a Cube; otherwise null")
    companion_computer: Optional[bool] = Field(
        default=None, description="true for autonomous / onboard AI / a named Jetson or companion computer; "
                                  "false only if the user explicitly excludes one; otherwise null")
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


class OptionsExplanation(BaseModel):
    why_it_failed: str = Field(description="Why the requested design does not close, in plain words")
    trade_offs: str = Field(description="How the listed options compare and what each costs the designer")


class FieldChange(BaseModel):
    field: ChangeField  # type: ignore[valid-type]
    value: str = Field(description="New value as text, e.g. \"4.5\", \"true\", \"series_hybrid\" or \"2\"")


# A change set as the LLM writes it: flat, so the strict schema has no nullable parameters. The app converts it
# to a `ChangeSet` (src/agent/design_chat.py), which validates types and bounds. (Docstrings of these models
# become schema descriptions the LLM reads, so implementation notes live in comments.)
class ProposedChange(BaseModel):
    """A design change: the fields to alter and their new values."""

    label: str = Field(description="Short name for the change, e.g. \"Endurance 4.5 h\"")
    changes: list[FieldChange] = Field(description="Each field to change, at most once")


# The what-if limit is enforced by the app after parsing, not by the JSON schema (`maxItems` is not supported by
# every provider's strict structured outputs). `proposal` is the schema's only nullable parameter (Anthropic
# limits optional/union parameters in strict schemas).
class DesignerTurn(BaseModel):
    """One round of the design chat: a reply, what-ifs to check first, and an optional proposal."""

    message: str = Field(description="Reply to the designer; cite only numbers given in the briefing or results")
    what_ifs: list[ProposedChange] = Field(
        description=f"Changes to check with the solver before answering (at most {MAX_WHAT_IFS})")
    proposal: Optional[ProposedChange] = Field(description="A change you recommend the designer accepts, or null")
    done: bool = Field(description="true when ready to answer without further checks")
