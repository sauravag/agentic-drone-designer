"""RequirementsDraft (what the user said, with units) -> validated SI OperationalRequirements.

Deterministic I/O boundary (SRS rule 3): unit conversion, documented defaults for unstated values
(each recorded as an assumption for the report), and the SRS clarification rule - payload mass and
endurance must come from the user.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Optional

from pydantic import ValidationError

from src.agent.agent_schemas import Altitude, Distance, Duration, Mass, Power, RequirementsDraft, Speed
from src.schemas.requirements import OperationalRequirements
from src.tools.database.matcher import ComponentNotFoundError, match_payload
from src.units import (DAY_TO_S, FT_TO_M, G_TO_KG, HOUR_TO_S, KM_TO_M, KMH_TO_MS, KNOT_TO_MS, KW_TO_W, LB_TO_KG,
                       MILE_TO_M, MIN_TO_S, MPH_TO_MS, NMI_TO_M, OZ_TO_KG)

DEFAULT_CRUISE_SPEED_MS: Final[float] = 22.0  # typical Group 1-3 ISR cruise
DEFAULT_RADIUS_KM: Final[float] = 50.0
FALLBACK_PAYLOAD_POWER_W: Final[float] = 50.0  # if the catalog has no matching sensor
ROUND_TRIP_LEGS: Final[float] = 2.0
ECHO_FIELDS: Final[tuple[str, ...]] = ("payload_mass", "payload_power", "endurance", "radius", "total_range",
                                        "cruise_speed", "service_ceiling", "max_stall_speed")
REQUIRED_FIELDS: Final[dict[str, str]] = {"payload_mass": "payload mass", "endurance": "endurance (time on station)"}

_MASS: Final[dict[str, float]] = {"kg": 1.0, "g": G_TO_KG, "lb": LB_TO_KG, "oz": OZ_TO_KG}
_DURATION_H: Final[dict[str, float]] = {"h": 1.0, "min": MIN_TO_S / HOUR_TO_S, "s": 1.0 / HOUR_TO_S,
                                         "day": DAY_TO_S / HOUR_TO_S}
_DISTANCE_KM: Final[dict[str, float]] = {"km": 1.0, "m": 1.0 / KM_TO_M, "mi": MILE_TO_M / KM_TO_M,
                                          "nmi": NMI_TO_M / KM_TO_M}
_SPEED: Final[dict[str, float]] = {"m/s": 1.0, "km/h": KMH_TO_MS, "kt": KNOT_TO_MS, "mph": MPH_TO_MS}
_ALTITUDE: Final[dict[str, float]] = {"m": 1.0, "km": KM_TO_M, "ft": FT_TO_M}
_POWER: Final[dict[str, float]] = {"W": 1.0, "kW": KW_TO_W}


@dataclass(frozen=True)
class ParseOutcome:
    requirements: Optional[OperationalRequirements]
    clarification: Optional[str]
    assumptions: tuple[str, ...] = ()


def kg(q: Mass) -> float:
    return q.value * _MASS[q.unit]


def hours(q: Duration) -> float:
    return q.value * _DURATION_H[q.unit]


def km(q: Distance) -> float:
    return q.value * _DISTANCE_KM[q.unit]


def ms(q: Speed) -> float:
    return q.value * _SPEED[q.unit]


def metres(q: Altitude) -> float:
    return q.value * _ALTITUDE[q.unit]


def watts(q: Power) -> float:
    return q.value * _POWER[q.unit]


def _radius_km(draft: RequirementsDraft, notes: list[str]) -> float:
    if draft.radius is not None:
        return km(draft.radius)
    if draft.total_range is not None:
        notes.append(f"Mission radius taken as half the stated total range ({km(draft.total_range):.0f} km).")
        return km(draft.total_range) / ROUND_TRIP_LEGS
    notes.append(f"Mission radius not stated: assumed {DEFAULT_RADIUS_KM:.0f} km.")
    return DEFAULT_RADIUS_KM


def _payload_power_w(draft: RequirementsDraft, payload_kg: float, db_path: Path, notes: list[str]) -> float:
    if draft.payload_power is not None:
        return watts(draft.payload_power)
    try:
        part = match_payload(db_path, payload_kg, draft.payload_sensor)
    except ComponentNotFoundError:
        notes.append(f"Payload power not stated and no catalog match: assumed {FALLBACK_PAYLOAD_POWER_W:.0f} W.")
        return FALLBACK_PAYLOAD_POWER_W
    notes.append(f"Payload power not stated: {part.power_w:.0f} W from the catalog sensor '{part.name}'.")
    return part.power_w


def _speed_ms(draft: RequirementsDraft, notes: list[str]) -> float:
    if draft.cruise_speed is not None:
        return ms(draft.cruise_speed)
    notes.append(f"Cruise speed not stated: assumed {DEFAULT_CRUISE_SPEED_MS:.0f} m/s.")
    return DEFAULT_CRUISE_SPEED_MS


def _airframe_fields(draft: RequirementsDraft, notes: list[str]) -> dict[str, object]:
    if draft.vtol_required:
        return {"airframe_topology": "quadplane_vtol", "runway_available": False}
    if draft.vtol_required is False and draft.runway_available is None:
        notes.append("VTOL not required and runway availability not stated: assumed a runway (fixed-wing).")
        return {"airframe_topology": "fixed_wing", "runway_available": True}
    if draft.runway_available is None:
        notes.append("Runway availability not stated: assumed runway-independent (VTOL quadplane).")
        return {"runway_available": False}
    return {"runway_available": draft.runway_available}


def _optional_fields(draft: RequirementsDraft) -> dict[str, float]:
    fields: dict[str, float] = {}
    if draft.service_ceiling is not None:
        fields["service_ceiling_m"] = metres(draft.service_ceiling)
    if draft.max_stall_speed is not None:
        fields["max_stall_speed_ms"] = ms(draft.max_stall_speed)
    return fields


def to_requirements(draft: RequirementsDraft, db_path: Path) -> ParseOutcome:
    """Convert a draft; returns a clarification question instead if the SRS-required facts are missing
    or the stated values are outside the supported envelope."""
    if draft.payload_mass is None or draft.endurance is None:
        missing: list[str] = [label for name, label in REQUIRED_FIELDS.items() if getattr(draft, name) is None]
        return ParseOutcome(None, f"Please specify the {' and '.join(missing)} for this mission.")
    notes: list[str] = []
    payload_kg: float = kg(draft.payload_mass)
    fields: dict[str, object] = {
        "mission_name": draft.mission_name, "payload_mass_kg": payload_kg, "endurance_hours": hours(draft.endurance),
        "payload_power_w": _payload_power_w(draft, payload_kg, db_path, notes),
        "combat_radius_km": _radius_km(draft, notes), "cruise_speed_ms": _speed_ms(draft, notes),
        "propulsion_type": draft.propulsion, "group_constraint": draft.dod_group,
        **_airframe_fields(draft, notes), **_optional_fields(draft),
    }
    try:
        req = OperationalRequirements(**fields)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        return ParseOutcome(None, f"Some requirements are outside the supported envelope ({problems}). "
                                  "Please revise them.")
    notes.insert(0, _parsed_echo(draft))
    notes += [f"Ambiguity noted by the parser: {text}" for text in draft.ambiguities]
    return ParseOutcome(req, None, tuple(notes))


def _parsed_echo(draft: RequirementsDraft) -> str:
    """What the LLM read from the request, in the user's own units, so a mis-read is visible."""
    stated: list[str] = [f"{name.replace('_', ' ')} {q.value:g} {q.unit}" for name in ECHO_FIELDS
                         if (q := getattr(draft, name)) is not None]
    return "Read from the request: " + (", ".join(stated) or "no quantities") + "."
