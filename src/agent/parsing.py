"""RequirementsDraft (what the user said, with units) -> validated SI OperationalRequirements.

Deterministic I/O boundary (SRS rule 3): unit conversion, documented defaults for unstated values
(each recorded as an assumption for the report), and the SRS clarification rule - payload mass and
endurance must come from the user (a named catalogue payload supplies its own mass).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Optional

from pydantic import ValidationError

from src.agent.agent_schemas import Altitude, Distance, Duration, Mass, Power, RequirementsDraft, Speed, Temperature
from src.schemas.components import Payload
from src.schemas.payload_catalog import SENSOR_LABELS, is_custom_name, payload_spec, resolve_payload_name
from src.schemas.requirements import (ARK_JETSON_PAB, CUBE_ORANGE_PLUS, CUSTOM_PAYLOAD, LAYOUT_LABELS,
                                      OperationalRequirements)
from src.tools.database.matcher import ComponentNotFoundError, match_payload
from src.units import (DAY_TO_S, FT_TO_M, G_TO_KG, HOUR_TO_S, KM_TO_M, KMH_TO_MS, KNOT_TO_MS, KW_TO_W, LB_TO_KG,
                       MILE_TO_M, MIN_TO_S, MPH_TO_MS, NMI_TO_M, OZ_TO_KG, FAHRENHEIT_OFFSET,
                       FAHRENHEIT_PER_CELSIUS, ISA_LAPSE_RATE_C_PER_M, ISA_SEA_LEVEL_TEMPERATURE_C)

DEFAULT_CRUISE_SPEED_MS: Final[float] = 22.0  # typical Group 1-3 ISR cruise
DEFAULT_RADIUS_KM: Final[float] = 50.0
FALLBACK_PAYLOAD_POWER_W: Final[float] = 50.0  # if the catalog has no matching sensor
ROUND_TRIP_LEGS: Final[float] = 2.0
ECHO_FIELDS: Final[tuple[str, ...]] = ("payload_mass", "payload_power", "endurance", "radius", "total_range",
                                        "cruise_speed", "service_ceiling", "max_stall_speed", "design_wind",
                                        "takeoff_altitude", "takeoff_air_temperature")
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


def _payload_fields(draft: RequirementsDraft, db_path: Path, notes: list[str]) -> Optional[dict[str, object]]:
    """Payload model, mass and power. A named catalogue model supplies its datasheet mass and power unless the user
    stated them; otherwise (Custom) the mass must be stated. None when neither a mass nor a known model was given."""
    model: Optional[str] = resolve_payload_name(draft.payload_model) if draft.payload_model else None
    if draft.payload_model and model is None and not is_custom_name(draft.payload_model):
        notes.append(f"Payload '{draft.payload_model}' is not in the catalogue; treated as a custom payload.")
    spec = payload_spec(model) if model is not None else None
    if spec is None:
        if draft.payload_mass is None:
            return None
        mass: float = kg(draft.payload_mass)
        return {"payload_model": CUSTOM_PAYLOAD, "payload_mass_kg": mass,
                "payload_power_w": _payload_power_w(draft, mass, db_path, notes)}
    notes += _model_notes(draft, spec)
    return {"payload_model": spec.name,
            "payload_mass_kg": spec.mass_kg if draft.payload_mass is None else kg(draft.payload_mass),
            "payload_power_w": spec.power_w if draft.payload_power is None else watts(draft.payload_power)}


def _model_notes(draft: RequirementsDraft, spec: Payload) -> list[str]:
    """Which datasheet values filled unstated ones, and a stated sensor class the named model overrides."""
    unstated: list[tuple[str, str]] = [(name, text) for name, text, stated in (
        ("mass", f"{spec.mass_kg:g} kg", draft.payload_mass), ("power", f"{spec.power_w:g} W", draft.payload_power))
        if stated is None]
    notes: list[str] = []
    if unstated:
        names, values = " and ".join(n for n, _ in unstated), ", ".join(v for _, v in unstated)
        notes.append(f"Payload {spec.name}: unstated {names} taken from its datasheet ({values}).")
    if draft.payload_sensor is not None and draft.payload_sensor != spec.sensor_type:
        sensor: str = SENSOR_LABELS.get(spec.sensor_type, spec.sensor_type)
        notes.append(f"Payload sensor '{draft.payload_sensor}' ignored: {spec.name} is an {sensor} payload.")
    return notes


def _speed_ms(draft: RequirementsDraft, notes: list[str]) -> float:
    if draft.cruise_speed is not None:
        return ms(draft.cruise_speed)
    notes.append(f"Cruise speed not stated: assumed {DEFAULT_CRUISE_SPEED_MS:.0f} m/s.")
    return DEFAULT_CRUISE_SPEED_MS


def _airframe_fields(draft: RequirementsDraft, notes: list[str]) -> dict[str, object]:
    layout: dict[str, object] = {} if draft.airframe_layout is None else {"airframe_layout": draft.airframe_layout}
    return {**layout, **_topology_fields(draft, notes)}


def _topology_fields(draft: RequirementsDraft, notes: list[str]) -> dict[str, object]:
    if draft.launch_method == "hand":
        if draft.vtol_required:
            notes.append("Both a hand launch and VTOL were requested: planned as a hand-launched fixed-wing "
                         "(a hand launch is not VTOL).")
        # A hand launch is its own fixed-wing mode: the runway stays false unless the user also stated one.
        return {"launch_method": "hand", "runway_available": bool(draft.runway_available)}
    if draft.launch_method == "runway":
        if draft.vtol_required:
            notes.append("Both a runway launch and VTOL were requested: planned as a runway-launched fixed-wing "
                         "(a stated launch method wins over the VTOL request).")
        return {"launch_method": "runway", "runway_available": draft.runway_available is not False}
    return _runway_fields(draft, notes)


def _runway_fields(draft: RequirementsDraft, notes: list[str]) -> dict[str, object]:
    if draft.vtol_required:
        return {"airframe_topology": "quadplane_vtol", "runway_available": False}
    if draft.vtol_required is False and draft.runway_available is None:
        notes.append("VTOL not required and runway availability not stated: assumed a runway (fixed-wing).")
        return {"airframe_topology": "fixed_wing", "runway_available": True}
    if draft.runway_available is None:
        notes.append("Runway availability not stated: assumed runway-independent (VTOL quadplane).")
        return {"runway_available": False}
    return {"runway_available": draft.runway_available}


def _avionics_fields(draft: RequirementsDraft, notes: list[str]) -> dict[str, object]:
    """New designs start autonomous: Cube Orange+ with a Jetson unless the request says otherwise."""
    if draft.flight_controller is None:
        notes.append("Flight controller not stated: assumed a Cube Orange+.")
    if draft.companion_computer is None and draft.flight_controller != ARK_JETSON_PAB:
        notes.append("Companion computer not stated: assumed a Jetson Orin NX is fitted (autonomous operation).")
    return {"flight_controller": draft.flight_controller or CUBE_ORANGE_PLUS,
            "companion_computer": True if draft.companion_computer is None else draft.companion_computer}


def celsius(q: Temperature) -> float:
    return q.value if q.unit == "C" else (q.value - FAHRENHEIT_OFFSET) / FAHRENHEIT_PER_CELSIUS


def isa_offset_c(air_temperature_c: float, altitude_m: float) -> float:
    """Deviation of a stated air temperature from the ISA (troposphere) temperature at that altitude."""
    return air_temperature_c - (ISA_SEA_LEVEL_TEMPERATURE_C - ISA_LAPSE_RATE_C_PER_M * altitude_m)


def _site_fields(draft: RequirementsDraft) -> dict[str, float]:
    """Take-off site: altitude, and an ISA offset stated directly (wins) or from the stated air temperature."""
    fields: dict[str, float] = {}
    altitude_m: float = metres(draft.takeoff_altitude) if draft.takeoff_altitude is not None else 0.0
    if draft.takeoff_altitude is not None:
        fields["takeoff_altitude_m"] = altitude_m
    if draft.isa_temperature_offset is not None:
        fields["isa_temperature_offset_c"] = draft.isa_temperature_offset
    elif draft.takeoff_air_temperature is not None:
        fields["isa_temperature_offset_c"] = isa_offset_c(celsius(draft.takeoff_air_temperature), altitude_m)
    return fields


def _optional_fields(draft: RequirementsDraft) -> dict[str, float]:
    fields: dict[str, float] = {}
    if draft.service_ceiling is not None:
        fields["service_ceiling_m"] = metres(draft.service_ceiling)
    if draft.max_stall_speed is not None:
        fields["max_stall_speed_ms"] = ms(draft.max_stall_speed)
    if draft.design_wind is not None:
        fields["design_wind_ms"] = ms(draft.design_wind)
    return {**fields, **_site_fields(draft)}


PYDANTIC_VALUE_ERROR_PREFIX: Final[str] = "Value error, "  # pydantic prepends this to a validator's ValueError


def issue_message(msg: str) -> str:
    """A validator's own wording, without pydantic's "Value error, " prefix."""
    return msg.removeprefix(PYDANTIC_VALUE_ERROR_PREFIX)


def _issue_text(error: dict[str, object], where: str) -> str:
    """"field: message", or just the message for a model-level error that names no field."""
    message: str = issue_message(str(error["msg"]))
    return f"{where}: {message}" if where else message


def to_requirements(draft: RequirementsDraft, db_path: Path) -> ParseOutcome:
    """Convert a draft; returns a clarification question instead if the SRS-required facts are missing
    or the stated values are outside the supported envelope."""
    notes: list[str] = []
    payload: Optional[dict[str, object]] = _payload_fields(draft, db_path, notes)
    missing: list[str] = [REQUIRED_FIELDS[name] for name, absent in
                          (("payload_mass", payload is None), ("endurance", draft.endurance is None)) if absent]
    if missing:
        return ParseOutcome(None, f"Please specify the {' and '.join(missing)} for this mission.")
    fields: dict[str, object] = {
        "mission_name": draft.mission_name, **payload, "endurance_hours": hours(draft.endurance),
        "combat_radius_km": _radius_km(draft, notes), "cruise_speed_ms": _speed_ms(draft, notes),
        "propulsion_type": draft.propulsion, "group_constraint": draft.dod_group,
        **_airframe_fields(draft, notes), **_avionics_fields(draft, notes), **_optional_fields(draft),
    }
    try:
        req = OperationalRequirements(**fields)
    except ValidationError as exc:
        problems = "; ".join(_issue_text(e, ".".join(map(str, e["loc"]))) for e in exc.errors())
        return ParseOutcome(None, f"Some requirements are outside the supported envelope ({problems}). "
                                  "Please revise them.")
    notes.insert(0, _parsed_echo(draft))
    notes += [f"Ambiguity noted by the parser: {text}" for text in draft.ambiguities]
    return ParseOutcome(req, None, tuple(notes))


def _parsed_echo(draft: RequirementsDraft) -> str:
    """What the LLM read from the request, in the user's own units, so a mis-read is visible."""
    stated: list[str] = [f"{name.replace('_', ' ')} {q.value:g} {q.unit}" for name in ECHO_FIELDS
                         if (q := getattr(draft, name)) is not None]
    if draft.airframe_layout is not None:
        stated.append(f"airframe layout {LAYOUT_LABELS[draft.airframe_layout]}")
    if draft.launch_method is not None:
        stated.append(f"{draft.launch_method} launch")
    return "Read from the request: " + (", ".join(stated) or "no quantities") + "."
