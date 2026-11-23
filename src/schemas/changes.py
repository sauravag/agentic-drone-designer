"""Change sets: the explicit allow-list of what the designer (or the LLM) may alter on a re-run.

Every sub-model field is `Optional[...] = None` with `extra="forbid"` so the strict LLM JSON schema
is an allow-list; `None` means "leave unchanged". Applying a change set never mutates its inputs and
re-validates through the target models, so bounds and cross-field validators always run.
"""

from decimal import Decimal
from typing import Final, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.assumptions import DesignAssumptions
from src.schemas.requirements import (ARK_JETSON_PAB, CUSTOM_PAYLOAD, LAYOUT_LABELS, AirframeLayout, FlightControllerId,
                                   OperationalRequirements)
from src.schemas.knobs import OVERRIDE_BOUNDS, OVERRIDE_DEFAULTS
from src.schemas.payload_catalog import CUSTOM_LABEL, payload_spec

LABEL_MAX_LEN: Final[int] = 120

FIELD_LABELS: Final[dict[str, str]] = {
    # requirements
    "propulsion_type": "Propulsion", "airframe_topology": "Airframe", "airframe_layout": "Airframe layout",
    "runway_available": "Runway available", "launch_method": "Launch",
    "group_constraint": "Group limit", "payload_mass_kg": "Payload mass (kg)", "payload_power_w": "Payload power (W)",
    "endurance_hours": "Endurance (h)", "combat_radius_km": "Combat radius (km)",
    "cruise_speed_ms": "Cruise speed (m/s)", "service_ceiling_m": "Service ceiling (m)",
    "max_stall_speed_ms": "Max stall speed (m/s)", "design_wind_ms": "Design headwind (m/s)",
    "takeoff_altitude_m": "Take-off altitude (m)", "isa_temperature_offset_c": "ISA temperature offset (°C)",
    "flight_controller": "Flight controller",
    "companion_computer": "Jetson companion computer", "payload_model": "Payload",
    # assumptions
    "liion_specific_energy_wh_kg": "Specific energy (Li-ion, Wh/kg)",
    "liion_energy_density_wh_l": "Energy density (Li-ion, Wh/L)",
    "liion_max_c_rate": "Max discharge (Li-ion)",
    "lipo_specific_energy_wh_kg": "Specific energy (LiPo, Wh/kg)",
    "lipo_energy_density_wh_l": "Energy density (LiPo, Wh/L)",
    "lipo_max_c_rate": "Max discharge (LiPo)",
    "battery_reserve_fraction": "Battery reserve", "fuel_reserve_fraction": "Fuel reserve",
    "bsfc_kg_per_kwh": "Generator BSFC (kg/kWh)", "drivetrain_efficiency": "Drivetrain efficiency",
    "hover_figure_of_merit": "Hover figure of merit", "lift_motor_efficiency": "Lift motor efficiency",
    "avionics_base_mass_kg": "Avionics allowance base (kg)",
    "avionics_mass_per_mtow": "Avionics allowance per kg MTOW (kg/kg)",
    "avionics_base_power_w": "Avionics power base (W)", "avionics_power_per_kg_w": "Avionics power per kg MTOW (W/kg)",
    "cube_orange_plus_mass_kg": "Cube Orange+ mass (kg)", "ark_pab_stack_mass_kg": "ARK PAB stack mass (kg)",
    "companion_computer_mass_kg": "Jetson companion mass (kg)", "jetson_power_w": "Jetson power (W)",
    # knobs
    "aspect_ratio": "Aspect ratio", "tail_volume_h": "Horizontal tail volume",
    "tail_volume_v": "Vertical tail volume", "fuselage_length_factor": "Fuselage length factor",
}


class RequirementChanges(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    propulsion_type: Optional[Literal["electric", "series_hybrid", "auto"]] = None
    airframe_topology: Optional[Literal["fixed_wing", "quadplane_vtol", "auto"]] = None
    airframe_layout: Optional[AirframeLayout] = None
    runway_available: Optional[bool] = None
    launch_method: Optional[Literal["auto", "runway", "hand"]] = None
    group_constraint: Optional[Literal[1, 2, 3]] = None
    payload_mass_kg: Optional[float] = None
    payload_power_w: Optional[float] = None
    endurance_hours: Optional[float] = None
    combat_radius_km: Optional[float] = None
    cruise_speed_ms: Optional[float] = None
    service_ceiling_m: Optional[float] = None
    max_stall_speed_ms: Optional[float] = None
    design_wind_ms: Optional[float] = None
    takeoff_altitude_m: Optional[float] = None
    isa_temperature_offset_c: Optional[float] = None
    flight_controller: Optional[FlightControllerId] = None
    companion_computer: Optional[bool] = None
    payload_model: Optional[str] = None  # a catalogue payload name or "custom"; checked when the change is applied


class AssumptionChanges(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    liion_specific_energy_wh_kg: Optional[float] = None
    liion_energy_density_wh_l: Optional[float] = None
    liion_max_c_rate: Optional[float] = None
    lipo_specific_energy_wh_kg: Optional[float] = None
    lipo_energy_density_wh_l: Optional[float] = None
    lipo_max_c_rate: Optional[float] = None
    battery_reserve_fraction: Optional[float] = None
    fuel_reserve_fraction: Optional[float] = None
    bsfc_kg_per_kwh: Optional[float] = None
    drivetrain_efficiency: Optional[float] = None
    hover_figure_of_merit: Optional[float] = None
    lift_motor_efficiency: Optional[float] = None
    avionics_base_mass_kg: Optional[float] = None
    avionics_mass_per_mtow: Optional[float] = None
    avionics_base_power_w: Optional[float] = None
    avionics_power_per_kg_w: Optional[float] = None
    cube_orange_plus_mass_kg: Optional[float] = None
    ark_pab_stack_mass_kg: Optional[float] = None
    companion_computer_mass_kg: Optional[float] = None
    jetson_power_w: Optional[float] = None


class KnobChanges(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    aspect_ratio: Optional[float] = None
    tail_volume_h: Optional[float] = None
    tail_volume_v: Optional[float] = None
    fuselage_length_factor: Optional[float] = None


def _set_fields(model: BaseModel) -> dict[str, object]:
    """The fields a change model actually sets (non-None)."""
    return {name: value for name, value in model.model_dump().items() if value is not None}


def _requirement_values(req: OperationalRequirements, changes: RequirementChanges) -> dict[str, object]:
    """Merged requirement values as they will be after re-validation, so `describe` and `apply_changes` agree.

    Moving off the ARK stack keeps its Jetson (as a separate companion) unless the change says otherwise; on the
    ARK stack a separate companion never applies (the schema normalises it away)."""
    values: dict[str, object] = {**req.model_dump(), **_set_fields(changes)}
    leaving_ark = req.flight_controller == ARK_JETSON_PAB and values["flight_controller"] != ARK_JETSON_PAB
    if leaving_ark and changes.companion_computer is None:
        values["companion_computer"] = True
    if values["flight_controller"] == ARK_JETSON_PAB:
        values["companion_computer"] = False
    return {**values, **_model_payload_values(req, changes)}


def _model_payload_values(req: OperationalRequirements, changes: RequirementChanges) -> dict[str, float]:
    """Choosing a different catalogue model brings its datasheet mass and power, unless the change states them.
    Re-stating the current model, switching to Custom (or an unknown name, rejected on apply) brings nothing."""
    switching: bool = changes.payload_model not in (None, req.payload_model)
    spec = payload_spec(changes.payload_model) if switching else None
    if spec is None:
        return {}
    implied: dict[str, float] = {"payload_mass_kg": spec.mass_kg, "payload_power_w": spec.power_w}
    return {name: value for name, value in implied.items() if getattr(changes, name) is None}


def _format(value: object) -> str:
    """Compact display: '3' not '3.0'; booleans as yes/no."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value == 0:
            return "0"  # folds -0.0
        # shortest round-trip digits, fixed-point (never exponent notation)
        return format(Decimal(repr(float(value))).normalize(), "f")
    return str(value)


def _format_requirement(name: str, value: object) -> str:
    """`_format`, except that Custom shows as 'Custom payload' rather than its stored value 'custom'."""
    if name == "payload_model" and value == CUSTOM_PAYLOAD:
        return CUSTOM_LABEL
    if name == "airframe_layout":
        return LAYOUT_LABELS.get(str(value), str(value))
    return _format(value)


class ChangeSet(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str = Field(..., min_length=1, max_length=LABEL_MAX_LEN)
    requirements: RequirementChanges
    assumptions: AssumptionChanges
    knobs: KnobChanges

    def is_empty(self) -> bool:
        return not (_set_fields(self.requirements) or _set_fields(self.assumptions) or _set_fields(self.knobs))

    def describe(self, req: OperationalRequirements, assumptions: DesignAssumptions,
                 current_knobs: Optional[Mapping[str, float]] = None) -> list[tuple[str, str, str]]:
        """(field label, old, new) for every changed field whose value actually differs.

        A knob's baseline is `current_knobs[name]` if given, else the solver's constant default."""
        rows: list[tuple[str, str, str]] = []
        merged = _requirement_values(req, self.requirements)
        # Requirement rows compare the effective result, so implied changes (the Jetson kept off ARK, a model's
        # datasheet mass and power) show and no-ops do not.
        for name in RequirementChanges.model_fields:
            if getattr(req, name) != merged[name]:
                rows.append((FIELD_LABELS[name], _format_requirement(name, getattr(req, name)),
                             _format_requirement(name, merged[name])))
        for name, new in _set_fields(self.assumptions).items():
            old = getattr(assumptions, name)
            if old != new:
                rows.append((FIELD_LABELS[name], _format(old), _format(new)))
        for name, new in _set_fields(self.knobs).items():
            old_knob = (current_knobs or {}).get(name, OVERRIDE_DEFAULTS[name])
            if old_knob != new:
                rows.append((FIELD_LABELS[name], _format(float(old_knob)), _format(new)))
        return rows


def apply_changes(
    req: OperationalRequirements, assumptions: DesignAssumptions, change: ChangeSet
) -> tuple[OperationalRequirements, DesignAssumptions, dict[str, float]]:
    """New (requirements, assumptions, knob overrides); inputs are untouched.

    Raises `pydantic.ValidationError` for out-of-bounds requirements/assumptions and `ValueError` for knobs.
    """
    new_req = OperationalRequirements.model_validate(_requirement_values(req, change.requirements))
    new_assumptions = DesignAssumptions.model_validate({**assumptions.model_dump(), **_set_fields(change.assumptions)})
    knobs: dict[str, float] = {name: float(value) for name, value in _set_fields(change.knobs).items()}
    for name, value in knobs.items():
        low, high = OVERRIDE_BOUNDS[name]
        if not low <= value <= high:
            raise ValueError(f"{FIELD_LABELS[name]} {value} is outside the allowed range {low}-{high}")
    return new_req, new_assumptions, knobs

