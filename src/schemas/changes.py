"""Change sets: the explicit allow-list of what the designer (or the LLM) may alter on a re-run.

Every sub-model field is `Optional[...] = None` with `extra="forbid"` so the strict LLM JSON schema
is an allow-list; `None` means "leave unchanged". Applying a change set never mutates its inputs and
re-validates through the target models, so bounds and cross-field validators always run.
"""

from typing import Final, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.assumptions import DesignAssumptions
from src.schemas.requirements import OperationalRequirements
from src.tools.sizing.mission_solver import OVERRIDE_BOUNDS, OVERRIDE_DEFAULTS

LABEL_MAX_LEN: Final[int] = 120
FLOAT_DECIMALS: Final[int] = 6  # fixed-point, trailing zeros stripped; never exponent notation

FIELD_LABELS: Final[dict[str, str]] = {
    # requirements
    "propulsion_type": "Propulsion", "airframe_topology": "Airframe", "runway_available": "Runway available",
    "group_constraint": "Group limit", "payload_mass_kg": "Payload mass (kg)", "payload_power_w": "Payload power (W)",
    "endurance_hours": "Endurance (h)", "combat_radius_km": "Combat radius (km)",
    "cruise_speed_ms": "Cruise speed (m/s)", "service_ceiling_m": "Service ceiling (m)",
    "max_stall_speed_ms": "Max stall speed (m/s)",
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
    # knobs
    "aspect_ratio": "Aspect ratio", "tail_volume_h": "Horizontal tail volume",
    "tail_volume_v": "Vertical tail volume", "fuselage_length_factor": "Fuselage length factor",
}


class RequirementChanges(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    propulsion_type: Optional[Literal["electric", "series_hybrid", "auto"]] = None
    airframe_topology: Optional[Literal["fixed_wing", "quadplane_vtol", "auto"]] = None
    runway_available: Optional[bool] = None
    group_constraint: Optional[Literal[1, 2, 3]] = None
    payload_mass_kg: Optional[float] = None
    payload_power_w: Optional[float] = None
    endurance_hours: Optional[float] = None
    combat_radius_km: Optional[float] = None
    cruise_speed_ms: Optional[float] = None
    service_ceiling_m: Optional[float] = None
    max_stall_speed_ms: Optional[float] = None


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


class KnobChanges(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    aspect_ratio: Optional[float] = None
    tail_volume_h: Optional[float] = None
    tail_volume_v: Optional[float] = None
    fuselage_length_factor: Optional[float] = None


def _set_fields(model: BaseModel) -> dict[str, object]:
    """The fields a change model actually sets (non-None)."""
    return {name: value for name, value in model.model_dump().items() if value is not None}


def _format(value: object) -> str:
    """Compact display: '3' not '3.0'; booleans as yes/no."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{FLOAT_DECIMALS}f}".rstrip("0").rstrip(".")
    return str(value)


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
        sources: list[tuple[BaseModel, BaseModel]] = [(self.requirements, req), (self.assumptions, assumptions)]
        for changes, current in sources:
            for name, new in _set_fields(changes).items():
                old = getattr(current, name)
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
    new_req = OperationalRequirements.model_validate({**req.model_dump(), **_set_fields(change.requirements)})
    new_assumptions = DesignAssumptions.model_validate({**assumptions.model_dump(), **_set_fields(change.assumptions)})
    knobs: dict[str, float] = {name: float(value) for name, value in _set_fields(change.knobs).items()}
    for name, value in knobs.items():
        low, high = OVERRIDE_BOUNDS[name]
        if not low <= value <= high:
            raise ValueError(f"{FIELD_LABELS[name]} {value} is outside the allowed range {low}-{high}")
    return new_req, new_assumptions, knobs
