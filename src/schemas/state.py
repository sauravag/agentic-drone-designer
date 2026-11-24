"""LangGraph DesignState and the tool-output schemas each node writes into it (all SI)."""

import math
from collections.abc import Sequence
from typing import Annotated, Final, Literal, Optional, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.schemas.assumptions import DesignAssumptions
from src.schemas.changes import ChangeSet
from src.schemas.components import SelectedBOM
from src.schemas.fabrication import FabricationSettings
from src.schemas.requirements import AirframeLayout, OperationalRequirements

Positive = Annotated[float, Field(gt=0)]
NonNegative = Annotated[float, Field(ge=0)]
Finite = Annotated[float, Field(allow_inf_nan=False)]


NO_KIT_PREFIX: Final[str] = "No laser kit: "  # CADArtifacts.kit_warning when the packer could not build a kit


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


MASS_CLOSURE_TOLERANCE: float = 1e-3


class PhaseResult(_Frozen):
    """Integrated result of one mission phase (time-stepped)."""

    name: Literal["VTOL_Takeoff", "Climb", "Ingress", "Loiter", "Egress", "VTOL_Land"]
    duration_s: Positive
    propulsive_power_w: NonNegative  # mean electrical power to propulsors
    total_power_w: NonNegative  # mean propulsion + payload + avionics
    energy_j: NonNegative  # electrical energy consumed
    fuel_burned_kg: NonNegative = 0.0


TAIL_AREA_REL_TOLERANCE: float = 1e-9


class TailSurface(_Frozen):
    """One tail surface as sizing defines it; the CAD layout and the flight model read it."""

    name: str  # "htail", "vtail"
    control: Literal["elevator", "rudder", "ruddervator"]
    area_m2: Positive  # whole surface (both panels if symmetric)
    span_m: Positive  # tip to tip if symmetric, else fin height
    taper_ratio: Positive
    dihedral_deg: float  # 0 horizontal, 90 vertical fin
    arm_m: Positive  # wing AC to surface AC
    is_symmetric: bool


def effective_tail_areas(surfaces: Sequence[TailSurface]) -> tuple[float, float]:
    """(horizontal, vertical) projected areas: sum of S cos^2(dihedral) and S sin^2(dihedral)."""
    horizontal: float = sum(t.area_m2 * math.cos(math.radians(t.dihedral_deg)) ** 2 for t in surfaces)
    vertical: float = sum(t.area_m2 * math.sin(math.radians(t.dihedral_deg)) ** 2 for t in surfaces)
    return horizontal, vertical


class SizingOutput(_Frozen):
    # --- SRS §4.1 fields ---
    mtow_kg: Positive
    empty_mass_kg: Positive  # everything except payload, battery and fuel
    # Itemised empty mass [kg] at the converged MTOW (structure incl. "tail_boom"/"lift_booms" tubes, propulsion,
    # gear, avionics, systems); sums to empty_mass_kg. Empty in records that predate it.
    empty_mass_breakdown: dict[str, NonNegative] = Field(default_factory=dict)
    fuel_mass_kg: NonNegative
    battery_mass_kg: NonNegative
    wing_area_m2: Positive
    wingspan_m: Positive
    mean_chord_m: Positive
    aspect_ratio: Positive
    fuselage_length_m: Positive
    fuselage_diameter_m: Positive
    cruise_power_w: Positive  # total electrical power at cruise (propulsion + hotel)
    hover_power_w: NonNegative  # peak electrical hover power; 0 for conventional fixed-wing
    v_stall_ms: Positive  # stall speed at MTOW as an equivalent (sea-level) airspeed, as ArduPilot reads airspeeds
    # True stall speed at the take-off site (altitude + ISA offset): the speed max_stall_speed_ms limits.
    # Optional so records sized before hot-and-high existed still load (they were sea-level sites).
    v_stall_takeoff_ms: Optional[Positive] = None
    v_loiter_ms: Positive
    v_cruise_ms: Optional[Positive] = None  # requested cruise speed; None in records that predate it
    # lift coefficients the sizing L/D assumed at cruise and loiter (the L/D gate evaluates VSPAERO at the same CLs)
    cl_cruise: Optional[Positive] = None  # None in records that predate the same-CL L/D gate
    cl_loiter: Optional[Positive] = None
    # --- extensions consumed by matcher / CAD / critic ---
    assumed_ld_cruise: Positive  # L/D the sizing loop assumed; critic compares to VSPAERO
    payload_mass_kg: Positive
    dod_group: Literal[1, 2, 3]
    propulsion_type: Literal["electric", "series_hybrid"]
    airframe_topology: Literal["fixed_wing", "quadplane_vtol"]
    airfoil: str
    cl_max: Positive  # 3D wing CLmax used for the stall constraint
    cd0: Positive
    oswald_e: Positive
    horizontal_tail_area_m2: NonNegative  # 0 for a layout without a horizontal tail
    vertical_tail_area_m2: NonNegative
    tail_arm_m: NonNegative
    launch_method: Literal["runway", "hand", "vertical"] = "runway"  # records that predate hand launch used runways
    gear_kind: Literal["none", "tricycle"] = "tricycle"  # records that predate hand launch had gear
    airframe_layout: AirframeLayout = "pod_boom_tractor"  # records that predate layouts were all tractors
    tail_surfaces: tuple[TailSurface, ...] = ()  # empty in records that predate layouts
    battery_energy_j: Positive  # installed capacity
    battery_peak_power_w: Positive
    generator_power_w: NonNegative  # 0 for pure electric
    cruise_motor_power_w: Positive  # peak forward-flight electrical propulsive power
    vtol_thrust_per_motor_n: NonNegative  # max (T/W-scaled) thrust per lift rotor; 0 for fixed-wing
    rotor_diameter_m: NonNegative
    energy_reserve_fraction: NonNegative  # electric: SoC reserve; hybrid: reserve fuel fraction
    phases: tuple[PhaseResult, ...]
    avionics_mass_kg: Positive  # what sizing used: shared allowance + flight-controller stack (+ Jetson)
    avionics_power_w: Positive  # whole-flight avionics draw incl. a fitted Jetson (excludes payload)

    @model_validator(mode="after")
    def _mass_closure(self) -> "SizingOutput":
        total: float = self.empty_mass_kg + self.payload_mass_kg + self.fuel_mass_kg + self.battery_mass_kg
        if abs(total - self.mtow_kg) > MASS_CLOSURE_TOLERANCE * self.mtow_kg:
            raise ValueError(f"mass closure failed: components sum to {total:.3f} kg vs mtow {self.mtow_kg:.3f} kg")
        return self

    @model_validator(mode="after")
    def _tail_areas_match_surfaces(self) -> "SizingOutput":
        if not self.tail_surfaces:
            return self
        horizontal, vertical = effective_tail_areas(self.tail_surfaces)
        for name, legacy, effective in (("horizontal", self.horizontal_tail_area_m2, horizontal),
                                        ("vertical", self.vertical_tail_area_m2, vertical)):
            if not math.isclose(legacy, effective, rel_tol=TAIL_AREA_REL_TOLERANCE, abs_tol=0.0):
                raise ValueError(f"{name} tail area {legacy} does not match its tail surfaces ({effective})")
        return self


class ControlSurfaceDerivatives(_Frozen):
    """Aerodynamic derivatives per rad of one servo-driven surface's own deflection (body axes, command convention)."""

    name: str  # "aileron", "elevator", "rudder", "ruddervator_left", "ruddervator_right"
    cl_per_rad: float
    cl_roll_per_rad: float
    cm_per_rad: float
    cn_per_rad: float


class AeroStabilityOutput(_Frozen):
    cl0: float
    cl_alpha_per_rad: float
    cd0: Positive
    ld_cruise: Positive  # at the sizing cruise CL
    # k in CD = CD0 + k*CL^2: VLM induced drag plus viscous lift-dependent drag; None in records that predate it
    induced_drag_factor: Optional[Positive] = None
    cm_alpha_per_rad: float
    cn_beta_per_rad: float
    cl_beta_per_rad: float
    cl_delta_a_per_rad: float
    cm_delta_e_per_rad: float
    cn_delta_r_per_rad: float
    x_np_m: Positive  # neutral point, metres aft of nose datum
    control_surfaces: tuple[ControlSurfaceDerivatives, ...] = ()  # one per servo-driven surface; empty in old records


class CADArtifacts(_Frozen):
    oml_vsp3_path: str
    oml_step_path: str
    assembly_step_path: str
    x_cg_m: Positive
    static_margin: float
    collision_count: int = Field(..., ge=0)
    as_built_mass_kg: Positive  # sized airframe + actual COTS parts; critic checks it against sized MTOW
    ixx_kgm2: Positive
    iyy_kgm2: Positive
    izz_kgm2: Positive
    ixz_kgm2: float
    viewer_model_path: Optional[str] = None  # display mesh for the 3D viewer; None if export failed
    kit_dir: Optional[str] = None  # laser-cut kit folder (DXF sheets, build sheet, parts list); None without kit files
    build_sheet_path: Optional[str] = None
    kit_mass_kg: Optional[NonNegative] = None  # packed kit mass (tubes, overwrap, covering, glue); None without a kit
    kit_warning: Optional[str] = None  # why there is no kit, or why its files or solids could not be written
    # Cruise-propeller hub of the packed layout (structural axes). None in records saved before it existed: those are
    # all nose tractors with the hub at the origin.
    propeller_x_m: Optional[Finite] = None
    propeller_z_m: Optional[Finite] = None


class PlantCaseResult(_Frozen):
    """Roll/pitch step response of one SITL plant (nominal, hot or sluggish)."""

    case: str
    roll_overshoot: NonNegative
    roll_settling_s: NonNegative
    pitch_overshoot: NonNegative
    pitch_settling_s: NonNegative
    error: Optional[str] = None  # set when the flight crashed; the metrics are then 0


class FCSArtifacts(_Frozen):
    param_path: str
    sitl_report_path: str
    passed: bool
    roll_overshoot: NonNegative  # fraction, 0.2 == 20 %; worst over the plants that completed a flight
    pitch_overshoot: NonNegative
    roll_settling_s: NonNegative
    pitch_settling_s: NonNegative
    max_transition_altitude_loss_m: NonNegative
    gain_scale: Positive = 1.0
    attempts: int = Field(default=1, ge=1)
    # worst over the plants that completed a flight; a crashed plant governs and is listed with its error
    governing_case: str = "nominal"
    perturbation_cases: tuple[PlantCaseResult, ...] = ()


class RunLineage(_Frozen):
    """Which run a re-run was derived from, and the change that was applied to it."""

    parent_run_id: str
    change_set: ChangeSet
    change_source: Literal["option", "chat", "manual"]
    fabrication_changes: tuple[str, ...] = ()  # R37: kit settings that differ from the parent's (field names)


class DesignState(TypedDict):
    user_prompt: str
    requirements: Optional[OperationalRequirements]
    clarification_needed: Optional[str]
    iteration_count: int  # hard cap: 8
    sizing_overrides: dict[str, float]
    sizing_results: Optional[SizingOutput]
    selected_bom: Optional[SelectedBOM]
    aero_stability: Optional[AeroStabilityOutput]
    cad_artifacts: Optional[CADArtifacts]
    fcs_artifacts: Optional[FCSArtifacts]
    validation_errors: list[str]
    final_report_path: Optional[str]
    # orchestration bookkeeping (beyond the SRS minimum)
    run_id: Optional[str]
    out_dir: Optional[str]
    assumptions: list[str]  # defaults the parser applied, reported verbatim
    design_assumptions: DesignAssumptions  # the designer's technology constants for this run
    fabrication: FabricationSettings  # laser-cut kit settings for this run (never sent to or changed by the LLM)
    gate_results: list[dict]  # latest critic gates: {name, passed, detail}
    critic_log: list[dict]  # one entry per critic pass
    stage_error: Optional[dict[str, str]]  # {stage, message} of a tool node that failed this iteration
    converged: bool  # every critic gate passed
    critic_exhausted: bool  # iteration cap reached, or no knob could move
    fatal_error: Optional[str]  # the run cannot continue (e.g. the LLM parser is unavailable)
    sitl_error: Optional[str]  # the SITL stage could not complete (distinct from completed-but-failed)
    explorer_mode: str  # closure | airframe | improvement | none, set when the report is compiled
    explore_options: list[dict]  # ExploreOption.to_json() of the closure options (empty otherwise)
    options_explanation: Optional[str]  # LLM prose about the options (audited, defanged in the report)
    explorer_note: Optional[str]  # set when a failed design closes unchanged with sizing + catalog
    lineage: Optional[RunLineage]  # set on a linked re-run
    viewer_error: Optional[str]  # why viewer.html was not built (a run warning, never a failed run)
