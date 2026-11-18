"""LangGraph DesignState and the tool-output schemas each node writes into it (all SI)."""

from typing import Annotated, Literal, Optional, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.schemas.assumptions import DesignAssumptions
from src.schemas.changes import ChangeSet
from src.schemas.components import SelectedBOM
from src.schemas.requirements import OperationalRequirements

Positive = Annotated[float, Field(gt=0)]
NonNegative = Annotated[float, Field(ge=0)]


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


class SizingOutput(_Frozen):
    # --- SRS §4.1 fields ---
    mtow_kg: Positive
    empty_mass_kg: Positive  # everything except payload, battery and fuel
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
    v_stall_ms: Positive  # sea-level stall speed at MTOW
    v_loiter_ms: Positive
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
    horizontal_tail_area_m2: Positive
    vertical_tail_area_m2: Positive
    tail_arm_m: Positive
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


class AeroStabilityOutput(_Frozen):
    cl0: float
    cl_alpha_per_rad: float
    cd0: Positive
    ld_cruise: Positive
    cm_alpha_per_rad: float
    cn_beta_per_rad: float
    cl_beta_per_rad: float
    cl_delta_a_per_rad: float
    cm_delta_e_per_rad: float
    cn_delta_r_per_rad: float
    x_np_m: Positive  # neutral point, metres aft of nose datum


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


class FCSArtifacts(_Frozen):
    param_path: str
    sitl_report_path: str
    passed: bool
    roll_overshoot: NonNegative  # fraction, 0.2 == 20 %
    pitch_overshoot: NonNegative
    roll_settling_s: NonNegative
    pitch_settling_s: NonNegative
    max_transition_altitude_loss_m: NonNegative
    gain_scale: Positive = 1.0
    attempts: int = Field(default=1, ge=1)


class RunLineage(_Frozen):
    """Which run a re-run was derived from, and the change that was applied to it."""

    parent_run_id: str
    change_set: ChangeSet
    change_source: Literal["option", "chat", "manual"]


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
