"""Deterministic half of the design critic (SRS 5.5): the four hard gates and how sizing knobs move.

Gates: (1) SM inside the layout's static-margin band; (2) VSPAERO L/D within +-15 % of the sizing L/D at the
sizing cruise and loiter CLs (VSPAERO with the viscous lift-dependent drag, stability.ld_deviations); (3) zero CAD
collisions and the as-built mass inside the DoD group limit; (4) end-of-mission energy reserve >= 15 %
(installed battery margin for electric, fuel reserve for series hybrid). A failed tool stage (e.g. the
CAD cannot be packed) is reported as a failed "pipeline" gate so the critic can react to it too.

The LLM proposes knob/direction pairs; this module applies fixed step sizes within the solver's
override bounds, and adds the rule-table fix for every failed gate so each iteration makes progress
even if the LLM's choice is poor (SRS Zero-LLM-Math: the model never picks a number).
"""

from dataclasses import asdict, dataclass
from typing import Final, Literal, Optional

from src.agent.agent_schemas import Adjustment, Knob
from src.schemas.assumptions import DEFAULT_ASSUMPTIONS, DesignAssumptions
from src.schemas.components import SelectedBOM
from src.schemas.knobs import OVERRIDE_BOUNDS, OVERRIDE_DEFAULTS
from src.schemas.requirements import OperationalRequirements
from src.schemas.state import NO_KIT_PREFIX, AeroStabilityOutput, CADArtifacts, SizingOutput
from src.tools.aero_cad.stability import LD_DEVIATION_TOLERANCE, ld_deviations
from src.tools.sizing import layouts
from src.tools.sizing.layouts import StaticMarginBand
from src.tools.sizing.mission_solver import reserve_fraction
from src.tools.sizing.topology import GROUP1_UPPER_KG, GROUP2_UPPER_KG, GROUP3_UPPER_KG
from src.units import WH_TO_J

RESERVE_MIN: Final[float] = 0.15  # SRS gate 4
GROUP_UPPER_KG: Final[dict[int, float]] = {1: GROUP1_UPPER_KG, 2: GROUP2_UPPER_KG, 3: GROUP3_UPPER_KG}
# (operation, step) per knob: multiplicative for geometric ratios, additive for coefficients/fractions
STEPS: Final[dict[str, tuple[Literal["mul", "add"], float]]] = {
    "aspect_ratio": ("mul", 1.15), "fuselage_length_factor": ("mul", 1.15),
    "tail_volume_h": ("add", 0.10), "tail_volume_v": ("add", 0.01), "reserve_fraction": ("add", 0.05),
}
TAILED_SM_KNOB: Final[str] = "tail_volume_h"  # the static-margin knob of the tailed layouts
VALUE_TOLERANCE: Final[float] = 1e-9
STEP_DECIMALS: Final[int] = 6

GateName = Literal["static_margin", "lift_to_drag", "collisions_and_group", "reserve", "pipeline"]


@dataclass(frozen=True)
class GateResult:
    name: GateName
    passed: bool
    detail: str
    side: Literal["", "low", "high"] = ""  # which side of a two-sided band a failed value fell on
    # L/D gate, low side: whether VSPAERO's extra drag at the worst point is mostly zero-lift (parasite) or induced
    # also "propulsion": no catalogue set matches, which no sizing knob can change
    driver: Literal["", "induced", "parasite", "propulsion"] = ""


@dataclass(frozen=True)
class AppliedChange:
    knob: str
    direction: str
    old: float
    new: float
    source: Literal["llm", "rule"]
    reason: str


def installed_reserve(sizing: SizingOutput, bom: SelectedBOM) -> float:
    """Electric: 1 - mission energy / installed COTS battery energy. Hybrid: the fuel reserve fraction."""
    if sizing.propulsion_type == "series_hybrid":
        return sizing.energy_reserve_fraction
    used_j: float = sum(phase.energy_j for phase in sizing.phases)
    return 1.0 - used_j / (bom.battery.capacity_wh * WH_TO_J)


def evaluate_gates(req: OperationalRequirements, sizing: SizingOutput, bom: SelectedBOM,
                   aero: AeroStabilityOutput, cad: CADArtifacts) -> list[GateResult]:
    sm: float = cad.static_margin
    band: StaticMarginBand = layouts.static_margin_band(sizing.airframe_layout)
    group: int = req.group_constraint or sizing.dod_group
    limit: float = GROUP_UPPER_KG[group]
    reserve: float = installed_reserve(sizing, bom)
    return [
        GateResult("static_margin", band.minimum <= sm <= band.maximum,
                   f"SM {sm:.3f} (band {band.minimum}-{band.maximum})"),
        _ld_gate(sizing, aero),
        GateResult("collisions_and_group", cad.collision_count == 0 and cad.as_built_mass_kg < limit,
                   f"{cad.collision_count} collision(s); as-built mass {cad.as_built_mass_kg:.2f} kg vs Group {group} "
                   f"limit {limit} kg" + _prop_strike_note(cad) + _missing_kit_chain(cad, group, limit)),
        GateResult("reserve", reserve >= RESERVE_MIN, f"energy reserve {reserve:.0%} (min {RESERVE_MIN:.0%})"),
    ]


def _prop_strike_note(cad: CADArtifacts) -> str:
    """What the propeller strikes, when one of the collisions is a propeller strike (tail or ground)."""
    return f"; {cad.prop_strike}" if cad.prop_strike else ""


def _missing_kit_chain(cad: CADArtifacts, group: int, limit: float) -> str:
    """Why a design without a kit is over its group: the packer fell back to the shell mass model, whose mass is
    the one the gate saw. Empty when there is a kit or the mass is within the limit."""
    if not (cad.kit_warning or "").startswith(NO_KIT_PREFIX) or cad.as_built_mass_kg < limit:
        return ""
    reason: str = cad.kit_warning.removeprefix(NO_KIT_PREFIX)
    return (f"; no laser kit ({reason}), so the shell mass model gives {cad.as_built_mass_kg:.2f} kg, over the "
            f"Group {group} limit of {limit} kg")


def _ld_gate(sizing: SizingOutput, aero: AeroStabilityOutput) -> GateResult:
    """Gate 2: VSPAERO and sizing L/D agree within the tolerance, either side, at both CLs. A low VSPAERO L/D also
    names its driver at the worst point: VSPAERO's extra zero-lift drag (CD0) against its extra induced drag
    (k CL^2). Only an induced gap is one a higher aspect ratio can close."""
    points = ld_deviations(sizing, aero)
    failed = [p for p in points if abs(p.deviation) > LD_DEVIATION_TOLERANCE]
    detail: str = "; ".join(f"{p.name} (CL {p.cl:.2f}): VSPAERO L/D {p.vspaero_ld:.1f} vs sizing {p.sizing_ld:.1f} "
                            f"({p.deviation:+.0%})" for p in points)
    detail += f" (band +-{LD_DEVIATION_TOLERANCE:.0%})"
    if not failed:
        return GateResult("lift_to_drag", True, detail)
    if not any(p.deviation < 0 for p in failed):
        return GateResult("lift_to_drag", False, detail, "high")
    worst = min(failed, key=lambda p: p.deviation)
    extra_parasite: float = worst.vspaero.cd0 - worst.sizing.cd0
    extra_induced: float = worst.vspaero.induced - worst.sizing.induced
    if extra_parasite < extra_induced:
        return GateResult("lift_to_drag", False, detail, "low", "induced")
    detail += (f"; the extra drag is mostly parasite (CD0 VSPAERO {aero.cd0:.4f} vs sizing {sizing.cd0:.4f}): "
               "a higher aspect ratio cannot close it")
    return GateResult("lift_to_drag", False, detail, "low", "parasite")


def evaluate_sizing_gates(req: OperationalRequirements, sizing: SizingOutput, bom: SelectedBOM) -> list[GateResult]:
    """Gates that need no aero/CAD (runs without OpenVSP): reserve, and the sized MTOW + COTS hardware
    estimate against the group limit. Static margin, L/D and collisions are not evaluated."""
    group: int = req.group_constraint or sizing.dod_group
    limit: float = GROUP_UPPER_KG[group]
    reserve: float = installed_reserve(sizing, bom)
    return [
        GateResult("collisions_and_group", sizing.mtow_kg < limit,
                   f"sized MTOW {sizing.mtow_kg:.2f} kg vs Group {group} limit {limit} kg (no CAD run)"),
        GateResult("reserve", reserve >= RESERVE_MIN, f"energy reserve {reserve:.0%} (min {RESERVE_MIN:.0%})"),
    ]


PROPULSION_STAGE: Final[str] = "propulsion_match"  # the stage_error name of a PropulsionMatchError


def pipeline_gate(stage: str, message: str) -> GateResult:
    if stage == PROPULSION_STAGE:
        return GateResult("pipeline", False, f"propulsion match failed: {message}; no sizing knob can fix a "
                          "catalogue with no matching hardware", driver="propulsion")
    return GateResult("pipeline", False, f"{stage} failed: {message}")


def static_margin_basis(layout: str) -> tuple[StaticMarginBand, str]:
    """The layout's static-margin band and the knob that moves the margin (both come from its spec)."""
    spec = layouts.layout_spec(layout)
    return spec.static_margin, spec.static_margin_knob


def _rule_fixes(gates: list[GateResult], cad: Optional[CADArtifacts], stage: Optional[str],
                band: Optional[StaticMarginBand], knob: str) -> list[Adjustment]:
    """Textbook response to each failed gate (the safety net under the LLM's choices). `knob` is the layout's
    static-margin knob; one the step table cannot move yet is an error, not a silent no-op."""
    fixes: list[Adjustment] = []
    for gate in (g for g in gates if not g.passed):
        if gate.name == "static_margin" and cad is not None:
            if band is None:
                raise ValueError("the static-margin fix needs the layout's band")
            if knob not in STEPS:
                raise ValueError(f"the static-margin knob {knob!r} cannot be adjusted yet")
            direction = "increase" if cad.static_margin < band.minimum else "decrease"
            fixes.append(Adjustment(knob=knob, direction=direction,
                                    reason=f"{knob} moves the neutral point and hence the SM"))
        elif gate.name == "lift_to_drag" and gate.side != "high" and gate.driver != "parasite":
            # no knob fix for a VSPAERO L/D above sizing, nor for a CD0 gap (AR only cuts induced drag)
            fixes.append(Adjustment(knob="aspect_ratio", direction="increase", reason="higher AR cuts induced drag"))
        elif gate.name == "collisions_and_group" and cad is not None and cad.collision_count:
            fixes.append(Adjustment(knob="fuselage_length_factor", direction="increase",
                                    reason="a longer pod gives the parts room"))
        elif gate.name == "reserve":
            fixes.append(Adjustment(knob="reserve_fraction", direction="increase",
                                    reason="size the energy store for a larger end-of-mission reserve"))
        elif gate.name == "pipeline" and stage == "aero_cad":
            fixes.append(Adjustment(knob="fuselage_length_factor", direction="increase",
                                    reason="the CAD could not be packed or balanced in the pod"))
    return fixes


def _current(knob: str, overrides: dict[str, float], propulsion: str,
             assumptions: DesignAssumptions = DEFAULT_ASSUMPTIONS) -> float:
    if knob == "reserve_fraction":
        return reserve_fraction(overrides, propulsion, assumptions)  # type: ignore[arg-type]
    return overrides.get(knob, OVERRIDE_DEFAULTS[knob])


def _step(knob: str, direction: str, value: float) -> float:
    op, step = STEPS[knob]
    sign: float = 1.0 if direction == "increase" else -1.0
    raw: float = value * step**sign if op == "mul" else value + sign * step
    low, high = OVERRIDE_BOUNDS[knob]
    return round(min(max(raw, low), high), STEP_DECIMALS)  # 0.10 + 0.05 must be exactly 0.15 for the gate


def apply_adjustments(overrides: dict[str, float], propulsion: str, llm_choices: list[Adjustment],
                      gates: list[GateResult], cad: Optional[CADArtifacts],
                      stage: Optional[str] = None,
                      assumptions: DesignAssumptions = DEFAULT_ASSUMPTIONS, *,
                      band: Optional[StaticMarginBand], sm_knob: str) -> tuple[dict[str, float], list[AppliedChange]]:
    """New overrides (a copy) plus the changes made. One move per knob. The rule table wins on every knob
    a failed gate governs (so a poor or prompt-injected LLM choice cannot push a knob the wrong way); the
    LLM may add moves on other knobs. The designer's reserve is a floor: a reserve step that would cross
    it is dropped (the critic may only raise the reserve)."""
    chosen: dict[Knob, tuple[Adjustment, Literal["llm", "rule"]]] = {}
    for adj in _rule_fixes(gates, cad, stage, band, sm_knob):
        chosen.setdefault(adj.knob, (adj, "rule"))
    for adj in llm_choices:
        if not _refused(adj, gates):
            chosen.setdefault(adj.knob, (adj, "llm"))
    updated: dict[str, float] = dict(overrides)
    changes: list[AppliedChange] = []
    for knob, (adj, source) in chosen.items():
        old: float = _current(knob, overrides, propulsion, assumptions)
        new: float = _step(knob, adj.direction, old)
        below_floor: bool = knob == "reserve_fraction" and new < assumptions.reserve_for(propulsion) - VALUE_TOLERANCE
        if abs(new - old) > VALUE_TOLERANCE and not below_floor:
            updated[knob] = new
            changes.append(AppliedChange(knob, adj.direction, old, new, source, adj.reason))
    return updated, changes


def _refused(adj: Adjustment, gates: list[GateResult]) -> bool:
    """An LLM move the gates rule out: any knob against a propulsion no-match (no knob fixes the catalogue), and a
    higher aspect ratio against a parasite-driven L/D gap (it cannot close it and walks the wing toward its
    stiffness limits)."""
    if any(g.name == "pipeline" and not g.passed and g.driver == "propulsion" for g in gates):
        return True
    parasite_gap: bool = any(g.name == "lift_to_drag" and not g.passed and g.driver == "parasite" for g in gates)
    return parasite_gap and adj.knob == "aspect_ratio" and adj.direction == "increase"


def unfixable(gates: list[GateResult], cad: Optional[CADArtifacts], stage: Optional[str],
              *, band: Optional[StaticMarginBand], sm_knob: str) -> list[str]:
    """Failed gates the rule table has no automatic fix for (e.g. a mass overrun, a sizing divergence)."""
    fixed_by_rule: set[str] = set()
    for gate in (g for g in gates if not g.passed):
        if _rule_fixes([gate], cad, stage, band, sm_knob):
            fixed_by_rule.add(gate.name)
    return [g.name for g in gates if not g.passed and g.name not in fixed_by_rule]


def gates_as_dicts(gates: list[GateResult]) -> list[dict[str, object]]:
    return [asdict(g) for g in gates]
