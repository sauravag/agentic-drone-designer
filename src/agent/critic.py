"""Deterministic half of the design critic (SRS 5.5): the four hard gates and how sizing knobs move.

Gates: (1) 0.08 <= SM <= 0.14; (2) VSPAERO cruise L/D >= 0.90 x the sizing assumption; (3) zero CAD
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
from src.schemas.state import AeroStabilityOutput, CADArtifacts, SizingOutput
from src.tools.aero_cad.cad_packer import SM_MAX, SM_MIN
from src.tools.sizing.mission_solver import reserve_fraction
from src.tools.sizing.topology import GROUP1_UPPER_KG, GROUP2_UPPER_KG, GROUP3_UPPER_KG
from src.units import WH_TO_J

LD_RATIO_MIN: Final[float] = 0.90  # SRS gate 2
RESERVE_MIN: Final[float] = 0.15  # SRS gate 4
GROUP_UPPER_KG: Final[dict[int, float]] = {1: GROUP1_UPPER_KG, 2: GROUP2_UPPER_KG, 3: GROUP3_UPPER_KG}
# (operation, step) per knob: multiplicative for geometric ratios, additive for coefficients/fractions
STEPS: Final[dict[str, tuple[Literal["mul", "add"], float]]] = {
    "aspect_ratio": ("mul", 1.15), "fuselage_length_factor": ("mul", 1.15),
    "tail_volume_h": ("add", 0.10), "tail_volume_v": ("add", 0.01), "reserve_fraction": ("add", 0.05),
}
VALUE_TOLERANCE: Final[float] = 1e-9
STEP_DECIMALS: Final[int] = 6

GateName = Literal["static_margin", "lift_to_drag", "collisions_and_group", "reserve", "pipeline"]


@dataclass(frozen=True)
class GateResult:
    name: GateName
    passed: bool
    detail: str


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
    sm, ld_ratio = cad.static_margin, aero.ld_cruise / sizing.assumed_ld_cruise
    group: int = req.group_constraint or sizing.dod_group
    limit: float = GROUP_UPPER_KG[group]
    reserve: float = installed_reserve(sizing, bom)
    return [
        GateResult("static_margin", SM_MIN <= sm <= SM_MAX, f"SM {sm:.3f} (band {SM_MIN}-{SM_MAX})"),
        GateResult("lift_to_drag", ld_ratio >= LD_RATIO_MIN,
                   f"VSPAERO L/D {aero.ld_cruise:.1f} = {ld_ratio:.0%} of the sizing assumption "
                   f"{sizing.assumed_ld_cruise:.1f} (min {LD_RATIO_MIN:.0%})"),
        GateResult("collisions_and_group", cad.collision_count == 0 and cad.as_built_mass_kg < limit,
                   f"{cad.collision_count} collision(s); as-built mass {cad.as_built_mass_kg:.2f} kg vs Group {group} "
                   f"limit {limit} kg"),
        GateResult("reserve", reserve >= RESERVE_MIN, f"energy reserve {reserve:.0%} (min {RESERVE_MIN:.0%})"),
    ]


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


def pipeline_gate(stage: str, message: str) -> GateResult:
    return GateResult("pipeline", False, f"{stage} failed: {message}")


def _rule_fixes(gates: list[GateResult], cad: Optional[CADArtifacts], stage: Optional[str]) -> list[Adjustment]:
    """Textbook response to each failed gate (the safety net under the LLM's choices)."""
    fixes: list[Adjustment] = []
    for gate in (g for g in gates if not g.passed):
        if gate.name == "static_margin" and cad is not None:
            direction = "increase" if cad.static_margin < SM_MIN else "decrease"
            fixes.append(Adjustment(knob="tail_volume_h", direction=direction,
                                    reason="horizontal tail volume moves the neutral point and hence the SM"))
        elif gate.name == "lift_to_drag":
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
                      assumptions: DesignAssumptions = DEFAULT_ASSUMPTIONS) -> tuple[dict[str, float], list[AppliedChange]]:
    """New overrides (a copy) plus the changes made. One move per knob. The rule table wins on every knob
    a failed gate governs (so a poor or prompt-injected LLM choice cannot push a knob the wrong way); the
    LLM may add moves on other knobs. The designer's reserve is a floor: a reserve step that would cross
    it is dropped (the critic may only raise the reserve)."""
    chosen: dict[Knob, tuple[Adjustment, Literal["llm", "rule"]]] = {}
    for adj in _rule_fixes(gates, cad, stage):
        chosen.setdefault(adj.knob, (adj, "rule"))
    for adj in llm_choices:
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


def unfixable(gates: list[GateResult], cad: Optional[CADArtifacts], stage: Optional[str]) -> list[str]:
    """Failed gates the rule table has no automatic fix for (e.g. a mass overrun, a sizing divergence)."""
    fixed_by_rule: set[str] = set()
    for gate in (g for g in gates if not g.passed):
        if _rule_fixes([gate], cad, stage):
            fixed_by_rule.add(gate.name)
    return [g.name for g in gates if not g.passed and g.name not in fixed_by_rule]


def gates_as_dicts(gates: list[GateResult]) -> list[dict[str, object]]:
    return [asdict(g) for g in gates]
