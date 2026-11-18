"""Option explorer (spec 6.1-6.3): single changes that make a failed design close, or sensitivities of a
converged one, judged with sizing + catalog only (`evaluator.evaluate`), never aero/CAD/SITL.

Search kinds: discrete levers are one evaluation; monotone continuous levers check their bound first and then
bisect between the bound and the requested value to the lever's resolution; cruise speed is not monotone, so it
is scanned then bisected at the closing transition nearest the request.

`relative_change` is |new - requested| / |requested| for continuous levers. Discrete levers (propulsion, runway,
group limit) use `DISCRETE_RELATIVE_CHANGE` = 1.0, i.e. "as big as a 100 % relaxation": they rank after any
partial continuous relaxation yet stay comparable with one that has to go all the way.
"""

import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Final, Literal, Optional

from src.agent.evaluator import Evaluation, evaluate, round_fact, with_reserve_floor
from src.schemas.assumptions import SERIES_HYBRID, DesignAssumptions
from src.schemas.changes import (FIELD_LABELS, AssumptionChanges, ChangeSet, KnobChanges, RequirementChanges,
                                 apply_changes)
from src.schemas.requirements import CUSTOM_PAYLOAD, OperationalRequirements
from src.schemas.state import DesignState

ExplorerMode = Literal["closure", "airframe", "improvement", "none"]

CLOSURE_BUDGET_S: Final[float] = 20.0
IMPROVEMENT_BUDGET_S: Final[float] = 10.0
DISCRETE_RELATIVE_CHANGE: Final[float] = 1.0
TIME_BUDGET_REASON: Final[str] = "not evaluated (time budget)"
OUT_OF_RANGE_REASON: Final[str] = "does not close within range"
CRUISE_SCAN_STEPS: Final[int] = 8
CRUISE_SCAN_FACTOR: Final[float] = 1.2  # scan [1.2 x stall, 1.2 x requested]
CRUISE_MIN_MARGIN_MS: Final[float] = 0.5  # keeps the scan strictly above the schema's exclusive minimum
VALUE_DECIMALS: Final[int] = 6  # snaps grid values (6.0 - 8 x 0.1 -> 5.2)
GROUP_MAX: Final[int] = 3
ELECTRIC: Final[str] = "electric"
AERO_CAD_STAGE: Final[str] = "aero_cad"
AIRFRAME_GATES: Final[frozenset[str]] = frozenset({"static_margin", "lift_to_drag"})
COLLISIONS_AND_GROUP_GATE: Final[str] = "collisions_and_group"
RESERVE_GATE: Final[str] = "reserve"
REFERENCE_LABEL: Final[str] = "Unchanged design"
UNCHANGED_CLOSES_NOTE: Final[str] = "Sizing + catalog close unchanged; the failure is in {where}"
BATTERY_LEVERS: Final[frozenset[str]] = frozenset({"liion_specific_energy", "liion_max_c_rate"})



def _cruise_limits() -> tuple[float, float]:
    """The requirements schema's cruise-speed bounds (exclusive minimum plus a margin, inclusive maximum)."""
    meta = OperationalRequirements.model_fields["cruise_speed_ms"].metadata
    low = next(m.gt for m in meta if getattr(m, "gt", None) is not None)
    high = next(m.le for m in meta if getattr(m, "le", None) is not None)
    return float(low) + CRUISE_MIN_MARGIN_MS, float(high)


CRUISE_MIN_MS, CRUISE_MAX_MS = _cruise_limits()
Section = Literal["requirements", "assumptions"]
Probe = Callable[[float], Optional[Evaluation]]


@dataclass(frozen=True)
class Lever:
    field: str
    section: Section
    bound: Optional[float]  # search limit; None = searched both ways (cruise speed)
    resolution: float
    improvement_step: float  # its sign is the relaxing direction


LEVERS: Final[dict[str, Lever]] = {
    "endurance": Lever("endurance_hours", "requirements", 0.5, 0.1, -1.0),
    "combat_radius": Lever("combat_radius_km", "requirements", 5.0, 1.0, -10.0),
    "payload_mass": Lever("payload_mass_kg", "requirements", 0.1, 0.1, -0.5),
    "cruise_speed": Lever("cruise_speed_ms", "requirements", None, 1.0, -2.0),
    "liion_specific_energy": Lever("liion_specific_energy_wh_kg", "assumptions", 500.0, 10.0, 50.0),
    "liion_max_c_rate": Lever("liion_max_c_rate", "assumptions", 30.0, 0.5, 2.0),
}


@dataclass(frozen=True)
class ExploreOption:
    lever: str
    change: ChangeSet
    result: Evaluation
    relative_change: float
    closes: bool
    mtow_saved_kg: Optional[float]  # improvement mode only
    group_drop: bool

    def to_json(self) -> dict[str, object]:
        return {
            "lever": self.lever, "change": self.change.model_dump(), "result": self.result.as_facts(),
            "relative_change": self.relative_change, "closes": self.closes,
            "mtow_saved_kg": round_fact(self.mtow_saved_kg), "group_drop": self.group_drop,
        }


@dataclass(frozen=True)
class _Search:
    req: OperationalRequirements
    assumptions: DesignAssumptions
    db_path: Path
    deadline: float  # time.monotonic() value

    def probe(self, change: ChangeSet) -> Optional[Evaluation]:
        """Evaluate one change set; None once the time budget is spent."""
        if time.monotonic() >= self.deadline:
            return None
        try:
            req, assumptions, knobs = apply_changes(self.req, self.assumptions, change)
        except ValueError as exc:  # pydantic ValidationError is a ValueError
            return Evaluation(False, None, None, None, None, None, f"invalid change: {exc}")
        return evaluate(req, assumptions, knobs, self.db_path)


# ------------------------------------------------------------------ mode ---
def explorer_mode(state: DesignState) -> ExplorerMode:
    """Which explorer fits why the run ended. Sizing-gate failures win over airframe ones (no airframe
    advice helps while sizing itself does not close)."""
    if state.get("fatal_error") or state.get("clarification_needed"):
        return "none"
    if state.get("converged"):
        return "improvement"
    stage_error = state.get("stage_error")
    if stage_error:
        return "airframe" if stage_error["stage"] == AERO_CAD_STAGE else "closure"
    failed = {gate["name"] for gate in state.get("gate_results") or [] if not gate["passed"]}
    cad = state.get("cad_artifacts")
    if RESERVE_GATE in failed:
        return "closure"
    if COLLISIONS_AND_GROUP_GATE in failed:
        return "airframe" if cad is not None and cad.collision_count > 0 else "closure"
    return "airframe" if failed & AIRFRAME_GATES else "closure"


def unchanged_closes(req: OperationalRequirements, assumptions: DesignAssumptions, db_path: Path) -> bool:
    """True when the design closes with sizing + catalog as it stands (default knobs, floored reserve): its
    failure then lies outside what the closure explorer can change."""
    return evaluate(req, assumptions, {}, db_path).closes


def unchanged_closes_note(state: DesignState) -> str:
    """Says where the failure is for a design that closes unchanged: the failed stage, else the failed gates."""
    stage_error = state.get("stage_error")
    failed = [gate["name"] for gate in state.get("gate_results") or [] if not gate["passed"]]
    where = f"the {stage_error['stage']} stage" if stage_error else ", ".join(failed) or "the final checks"
    return UNCHANGED_CLOSES_NOTE.format(where=where)


# --------------------------------------------------------------- helpers ---
def _change(label: str, requirements: Optional[dict] = None, assumptions: Optional[dict] = None) -> ChangeSet:
    return ChangeSet(label=label, requirements=RequirementChanges(**(requirements or {})),
                     assumptions=AssumptionChanges(**(assumptions or {})), knobs=KnobChanges())


def _lever_change(lever: Lever, value: float, req: OperationalRequirements) -> ChangeSet:
    fields: dict[str, object] = {lever.field: value}
    if lever.field == "payload_mass_kg" and req.payload_model != CUSTOM_PAYLOAD:
        fields["payload_model"] = CUSTOM_PAYLOAD  # a lighter payload is a different unit, not an edited catalogue one
    return _change(f"{FIELD_LABELS[lever.field]} → {value:g}", **{lever.section: fields})


def _lever_probe(search: _Search, lever: Lever) -> Probe:
    return lambda value: search.probe(_lever_change(lever, value, search.req))


def _current(search: _Search, lever: Lever) -> float:
    source = search.req if lever.section == "requirements" else search.assumptions
    return float(getattr(source, lever.field))


def _relative(requested: float, value: float) -> float:
    return abs(value - requested) / abs(requested)


def _option(search: _Search, lever: str, change: ChangeSet, result: Evaluation, relative: float) -> ExploreOption:
    final = with_reserve_floor(change, result, search.assumptions)
    return ExploreOption(lever, final, result, relative, result.closes, None, False)


def _unevaluated(lever: str, change: ChangeSet, relative: float) -> ExploreOption:
    result = Evaluation(False, None, None, None, None, None, TIME_BUDGET_REASON)
    return ExploreOption(lever, change, result, relative, False, None, False)


def _out_of_range(result: Evaluation) -> Evaluation:
    return replace(result, reason=f"{OUT_OF_RANGE_REASON}: {result.reason}" if result.reason else OUT_OF_RANGE_REASON)


def _bisect(probe: Probe, closing: float, closing_result: Evaluation, failing: float,
            resolution: float) -> tuple[float, Evaluation]:
    """Closing grid value (step `resolution` from `closing`) nearest `failing`, assumed not to close.
    Stops early on an exhausted budget, keeping the best closing value found so far."""
    sign = 1.0 if failing > closing else -1.0
    lo, hi = 0, math.ceil(round(abs(failing - closing) / resolution, VALUE_DECIMALS))
    best = closing_result

    def grid(step: int) -> float:
        return round(closing + sign * step * resolution, VALUE_DECIMALS)

    while hi - lo > 1:
        mid = (lo + hi) // 2
        result = probe(grid(mid))
        if result is None:
            break
        if result.closes:
            lo, best = mid, result
        else:
            hi = mid
    return grid(lo), best


# -------------------------------------------------------------- searches ---
def _search_discrete(search: _Search, lever: str, change: ChangeSet) -> ExploreOption:
    result = search.probe(change)
    if result is None:
        return _unevaluated(lever, change, DISCRETE_RELATIVE_CHANGE)
    return _option(search, lever, change, result, DISCRETE_RELATIVE_CHANGE)


def _discrete_candidates(req: OperationalRequirements) -> list[tuple[str, ChangeSet]]:
    """Propulsion flips to the other powertrain; the runway lever is skipped if already fixed-wing with runway."""
    target = ELECTRIC if req.propulsion_type == SERIES_HYBRID else SERIES_HYBRID
    candidates = [("propulsion", _change(f"Propulsion → {target}", {"propulsion_type": target}))]
    if not (req.runway_available and req.airframe_topology == "fixed_wing"):
        runway = {"runway_available": True, "airframe_topology": "fixed_wing"}
        candidates.append(("runway", _change("Allow runway, fixed-wing instead of VTOL", runway)))
    return candidates


def _search_group(search: _Search) -> list[ExploreOption]:
    """Raise the group limit. Probed at the loosest limit (sizing ignores the limit, only the gate uses it);
    a closing option names the smallest limit above the current one that still holds the sized MTOW."""
    current = search.req.group_constraint
    if current is None or current >= GROUP_MAX:
        return []
    loosest = _change(f"Group limit → {GROUP_MAX}", {"group_constraint": GROUP_MAX})
    option = _search_discrete(search, "group", loosest)
    if not option.closes or option.result.dod_group is None:
        return [option]
    group = max(option.result.dod_group, current + 1)
    change = _change(f"Group limit → {group}", {"group_constraint": group})
    return [_option(search, "group", change, option.result, DISCRETE_RELATIVE_CHANGE)]


def _search_monotone(search: _Search, name: str, lever: Lever) -> Optional[ExploreOption]:
    requested, bound = _current(search, lever), float(lever.bound)
    if (bound - requested) * lever.improvement_step <= 0:
        return None  # already at (or past) the bound
    probe = _lever_probe(search, lever)
    at_bound = probe(bound)
    if at_bound is None:
        return _unevaluated(name, _lever_change(lever, bound, search.req), _relative(requested, bound))
    if not at_bound.closes:
        return _option(search, name, _lever_change(lever, bound, search.req), _out_of_range(at_bound), _relative(requested, bound))
    value, result = _bisect(probe, bound, at_bound, requested, lever.resolution)
    return _option(search, name, _lever_change(lever, value, search.req), result, _relative(requested, value))


def _cruise_floor(req: OperationalRequirements) -> float:
    return max(CRUISE_SCAN_FACTOR * req.max_stall_speed_ms, CRUISE_MIN_MS)


def _cruise_scan_points(req: OperationalRequirements) -> list[float]:
    low = _cruise_floor(req)
    high = min(CRUISE_SCAN_FACTOR * req.cruise_speed_ms, CRUISE_MAX_MS)
    span = (high - low) / (CRUISE_SCAN_STEPS - 1)
    return sorted({round(low + i * span, VALUE_DECIMALS) for i in range(CRUISE_SCAN_STEPS)})


def _search_cruise(search: _Search, name: str, lever: Lever) -> ExploreOption:
    requested = search.req.cruise_speed_ms
    probe = _lever_probe(search, lever)
    points = _cruise_scan_points(search.req)
    scanned: list[tuple[float, Evaluation]] = []
    for value in points:
        result = probe(value)
        if result is None:
            break
        scanned.append((value, result))
    if not scanned:
        return _unevaluated(name, _lever_change(lever, points[0], search.req), _relative(requested, points[0]))
    closing = [(value, result) for value, result in scanned if result.closes]
    if not closing:
        value, result = min(scanned, key=lambda pair: abs(pair[0] - requested))
        return _option(search, name, _lever_change(lever, value, search.req), _out_of_range(result), _relative(requested, value))
    value, result = min(closing, key=lambda pair: abs(pair[0] - requested))
    # the failing end: the request, or a nearer non-closing scan point between it and the closing point
    between = [v for v, r in scanned if not r.closes and min(value, requested) < v < max(value, requested)]
    failing = min([requested, *between], key=lambda v: abs(v - value))
    value, result = _bisect(probe, value, result, failing, lever.resolution)
    return _option(search, name, _lever_change(lever, value, search.req), result, _relative(requested, value))


def explore_closure(req: OperationalRequirements, assumptions: DesignAssumptions, db_path: Path,
                    time_budget_s: float = CLOSURE_BUDGET_S) -> list[ExploreOption]:
    """Single changes that make `req` close; closing first by relative change, then non-closing. None when the
    unchanged design already closes (any option would only look like a fix; see `unchanged_closes`)."""
    search = _Search(req, assumptions, db_path, time.monotonic() + time_budget_s)
    reference = search.probe(_change(REFERENCE_LABEL))
    if reference is not None and reference.closes:
        return []
    options = [_search_discrete(search, name, change) for name, change in _discrete_candidates(req)]
    options += _search_group(search)
    for name, lever in LEVERS.items():
        option = _search_cruise(search, name, lever) if lever.bound is None else _search_monotone(search, name, lever)
        if option is not None:
            options.append(option)
    return sorted(options, key=lambda o: (not o.closes, o.relative_change if o.closes else 0.0))


# ----------------------------------------------------------- improvement ---
def _improvement_target(search: _Search, lever: Lever) -> Optional[float]:
    """Current value plus the lever's step, clipped to its limit; None if no step is possible."""
    current = _current(search, lever)
    limit = _cruise_floor(search.req) if lever.bound is None else lever.bound  # cruise: stall-based floor
    stepped = current + lever.improvement_step
    target = round(max(stepped, limit) if lever.improvement_step < 0 else min(stepped, limit), VALUE_DECIMALS)
    return target if (target - current) * lever.improvement_step > 0 else None


def _improve_lever(search: _Search, name: str, lever: Lever,
                   reference: Optional[Evaluation]) -> Optional[ExploreOption]:
    """One step of `lever`, compared against `reference` (the unchanged design through the same evaluator)."""
    is_hybrid = reference is not None and reference.propulsion == SERIES_HYBRID
    target = _improvement_target(search, lever)
    if target is None or (is_hybrid and name in BATTERY_LEVERS):  # battery constants are no-ops on a hybrid
        return None
    change, relative = _lever_change(lever, target, search.req), _relative(_current(search, lever), target)
    result = search.probe(change)
    if result is None:
        return _unevaluated(name, change, relative)
    has_reference = reference is not None and reference.mtow_kg is not None and result.mtow_kg is not None
    saved = reference.mtow_kg - result.mtow_kg if has_reference else None
    drop = has_reference and result.dod_group is not None and reference.dod_group is not None \
        and result.dod_group < reference.dod_group
    return replace(_option(search, name, change, result, relative), mtow_saved_kg=saved, group_drop=bool(drop))


def explore_improvement(req: OperationalRequirements, assumptions: DesignAssumptions, db_path: Path,
                        time_budget_s: float = IMPROVEMENT_BUDGET_S) -> list[ExploreOption]:
    """Per-lever sensitivity of a converged design: MTOW saved for one stated step, ranked by MTOW saved per
    relative change (closing first).

    Savings and group drops are measured against the unchanged design evaluated once through the same path
    (reserve floor, default knobs; counts against the budget), not against the run's own sizing: that may use
    unfloored reserves and critic-adjusted knobs, so comparing to it would mix two different evaluations."""
    search = _Search(req, assumptions, db_path, time.monotonic() + time_budget_s)
    reference = search.probe(_change(REFERENCE_LABEL))
    options = [o for name, lever in LEVERS.items() if (o := _improve_lever(search, name, lever, reference))]

    def rank(option: ExploreOption) -> tuple[bool, float]:
        saved = option.mtow_saved_kg
        return (not option.closes, -saved / option.relative_change if saved is not None else math.inf)

    return sorted(options, key=rank)
