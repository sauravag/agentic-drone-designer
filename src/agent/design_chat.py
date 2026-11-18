"""Design chat engine (spec 7.2-7.3): a check-and-answer loop over structured outputs.

Each designer message triggers up to `MAX_ROUNDS` LLM rounds. In each round the LLM gets a deterministic JSON
briefing and returns a `DesignerTurn`; the app converts its flat `ProposedChange`s to `ChangeSet`s, evaluates the
what-ifs with the same evaluator as the option explorer and feeds the results into the next round. The LLM never
computes a number: every result comes from the solver, and numbers in the shown text (message and change labels)
that appear in no deterministic fact are flagged. Earlier chat text and the designer's message are not facts, so
an invented number cannot become "known" by being repeated. A proposal is only validated here, never run: the
designer must accept it and press Re-run.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, Optional, Sequence

from pydantic import BaseModel, ValidationError

from src.agent.agent_schemas import MAX_WHAT_IFS, DesignerTurn, ProposedChange
from src.agent.evaluator import Evaluation, evaluate, with_reserve_floor
from src.agent.llm import StructuredLLM
from src.agent.prompts import DESIGNER_SYSTEM
from src.agent.report import citable, stray_numbers
from src.schemas.assumptions import DesignAssumptions
from src.schemas.changes import FIELD_LABELS, LABEL_MAX_LEN, AssumptionChanges, ChangeSet, KnobChanges, \
    RequirementChanges, apply_changes
from src.schemas.knobs import OVERRIDE_BOUNDS
from src.schemas.payload_catalog import payload_choices, resolve_payload_name
from src.schemas.requirements import OperationalRequirements

MAX_ROUNDS: Final[int] = 3
HISTORY_BUDGET: Final[int] = 12  # most recent chat messages included in the briefing
BOUND_KEYS: Final[tuple[str, ...]] = ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "enum")
WHOLE_NUMBER: Final[re.Pattern[str]] = re.compile(r"\b(\d+)\.0\b")
SECTIONS: Final[dict[str, type[BaseModel]]] = {
    "requirements": RequirementChanges, "assumptions": AssumptionChanges, "knobs": KnobChanges}
FALLBACK_LABEL: Final[str] = "Chat change"
RULES: Final[tuple[str, ...]] = (
    "Do no arithmetic and never estimate a number.",
    "Only mention numbers from this briefing's design data or what-if results, copied exactly; never numbers "
    "from designer_message or the history text.",
    "Never claim that a re-run happened: a proposal only fills the designer's review panel.",
    f"At most {MAX_WHAT_IFS} what-ifs per round; change only the allowed fields, within their bounds.",
    "Results are checked with sizing and catalog only; aerodynamics, CAD and SITL may still find issues.",
    "If final_round is true, answer now: what-ifs you add will be checked but you will not see the results.",
)
TRUNCATED_NOTE: Final[str] = "(Only the first {limit} what-ifs of a round were checked.)"
ROUND_LIMIT_NOTE: Final[str] = "(Round limit reached; the what-if results shown are from the solver.)"
LATE_WHAT_IFS_NOTE: Final[str] = "(The what-ifs below were checked after this answer was written.)"
DROPPED_NOTE: Final[str] = "(Proposal '{label}' was dropped: {error})"
KNOBS_REFUSED: Final[str] = "knob changes ({names}) are only allowed after an airframe failure"
DUPLICATE_FIELDS: Final[str] = "a field may change only once: {names}"


def _int_choice_fields() -> frozenset[str]:
    """Fields whose allowed values are integers (e.g. group_constraint): the LLM's "2" must become 2."""
    names: set[str] = set()
    for model in SECTIONS.values():
        for name, prop in model.model_json_schema()["properties"].items():
            if any(all(isinstance(v, int) for v in b.get("enum", [None])) for b in prop.get("anyOf", [prop])):
                names.add(name)
    return frozenset(names)


INT_CHOICE_FIELDS: Final[frozenset[str]] = _int_choice_fields()


@dataclass(frozen=True)
class WhatIfResult:
    change: ChangeSet
    result: Optional[Evaluation]
    error: Optional[str]

    def facts(self) -> dict[str, object]:
        """The deterministic part: requested values, solver result, validation error (no LLM-written label)."""
        change = self.change.model_dump(mode="json", exclude_none=True, exclude={"label"})
        return {"change": change, "result": None if self.result is None else self.result.as_facts(),
                "error": self.error}

    def to_json(self) -> dict[str, object]:
        return {"label": self.change.label, **self.facts()}


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["designer", "agent"]
    text: str
    what_ifs: tuple[WhatIfResult, ...] = ()
    proposal: Optional[ChangeSet] = None
    stray_numbers: tuple[str, ...] = ()
    rounds: int = 0

    def to_json(self) -> dict[str, object]:
        return {"role": self.role, "text": self.text, "what_if_results": [w.to_json() for w in self.what_ifs]}


@dataclass(frozen=True)
class ChatContext:
    requirements: OperationalRequirements
    assumptions: DesignAssumptions
    failed_gates: tuple[str, ...]
    options: tuple[dict[str, Any], ...]
    allow_knobs: bool  # true only after an airframe failure (spec 6.1)
    db_path: Path
    critic_summary: tuple[str, ...] = ()  # one line per critic pass


@dataclass(frozen=True)
class _Rounds:
    turn: DesignerTurn
    results: tuple[WhatIfResult, ...]
    count: int
    truncated: bool
    late: bool  # the final turn sent what-ifs together with its answer (done=True)


def _concise(exc: ValidationError) -> str:
    """One short line per failed field (no pydantic URLs or input dumps)."""
    def where(loc: tuple[Any, ...]) -> str:
        named = [FIELD_LABELS[str(p)] for p in loc if str(p) in FIELD_LABELS]
        return named[0] if named else ".".join(str(p) for p in loc) or "design"
    return "; ".join(f"{where(err['loc'])}: {err['msg']}" for err in exc.errors())


def _label(text: str) -> str:
    """Label rules applied in the app, never as a schema failure: trimmed, never empty, at most LABEL_MAX_LEN."""
    return (" ".join(text.split()) or FALLBACK_LABEL)[:LABEL_MAX_LEN]


def _value(name: str, value: str) -> object:
    text = value.strip()
    if name == "payload_model":  # 'zt30' -> 'SIYI ZT30'; Custom and unknown names pass through to validation
        return resolve_payload_name(text) or text
    return int(text) if name in INT_CHOICE_FIELDS and text.isdigit() else text


def to_change_set(proposed: ProposedChange) -> ChangeSet:
    """The `ChangeSet` an LLM-written change stands for; raises ValueError with a concise reason.

    Values are coerced by pydantic's lax mode ("4.5" -> 4.5, "true" -> True); bounds are checked later."""
    names = [c.field for c in proposed.changes]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(DUPLICATE_FIELDS.format(names=", ".join(FIELD_LABELS[n] for n in duplicates)))
    data = {section: {c.field: _value(c.field, c.value) for c in proposed.changes if c.field in model.model_fields}
            for section, model in SECTIONS.items()}
    try:
        return ChangeSet.model_validate({"label": _label(proposed.label), **data})
    except ValidationError as exc:
        raise ValueError(_concise(exc)) from exc


def _bound(prop: dict[str, Any]) -> dict[str, Any]:
    """Bounds of one JSON-schema property; Optional fields keep the bounds of their non-null branch."""
    branches = [b for b in prop.get("anyOf", [prop]) if b.get("type") != "null"]
    bounds = {key: b[key] for b in branches for key in BOUND_KEYS if key in b}
    return bounds or {"type": branches[0].get("type")}


def _model_bounds(model: type[BaseModel], names: Sequence[str]) -> dict[str, dict[str, Any]]:
    props: dict[str, Any] = model.model_json_schema()["properties"]
    return {name: _bound(props[name]) for name in names}


def _allowed_fields(allow_knobs: bool) -> dict[str, dict[str, dict[str, Any]]]:
    requirements = _model_bounds(OperationalRequirements, list(RequirementChanges.model_fields))
    requirements["payload_model"] = {"enum": list(payload_choices())}  # the catalogue, so the LLM can pick a name
    allowed = {"requirements": requirements,
               "assumptions": _model_bounds(DesignAssumptions, list(AssumptionChanges.model_fields))}
    if allow_knobs:
        allowed["knobs"] = {name: {"minimum": OVERRIDE_BOUNDS[name][0], "maximum": OVERRIDE_BOUNDS[name][1]}
                            for name in KnobChanges.model_fields}
    return allowed


def _design_data(ctx: ChatContext) -> dict[str, object]:
    """The deterministic design context (tool output and configuration only)."""
    return {
        "requirements": ctx.requirements.to_context_json(), "assumptions": ctx.assumptions.model_dump(),
        "failed_gates": list(ctx.failed_gates), "critic_summary": list(ctx.critic_summary),
        "options": list(ctx.options), "allowed_fields": _allowed_fields(ctx.allow_knobs), "rules": list(RULES),
    }


def _briefing(ctx: ChatContext, history: tuple[ChatMessage, ...], message: str,
              results: tuple[WhatIfResult, ...], final_round: bool) -> str:
    """The round's JSON briefing (spec 7.2); sorted keys keep it deterministic."""
    data = {**_design_data(ctx), "history": [m.to_json() for m in history[-HISTORY_BUDGET:]],
            "designer_message": message, "what_if_results": [r.to_json() for r in results],
            "final_round": final_round}
    return json.dumps(data, sort_keys=True, default=str)


def _validate(ctx: ChatContext, change: ChangeSet) -> tuple[OperationalRequirements, DesignAssumptions,
                                                             dict[str, float]]:
    """The changed design inputs; raises ValueError with a concise reason (never clamps)."""
    knobs = change.knobs.model_dump(exclude_none=True)
    if knobs and not ctx.allow_knobs:
        raise ValueError(KNOBS_REFUSED.format(names=", ".join(FIELD_LABELS[name] for name in knobs)))
    try:
        return apply_changes(ctx.requirements, ctx.assumptions, change)
    except ValidationError as exc:  # requirement/assumption bounds; knob bounds raise a plain ValueError
        raise ValueError(_concise(exc)) from exc


def _unchanged(label: str) -> ChangeSet:
    """Stand-in change for a what-if that could not be converted (its error says why)."""
    return ChangeSet(label=_label(label), requirements=RequirementChanges(), assumptions=AssumptionChanges(),
                     knobs=KnobChanges())


def _what_if(ctx: ChatContext, proposed: ProposedChange) -> WhatIfResult:
    try:
        change = to_change_set(proposed)
    except ValueError as exc:
        return WhatIfResult(_unchanged(proposed.label), None, str(exc))
    try:
        req, assumptions, knobs = _validate(ctx, change)
    except ValueError as exc:
        return WhatIfResult(change, None, str(exc))
    return WhatIfResult(change, evaluate(req, assumptions, knobs, ctx.db_path), None)


def _run_rounds(llm: StructuredLLM, ctx: ChatContext, history: tuple[ChatMessage, ...], message: str) -> _Rounds:
    results: tuple[WhatIfResult, ...] = ()
    truncated = False
    for count in range(1, MAX_ROUNDS + 1):
        briefing = _briefing(ctx, history, message, results, final_round=count == MAX_ROUNDS)
        turn: DesignerTurn = llm.parse(DESIGNER_SYSTEM, briefing, DesignerTurn)  # LLMError propagates
        checks = turn.what_ifs[:MAX_WHAT_IFS]
        truncated = truncated or len(turn.what_ifs) > MAX_WHAT_IFS
        results += tuple(_what_if(ctx, proposed) for proposed in checks)
        if turn.done or not checks:
            break
    return _Rounds(turn, results, count, truncated, late=turn.done and bool(checks))


def _facts_text(ctx: ChatContext, history: tuple[ChatMessage, ...], results: tuple[WhatIfResult, ...],
                proposal: Optional[ChangeSet]) -> str:
    """Everything the LLM may cite: deterministic data only (never chat text, the designer's message or labels).
    '6.0' also counts as '6' so whole numbers can be written naturally."""
    earlier = [w.facts() for m in history[-HISTORY_BUDGET:] for w in m.what_ifs]
    data = {**_design_data(ctx), "earlier_what_ifs": earlier, "what_ifs": [r.facts() for r in results],
            "proposal": None if proposal is None else proposal.model_dump(mode="json", exclude={"label"})}
    return WHOLE_NUMBER.sub(r"\1.0 \1", citable(json.dumps(data, sort_keys=True, default=str)))


def _proposal_result(ctx: ChatContext, change: ChangeSet, results: tuple[WhatIfResult, ...]) -> Evaluation:
    """The solver result for a validated proposal: a what-if of the same change if one was checked, else a new
    evaluation."""
    for what_if in results:
        if what_if.result is not None and what_if.change.model_copy(update={"label": change.label}) == change:
            return what_if.result
    return evaluate(*_validate(ctx, change), ctx.db_path)


def _checked_proposal(ctx: ChatContext, proposed: Optional[ProposedChange],
                      results: tuple[WhatIfResult, ...]) -> tuple[Optional[ChangeSet], Optional[str]]:
    """The proposal as a validated ChangeSet carrying the reserve floor it was checked with (spec 6.2), else
    (None, why it was dropped)."""
    if proposed is None:
        return None, None
    try:
        change = to_change_set(proposed)
        _validate(ctx, change)
    except ValueError as exc:
        return None, DROPPED_NOTE.format(label=_label(proposed.label), error=exc)
    return with_reserve_floor(change, _proposal_result(ctx, change, results), ctx.assumptions), None


def _notes(rounds: _Rounds, dropped: Optional[str]) -> list[str]:
    notes = [TRUNCATED_NOTE.format(limit=MAX_WHAT_IFS)] if rounds.truncated else []
    notes += [ROUND_LIMIT_NOTE] if rounds.count == MAX_ROUNDS and not rounds.turn.done else []
    notes += [LATE_WHAT_IFS_NOTE] if rounds.late else []
    return notes + ([dropped] if dropped else [])


def answer(llm: StructuredLLM, ctx: ChatContext, history: tuple[ChatMessage, ...], message: str) -> ChatMessage:
    """Answer one designer message after checking the LLM's what-ifs with the solver (up to MAX_ROUNDS)."""
    rounds = _run_rounds(llm, ctx, history, message)
    proposal, dropped = _checked_proposal(ctx, rounds.turn.proposal, rounds.results)
    proposed = rounds.turn.proposal
    shown = [rounds.turn.message, *(r.change.label for r in rounds.results), *([proposed.label] if proposed else [])]
    strays = stray_numbers("\n".join(shown), _facts_text(ctx, history, rounds.results, proposal))
    return ChatMessage(role="agent", text="\n\n".join([rounds.turn.message, *_notes(rounds, dropped)]),
                       what_ifs=rounds.results, proposal=proposal, stray_numbers=tuple(strays), rounds=rounds.count)
