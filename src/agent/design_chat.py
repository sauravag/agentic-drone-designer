"""Design chat engine (spec 7.2-7.3): a check-and-answer loop over structured outputs.

Each designer message triggers up to `MAX_ROUNDS` LLM rounds. In each round the LLM gets a deterministic JSON
briefing and returns a `DesignerTurn`; the app evaluates its what-ifs with the same evaluator as the option
explorer and feeds the results into the next round. The LLM never computes a number: every result comes from
the solver, and numbers in the final message that appear nowhere in the briefing or results are flagged.
A proposal is only validated here, never run: the designer must accept it and press Re-run.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, Optional, Sequence

from pydantic import BaseModel, ValidationError

from src.agent.agent_schemas import MAX_WHAT_IFS, DesignerTurn
from src.agent.evaluator import Evaluation, evaluate
from src.agent.llm import StructuredLLM
from src.agent.prompts import DESIGNER_SYSTEM
from src.agent.report import stray_numbers
from src.schemas.assumptions import DesignAssumptions
from src.schemas.changes import FIELD_LABELS, AssumptionChanges, ChangeSet, KnobChanges, RequirementChanges, \
    apply_changes
from src.schemas.knobs import OVERRIDE_BOUNDS
from src.schemas.requirements import OperationalRequirements

MAX_ROUNDS: Final[int] = 3
HISTORY_BUDGET: Final[int] = 12  # most recent chat messages included in the briefing
BOUND_KEYS: Final[tuple[str, ...]] = ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "enum")
WHOLE_NUMBER: Final[re.Pattern[str]] = re.compile(r"\b(\d+)\.0\b")
RULES: Final[tuple[str, ...]] = (
    "Do no arithmetic and never estimate a number.",
    "Only mention numbers that appear in this briefing or in what_if_results, copied exactly.",
    "Never claim that a re-run happened: a proposal only fills the designer's review panel.",
    f"At most {MAX_WHAT_IFS} what-ifs per round; change only the allowed fields, within their bounds.",
    "Results are checked with sizing and catalog only; aerodynamics, CAD and SITL may still find issues.",
)
TRUNCATED_NOTE: Final[str] = "(Only the first {limit} what-ifs of a round were checked.)"
DROPPED_NOTE: Final[str] = "(Proposal '{label}' was dropped: {error})"
KNOBS_REFUSED: Final[str] = "knob changes ({names}) are only allowed after an airframe failure"


@dataclass(frozen=True)
class WhatIfResult:
    change: ChangeSet
    result: Optional[Evaluation]
    error: Optional[str]

    def to_json(self) -> dict[str, object]:
        return {"change": self.change.model_dump(mode="json", exclude_none=True),
                "result": None if self.result is None else self.result.as_facts(), "error": self.error}


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


@dataclass(frozen=True)
class _Rounds:
    turn: DesignerTurn
    results: tuple[WhatIfResult, ...]
    briefing: str  # the last round's briefing
    count: int
    truncated: bool


def _bound(prop: dict[str, Any]) -> dict[str, Any]:
    """Bounds of one JSON-schema property; Optional fields keep the bounds of their non-null branch."""
    branches = [b for b in prop.get("anyOf", [prop]) if b.get("type") != "null"]
    bounds = {key: b[key] for b in branches for key in BOUND_KEYS if key in b}
    return bounds or {"type": branches[0].get("type")}


def _model_bounds(model: type[BaseModel], names: Sequence[str]) -> dict[str, dict[str, Any]]:
    props: dict[str, Any] = model.model_json_schema()["properties"]
    return {name: _bound(props[name]) for name in names}


def _allowed_fields(allow_knobs: bool) -> dict[str, dict[str, dict[str, Any]]]:
    allowed = {"requirements": _model_bounds(OperationalRequirements, list(RequirementChanges.model_fields)),
               "assumptions": _model_bounds(DesignAssumptions, list(AssumptionChanges.model_fields))}
    if allow_knobs:
        allowed["knobs"] = {name: {"minimum": OVERRIDE_BOUNDS[name][0], "maximum": OVERRIDE_BOUNDS[name][1]}
                            for name in KnobChanges.model_fields}
    return allowed


def _briefing(ctx: ChatContext, history: tuple[ChatMessage, ...], message: str,
              results: tuple[WhatIfResult, ...]) -> str:
    """The round's JSON briefing (spec 7.2); sorted keys keep it deterministic."""
    data = {
        "requirements": ctx.requirements.model_dump(mode="json"), "assumptions": ctx.assumptions.model_dump(),
        "failed_gates": list(ctx.failed_gates), "options": list(ctx.options),
        "history": [m.to_json() for m in history[-HISTORY_BUDGET:]], "designer_message": message,
        "what_if_results": [r.to_json() for r in results], "allowed_fields": _allowed_fields(ctx.allow_knobs),
        "rules": list(RULES),
    }
    return json.dumps(data, sort_keys=True, default=str)


def _concise(exc: ValidationError) -> str:
    """One short line per failed field (no pydantic URLs or input dumps)."""
    parts = [f"{FIELD_LABELS.get(str(err['loc'][0]), err['loc'][0]) if err['loc'] else 'design'}: {err['msg']}"
             for err in exc.errors()]
    return "; ".join(parts)


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


def _what_if(ctx: ChatContext, change: ChangeSet) -> WhatIfResult:
    try:
        req, assumptions, knobs = _validate(ctx, change)
    except ValueError as exc:
        return WhatIfResult(change, None, str(exc))
    return WhatIfResult(change, evaluate(req, assumptions, knobs, ctx.db_path), None)


def _run_rounds(llm: StructuredLLM, ctx: ChatContext, history: tuple[ChatMessage, ...], message: str) -> _Rounds:
    results: tuple[WhatIfResult, ...] = ()
    truncated = False
    for count in range(1, MAX_ROUNDS + 1):
        briefing = _briefing(ctx, history, message, results)
        turn: DesignerTurn = llm.parse(DESIGNER_SYSTEM, briefing, DesignerTurn)  # LLMError propagates
        checks = turn.what_ifs[:MAX_WHAT_IFS]
        truncated = truncated or len(turn.what_ifs) > MAX_WHAT_IFS
        results += tuple(_what_if(ctx, change) for change in checks)
        if turn.done or not checks:
            break
    return _Rounds(turn, results, briefing, count, truncated)


def _facts_text(rounds: _Rounds, proposal: Optional[ChangeSet]) -> str:
    """Everything the LLM may cite; '6.0' also counts as '6' so whole numbers can be written naturally."""
    extra = [r.to_json() for r in rounds.results]
    if proposal is not None:
        extra.append({"proposal": proposal.model_dump(mode="json", exclude_none=True)})
    text = f"{rounds.briefing}\n{json.dumps(extra, sort_keys=True, default=str)}"
    return WHOLE_NUMBER.sub(r"\1.0 \1", text)


def _checked_proposal(ctx: ChatContext, proposal: Optional[ChangeSet]) -> tuple[Optional[ChangeSet], Optional[str]]:
    """The proposal if it validates, else (None, why it was dropped)."""
    if proposal is None:
        return None, None
    try:
        _validate(ctx, proposal)
    except ValueError as exc:
        return None, DROPPED_NOTE.format(label=proposal.label, error=exc)
    return proposal, None


def answer(llm: StructuredLLM, ctx: ChatContext, history: tuple[ChatMessage, ...], message: str) -> ChatMessage:
    """Answer one designer message after checking the LLM's what-ifs with the solver (up to MAX_ROUNDS)."""
    rounds = _run_rounds(llm, ctx, history, message)
    proposal, dropped = _checked_proposal(ctx, rounds.turn.proposal)
    notes = [TRUNCATED_NOTE.format(limit=MAX_WHAT_IFS)] if rounds.truncated else []
    notes += [dropped] if dropped else []
    strays = stray_numbers(rounds.turn.message, _facts_text(rounds, proposal))
    return ChatMessage(role="agent", text="\n\n".join([rounds.turn.message, *notes]), what_ifs=rounds.results,
                       proposal=proposal, stray_numbers=tuple(strays), rounds=rounds.count)
