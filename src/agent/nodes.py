"""LangGraph node functions (SRS 5). Tool nodes run the deterministic modules and write their artifacts to
outputs/<run_id>/; LLM nodes go through the StructuredLLM protocol. A design error in a tool node does
not crash the run: it is recorded as `stage_error` and handed to the critic, like a failed gate.
"""

import json
import logging
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Final, Optional

from pydantic import BaseModel

from src import pipeline
from src.agent import critic, report
from src.agent.agent_schemas import CriticDecision, OptionsExplanation, ReportNarrative, RequirementsDraft
from src.agent.llm import LLMError, StructuredLLM
from src.agent.parsing import to_requirements
from src.agent.explorer import (ExploreOption, explore_closure, explorer_mode, unchanged_closes,
                                unchanged_closes_note)
from src.agent.prompts import CRITIC_SYSTEM, OPTIONS_SYSTEM, PARSER_SYSTEM, REPORT_SYSTEM
from src.schemas.assumptions import DEFAULT_ASSUMPTIONS, DesignAssumptions
from src.schemas.requirements import OperationalRequirements
from src.schemas.state import DesignState
from src.tools.database.matcher import select_bom
from src.tools.database.seed_data import DEFAULT_DB_PATH
from src.tools.sizing.mission_solver import component_needs, size_aircraft
from src.viewer.build import viewer_warning

logger = logging.getLogger(__name__)

MAX_ITERATIONS: Final[int] = 8  # SRS hard cap on critic loops
Update = dict[str, Any]


class AgentStateError(RuntimeError):
    """A node ran without the state an earlier node should have produced (a routing bug)."""


@dataclass(frozen=True)
class AgentContext:
    llm: Optional[StructuredLLM]
    outputs_root: Path = pipeline.DEFAULT_OUTPUTS_ROOT
    db_path: Path = DEFAULT_DB_PATH
    with_aero_cad: bool = True
    with_sitl: bool = True
    max_iterations: int = MAX_ITERATIONS
    progress: Callable[[str], None] = field(default=lambda message: None)


def _require(state: DesignState, *keys: str) -> list[Any]:
    values: list[Any] = [state.get(key) for key in keys]  # type: ignore[misc]
    missing: list[str] = [key for key, value in zip(keys, values) if value is None]
    if missing:
        raise AgentStateError(f"node reached without {missing}")
    return values


def _write(state: DesignState, name: str, payload: BaseModel | list | dict) -> None:
    (out_dir,) = _require(state, "out_dir")
    data = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    (Path(out_dir) / name).write_text(json.dumps(data, indent=2))


def _tool(ctx: AgentContext, message: str) -> None:
    logger.info("[Tool] %s", message)
    ctx.progress(message)


# ------------------------------------------------------------------ parse ---
def parse_requirements(state: DesignState, ctx: AgentContext) -> Update:
    """LLM -> RequirementsDraft -> deterministic SI conversion. Structured requirements skip the LLM."""
    pipeline.ensure_catalog(ctx.db_path)
    req: Optional[OperationalRequirements] = state.get("requirements")
    assumptions: tuple[str, ...] = ()
    if req is None:
        if ctx.llm is None:
            return {"fatal_error": "No LLM configured to parse a natural-language request."}
        _tool(ctx, "parse_requirements: extracting the mission requirements")
        try:
            draft: RequirementsDraft = ctx.llm.parse(PARSER_SYSTEM, state["user_prompt"], RequirementsDraft)
        except LLMError as exc:
            return {"fatal_error": f"Could not parse the request: {exc}"}
        outcome = to_requirements(draft, ctx.db_path)
        if outcome.requirements is None:
            return {"clarification_needed": outcome.clarification}
        req, assumptions = outcome.requirements, outcome.assumptions
    run_id, out_dir = pipeline.create_run_dir(ctx.outputs_root, pipeline.new_run_id(req.mission_name))
    update: Update = {"requirements": req, "run_id": run_id, "out_dir": str(out_dir),
                      "assumptions": list(assumptions), "clarification_needed": None}
    _write(update, "requirements.json", req)  # type: ignore[arg-type]
    _write(update, "assumptions.json", state.get("design_assumptions") or DEFAULT_ASSUMPTIONS)  # type: ignore[arg-type]
    return update


# ------------------------------------------------------------- tool nodes ---
def _assumptions(state: DesignState) -> DesignAssumptions:
    return state.get("design_assumptions") or DEFAULT_ASSUMPTIONS


def run_sizing(state: DesignState, ctx: AgentContext) -> Update:
    (req,) = _require(state, "requirements")
    overrides: dict[str, float] = state.get("sizing_overrides") or {}
    _tool(ctx, f"run_sizing (iteration {state.get('iteration_count', 0)}, overrides {overrides or 'none'})")
    cleared: Update = {"selected_bom": None, "aero_stability": None, "cad_artifacts": None, "fcs_artifacts": None,
                       "stage_error": None, "sitl_error": None}
    try:
        sizing = size_aircraft(req, overrides, _assumptions(state))
    except pipeline.DESIGN_ERRORS as exc:
        return cleared | {"sizing_results": None, "stage_error": {"stage": "sizing", "message": str(exc)}}
    _write(state, "sizing.json", sizing)
    return cleared | {"sizing_results": sizing}


def match_cots(state: DesignState, ctx: AgentContext) -> Update:
    req, sizing = _require(state, "requirements", "sizing_results")
    _tool(ctx, "match_cots: selecting catalog components")
    try:
        bom = select_bom(ctx.db_path, component_needs(sizing, req), _assumptions(state))
    except pipeline.DESIGN_ERRORS as exc:
        return {"stage_error": {"stage": "cots", "message": str(exc)}}
    _write(state, "bom.json", bom)
    return {"selected_bom": bom}


def build_aero_cad(state: DesignState, ctx: AgentContext) -> Update:
    if not ctx.with_aero_cad:
        return {}
    req, sizing, bom, out_dir = _require(state, "requirements", "sizing_results", "selected_bom", "out_dir")
    try:
        aero, cad = pipeline.run_aero_cad(sizing, bom, req, Path(out_dir), ctx.progress)
    except pipeline.DESIGN_ERRORS as exc:
        return {"stage_error": {"stage": "aero_cad", "message": str(exc)}}
    _write(state, "aero.json", aero)
    _write(state, "cad.json", cad)
    return {"aero_stability": aero, "cad_artifacts": cad}


def generate_and_verify_fcs(state: DesignState, ctx: AgentContext) -> Update:
    req, sizing, bom, aero, cad, out_dir = _require(state, "requirements", "sizing_results", "selected_bom",
                                                    "aero_stability", "cad_artifacts", "out_dir")
    try:
        fcs = pipeline.run_sitl(req, sizing, bom, aero, cad, Path(out_dir), ctx.progress)
    except pipeline.DESIGN_ERRORS as exc:
        return {"sitl_error": f"flight control / SITL could not complete: {exc}"}
    _write(state, "fcs.json", fcs)
    return {"fcs_artifacts": fcs}


# ----------------------------------------------------------------- critic ---
def _gates(state: DesignState) -> list[critic.GateResult]:
    stage_error = state.get("stage_error")
    if stage_error:
        return [critic.pipeline_gate(stage_error["stage"], stage_error["message"])]
    req, sizing, bom = _require(state, "requirements", "sizing_results", "selected_bom")
    aero, cad = state.get("aero_stability"), state.get("cad_artifacts")
    if aero is None or cad is None:
        return critic.evaluate_sizing_gates(req, sizing, bom)
    return critic.evaluate_gates(req, sizing, bom, aero, cad)


def _critic_prompt(state: DesignState, failed: list[critic.GateResult]) -> str:
    sizing = state.get("sizing_results")
    design: str = (f"{sizing.airframe_topology}, {sizing.propulsion_type}, MTOW {sizing.mtow_kg:.2f} kg, "
                   f"AR {sizing.aspect_ratio:.1f}") if sizing else "not sized"
    lines: list[str] = [f"Design: {design}", f"Current overrides: {state.get('sizing_overrides') or 'none'}",
                        "Failed gates:"] + [f"- {g.name}: {g.detail}" for g in failed]
    return "\n".join(lines)


def _llm_choices(state: DesignState, failed: list[critic.GateResult], ctx: AgentContext) -> CriticDecision:
    if ctx.llm is None:
        return CriticDecision(summary="rule-table adjustments only (no LLM)")
    try:
        return ctx.llm.parse(CRITIC_SYSTEM, _critic_prompt(state, failed), CriticDecision)
    except LLMError as exc:
        logger.warning("critic LLM unavailable, using the rule table: %s", exc)
        return CriticDecision(summary=f"rule-table adjustments only (LLM error: {exc})")


def design_critic(state: DesignState, ctx: AgentContext) -> Update:
    gates = _gates(state)
    failed = [g for g in gates if not g.passed]
    iteration: int = state.get("iteration_count", 0)
    entry: dict[str, Any] = {"iteration": iteration, "gates": critic.gates_as_dicts(gates),
                             "failed": [g.name for g in failed], "changes": [], "summary": ""}
    update: Update = {"gate_results": critic.gates_as_dicts(gates), "validation_errors": [g.detail for g in failed]}
    if not failed:
        update |= {"converged": True}
    elif iteration >= ctx.max_iterations:
        update |= {"critic_exhausted": True}
        entry["summary"] = f"iteration cap ({ctx.max_iterations}) reached"
    else:
        update |= _adjust(state, gates, failed, entry, ctx)
    _tool(ctx, f"design_critic: iteration {iteration}, failed gates {entry['failed'] or 'none'}")
    log: list[dict[str, Any]] = [*state.get("critic_log", []), entry]
    _write(state, "critic_log.json", log)
    return update | {"critic_log": log}


def _propulsion(state: DesignState) -> str:
    sizing, req = state.get("sizing_results"), state.get("requirements")
    if sizing is not None:
        return sizing.propulsion_type
    return req.propulsion_type if req is not None and req.propulsion_type != "auto" else "electric"


def _adjust(state: DesignState, gates: list[critic.GateResult], failed: list[critic.GateResult],
            entry: dict[str, Any], ctx: AgentContext) -> Update:
    decision = _llm_choices(state, failed, ctx)
    stage = (state.get("stage_error") or {}).get("stage")
    cad = state.get("cad_artifacts")
    overrides, changes = critic.apply_adjustments(state.get("sizing_overrides") or {}, _propulsion(state),
                                                  decision.adjustments, gates, cad, stage,
                                                  _assumptions(state))
    stuck: list[str] = critic.unfixable(gates, cad, stage)
    entry["summary"] = decision.summary + (f" No automatic fix for: {', '.join(stuck)}." if stuck else "")
    entry["changes"] = [f"{c.knob} {c.old:.3g} -> {c.new:.3g} ({c.source}: {c.reason})" for c in changes]
    entry["applied"] = [asdict(c) for c in changes]
    if not changes:
        return {"critic_exhausted": True}
    return {"sizing_overrides": overrides, "iteration_count": state.get("iteration_count", 0) + 1}


# ----------------------------------------------------------------- report ---
def sitl_passed(state: DesignState) -> Optional[bool]:
    """True/False once the SITL test ran (an aborted test counts as failed); None if it never ran."""
    if state.get("sitl_error"):
        return False
    fcs = state.get("fcs_artifacts")
    return None if fcs is None else fcs.passed


def run_status(state: DesignState) -> str:
    if not state.get("converged"):
        return "not_converged"
    return "converged_sitl_failed" if sitl_passed(state) is False else "converged"


def run_warnings(state: DesignState) -> list[str]:
    """Everything the report (and the web form) must flag, most serious first."""
    sizing, aero, cad, fcs = (state.get("sizing_results"), state.get("aero_stability"),
                              state.get("cad_artifacts"), state.get("fcs_artifacts"))
    warnings: list[str] = list(pipeline.design_warnings(sizing, aero, cad, fcs)) if sizing else []
    if state.get("requirements") is not None:
        warnings += pipeline.payload_warnings(state["requirements"], state.get("selected_bom"))
    if not state.get("converged"):
        warnings.insert(0, "NON-CONVERGENCE: the design did not pass every critic gate: "
                           + "; ".join(state.get("validation_errors", [])))
    if state.get("sitl_error"):
        warnings.insert(0, state["sitl_error"])
    if state.get("viewer_error"):
        warnings.append(state["viewer_error"])
    return warnings


def _build_viewer(state: DesignState) -> DesignState:
    """Write viewer.html when the CAD stage exported a mesh; a failure becomes a run warning."""
    if not state.get("out_dir"):
        return state
    return {**state, "viewer_error": viewer_warning(Path(state["out_dir"]))}


def _report_inputs(state: DesignState) -> report.ReportInputs:
    req, run_id, out_dir = _require(state, "requirements", "run_id", "out_dir")
    sizing, aero, cad, fcs = (state.get("sizing_results"), state.get("aero_stability"),
                              state.get("cad_artifacts"), state.get("fcs_artifacts"))
    warnings: list[str] = run_warnings(state)
    return report.ReportInputs(
        run_id=run_id, out_dir=Path(out_dir), requirements=req, sizing=sizing, bom=state.get("selected_bom"),
        aero=aero, cad=cad, fcs=fcs, converged=bool(state.get("converged")),
        iterations=state.get("iteration_count", 0), final_gates=state.get("gate_results", []),
        critic_log=state.get("critic_log", []), assumptions=state.get("assumptions", []), design_assumptions=_assumptions(state),
        warnings=tuple(warnings), lineage=state.get("lineage"))


def _options_facts(options: list[ExploreOption], state: DesignState) -> str:
    return json.dumps({"failure": state.get("validation_errors", []),
                       "options": [option.to_json() for option in options]}, indent=1)


def _explain_options(ctx: AgentContext, facts: str) -> Optional[OptionsExplanation]:
    """The optional LLM prose; an outage costs the explanation, never the run."""
    if ctx.llm is None:
        return None
    try:
        return ctx.llm.parse(OPTIONS_SYSTEM, facts, OptionsExplanation)
    except LLMError as exc:
        logger.warning("options explanation unavailable: %s", exc)
        return None


def compile_report(state: DesignState, ctx: AgentContext) -> Update:
    state = _build_viewer(state)  # before the report inputs, so the report sees any viewer warning
    inputs = _report_inputs(state)
    mode = explorer_mode(state)
    options: list[ExploreOption] = []
    explanation: Optional[OptionsExplanation] = None
    note: Optional[str] = None
    if mode == "closure" and unchanged_closes(inputs.requirements, inputs.design_assumptions, ctx.db_path):
        mode, note = "airframe", unchanged_closes_note(state)  # no closure option would be a real fix
        inputs = replace(inputs, explorer_note=note)
    if mode == "closure":
        _tool(ctx, "compile_report: exploring options that close the design")
        options = explore_closure(inputs.requirements, inputs.design_assumptions, ctx.db_path)
        _write(state, "options.json", [option.to_json() for option in options])
        facts = _options_facts(options, state)
        explanation = _explain_options(ctx, facts) if options else None
        inputs = replace(inputs, options=tuple(options), options_explanation=explanation, options_facts=facts)
    _tool(ctx, "compile_report: writing design_report.md")
    narrative: ReportNarrative
    if ctx.llm is None:
        narrative = report.fallback_narrative(inputs, "no LLM configured")
    else:
        try:
            narrative = ctx.llm.parse(REPORT_SYSTEM, report.facts(inputs), ReportNarrative)
        except LLMError as exc:
            narrative = report.fallback_narrative(inputs, str(exc))
    path: Path = report.write_report(inputs, narrative)
    lineage = state.get("lineage")
    _write(state, "run_summary.json", {
        "run_id": inputs.run_id, "status": run_status(state), "converged": inputs.converged,
        "sitl_passed": sitl_passed(state), "iterations": inputs.iterations,
        "final_overrides": state.get("sizing_overrides") or {}, "report": path.name,
        "non_default_assumptions": {k: list(v) for k, v in inputs.design_assumptions.non_default().items()},
        "explorer_note": note, "lineage": lineage.model_dump(mode="json") if lineage else None})
    return {"final_report_path": str(path), "viewer_error": state.get("viewer_error"),
            "explorer_mode": mode, "explorer_note": note,
            "explore_options": [option.to_json() for option in options],
            "options_explanation": None if explanation is None else
            f"{explanation.why_it_failed}\n\n{explanation.trade_offs}"}
