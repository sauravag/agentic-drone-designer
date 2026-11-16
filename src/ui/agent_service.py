"""Web form <-> design agent (M6). Builds the AgentContext, runs the LangGraph critic loop, and flattens the
final DesignState into an immutable view for app.py. Kept free of Streamlit so it is unit-testable.

An LLM is optional: a natural-language request needs one (parser), while structured form input runs the
same loop with the rule-table critic and a template narrative when no API key is configured. The provider
and model come only from the environment (.env) via src/agent/providers.py; the form cannot change them.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final, Literal, Optional

from src import pipeline
from src.agent.graph import run_agent
from src.agent import providers
from src.agent.explorer import ExploreOption, explore_improvement
from src.agent.report import inline
from src.agent.llm import StructuredLLM
from src.agent.nodes import MAX_ITERATIONS, AgentContext, run_status, run_warnings
from src.schemas.assumptions import DEFAULT_ASSUMPTIONS, DesignAssumptions
from src.schemas.changes import ChangeSet, apply_changes
from src.schemas.components import SelectedBOM
from src.schemas.requirements import OperationalRequirements
from src.schemas.state import AeroStabilityOutput, CADArtifacts, DesignState, FCSArtifacts, RunLineage, SizingOutput

STATUS_CONVERGED: Final[str] = "converged"
STATUS_CLARIFY: Final[str] = "clarification"
STATUS_ERROR: Final[str] = "error"
NO_LLM_TEMPLATE: Final[str] = ("Designing from a description needs an LLM: {problem} (in .env for Docker Compose, "
                               "or in your shell on the host), or use the structured form below.")


@dataclass(frozen=True)
class AgentRunView:
    status: str  # converged | not_converged | converged_sitl_failed | clarification | error
    message: Optional[str] = None  # the clarification question or the error
    run_id: Optional[str] = None
    out_dir: Optional[Path] = None
    requirements: Optional[OperationalRequirements] = None
    sizing: Optional[SizingOutput] = None
    bom: Optional[SelectedBOM] = None
    aero: Optional[AeroStabilityOutput] = None
    cad: Optional[CADArtifacts] = None
    fcs: Optional[FCSArtifacts] = None
    iterations: int = 0
    warnings: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    gates: tuple[dict[str, Any], ...] = ()
    critic_log: tuple[dict[str, Any], ...] = ()
    report_path: Optional[Path] = None
    used_llm: bool = False
    design_assumptions: DesignAssumptions = DEFAULT_ASSUMPTIONS
    explorer_mode: str = "none"
    options: tuple[dict[str, Any], ...] = ()
    options_explanation: Optional[str] = None
    lineage: Optional[RunLineage] = None

    @property
    def has_design(self) -> bool:
        return self.status not in (STATUS_CLARIFY, STATUS_ERROR)


def llm_problem() -> Optional[str]:
    """Why designing from a description is unavailable, or None when an LLM is configured.
    Only checks that a key is present; the key itself is read by the provider SDK, never shown."""
    problem: Optional[str] = providers.config_problem()
    return None if problem is None else NO_LLM_TEMPLATE.format(problem=problem)


def llm_configured() -> bool:
    return llm_problem() is None


def llm_label() -> str:
    """`provider · model` for the form; raises providers.LLMConfigError on a bad setting."""
    return providers.describe()


def make_llm() -> Optional[StructuredLLM]:
    """The LLM configured in the environment, or None when no key is set.
    Raises providers.LLMConfigError for an invalid provider or model id."""
    return providers.make_llm() if llm_configured() else None


def _optional_path(value: Optional[str]) -> Optional[Path]:
    return Path(value) if value else None


def to_view(state: DesignState, used_llm: bool) -> AgentRunView:
    if state.get("fatal_error"):
        return AgentRunView(STATUS_ERROR, state["fatal_error"], run_id=state.get("run_id"), used_llm=used_llm)
    if state.get("clarification_needed"):
        return AgentRunView(STATUS_CLARIFY, state["clarification_needed"], used_llm=used_llm)
    return AgentRunView(
        status=run_status(state), run_id=state.get("run_id"), out_dir=_optional_path(state.get("out_dir")),
        requirements=state.get("requirements"), sizing=state.get("sizing_results"), bom=state.get("selected_bom"),
        aero=state.get("aero_stability"), cad=state.get("cad_artifacts"), fcs=state.get("fcs_artifacts"),
        iterations=state.get("iteration_count", 0), warnings=tuple(run_warnings(state)),
        assumptions=tuple(state.get("assumptions", [])), gates=tuple(state.get("gate_results", [])),
        critic_log=tuple(state.get("critic_log", [])), report_path=_optional_path(state.get("final_report_path")),
        used_llm=used_llm, design_assumptions=state.get("design_assumptions") or DEFAULT_ASSUMPTIONS,
        explorer_mode=state.get("explorer_mode", "none"), options=tuple(state.get("explore_options", [])),
        options_explanation=state.get("options_explanation"), lineage=state.get("lineage"),
    )


class NoChangeError(ValueError):
    """A re-run was requested with a change that alters nothing."""


def _run_agent(ctx_args: dict[str, Any], prompt: str, requirements: Optional[OperationalRequirements],
               sizing_overrides: Optional[dict[str, float]], design_assumptions: Optional[DesignAssumptions],
               lineage: Optional[RunLineage], llm: Optional[StructuredLLM]) -> AgentRunView:
    ctx = AgentContext(llm=llm, **ctx_args)
    return to_view(run_agent(ctx, prompt, requirements, sizing_overrides, design_assumptions, lineage),
                   used_llm=llm is not None)


def _context_args(outputs_root: Path, db_path: Path, with_aero_cad: Optional[bool], with_sitl: Optional[bool],
                  progress: Callable[[str], None], max_iterations: int) -> dict[str, Any]:
    stages: dict[str, bool] = pipeline.resolve_stages(with_aero_cad, with_sitl)
    return {"outputs_root": outputs_root, "db_path": db_path, "with_aero_cad": stages["aero_cad"],
            "with_sitl": stages["sitl"], "max_iterations": max_iterations, "progress": progress}


def run(outputs_root: Path, db_path: Path, with_aero_cad: Optional[bool], with_sitl: Optional[bool],
        progress: Callable[[str], None], prompt: str = "", requirements: Optional[OperationalRequirements] = None,
        max_iterations: int = MAX_ITERATIONS,
        design_assumptions: Optional[DesignAssumptions] = None) -> AgentRunView:
    """Run the agent on a description (`prompt`) or on validated form input (`requirements`), with the
    designer's `design_assumptions` (defaults when omitted).
    Raises pipeline.PipelineError when a requested stage's toolchain is missing and
    providers.LLMConfigError for an invalid provider or model setting."""
    if bool(prompt.strip()) == (requirements is not None):
        raise ValueError("give either a description or structured requirements")
    args = _context_args(outputs_root, db_path, with_aero_cad, with_sitl, progress, max_iterations)
    llm: Optional[StructuredLLM] = make_llm()
    if requirements is None and llm is None:
        return AgentRunView(STATUS_ERROR, llm_problem())
    return _run_agent(args, prompt.strip(), requirements, None, design_assumptions, None, llm)


def rerun(view: AgentRunView, change: ChangeSet, source: Literal["option", "chat", "manual"], outputs_root: Path,
          db_path: Path, with_aero_cad: Optional[bool], with_sitl: Optional[bool],
          progress: Callable[[str], None], max_iterations: int = MAX_ITERATIONS) -> AgentRunView:
    """Apply `change` to the view's design and run the structured path again, linked to the parent run.

    The critic starts from default knobs (the change's knobs, if any, are the only overrides). Raises
    NoChangeError when nothing would differ, ValueError for a view without requirements, pydantic
    ValidationError / ValueError for out-of-bounds values, and pipeline.PipelineError for a missing toolchain."""
    if view.requirements is None or view.run_id is None:
        raise ValueError("this run has no requirements to change")
    if change.is_empty() or not change.describe(view.requirements, view.design_assumptions):
        raise NoChangeError(f"'{inline(change.label)}' does not change anything in this design")
    req, assumptions, knobs = apply_changes(view.requirements, view.design_assumptions, change)
    args = _context_args(outputs_root, db_path, with_aero_cad, with_sitl, progress, max_iterations)
    lineage = RunLineage(parent_run_id=view.run_id, change_set=change, change_source=source)
    return _run_agent(args, "", req, knobs, assumptions, lineage, make_llm())


def explore(view: AgentRunView, db_path: Path) -> tuple[ExploreOption, ...]:
    """On-demand improvement mode: what shrinks a converged design the most per stated step."""
    if view.status != STATUS_CONVERGED or view.requirements is None or view.sizing is None:
        raise ValueError("improvement options need a converged design with sizing results")
    return tuple(explore_improvement(view.requirements, view.design_assumptions, view.sizing, db_path))
