"""Web form <-> design agent (M6). Builds the AgentContext, runs the LangGraph critic loop, and flattens the
final DesignState into an immutable view for app.py. Kept free of Streamlit so it is unit-testable.

Claude is optional: a natural-language request needs it (parser), while structured form input runs the
same loop with the rule-table critic and a template narrative when no API key is configured.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final, Optional

from src import pipeline
from src.agent.graph import run_agent
from src.agent.llm import ClaudeLLM, StructuredLLM
from src.agent.nodes import MAX_ITERATIONS, AgentContext, run_status, run_warnings
from src.schemas.components import SelectedBOM
from src.schemas.requirements import OperationalRequirements
from src.schemas.state import AeroStabilityOutput, CADArtifacts, DesignState, FCSArtifacts, SizingOutput

API_KEY_ENV: Final[str] = "ANTHROPIC_API_KEY"
STATUS_CONVERGED: Final[str] = "converged"
STATUS_CLARIFY: Final[str] = "clarification"
STATUS_ERROR: Final[str] = "error"
NO_LLM_MESSAGE: Final[str] = (f"Designing from a description needs Claude: set {API_KEY_ENV} (in .env for Docker "
                              "Compose, or in your shell on the host), or use the structured form below.")


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

    @property
    def has_design(self) -> bool:
        return self.status not in (STATUS_CLARIFY, STATUS_ERROR)


def llm_configured() -> bool:
    """Only checks that a key is present; the key itself is read by the Anthropic SDK, never by us."""
    return bool(os.environ.get(API_KEY_ENV, "").strip())


def make_llm() -> Optional[StructuredLLM]:
    return ClaudeLLM() if llm_configured() else None


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
        used_llm=used_llm,
    )


def run(outputs_root: Path, db_path: Path, with_aero_cad: Optional[bool], with_sitl: Optional[bool],
        progress: Callable[[str], None], prompt: str = "", requirements: Optional[OperationalRequirements] = None,
        max_iterations: int = MAX_ITERATIONS) -> AgentRunView:
    """Run the agent on a description (`prompt`) or on validated form input (`requirements`).
    Raises pipeline.PipelineError when a requested stage's toolchain is missing."""
    if bool(prompt.strip()) == (requirements is not None):
        raise ValueError("give either a description or structured requirements")
    stages: dict[str, bool] = pipeline.resolve_stages(with_aero_cad, with_sitl)
    llm: Optional[StructuredLLM] = make_llm()
    if requirements is None and llm is None:
        return AgentRunView(STATUS_ERROR, NO_LLM_MESSAGE)
    ctx = AgentContext(llm=llm, outputs_root=outputs_root, db_path=db_path, with_aero_cad=stages["aero_cad"],
                       with_sitl=stages["sitl"], max_iterations=max_iterations, progress=progress)
    return to_view(run_agent(ctx, prompt.strip(), requirements), used_llm=llm is not None)
