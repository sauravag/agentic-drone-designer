"""ISR-Forge LangGraph state machine (SRS 5):

    parse_requirements -> run_sizing -> match_cots -> build_aero_cad -> design_critic
        design_critic: all gates pass         -> generate_and_verify_fcs -> compile_report -> END
                       gate failed, iter < 8  -> run_sizing (with updated sizing_overrides)
                       iter >= 8 / stalled    -> compile_report (non-convergence warnings)
    parse_requirements: clarification or fatal error -> END
    any tool node with a design error -> design_critic (treated as a failed gate)
"""

import json
import logging
from functools import partial
from pathlib import Path
from typing import Any, Callable, Final, Optional

from langgraph.graph import END, START, StateGraph

from src.agent import nodes
from src.agent.nodes import AgentContext
from src.schemas.requirements import OperationalRequirements
from src.schemas.state import DesignState

logger = logging.getLogger(__name__)

NODES_PER_ITERATION: Final[int] = 4  # run_sizing, match_cots, build_aero_cad, design_critic
FIXED_NODES: Final[int] = 10  # parse, fcs, report + slack
NODE_NAMES: Final[tuple[str, ...]] = ("parse_requirements", "run_sizing", "match_cots", "build_aero_cad",
                                      "design_critic", "generate_and_verify_fcs", "compile_report")


def _after_parse(state: DesignState) -> str:
    if state.get("fatal_error") or state.get("clarification_needed"):
        return END
    return "run_sizing"


def _next_or_critic(next_node: str) -> Callable[[DesignState], str]:
    def route(state: DesignState) -> str:
        return "design_critic" if state.get("stage_error") else next_node
    return route


def _after_critic(ctx: AgentContext) -> Callable[[DesignState], str]:
    def route(state: DesignState) -> str:
        if state.get("converged"):
            has_cad: bool = state.get("cad_artifacts") is not None
            return "generate_and_verify_fcs" if ctx.with_sitl and has_cad else "compile_report"
        if state.get("critic_exhausted"):
            return "compile_report"
        return "run_sizing"
    return route


def build_graph(ctx: AgentContext) -> Any:
    graph = StateGraph(DesignState)
    for name in NODE_NAMES:
        graph.add_node(name, partial(getattr(nodes, name), ctx=ctx))
    graph.add_edge(START, "parse_requirements")
    graph.add_conditional_edges("parse_requirements", _after_parse, ["run_sizing", END])
    graph.add_conditional_edges("run_sizing", _next_or_critic("match_cots"), ["match_cots", "design_critic"])
    graph.add_conditional_edges("match_cots", _next_or_critic("build_aero_cad"), ["build_aero_cad", "design_critic"])
    graph.add_edge("build_aero_cad", "design_critic")
    graph.add_conditional_edges("design_critic", _after_critic(ctx),
                                ["generate_and_verify_fcs", "compile_report", "run_sizing"])
    graph.add_edge("generate_and_verify_fcs", "compile_report")
    graph.add_edge("compile_report", END)
    return graph.compile()


def initial_state(user_prompt: str, requirements: Optional[OperationalRequirements] = None,
                  sizing_overrides: Optional[dict[str, float]] = None) -> DesignState:
    return DesignState(
        user_prompt=user_prompt, requirements=requirements, clarification_needed=None, iteration_count=0,
        sizing_overrides=dict(sizing_overrides or {}), sizing_results=None, selected_bom=None, aero_stability=None,
        cad_artifacts=None, fcs_artifacts=None, validation_errors=[], final_report_path=None, run_id=None,
        out_dir=None, assumptions=[], gate_results=[], critic_log=[], stage_error=None, converged=False,
        critic_exhausted=False, fatal_error=None, sitl_error=None,
    )


def run_agent(ctx: AgentContext, user_prompt: str = "", requirements: Optional[OperationalRequirements] = None,
              sizing_overrides: Optional[dict[str, float]] = None) -> DesignState:
    """Run the full design loop from a natural-language prompt (or structured requirements)."""
    if not user_prompt and requirements is None:
        raise ValueError("give a natural-language prompt or structured requirements")
    state: DesignState = initial_state(user_prompt, requirements, sizing_overrides)
    limit: int = NODES_PER_ITERATION * (ctx.max_iterations + 1) + FIXED_NODES
    try:
        for snapshot in build_graph(ctx).stream(state, {"recursion_limit": limit}, stream_mode="values"):
            state = snapshot
    except Exception as exc:  # leave a failed summary in the run folder, then surface the error unchanged
        _write_failed_summary(state, exc)
        raise
    return state


def _write_failed_summary(state: DesignState, exc: Exception) -> None:
    out_dir = state.get("out_dir")
    if not out_dir:
        return
    summary = {"run_id": state.get("run_id"), "status": "failed", "error": f"{type(exc).__name__}: {exc}",
               "iterations": state.get("iteration_count", 0)}
    (Path(out_dir) / "run_summary.json").write_text(json.dumps(summary, indent=2))
    logger.error("agent run %s failed: %s", state.get("run_id"), exc)
