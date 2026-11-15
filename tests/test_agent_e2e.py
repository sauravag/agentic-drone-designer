"""Milestone 5 acceptance test (SRS 7): a full end-to-end run from a natural-language prompt through the
LangGraph agent - parse, size, COTS, OpenVSP/VSPAERO + Build123d CAD, critic loop, ArduPilot SITL and the
report - that is forced through at least one critic iteration by an initially tight constraint.

The tight constraint is the SRS's own: series-hybrid sizing carries a 10 % fuel reserve (Module A) while
the critic's reserve gate demands >= 15 % (gate 4). The SRS example mission sizes to a Group 2 hybrid,
so the first critique must fail and the critic must raise `reserve_fraction` before the design closes.

Needs the Docker image (OpenVSP + arduplane):
    docker compose run --rm isr-forge pytest -q tests/test_agent_e2e.py
The LLM is a deterministic scripted stand-in unless ISR_FORGE_LIVE_LLM=1 and Claude credentials are
available, in which case Claude parses the prompt, critiques and writes the narrative.
"""

import os
from pathlib import Path
from typing import Any

import pytest

from src import pipeline
from src.agent.graph import run_agent
from src.agent.nodes import AgentContext
from tests.agent_fakes import SRS_PROMPT, ScriptedLLM

REQUIRED_ARTIFACTS: tuple[str, ...] = ("oml.step", "assembly.step", "bom.json", "ardupilot.param",
                                       "sitl_report.json", "design_report.md")


def _llm() -> tuple[Any, str]:
    if os.environ.get("ISR_FORGE_LIVE_LLM") == "1":
        from src.agent.llm import ClaudeLLM

        llm = ClaudeLLM()
        return llm, f"live {llm.model}"
    return ScriptedLLM(), "scripted"


@pytest.mark.docker
def test_agent_designs_the_srs_example_end_to_end(tmp_path: Path) -> None:
    if not (pipeline.aero_cad_available() and pipeline.sitl_available()):
        pytest.skip("needs OpenVSP and arduplane (run in the Docker image)")
    llm, kind = _llm()
    ctx = AgentContext(llm=llm, outputs_root=tmp_path / "outputs", db_path=tmp_path / "catalog.db",
                       with_aero_cad=True, with_sitl=True)
    state = run_agent(ctx, SRS_PROMPT)

    assert state.get("fatal_error") is None and state.get("clarification_needed") is None, kind
    assert state["converged"], (kind, state["critic_log"])
    assert state["iteration_count"] >= 1, "the reserve gate must force at least one critic loop"
    assert "reserve" in state["critic_log"][0]["failed"]
    assert state["sizing_overrides"].get("reserve_fraction", 0.0) >= 0.15
    out = Path(state["out_dir"])
    assert out.parent == tmp_path / "outputs"
    for name in REQUIRED_ARTIFACTS:
        assert (out / name).is_file() and (out / name).stat().st_size > 0, name
    req = state["requirements"]
    assert req.group_constraint == 2 and req.payload_mass_kg == pytest.approx(3.5)
    assert req.endurance_hours == pytest.approx(6.0) and req.combat_radius_km == pytest.approx(80.0)
    assert state["sizing_results"].dod_group == 2 and state["fcs_artifacts"].passed
    report = (out / "design_report.md").read_text()
    for heading in ("## Executive summary", "## Bill of materials", "## Aerodynamics and stability",
                    "## Flight control and SITL validation", "status **CONVERGED**"):
        assert heading in report, heading
