"""Web form <-> agent adapter (M6): structured and natural-language runs, clarification, errors, the view."""

from pathlib import Path

import pytest

from src import pipeline
from src.agent.graph import initial_state
from src.agent.providers import LLMConfigError
from src.schemas.state import DesignState
from src.tools.sizing.mission_solver import size_aircraft
from src.ui import agent_service
from tests.agent_fakes import SRS_DRAFT, SRS_PROMPT, ScriptedLLM
from tests.reference_designs import G1_ELECTRIC_FW


def _run(tmp_path: Path, **kwargs: object) -> agent_service.AgentRunView:
    return agent_service.run(tmp_path / "o", tmp_path / "c.db", with_aero_cad=False, with_sitl=False,
                             progress=lambda _: None, **kwargs)


@pytest.fixture
def no_key() -> None:
    """tests/conftest.py already clears every provider key; named for readability."""


def test_form_input_runs_the_critic_loop_and_writes_a_report_without_an_llm(tmp_path: Path, no_key: None) -> None:
    view = _run(tmp_path, requirements=G1_ELECTRIC_FW)
    assert view.has_design and not view.used_llm
    assert view.status == agent_service.STATUS_CONVERGED and view.iterations == 0
    assert view.sizing is not None and view.bom is not None and view.requirements == G1_ELECTRIC_FW
    assert view.gates and all({"name", "passed"} <= set(g) for g in view.gates)
    assert view.report_path is not None and view.report_path.is_file()
    assert view.out_dir is not None and (view.out_dir / "run_summary.json").is_file()


def test_description_without_an_api_key_is_refused_before_any_run(tmp_path: Path, no_key: None) -> None:
    view = _run(tmp_path, prompt=SRS_PROMPT)
    assert view.status == agent_service.STATUS_ERROR and view.message == agent_service.llm_problem()
    assert "OPENROUTER_API_KEY" in view.message and "ANTHROPIC_API_KEY" in view.message
    assert not view.has_design
    assert not (tmp_path / "o").exists()


def test_description_with_an_llm_parses_iterates_and_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    llm = ScriptedLLM()
    monkeypatch.setattr(agent_service, "make_llm", lambda: llm)
    view = _run(tmp_path, prompt=f"  {SRS_PROMPT}  ")
    assert view.used_llm and view.has_design
    assert view.requirements is not None and view.requirements.payload_mass_kg == pytest.approx(3.5)
    assert view.assumptions[0].startswith("Read from the request")
    assert view.iterations == 1 and len(view.critic_log) == 2  # the SRS 10 % fuel reserve fails the 15 % gate
    assert view.critic_log[0]["failed"] and not view.critic_log[1]["failed"]
    assert view.status == agent_service.STATUS_CONVERGED
    assert llm.calls[0] == ("RequirementsDraft", SRS_PROMPT)  # prompt is stripped before it reaches the LLM


def test_a_missing_required_fact_returns_the_clarification_question(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_service, "make_llm", lambda: ScriptedLLM(draft=SRS_DRAFT.model_copy(
        update={"endurance": None})))
    view = _run(tmp_path, prompt="Design a drone with a 3.5 kg gimbal")
    assert view.status == agent_service.STATUS_CLARIFY and "endurance" in (view.message or "")
    assert view.run_id is None and not view.has_design


@pytest.mark.parametrize("kwargs", [{}, {"prompt": "   "}, {"prompt": "x", "requirements": G1_ELECTRIC_FW}])
def test_exactly_one_input_is_required(tmp_path: Path, kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="either"):
        _run(tmp_path, **kwargs)


def test_missing_toolchain_is_rejected_before_a_run_folder_exists(tmp_path: Path) -> None:
    with pytest.raises(pipeline.PipelineError):
        agent_service.run(tmp_path / "o", tmp_path / "c.db", with_aero_cad=False, with_sitl=True,
                          progress=lambda _: None, requirements=G1_ELECTRIC_FW)
    assert not (tmp_path / "o").exists()


@pytest.mark.parametrize("env", ["OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"])
@pytest.mark.parametrize(("value", "expected"), [("sk-test", True), ("   ", False), ("", False)])
def test_llm_configured_only_checks_for_a_non_blank_key(monkeypatch: pytest.MonkeyPatch, env: str, value: str,
                                                        expected: bool) -> None:
    monkeypatch.setenv(env, value)
    assert agent_service.llm_configured() is expected


def test_an_invalid_model_in_the_environment_is_rejected_before_a_run_folder_exists(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("ISR_FORGE_MODEL", "gpt; rm -rf /")
    with pytest.raises(LLMConfigError, match="invalid model id"):
        _run(tmp_path, requirements=G1_ELECTRIC_FW)
    assert not (tmp_path / "o").exists()


def test_the_form_label_shows_the_model_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    assert agent_service.llm_label() == "openrouter · anthropic/claude-opus-5.5"
    monkeypatch.setenv("ISR_FORGE_MODEL", "google/gemini-3.8-flash")
    assert agent_service.llm_label() == "openrouter · google/gemini-3.8-flash"


def test_run_has_no_model_override() -> None:
    import inspect

    assert "model" not in inspect.signature(agent_service.run).parameters  # .env is the only switch


def test_no_key_means_no_llm(no_key: None) -> None:
    assert agent_service.make_llm() is None


def _state(**update: object) -> DesignState:
    return initial_state("", G1_ELECTRIC_FW) | {"run_id": "r1", "out_dir": "/tmp/r1"} | update  # type: ignore[return-value]


def test_a_sitl_abort_after_convergence_maps_to_converged_sitl_failed() -> None:
    view = agent_service.to_view(_state(converged=True, sitl_error="SITL aborted: arduplane crashed"), False)
    assert view.status == "converged_sitl_failed" and view.warnings[0] == "SITL aborted: arduplane crashed"


def test_a_final_sizing_failure_yields_a_view_without_a_design() -> None:
    view = agent_service.to_view(_state(validation_errors=["sizing: mass spiral"]), False)
    assert view.status == "not_converged" and view.sizing is None and view.bom is None
    assert view.warnings[0].startswith("NON-CONVERGENCE") and "mass spiral" in view.warnings[0]


def test_a_catalog_failure_keeps_the_sizing_and_skips_payload_warnings() -> None:
    sizing = size_aircraft(G1_ELECTRIC_FW)
    view = agent_service.to_view(_state(sizing_results=sizing, stage_error={"stage": "cots", "message": "x"}), False)
    assert view.sizing == sizing and view.bom is None
    assert not any("Catalog payload" in w for w in view.warnings)


def test_a_fatal_agent_error_becomes_an_error_view() -> None:
    view = agent_service.to_view({"fatal_error": "parser unavailable", "run_id": None}, used_llm=True)  # type: ignore[typeddict-item]
    assert view.status == agent_service.STATUS_ERROR and view.message == "parser unavailable"
