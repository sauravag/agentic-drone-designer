"""Web form <-> agent adapter (M6): structured and natural-language runs, clarification, errors, the view."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src import pipeline
from src.agent.graph import initial_state
from src.agent.providers import LLMConfigError
from src.schemas.state import DesignState, RunLineage
from src.tools.sizing.mission_solver import size_aircraft
from src.ui import agent_service
from tests.agent_fakes import SRS_DRAFT, SRS_PROMPT, ScriptedLLM
from src.schemas.assumptions import DesignAssumptions
from src.schemas.fabrication import DEFAULT_FABRICATION, FabricationSettings
from src.schemas.changes import AssumptionChanges, ChangeSet, KnobChanges, RequirementChanges
from tests.reference_designs import G1_ELECTRIC_FW, REQ_FAILED


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


def test_rerun_links_to_the_parent_and_applies_the_change(tmp_path: Path, no_key: None) -> None:
    first = _run(tmp_path, requirements=REQ_FAILED)
    closing = [o for o in first.options if o["closes"]]
    assert closing, "REQ_FAILED must offer at least one closing option"
    option = closing[0]
    change = ChangeSet.model_validate(option["change"])
    second = agent_service.rerun(first, change, "option", tmp_path / "o", tmp_path / "c.db", False, False, lambda _: None)
    assert second.lineage.parent_run_id == first.run_id and second.status == agent_service.STATUS_CONVERGED
    assert json.loads((second.out_dir / "run_summary.json").read_text())["lineage"]["change_source"] == "option"
    assert f"Derived from run {first.run_id} (option)" in second.report_path.read_text()


def test_a_change_that_changes_nothing_is_refused(tmp_path: Path, no_key: None) -> None:  # Review Focus 2
    first = _run(tmp_path, requirements=G1_ELECTRIC_FW)
    same = ChangeSet(label="same", requirements=RequirementChanges(endurance_hours=G1_ELECTRIC_FW.endurance_hours),
                     assumptions=AssumptionChanges(), knobs=KnobChanges())
    with pytest.raises(agent_service.NoChangeError):
        agent_service.rerun(first, same, "chat", tmp_path / "o", tmp_path / "c.db", False, False, lambda _: None)


def test_description_runs_use_the_panel_assumptions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_service, "make_llm", lambda: ScriptedLLM())
    a = DesignAssumptions(bsfc_kg_per_kwh=0.8)  # the SRS design is a series hybrid, so fuel burn matters
    view = _run(tmp_path, prompt=SRS_PROMPT, design_assumptions=a)
    baseline = _run(tmp_path / "base", prompt=SRS_PROMPT)
    assert view.design_assumptions == a
    assert view.sizing is not None and baseline.sizing is not None
    assert view.sizing.fuel_mass_kg != baseline.sizing.fuel_mass_kg  # the assumption reached the design


def test_rerun_needs_a_view_with_requirements(tmp_path: Path, no_key: None) -> None:
    change = ChangeSet(label="x", requirements=RequirementChanges(endurance_hours=2.0),
                       assumptions=AssumptionChanges(), knobs=KnobChanges())
    with pytest.raises(ValueError, match="requirements"):
        agent_service.rerun(agent_service.AgentRunView(agent_service.STATUS_ERROR, "boom"), change, "manual",
                            tmp_path / "o", tmp_path / "c.db", False, False, lambda _: None)


def test_explore_is_limited_to_converged_views(tmp_path: Path, no_key: None) -> None:
    with pytest.raises(ValueError, match="converged"):
        agent_service.explore(agent_service.AgentRunView(agent_service.STATUS_ERROR, "boom"), tmp_path / "c.db")
    view = _run(tmp_path, requirements=G1_ELECTRIC_FW)
    options = agent_service.explore(view, tmp_path / "c.db")
    assert isinstance(options, tuple) and options


def test_run_lineage_needs_no_import_order(tmp_path: Path) -> None:
    code = ("from src.schemas.state import RunLineage; RunLineage.model_validate({'parent_run_id': 'r', "
            "'change_source': 'chat', 'change_set': {'label': 'x', 'requirements': {}, 'assumptions': {}, 'knobs': {}}})")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).resolve().parent.parent)


CUSTOM_FABRICATION = FabricationSettings(bed_width_m=0.5, bed_height_m=0.3)


def test_run_carries_fabrication_settings(tmp_path: Path, no_key: None) -> None:
    view = _run(tmp_path, requirements=G1_ELECTRIC_FW, fabrication=CUSTOM_FABRICATION)
    assert view.fabrication == CUSTOM_FABRICATION
    written = json.loads((view.out_dir / "fabrication.json").read_text())
    assert FabricationSettings.model_validate(written) == CUSTOM_FABRICATION


def test_a_run_without_settings_views_the_default(tmp_path: Path, no_key: None) -> None:
    assert _run(tmp_path, requirements=G1_ELECTRIC_FW).fabrication == DEFAULT_FABRICATION


def test_rerun_keeps_the_parent_settings(tmp_path: Path, no_key: None) -> None:
    first = _run(tmp_path, requirements=G1_ELECTRIC_FW, fabrication=CUSTOM_FABRICATION)
    change = ChangeSet(label="longer", requirements=RequirementChanges(endurance_hours=G1_ELECTRIC_FW.endurance_hours + 0.25),
                       assumptions=AssumptionChanges(), knobs=KnobChanges())
    second = agent_service.rerun(first, change, "manual", tmp_path / "o", tmp_path / "c.db", False, False, lambda _: None)
    assert second.fabrication == CUSTOM_FABRICATION
    assert FabricationSettings.model_validate_json((second.out_dir / "fabrication.json").read_text()) == CUSTOM_FABRICATION


def test_rerun_with_edited_settings_uses_them_and_records_the_change(tmp_path: Path, no_key: None) -> None:
    """R37: a re-run uses the panel's settings; the lineage names the fields that differ from the parent's."""
    first = _run(tmp_path, requirements=G1_ELECTRIC_FW)
    change = ChangeSet(label="longer", requirements=RequirementChanges(endurance_hours=G1_ELECTRIC_FW.endurance_hours + 0.25),
                       assumptions=AssumptionChanges(), knobs=KnobChanges())
    second = agent_service.rerun(first, change, "manual", tmp_path / "o", tmp_path / "c.db", False, False,
                                 lambda _: None, fabrication=CUSTOM_FABRICATION)
    assert second.fabrication == CUSTOM_FABRICATION
    assert second.lineage.fabrication_changes == ("bed_width_m", "bed_height_m")
    assert FabricationSettings.model_validate_json((second.out_dir / "fabrication.json").read_text()) == CUSTOM_FABRICATION
    summary = json.loads((second.out_dir / "run_summary.json").read_text())
    assert summary["lineage"]["fabrication_changes"] == ["bed_width_m", "bed_height_m"]


def test_rerun_with_the_parents_settings_records_no_change(tmp_path: Path, no_key: None) -> None:
    first = _run(tmp_path, requirements=G1_ELECTRIC_FW, fabrication=CUSTOM_FABRICATION)
    change = ChangeSet(label="longer", requirements=RequirementChanges(endurance_hours=G1_ELECTRIC_FW.endurance_hours + 0.25),
                       assumptions=AssumptionChanges(), knobs=KnobChanges())
    second = agent_service.rerun(first, change, "manual", tmp_path / "o", tmp_path / "c.db", False, False,
                                 lambda _: None, fabrication=CUSTOM_FABRICATION)
    assert second.lineage.fabrication_changes == ()


def test_old_lineage_records_load_without_fabrication_changes() -> None:
    lineage = RunLineage.model_validate({"parent_run_id": "r", "change_source": "chat", "change_set": {
        "label": "x", "requirements": {}, "assumptions": {}, "knobs": {}}})
    assert lineage.fabrication_changes == ()
