"""Local web form: form parsing, result presenters, and a headless Streamlit smoke test."""

import json
from pathlib import Path
from typing import Any, Optional

import pytest

from src import pipeline
from src.ui import presenters
from src.ui.examples import EXAMPLE_MISSIONS
from src.agent.llm import StructuredLLM
from src.schemas.requirements import OperationalRequirements
from tests.reference_designs import G1_ELECTRIC_FW, REQ_FAILED

APP_PATH = Path(__file__).resolve().parents[1] / "src" / "ui" / "app.py"


def test_examples_are_valid_requirements() -> None:
    assert "G1 electric fixed-wing (reference)" in EXAMPLE_MISSIONS
    assert all(req.mission_name for req in EXAMPLE_MISSIONS.values())


def _schema_limits(field: str) -> tuple[float, float]:
    from src.schemas.requirements import OperationalRequirements

    lo, hi = float("-inf"), float("inf")
    for rule in OperationalRequirements.model_fields[field].metadata:
        for attr in ("gt", "ge"):
            lo = getattr(rule, attr, None) if getattr(rule, attr, None) is not None else lo
        for attr in ("lt", "le"):
            hi = getattr(rule, attr, None) if getattr(rule, attr, None) is not None else hi
    return lo, hi


def test_form_bounds_sit_inside_the_schema_and_fit_every_preset() -> None:
    for fields in presenters.NUMBER_FIELDS.values():
        for name, lo, hi, _, _ in fields:
            s_lo, s_hi = _schema_limits(name)
            assert s_lo <= lo < hi <= s_hi, name
            for preset in EXAMPLE_MISSIONS.values():
                assert lo <= getattr(preset, name) <= hi, (name, preset.mission_name)


def test_parse_form_accepts_valid_values() -> None:
    req, errors = presenters.parse_form(G1_ELECTRIC_FW.model_dump())
    assert errors == [] and req == G1_ELECTRIC_FW


def test_parse_form_maps_auto_group_to_none() -> None:
    req, _ = presenters.parse_form({**G1_ELECTRIC_FW.model_dump(), "group_constraint": "auto"})
    assert req is not None and req.group_constraint is None


def test_parse_form_reports_friendly_field_errors() -> None:
    req, errors = presenters.parse_form({**G1_ELECTRIC_FW.model_dump(), "payload_mass_kg": -1.0, "mission_name": ""})
    assert req is None
    assert any(e.startswith("Payload mass") for e in errors)
    assert any(e.startswith("Mission name") for e in errors)


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> pipeline.DesignRun:
    root = tmp_path_factory.mktemp("ui")
    return pipeline.run_design(G1_ELECTRIC_FW, outputs_root=root / "o", db_path=root / "c.db", with_aero_cad=False)


def test_sizing_summary_has_headline_numbers(run: pipeline.DesignRun) -> None:
    summary = dict(presenters.sizing_summary(run.sizing))
    assert summary["MTOW"] == f"{run.sizing.mtow_kg:.2f} kg"
    assert summary["DoD group"] == "Group 1"
    assert "Wingspan" in summary and "Cruise L/D" in summary


def test_phase_rows_cover_the_mission(run: pipeline.DesignRun) -> None:
    rows = presenters.phase_rows(run.sizing)
    assert [r["Phase"] for r in rows] == [p.name for p in run.sizing.phases]


def test_bom_rows_list_parts_and_total(run: pipeline.DesignRun) -> None:
    rows = presenters.bom_rows(run.bom)
    parts = [r["Part"] for r in rows]
    assert {"Cruise motor", "Battery", "Payload"} <= set(parts)
    assert rows[-1]["Part"] == "Total" and rows[-1]["Mass (kg)"] == pytest.approx(run.bom.total_hardware_mass_kg)


def test_fcs_rows_compare_against_srs_limits() -> None:
    from src.schemas.state import FCSArtifacts

    fcs = FCSArtifacts(param_path="a", sitl_report_path="b", passed=False, roll_overshoot=0.05, pitch_overshoot=0.25,
                       roll_settling_s=1.5, pitch_settling_s=1.0, max_transition_altitude_loss_m=0.3)
    rows = {r["Check"]: r for r in presenters.fcs_rows(fcs)}
    assert rows["Roll overshoot (+15° step)"]["Pass"] == "yes" and rows["Roll overshoot (+15° step)"]["Result"] == "5%"
    assert rows["Pitch overshoot (+10° step)"]["Pass"] == "no"
    assert rows["Transition altitude loss"]["Result"] == "0.30 m"


def test_downloadable_files_stay_inside_the_run_folder(run: pipeline.DesignRun) -> None:
    files = presenters.downloadable_files(run.out_dir)
    assert {f.name for f in files} >= {"requirements.json", "sizing.json", "bom.json"}
    assert all(f.resolve().parent == run.out_dir.resolve() for f in files)


def test_app_runs_a_sizing_only_design_headlessly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("ISR_FORGE_OUTPUTS", str(tmp_path / "outputs"))
    monkeypatch.setenv("ISR_FORGE_DB", str(tmp_path / "catalog.db"))
    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    assert not at.exception
    assert at.checkbox(key="with_sitl").disabled == (not (pipeline.aero_cad_available() and pipeline.sitl_available()))
    aero_box = at.checkbox(key="with_aero_cad")
    if not aero_box.disabled:  # enabled only where OpenVSP exists; keep this smoke test fast
        aero_box.uncheck()
    assert aero_box.disabled == (not pipeline.aero_cad_available())
    at.button(key="submit").click()
    at.run()
    assert not at.exception
    assert any(m.label == "MTOW" for m in at.metric)
    assert any(m.label == "Critic iterations" for m in at.metric)  # the form runs the agent's critic loop
    assert any((tmp_path / "outputs").glob("*/bom.json"))
    assert any((tmp_path / "outputs").glob("*/design_report.md"))
    assert at.text_area(key="prompt").disabled  # no API key: description input is off, the form still works


def test_app_designs_from_a_description_when_claude_is_configured(tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit.testing.v1 import AppTest

    from src.ui import agent_service
    from tests.agent_fakes import SRS_PROMPT, ScriptedLLM

    monkeypatch.setattr(agent_service, "llm_problem", lambda: None)
    monkeypatch.setattr(agent_service, "make_llm", lambda: ScriptedLLM())
    monkeypatch.setenv("ISR_FORGE_OUTPUTS", str(tmp_path / "outputs"))
    monkeypatch.setenv("ISR_FORGE_DB", str(tmp_path / "catalog.db"))
    at = AppTest.from_file(str(APP_PATH), default_timeout=180)
    at.run()
    if not at.checkbox(key="with_aero_cad").disabled:
        at.checkbox(key="with_aero_cad").uncheck()
    at.text_area(key="prompt").set_value(SRS_PROMPT)
    at.button(key="submit_prompt").click()
    at.run()
    assert not at.exception
    assert any(m.label == "MTOW" for m in at.metric)
    assert any(e.label.startswith("Assumptions") for e in at.expander)
    assert not any(t.key == "llm_model" for t in at.text_input)  # the model is set only in .env


def _llm_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    from streamlit.testing.v1 import AppTest

    from src.ui import agent_service

    monkeypatch.setattr(agent_service, "llm_problem", lambda: None)
    monkeypatch.setenv("ISR_FORGE_OUTPUTS", str(tmp_path / "outputs"))
    monkeypatch.setenv("ISR_FORGE_DB", str(tmp_path / "catalog.db"))
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    return at


def test_a_blank_description_is_rejected_without_running(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _llm_app(tmp_path, monkeypatch)
    at.text_area(key="prompt").set_value("   ")
    at.button(key="submit_prompt").click()
    at.run()
    assert not at.exception
    assert any("Describe the mission first" in e.value for e in at.error)
    assert not (tmp_path / "outputs").exists()


def test_an_unexpected_failure_shows_a_generic_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.ui import agent_service

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("read-only file system")

    at = _llm_app(tmp_path, monkeypatch)
    monkeypatch.setattr(agent_service, "run", boom)
    at.button(key="submit_prompt").click()
    at.run()
    assert not at.exception
    assert any("failed unexpectedly" in e.value for e in at.error)


def test_a_run_whose_final_sizing_failed_renders_without_design_tabs(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    from src.ui import agent_service

    view = agent_service.AgentRunView("not_converged", run_id="r1", out_dir=tmp_path, requirements=G1_ELECTRIC_FW,
                                      warnings=("NON-CONVERGENCE: sizing: mass spiral",))
    at = _llm_app(tmp_path, monkeypatch)
    monkeypatch.setattr(agent_service, "run", lambda *a, **k: view)
    at.button(key="submit_prompt").click()
    at.run()
    assert not at.exception
    assert any("Sizing did not converge" in e.value for e in at.error)
    assert not any(m.label == "MTOW" for m in at.metric)


def test_read_report_is_confined_to_the_run_folder(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "design_report.md").write_text("# ok", encoding="utf-8")
    (tmp_path / "outside.md").write_text("secret")
    assert presenters.read_report(run_dir, run_dir / "design_report.md") == "# ok"
    assert presenters.read_report(run_dir, tmp_path / "outside.md") is None
    assert presenters.read_report(run_dir, run_dir / "missing.md") is None
    assert presenters.read_report(None, run_dir / "design_report.md") is None


def test_app_shows_the_clarification_question(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit.testing.v1 import AppTest

    from src.ui import agent_service
    from tests.agent_fakes import SRS_DRAFT, ScriptedLLM

    monkeypatch.setattr(agent_service, "llm_problem", lambda: None)
    monkeypatch.setattr(agent_service, "make_llm",
                        lambda: ScriptedLLM(draft=SRS_DRAFT.model_copy(update={"payload_mass": None})))
    monkeypatch.setenv("ISR_FORGE_OUTPUTS", str(tmp_path / "outputs"))
    monkeypatch.setenv("ISR_FORGE_DB", str(tmp_path / "catalog.db"))
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    at.button(key="submit_prompt").click()
    at.run()
    assert not at.exception
    assert any("payload mass" in i.value for i in at.info)


def test_app_shows_validation_errors_instead_of_running(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("ISR_FORGE_OUTPUTS", str(tmp_path / "outputs"))
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    at.text_input(key="mission_name").set_value("")
    at.button(key="submit").click()
    at.run()
    assert not at.exception
    assert any("Mission name" in e.value for e in at.error)
    assert not (tmp_path / "outputs").exists()


def test_assumptions_panel_shows_defaults_and_applies_to_description_runs(tmp_path: Path,
                                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    from tests.agent_fakes import SRS_PROMPT, ScriptedLLM

    from src.ui import agent_service

    monkeypatch.setattr(agent_service, "make_llm", lambda: ScriptedLLM())
    at = _llm_app(tmp_path, monkeypatch)
    if not at.checkbox(key="with_aero_cad").disabled:
        at.checkbox(key="with_aero_cad").uncheck()
    assert at.number_input(key="assume_liion_specific_energy_wh_kg").value == 200
    at.number_input(key="assume_liion_max_c_rate").set_value(6.0)
    at.text_area(key="prompt").set_value(SRS_PROMPT)
    at.button(key="submit_prompt").click()
    at.run()
    assert not at.exception
    (saved,) = (tmp_path / "outputs").glob("*/assumptions.json")
    assert json.loads(saved.read_text())["liion_max_c_rate"] == 6.0


def test_reset_restores_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _llm_app(tmp_path, monkeypatch)
    at.number_input(key="assume_drivetrain_efficiency").set_value(0.9)
    at.run()
    at.button(key="reset_assumptions").click()
    at.run()
    assert at.number_input(key="assume_drivetrain_efficiency").value == pytest.approx(0.68)


def test_assumptions_apply_to_the_structured_form_too(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.ui import agent_service

    seen: list[object] = []

    def fake_run(*args: object, **kwargs: object) -> object:
        seen.append(kwargs.get("design_assumptions"))
        return agent_service.AgentRunView("error", message="stop")

    at = _llm_app(tmp_path, monkeypatch)
    monkeypatch.setattr(agent_service, "run", fake_run)
    at.number_input(key="assume_bsfc_kg_per_kwh").set_value(0.4)
    at.button(key="submit").click()
    at.run()
    assert not at.exception
    assert seen and seen[0].bsfc_kg_per_kwh == pytest.approx(0.4)


SELECT_FIELDS = ("airframe_topology", "propulsion_type", "group_constraint")
NO_LLM_PROBLEM = "no LLM in this test"


def _fill_form_widget(at: Any, key: str, value: object) -> None:
    if isinstance(value, bool):
        box = at.checkbox(key=key)
        box.check() if value else box.uncheck()
    elif key in SELECT_FIELDS:
        at.selectbox(key=key).set_value(value)
    elif isinstance(value, str):
        at.text_input(key=key).set_value(value)
    else:
        at.number_input(key=key).set_value(value)


def _form_app_with(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, req: OperationalRequirements,
                   llm: Optional[StructuredLLM] = None) -> Any:
    """Fill the structured form from `req`, aero/CAD off, (optionally) a scripted LLM; submit and return the app."""
    from streamlit.testing.v1 import AppTest

    from src.ui import agent_service

    monkeypatch.setattr(agent_service, "llm_problem", lambda: None if llm is not None else NO_LLM_PROBLEM)
    monkeypatch.setattr(agent_service, "make_llm", lambda: llm)
    monkeypatch.setenv("ISR_FORGE_OUTPUTS", str(tmp_path / "outputs"))
    monkeypatch.setenv("ISR_FORGE_DB", str(tmp_path / "catalog.db"))
    at = AppTest.from_file(str(APP_PATH), default_timeout=300)
    at.run()
    if not at.checkbox(key="with_aero_cad").disabled:
        at.checkbox(key="with_aero_cad").uncheck()
    for key, value in presenters.form_defaults(req).items():
        if key in presenters.FIELD_LABELS:
            _fill_form_widget(at, key, value)
    at.button(key="submit").click()
    at.run()
    assert not at.exception
    return at


def test_failed_design_offers_cards_and_a_linked_rerun(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _form_app_with(tmp_path, monkeypatch, REQ_FAILED)
    assert any(b.key.startswith("select_option_") for b in at.button)
    assert any("checked with sizing + catalog only" in c.value for c in at.caption)
    at.button(key="select_option_0").click()
    at.run()
    at.button(key="rerun").click()
    at.run()
    assert not at.exception
    summaries = sorted((tmp_path / "outputs").glob("*/run_summary.json"))
    assert json.loads(summaries[-1].read_text())["lineage"]["change_source"] == "option"
    assert len([b for b in at.button if b.key.startswith("open_run_")]) == 2
    assert at.session_state["pending_change"] is None


def test_reopening_a_run_shows_its_own_assumptions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _form_app_with(tmp_path, monkeypatch, G1_ELECTRIC_FW)
    first = at.session_state["last_run"].run_id
    at.number_input(key="assume_liion_specific_energy_wh_kg").set_value(300.0)
    at.button(key="submit").click()
    at.run()
    at.button(key=f"open_run_{first}").click()
    at.run()
    assert not at.exception
    assert at.session_state["last_run"].design_assumptions.liion_specific_energy_wh_kg == 200
    assert at.number_input(key="assume_liion_specific_energy_wh_kg").value == 300  # the panel keeps the designer's


def test_converged_design_can_explore_improvements(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _form_app_with(tmp_path, monkeypatch, G1_ELECTRIC_FW)
    run_id = at.session_state["last_run"].run_id
    at.button(key="explore_alternatives").click()
    at.run()
    assert not at.exception
    assert any("kg" in m.value for m in at.markdown if "−" in m.value or "-" in m.value)
    explored = at.session_state["explored_options"][run_id]
    assert explored and any(o["closes"] for o in explored)
    assert any(m.value.startswith("**") and "MTOW \u2212 " in m.value and m.value.endswith(("kg", "group"))
               for m in at.markdown)  # a saving card, not the headline metric


def test_a_change_that_alters_nothing_shows_an_error_and_does_not_run(tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    at = _form_app_with(tmp_path, monkeypatch, REQ_FAILED)
    at.button(key="select_option_0").click()
    at.run()
    view = at.session_state["last_run"]
    widgets = [w for w in [*at.number_input, *at.selectbox, *at.checkbox] if (w.key or "").startswith("review_")]
    assert widgets
    for widget in widgets:  # put every proposed value back to the run's own value
        _, section, field = widget.key.split("_", 2)
        source = view.requirements if section == "requirements" else view.design_assumptions
        widget.set_value(getattr(source, field))
    at.button(key="rerun").click()
    at.run()
    assert not at.exception
    assert any("does not change anything" in e.value for e in at.error)
    assert len(list((tmp_path / "outputs").glob("*/run_summary.json"))) == 1


def test_cancel_clears_the_pending_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _form_app_with(tmp_path, monkeypatch, REQ_FAILED)
    at.button(key="select_option_0").click()
    at.run()
    assert any(b.key == "rerun" for b in at.button)
    at.button(key="cancel_change").click()
    at.run()
    assert at.session_state["pending_change"] is None
    assert not any(b.key == "rerun" for b in at.button)


def test_improvement_savings_read_with_a_true_minus_and_gains_with_a_plus() -> None:
    from src.ui.design_loop import saving_text

    assert saving_text(3.14) == "MTOW − 3.1 kg"
    assert saving_text(-0.12) == "MTOW + 0.1 kg"



def test_rerun_uses_the_stage_checkboxes_at_click_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.ui import agent_service

    at = _form_app_with(tmp_path, monkeypatch, REQ_FAILED)  # parent run: aero/CAD and SITL off
    assert at.session_state["last_run"].cad is None
    seen: dict[str, object] = {}

    def fake_rerun(*args: object, **kwargs: object) -> object:
        seen.update(kwargs)
        return agent_service.AgentRunView("error", message="stop")  # never touches the toolchain

    monkeypatch.setattr(agent_service, "rerun", fake_rerun)
    at.button(key="select_option_0").click()
    at.run()
    at.session_state["with_aero_cad"] = True
    at.session_state["with_sitl"] = True
    at.button(key="rerun").click()
    at.run()
    assert not at.exception
    assert seen["with_aero_cad"] is True and seen["with_sitl"] is True


def test_an_out_of_bounds_review_value_is_rejected_without_running(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    at = _form_app_with(tmp_path, monkeypatch, REQ_FAILED)
    at.button(key="select_option_0").click()  # hybrid: propulsion + fuel reserve
    at.run()
    at.number_input(key="review_assumptions_fuel_reserve_fraction").set_value(5.0)
    at.button(key="rerun").click()
    at.run()
    assert not at.exception
    assert any("fuel_reserve_fraction" in e.value for e in at.error)
    assert len(list((tmp_path / "outputs").glob("*/run_summary.json"))) == 1
    assert at.session_state["pending_change"] is not None  # the panel stays open to fix the value


def test_selecting_another_option_reseeds_the_review_widgets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _form_app_with(tmp_path, monkeypatch, REQ_FAILED)
    at.button(key="select_option_0").click()
    at.run()
    at.number_input(key="review_assumptions_fuel_reserve_fraction").set_value(0.3)
    at.run()
    at.button(key="select_option_1").click()  # Li-ion specific energy
    at.run()
    keys = {w.key for w in [*at.number_input, *at.selectbox] if (w.key or "").startswith("review_")}
    assert keys == {"review_assumptions_liion_specific_energy_wh_kg"}
    proposed = at.session_state["pending_change"][0].assumptions.liion_specific_energy_wh_kg
    assert at.number_input(key="review_assumptions_liion_specific_energy_wh_kg").value == proposed
    at.button(key="cancel_change").click()
    at.run()
    at.button(key="select_option_0").click()
    at.run()
    assert at.number_input(key="review_assumptions_fuel_reserve_fraction").value == pytest.approx(0.15)


def test_options_explanation_and_non_closing_options_render_safely(tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    from src.schemas.changes import ChangeSet
    from src.ui import agent_service

    def change(label: str, **req: object) -> dict[str, object]:
        return ChangeSet.model_validate({"label": label, "requirements": req, "assumptions": {},
                                         "knobs": {}}).model_dump()

    result = {"closes": True, "mtow_kg": 16.9, "dod_group": 2, "propulsion": "series_hybrid",
              "battery_mass_kg": 0.8, "fuel_mass_kg": 1.7, "reason": ""}
    options = (
        {"lever": "propulsion", "change": change("hybrid", propulsion_type="series_hybrid"), "result": result,
         "relative_change": 1.0, "closes": True, "mtow_saved_kg": None, "group_drop": False},
        {"lever": "endurance", "change": change("endurance", endurance_hours=0.5),
         "result": {**result, "closes": False, "reason": "mass spiral"}, "relative_change": 0.0, "closes": False,
         "mtow_saved_kg": None, "group_drop": False},
    )
    view = agent_service.AgentRunView("not_converged", run_id="r1", out_dir=tmp_path, requirements=REQ_FAILED,
                                      explorer_mode="closure", options=options,
                                      options_explanation="Try ![x](https://evil.example/a.png) or [this](http://x)")
    at = _llm_app(tmp_path, monkeypatch)
    monkeypatch.setattr(agent_service, "run", lambda *a, **k: view)
    at.button(key="submit_prompt").click()
    at.run()
    assert not at.exception
    texts = [m.value for m in at.markdown]
    assert any("!\\[x\\]" in t for t in texts)
    assert not any("![x](" in t or "[this](" in t for t in texts)
    assert any("Endurance (h): tried down to 0.5 — mass spiral" in t for t in texts)
