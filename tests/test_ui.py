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
from src.schemas.state import FCSArtifacts, PlantCaseResult
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
    fcs = FCSArtifacts(param_path="a", sitl_report_path="b", passed=False, roll_overshoot=0.05, pitch_overshoot=0.25,
                       roll_settling_s=1.5, pitch_settling_s=1.0, max_transition_altitude_loss_m=0.3)
    rows = {r["Check"]: r for r in presenters.fcs_rows(fcs)}
    assert rows["Roll overshoot (+15° step)"]["Pass"] == "yes" and rows["Roll overshoot (+15° step)"]["Result"] == "5%"
    assert rows["Pitch overshoot (+10° step)"]["Pass"] == "no"
    assert rows["Transition altitude loss"]["Result"] == "0.30 m"


def _fcs_with_cases(**updates: object) -> FCSArtifacts:
    cases = (PlantCaseResult(case="nominal", roll_overshoot=0.05, roll_settling_s=1.0, pitch_overshoot=0.04,
                             pitch_settling_s=0.9),
             PlantCaseResult(case="hot", roll_overshoot=0.12, roll_settling_s=1.4, pitch_overshoot=0.08,
                             pitch_settling_s=1.1),
             PlantCaseResult(case="ringing", roll_overshoot=0.25, roll_settling_s=1.0, pitch_overshoot=0.04,
                             pitch_settling_s=0.9),
             PlantCaseResult(case="sluggish", roll_overshoot=0.0, roll_settling_s=0.0, pitch_overshoot=0.0,
                             pitch_settling_s=0.0, error="SitlFlightError: stalled"))
    base: dict[str, object] = dict(param_path="a", sitl_report_path="b", passed=False, roll_overshoot=0.12,
                                   pitch_overshoot=0.08, roll_settling_s=1.4, pitch_settling_s=1.1,
                                   max_transition_altitude_loss_m=0.3, governing_case="sluggish",
                                   perturbation_cases=cases)
    return FCSArtifacts(**{**base, **updates})


def test_fcs_rows_name_the_governing_case_and_list_each_plant() -> None:
    rows = {r["Check"]: r for r in presenters.fcs_rows(_fcs_with_cases())}
    worst = rows["Worst case (of nominal, hot, ringing, sluggish)"]
    assert worst["Result"] == "sluggish" and worst["Pass"] == "no"
    assert rows["Plant: ringing"]["Pass"] == "no" and rows["Plant: ringing"]["Result"].startswith("roll 25%")
    assert rows["Plant: hot"]["Result"] == "roll 12% / 1.4 s, pitch 8% / 1.1 s"
    assert rows["Plant: hot"]["Pass"] == "yes"
    assert rows["Plant: sluggish"]["Result"] == "crashed: SitlFlightError: stalled"
    assert rows["Plant: sluggish"]["Pass"] == "no"
    assert rows["Roll overshoot, worst plant (+15° step)"]["Result"] == "12%"
    assert {"Roll settling time, worst plant", "Pitch overshoot, worst plant (+10° step)",
            "Pitch settling time, worst plant"} <= set(rows)
    old = FCSArtifacts(param_path="a", sitl_report_path="b", passed=True, roll_overshoot=0.05, pitch_overshoot=0.05,
                       roll_settling_s=1.0, pitch_settling_s=1.0, max_transition_altitude_loss_m=0.0)
    assert [r["Check"] for r in presenters.fcs_rows(old)] == [
        "Roll overshoot (+15° step)", "Roll settling time", "Pitch overshoot (+10° step)", "Pitch settling time",
        "Transition altitude loss", "Rate-gain scale / attempts"]


def test_downloadable_files_stay_inside_the_run_folder(run: pipeline.DesignRun) -> None:
    files = presenters.downloadable_files(run.out_dir)
    assert {f.name for f in files} >= {"requirements.json", "sizing.json", "bom.json"}
    assert all(f.resolve().parent == run.out_dir.resolve() for f in files)


def test_the_pdf_report_is_downloadable(tmp_path: Path) -> None:
    (tmp_path / "design_report.pdf").write_bytes(b"%PDF-1.4")
    assert [f.name for f in presenters.downloadable_files(tmp_path)] == ["design_report.pdf"]


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


SELECT_FIELDS = ("airframe_topology", "launch_method", "propulsion_type", "group_constraint")
NO_LLM_PROBLEM = "no LLM in this test"


def _fill_form_widget(at: Any, key: str, value: object) -> None:
    if key in ("flight_controller", "payload_model"):  # these redraw (Jetson box) or refill (mass, power) others
        at.selectbox(key=key).set_value(value)
        at.run()
    elif key == "companion_computer":
        if at.session_state["flight_controller"] == "cube_orange_plus":
            box = at.checkbox(key="companion_computer_cube")
            box.check() if value else box.uncheck()
            at.run()
    elif isinstance(value, bool):
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
    # Payload model and avionics first: choosing a model refills mass/power; the stack decides the Jetson box key.
    first = ("payload_model", *presenters.HARDWARE_KEYS)
    values = sorted(presenters.form_defaults(req).items(), key=lambda kv: kv[0] not in first)
    for key, value in values:
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
    at.button(key="select_option_0").click()  # same option again, panel still open: the edit must not stick
    at.run()
    assert at.number_input(key="review_assumptions_fuel_reserve_fraction").value == pytest.approx(0.15)
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


def test_chat_tests_a_what_if_and_hands_a_proposal_to_the_review_panel(tmp_path: Path,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    from src.agent.agent_schemas import DesignerTurn
    from src.agent.design_chat import to_change_set
    from tests.agent_fakes import ScriptedLLM
    from tests.test_design_chat import _endurance

    turns = [DesignerTurn(message="3 h closes", what_ifs=[_endurance(3.0)], proposal=_endurance(3.0), done=True)]
    at = _form_app_with(tmp_path, monkeypatch, REQ_FAILED, llm=ScriptedLLM(designer_turns=turns))
    at.chat_input(key="design_chat").set_value("Keep electric?").run()
    assert not at.exception
    assert any("3 h closes" in m.markdown[0].value for m in at.chat_message if m.name == "assistant")
    assert any("Keep electric?" in m.markdown[0].value for m in at.chat_message if m.name == "user")
    assert any("1 LLM round" in c.value for c in at.caption)
    assert any("What-if" in d.value.columns for d in at.dataframe)  # the what-if table under the answer
    at.button(key="use_proposal_0").click()
    at.run()
    assert not at.exception
    assert at.session_state["pending_change"][1] == "chat"
    assert at.session_state["pending_change"][0] == to_change_set(_endurance(3.0))
    assert at.number_input(key="review_requirements_endurance_hours").value == pytest.approx(3.0)
    saved = json.loads(next((tmp_path / "outputs").glob("*/design_chat.json")).read_text())
    assert [m["role"] for m in saved["messages"]] == ["designer", "agent"]
    agent = saved["messages"][1]
    assert agent["rounds"] == 1 and agent["proposal"]["requirements"]["endurance_hours"] == 3.0
    what_if = agent["what_if_results"][0]
    assert what_if["error"] is None and what_if["result"]["reason"]  # the solver's verdict, saved verbatim
    assert what_if["change"]["requirements"]["endurance_hours"] == 3.0


def test_chat_is_disabled_without_a_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _form_app_with(tmp_path, monkeypatch, REQ_FAILED, llm=None)
    assert not [c for c in at.chat_input if c.key == "design_chat"] or at.chat_input(key="design_chat").disabled
    assert any("design chat needs an LLM" in c.value for c in at.caption)
    assert any(b.key.startswith("select_option_") for b in at.button)  # option cards still work


def test_chat_llm_error_shows_a_short_error_and_keeps_the_history(tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    from src.agent.agent_schemas import DesignerTurn
    from tests.agent_fakes import ScriptedLLM

    turns = [DesignerTurn(message="Ask me about [x](http://evil.example) margins", what_ifs=[], proposal=None,
                          done=True)]
    llm = ScriptedLLM(designer_turns=turns)
    at = _form_app_with(tmp_path, monkeypatch, REQ_FAILED, llm=llm)
    at.chat_input(key="design_chat").set_value("Hello").run()
    assistant = [m.markdown[0].value for m in at.chat_message if m.name == "assistant"]
    assert assistant and "\\[x\\]" in assistant[0] and "[x](" not in assistant[0]
    assert not [b for b in at.button if b.key.startswith("use_proposal_")]  # no proposal, no button
    llm.fail_on = (DesignerTurn,)
    at.chat_input(key="design_chat").set_value("Again").run()
    assert not at.exception
    assert any("scripted failure" in e.value for e in at.error)
    assert len([m for m in at.chat_message if m.name == "assistant"]) == 1
    assert any(b.key.startswith("select_option_") for b in at.button)


# ------------------------------------------------------ final review fixes ---
def _change_json(label: str, **req: object) -> dict[str, object]:
    from src.schemas.changes import ChangeSet

    return ChangeSet.model_validate({"label": label, "requirements": req, "assumptions": {}, "knobs": {}}).model_dump()


def _show(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, view: Any) -> Any:
    """Render `view` as the result of a description run (no solver, no LLM call)."""
    from src.ui import agent_service

    at = _llm_app(tmp_path, monkeypatch)
    monkeypatch.setattr(agent_service, "run", lambda *a, **k: view)
    at.button(key="submit_prompt").click()
    at.run()
    assert not at.exception
    return at


def test_what_if_rows_are_plain_text_at_solver_precision_with_the_floor_note() -> None:  # findings 2, 3, 9
    from src.agent.design_chat import ChatMessage, WhatIfResult
    from src.agent.evaluator import Evaluation
    from src.schemas.changes import ChangeSet
    from src.ui import agent_service, chat_panel

    change = ChangeSet.model_validate({"label": "Try [x] *bold*", "requirements": {}, "assumptions": {}, "knobs": {}})
    result = Evaluation(True, 16.623456789, 2, "series_hybrid", 0.8, 3.4, "", reserve_floor_applied=0.12)
    message = ChatMessage(role="agent", text="x", what_ifs=(WhatIfResult(change, result, "bad \\ value [y]"),))
    view = agent_service.AgentRunView("not_converged", run_id="r1", requirements=REQ_FAILED)
    (row,) = chat_panel._what_if_rows(message, view)
    assert row["What-if"] == "Try [x] *bold*" and row["Change"] == "Try [x] *bold*"
    assert row["MTOW (kg)"] == "16.62"
    assert row["Note"].startswith("bad \\ value [y]")
    assert "sized with the 15 % reserve floor (requested 12 %)" in row["Note"]


def test_option_outcome_uses_solver_precision_and_names_the_reserve_floor() -> None:  # findings 2, 3
    from src.ui.design_loop import _outcome_text

    text = _outcome_text({"mtow_kg": 16.623456789, "dod_group": 2, "battery_mass_kg": 0.8,
                          "fuel_mass_kg": 3.4567891, "reserve_floor_applied": 0.1})
    assert "MTOW 16.62 kg" in text and "fuel 3.46 kg" in text
    assert "sized with the 15 % reserve floor (requested 10 %)" in text


def test_no_closing_option_gets_an_honest_heading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # 7
    from src.ui import agent_service

    result = {"closes": False, "mtow_kg": 40.0, "dod_group": 2, "propulsion": "electric", "battery_mass_kg": 20.0,
              "fuel_mass_kg": 0.0, "reason": "mass spiral"}
    options = ({"lever": "endurance", "change": _change_json("endurance", endurance_hours=0.5), "result": result,
                "relative_change": 0.0, "closes": False, "mtow_saved_kg": None, "group_drop": False},)
    view = agent_service.AgentRunView("not_converged", run_id="r1", out_dir=tmp_path, requirements=REQ_FAILED,
                                      explorer_mode="closure", options=options)
    headings = [s.value for s in _show(tmp_path, monkeypatch, view).subheader]
    assert "No single change closes the design (checked with sizing + catalog)" in headings
    assert "Options that would close the design" not in headings


def test_airframe_chat_opens_with_the_failed_gates_and_the_knobs(tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:  # 1, 8
    from src.ui import agent_service

    gates = ({"name": "static_margin", "passed": False, "detail": "SM 3 % below the 5 % minimum"},
             {"name": "lift_to_drag", "passed": True, "detail": "L/D fine"})
    note = "Sizing + catalog close unchanged; the failure is in static_margin"
    view = agent_service.AgentRunView("not_converged", run_id="r1", out_dir=tmp_path, requirements=G1_ELECTRIC_FW,
                                      explorer_mode="airframe", gates=gates, explorer_note=note)
    at = _show(tmp_path, monkeypatch, view)
    first = next(m for m in at.chat_message if m.name == "assistant").markdown[0].value
    assert "static_margin" in first and "SM 3 % below the 5 % minimum" in first and "L/D fine" not in first
    assert "Aspect ratio (4–25)" in first and "Fuselage length factor (0.6–2.5)" in first
    assert any(note in i.value for i in at.info)
    assert not (tmp_path / "design_chat.json").exists()  # the opening is not an LLM turn and is not saved


# ------------------------------------------------------------ 3D viewer tab ---
def _viewer_view(run: pipeline.DesignRun, out_dir: Optional[Path]) -> Any:
    from src.ui import agent_service

    return agent_service.AgentRunView("converged", run_id="r1", out_dir=out_dir, requirements=G1_ELECTRIC_FW,
                                      sizing=run.sizing)


def test_3d_tab_offers_the_viewer_download_when_the_file_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                run: pipeline.DesignRun) -> None:
    (tmp_path / "viewer.html").write_text("<html><body>viewer</body></html>", encoding="utf-8")
    at = _show(tmp_path, monkeypatch, _viewer_view(run, tmp_path))
    assert any(b.label == "Download viewer.html" for b in at.get("download_button"))
    labels = [t.label for t in at.tabs]
    assert labels[:5] == ["Sizing", "Components", "Aero & CAD", "3D", "Flight test"]
    assert not any("3D viewer" in i.value for i in at.info)


def test_3d_tab_explains_when_there_is_no_output_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                        run: pipeline.DesignRun) -> None:
    at = _show(tmp_path, monkeypatch, _viewer_view(run, None))
    assert any("3D viewer" in i.value for i in at.info)
    assert not any(b.label == "Download viewer.html" for b in at.get("download_button"))


def test_3d_tab_explains_when_the_run_folder_has_no_viewer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                           run: pipeline.DesignRun) -> None:
    at = _show(tmp_path, monkeypatch, _viewer_view(run, tmp_path))
    assert any("3D viewer" in i.value and "Docker" in i.value for i in at.info)
    assert not any(b.label == "Download viewer.html" for b in at.get("download_button"))


def test_render_viewer_embeds_the_page_at_the_fixed_height(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                           run: pipeline.DesignRun) -> None:
    from src.ui import viewer_panel

    (tmp_path / "viewer.html").write_text("<html>PAGE</html>", encoding="utf-8")
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(viewer_panel.components, "html", lambda html, height: calls.append((html, height)))
    viewer_panel.render_viewer(_viewer_view(run, tmp_path))
    assert calls == [("<html>PAGE</html>", viewer_panel.VIEWER_HEIGHT_PX)]
    assert viewer_panel.VIEWER_HEIGHT_PX == 640


def test_render_viewer_sends_the_page_once_and_the_download_lazily(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                   run: pipeline.DesignRun) -> None:
    from src.ui import viewer_panel

    (tmp_path / "viewer.html").write_text("<html>PAGE</html>", encoding="utf-8")
    pages: list[str] = []
    downloads: list[object] = []
    monkeypatch.setattr(viewer_panel.components, "html", lambda html, height: pages.append(html))
    monkeypatch.setattr(viewer_panel.st, "download_button", lambda label, data, **_kw: downloads.append(data))
    viewer_panel.render_viewer(_viewer_view(run, tmp_path))
    assert pages == ["<html>PAGE</html>"]
    (data,) = downloads
    assert callable(data) and data() == b"<html>PAGE</html>"  # read only when the user downloads


def test_render_viewer_reads_an_unchanged_page_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                    run: pipeline.DesignRun) -> None:
    from src.ui import viewer_panel

    viewer_panel.read_viewer_page.clear()
    page = tmp_path / "viewer.html"
    page.write_text("<html>PAGE</html>", encoding="utf-8")
    monkeypatch.setattr(viewer_panel.components, "html", lambda html, height: None)
    reads: list[str] = []
    real = viewer_panel.Path.read_text
    monkeypatch.setattr(viewer_panel.Path, "read_text", lambda self, **kw: reads.append(str(self)) or real(self, **kw))
    for _ in range(3):
        viewer_panel.render_viewer(_viewer_view(run, tmp_path))
    assert len(reads) == 1


def test_render_viewer_warns_when_the_page_is_not_valid_utf8(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                              run: pipeline.DesignRun) -> None:
    from src.ui import viewer_panel

    (tmp_path / "viewer.html").write_bytes(b"\xff\xfe\x00bad")
    warnings: list[str] = []
    monkeypatch.setattr(viewer_panel.st, "warning", warnings.append)
    viewer_panel.render_viewer(_viewer_view(run, tmp_path))
    assert len(warnings) == 1 and "could not be read" in warnings[0]


# ------------------------------------------------------------------ avionics ---
from src.schemas.assumptions import DEFAULT_ASSUMPTIONS  # noqa: E402


def _avionics_req(**update: object) -> OperationalRequirements:
    return OperationalRequirements(**{**G1_ELECTRIC_FW.model_dump(), **update})


def test_avionics_inputs_default_to_cube_with_jetson(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _llm_app(tmp_path, monkeypatch)
    assert at.selectbox(key="flight_controller").value == "cube_orange_plus"
    box = at.checkbox(key="companion_computer_cube")
    assert box.value is True and not box.disabled
    assert at.session_state["companion_computer"] is True


def test_jetson_checkbox_greys_out_live_under_ark(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _llm_app(tmp_path, monkeypatch)
    at.checkbox(key="companion_computer_cube").uncheck()
    at.run()
    at.selectbox(key="flight_controller").set_value("ark_jetson_pab")
    at.run()
    shown = at.checkbox(key="companion_computer_ark")
    assert shown.disabled and shown.value is True
    at.selectbox(key="flight_controller").set_value("cube_orange_plus")
    at.run()
    assert at.checkbox(key="companion_computer_cube").value is False  # the earlier Cube choice survives
    assert not at.checkbox(key="companion_computer_cube").disabled


def test_preset_change_keeps_avionics_choice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _llm_app(tmp_path, monkeypatch)
    at.checkbox(key="companion_computer_cube").uncheck()
    at.run()
    at.selectbox(key="preset").set_value(list(EXAMPLE_MISSIONS)[1])
    at.run()
    assert at.session_state["companion_computer"] is False
    assert at.checkbox(key="companion_computer_cube").value is False


def test_avionics_group_in_assumptions_panel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _llm_app(tmp_path, monkeypatch)
    assert at.number_input(key="assume_jetson_power_w").value == pytest.approx(25.0)
    assert at.number_input(key="assume_ark_pab_stack_mass_kg").value == pytest.approx(0.166)


def test_avionics_summary_texts() -> None:
    summary = presenters.avionics_summary
    assert summary(_avionics_req(companion_computer=True), DEFAULT_ASSUMPTIONS) == \
        "Cube Orange+ · Jetson Orin NX companion (25 W, 0.2 kg)"
    assert summary(_avionics_req(), DEFAULT_ASSUMPTIONS) == "Cube Orange+ · no Jetson"
    assert summary(_avionics_req(flight_controller="ark_jetson_pab"), DEFAULT_ASSUMPTIONS) == \
        "ARK Jetson PAB + Orin NX 16 GB (25 W, 0.166 kg)"


def test_report_requirements_list_avionics() -> None:
    from src.agent import report
    rows = dict(report.requirement_rows(_avionics_req(flight_controller="ark_jetson_pab")))
    assert rows["Flight controller"] == "ARK Jetson PAB (Orin NX 16 GB)"
    assert rows["Jetson companion computer"] == "built into the ARK stack"


def test_avionics_column_caption(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _llm_app(tmp_path, monkeypatch)
    assert any(c.value == "A description sets its own avionics." for c in at.caption)
