"""Milestone 5 unit tests: parsing at the I/O boundary, the critic gates and knob steps, graph routing
(clarification, convergence, iteration cap, tool failures, LLM outages), the report and the CLI.
All run on the host with a scripted LLM and aero/CAD off; tests/test_agent_e2e.py runs the full stack."""

import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from src import pipeline
from src.agent import critic, nodes, report
from src.agent.agent_schemas import (Adjustment, CriticDecision, Distance, Duration, Mass, Power, ReportNarrative,
                                     OptionsExplanation, RequirementsDraft, Speed)
from src.agent.graph import run_agent
from src.agent.llm import ClaudeLLM, LLMError
from src.agent.nodes import AgentContext
from src.agent.parsing import DEFAULT_CRUISE_SPEED_MS, to_requirements
from src.schemas.assumptions import DesignAssumptions
from src.schemas.fabrication import DEFAULT_FABRICATION, FabricationSettings
from src.schemas.state import CADArtifacts
from src.tools.aero_cad import stability
from src.viewer.build import ViewerError
from src.main import EXIT_CLARIFY, EXIT_CONVERGED, EXIT_ERROR, EXIT_NOT_CONVERGED, main
from src.tools.sizing.layouts import TAILED_BAND, StaticMarginBand
from src.tools.sizing.mission_solver import SizingDivergenceError, size_aircraft
from tests.agent_fakes import SRS_DRAFT, SRS_PROMPT, ScriptedLLM
from tests.reference_designs import G1_ELECTRIC_FW, REQ_FAILED
from tests.test_flight_control import load_design


@pytest.fixture(scope="module")
def db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from src.tools.database.seed_data import seed_database

    path = tmp_path_factory.mktemp("db") / "catalog.db"
    seed_database(path)
    return path


def _ctx(tmp_path: Path, db: Path, llm: Any, **kw: Any) -> AgentContext:
    return AgentContext(llm=llm, outputs_root=tmp_path / "outputs", db_path=db, with_aero_cad=False,
                        with_sitl=False, **kw)


# ------------------------------------------------------------------- parsing ---
def test_draft_units_are_converted_to_si(db: Path) -> None:
    draft = RequirementsDraft(mission_name="Imperial", payload_mass=Mass(value=11, unit="lb"),
                              payload_power=Power(value=0.1, unit="kW"), endurance=Duration(value=90, unit="min"),
                              radius=Distance(value=40, unit="nmi"), cruise_speed=Speed(value=50, unit="kt"),
                              runway_available=True)
    req = to_requirements(draft, db).requirements
    assert req is not None
    assert req.payload_mass_kg == pytest.approx(4.9895, abs=1e-3) and req.payload_power_w == pytest.approx(100.0)
    assert req.endurance_hours == pytest.approx(1.5) and req.combat_radius_km == pytest.approx(74.08)
    assert req.cruise_speed_ms == pytest.approx(25.72, abs=0.01) and req.runway_available


def test_unstated_values_become_reported_assumptions(db: Path) -> None:
    draft = SRS_DRAFT.model_copy(update={"radius": None, "total_range": Distance(value=200, unit="km")})
    outcome = to_requirements(draft, db)
    assert outcome.requirements is not None
    assert outcome.requirements.combat_radius_km == pytest.approx(100.0)
    assert outcome.requirements.cruise_speed_ms == DEFAULT_CRUISE_SPEED_MS
    assert outcome.requirements.airframe_topology == "quadplane_vtol" and outcome.requirements.group_constraint == 2
    text = " ".join(outcome.assumptions)
    assert "half the stated total range" in text and "Cruise speed not stated" in text and "catalog sensor" in text


@pytest.mark.parametrize("missing", ["payload_mass", "endurance"])
def test_missing_payload_or_endurance_asks_for_clarification(db: Path, missing: str) -> None:
    outcome = to_requirements(SRS_DRAFT.model_copy(update={missing: None}), db)
    assert outcome.requirements is None and "Please specify the" in outcome.clarification


def test_out_of_envelope_values_ask_for_revision(db: Path) -> None:
    outcome = to_requirements(SRS_DRAFT.model_copy(update={"payload_mass": Mass(value=500, unit="kg")}), db)
    assert outcome.requirements is None and "outside the supported envelope" in outcome.clarification


# -------------------------------------------------------------------- critic ---
def test_gates_on_a_balanced_design_pass() -> None:
    req, sizing, bom, aero, cad = load_design("g2")
    gates = critic.evaluate_gates(req, sizing, bom, aero, cad)
    assert [g.name for g in gates] == ["static_margin", "lift_to_drag", "collisions_and_group", "reserve"]
    assert all(g.passed for g in gates), gates


def test_each_gate_detects_its_failure() -> None:
    req, sizing, bom, aero, cad = load_design("g2")
    bad_cad = cad.model_copy(update={"static_margin": 0.02, "collision_count": 2})
    bad_aero = aero.model_copy(update={"induced_drag_factor": 3.0 * aero.induced_drag_factor})
    hybrid = sizing.model_copy(update={"propulsion_type": "series_hybrid", "energy_reserve_fraction": 0.10})
    failed = {g.name for g in critic.evaluate_gates(req, hybrid, bom, bad_aero, bad_cad) if not g.passed}
    assert failed == {"static_margin", "lift_to_drag", "collisions_and_group", "reserve"}
    heavy = cad.model_copy(update={"as_built_mass_kg": 30.0})
    group = next(g for g in critic.evaluate_gates(req, sizing, bom, aero, heavy) if g.name == "collisions_and_group")
    assert not group.passed and "Group 2" in group.detail


def test_ld_gate_compares_both_models_at_the_sizing_cruise_and_loiter_lift_coefficients() -> None:
    _, sizing, _, aero, _ = load_design("g2")
    points = {p.name: p for p in stability.ld_deviations(sizing, aero)}
    assert set(points) == {"cruise", "loiter"}
    k_sizing: float = 1.0 / (math.pi * sizing.oswald_e * sizing.aspect_ratio)
    for name, cl in (("cruise", sizing.cl_cruise), ("loiter", sizing.cl_loiter)):
        assert points[name].cl == cl
        assert points[name].sizing_ld == pytest.approx(cl / (sizing.cd0 + k_sizing * cl**2))
        assert points[name].vspaero_ld == pytest.approx(cl / (aero.cd0 + aero.induced_drag_factor * cl**2))


def test_ld_gate_is_two_sided_and_only_a_low_vspaero_ld_raises_the_aspect_ratio() -> None:
    req, sizing, bom, aero, cad = load_design("g2")
    low = aero.model_copy(update={"induced_drag_factor": 2.0 * aero.induced_drag_factor})
    high = aero.model_copy(update={"cd0": 0.5 * aero.cd0, "induced_drag_factor": 0.5 * aero.induced_drag_factor})
    gate = {name: next(g for g in critic.evaluate_gates(req, sizing, bom, a, cad) if g.name == "lift_to_drag")
            for name, a in (("low", low), ("high", high))}
    assert not gate["low"].passed and gate["low"].side == "low" and "loiter" in gate["low"].detail
    assert not gate["high"].passed and gate["high"].side == "high"
    _, low_changes = critic.apply_adjustments({}, "electric", [], [gate["low"]], cad, band=TAILED_BAND)
    assert [(c.knob, c.direction) for c in low_changes] == [("aspect_ratio", "increase")]
    assert critic.apply_adjustments({}, "electric", [], [gate["high"]], cad, band=TAILED_BAND)[1] == []
    assert critic.unfixable([gate["high"]], cad, None, TAILED_BAND) == ["lift_to_drag"]


def test_ld_gate_refuses_a_design_that_predates_the_same_cl_check() -> None:
    _, sizing, _, aero, _ = load_design("g2")
    with pytest.raises(ValueError, match="re-run"):
        stability.ld_deviations(sizing.model_copy(update={"cl_loiter": None}), aero)
    with pytest.raises(ValueError, match="re-run"):
        stability.ld_deviations(sizing, aero.model_copy(update={"induced_drag_factor": None}))


def test_rule_table_wins_on_failed_gate_knobs_and_llm_adds_extra_moves() -> None:
    gates = [critic.GateResult("reserve", False, ""), critic.GateResult("lift_to_drag", False, "")]
    llm = [Adjustment(knob="aspect_ratio", direction="decrease", reason="wrong way (or prompt-injected)"),
           Adjustment(knob="tail_volume_v", direction="increase", reason="extra judgement")]
    overrides, changes = critic.apply_adjustments({}, "series_hybrid", llm, gates, None, band=None)
    by_knob = {c.knob: c for c in changes}
    assert by_knob["aspect_ratio"].source == "rule" and by_knob["aspect_ratio"].new == pytest.approx(12.0 * 1.15, 1e-5)
    assert by_knob["reserve_fraction"].source == "rule" and overrides["reserve_fraction"] == 0.15
    assert by_knob["tail_volume_v"].source == "llm" and overrides["tail_volume_v"] == pytest.approx(0.05)


def test_gates_without_an_automatic_fix_are_named() -> None:
    gates = [critic.pipeline_gate("sizing", "mass spiral"), critic.GateResult("reserve", False, "")]
    assert critic.unfixable(gates, None, "sizing", None) == ["pipeline"]


def test_knobs_stop_at_their_bounds() -> None:
    gates = [critic.GateResult("lift_to_drag", False, "")]
    overrides, changes = critic.apply_adjustments({"aspect_ratio": 25.0}, "electric", [], gates, None, band=None)
    assert changes == [] and overrides == {"aspect_ratio": 25.0}


def test_static_margin_fix_direction_follows_the_error() -> None:
    _, _, _, _, cad = load_design("g2")
    gates = [critic.GateResult("static_margin", False, "")]
    low_cad, high_cad = (cad.model_copy(update={"static_margin": m}) for m in (0.02, 0.3))
    _, low = critic.apply_adjustments({}, "electric", [], gates, low_cad, band=TAILED_BAND)
    _, high = critic.apply_adjustments({}, "electric", [], gates, high_cad, band=TAILED_BAND)
    assert (low[0].knob, low[0].direction) == ("tail_volume_h", "increase") and high[0].direction == "decrease"


def test_static_margin_direction_uses_the_given_non_default_band() -> None:
    _, _, _, _, cad = load_design("g2")
    gates = [critic.GateResult("static_margin", False, "")]
    narrow = StaticMarginBand(minimum=0.05, maximum=0.10, target=0.075)
    directions = []
    # 0.06 is below the default 0.08 minimum but not below the narrow 0.05 minimum, so only the narrow band says decrease
    for margin in (0.04, 0.06, 0.12):
        _, changes = critic.apply_adjustments({}, "electric", [], gates, cad.model_copy(update={"static_margin": margin}),
                                              band=narrow)
        directions.append(changes[0].direction)
    assert directions == ["increase", "decrease", "decrease"]


# --------------------------------------------------------------------- graph ---
def test_hybrid_reserve_gate_forces_one_critic_iteration(tmp_path: Path, db: Path) -> None:
    llm = ScriptedLLM()
    state = run_agent(_ctx(tmp_path, db, llm), SRS_PROMPT)
    assert state["converged"] and state["iteration_count"] == 1
    assert state["sizing_overrides"] == {"reserve_fraction": 0.15}
    assert state["critic_log"][0]["failed"] == ["reserve"] and state["critic_log"][1]["failed"] == []
    out = Path(state["out_dir"])
    for name in ("requirements.json", "sizing.json", "bom.json", "critic_log.json", "design_report.md",
                 "design_report.pdf", "run_summary.json"):
        assert (out / name).is_file(), name
    assert json.loads((out / "run_summary.json").read_text())["report_pdf"] == "design_report.pdf"
    text = (out / "design_report.md").read_text()
    assert "status **CONVERGED** after 1 critic iteration" in text and "reserve_fraction 0.1 -> 0.15" in text
    assert [name for name, _ in llm.calls] == ["RequirementsDraft", "CriticDecision", "ReportNarrative"]


def test_clarification_ends_the_run_without_a_run_folder(tmp_path: Path, db: Path) -> None:
    llm = ScriptedLLM(draft=SRS_DRAFT.model_copy(update={"endurance": None}))
    state = run_agent(_ctx(tmp_path, db, llm), "Design a drone with a 3.5 kg gimbal")
    assert "endurance" in state["clarification_needed"] and state["run_id"] is None
    assert not (tmp_path / "outputs").exists()


def test_parser_outage_is_a_fatal_error(tmp_path: Path, db: Path) -> None:
    state = run_agent(_ctx(tmp_path, db, ScriptedLLM(fail_on=(RequirementsDraft,))), SRS_PROMPT)
    assert "Could not parse" in state["fatal_error"] and state["final_report_path"] is None


def test_critic_and_report_outages_fall_back_to_rules_and_template(tmp_path: Path, db: Path) -> None:
    llm = ScriptedLLM(fail_on=(CriticDecision, ReportNarrative))
    state = run_agent(_ctx(tmp_path, db, llm), SRS_PROMPT)
    assert state["converged"] and "LLM error" in state["critic_log"][0]["summary"]
    assert "Narrative generated without an LLM" in Path(state["final_report_path"]).read_text()


def test_iteration_cap_reports_non_convergence(tmp_path: Path, db: Path) -> None:
    state = run_agent(_ctx(tmp_path, db, ScriptedLLM(), max_iterations=0), SRS_PROMPT)
    assert not state["converged"] and state["critic_exhausted"] and state["iteration_count"] == 0
    assert "NON-CONVERGENCE" in Path(state["final_report_path"]).read_text()


def test_tool_failure_goes_to_the_critic_and_then_the_report(tmp_path: Path, db: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    def diverge(*args: Any, **kwargs: Any) -> Any:
        raise SizingDivergenceError("mass spiral")

    monkeypatch.setattr("src.agent.nodes.size_aircraft", diverge)
    state = run_agent(_ctx(tmp_path, db, ScriptedLLM()), SRS_PROMPT)
    assert not state["converged"] and state["critic_log"][0]["failed"] == ["pipeline"]
    assert "sizing failed: mass spiral" in Path(state["final_report_path"]).read_text()


def test_structured_requirements_skip_the_parser(tmp_path: Path, db: Path) -> None:
    llm = ScriptedLLM(draft=None)
    state = run_agent(_ctx(tmp_path, db, llm), requirements=G1_ELECTRIC_FW)
    assert state["converged"] and state["iteration_count"] == 0 and "RequirementsDraft" not in dict(llm.calls)


# -------------------------------------------------------------------- report ---
def test_polar_and_tables() -> None:
    _, sizing, _, aero, _ = load_design("g1")
    rows = report.polar_rows(aero, sizing)
    assert [r["alpha (deg)"] for r in rows] == list(report.POLAR_ALPHAS_DEG)
    assert rows[1]["CL"] == pytest.approx(aero.cl0, abs=1e-3) and all(r["CD"] > aero.cd0 for r in rows)
    assert report.table([]) == "_none_\n" and report.table([{"a": 1, "b": None}]).splitlines()[2] == "| 1 |  |"


# ----------------------------------------------------------------------- CLI ---
def test_cli_with_a_spec_file_runs_without_an_llm(tmp_path: Path, db: Path) -> None:
    spec = tmp_path / "req.json"
    spec.write_text(G1_ELECTRIC_FW.model_dump_json())
    code = main([f"--spec={spec}", f"--outputs={tmp_path / 'o'}", f"--db={db}", "--no-aero", "--quiet"])
    assert code == EXIT_CONVERGED
    (summary,) = (tmp_path / "o").glob("*/run_summary.json")
    assert json.loads(summary.read_text())["converged"] is True


def test_cli_reports_clarification_and_errors(tmp_path: Path, db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.main.make_llm",
                        lambda: ScriptedLLM(draft=SRS_DRAFT.model_copy(update={"payload_mass": None})))
    assert main(["a drone", f"--outputs={tmp_path / 'o'}", f"--db={db}", "--no-aero", "--quiet"]) == EXIT_CLARIFY
    assert main([]) == EXIT_ERROR
    from src.agent import providers

    monkeypatch.setattr("src.main.make_llm", providers.make_llm)  # real factory, no key -> fail fast, not mid-run
    monkeypatch.setenv("ISR_FORGE_LLM_PROVIDER", "openrouter")
    assert main(["a drone", f"--outputs={tmp_path / 'o2'}", "--no-aero", "--quiet"]) == EXIT_ERROR
    assert not (tmp_path / "o2").exists()
    assert main([f"--spec={tmp_path / 'missing.json'}", "--no-aero", "--quiet"]) == EXIT_ERROR


# ------------------------------------------------------------- Claude adapter ---
def _fake_client(response: Any) -> Any:
    messages = SimpleNamespace(parse=lambda **kwargs: response)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages))


def test_claude_adapter_returns_parsed_output_and_rejects_refusals() -> None:
    ok = SimpleNamespace(stop_reason="end_turn", parsed_output=CriticDecision(summary="x"), _request_id="req_1")
    assert ClaudeLLM("claude-opus-5-5", _fake_client(ok)).parse("s", "p", CriticDecision).summary == "x"
    refused = SimpleNamespace(stop_reason="refusal", parsed_output=None, _request_id="req_2")
    with pytest.raises(LLMError, match="declined"):
        ClaudeLLM("claude-opus-5-5", _fake_client(refused)).parse("s", "p", CriticDecision)
    truncated = SimpleNamespace(stop_reason="max_tokens", parsed_output=None, _request_id="req_3")
    with pytest.raises(LLMError, match="no CriticDecision"):
        ClaudeLLM("claude-opus-5-5", _fake_client(truncated)).parse("s", "p", CriticDecision)


def test_claude_adapter_sends_structured_output_with_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def capture(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return SimpleNamespace(stop_reason="end_turn", parsed_output=CriticDecision(summary="x"), _request_id="r")

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(parse=capture)))
    monkeypatch.setenv("ISR_FORGE_MODEL", "claude-sonnet-5-5")
    ClaudeLLM(client=client).parse("sys", "prompt", CriticDecision, effort="low")
    assert seen["model"] == "claude-sonnet-5-5" and seen["output_format"] is CriticDecision
    assert seen["fallbacks"] == "default" and seen["output_config"] == {"effort": "low"}


# ------------------------------------------------------- review regressions ---
def test_missing_credentials_and_schema_errors_become_llm_errors() -> None:
    def no_auth(**kwargs: Any) -> Any:
        raise TypeError("Could not resolve authentication method")

    def bad_json(**kwargs: Any) -> Any:
        raise ValueError("invalid JSON in structured output")

    for failure in (no_auth, bad_json):
        client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(parse=failure)))
        with pytest.raises(LLMError, match="Claude call failed"):
            ClaudeLLM("claude-opus-5-5", client).parse("s", "p", CriticDecision)


def test_rejected_credentials_give_a_fixed_message_without_the_response_body() -> None:
    import anthropic
    import httpx

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    body = "invalid x-api-key sk-ant-SECRET"

    def unauthorized(**kwargs: Any) -> Any:
        raise anthropic.AuthenticationError(body, response=httpx.Response(401, request=request), body=None)

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(parse=unauthorized)))
    with pytest.raises(LLMError, match="rejected the credentials") as info:
        ClaudeLLM("claude-opus-5-5", client).parse("s", "p", CriticDecision)
    assert "SECRET" not in str(info.value)


def test_llm_narrative_cannot_render_links_or_remote_images(tmp_path: Path) -> None:
    hostile = "See ![x](https://evil.example/?q=mtow) and [docs](https://evil.example) <img src=x>"
    assert "](" not in report.defang(hostile).replace("\\](", "")  # every ']' before '(' is escaped
    assert "<img" not in report.defang(hostile)
    assert report.inline("[a](b)") == "\\[a\\](b)"


def test_report_cells_and_headings_resist_markdown_injection() -> None:
    assert report.cell("a | b\n| injected |") == "a \\| b \\| injected \\|"
    assert report.inline("## Title\n# more") == "Title # more"
    assert report.table([{"x": "1|2"}]).splitlines()[2] == "| 1\\|2 |"


def test_narrative_numbers_not_in_the_facts_are_flagged() -> None:
    narrative = ReportNarrative(executive_summary="MTOW is 13.43 kg with 42.7 km of range.", trade_offs="Group 2.",
                                risks_and_next_steps="Fly 3 tests.")
    assert report.unsupported_numbers(narrative, '{"MTOW": "13.43 kg"}') == ["42.7"]


def test_unexpected_exception_leaves_a_failed_summary(tmp_path: Path, db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise ZeroDivisionError("bug in a tool")

    monkeypatch.setattr("src.agent.nodes.select_bom", boom)
    with pytest.raises(ZeroDivisionError):
        run_agent(_ctx(tmp_path, db, ScriptedLLM()), SRS_PROMPT)
    (summary,) = (tmp_path / "outputs").glob("*/run_summary.json")
    data = json.loads(summary.read_text())
    assert data["status"] == "failed" and "ZeroDivisionError" in data["error"]


def test_sizing_failure_keeps_the_requested_powertrain_for_the_reserve_baseline() -> None:
    from src.agent.nodes import _propulsion

    hybrid_req = SimpleNamespace(propulsion_type="series_hybrid")
    assert _propulsion({"sizing_results": None, "requirements": hybrid_req}) == "series_hybrid"  # type: ignore[arg-type]
    assert _propulsion({"sizing_results": None, "requirements": SimpleNamespace(propulsion_type="auto")}) == "electric"  # type: ignore[arg-type]


def test_sitl_failure_is_reported_in_status_and_exit_code() -> None:
    from src.agent.nodes import run_status, sitl_passed

    failed_fcs = SimpleNamespace(passed=False)
    assert run_status({"converged": True, "fcs_artifacts": failed_fcs}) == "converged_sitl_failed"  # type: ignore[arg-type]
    assert run_status({"converged": True, "sitl_error": "aborted"}) == "converged_sitl_failed"  # type: ignore[arg-type]
    assert sitl_passed({"converged": True}) is None and run_status({"converged": False}) == "not_converged"  # type: ignore[arg-type]


def test_parser_echoes_quantities_and_respects_no_vtol(db: Path) -> None:
    outcome = to_requirements(SRS_DRAFT.model_copy(update={"vtol_required": False, "runway_available": None}), db)
    assert outcome.requirements is not None and outcome.requirements.airframe_topology == "fixed_wing"
    assert outcome.assumptions[0].startswith("Read from the request: payload mass 3.5 kg, endurance 6 h, radius 80 km")


def test_cli_validates_iterations_and_rejects_prompt_plus_spec(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["x", "--max-iterations=-1"])
    with pytest.raises(SystemExit):
        main(["x", "--max-iterations=999"])
    spec = tmp_path / "s.json"
    spec.write_text(G1_ELECTRIC_FW.model_dump_json())
    assert main(["a prompt", f"--spec={spec}"]) == EXIT_ERROR


# ------------------------------------------------------------ design assumptions ---
def test_run_uses_and_saves_the_designer_assumptions(tmp_path: Path, db: Path) -> None:
    a = DesignAssumptions(liion_specific_energy_wh_kg=260)
    state = run_agent(_ctx(tmp_path, db, None), requirements=G1_ELECTRIC_FW, design_assumptions=a)
    out = Path(state["out_dir"])
    assert DesignAssumptions.model_validate_json((out / "assumptions.json").read_text()) == a
    assert state["sizing_results"].mtow_kg < size_aircraft(G1_ELECTRIC_FW).mtow_kg
    text = (out / "design_report.md").read_text()
    assert "## Design assumptions" in text and "catalog battery packs are notional" in text
    assert json.loads((out / "run_summary.json").read_text())["non_default_assumptions"] == {
        "liion_specific_energy_wh_kg": [200.0, 260.0]}


def test_default_assumptions_report_no_notional_battery_note(tmp_path: Path, db: Path) -> None:
    state = run_agent(_ctx(tmp_path, db, None), requirements=G1_ELECTRIC_FW)
    out = Path(state["out_dir"])
    assert "notional" not in (out / "design_report.md").read_text()
    assert json.loads((out / "run_summary.json").read_text())["non_default_assumptions"] == {}


def test_critic_never_lowers_the_reserve_below_the_designer_floor() -> None:
    a = DesignAssumptions(battery_reserve_fraction=0.20)
    llm = [Adjustment(knob="reserve_fraction", direction="decrease", reason="x")]
    overrides, changes = critic.apply_adjustments({}, "electric", llm, [], None, None, a, band=None)
    assert overrides.get("reserve_fraction", 0.20) >= 0.20 and not changes


def test_critic_may_still_raise_the_reserve_above_the_floor() -> None:
    a = DesignAssumptions(battery_reserve_fraction=0.20)
    llm = [Adjustment(knob="reserve_fraction", direction="increase", reason="x")]
    overrides, _ = critic.apply_adjustments({}, "electric", llm, [], None, None, a, band=None)
    assert overrides["reserve_fraction"] == pytest.approx(0.25)


def test_cli_spec_accepts_an_assumptions_block(tmp_path: Path, db: Path) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({**G1_ELECTRIC_FW.model_dump(mode="json"), "assumptions": {"liion_max_c_rate": 6}}))
    assert main([f"--spec={spec}", f"--outputs={tmp_path / 'o'}", f"--db={db}", "--no-aero", "--quiet"]) == EXIT_CONVERGED
    (saved,) = (tmp_path / "o").glob("*/assumptions.json")
    assert json.loads(saved.read_text())["liion_max_c_rate"] == 6


def test_cli_spec_rejects_unknown_assumption_fields(tmp_path: Path, db: Path) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({**G1_ELECTRIC_FW.model_dump(mode="json"), "assumptions": {"warp_drive": 9}}))
    assert main([f"--spec={spec}", f"--outputs={tmp_path / 'o'}", f"--db={db}", "--no-aero", "--quiet"]) == EXIT_ERROR


# --------------------------------------------------------------- options (task 8) ---
def test_a_non_converged_sizing_failure_offers_options(tmp_path: Path, db: Path) -> None:
    state = run_agent(_ctx(tmp_path, db, ScriptedLLM()), requirements=REQ_FAILED)
    assert state["explorer_mode"] == "closure" and any(o["closes"] for o in state["explore_options"])
    out = Path(state["out_dir"])
    assert (out / "options.json").is_file()
    assert "## Options that close the design" in (out / "design_report.md").read_text()
    assert state["options_explanation"]


def test_a_closure_option_that_fails_an_airframe_gate_comes_back_in_airframe_mode(
        tmp_path: Path, db: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # Review Focus 3
    monkeypatch.setattr("src.agent.nodes._gates", lambda state: [critic.GateResult("static_margin", False, "SM 3 %")])
    state = run_agent(_ctx(tmp_path, db, None, max_iterations=0),
                      requirements=REQ_FAILED.model_copy(update={"propulsion_type": "series_hybrid"}))
    assert state["explorer_mode"] == "airframe" and state["explore_options"] == []
    assert not (Path(state["out_dir"]) / "options.json").exists()


def test_a_design_that_closes_unchanged_ends_in_airframe_mode_with_a_note(
        tmp_path: Path, db: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # final review finding 1
    gate = critic.GateResult("collisions_and_group", False, "as-built mass 26 kg is over the Group 1 limit")
    monkeypatch.setattr("src.agent.nodes._gates", lambda state: [gate])
    state = run_agent(_ctx(tmp_path, db, None, max_iterations=0), requirements=G1_ELECTRIC_FW)
    note = "Sizing + catalog close unchanged; the failure is in collisions_and_group"
    assert state["explorer_mode"] == "airframe" and state["explore_options"] == []
    assert state["explorer_note"] == note
    assert note in Path(state["final_report_path"]).read_text()
    assert not (Path(state["out_dir"]) / "options.json").exists()


def test_options_explanation_citing_rounded_solver_numbers_is_not_flagged(
        tmp_path: Path, db: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # final review finding 3
    from src.agent.evaluator import Evaluation
    from src.agent.explorer import ExploreOption
    from src.schemas.changes import ChangeSet

    change = ChangeSet.model_validate({"label": "hybrid", "requirements": {"propulsion_type": "series_hybrid"},
                                      "assumptions": {}, "knobs": {}})
    result = Evaluation(True, 16.623456789, 2, "series_hybrid", 0.81234567, 3.4567891, "")
    option = ExploreOption("propulsion", change, result, 1.0, True, None, False)
    monkeypatch.setattr("src.agent.nodes.explore_closure", lambda *args, **kwargs: [option])
    llm = ScriptedLLM(options=OptionsExplanation(why_it_failed="A hybrid closes at 16.6 kg.",
                                                 trade_offs="It carries 3.46 kg of fuel."))
    state = run_agent(_ctx(tmp_path, db, llm), requirements=REQ_FAILED)
    assert "explanation of the options mentions" not in Path(state["final_report_path"]).read_text()


def test_an_options_llm_outage_costs_only_the_explanation(tmp_path: Path, db: Path) -> None:
    llm = ScriptedLLM(fail_on=(OptionsExplanation,))
    state = run_agent(_ctx(tmp_path, db, llm), requirements=REQ_FAILED)
    assert state["explore_options"] and state["options_explanation"] is None and state["final_report_path"]


def test_numbers_the_options_explanation_invents_are_flagged(tmp_path: Path, db: Path) -> None:
    llm = ScriptedLLM(options=OptionsExplanation(why_it_failed="Needs 987.6 kg of battery.", trade_offs="None."))
    state = run_agent(_ctx(tmp_path, db, llm), requirements=REQ_FAILED)
    assert "987.6" in Path(state["final_report_path"]).read_text().split("## Warnings")[1]


def test_stray_numbers_ignores_known_and_trivial_figures() -> None:
    assert report.stray_numbers("13.43 kg, 42.7 kg and 2 more", '{"m": 13.43}') == ["42.7"]


def test_cli_prints_options_after_a_failed_run(tmp_path: Path, db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text(REQ_FAILED.model_dump_json())
    assert main([f"--spec={spec}", f"--outputs={tmp_path / 'o'}", f"--db={db}", "--no-aero", "--quiet"]) == EXIT_NOT_CONVERGED
    assert "Options that close the design" in capsys.readouterr().out


# ------------------------------------------------------------------ 3D viewer ---
VIEWER_MODEL: dict[str, Any] = {"frame": "x_aft_y_right_z_up", "cg_m": [0.5, 0, 0], "mass_kg": 10.0,
                                "parts": [{"name": "fuselage", "group": "component", "mass_kg": 1.0,
                                           "positions_b64": "", "indices_b64": ""}]}


def _cad_with_model(model_path: Optional[str]) -> CADArtifacts:
    return CADArtifacts(oml_vsp3_path="a.vsp3", oml_step_path="b.step", assembly_step_path="c.step", x_cg_m=0.3,
                        static_margin=0.11, collision_count=0, as_built_mass_kg=10.0, ixx_kgm2=1.0, iyy_kgm2=1.0,
                        izz_kgm2=1.0, ixz_kgm2=0.0, viewer_model_path=model_path)


def test_compile_report_builds_the_viewer_from_the_exported_model(tmp_path: Path, db: Path) -> None:
    ctx = _ctx(tmp_path, db, ScriptedLLM())
    state = run_agent(ctx, requirements=G1_ELECTRIC_FW)
    out = Path(state["out_dir"])
    assert not (out / "viewer.html").exists()  # the aero/CAD-free run exported no mesh
    (out / "viewer_model.json").write_text(json.dumps(VIEWER_MODEL))
    update = nodes.compile_report(
        {**state, "cad_artifacts": _cad_with_model(str(out / "viewer_model.json"))}, ctx)
    assert (out / "viewer.html").is_file() and update["viewer_error"] is None


@pytest.mark.parametrize("cad", [None, _cad_with_model(None)], ids=["no_cad", "no_mesh_path"])
def test_a_stale_viewer_model_never_becomes_the_viewer(tmp_path: Path, db: Path, cad: Optional[CADArtifacts]) -> None:
    ctx = _ctx(tmp_path, db, ScriptedLLM())
    state = run_agent(ctx, requirements=G1_ELECTRIC_FW)
    out = Path(state["out_dir"])
    (out / "viewer.html").unlink(missing_ok=True)
    (out / "viewer_model.json").write_text(json.dumps(VIEWER_MODEL))  # left behind by an earlier iteration
    update = nodes.compile_report({**state, "cad_artifacts": cad}, ctx)
    assert not (out / "viewer.html").exists()
    assert update["viewer_error"] is None
    assert not any("3D viewer not built" in w for w in nodes.run_warnings({**state, **update}))


def test_a_viewer_failure_is_a_warning_not_a_failed_run(tmp_path: Path, db: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(run_dir: Path) -> Path:
        raise ViewerError("bad model")

    monkeypatch.setattr("src.viewer.build.write_viewer", broken)
    ctx = _ctx(tmp_path, db, ScriptedLLM())
    state = run_agent(ctx, requirements=G1_ELECTRIC_FW)
    update = nodes.compile_report({**state, "cad_artifacts": _cad_with_model("viewer_model.json")}, ctx)
    assert Path(update["final_report_path"]).is_file()
    assert update["viewer_error"] == "3D viewer not built: bad model"
    assert "bad model" in Path(update["final_report_path"]).read_text()


def test_the_report_lists_the_viewer_file_only_when_it_was_built(tmp_path: Path) -> None:
    cad = CADArtifacts(oml_vsp3_path="a.vsp3", oml_step_path="b.step", assembly_step_path="c.step", x_cg_m=0.3,
                       static_margin=0.11, collision_count=0, as_built_mass_kg=10.0, ixx_kgm2=1.0, iyy_kgm2=1.0,
                       izz_kgm2=1.0, ixz_kgm2=0.0, viewer_model_path="viewer_model.json")
    inputs = report.ReportInputs(run_id="r", out_dir=tmp_path, requirements=G1_ELECTRIC_FW, sizing=None,
                                 bom=None, aero=None, cad=cad, fcs=None, converged=True, iterations=0)
    assert "viewer.html" not in "\n".join(report._sections(inputs))  # mesh exported, but the build failed
    no_mesh = report.ReportInputs(run_id="r", out_dir=tmp_path, requirements=G1_ELECTRIC_FW, sizing=None, bom=None,
                                  aero=None, cad=_cad_with_model(None), fcs=None, converged=True, iterations=0)
    assert "viewer.html" not in "\n".join(report._sections(no_mesh))  # no mesh exported: nothing to list
    (tmp_path / "viewer.html").write_text("<html></html>")
    assert "`viewer.html` (3D viewer)" in "\n".join(report._sections(inputs))


# ---------------------------------------------------------- avionics parsing ---
def test_parser_defaults_to_cube_with_jetson(db: Path) -> None:
    req = to_requirements(SRS_DRAFT, db).requirements
    assert req is not None and req.flight_controller == "cube_orange_plus" and req.companion_computer is True


def test_parser_reads_ark(db: Path) -> None:
    req = to_requirements(SRS_DRAFT.model_copy(update={"flight_controller": "ark_jetson_pab"}), db).requirements
    assert req is not None and req.flight_controller == "ark_jetson_pab" and req.jetson_fitted


def test_parser_respects_explicit_no_companion(db: Path) -> None:
    req = to_requirements(SRS_DRAFT.model_copy(update={"companion_computer": False}), db).requirements
    assert req is not None and req.companion_computer is False and not req.jetson_fitted


# ----------------------------------------------------------- payload model parsing ---
def test_named_model_without_mass_uses_catalogue(db: Path) -> None:
    outcome = to_requirements(SRS_DRAFT.model_copy(update={"payload_model": "zt30", "payload_mass": None,
                                                           "payload_power": None}), db)
    req = outcome.requirements
    assert outcome.clarification is None and req is not None
    assert (req.payload_model, req.payload_mass_kg, req.payload_power_w) == ("SIYI ZT30", 0.854, 9.0)
    assert not req.payload_edited


def test_named_model_with_stated_mass_is_edited(db: Path) -> None:
    req = to_requirements(SRS_DRAFT.model_copy(update={"payload_model": "SIYI ZT30", "payload_mass": Mass(value=1.2, unit="kg"),
                                                       "payload_power": None}), db).requirements
    assert req is not None and (req.payload_mass_kg, req.payload_power_w, req.payload_edited) == (1.2, 9.0, True)


def test_unknown_model_note_and_mass_required(db: Path) -> None:
    unknown = SRS_DRAFT.model_copy(update={"payload_model": "Foo 9000"})
    assert "payload mass" in to_requirements(unknown.model_copy(update={"payload_mass": None}), db).clarification
    outcome = to_requirements(unknown, db)
    assert outcome.requirements is not None and outcome.requirements.payload_model == "custom"
    assert "Payload 'Foo 9000' is not in the catalogue; treated as a custom payload." in outcome.assumptions


def test_custom_named_payload_has_no_catalogue_note(db: Path) -> None:
    outcome = to_requirements(SRS_DRAFT.model_copy(update={"payload_model": "Custom payload"}), db)
    assert outcome.requirements is not None and outcome.requirements.payload_model == "custom"
    assert not any("not in the catalogue" in note for note in outcome.assumptions)


def test_datasheet_note_names_only_the_unstated_values(db: Path) -> None:
    named = SRS_DRAFT.model_copy(update={"payload_model": "SIYI ZT30", "payload_power": None})
    power_only = to_requirements(named.model_copy(update={"payload_mass": Mass(value=1.2, unit="kg")}), db)
    assert "Payload SIYI ZT30: unstated power taken from its datasheet (9 W)." in power_only.assumptions
    both = to_requirements(named.model_copy(update={"payload_mass": None}), db)
    assert "Payload SIYI ZT30: unstated mass and power taken from its datasheet (0.854 kg, 9 W)." in both.assumptions


def test_conflicting_sensor_type_is_noted(db: Path) -> None:
    zr10 = to_requirements(SRS_DRAFT.model_copy(update={"payload_model": "SIYI ZR10"}), db)  # SRS says eo_ir
    assert "Payload sensor 'eo_ir' ignored: SIYI ZR10 is an EO payload." in zr10.assumptions
    zt30 = to_requirements(SRS_DRAFT.model_copy(update={"payload_model": "SIYI ZT30"}), db)
    assert not any("ignored" in note for note in zt30.assumptions)


def test_no_model_stays_custom(db: Path) -> None:
    req = to_requirements(SRS_DRAFT, db).requirements
    assert req is not None and req.payload_model == "custom"


def test_parser_prompt_lists_gimbals() -> None:
    from src.agent.prompts import PARSER_SYSTEM
    from src.schemas.payload_catalog import PAYLOADS

    assert all(p.name in PARSER_SYSTEM for p in PAYLOADS if p.vendor is not None)


# ------------------------------------------------------------ fabrication ---
CUSTOM_FABRICATION = FabricationSettings(bed_width_m=0.5, bed_height_m=0.3)


def test_fabrication_settings_are_written_to_the_run_folder(tmp_path: Path, db: Path) -> None:
    state = run_agent(_ctx(tmp_path, db, None), requirements=G1_ELECTRIC_FW, fabrication=CUSTOM_FABRICATION)
    written = json.loads((Path(state["out_dir"]) / "fabrication.json").read_text())
    assert FabricationSettings.model_validate(written) == CUSTOM_FABRICATION
    assert state["fabrication"] == CUSTOM_FABRICATION


def test_a_run_without_settings_uses_the_default(tmp_path: Path, db: Path) -> None:
    state = run_agent(_ctx(tmp_path, db, None), requirements=G1_ELECTRIC_FW)
    assert state["fabrication"] == DEFAULT_FABRICATION
    written = json.loads((Path(state["out_dir"]) / "fabrication.json").read_text())
    assert FabricationSettings.model_validate(written) == DEFAULT_FABRICATION


def test_the_cad_stage_receives_the_run_settings(tmp_path: Path, db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = run_agent(_ctx(tmp_path, db, None), requirements=G1_ELECTRIC_FW, fabrication=CUSTOM_FABRICATION)
    seen: list[FabricationSettings] = []

    def fake(sizing: Any, bom: Any, req: Any, out_dir: Path, progress: Any,
             settings: FabricationSettings = DEFAULT_FABRICATION) -> Any:
        seen.append(settings)
        raise pipeline.PipelineError("stop here")

    monkeypatch.setattr("src.agent.nodes.pipeline.run_aero_cad", fake)
    monkeypatch.setattr("src.agent.nodes.pipeline.DESIGN_ERRORS", (pipeline.PipelineError,))
    nodes.build_aero_cad(state, replace(_ctx(tmp_path, db, None), with_aero_cad=True))
    assert seen == [CUSTOM_FABRICATION]


def _kit_cad(warning: Optional[str]) -> CADArtifacts:
    return _cad_with_model(None).model_copy(update={"kit_warning": warning})


def test_a_kit_warning_is_a_design_warning_exactly_once(tmp_path: Path, db: Path) -> None:
    state = run_agent(_ctx(tmp_path, db, None), requirements=G1_ELECTRIC_FW)
    with_kit_warning = {**state, "cad_artifacts": _kit_cad("no kit: shell model used")}
    hits = [w for w in nodes.run_warnings(with_kit_warning) if "no kit: shell model used" in w]
    assert len(hits) == 1
    assert not any("kit" in w.lower() for w in nodes.run_warnings({**state, "cad_artifacts": _kit_cad(None)}))


def test_run_aero_cad_hands_the_settings_and_design_day_to_build_aero_cad(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Any] = []
    monkeypatch.setitem(__import__("sys").modules, "src.tools.aero_cad.build", SimpleNamespace(
        build_aero_cad=lambda *args: seen.append(args[-2:]) or ("aero", "cad")))
    req = SimpleNamespace(service_ceiling_m=1.0, isa_temperature_offset_c=7.0)
    out = pipeline.run_aero_cad("sizing", "bom", req, Path("."), lambda _: None, CUSTOM_FABRICATION)
    assert out == ("aero", "cad") and seen == [(CUSTOM_FABRICATION, 7.0)]


def test_cli_fabrication_file_reaches_the_run(tmp_path: Path, db: Path) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text(G1_ELECTRIC_FW.model_dump_json())
    fab = tmp_path / "fab.json"
    fab.write_text(CUSTOM_FABRICATION.model_dump_json())
    args = [f"--spec={spec}", f"--outputs={tmp_path / 'o'}", f"--db={db}", "--no-aero", "--quiet"]
    assert main([*args, f"--fabrication={fab}"]) in (EXIT_CONVERGED, EXIT_NOT_CONVERGED)
    (written,) = (tmp_path / "o").glob("*/fabrication.json")
    assert FabricationSettings.model_validate_json(written.read_text()) == CUSTOM_FABRICATION


@pytest.mark.parametrize("content", [None, "not json", '{"bed_width_m": -1}', '{"bogus": 1}'],
                         ids=["missing", "not_json", "out_of_bounds", "unknown_field"])
def test_cli_rejects_a_bad_fabrication_file(tmp_path: Path, db: Path, content: Optional[str],
                                            capsys: pytest.CaptureFixture[str]) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text(G1_ELECTRIC_FW.model_dump_json())
    fab = tmp_path / "fab.json"
    if content is not None:
        fab.write_text(content)
    code = main([f"--spec={spec}", f"--outputs={tmp_path / 'o'}", f"--db={db}", "--no-aero", "--quiet",
                 f"--fabrication={fab}"])
    assert code == EXIT_ERROR and "fabrication file" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


def test_cli_rejects_a_non_utf8_fabrication_file_naming_the_path(tmp_path: Path, db: Path,
                                                                 capsys: pytest.CaptureFixture[str]) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text(G1_ELECTRIC_FW.model_dump_json())
    fab = tmp_path / "fab.json"
    fab.write_bytes(b'{"bed_width_m": \xff\xfe}')
    code = main([f"--spec={spec}", f"--outputs={tmp_path / 'o'}", f"--db={db}", "--no-aero", "--quiet",
                 f"--fabrication={fab}"])
    err = capsys.readouterr().err
    assert code == EXIT_ERROR and "fabrication file" in err and str(fab) in err
    assert not (tmp_path / "o").exists()


def test_run_design_writes_and_threads_the_settings(tmp_path: Path, db: Path) -> None:
    run = pipeline.run_design(G1_ELECTRIC_FW, tmp_path / "o", db, with_aero_cad=False, with_sitl=False,
                              settings=CUSTOM_FABRICATION)
    written = (run.out_dir / "fabrication.json").read_text()
    assert FabricationSettings.model_validate_json(written) == CUSTOM_FABRICATION
