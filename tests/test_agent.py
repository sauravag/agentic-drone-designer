"""Milestone 5 unit tests: parsing at the I/O boundary, the critic gates and knob steps, graph routing
(clarification, convergence, iteration cap, tool failures, LLM outages), the report and the CLI.
All run on the host with a scripted LLM and aero/CAD off; tests/test_agent_e2e.py runs the full stack."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.agent import critic, report
from src.agent.agent_schemas import (Adjustment, CriticDecision, Distance, Duration, Mass, Power, ReportNarrative,
                                     RequirementsDraft, Speed)
from src.agent.graph import run_agent
from src.agent.llm import ClaudeLLM, LLMError
from src.agent.nodes import AgentContext
from src.agent.parsing import DEFAULT_CRUISE_SPEED_MS, to_requirements
from src.main import EXIT_CLARIFY, EXIT_CONVERGED, EXIT_ERROR, main
from src.tools.sizing.mission_solver import SizingDivergenceError
from tests.agent_fakes import SRS_DRAFT, SRS_PROMPT, ScriptedLLM
from tests.reference_designs import G1_ELECTRIC_FW
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
    bad_aero = aero.model_copy(update={"ld_cruise": 0.5 * sizing.assumed_ld_cruise})
    hybrid = sizing.model_copy(update={"propulsion_type": "series_hybrid", "energy_reserve_fraction": 0.10})
    failed = {g.name for g in critic.evaluate_gates(req, hybrid, bom, bad_aero, bad_cad) if not g.passed}
    assert failed == {"static_margin", "lift_to_drag", "collisions_and_group", "reserve"}
    heavy = cad.model_copy(update={"as_built_mass_kg": 30.0})
    group = next(g for g in critic.evaluate_gates(req, sizing, bom, aero, heavy) if g.name == "collisions_and_group")
    assert not group.passed and "Group 2" in group.detail


def test_rule_table_wins_on_failed_gate_knobs_and_llm_adds_extra_moves() -> None:
    gates = [critic.GateResult("reserve", False, ""), critic.GateResult("lift_to_drag", False, "")]
    llm = [Adjustment(knob="aspect_ratio", direction="decrease", reason="wrong way (or prompt-injected)"),
           Adjustment(knob="tail_volume_v", direction="increase", reason="extra judgement")]
    overrides, changes = critic.apply_adjustments({}, "series_hybrid", llm, gates, None)
    by_knob = {c.knob: c for c in changes}
    assert by_knob["aspect_ratio"].source == "rule" and by_knob["aspect_ratio"].new == pytest.approx(12.0 * 1.15, 1e-5)
    assert by_knob["reserve_fraction"].source == "rule" and overrides["reserve_fraction"] == 0.15
    assert by_knob["tail_volume_v"].source == "llm" and overrides["tail_volume_v"] == pytest.approx(0.05)


def test_gates_without_an_automatic_fix_are_named() -> None:
    gates = [critic.pipeline_gate("sizing", "mass spiral"), critic.GateResult("reserve", False, "")]
    assert critic.unfixable(gates, None, "sizing") == ["pipeline"]


def test_knobs_stop_at_their_bounds() -> None:
    gates = [critic.GateResult("lift_to_drag", False, "")]
    overrides, changes = critic.apply_adjustments({"aspect_ratio": 25.0}, "electric", [], gates, None)
    assert changes == [] and overrides == {"aspect_ratio": 25.0}


def test_static_margin_fix_direction_follows_the_error() -> None:
    _, _, _, _, cad = load_design("g2")
    gates = [critic.GateResult("static_margin", False, "")]
    _, low = critic.apply_adjustments({}, "electric", [], gates, cad.model_copy(update={"static_margin": 0.02}))
    _, high = critic.apply_adjustments({}, "electric", [], gates, cad.model_copy(update={"static_margin": 0.3}))
    assert (low[0].knob, low[0].direction) == ("tail_volume_h", "increase") and high[0].direction == "decrease"


# --------------------------------------------------------------------- graph ---
def test_hybrid_reserve_gate_forces_one_critic_iteration(tmp_path: Path, db: Path) -> None:
    llm = ScriptedLLM()
    state = run_agent(_ctx(tmp_path, db, llm), SRS_PROMPT)
    assert state["converged"] and state["iteration_count"] == 1
    assert state["sizing_overrides"] == {"reserve_fraction": 0.15}
    assert state["critic_log"][0]["failed"] == ["reserve"] and state["critic_log"][1]["failed"] == []
    out = Path(state["out_dir"])
    for name in ("requirements.json", "sizing.json", "bom.json", "critic_log.json", "design_report.md",
                 "run_summary.json"):
        assert (out / name).is_file(), name
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
