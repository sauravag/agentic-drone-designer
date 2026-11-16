"""Design chat (spec 7): the check-and-answer loop tests what-ifs with the solver before it answers.

The LLM writes flat `ProposedChange`s; `to_change_set` gives the `ChangeSet` the app validates, evaluates and
returns, so tests compare `reply.proposal` with `to_change_set(<helper>)`."""

import json
from pathlib import Path

import pytest

from src.agent.agent_schemas import DesignerTurn, FieldChange, ProposedChange
from src.agent.design_chat import (HISTORY_BUDGET, MAX_ROUNDS, MAX_WHAT_IFS, ROUND_LIMIT_NOTE, ChatContext,
                                   ChatMessage, answer, to_change_set)
from src.agent.llm import LLMError
from src.schemas.assumptions import DEFAULT_ASSUMPTIONS
from src.schemas.changes import LABEL_MAX_LEN
from src.ui import agent_service
from tests.agent_fakes import ScriptedLLM
from tests.reference_designs import REQ_FAILED


@pytest.fixture(scope="module")
def db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from src.tools.database.seed_data import seed_database

    path = tmp_path_factory.mktemp("db") / "catalog.db"
    seed_database(path)
    return path


def _ctx(db: Path, allow_knobs: bool = False) -> ChatContext:
    return ChatContext(requirements=REQ_FAILED, assumptions=DEFAULT_ASSUMPTIONS, failed_gates=(), options=(),
                       allow_knobs=allow_knobs, db_path=db)


def _change(label: str, **values: object) -> ProposedChange:
    return ProposedChange(label=label, changes=[FieldChange(field=k, value=str(v)) for k, v in values.items()])


def _endurance(h: float) -> ProposedChange:
    return _change("Shorter endurance", endurance_hours=h)


def _c_rate(c: float) -> ProposedChange:
    return _change("Higher C-rate", liion_max_c_rate=c)


def _knob(**kw: float) -> ProposedChange:
    return _change("knob", **kw)


def _final(message: str, proposal: ProposedChange | None = None) -> DesignerTurn:
    return DesignerTurn(message=message, what_ifs=[], proposal=proposal, done=True)


def _checking(*what_ifs: ProposedChange) -> DesignerTurn:
    return DesignerTurn(message="checking", what_ifs=list(what_ifs), proposal=None, done=False)


def test_the_agent_tests_a_what_if_before_answering(db: Path) -> None:
    turns = [DesignerTurn(message="checking", what_ifs=[_endurance(4.0)], proposal=None, done=False),
             DesignerTurn(message="4 h closes", what_ifs=[], proposal=_endurance(4.0), done=True)]
    llm = ScriptedLLM(designer_turns=turns)
    reply = answer(llm, _ctx(db), (), "Can we keep electric with less endurance?")
    assert reply.rounds == 2 and reply.proposal == to_change_set(_endurance(4.0))
    assert len(reply.what_ifs) == 1 and reply.what_ifs[0].result is not None
    assert "what_if_results" in llm.calls[-1][1]  # the second round saw the solver's numbers


def test_the_second_round_briefing_carries_the_solver_result(db: Path) -> None:
    llm = ScriptedLLM(designer_turns=[_checking(_endurance(4.0)), _final("done")])
    reply = answer(llm, _ctx(db), (), "less endurance?")
    first, second = (json.loads(prompt) for _, prompt in llm.calls)
    assert first["what_if_results"] == [] and first["designer_message"] == "less endurance?"
    assert first["final_round"] is False
    (seen,) = second["what_if_results"]
    assert seen["result"] == reply.what_ifs[0].result.as_facts()  # type: ignore[union-attr]
    assert seen["change"]["requirements"] == {"endurance_hours": 4.0}
    assert reply.role == "agent" and reply.text == "done"


def test_round_limit_and_invalid_changes(db: Path) -> None:
    loop = [DesignerTurn(message="more", what_ifs=[_c_rate(99)], proposal=None, done=False)] * 5
    llm = ScriptedLLM(designer_turns=loop)
    reply = answer(llm, _ctx(db), (), "try 99C")
    assert reply.rounds == MAX_ROUNDS and reply.what_ifs[0].error
    assert reply.what_ifs[0].result is None and "Max discharge (Li-ion)" in reply.what_ifs[0].error
    assert reply.text == f"more\n\n{ROUND_LIMIT_NOTE}"
    assert [json.loads(p)["final_round"] for _, p in llm.calls] == [False] * (MAX_ROUNDS - 1) + [True]


def test_knob_changes_are_refused_outside_airframe_mode(db: Path) -> None:
    turn = DesignerTurn(message="x", what_ifs=[_knob(aspect_ratio=14)], proposal=None, done=True)
    reply = answer(ScriptedLLM(designer_turns=[turn]), _ctx(db, allow_knobs=False), (), "raise AR")
    assert "knob" in (reply.what_ifs[0].error or "")


def test_knob_changes_are_checked_in_airframe_mode_and_never_clamped(db: Path) -> None:
    turn = DesignerTurn(message="x", what_ifs=[_knob(aspect_ratio=14), _knob(aspect_ratio=40)], proposal=None,
                        done=True)
    ok, too_high = answer(ScriptedLLM(designer_turns=[turn]), _ctx(db, allow_knobs=True), (), "AR?").what_ifs
    assert ok.error is None and ok.result is not None
    assert too_high.result is None and "outside the allowed range" in (too_high.error or "")


def test_text_values_are_coerced_to_the_field_types() -> None:
    change = to_change_set(_change("mix", runway_available="true", group_constraint=2,
                                   propulsion_type="series_hybrid", endurance_hours="4.5"))
    assert change.requirements.runway_available is True and change.requirements.group_constraint == 2
    assert change.requirements.propulsion_type == "series_hybrid" and change.requirements.endurance_hours == 4.5


@pytest.mark.parametrize(("proposed", "match"), [
    (_change("x", endurance_hours="four"), "Endurance (h): Input should be a valid number"),
    (_change("x", runway_available="maybe"), "Runway available"),
    (_change("x", group_constraint=4), "Group limit"),
    (ProposedChange(label="x", changes=[FieldChange(field="endurance_hours", value="4"),
                                        FieldChange(field="endurance_hours", value="5")]), "only once"),
])
def test_malformed_values_become_concise_what_if_errors(db: Path, proposed: ProposedChange, match: str) -> None:
    turn = DesignerTurn(message="x", what_ifs=[proposed], proposal=None, done=True)
    (result,) = answer(ScriptedLLM(designer_turns=[turn]), _ctx(db), (), "q").what_ifs
    assert result.result is None and match in (result.error or "")
    assert "pydantic" not in (result.error or "") and "\n" not in (result.error or "")


def test_labels_are_trimmed_in_the_app_not_rejected() -> None:
    assert len(to_change_set(_change("L" * 500, endurance_hours=4)).label) == LABEL_MAX_LEN
    assert to_change_set(_change("   ", endurance_hours=4)).label


def test_numbers_the_solver_never_produced_are_flagged(db: Path) -> None:
    turn = DesignerTurn(message="It will weigh 12.34 kg", what_ifs=[], proposal=None, done=True)
    assert "12.34" in answer(ScriptedLLM(designer_turns=[turn]), _ctx(db), (), "weight?").stray_numbers


def test_numbers_from_the_briefing_and_results_are_not_flagged(db: Path) -> None:
    first = answer(ScriptedLLM(designer_turns=[_checking(_endurance(4.0)), _final("x")]), _ctx(db), (), "q")
    evaluation = first.what_ifs[0].result
    assert evaluation is not None
    text = f"Endurance 6 h and radius 100 km; at 4 h the MTOW is {evaluation.mtow_kg} kg."
    llm = ScriptedLLM(designer_turns=[_checking(_endurance(4.0)), _final(text)])
    assert answer(llm, _ctx(db), (), "q").stray_numbers == ()


def test_an_invented_number_stays_flagged_when_repeated_in_a_later_turn(db: Path) -> None:
    first = answer(ScriptedLLM(designer_turns=[_final("It will weigh 12.34 kg")]), _ctx(db), (), "weight?")
    history = (ChatMessage(role="designer", text="weight?"), first)
    second = answer(ScriptedLLM(designer_turns=[_final("As I said, 12.34 kg")]), _ctx(db), history, "sure?")
    assert "12.34" in second.stray_numbers


def test_a_number_the_designer_typed_is_not_a_fact(db: Path) -> None:
    reply = answer(ScriptedLLM(designer_turns=[_final("At 99 kg it would close")]), _ctx(db), (), "at 99 kg?")
    assert "99" in reply.stray_numbers


def test_earlier_what_if_results_stay_citable(db: Path) -> None:
    first = answer(ScriptedLLM(designer_turns=[_checking(_endurance(4.0)), _final("x")]), _ctx(db), (), "q")
    mtow = first.what_ifs[0].result.mtow_kg  # type: ignore[union-attr]
    reply = answer(ScriptedLLM(designer_turns=[_final(f"Earlier: {mtow} kg")]), _ctx(db), (first,), "again?")
    assert reply.stray_numbers == ()


def test_change_labels_are_audited_too(db: Path) -> None:
    turn = DesignerTurn(message="ok", what_ifs=[_change("MTOW 77.7 kg variant", endurance_hours=4)],
                        proposal=_change("Saves 88.8 kg", endurance_hours=4), done=True)
    strays = answer(ScriptedLLM(designer_turns=[turn]), _ctx(db), (), "q").stray_numbers
    assert {"77.7", "88.8"} <= set(strays)


def test_an_invalid_proposal_is_dropped_and_its_error_shown(db: Path) -> None:
    reply = answer(ScriptedLLM(designer_turns=[_final("accept this", _c_rate(99))]), _ctx(db), (), "q")
    assert reply.proposal is None
    assert reply.text.startswith("accept this") and "Max discharge (Li-ion)" in reply.text


def test_a_knob_proposal_is_dropped_outside_airframe_mode(db: Path) -> None:
    reply = answer(ScriptedLLM(designer_turns=[_final("raise AR", _knob(aspect_ratio=14))]), _ctx(db), (), "q")
    assert reply.proposal is None and "knob" in reply.text


def test_what_ifs_beyond_the_limit_are_not_evaluated(db: Path) -> None:
    many = [_endurance(5.0 - 0.25 * i) for i in range(MAX_WHAT_IFS + 2)]
    turn = DesignerTurn(message="m", what_ifs=many, proposal=None, done=True)
    reply = answer(ScriptedLLM(designer_turns=[turn]), _ctx(db), (), "q")
    assert len(reply.what_ifs) == MAX_WHAT_IFS and f"first {MAX_WHAT_IFS}" in reply.text


def test_the_briefing_trims_history_and_lists_only_allowed_fields(db: Path) -> None:
    history = tuple(ChatMessage(role="designer" if i % 2 else "agent", text=f"msg {i}") for i in range(20))
    llm = ScriptedLLM(designer_turns=[_final("ok")])
    answer(llm, _ctx(db), history, "q")
    briefing = json.loads(llm.calls[0][1])
    assert [m["text"] for m in briefing["history"]] == [f"msg {i}" for i in range(20 - HISTORY_BUDGET, 20)]
    assert set(briefing["allowed_fields"]) == {"requirements", "assumptions"}
    assert briefing["allowed_fields"]["assumptions"]["liion_max_c_rate"] == {"minimum": 1.0, "maximum": 30.0}
    assert briefing["allowed_fields"]["requirements"]["endurance_hours"] == {"exclusiveMinimum": 0.1, "maximum": 48.0}
    llm_knobs = ScriptedLLM(designer_turns=[_final("ok")])
    answer(llm_knobs, _ctx(db, allow_knobs=True), (), "q")
    knobs = json.loads(llm_knobs.calls[0][1])["allowed_fields"]["knobs"]
    assert knobs["aspect_ratio"] == {"minimum": 4.0, "maximum": 25.0} and "reserve_fraction" not in knobs


def test_llm_errors_propagate(db: Path) -> None:
    with pytest.raises(LLMError):
        answer(ScriptedLLM(fail_on=(DesignerTurn,)), _ctx(db), (), "q")


# ------------------------------------------------------------- service ---
def _view(**kw: object) -> agent_service.AgentRunView:
    gates = ({"name": "mass", "passed": False, "detail": "too heavy"}, {"name": "cg", "passed": True, "detail": ""})
    log = ({"iteration": 0, "failed": ["mass"], "changes": ["aspect_ratio 12 -> 13 (rule: L/D)"], "summary": ""},)
    return agent_service.AgentRunView(status="not_converged", requirements=REQ_FAILED, gates=gates,
                                      critic_log=log, **kw)


def test_the_service_builds_the_context_from_the_view(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    llm = ScriptedLLM(designer_turns=[_final("ok"), _final("ok")])
    monkeypatch.setattr(agent_service, "make_llm", lambda: llm)
    assert agent_service.chat(_view(explorer_mode="closure"), (), "q", db).text == "ok"
    briefing = json.loads(llm.calls[0][1])
    assert briefing["failed_gates"] == ["mass: too heavy"] and "knobs" not in briefing["allowed_fields"]
    assert briefing["critic_summary"] == ["iteration 0: failed mass; changes aspect_ratio 12 -> 13 (rule: L/D)"]
    agent_service.chat(_view(explorer_mode="airframe"), (), "q", db)
    assert "knobs" in json.loads(llm.calls[1][1])["allowed_fields"]


def test_the_service_needs_an_llm_and_requirements(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="LLM"):
        agent_service.chat(_view(), (), "q", db)
    monkeypatch.setattr(agent_service, "make_llm", lambda: ScriptedLLM(designer_turns=[_final("ok")]))
    with pytest.raises(ValueError, match="requirements"):
        agent_service.chat(agent_service.AgentRunView(status="error"), (), "q", db)
