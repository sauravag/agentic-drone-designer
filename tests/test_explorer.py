from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agent.critic import RESERVE_MIN
from src.agent.evaluator import evaluate
from src.agent.explorer import OUT_OF_RANGE_REASON, explore_closure, explore_improvement, explorer_mode
from src.agent.graph import initial_state
from src.schemas.assumptions import DEFAULT_ASSUMPTIONS
from src.schemas.changes import apply_changes
from src.tools.sizing.mission_solver import size_aircraft
from tests.reference_designs import G1_ELECTRIC_FW, G3_HYBRID_VTOL, REQ_FAILED


@pytest.fixture(scope="module")
def db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from src.tools.database.seed_data import seed_database

    path = tmp_path_factory.mktemp("db") / "catalog.db"
    seed_database(path)
    return path


@pytest.fixture(scope="module")
def closure_options(db: Path) -> list:
    return explore_closure(REQ_FAILED, DEFAULT_ASSUMPTIONS, db)


def test_closure_options_for_the_failed_run(db: Path, closure_options: list) -> None:
    options = closure_options
    by_lever = {o.lever: o for o in options}
    assert by_lever["propulsion"].closes
    # Measured (brief expected endurance to close): the 2 x 100 km transit at 30 m/s alone spirals the electric
    # quadplane, so even the 0.5 h endurance bound does not close; the option says so instead.
    endurance = by_lever["endurance"]
    assert not endurance.closes and OUT_OF_RANGE_REASON in endurance.result.reason
    assert endurance.change.requirements.endurance_hours == 0.5
    battery = by_lever["liion_specific_energy"]
    assert (not battery.closes) or battery.change.assumptions.liion_specific_energy_wh_kg > 350  # top of range only
    closing = [o for o in options if o.closes]
    assert closing == sorted(closing, key=lambda o: o.relative_change)
    for o in closing:  # every option re-validates and re-evaluates as closing
        req, a, knobs = apply_changes(REQ_FAILED, DEFAULT_ASSUMPTIONS, o.change)
        assert evaluate(req, a, knobs, db).closes


def test_closing_options_come_first_and_runway_lever_sets_both_fields(closure_options: list) -> None:
    flags = [o.closes for o in closure_options]
    assert flags == sorted(flags, reverse=True)
    runway = {o.lever: o for o in closure_options}["runway"].change.requirements
    assert runway.runway_available is True and runway.airframe_topology == "fixed_wing"


def test_hybrid_option_carries_the_reserve_it_was_checked_with(closure_options: list) -> None:  # spec 6.2
    propulsion = {o.lever: o for o in closure_options}["propulsion"]
    assert propulsion.change.requirements.propulsion_type == "series_hybrid"
    assert propulsion.change.assumptions.fuel_reserve_fraction == RESERVE_MIN
    assert propulsion.change.label.startswith("Propulsion")


def test_options_serialise_to_json(closure_options: list) -> None:
    payload = closure_options[0].to_json()
    assert {"lever", "change", "result", "relative_change", "closes", "mtow_saved_kg", "group_drop"} <= payload.keys()
    assert payload["result"]["closes"] == closure_options[0].closes


def test_time_budget_marks_unevaluated_levers(db: Path) -> None:
    options = explore_closure(REQ_FAILED, DEFAULT_ASSUMPTIONS, db, time_budget_s=0.0)
    assert options and all(not o.closes and "time budget" in o.result.reason for o in options)


def test_improvement_mode_reports_savings(db: Path) -> None:
    baseline = size_aircraft(G1_ELECTRIC_FW)
    options = explore_improvement(G1_ELECTRIC_FW, DEFAULT_ASSUMPTIONS, baseline, db)
    assert any(o.mtow_saved_kg and o.mtow_saved_kg > 0 for o in options)
    by_lever = {o.lever: o for o in options}
    assert by_lever["endurance"].change.requirements.endurance_hours == G1_ELECTRIC_FW.endurance_hours - 1.0


@pytest.mark.parametrize(("update", "mode"), [
    ({"converged": True}, "improvement"),
    ({"stage_error": {"stage": "sizing", "message": "spiral"}}, "closure"),
    ({"stage_error": {"stage": "cots", "message": "no motor"}}, "closure"),
    ({"gate_results": [{"name": "static_margin", "passed": False, "detail": ""}]}, "airframe"),
    ({"gate_results": [{"name": "lift_to_drag", "passed": False, "detail": ""}]}, "airframe"),
    ({"gate_results": [{"name": "reserve", "passed": False, "detail": ""}]}, "closure"),
    ({"gate_results": [{"name": "collisions_and_group", "passed": False, "detail": ""}]}, "closure"),
    ({"gate_results": [{"name": "collisions_and_group", "passed": False, "detail": ""}],
      "cad_artifacts": SimpleNamespace(collision_count=2)}, "airframe"),
    ({"stage_error": {"stage": "aero_cad", "message": "pack"}}, "airframe"),
    ({"clarification_needed": "which payload?"}, "none"),
    ({"fatal_error": "x"}, "none")])
def test_explorer_mode(update: dict, mode: str) -> None:
    assert explorer_mode(initial_state("", G1_ELECTRIC_FW) | update) == mode


def test_endurance_bisects_to_the_closing_value_nearest_the_request(db: Path) -> None:
    req = REQ_FAILED.model_copy(update={"combat_radius_km": 20.0})  # short transit: endurance can close it
    endurance = {o.lever: o for o in explore_closure(req, DEFAULT_ASSUMPTIONS, db)}["endurance"]
    value = endurance.change.requirements.endurance_hours
    assert endurance.closes and 0.5 <= value < req.endurance_hours
    assert endurance.relative_change == pytest.approx((req.endurance_hours - value) / req.endurance_hours)
    one_step_more = req.model_copy(update={"endurance_hours": round(value + 0.1, 6)})
    assert not evaluate(one_step_more, DEFAULT_ASSUMPTIONS, {}, db).closes


def test_group_lever_names_the_group_the_design_closes_in(db: Path) -> None:
    req = REQ_FAILED.model_copy(update={"propulsion_type": "series_hybrid", "group_constraint": 1})
    group = {o.lever: o for o in explore_closure(req, DEFAULT_ASSUMPTIONS, db)}["group"]
    assert group.closes and group.change.requirements.group_constraint == 2
    new_req, a, knobs = apply_changes(req, DEFAULT_ASSUMPTIONS, group.change)
    assert evaluate(new_req, a, knobs, db).closes


def test_improvement_compares_against_the_evaluated_unchanged_design(db: Path) -> None:
    baseline = size_aircraft(G3_HYBRID_VTOL)  # unfloored reserve: not the reference for savings
    by_lever = {o.lever: o for o in explore_improvement(G3_HYBRID_VTOL, DEFAULT_ASSUMPTIONS, baseline, db)}
    assert by_lever["endurance"].mtow_saved_kg > 0
    assert "liion_max_c_rate" not in by_lever and "liion_specific_energy" not in by_lever  # no-ops on a hybrid


def test_a_no_op_step_saves_nothing(db: Path) -> None:
    baseline = size_aircraft(G1_ELECTRIC_FW)
    c_rate = {o.lever: o for o in explore_improvement(G1_ELECTRIC_FW, DEFAULT_ASSUMPTIONS, baseline, db)}["liion_max_c_rate"]
    assert abs(c_rate.mtow_saved_kg) < 0.01
