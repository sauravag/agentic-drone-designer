import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.agent.critic import RESERVE_MIN
from src.agent.evaluator import Evaluation, evaluate, round_fact, with_reserve_floor
from src.agent.report import citable, stray_numbers
from src.schemas.assumptions import DEFAULT_ASSUMPTIONS
from src.schemas.changes import AssumptionChanges, ChangeSet, KnobChanges, RequirementChanges, apply_changes
from tests.reference_designs import REQ_FAILED


@pytest.fixture(scope="module")
def db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from src.tools.database.seed_data import seed_database

    path = tmp_path_factory.mktemp("db") / "catalog.db"
    seed_database(path)
    return path


def _cs(**kw) -> ChangeSet:
    return ChangeSet(label="x", requirements=kw.get("r", RequirementChanges()),
                     assumptions=kw.get("a", AssumptionChanges()), knobs=kw.get("k", KnobChanges()))


def test_the_failed_run_does_not_close_but_hybrid_does(db: Path) -> None:
    assert not evaluate(REQ_FAILED, DEFAULT_ASSUMPTIONS, {}, db).closes
    hybrid = evaluate(REQ_FAILED.model_copy(update={"propulsion_type": "series_hybrid"}), DEFAULT_ASSUMPTIONS, {}, db)
    assert hybrid.closes and hybrid.dod_group == 2 and 12 < hybrid.mtow_kg < 25


def test_an_option_over_the_requested_group_is_not_closing(db: Path) -> None:  # Review Focus 4
    req = REQ_FAILED.model_copy(update={"propulsion_type": "series_hybrid", "group_constraint": 1})
    result = evaluate(req, DEFAULT_ASSUMPTIONS, {}, db)
    assert not result.closes and "Group 1" in result.reason


def test_a_requested_reserve_below_the_floor_is_recorded_not_silently_raised(db: Path) -> None:  # spec 7.3
    hybrid = REQ_FAILED.model_copy(update={"propulsion_type": "series_hybrid"})
    low = DEFAULT_ASSUMPTIONS.model_copy(update={"fuel_reserve_fraction": 0.12})
    result = evaluate(hybrid, low, {}, db)
    assert result.reserve_floor_applied == 0.12
    assert result.floor_note == "sized with the 15 % reserve floor (requested 12 %)"
    assert result.as_facts()["reserve_floor_applied"] == 0.12
    electric = evaluate(REQ_FAILED, DEFAULT_ASSUMPTIONS, {}, db)  # battery reserve 0.20 is above the floor
    assert electric.reserve_floor_applied is None and electric.floor_note == ""


def test_the_reserve_floor_is_carried_into_a_change_set_for_the_powertrain_it_was_sized_with() -> None:
    result = Evaluation(True, 10.0, 2, "series_hybrid", 0.5, 1.0, "")
    carried = with_reserve_floor(_cs(r=RequirementChanges(propulsion_type="series_hybrid")), result,
                                 DEFAULT_ASSUMPTIONS)
    assert carried.assumptions.fuel_reserve_fraction == RESERVE_MIN
    assert carried.assumptions.battery_reserve_fraction is None
    above = _cs(a=AssumptionChanges(fuel_reserve_fraction=0.3))
    assert with_reserve_floor(above, result, DEFAULT_ASSUMPTIONS) == above
    below = _cs(a=AssumptionChanges(fuel_reserve_fraction=0.12))
    assert with_reserve_floor(below, result, DEFAULT_ASSUMPTIONS).assumptions.fuel_reserve_fraction == RESERVE_MIN


def test_solver_facts_are_rounded_so_display_precision_citations_are_not_flagged() -> None:  # finding 3
    facts = Evaluation(True, 16.623456789, 2, "series_hybrid", 0.81234567, 3.4567891, "").as_facts()
    assert facts["mtow_kg"] == 16.62 and facts["fuel_mass_kg"] == 3.46 and round_fact(None) is None
    known = citable(json.dumps(facts))
    assert stray_numbers("closes at 16.6 kg with 3.5 kg of fuel", known) == []
    assert stray_numbers("closes at 16.62 kg with 3.46 kg of fuel", known) == []
    assert stray_numbers("closes at 16.7 kg", known) == ["16.7"]


def test_change_sets_apply_validate_and_describe() -> None:
    cs = ChangeSet(label="runway", requirements=RequirementChanges(runway_available=True, airframe_topology="fixed_wing"),
                   assumptions=AssumptionChanges(liion_max_c_rate=5), knobs=KnobChanges())
    req, a, knobs = apply_changes(REQ_FAILED, DEFAULT_ASSUMPTIONS, cs)
    assert req.airframe_topology == "fixed_wing" and a.liion_max_c_rate == 5 and knobs == {}
    assert REQ_FAILED.airframe_topology != "fixed_wing" and DEFAULT_ASSUMPTIONS.liion_max_c_rate == 3.0
    assert ("Max discharge (Li-ion)", "3", "5") in cs.describe(REQ_FAILED, DEFAULT_ASSUMPTIONS)
    with pytest.raises(ValidationError):
        apply_changes(REQ_FAILED, DEFAULT_ASSUMPTIONS, cs.model_copy(update={"assumptions": AssumptionChanges(liion_max_c_rate=99)}))
    assert _cs().is_empty() and not cs.is_empty()


def test_knob_changes_are_bounded_and_returned() -> None:
    _, _, knobs = apply_changes(REQ_FAILED, DEFAULT_ASSUMPTIONS, _cs(k=KnobChanges(aspect_ratio=10)))
    assert knobs == {"aspect_ratio": 10.0}
    with pytest.raises(ValueError):
        apply_changes(REQ_FAILED, DEFAULT_ASSUMPTIONS, _cs(k=KnobChanges(aspect_ratio=99)))


def test_requirement_changes_run_model_validation() -> None:
    with pytest.raises(ValidationError):
        apply_changes(REQ_FAILED, DEFAULT_ASSUMPTIONS, _cs(r=RequirementChanges(payload_mass_kg=-1)))


def test_describe_skips_no_op_changes() -> None:
    from src.schemas.knobs import OVERRIDE_DEFAULTS

    default_ar = OVERRIDE_DEFAULTS["aspect_ratio"]
    assert _cs(k=KnobChanges(aspect_ratio=default_ar)).describe(REQ_FAILED, DEFAULT_ASSUMPTIONS) == []
    same = _cs(k=KnobChanges(aspect_ratio=12.0))
    assert same.describe(REQ_FAILED, DEFAULT_ASSUMPTIONS, {"aspect_ratio": 12.0}) == []
    assert same.describe(REQ_FAILED, DEFAULT_ASSUMPTIONS, {"aspect_ratio": 10.0}) == [("Aspect ratio", "10", "12")]
    noop = _cs(r=RequirementChanges(payload_mass_kg=REQ_FAILED.payload_mass_kg),
               a=AssumptionChanges(liion_max_c_rate=DEFAULT_ASSUMPTIONS.liion_max_c_rate))
    assert noop.describe(REQ_FAILED, DEFAULT_ASSUMPTIONS) == []


def test_describe_never_uses_exponent_notation() -> None:
    cs = _cs(r=RequirementChanges(service_ceiling_m=12000), a=AssumptionChanges(battery_reserve_fraction=0.15))
    rows = dict((label, (old, new)) for label, old, new in cs.describe(REQ_FAILED, DEFAULT_ASSUMPTIONS))
    assert rows["Service ceiling (m)"][1] == "12000"
    assert rows["Battery reserve"] == ("0.2", "0.15")


def test_number_formatting_is_lossless() -> None:
    from src.schemas.changes import _format

    assert _format(1e-7) != _format(4e-7) and _format(1e-7) == "0.0000001"
    assert _format(0.1234567) != _format(0.1234568)
    assert _format(-0.0) == "0" and _format(-1e-7) == "-0.0000001"
    assert (_format(12000.0), _format(3.0), _format(0.15)) == ("12000", "3", "0.15")
