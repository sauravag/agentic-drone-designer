from pathlib import Path

import pytest
from pydantic import ValidationError

from src.agent.evaluator import evaluate
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
