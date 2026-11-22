"""Airframe layout ids, the layout spec and the requirement validation that gates unbuilt layouts."""

import pytest
from pydantic import ValidationError

from src.agent.agent_schemas import RequirementsDraft
from src.agent.parsing import _airframe_fields
from src.schemas import requirements as requirements_module
from src.schemas.assumptions import DEFAULT_ASSUMPTIONS
from src.schemas.changes import AssumptionChanges, ChangeSet, KnobChanges, RequirementChanges, apply_changes
from src.schemas.requirements import OperationalRequirements
from src.tools.sizing.layouts import StaticMarginBand, layout_spec, tail_arm_m
from tests.reference_designs import G1_ELECTRIC_FW

FUSELAGE_LENGTH_M = 1.5
EXPECTED_TAIL_ARM_M = 0.9


def _with(**changes: object) -> OperationalRequirements:
    return OperationalRequirements(**{**G1_ELECTRIC_FW.model_dump(), **changes})


def test_default_layout_is_the_tractor() -> None:
    assert G1_ELECTRIC_FW.airframe_layout == "pod_boom_tractor"


def test_unbuilt_layout_is_rejected_with_reason() -> None:
    with pytest.raises(ValidationError) as err:
        _with(airframe_layout="flying_wing")
    assert "flying_wing" in str(err.value) and "not available yet" in str(err.value)


def test_auto_topology_needs_layout_enabled_for_both(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(requirements_module, "ENABLED_LAYOUTS", frozenset({("pod_boom_tractor", "fixed_wing")}))
    with pytest.raises(ValidationError):
        _with(airframe_topology="auto")
    assert _with(airframe_topology="fixed_wing").airframe_topology == "fixed_wing"


def test_tractor_spec_and_tail_arm() -> None:
    spec = layout_spec("pod_boom_tractor")
    assert spec.static_margin == StaticMarginBand(0.08, 0.14, 0.11)
    assert tail_arm_m(spec, FUSELAGE_LENGTH_M) == pytest.approx(EXPECTED_TAIL_ARM_M)
    with pytest.raises(ValueError):
        layout_spec("flying_wing")


def test_parser_maps_flying_wing_and_validation_explains() -> None:
    draft = RequirementsDraft(mission_name="Wing", airframe_layout="flying_wing", runway_available=True)
    fields = _airframe_fields(draft, [])
    assert fields["airframe_layout"] == "flying_wing"
    with pytest.raises(ValidationError, match="not available yet"):
        _with(**fields)


def test_design_chat_change_carries_layout() -> None:
    change = ChangeSet(label="Layout", requirements=RequirementChanges(airframe_layout="pod_boom_tractor"),
                       assumptions=AssumptionChanges(), knobs=KnobChanges())
    new_req, _, _ = apply_changes(G1_ELECTRIC_FW, DEFAULT_ASSUMPTIONS, change)
    assert new_req.airframe_layout == "pod_boom_tractor"
