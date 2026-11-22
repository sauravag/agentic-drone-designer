"""Airframe layout ids, the layout spec and the requirement validation that gates unbuilt layouts."""

import dataclasses
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.agent import report
from src.agent.agent_schemas import RequirementsDraft
from src.agent.parsing import _airframe_fields
from src.schemas import requirements as requirements_module
from src.schemas.assumptions import DEFAULT_ASSUMPTIONS
from src.schemas.changes import AssumptionChanges, ChangeSet, KnobChanges, RequirementChanges, apply_changes
from src.schemas.requirements import OperationalRequirements
from src.schemas.state import SizingOutput, effective_tail_areas
from src.tools.aero_cad.layout import AircraftLayout, layout_from_sizing
from src.tools.sizing.layouts import StaticMarginBand, layout_spec, tail_arm_m
from src.tools.sizing.mission_solver import size_aircraft
from src.ui import presenters
from tests.reference_designs import G1_ELECTRIC_FW

G1_SIZING_RECORD = Path(__file__).parent / "data" / "flight_control" / "g1" / "sizing.json"

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


def test_tractor_sizing_has_htail_and_fin() -> None:
    sizing = size_aircraft(G1_ELECTRIC_FW)

    assert [t.name for t in sizing.tail_surfaces] == ["htail", "vtail"]
    assert tuple(t.dihedral_deg for t in sizing.tail_surfaces) == (0.0, 90.0)
    assert tuple(t.control for t in sizing.tail_surfaces) == ("elevator", "rudder")
    assert sizing.airframe_layout == "pod_boom_tractor"
    assert effective_tail_areas(sizing.tail_surfaces) == pytest.approx(
        (sizing.horizontal_tail_area_m2, sizing.vertical_tail_area_m2), rel=1e-12)


def test_effective_areas_must_match_surfaces() -> None:
    sizing = size_aircraft(G1_ELECTRIC_FW)

    with pytest.raises(ValidationError):
        SizingOutput.model_validate({**sizing.model_dump(), "horizontal_tail_area_m2": 9.9})


def test_pre_layout_sizing_record_loads() -> None:
    sizing = SizingOutput.model_validate_json(G1_SIZING_RECORD.read_text())

    assert sizing.airframe_layout == "pod_boom_tractor"
    assert sizing.tail_surfaces == ()


def test_sizing_summary_shows_layout() -> None:
    sizing = size_aircraft(G1_ELECTRIC_FW)
    rows = presenters.sizing_summary(sizing)

    assert ("Layout", "Pod-and-boom tractor") in rows
    assert "| Layout | Pod-and-boom tractor |" in report.pairs(rows)


def test_layout_places_every_tail_surface(g1_layout: AircraftLayout) -> None:
    assert [t.name for t in g1_layout.tail_surfaces] == ["htail", "vtail"]
    assert g1_layout.htail is g1_layout.tail_surfaces[0].planform
    assert len(g1_layout.tail_booms) == 1
    assert g1_layout.propeller.is_pusher is False
    assert g1_layout.propeller.x_m == 0.0


def test_missing_surface_property_fails_loudly(g1_layout: AircraftLayout) -> None:
    bare = dataclasses.replace(g1_layout, tail_surfaces=())
    with pytest.raises(ValueError, match="htail"):
        bare.htail


def test_legacy_record_without_tail_surfaces_gives_same_layout(designs: dict) -> None:
    sizing = designs["g1"][0]
    legacy = sizing.model_copy(update={"tail_surfaces": ()})
    new, old = layout_from_sizing(sizing), layout_from_sizing(legacy)
    assert old.tail_booms == new.tail_booms
    assert old.htail.area_m2 == pytest.approx(new.htail.area_m2, rel=1e-12)
    assert old.vtail.semi_span_m == pytest.approx(new.vtail.semi_span_m, rel=1e-12)
