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
from src.tools.flight_control.control_channels import control_channels
from src.tools.flight_control.flight_model import build_flight_model
from src.tools.flight_control.param_generator import generate_params
from src.tools.sizing.layouts import StaticMarginBand, layout_spec, tail_arm_m
from src.tools.sizing.mission_solver import size_aircraft
from src.ui import presenters
from tests.reference_designs import G1_ELECTRIC_FW
from tests.test_flight_control import load_design

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


def test_tail_booms_reject_empty_surfaces() -> None:
    from src.tools.aero_cad.layout import _tail_booms

    with pytest.raises(ValueError, match="no tail surfaces"):
        _tail_booms(1.0, 0.1, ())


# ---------------------------------------------------------------- static-margin band ---
NARROW_BAND = StaticMarginBand(0.05, 0.10, 0.07)
BAND_TOLERANCE = 0.01  # how closely the battery slide can reach the target on a real packing


@pytest.fixture
def narrow_band(monkeypatch: pytest.MonkeyPatch) -> StaticMarginBand:
    """Every reader asks `layouts.layout_spec`; one patch point swaps the band for all of them."""
    from src.tools.sizing import layouts

    spec = dataclasses.replace(layouts.layout_spec("pod_boom_tractor"), static_margin=NARROW_BAND)
    monkeypatch.setattr(layouts, "layout_spec", lambda layout: spec)
    return NARROW_BAND


def test_every_reader_uses_the_layout_band(narrow_band: StaticMarginBand, designs: dict, tmp_path: Path) -> None:
    from src import pipeline
    from src.agent import critic
    from src.tools.aero_cad import cad_packer
    from src.tools.flight_control.flight_model import build_flight_model
    from src.tools.flight_control.perturbation import HOT, SLUGGISH, perturbed_model
    from tests.cad_helpers import _fitted_layout, _synthetic_aero, _synthetic_oml
    from tests.test_flight_control import load_design

    req, sizing, bom, aero, cad = load_design("g2")
    inside, outside = (cad.model_copy(update={"static_margin": m}) for m in (0.06, 0.12))

    def margin_gate(candidate):
        return next(g for g in critic.evaluate_gates(req, sizing, bom, aero, candidate) if g.name == "static_margin")

    assert margin_gate(inside).passed and not margin_gate(outside).passed
    assert "band 0.05-0.1" in margin_gate(outside).detail
    assert not any("static margin" in n for n in pipeline.design_warnings(sizing, aero, inside))
    assert any("outside 0.05-0.1" in n for n in pipeline.design_warnings(sizing, aero, outside))

    flight_model = build_flight_model(req, sizing, bom, aero, cad)
    assert perturbed_model(flight_model, HOT, narrow_band).static_margin == 0.05
    assert perturbed_model(flight_model, SLUGGISH, narrow_band).static_margin == 0.10

    g1_sizing, g1_bom = designs["g1"]
    layout = _fitted_layout(g1_sizing, g1_bom)
    packed = cad_packer.pack_layout(g1_sizing, g1_bom, _synthetic_aero(layout), layout,
                                    _synthetic_oml(layout, tmp_path / "oml.step"))
    assert packed.band == narrow_band
    assert packed.static_margin == pytest.approx(narrow_band.target, abs=BAND_TOLERANCE)


def test_gear_is_one_description(tmp_path: Path) -> None:
    import xml.etree.ElementTree as ET

    from src.tools.fabrication.kit_mass import residual_gear_item
    from src.tools.flight_control.flight_model import build_flight_model
    from src.tools.flight_control.jsbsim_exporter import export_jsbsim
    from src.tools.gear import gear_mass_kg, tricycle_gear
    from tests.test_flight_control import load_design

    inputs = load_design("g1")
    sizing, model = inputs[1], build_flight_model(*inputs)
    gear = tricycle_gear(model.fuselage_length_m, model.fuselage_diameter_m, model.cruise_prop_diameter_m,
                         model.mean_chord_m, model.wingspan_m, model.x_cg_m)
    jsb = export_jsbsim(model, tmp_path)
    xml = ET.parse(jsb.xml_path).getroot()
    assert gear.kind == "tricycle"
    exported = {c.get("name"): [float(c.find(f"location/{a}").text) for a in "xyz"]
                for c in xml.findall("ground_reactions/contact")}
    assert exported == {c.name: pytest.approx([c.x_m, c.y_m, c.z_m], rel=1e-5, abs=1e-9) for c in gear.contacts}
    assert residual_gear_item(sizing, layout_from_sizing(sizing)).mass_kg == gear_mass_kg(
        sizing.airframe_topology, sizing.mtow_kg)


# ------------------------------------------------------- control channels (Task 7) ---
def test_tractor_channels() -> None:
    model = build_flight_model(*load_design("g1"))
    channels = control_channels(model.aero)
    assert [(c.surface, c.servo, c.servo_function) for c in channels] == [
        ("aileron", 1, 4), ("elevator", 2, 19), ("rudder", 4, 21)]
    assert model.channels == channels


def test_servo_functions_come_from_channels() -> None:

    inputs = load_design("g1")
    params = generate_params(build_flight_model(*inputs), inputs[1], inputs[2])
    assert [params[f"SERVO{n}_FUNCTION"] for n in (1, 2, 3, 4)] == [4, 19, 70, 21]


def test_bridge_applies_each_channel() -> None:
    # imports jsbsim, which only the Docker image is guaranteed to have
    bridge = pytest.importorskip("src.tools.flight_control.sitl_bridge", exc_type=ImportError)
    channels = control_channels(build_flight_model(*load_design("g1")).aero)
    pwm = [bridge.PWM_TRIM] * 16
    pwm[1] = 1600
    fdm: dict[str, float] = {}
    bridge._apply_channels(fdm, channels, tuple(pwm))
    assert fdm["fcs/elevator-cmd-norm"] == -bridge._centred(1600)
    assert fdm["fcs/aileron-cmd-norm"] == 0.0 and fdm["fcs/rudder-cmd-norm"] == 0.0
