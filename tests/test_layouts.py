"""Airframe layout ids, the layout spec and the requirement validation that gates unbuilt layouts."""

import dataclasses
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.agent import report
from src.agent.agent_schemas import RequirementsDraft
from src.agent.design_chat import _concise
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
from src.tools.sizing.layouts import (StaticMarginBand, layout_spec, lift_boom_geometry, tail_arm_m,
                                     tail_boom_diameter_m)
from src.tools.sizing.mission_solver import size_aircraft
from src.tools.sizing.topology import resolved_launch_method, select_airframe
from src.ui import presenters
from tests.reference_designs import G1_ELECTRIC_FW, G2_ELECTRIC_VTOL
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


def test_design_chat_error_text_drops_the_pydantic_prefix() -> None:
    with pytest.raises(ValidationError) as err:
        _with(airframe_layout="flying_wing")
    text = _concise(err.value)
    assert "not available yet" in text and "Value error" not in text


def test_placed_tail_true_dihedral_matches_the_sizing_convention() -> None:
    sizing = size_aircraft(G1_ELECTRIC_FW)
    placed = layout_from_sizing(sizing).tail_surfaces
    assert tuple(t.dihedral_deg for t in placed) == tuple(t.dihedral_deg for t in sizing.tail_surfaces) == (0.0, 90.0)
    assert placed[1].planform.dihedral_deg == 0.0  # the stored geometry keeps the OpenVSP convention


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
        _tail_booms(layout_spec("pod_boom_tractor"), 1.0, 0.1, ())


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
    from src.tools.gear import GearInputs, PropellerDisc, gear_mass_kg, landing_gear
    from src.tools.sizing.mass_model import sizing_output_gear
    from tests.test_flight_control import load_design

    inputs = load_design("g1")
    sizing, model = inputs[1], build_flight_model(*inputs)
    gear = landing_gear(GearInputs("runway", model.fuselage_length_m, model.fuselage_diameter_m, model.mean_chord_m,
                                   model.wingspan_m, model.x_cg_m,
                                   PropellerDisc(0.0, 0.0, model.cruise_prop_diameter_m)))
    jsb = export_jsbsim(model, tmp_path)
    xml = ET.parse(jsb.xml_path).getroot()
    assert gear.kind == "tricycle"
    exported = {c.get("name"): [float(c.find(f"location/{a}").text) for a in "xyz"]
                for c in xml.findall("ground_reactions/contact")}
    assert exported == {c.name: pytest.approx([c.x_m, c.y_m, c.z_m], rel=1e-5, abs=1e-9) for c in gear.contacts}
    sized_gear = sizing_output_gear(sizing)
    assert residual_gear_item(sizing, layout_from_sizing(sizing), sized_gear).mass_kg == gear_mass_kg(
        sized_gear, sizing.airframe_topology, sizing.mtow_kg)


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


def test_hand_launch_resolves_fixed_wing() -> None:
    req = _with(runway_available=False, launch_method="hand")
    assert resolved_launch_method(req) == "hand"
    assert select_airframe(req) == "fixed_wing"


def test_fixed_wing_without_runway_must_say_how_it_launches() -> None:
    with pytest.raises(ValidationError, match="needs launch_method='hand'"):
        _with(runway_available=False, airframe_topology="fixed_wing")


def test_runway_launch_without_a_runway_is_rejected() -> None:
    with pytest.raises(ValidationError, match="launch_method='runway' needs runway_available=True"):
        _with(runway_available=False, launch_method="runway")


def test_quadplane_rejects_hand_launch() -> None:
    with pytest.raises(ValidationError, match="launches vertically"):
        _with(launch_method="hand", airframe_topology="quadplane_vtol", runway_available=False)


def test_auto_without_runway_stays_a_vertical_quadplane() -> None:
    req = _with(runway_available=False)
    assert (select_airframe(req), resolved_launch_method(req)) == ("quadplane_vtol", "vertical")


def test_runway_fixed_wing_resolves_to_a_runway_launch() -> None:
    assert resolved_launch_method(G1_ELECTRIC_FW) == "runway"


def test_old_requirements_record_defaults_launch() -> None:
    data = G1_ELECTRIC_FW.model_dump()
    data.pop("launch_method")
    assert OperationalRequirements.model_validate(data).launch_method == "auto"


def test_sizing_summary_has_a_launch_row() -> None:
    sizing = size_aircraft(G1_ELECTRIC_FW)
    assert ("Launch", "runway") in presenters.sizing_summary(sizing)


def _launch_change(**fields: object) -> ChangeSet:
    return ChangeSet(label="launch", requirements=RequirementChanges(**fields), assumptions=AssumptionChanges(),
                     knobs=KnobChanges())


def test_change_set_switches_a_runway_fixed_wing_to_hand_launch() -> None:
    req, _, _ = apply_changes(G1_ELECTRIC_FW, DEFAULT_ASSUMPTIONS, _launch_change(launch_method="hand"))
    assert resolved_launch_method(req) == "hand"
    rows = _launch_change(launch_method="hand").describe(G1_ELECTRIC_FW, DEFAULT_ASSUMPTIONS)
    assert ("Launch", "auto", "hand") in rows


def test_change_set_hand_launch_on_a_quadplane_is_a_readable_error() -> None:
    change = _launch_change(launch_method="hand", airframe_topology="quadplane_vtol")
    with pytest.raises(ValidationError) as exc_info:
        apply_changes(G1_ELECTRIC_FW, DEFAULT_ASSUMPTIONS, change)
    text = _concise(exc_info.value)
    assert "launches vertically" in text and "Traceback" not in text and "pydantic" not in text


def test_change_set_hand_launch_with_the_airframe_switch_succeeds() -> None:
    change = _launch_change(launch_method="hand", airframe_topology="fixed_wing")
    req, _, _ = apply_changes(G2_ELECTRIC_VTOL, DEFAULT_ASSUMPTIONS, change)
    assert (select_airframe(req), resolved_launch_method(req)) == ("fixed_wing", "hand")


def test_legacy_fixed_wing_without_runway_loads_as_hand_launch() -> None:
    """The form once allowed fixed_wing with the runway box off, so such records exist: they were hand launches."""
    data = G1_ELECTRIC_FW.model_dump()
    data.pop("launch_method")
    legacy = OperationalRequirements.model_validate({**data, "airframe_topology": "fixed_wing",
                                                     "runway_available": False})
    assert legacy.launch_method == "hand" and resolved_launch_method(legacy) == "hand"


def test_explicit_launch_method_is_not_migrated() -> None:
    with pytest.raises(ValidationError, match="launch_method='runway' needs runway_available=True"):
        _with(airframe_topology="fixed_wing", runway_available=False, launch_method="runway")


def test_layout_lift_booms_use_shared_geometry() -> None:
    sizing = size_aircraft(G2_ELECTRIC_VTOL)
    layout = layout_from_sizing(sizing)
    length, diameter = lift_boom_geometry(sizing.rotor_diameter_m, layout.wing.root_chord_m,
                                          layout.wing.x_le_root_m, layout.wing.x_ac_m)
    assert layout.vtol_booms[0].length_m == length
    assert layout.vtol_booms[0].diameter_m == diameter


def test_layout_tail_boom_diameter_is_shared() -> None:
    layout = layout_from_sizing(size_aircraft(G1_ELECTRIC_FW))
    assert layout.tail_boom is not None
    assert layout.tail_boom.diameter_m == tail_boom_diameter_m(layout.fuselage_diameter_m)


# ------------------------------------------------------------- V-tail pusher (phase 2, Task 6) ---
import math  # noqa: E402

from src.schemas.assumptions import DesignAssumptions  # noqa: E402
from src.tools.aero_cad.layout import PropellerStation  # noqa: E402
from src.tools.sizing import layouts  # noqa: E402
from src.tools.sizing import mass_model as mm  # noqa: E402
from src.tools.sizing.layouts import (PUSHER_MOTOR_MOUNT_FRACTION, PUSHER_TAIL_GAP_FRACTION, TAILED_BAND,  # noqa: E402
                                      VTAIL_PAIR_ASPECT_RATIO, LayoutSpec)
from src.tools.sizing.mission_solver import cruise_drivetrain_efficiency  # noqa: E402

VTAIL = "single_boom_vtail_pusher"
G1_PROP_DIAMETER_M = 0.279


def _vtail_geometry() -> mm.Geometry:
    return mm.size_geometry(5.0, 0.6, 9.0, 0.45, 0.035, 1.0, layout_spec(VTAIL))


def test_vtail_pusher_spec_is_built_but_not_enabled() -> None:
    assert layout_spec(VTAIL) == LayoutSpec("v", "centre", "pusher_boom_end", "straight", "tail_volume_h", TAILED_BAND)
    assert not any(layout == VTAIL for layout, _ in requirements_module.ENABLED_LAYOUTS)
    assert VTAIL_PAIR_ASPECT_RATIO == 4.0 and PUSHER_TAIL_GAP_FRACTION == 0.10


def test_vtail_area_and_dihedral() -> None:
    g = _vtail_geometry()
    (t,) = g.tail_surfaces
    s_h = 0.45 * 0.6 * g.mean_chord_m / g.tail_arm_m
    s_v = 0.035 * 0.6 * g.wingspan_m / g.tail_arm_m
    assert t.name == "vtail_pair" and t.control == "ruddervator" and t.is_symmetric
    assert t.area_m2 == pytest.approx(s_h + s_v)
    assert t.span_m == pytest.approx(math.sqrt(VTAIL_PAIR_ASPECT_RATIO * (s_h + s_v)))
    assert t.dihedral_deg == pytest.approx(math.degrees(math.atan(math.sqrt(s_v / s_h))))
    assert effective_tail_areas(g.tail_surfaces) == pytest.approx((s_h, s_v))
    assert (g.horizontal_tail_area_m2, g.vertical_tail_area_m2) == pytest.approx((s_h, s_v))
    assert g.tail_arm_m == pytest.approx(tail_arm_m(layout_spec(VTAIL), g.fuselage_length_m))


def test_unknown_tail_kind_has_no_sizing_rule() -> None:
    spec = dataclasses.replace(layout_spec(VTAIL), tail="inverted_v")
    with pytest.raises(ValueError, match="inverted_v"):
        mm.size_geometry(5.0, 0.6, 9.0, 0.45, 0.035, 1.0, spec)


def test_pusher_efficiency_applies_only_to_pushers() -> None:
    assumptions = DesignAssumptions(drivetrain_efficiency=0.70)
    tractor = cruise_drivetrain_efficiency(layout_spec("pod_boom_tractor"), assumptions)
    pusher = cruise_drivetrain_efficiency(layout_spec(VTAIL), assumptions)
    assert tractor == 0.70
    assert pusher == pytest.approx(0.95 * tractor)
    custom = assumptions.model_copy(update={"pusher_efficiency_factor": 0.85})
    assert cruise_drivetrain_efficiency(layout_spec(VTAIL), custom) == pytest.approx(0.85 * 0.70)


@pytest.mark.parametrize("bad", [0.79, 1.01])
def test_pusher_efficiency_factor_is_bounded(bad: float) -> None:
    assert DEFAULT_ASSUMPTIONS.pusher_efficiency_factor == 0.95
    with pytest.raises(ValidationError):
        DesignAssumptions(pusher_efficiency_factor=bad)


def test_change_set_can_set_the_pusher_efficiency_factor() -> None:
    change = ChangeSet(label="pusher", requirements=RequirementChanges(),
                       assumptions=AssumptionChanges(pusher_efficiency_factor=0.9), knobs=KnobChanges())
    _, assumptions, _ = apply_changes(G1_ELECTRIC_FW, DEFAULT_ASSUMPTIONS, change)
    assert assumptions.pusher_efficiency_factor == 0.9


def test_pusher_tail_boom_reaches_past_the_tail_by_the_gap() -> None:
    g = _vtail_geometry()
    spec = layout_spec(VTAIL)
    as_tractor = dataclasses.replace(spec, propeller="tractor_nose")
    (pusher_boom,) = mm.sizing_booms(g, spec, 0.0)
    (plain_boom,) = mm.sizing_booms(g, as_tractor, 0.0)
    gap = PUSHER_TAIL_GAP_FRACTION * mm.PROP_TO_FUSELAGE_DIAMETER_RATIO * g.fuselage_diameter_m  # sizing's estimate
    assert pusher_boom.length_m == pytest.approx(plain_boom.length_m + gap)
    assert mm.boom_mass_kg(pusher_boom) > mm.boom_mass_kg(plain_boom)


def test_sizing_gear_puts_the_pusher_disc_at_the_boom_end() -> None:
    from src.tools.gear import GearInputs, PropellerDisc, landing_gear

    g, spec = _vtail_geometry(), layout_spec(VTAIL)
    (boom,) = mm.sizing_booms(g, spec, 0.0)
    boom_end = mm.TAIL_BOOM_START_FRACTION * g.fuselage_length_m + boom.length_m
    diameter = mm.PROP_TO_FUSELAGE_DIAMETER_RATIO * g.fuselage_diameter_m
    _, _, x_ac = mm._default_station_wing_m(g)
    x_cg = x_ac - spec.static_margin.target * g.mean_chord_m
    expected = landing_gear(GearInputs("runway", g.fuselage_length_m, diameter, g.mean_chord_m, g.wingspan_m, x_cg,
                                       PropellerDisc(boom_end + PUSHER_MOTOR_MOUNT_FRACTION * diameter, 0.0, diameter)))
    gear = mm.sizing_gear(g, spec, "runway")
    assert gear == expected
    assert gear.leg_height_m > gear.base_leg_height_m  # the take-off rotation sets the leg, not the fuselage


def test_pusher_pod_has_no_estimate_yet() -> None:
    spec = dataclasses.replace(layout_spec(VTAIL), propeller="pusher_pod")
    with pytest.raises(ValueError, match="not built yet"):
        mm.sizing_gear(_vtail_geometry(), spec, "runway")


def test_vtail_pusher_sizes_with_a_ruddervator_pair(vtail_g1_sizing: SizingOutput) -> None:
    sizing = vtail_g1_sizing
    assert sizing.airframe_layout == VTAIL
    assert [(t.name, t.control) for t in sizing.tail_surfaces] == [("vtail_pair", "ruddervator")]
    assert sizing.gear_kind == "tricycle"
    assert sizing.empty_mass_breakdown["tail_boom"] > 0


def test_vtail_layout_places_one_mirrored_surface(vtail_g1_sizing: SizingOutput) -> None:
    layout = layout_from_sizing(vtail_g1_sizing, propeller_diameter_m=G1_PROP_DIAMETER_M)
    (pair,) = layout.tail_surfaces
    assert pair.control == "ruddervator" and not pair.is_vertical and pair.planform.is_symmetric
    assert pair.dihedral_deg == pytest.approx(vtail_g1_sizing.tail_surfaces[0].dihedral_deg)
    assert pair.planform.x_ac_m - layout.wing.x_ac_m == pytest.approx(vtail_g1_sizing.tail_arm_m)


def test_boom_end_station_behind_vtail(vtail_g1_sizing: SizingOutput) -> None:
    layout = layout_from_sizing(vtail_g1_sizing, propeller_diameter_m=G1_PROP_DIAMETER_M)
    te = max(t.planform.x_le_root_m + t.planform.root_chord_m for t in layout.tail_surfaces)
    assert layout.propeller.is_pusher and layout.tail_boom.x_end_m == pytest.approx(te + 0.0279)
    assert layout.propeller.x_m == pytest.approx(layout.tail_boom.x_end_m
                                                 + PUSHER_MOTOR_MOUNT_FRACTION * G1_PROP_DIAMETER_M)
    assert layout.propeller.z_m == layout.tail_boom.z_m
    assert layout.propeller.diameter_m == G1_PROP_DIAMETER_M


def test_tractor_station_gains_only_the_diameter(designs: dict) -> None:
    sizing = designs["g1"][0]
    plain, with_d = layout_from_sizing(sizing), layout_from_sizing(sizing, propeller_diameter_m=G1_PROP_DIAMETER_M)
    assert with_d.propeller == PropellerStation(0.0, 0.0, 0.0, False, G1_PROP_DIAMETER_M)
    assert dataclasses.replace(with_d, propeller=plain.propeller) == plain


def test_pusher_pod_station_is_not_built_yet(vtail_g1_sizing: SizingOutput, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(layouts._SPECS, VTAIL, dataclasses.replace(layout_spec(VTAIL), propeller="pusher_pod"))
    with pytest.raises(ValueError, match="not built yet"):
        layout_from_sizing(vtail_g1_sizing, propeller_diameter_m=G1_PROP_DIAMETER_M)


def test_flight_model_gear_uses_the_cad_propeller_station() -> None:
    from src.tools.gear import GearInputs, PropellerDisc, landing_gear

    req, sizing, bom, aero, cad = load_design("g1")
    tractor = build_flight_model(req, sizing, bom, aero, cad)
    at_nose = cad.model_copy(update={"propeller_x_m": 0.0, "propeller_z_m": 0.0})
    assert build_flight_model(req, sizing, bom, aero, at_nose).gear == tractor.gear
    x_hub = 1.3
    pusher = build_flight_model(req, sizing.model_copy(update={"airframe_layout": VTAIL}), bom, aero,
                                cad.model_copy(update={"propeller_x_m": x_hub, "propeller_z_m": 0.0}))
    expected = landing_gear(GearInputs("runway", sizing.fuselage_length_m, sizing.fuselage_diameter_m,
                                       sizing.mean_chord_m, sizing.wingspan_m, cad.x_cg_m,
                                       PropellerDisc(x_hub, 0.0, bom.cruise_propeller.diameter_m)))
    assert pusher.gear == expected
    assert pusher.gear.leg_height_m > tractor.gear.leg_height_m


def test_pusher_flight_model_needs_the_cad_propeller_station() -> None:
    req, sizing, bom, aero, cad = load_design("g1")
    with pytest.raises(ValueError, match="propeller station"):
        build_flight_model(req, sizing.model_copy(update={"airframe_layout": VTAIL}), bom, aero, cad)


# The catalog snaps the cruise propeller to discrete sizes (11 -> 13 in near G1 is an 18 % step), so a single ratio
# cannot beat about half a step (9 %) on any one design; 10 % allows that plus the spread across the references.
PROP_ESTIMATE_REL_TOL = 0.10


def test_sizing_propeller_estimate_tracks_the_real_bom_propeller(designs: dict) -> None:
    sizing, bom = designs["g1"]
    estimate = mm.PROP_TO_FUSELAGE_DIAMETER_RATIO * sizing.fuselage_diameter_m
    assert estimate == pytest.approx(bom.cruise_propeller.diameter_m, rel=PROP_ESTIMATE_REL_TOL)


def test_sizing_gear_and_boom_use_the_propeller_estimate() -> None:
    g, spec = _vtail_geometry(), layout_spec(VTAIL)
    diameter = mm.PROP_TO_FUSELAGE_DIAMETER_RATIO * g.fuselage_diameter_m
    (boom,) = mm.sizing_booms(g, spec, 0.0)
    (plain,) = mm.sizing_booms(g, dataclasses.replace(spec, propeller="tractor_nose"), 0.0)
    assert boom.length_m == pytest.approx(plain.length_m + PUSHER_TAIL_GAP_FRACTION * diameter)
    tractor = mm.sizing_gear(g, layout_spec("pod_boom_tractor"), "runway")
    assert tractor.base_leg_height_m == pytest.approx(diameter / 2 * 1.25)
    assert tractor.leg_height_m == tractor.base_leg_height_m  # a nose disc never needs a taller leg: mass ratio 1


LONG_POD_FACTOR = 4.0  # a pod so long that the V-tail and the propeller gap end ahead of its tail


def test_sizing_books_the_pusher_boom_whenever_the_layout_builds_it() -> None:
    spec = layout_spec(VTAIL)
    g = mm.size_geometry(5.0, 0.6, 9.0, 0.45, 0.035, LONG_POD_FACTOR, spec)
    assert mm._tail_boom_end_m(g, spec) < g.fuselage_length_m
    assert mm.sizing_booms(g, dataclasses.replace(spec, propeller="tractor_nose"), 0.0) == ()  # tail on the pod
    (boom,) = mm.sizing_booms(g, spec, 0.0)  # but the pusher's boom carries the motor
    start = mm.TAIL_BOOM_START_FRACTION * g.fuselage_length_m
    assert boom.length_m == pytest.approx(mm._tail_boom_end_m(g, spec) - start)


def test_sizing_refuses_a_pusher_boom_ending_ahead_of_its_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(layouts, "PUSHER_TAIL_GAP_FRACTION", -5.0)
    with pytest.raises(ValueError, match="ahead of its root"):
        mm.sizing_booms(_vtail_geometry(), layout_spec(VTAIL), 0.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_cad_record_rejects_a_non_finite_propeller_station(bad: float) -> None:
    from src.schemas.state import CADArtifacts

    _, _, _, _, cad = load_design("g1")
    for field in ("propeller_x_m", "propeller_z_m"):
        with pytest.raises(ValidationError):
            CADArtifacts.model_validate({**cad.model_dump(), field: bad})
