"""Task 6: `build_kit` assembles every part, tube and strip of the laser-cut kit; `kit_mass_items` weighs it."""

import dataclasses
import math
import re

import build123d as bd
import numpy as np
import pytest

from src.schemas.fabrication import DEFAULT_FABRICATION, TubeStock
from src.tools.aero_cad import components
from src.tools.aero_cad import mass_properties as mp
from src.tools.aero_cad.airfoil_geometry import load_airfoil
from src.tools.aero_cad.clearance import prop_clearance
from src.tools.aero_cad.layout import FUSELAGE_PROFILE, TAIL_AIRFOIL_NAME, fuselage_radius_at
from src.tools.fabrication.build_kit import MITRE_LABELS, _covering, build_kit, socket_end_m, spar_tubes
from src.tools.fabrication.fittings import join_notes
from src.tools.fabrication.fuselage_parts import STRINGER_COUNT, former_stations, socket_axis_m
from src.tools.fabrication.kit import (
    BOOM_MOTOR_WIRING_NOTE,
    FOLDING_PROP_NOTE,
    TAIL_CONE_SLOT_NOTE,
    Kit,
    KitUnavailable,
    TubePart,
    with_clearance_notes,
)
from src.tools.fabrication.kit_export import build_sheet_text
from src.tools.fabrication.kit_mass import kit_mass_items, kit_mass_kg, residual_gear_item, tube_item
from src.tools.fabrication.tubes import SparPlan, SparSegment
from src.tools.fabrication.wing_parts import PIN_CHORD_FRACTION, panel_rib_stations, spar_depth_at, spar_member_at
from src.tools.fabrication.wing_stiffness import dbox_material_m2, dbox_web_x_m
from src.tools.gear import GEAR_FRACTION, GearGeometry
from src.tools.sizing.mass_model import sizing_output_gear
from tests.cad_helpers import _fitted_layout, hole_diameters, part_solid, tube_solid
from tests.test_fabrication_fuselage import _assert_no_intersections, _real_components, _test_battery

SETTINGS = DEFAULT_FABRICATION
MASS_REL = 1e-6
INERTIA_REL = 1e-3
PIN_REACH_M = 0.020  # R18: the pin runs this far outboard of the root doubler
USABLE_LONG_M, USABLE_SHORT_M = sorted(DEFAULT_FABRICATION.usable_bed_m, reverse=True)
RIM_ALLOWANCE_M = 0.003  # a square-cut tube end may stop this far short so its leaning rim clears a rib


def _placed(designs, case: str, layout):
    return (*_real_components(designs, case, layout), *components.vtol_motors(designs[case][1], layout))


def _build(designs, inputs, case: str):
    layout, airfoil = inputs[0], inputs[1]
    return build_kit(designs[case][0], layout, airfoil, _placed(designs, case, layout), SETTINGS)


@pytest.fixture(scope="module")
def g1_kit(designs, g1_kit_inputs):
    return _build(designs, g1_kit_inputs, "g1")


@pytest.fixture(scope="module")
def g2_kit(designs, g2_kit_inputs):
    return _build(designs, g2_kit_inputs, "g2")


def _items(kit) -> dict[str, mp.MassItem]:
    return {i.name: i for i in kit_mass_items(kit, SETTINGS)}


def _assert_item_matches(item: mp.MassItem, oracle: mp.MassItem, label: str) -> None:
    assert item.mass_kg == pytest.approx(oracle.mass_kg, rel=MASS_REL), label
    scale = max(abs(c) for c in oracle.cg_m)
    assert np.allclose(item.cg_m, oracle.cg_m, rtol=0.0, atol=MASS_REL * scale), label
    tensor, expected = np.array(item.inertia_kgm2), np.array(oracle.inertia_kgm2)
    assert np.abs(tensor - expected).max() <= INERTIA_REL * np.abs(expected).max(), label


# ------------------------------------------------------------------------------------------ group ---
def test_group_3_is_unavailable(designs, g1_kit_inputs):
    sizing = designs["g3"][0]
    assert sizing.dod_group == 3
    layout, airfoil = g1_kit_inputs[0], g1_kit_inputs[1]
    with pytest.raises(KitUnavailable, match="Group 1–2"):
        build_kit(sizing, layout, airfoil, (), SETTINGS)


def _mitre_name(label: str) -> str:
    """`_assert_no_intersections` skips pairs whose names both start with 'socket'."""
    return f"socket {label}" if label in MITRE_LABELS else label


@pytest.mark.parametrize("case", ["g1", "g2"])
def test_every_reference_design_in_group_1_2_gets_a_kit(case, request):
    """g1 (4.1 kg fixed wing) and g2 (19.4 kg quadplane) with real components: nothing intersects but the
    centreline mitre of the sockets and the spar roots inside them (R14)."""
    kit = request.getfixturevalue(f"{case}_kit")
    spar = request.getfixturevalue(f"{case}_kit_inputs")[2]
    assert kit.spar == spar
    assert kit.parts and kit.tubes and kit.linear
    solids = [(p.label, part_solid(p, SETTINGS)) for p in kit.parts]
    solids += [(_mitre_name(t.label), tube_solid(t)) for t in kit.tubes]
    _assert_no_intersections(solids)


def test_only_the_centreline_mitre_is_exempt():
    assert set(MITRE_LABELS) == {"SK-R", "SK-L", "SP-R1", "SP-L1"}


@pytest.mark.parametrize("case", ["g1", "g2"])
def test_same_side_socket_and_spar_root_do_not_intersect(case, request):
    kit = request.getfixturevalue(f"{case}_kit")
    by = {t.label: t for t in kit.tubes}
    for side in "RL":
        common = tube_solid(by[f"SK-{side}"]).intersect(tube_solid(by[f"SP-{side}1"]))
        pieces = common if isinstance(common, list) else ([] if common is None else [common])
        assert sum(p.volume for p in pieces) < 1e-12


def test_every_part_fits_the_bed(g1_kit, g2_kit):
    for kit in (g1_kit, g2_kit):
        for part in kit.parts:
            size = part.outline.bounding_box().size
            long, short = max(size.X, size.Y), min(size.X, size.Y)
            assert long <= USABLE_LONG_M + 1e-9 and short <= USABLE_SHORT_M + 1e-9, part.label


def test_joins_describe_every_split(g2_kit):
    assert g2_kit.joins == join_notes(g2_kit.parts)
    assert len({p.label for p in g2_kit.parts}) == len(g2_kit.parts)


def test_kit_is_deterministic(designs, g1_kit_inputs, g1_kit):
    again = _build(designs, g1_kit_inputs, "g1")
    assert [p.label for p in again.parts] == [p.label for p in g1_kit.parts]
    assert [p.area_m2 for p in again.parts] == [p.area_m2 for p in g1_kit.parts]
    assert [t.label for t in again.tubes] == [t.label for t in g1_kit.tubes]
    assert [(i.name, i.mass_kg) for i in kit_mass_items(again, SETTINGS)] == [
        (i.name, i.mass_kg) for i in kit_mass_items(g1_kit, SETTINGS)
    ]


# ------------------------------------------------------------------------------------------ tubes ---
def _axis_point(layout, side: float, y: float, fraction: float = 0.25):
    x, z = socket_axis_m(layout, y, fraction)
    return np.array([x, side * y, z])


def _tube_end(tube: TubePart):
    return np.array(tube.start_m) + tube.length_m * np.array(tube.axis)


@pytest.mark.parametrize("case", ["g1", "g2"])
def test_spar_members_telescope_with_bonded_overlaps(case, request):
    """R18: each inner member reaches min_insertion_diameters x its OD into its outer member; the root runs
    from the centreline (inside the socket) to the first step."""
    kit = request.getfixturevalue(f"{case}_kit")
    layout, _, spar, _ = request.getfixturevalue(f"{case}_kit_inputs")
    by = {t.label: t for t in kit.tubes}
    for side, name in ((1.0, "R"), (-1.0, "L")):
        assert f"SP-{name}{len(spar.segments) + 1}" not in by
        previous_end = _axis_point(layout, side, 0.0)
        for k, seg in enumerate(spar.segments):
            member = by[f"SP-{name}{k + 1}"]
            assert member.role == "spar" and member.tube == seg.tube
            end = _tube_end(member)
            assert np.allclose(np.cross(end - _axis_point(layout, side, 0.0), member.axis), 0.0, atol=1e-9)
            assert seg.y_end_m - RIM_ALLOWANCE_M <= abs(end[1]) <= seg.y_end_m + 1e-9  # trimmed clear of a rib
            overlap = 0.0 if k == 0 else SETTINGS.min_insertion_diameters * seg.tube.od_m
            start = previous_end - overlap * np.array(member.axis)  # from the outer member's actual end
            assert member.start_m == pytest.approx(tuple(start), abs=1e-9)
            previous_end = end


@pytest.mark.parametrize("case", ["g1", "g2"])
def test_sockets_run_from_the_centreline_to_the_skin_joint(case, request):
    kit = request.getfixturevalue(f"{case}_kit")
    layout, _, spar, _ = request.getfixturevalue(f"{case}_kit_inputs")
    by = {t.label: t for t in kit.tubes}
    for side, name in ((1.0, "R"), (-1.0, "L")):
        socket = by[f"SK-{name}"]
        assert socket.role == "socket" and socket.tube == spar.socket
        assert socket.start_m == pytest.approx(tuple(_axis_point(layout, side, 0.0)), abs=1e-9)
        end = socket_end_m(layout, spar)
        assert spar.joint_y_m - RIM_ALLOWANCE_M < end < spar.joint_y_m
        assert _tube_end(socket) == pytest.approx(_axis_point(layout, side, end), abs=1e-9)


def test_pins_run_from_the_plate_to_past_the_root_doubler(g1_kit, g1_kit_inputs):
    layout, _, spar, _ = g1_kit_inputs
    parts, by = {p.label: p for p in g1_kit.parts}, {t.label: t for t in g1_kit.tubes}
    t_plate, t_doubler = SETTINGS.sheet_for("plate").thickness_m, SETTINGS.sheet_for("doubler").thickness_m
    inner = parts["WM-R"].location.position.Y - t_plate
    outer = spar.joint_y_m + t_doubler + PIN_REACH_M
    for side, name in ((1.0, "R"), (-1.0, "L")):
        pin = by[f"PN-{name}"]
        assert pin.role == "pin" and (pin.tube.od_m, pin.tube.id_m) == (0.004, 0.0)
        assert pin.start_m == pytest.approx(tuple(_axis_point(layout, side, inner, PIN_CHORD_FRACTION)), abs=1e-9)
        assert _tube_end(pin) == pytest.approx(_axis_point(layout, side, outer, PIN_CHORD_FRACTION), abs=1e-9)


def test_booms_and_tail_spars_are_in_the_kit(g1_kit, g2_kit):
    def labels(kit) -> list[str]:
        return sorted(t.label for t in kit.tubes if t.role in ("tail_boom", "lift_boom", "tail_spar"))

    assert labels(g1_kit) == ["TB-1", "TS-HL", "TS-HR", "TS-V"]
    assert labels(g2_kit) == ["LB-L", "LB-R", "TB-1", "TS-HL", "TS-HR", "TS-V"]


@pytest.mark.parametrize("case", ["g1", "g2"])
def test_tail_spars_run_out_as_far_as_the_tail_depth_admits(case, request):
    """Each tail spar reaches the tip, or the station where the tail grows too thin for it (never 'one rib')."""
    kit = request.getfixturevalue(f"{case}_kit")
    layout = request.getfixturevalue(f"{case}_kit_inputs")[0]
    tail = load_airfoil(TAIL_AIRFOIL_NAME)
    by = {t.label: t for t in kit.tubes}
    for label, surface, coord, origin in (("TS-HR", layout.htail, 1, 0.0), ("TS-V", layout.vtail, 2, layout.vtail.z_root_m)):
        spar = by[label]
        reach = _tube_end(spar)[coord] - origin
        depth = spar_depth_at(surface, tail, SETTINGS, root_doubler_m=None)
        assert depth(reach) >= spar.tube.od_m - 1e-6, label
        assert reach == pytest.approx(surface.semi_span_m, abs=1e-6) or depth(reach + 0.005) < spar.tube.od_m, label


# ----------------------------------------------------------------------------------------- linear ---
def test_stringers_run_between_the_first_and_last_former(designs, g1_kit, g1_kit_inputs):
    layout, _, spar, _ = g1_kit_inputs
    stations = former_stations(layout, spar, SETTINGS, _placed(designs, "g1", layout))
    stringers = [s for s in g1_kit.linear if s.kind == "stringer"]
    assert len(stringers) == STRINGER_COUNT
    t_former, side = SETTINGS.sheet_for("former").thickness_m, SETTINGS.stringer_side_m
    first, last = stations[1], stations[-1] + t_former
    for s in stringers:
        assert s.start_m[0] == pytest.approx(first) and s.end_m[0] == pytest.approx(last)
        for x, (_, y, z) in ((first, s.start_m), (last, s.end_m)):
            r_in = fuselage_radius_at(layout, x) - SETTINGS.fuselage_cover_thickness_m
            assert math.hypot(y, z) == pytest.approx(r_in - side / 2.0, abs=1e-9)
        assert s.section_area_m2 == pytest.approx(side**2)
        assert s.length_m == pytest.approx(math.dist(s.start_m, s.end_m))


def test_le_and_te_strips_per_panel_and_tail_surface(g1_kit):
    for kind in ("le", "te"):
        strips = sorted(s.label for s in g1_kit.linear if s.kind == kind)
        assert strips == sorted(f"{kind.upper()}-{s}" for s in ("WR", "WL", "HR", "HL", "V"))


# ------------------------------------------------------------------------------------------- mass ---
def test_plate_mass_matches_extruded_solid(g1_kit):
    items = _items(g1_kit)
    for part in g1_kit.parts:
        density = SETTINGS.sheet_for(part.family).density_kg_m3
        oracle = mp.solid_item(part.label, part_solid(part, SETTINGS), density)
        _assert_item_matches(items[part.label], oracle, part.label)


def test_tube_mass_matches_hollow_solid(g2_kit):
    items = _items(g2_kit)
    for tube in g2_kit.tubes:
        oracle = mp.solid_item(tube.label, tube_solid(tube), SETTINGS.carbon_density_kg_m3)
        _assert_item_matches(items[tube.label], oracle, tube.label)


def test_tube_item_follows_an_oblique_axis():
    direction = np.array([0.3, 0.8, -0.2])
    axis = tuple(float(v) for v in direction / np.linalg.norm(direction))
    tube = TubePart("T", "spar", SETTINGS.tubes[5], 0.4, (0.1, -0.2, 0.05), axis)
    density = SETTINGS.carbon_density_kg_m3
    item = tube_item("T", tube.tube.area_m2 * tube.length_m * density, tube.start_m, tube.axis, tube.length_m,
                     tube.tube.od_m, tube.tube.id_m)
    _assert_item_matches(item, mp.solid_item("T", tube_solid(tube), density), "oblique tube")


def _frustum_area(layout) -> float:
    length = layout.fuselage_length_m
    xs = [0.0, *(f * length for f, _ in FUSELAGE_PROFILE), length]
    rs = [fuselage_radius_at(layout, x) for x in xs]
    return sum(math.pi * (r0 + r1) * math.hypot(x1 - x0, r1 - r0) for x0, x1, r0, r1 in zip(xs, xs[1:], rs, rs[1:]))


@pytest.mark.parametrize("case", ["g1", "g2"])
def test_kit_mass_is_sum_of_parts_tubes_stock_covering_and_glue(case, request):
    kit = request.getfixturevalue(f"{case}_kit")
    layout = request.getfixturevalue(f"{case}_kit_inputs")[0]
    items = kit_mass_items(kit, SETTINGS)
    by = {i.name: i for i in items}
    assert len(by) == len(items)
    assert kit_mass_kg(kit, SETTINGS) == pytest.approx(sum(i.mass_kg for i in items), rel=1e-12)
    structure = [by[p.label] for p in kit.parts] + [by[t.label] for t in kit.tubes] + [by[s.label] for s in kit.linear]
    structure += [by[c.label] for c in kit.covering if c.kind == "dbox"]  # glued balsa joins the glue base
    glue = by["glue_hardware"]
    assert glue.mass_kg == pytest.approx(SETTINGS.glue_fraction * sum(i.mass_kg for i in structure), rel=1e-12)
    assert glue.cg_m == pytest.approx(mp.combine(structure).cg_m, abs=1e-12)
    assert kit.film_area_m2 == pytest.approx(2.0 * _exposed_planform_m2(layout, kit), rel=1e-9)  # M3
    assert kit.film_area_m2 < 2.0 * (layout.wing.area_m2 + layout.htail.area_m2 + layout.vtail.area_m2)
    assert kit.fuselage_cover_area_m2 == pytest.approx(_frustum_area(layout), rel=1e-6)
    film = sum(i.mass_kg for n, i in by.items() if n.startswith("film_"))
    assert film == pytest.approx(SETTINGS.film_kg_m2 * kit.film_area_m2, rel=1e-9)
    cover = SETTINGS.fuselage_cover_kg_m2 * kit.fuselage_cover_area_m2
    assert by["fuselage_cover"].mass_kg == pytest.approx(cover, rel=1e-9)
    named = {p.label for p in kit.parts} | {t.label for t in kit.tubes} | {s.label for s in kit.linear}
    expected_extra = {"glue_hardware", "fuselage_cover", "film_wing", "film_htail", "film_vtail", "OW-1"}
    assert set(by) - named == expected_extra | ({"dbox_wing"} if kit.spar.dbox else set())
    assert by["OW-1"].mass_kg == pytest.approx(kit.overwrap.mass_kg, rel=1e-12)  # R24: its own mass, glue-free


def _exposed_m2(surface, start_m: float) -> float:
    """Surface area outboard of `start_m` (both panels of a symmetric surface): a trapezoid per panel, in its
    dihedral plane (the film covers the true surface, 1 / cos(dihedral) more than the projected planform)."""
    panels = 2.0 if surface.is_symmetric else 1.0
    slope = 1.0 / math.cos(math.radians(surface.dihedral_deg)) if surface.is_symmetric else 1.0
    return panels * slope * 0.5 * (surface.chord_at(start_m) + surface.chord_at(surface.semi_span_m)) * (
        surface.semi_span_m - start_m)


def _exposed_planform_m2(layout, kit) -> float:
    from src.tools.fabrication.wing_parts import fin_start_m, tail_root_radius_m

    return (_exposed_m2(layout.wing, kit.spar.joint_y_m) + _exposed_m2(layout.htail, tail_root_radius_m(layout))
            + _exposed_m2(layout.vtail, fin_start_m(layout, SETTINGS)))


def test_film_sits_on_the_exposed_wing_planform_centroid(g1_kit, g1_kit_inputs):
    """M3: the film covers the planform outboard of the skin joint, not the part inside the pod."""
    wing = g1_kit_inputs[0].wing
    c0, c1 = wing.chord_at(g1_kit.spar.joint_y_m), wing.chord_at(wing.semi_span_m)
    x_centroid = wing.x_le_root_m + (c0 * c0 + c0 * c1 + c1 * c1) / (3 * (c0 + c1))
    item = _items(g1_kit)["film_wing"]
    assert item.cg_m[0] == pytest.approx(x_centroid, rel=1e-9)
    assert item.cg_m[1] == pytest.approx(0.0, abs=1e-12)


def test_residual_gear_item_is_the_gear_fraction_over_the_pod(designs, g2_kit_inputs):
    sizing, layout = designs["g2"][0], g2_kit_inputs[0]
    item = residual_gear_item(sizing, layout, sizing_output_gear(sizing))
    assert item.name == "landing_gear"
    assert item.mass_kg == pytest.approx(GEAR_FRACTION[sizing.airframe_topology] * sizing.mtow_kg)
    assert item.cg_m == pytest.approx((layout.fuselage_length_m / 2.0, 0.0, 0.0))
    expected = mp.cylinder_item("landing_gear", item.mass_kg, item.cg_m, layout.fuselage_length_m,
                                layout.fuselage_diameter_m)
    assert np.allclose(item.inertia_kgm2, expected.inertia_kgm2, rtol=1e-12, atol=0.0)


# ------------------------------------------------------------------------------------ fix round ---

def test_kit_requires_its_covering_and_derives_the_areas(g1_kit):
    field = next(f for f in dataclasses.fields(Kit) if f.name == "covering")
    assert field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING
    assert {f.name for f in dataclasses.fields(Kit)}.isdisjoint({"film_area_m2", "fuselage_cover_area_m2"})
    assert g1_kit.film_area_m2 == sum(c.area_m2 for c in g1_kit.covering if c.kind == "film")
    assert g1_kit.fuselage_cover_area_m2 == sum(c.area_m2 for c in g1_kit.covering if c.kind == "fuselage")


def test_inner_member_overlap_survives_a_trimmed_outer_end(g1_kit_inputs):
    """R18 after a trim: a step just inboard of a rib pulls the outer end in; the inner member still reaches
    min_insertion_diameters x its OD inside the outer member's actual end."""
    layout, _, spar, settings = g1_kit_inputs
    joint, semi = spar.joint_y_m, layout.wing.semi_span_m
    station = panel_rib_stations(joint, semi)[2]
    step = station - settings.sheet_for("rib").thickness_m / 2.0 + 0.0002  # the rib's inner face is inboard of it
    outer, inner = TubeStock(od_m=0.012, id_m=0.010), TubeStock(od_m=0.010, id_m=0.008)
    plan = SparPlan((SparSegment(outer, joint, step), SparSegment(inner, step, semi)), spar.socket, joint, joint,
                    spar.centre_moment_nm, spar.pin)
    by = {t.label: t for t in spar_tubes(layout, plan, settings)}
    end = _tube_end(by["SP-R1"])
    assert end[1] < step - 1e-6  # trimmed
    axis = np.array(by["SP-R2"].axis)
    overlap = float(np.dot(end - np.array(by["SP-R2"].start_m), axis))
    assert overlap >= settings.min_insertion_diameters * inner.od_m - 1e-9


def test_g2_lift_booms_are_strength_sized(g2_kit):
    lift = [t for t in g2_kit.tubes if t.role == "lift_boom"]
    mass = sum(t.tube.area_m2 * t.length_m * SETTINGS.carbon_density_kg_m3 for t in lift)
    assert len(lift) == 2 and mass < 1.0  # R13's widest stock tube weighed 1.18 kg


def test_g2_lift_booms_are_stiff_enough(g2_kit):
    """R22: the hand estimate (68.8 N at 0.691 m needs I >= 1.56e-8 m^4) gives 32 x 28 mm."""
    for tube in (t for t in g2_kit.tubes if t.role == "lift_boom"):
        assert (round(tube.tube.od_m * 1e3), round(tube.tube.id_m * 1e3)) == (32, 28)


# ------------------------------------------------------------------------------------------- D-box (R45) ---
def test_dbox_covering_area_and_mass(designs, g2_kit_inputs):
    # Arrange: a non-default wood density, so a missing areal-density mapping cannot pass by coincidence
    layout, airfoil, _, _ = g2_kit_inputs
    settings = SETTINGS.model_copy(update={"stringer_density_kg_m3": 200.0})
    kit = build_kit(designs["g2"][0], layout, airfoil, _placed(designs, "g2", layout), settings)
    # Act
    by = {i.name: i for i in kit_mass_items(kit, settings)}
    # Assert: the area is the D-box material area (independently integrated in test_wing_stiffness)
    assert kit.spar.dbox and kit.dbox_area_m2 == pytest.approx(
        dbox_material_m2(kit.spar, airfoil, layout.wing.semi_span_m, layout.wing.chord_at, settings), rel=1e-9)
    dbox_kg = kit.dbox_area_m2 * settings.dbox_sheet_thickness_m * 200.0
    assert by["dbox_wing"].mass_kg == pytest.approx(dbox_kg, rel=1e-9)
    structure = [by[p.label] for p in kit.parts] + [by[t.label] for t in kit.tubes] + [by[s.label] for s in kit.linear]
    glue_base = sum(i.mass_kg for i in structure) + dbox_kg  # glued wood; film and fuselage cover are not
    assert by["glue_hardware"].mass_kg == pytest.approx(settings.glue_fraction * glue_base, rel=1e-9)


def test_dbox_strip_spans_the_leading_edge_to_the_web_on_both_panels(g2_kit, g2_kit_inputs):
    # Arrange
    layout, _, spar, settings = g2_kit_inputs
    cover = next(c for c in g2_kit.covering if c.kind == "dbox")
    wing, root = layout.wing, spar.joint_y_m
    c_root = wing.chord_at(root)
    web_x = wing.x_le_root_m + dbox_web_x_m(c_root, spar_member_at(spar, root).od_m, settings)
    # Act
    box = cover.surface.bounding_box()
    root_edge = [v for v in cover.surface.vertices() if abs(abs(v.Y) - root) < 1e-9]
    # Assert: both panels, from the joint to the tip, from the leading edge back to the root web
    assert cover.label == "dbox_wing"
    assert box.max.Y == pytest.approx(wing.semi_span_m, abs=1e-9) and box.min.Y == pytest.approx(-wing.semi_span_m, abs=1e-9)
    assert sorted(v.X for v in root_edge if v.Y > 0) == pytest.approx(
        [wing.x_le_root_m, web_x], abs=1e-9)
    assert cover.area_m2 > 2.0 * cover.surface.area  # the web and both faces of the sheet


def test_kit_without_dbox_has_no_dbox_covering(g2_kit_inputs, g2_kit):
    # Arrange: the g2 inputs with the D-box switched off, independent of any design's reversal margin
    layout, airfoil, spar, settings = g2_kit_inputs
    plain_spar = dataclasses.replace(spar, dbox=False)
    # Act
    covering = _covering(layout, airfoil, plain_spar, settings)
    plain = dataclasses.replace(g2_kit, spar=plain_spar, covering=covering)
    # Assert
    assert "dbox" not in {c.kind for c in covering} and plain.dbox_area_m2 == 0.0
    assert "dbox_wing" not in {i.name for i in kit_mass_items(plain, SETTINGS)}


# ------------------------------------------------------------------------------ V-tail pusher kit (Task 8) ---
VTAIL_WING_STATIONS: dict[str, float] = {"g1": 0.45, "g2": 0.35}  # the balanced stations the phase-2 probe found
ANGLE_TOL_DEG = 0.5
TOUCH_M = 1e-6  # glued faces: solids meet within this distance


@dataclasses.dataclass(frozen=True)
class VTailCase:
    sizing: object
    bom: object
    layout: object
    placed: tuple
    kit: Kit


def _vtail_placed(bom, layout) -> tuple:
    battery = _test_battery(layout).at_x(socket_axis_m(layout, 0.0)[0])
    return (components.payload(bom, layout), components.cruise_motor(bom, layout), battery,
            *components.vtol_motors(bom, layout))


# The G2 V-tail (21.5 kg, heavier than its 19.9 kg tractor) fails two *wing* rules with today's stock: the R31
# anti-rotation pin (300 N on the 5 mm rod with bosses) and, with a 6 mm rod, aileron reversal (34.5 vs 35.4 m/s with
# the 1.5 mm D-box). Its V-tail parts are exercised with that wing stock added (6 mm rod, 3 mm D-box sheet).
G2_WING_STOCK = SETTINGS.model_copy(update={"rods_m": (*SETTINGS.rods_m, 0.006), "dbox_sheet_thickness_m": 0.003})
VTAIL_SETTINGS = {"g1": SETTINGS, "g2": G2_WING_STOCK}


def _vtail_layout(design, case: str):
    return _fitted_layout(*design, VTAIL_WING_STATIONS[case], with_propeller=True)


def _vtail_case(design, case: str) -> VTailCase:
    sizing, bom = design
    layout = _vtail_layout(design, case)
    placed = _vtail_placed(bom, layout)
    kit = build_kit(sizing, layout, load_airfoil(layout.airfoil), placed, VTAIL_SETTINGS[case])
    return VTailCase(sizing, bom, layout, placed, kit)


@pytest.fixture(scope="module")
def vtail_kits(vtail_g1, vtail_g2) -> dict[str, VTailCase]:
    return {"g1": _vtail_case(vtail_g1, "g1"), "g2": _vtail_case(vtail_g2, "g2")}


def test_g2_vtail_kit_needs_more_wing_pin_stock_than_today(vtail_g2):
    """Documents today's limit (a wing rule, not the tail): the heavier G2 V-tail's root torque overloads the pin."""
    layout = _vtail_layout(vtail_g2, "g2")
    with pytest.raises(KitUnavailable, match="anti-rotation pin"):
        build_kit(vtail_g2[0], layout, load_airfoil(layout.airfoil), _vtail_placed(vtail_g2[1], layout), SETTINGS)


def _vtail_surface(layout):
    return next(t for t in layout.tail_surfaces if t.name == "vtail_pair")


def test_tractor_kit_unchanged_after_generalisation():
    """The tractor kit (parts, positions, labels, tubes) still matches its golden snapshot after the tail
    generalisation (the same gate as tests/test_layout_golden.py)."""
    from tests.golden import GOLDEN_DIR, _kit_text

    for case in ("g1", "g2"):
        assert _kit_text(case) == (GOLDEN_DIR / case / "kit.json").read_text(), case


@pytest.mark.parametrize("case", ["g1", "g2"])
def test_vtail_kit_has_two_panels_at_dihedral(case, vtail_kits):
    c = vtail_kits[case]
    spars = [t for t in c.kit.tubes if t.role == "tail_spar"]
    assert sorted(t.label for t in spars) == ["TS-VL", "TS-VR"]
    gamma = _vtail_surface(c.layout).dihedral_deg
    for t in spars:
        assert math.degrees(math.atan2(abs(t.axis[2]), abs(t.axis[1]))) == pytest.approx(gamma, abs=ANGLE_TOL_DEG)
    by = {t.label: t for t in spars}
    right, left = by["TS-VR"], by["TS-VL"]
    assert right.axis[1] > 0 > left.axis[1] and right.axis[2] > 0 and left.axis[2] > 0
    assert left.start_m == pytest.approx((right.start_m[0], -right.start_m[1], right.start_m[2]), abs=1e-9)
    assert left.tube == right.tube and left.length_m == pytest.approx(right.length_m, abs=1e-9)
    labels = {p.label for p in c.kit.parts}
    assert {"VR-1", "VR-01", "VL-01"} <= labels
    assert not labels & {"HR-01", "HL-01", "V-01", "TM-1"}


@pytest.mark.parametrize("case", ["g1", "g2"])
def test_vtail_ribs_stand_square_to_their_panel(case, vtail_kits):
    """Each panel's ribs lie in planes normal to its span direction (0, +-cos G, sin G); the left panel mirrors the
    right one."""
    c = vtail_kits[case]
    g = math.radians(_vtail_surface(c.layout).dihedral_deg)
    parts = {p.label: p for p in c.kit.parts}
    right = sorted(label for label in parts if re.fullmatch(r"VR-\d\d", label))
    assert right and right == sorted(label.replace("VL-", "VR-") for label in parts if re.fullmatch(r"VL-\d\d", label))
    for label in right:
        for name, side in ((label, 1.0), (label.replace("VR-", "VL-"), -1.0)):
            loc = parts[name].location
            normal = np.array(tuple((loc * bd.Location((0.0, 0.0, 1.0))).position - loc.position))
            assert abs(np.dot(normal, (0.0, side * math.cos(g), math.sin(g)))) == pytest.approx(1.0, abs=1e-9), name
        mirrored = part_solid(parts[label], SETTINGS).mirror(bd.Plane.XZ).bounding_box()
        box = part_solid(parts[label.replace("VR-", "VL-")], SETTINGS).bounding_box()
        assert tuple(mirrored.min) == pytest.approx(tuple(box.min), abs=1e-9)
        assert tuple(mirrored.max) == pytest.approx(tuple(box.max), abs=1e-9)


@pytest.mark.parametrize("case", ["g1", "g2"])
def test_vtail_root_block_rings_the_boom_and_carries_both_root_ribs(case, vtail_kits):
    """`VR-1` is a ply clamp threaded on the tail boom; each panel's root rib is glued to one of its two faces at
    +-G, and each panel spar starts at that face."""
    c = vtail_kits[case]
    parts, tubes = {p.label: p for p in c.kit.parts}, {t.label: t for t in c.kit.tubes}
    block = parts["VR-1"]
    boom = tubes["TB-1"]
    assert block.family == "clamp"
    assert hole_diameters(block.outline) == pytest.approx([boom.tube.od_m + SETTINGS.fit_clearance_m], abs=1e-9)
    solid = part_solid(block, SETTINGS)
    for rib, spar in (("VR-01", "TS-VR"), ("VL-01", "TS-VL")):
        assert solid.distance_to(part_solid(parts[rib], SETTINGS)) < TOUCH_M, rib
        assert solid.distance_to(tube_solid(tubes[spar])) < SETTINGS.sheet_for("rib").thickness_m, spar


@pytest.mark.parametrize("case", ["g1", "g2"])
def test_vtail_kit_parts_and_tubes_do_not_intersect(case, vtail_kits):
    c = vtail_kits[case]
    solids = [(p.label, part_solid(p, SETTINGS)) for p in c.kit.parts]
    solids += [(_mitre_name(t.label), tube_solid(t)) for t in c.kit.tubes]
    motor = next(p for p in c.placed if p.name == "cruise_motor")
    _assert_no_intersections([*solids, ("cruise_motor", motor.solid())])


def test_vtail_kit_parts_fit_the_bed(vtail_kits):
    for c in vtail_kits.values():
        for part in c.kit.parts:
            size = part.outline.bounding_box().size
            long, short = max(size.X, size.Y), min(size.X, size.Y)
            assert long <= USABLE_LONG_M + 1e-9 and short <= USABLE_SHORT_M + 1e-9, part.label


def test_vtail_edge_strips_film_and_mass(vtail_kits):
    c = vtail_kits["g1"]
    for kind in ("le", "te"):
        strips = sorted(s.label for s in c.kit.linear if s.kind == kind)
        assert strips == sorted(f"{kind.upper()}-{s}" for s in ("WR", "WL", "VR", "VL"))
    names = {i.name for i in kit_mass_items(c.kit, SETTINGS)}
    assert {"film_wing", "film_vtail_pair", "VR-1", "MP-B", "MC-B"} <= names
    assert not names & {"film_htail", "film_vtail"}


@pytest.mark.parametrize("case", ["g1", "g2"])
def test_boom_end_motor_plate_replaces_nose_plate(case, vtail_kits):
    labels = {p.label for p in vtail_kits[case].kit.parts}
    assert "MP-B" in labels and "MC-B" in labels and "MP-C" not in labels


@pytest.mark.parametrize("case", ["g1", "g2"])
def test_boom_end_motor_plate_sits_on_the_boom_end_ahead_of_the_motor(case, vtail_kits):
    """`MP-B` covers the boom's end face, on the boom axis; the `MC-B` ring clamps it on the boom just ahead."""
    c = vtail_kits[case]
    parts, boom = {p.label: p for p in c.kit.parts}, c.layout.tail_boom
    t = SETTINGS.sheet_for("plate").thickness_m
    plate, clamp = parts["MP-B"], parts["MC-B"]
    assert plate.location.position.X == pytest.approx(boom.x_end_m, abs=1e-9)
    assert clamp.location.position.X == pytest.approx(boom.x_end_m - SETTINGS.sheet_for("clamp").thickness_m,
                                                     abs=1e-9)
    for part in (plate, clamp):
        centre = part_solid(part, SETTINGS).bounding_box().center()
        assert (centre.Y, centre.Z) == pytest.approx((boom.y_m, boom.z_m), abs=1e-6), part.label
    od = next(tb for tb in c.kit.tubes if tb.label == "TB-1").tube.od_m
    assert hole_diameters(clamp.outline) == pytest.approx([od + SETTINGS.fit_clearance_m], abs=1e-9)
    motor = next(p for p in c.placed if p.name == "cruise_motor")
    assert motor.x_min >= boom.x_end_m + t - 1e-9


def test_boom_end_motor_reaching_back_into_its_plate_is_unavailable(vtail_kits):
    c = vtail_kits["g1"]
    motor = next(p for p in c.placed if p.name == "cruise_motor")
    forward = dataclasses.replace(motor, center_m=(c.layout.tail_boom.x_end_m, *motor.center_m[1:]))
    placed = tuple(forward if p is motor else p for p in c.placed)
    with pytest.raises(KitUnavailable, match="MP-B: the cruise motor reaches back into its mount plate"):
        build_kit(c.sizing, c.layout, load_airfoil(c.layout.airfoil), placed, SETTINGS)


def test_boom_end_pusher_notes_the_motor_wiring(vtail_kits, g1_kit):
    assert BOOM_MOTOR_WIRING_NOTE in vtail_kits["g1"].kit.notes
    assert g1_kit.notes == ()  # the tractor's build sheet is unchanged


def test_folding_prop_note_for_hand_launched_pusher(vtail_kits):
    c = vtail_kits["g1"]
    belly = prop_clearance(c.layout, GearGeometry(kind="none", contacts=(), leg_height_m=0.0), "hand")
    assert belly.needs_folding_prop
    text = build_sheet_text(with_clearance_notes(c.kit, belly.needs_folding_prop), SETTINGS, {})
    assert "Folding propeller required (belly landing)" in text
    assert BOOM_MOTOR_WIRING_NOTE in text
    assert FOLDING_PROP_NOTE not in build_sheet_text(with_clearance_notes(c.kit, False), SETTINGS, {})


def test_folding_prop_note_follows_the_flag_for_a_tractor_too(g1_kit):
    """Controller ruling (Task 6): any hand-launched aircraft whose tip is below the belly line folds its prop."""
    assert with_clearance_notes(g1_kit, True).notes == (FOLDING_PROP_NOTE,)
    assert with_clearance_notes(g1_kit, False) == g1_kit


@pytest.mark.parametrize("change", [{"booms": "twin"}, {"propeller": "pusher_pod"}, {"tail": "inverted_v"},
                                    {"propeller": "pusher_boom_end"}, {"tail": "v"}])
def test_unbuilt_layouts_still_get_no_kit(change, designs, g1_kit_inputs, monkeypatch):
    """The kit cuts exactly two (tail, propeller, booms) combinations, the tractor's and the V-tail pusher's; any
    other, even one mixing their parts (a V-tail tractor, a conventional boom-end pusher), gets a clean refusal naming
    the layout and its parts."""
    from src.tools.fabrication import build_kit as build_kit_module

    real = build_kit_module.layout_spec
    monkeypatch.setattr(build_kit_module, "layout_spec", lambda name: dataclasses.replace(real(name), **change))
    layout, airfoil = g1_kit_inputs[0], g1_kit_inputs[1]
    spec = dataclasses.replace(real("pod_boom_tractor"), **change)
    parts = f"({spec.tail} tail, {spec.propeller} propeller, {spec.booms} booms)"
    with pytest.raises(KitUnavailable, match=re.escape(f"no parts for the Pod-and-boom tractor layout yet {parts}")):
        build_kit(designs["g1"][0], layout, airfoil, (), SETTINGS)


def test_built_layout_parts_are_the_tractor_and_the_vtail_pusher():
    from src.tools.fabrication.build_kit import BUILT_LAYOUT_PARTS
    from src.tools.sizing.layouts import layout_spec

    specs = [layout_spec(name) for name in ("pod_boom_tractor", "single_boom_vtail_pusher")]
    assert BUILT_LAYOUT_PARTS == {(s.tail, s.propeller, s.booms) for s in specs}


def test_skin_governed_vtail_root_still_meets_its_root_block(vtail_g1):
    """Constructed case: at wing station 0.30 the V-tail root chord lies over the pod's tail cone, so the fuselage
    skin, not the boom web, sets the panels' root station. VR-1's faces move out with it: both root ribs are still
    glued to it and both spars still butt it (no disconnected joint)."""
    from src.tools.fabrication.wing_parts import panel_start_m, root_block_face_m

    sizing, bom = vtail_g1
    layout = _fitted_layout(sizing, bom, 0.30, with_propeller=True)
    surface = _vtail_surface(layout)
    assert panel_start_m(layout, surface, SETTINGS) > root_block_face_m(layout, SETTINGS) + 0.01  # the skin governs
    kit = build_kit(sizing, layout, load_airfoil(layout.airfoil), _vtail_placed(bom, layout), SETTINGS)
    parts, tubes = {p.label: p for p in kit.parts}, {t.label: t for t in kit.tubes}
    block = part_solid(parts["VR-1"], SETTINGS)
    for rib, spar in (("VR-01", "TS-VR"), ("VL-01", "TS-VL")):
        assert block.distance_to(part_solid(parts[rib], SETTINGS)) < TOUCH_M, rib
        assert block.distance_to(tube_solid(tubes[spar])) < SETTINGS.sheet_for("rib").thickness_m, spar


# ------------------------------------------------------------ V-tail root inside the tail cone (M5 ruling) ---
@pytest.mark.parametrize("case,station,in_pod", [("g1", 0.30, True), ("g1", 0.35, False), ("g2", 0.30, True),
                                                 ("g2", 0.35, False)])
def test_vtail_root_block_in_the_tail_cone_clears_every_part(case, station, in_pod, request):
    """Where the V-tail root chord lies over the pod's tail cone (the skin sets its root station), VR-1, the root ribs
    and the spars clear every former, the crutch and every other kit part. At 0.30 VR-1 itself lies inside the pod (a
    blended root) and the build sheet says to slot the tail-cone covering; at 0.35 it is already aft of the pod."""
    sizing, bom = request.getfixturevalue(f"vtail_{case}")
    layout = _fitted_layout(sizing, bom, station, with_propeller=True)
    placed = _vtail_placed(bom, layout)
    kit = build_kit(sizing, layout, load_airfoil(layout.airfoil), placed, VTAIL_SETTINGS[case])
    block = next(p for p in kit.parts if p.label == "VR-1")
    assert (block.location.position.X < layout.fuselage_length_m) == in_pod
    assert (TAIL_CONE_SLOT_NOTE in kit.notes) == in_pod
    solids = [(p.label, part_solid(p, SETTINGS)) for p in kit.parts]
    solids += [(_mitre_name(t.label), tube_solid(t)) for t in kit.tubes]
    motor = next(p for p in placed if p.name == "cruise_motor")
    _assert_no_intersections([*solids, ("cruise_motor", motor.solid())])


def test_no_tail_cone_slot_note_when_the_root_block_is_aft_of_the_pod(vtail_kits):
    c = vtail_kits["g1"]  # station 0.45: the V-tail root is behind the pod
    block = next(p for p in c.kit.parts if p.label == "VR-1")
    assert block.location.position.X > c.layout.fuselage_length_m
    assert TAIL_CONE_SLOT_NOTE not in c.kit.notes


def test_root_block_clashing_with_a_fuselage_part_refuses_the_kit(vtail_kits):
    """No relief exists for VR-1 in a former: a clash makes the kit unavailable, naming the part."""
    from src.tools.fabrication.build_kit import _check_root_block_clear

    block = next(p for p in vtail_kits["g1"].kit.parts if p.label == "VR-1")
    former = dataclasses.replace(block, label="F-9", family="former")
    with pytest.raises(KitUnavailable, match="VR-1: the V-tail root block inside the tail cone runs into F-9"):
        _check_root_block_clear((block,), (former,), SETTINGS)
