"""Task 3: kit model and wing/tail rib outlines (flat faces, metres, located into the aircraft frame)."""

import dataclasses
import itertools
import math

import build123d as bd
import pytest

from src.schemas.fabrication import TubeStock
from src.tools.aero_cad.airfoil_geometry import camber_at, load_airfoil
from src.tools.aero_cad.layout import AILERON_ETA, TAIL_AIRFOIL_NAME, TAIL_CONTROL_ETA
from src.tools.aero_cad.structure import RIB_PITCH_M
from src.tools.fabrication.kit import KitPart, KitUnavailable
from src.tools.fabrication.rib_outline import notched_profile
from src.tools.fabrication.skin import max_skin_radius_m
from src.tools.fabrication.tubes import MIN_WEB_M, SparPlan, SparSegment, plan_spar, spar_member_at
from src.tools.fabrication.wing_parts import (
    AILERON_HINGE_FRACTION,
    PIN_CHORD_FRACTION,
    TAIL_HINGE_FRACTION,
    line_lean,
    socket_axis_m,
    spar_depth_at,
    tail_ribs,
    tail_root_radius_m,
    tail_spar_tubes,
    wing_ribs,
)
from src.tools.fabrication.wing_stiffness import dbox_web_x_m
from tests.cad_helpers import hole_centre, hole_diameters, part_solid, ribs_by_label

TOL_M = 1e-6


@pytest.fixture(scope="module")
def ribs(g1_kit_inputs):
    return wing_ribs(*g1_kit_inputs)


@pytest.fixture(scope="module")
def g1_tail_spars(designs, g1_kit_inputs):
    layout, _, _, settings = g1_kit_inputs
    return tail_spar_tubes(designs["g1"][0], layout, load_airfoil(TAIL_AIRFOIL_NAME), settings)


@pytest.fixture(scope="module")
def tails(g1_kit_inputs, g1_tail_spars):
    layout, _, _, settings = g1_kit_inputs
    return tail_ribs(layout, load_airfoil(TAIL_AIRFOIL_NAME), g1_tail_spars, settings)


def _main_ribs(parts, prefix: str) -> list[KitPart]:
    """Whole ribs / fore parts (`WR-07`), not aileron parts (`WR-07A`) or plates (`WR-SV`)."""
    return [p for p in parts if p.label.startswith(prefix) and p.label[len(prefix):].isdigit()]


def _station(part: KitPart, settings) -> float:
    """Span station of a right-panel rib: slab centre, or the inboard face of a panel's root rib."""
    box = part_solid(part, settings).bounding_box()
    return box.min.Y if part.label.endswith("-01") else 0.5 * (box.min.Y + box.max.Y)


def _x_range(face: bd.Face) -> tuple[float, float]:
    box = face.bounding_box()
    return box.min.X, box.max.X


def _wing_hole_m(layout, od_m: float, thickness_m: float, clearance_m: float, fraction: float = 0.25) -> float:
    """R19: OD/cos(lean) + t tan(lean) + clearance for a rod on the wing's `fraction` line."""
    span = layout.wing.semi_span_m
    (x0, z0), (x1, z1) = socket_axis_m(layout, 0.0, fraction), socket_axis_m(layout, span, fraction)
    lean = math.hypot(x1 - x0, z1 - z0) / span
    return od_m * math.hypot(1.0, lean) + thickness_m * lean + clearance_m


def test_rib_hole_matches_the_tube_through_it(g1_kit_inputs, ribs):
    outer = ribs_by_label(ribs)["WR-05"]
    layout, spar, settings = g1_kit_inputs[0], g1_kit_inputs[2], g1_kit_inputs[3]
    y = _station(outer, settings)
    tube = next(s.tube for s in spar.segments if s.y_start_m < y <= s.y_end_m)  # inboard member on a step
    expected = _wing_hole_m(layout, tube.od_m, settings.sheet_for("rib").thickness_m, 0.0002)
    assert hole_diameters(outer.outline)[0] == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("inputs", ["g1_kit_inputs", "g2_kit_inputs"])
def test_every_wing_rib_hole_is_its_spar_member_plus_clearance(inputs, request):
    """The socket stays inside the fuselage, so no rib carries a socket hole (R11)."""
    kit_inputs = request.getfixturevalue(inputs)
    layout, _, spar, settings = kit_inputs
    ribs = wing_ribs(*kit_inputs)
    for rib in _main_ribs(ribs, "WR-"):
        y = _station(rib, settings)
        tube = next(s.tube for s in spar.segments if y <= s.y_end_m + 1e-12)  # step on a rib: inboard member
        expected = _wing_hole_m(layout, tube.od_m, settings.sheet_for(rib.family).thickness_m, settings.fit_clearance_m)
        assert hole_diameters(rib.outline)[0] == pytest.approx(expected, abs=TOL_M), rib.label


def _assert_mirrored(parts, right_prefix: str, left_prefix: str, settings) -> None:
    by = ribs_by_label(parts)
    rights = [p for p in parts if p.label.startswith(right_prefix)]
    assert rights
    for right in rights:
        left = by[right.label.replace(right_prefix, left_prefix, 1)]
        r, l = part_solid(right, settings).bounding_box(), part_solid(left, settings).bounding_box()
        assert (l.min.X, l.max.X, l.min.Z, l.max.Z) == pytest.approx((r.min.X, r.max.X, r.min.Z, r.max.Z), abs=1e-9)
        assert (l.min.Y, l.max.Y) == pytest.approx((-r.max.Y, -r.min.Y), abs=1e-9), right.label


def test_left_parts_are_exact_mirrors_of_the_right(g1_kit_inputs, ribs, tails):
    settings = g1_kit_inputs[3]
    _assert_mirrored(ribs, "WR-", "WL-", settings)
    _assert_mirrored(tails, "HR-", "HL-", settings)


def test_rib_slabs_are_centred_on_their_stations(g1_kit_inputs, ribs):
    layout, _, spar, settings = g1_kit_inputs
    root = part_solid(ribs_by_label(ribs)["WR-01"], settings).bounding_box()
    assert root.min.Y == pytest.approx(spar.joint_y_m, abs=1e-9)  # root doubler lies outboard of the skin joint
    assert root.max.Y - root.min.Y == pytest.approx(0.003, abs=1e-9)
    for k, rib in enumerate(_main_ribs(ribs, "WR-")[1:], start=1):
        box = part_solid(rib, settings).bounding_box()
        assert 0.5 * (box.min.Y + box.max.Y) == pytest.approx(spar.joint_y_m + k * RIB_PITCH_M, abs=1e-9)
        assert box.max.Y - box.min.Y == pytest.approx(0.0015, abs=1e-9)


def _common_volume(a: bd.Solid, b: bd.Solid) -> float:
    common = a.intersect(b)
    if common is None:
        return 0.0
    return sum(shape.volume for shape in (common if isinstance(common, list) else [common]))


def test_no_two_wing_or_tail_parts_intersect(g1_kit_inputs, ribs, tails):
    settings = g1_kit_inputs[3]
    solids = [(p.label, part_solid(p, settings)) for p in (*ribs, *tails)]
    for (la, a), (lb, b) in itertools.combinations(solids, 2):
        ba, bb = a.bounding_box(), b.bounding_box()
        if any(lo_a >= hi_b or lo_b >= hi_a for lo_a, hi_a, lo_b, hi_b in zip(ba.min, ba.max, bb.min, bb.max)):
            continue  # bounding boxes apart: no need for a boolean
        volume = _common_volume(a, b)
        assert volume < 1e-12, f"{la} intersects {lb}: {volume:.3e} m^3"


def test_oversized_spar_tube_breaking_the_web_is_unavailable(g1_kit_inputs):
    layout, airfoil, spar, settings = g1_kit_inputs
    root = dataclasses.replace(spar.segments[0], tube=TubeStock(od_m=0.016, id_m=0.014))  # inside the rib, web < 2 mm
    fat = dataclasses.replace(spar, segments=(root, *spar.segments[1:]))
    with pytest.raises(KitUnavailable, match="WR-01"):
        wing_ribs(layout, airfoil, fat, settings)


def test_rib_location_maps_chord_aft_and_thickness_up(g1_kit_inputs, ribs):
    layout, settings = g1_kit_inputs[0], g1_kit_inputs[3]
    rib = ribs_by_label(ribs)["WR-02"]
    y = _station(rib, settings)
    le = rib.location * bd.Location((0, 0, settings.sheet_for("rib").thickness_m / 2))  # mid-plane LE
    up = rib.location * bd.Location((0, 1, settings.sheet_for("rib").thickness_m / 2))
    aft = rib.location * bd.Location((1, 0, settings.sheet_for("rib").thickness_m / 2))
    assert tuple(le.position) == pytest.approx((layout.wing.x_le_root_m, y, layout.wing.z_at(y)))
    assert tuple(up.position - le.position) == pytest.approx((0, 0, 1), abs=1e-9)
    assert tuple(aft.position - le.position) == pytest.approx((1, 0, 0), abs=1e-9)


def test_aileron_ribs_split_at_75_percent(g1_kit_inputs, ribs):
    layout, settings = g1_kit_inputs[0], g1_kit_inputs[3]
    by = ribs_by_label(ribs)
    split_count = 0
    for rib in _main_ribs(ribs, "WR-"):
        y = _station(rib, settings)
        eta = y / layout.wing.semi_span_m
        aft = by.get(rib.label + "A")
        assert (aft is not None) == (AILERON_ETA[0] <= eta <= AILERON_ETA[1]), rib.label
        if aft is None:
            continue
        split_count += 1
        hinge = 0.75 * layout.wing.chord_at(y)
        assert _x_range(rib.outline)[1] == pytest.approx(hinge, abs=TOL_M)
        assert _x_range(aft.outline)[0] == pytest.approx(hinge, abs=TOL_M)
        assert tuple(aft.location.position) == pytest.approx(tuple(rib.location.position))
    assert split_count >= 1
    assert AILERON_HINGE_FRACTION == pytest.approx(0.75)


def _assert_minimum_webs(parts) -> None:
    for part in parts:
        if part.family not in ("rib", "doubler"):
            continue
        outer, holes = part.outline.outer_wire(), part.outline.inner_wires()
        for i, hole in enumerate(holes):
            assert hole.distance_to(outer) >= MIN_WEB_M - TOL_M, part.label
            for other in holes[i + 1:]:
                assert hole.distance_to(other) >= MIN_WEB_M - TOL_M, part.label


def test_every_rib_keeps_a_minimum_web(ribs, tails):
    _assert_minimum_webs([*ribs, *tails])


def test_g2_wing_ribs_keep_a_minimum_web(g2_kit_inputs):
    ribs = wing_ribs(*g2_kit_inputs)
    assert len(ribs) > 0
    _assert_minimum_webs(ribs)


def test_root_rib_is_a_3mm_doubler_with_pin_hole(g1_kit_inputs, ribs):
    layout, airfoil, _, settings = g1_kit_inputs
    for label in ("WR-01", "WL-01"):
        root = ribs_by_label(ribs)[label]
        assert root.family == "doubler"
        assert settings.sheet_for(root.family).thickness_m == pytest.approx(0.003)
    root = ribs_by_label(ribs)["WR-01"]
    chord = layout.wing.chord_at(_station(root, settings))
    pin = next(w for w in root.outline.inner_wires()
               if hole_centre(w).X == pytest.approx(PIN_CHORD_FRACTION * chord, abs=1e-3))  # R19: at mid-slab (exact check below)
    assert PIN_CHORD_FRACTION == pytest.approx(0.70)
    assert hole_centre(pin).Y == pytest.approx(camber_at(airfoil, 0.70) * chord, abs=1e-3)
    expected = _wing_hole_m(layout, 0.004, 0.003, settings.fit_clearance_m, PIN_CHORD_FRACTION)
    assert 2 * pin.edges()[0].radius == pytest.approx(expected, abs=TOL_M)
    assert all(p.family == "rib" for p in _main_ribs(ribs, "WR-")[1:])


def test_rib_count_follows_the_20cm_pitch(g1_kit_inputs, ribs):
    layout, _, spar, settings = g1_kit_inputs
    expected = math.floor((layout.wing.semi_span_m - spar.joint_y_m) / RIB_PITCH_M + 1e-9) + 1
    stations = [_station(r, settings) for r in _main_ribs(ribs, "WR-")]
    assert len(stations) == expected == len(_main_ribs(ribs, "WL-"))
    assert stations[0] == pytest.approx(spar.joint_y_m)
    assert [b - a for a, b in zip(stations, stations[1:])] == pytest.approx([RIB_PITCH_M] * (expected - 1))


def test_one_servo_plate_per_aileron(g1_kit_inputs, ribs):
    layout = g1_kit_inputs[0]
    by = ribs_by_label(ribs)
    for label, sign in (("WR-SV", 1.0), ("WL-SV", -1.0)):
        plate = by[label]
        assert plate.family == "plate"
        centre = plate.location * bd.Location(plate.outline.center())
        eta = sign * centre.position.Y / layout.wing.semi_span_m
        assert AILERON_ETA[0] <= eta <= AILERON_ETA[1]
    assert sum(p.label.endswith("-SV") for p in ribs) == 2


def test_tail_ribs_split_at_65_percent(g1_kit_inputs, tails):
    layout, settings = g1_kit_inputs[0], g1_kit_inputs[3]
    by = ribs_by_label(tails)
    assert TAIL_HINGE_FRACTION == pytest.approx(0.65)
    assert _main_ribs(tails, "HR-") and _main_ribs(tails, "HL-") and _main_ribs(tails, "V-")
    for prefix, surface, axis in (("HR-", layout.htail, "Y"), ("V-", layout.vtail, "Z")):
        for rib in _main_ribs(tails, prefix):
            span = rib.location.position.Z - surface.z_root_m if axis == "Z" else _station(rib, settings)
            hinge = 0.65 * surface.chord_at(span)
            eta = span / surface.semi_span_m
            aft = by.get(rib.label + "A")
            assert (aft is not None) == (TAIL_CONTROL_ETA[0] <= eta <= TAIL_CONTROL_ETA[1]), rib.label
            if aft is not None:
                assert _x_range(rib.outline)[1] == pytest.approx(hinge, abs=TOL_M)
                assert _x_range(aft.outline)[0] == pytest.approx(hinge, abs=TOL_M)


def test_each_tail_surface_root_rib_is_drilled_for_its_own_spar(g1_kit_inputs, g1_tail_spars, tails):
    settings = g1_kit_inputs[3]
    by = ribs_by_label(tails)
    for label, tube in (("HR-01", g1_tail_spars.htail), ("V-01", g1_tail_spars.fin)):
        hole = hole_diameters(by[label].outline)[0]
        assert tube.od_m + settings.fit_clearance_m <= hole <= tube.od_m + 0.001, label  # R19


def test_kit_part_area_is_the_outline_area(ribs):
    part = ribs[0]
    assert part.area_m2 == pytest.approx(part.outline.area)
    assert part.quantity == 1


def test_wiring_hole_runs_from_the_servo_bay_to_the_root(g1_kit_inputs, ribs):
    by = ribs_by_label(ribs)
    servo_y = by["WR-SV"].location.position.Y
    for rib in _main_ribs(ribs, "WR-"):
        wiring = [d for d in hole_diameters(rib.outline) if d == pytest.approx(0.008, abs=TOL_M)]
        assert len(wiring) == (1 if _station(rib, g1_kit_inputs[3]) < servo_y else 0), rib.label


def test_fin_sits_on_a_mount_saddle_on_the_boom_where_the_boom_is_the_limit(g1_kit_inputs, g1_tail_spars):
    """R9 / R20: a fin whose root chord lies aft of the fuselage sits on TM-1 on top of the boom."""
    layout, _, _, settings = g1_kit_inputs
    vt = dataclasses.replace(layout.vtail, x_le_root_m=layout.fuselage_length_m)
    aft = dataclasses.replace(layout, vtail=vt)
    by = ribs_by_label(tail_ribs(aft, load_airfoil(TAIL_AIRFOIL_NAME), g1_tail_spars, settings))
    r_boom = aft.tail_boom.z_m + aft.tail_boom.diameter_m / 2
    t_plate = settings.sheet_for("plate").thickness_m
    mount = part_solid(by["TM-1"], settings).bounding_box()
    assert (mount.min.Z, mount.max.Z) == pytest.approx((r_boom, r_boom + t_plate), abs=1e-9)
    assert (mount.min.X, mount.max.X) == pytest.approx((vt.x_le_root_m, vt.x_le_root_m + vt.root_chord_m), abs=1e-9)
    assert part_solid(by["V-01"], settings).bounding_box().min.Z == pytest.approx(r_boom + t_plate, abs=1e-9)


def test_tail_roots_start_outside_the_fuselage_skin(g1_kit_inputs, tails):
    """R20: on g1 the tail roots lie over the tail cone, so HR-01/HL-01 and V-01 start on the skin; no TM-1."""
    layout, _, _, settings = g1_kit_inputs
    by = ribs_by_label(tails)
    ht, vt, boom = layout.htail, layout.vtail, layout.tail_boom
    root_y = max(boom.diameter_m / 2, max_skin_radius_m(layout, ht.x_le_root_m, ht.x_le_root_m + ht.root_chord_m))
    assert root_y > boom.diameter_m / 2  # the skin is the limit on g1
    assert part_solid(by["HR-01"], settings).bounding_box().min.Y == pytest.approx(root_y, abs=1e-9)
    assert part_solid(by["HL-01"], settings).bounding_box().max.Y == pytest.approx(-root_y, abs=1e-9)
    skin = max_skin_radius_m(layout, vt.x_le_root_m, vt.x_le_root_m + vt.root_chord_m)
    fin_z = max(boom.z_m + boom.diameter_m / 2, skin)
    assert part_solid(by["V-01"], settings).bounding_box().min.Z == pytest.approx(fin_z, abs=1e-9)
    assert "TM-1" not in by


def _wing_lean(layout, fraction: float) -> float:
    span = layout.wing.semi_span_m
    (x0, z0), (x1, z1) = socket_axis_m(layout, 0.0, fraction), socket_axis_m(layout, span, fraction)
    return math.hypot(x1 - x0, z1 - z0) / span


def _hole_near(part: KitPart, x_m: float) -> bd.Wire:
    return min(part.outline.inner_wires(), key=lambda w: abs(hole_centre(w).X - x_m))


def test_rib_holes_follow_the_leaning_line_through_the_slab(g1_kit_inputs, ribs):
    """R19: a hole crossed by a leaning line sits where the line crosses the slab mid-plane and is
    OD/cos(lean) + t tan(lean) + clearance across."""
    layout, _, spar, settings = g1_kit_inputs
    by, wing, clr = ribs_by_label(ribs), layout.wing, settings.fit_clearance_m
    t_root, t_rib = settings.sheet_for("doubler").thickness_m, settings.sheet_for("rib").thickness_m
    joint, station = spar.joint_y_m, spar.joint_y_m + 2 * RIB_PITCH_M
    member = next(s.tube for s in spar.segments if station <= s.y_end_m + 1e-12)
    cases = (("WR-01", joint, joint + t_root / 2, 0.25, spar.segments[0].tube.od_m, t_root),
             ("WR-01", joint, joint + t_root / 2, PIN_CHORD_FRACTION, 0.004, t_root),
             ("WR-03", station, station, 0.25, member.od_m, t_rib))
    for label, y, mid, fraction, od, t in cases:
        x, z = socket_axis_m(layout, mid, fraction)
        lean = _wing_lean(layout, fraction)
        wire = _hole_near(by[label], x - wing.x_le_root_m)
        centre = hole_centre(wire)
        assert (centre.X, centre.Y) == pytest.approx((x - wing.x_le_root_m, z - wing.z_at(y)), abs=1e-7), label
        expected = od * math.hypot(1.0, lean) + t * lean + clr
        assert 2 * wire.edges()[0].radius == pytest.approx(expected, abs=1e-7), label


# ------------------------------------------------------------------------------ root doubler depth ---


def test_root_station_depth_uses_the_doubler_thickness(g1_kit_inputs):
    layout, airfoil, spar, settings = g1_kit_inputs
    wing, joint = layout.wing, spar.joint_y_m
    plain = spar_depth_at(wing, airfoil, settings, root_doubler_m=None)
    root = spar_depth_at(wing, airfoil, settings, root_doubler_m=joint)
    lean = line_lean(wing, airfoil, 0.25)
    extra = (settings.sheet_for("doubler").thickness_m - settings.sheet_for("rib").thickness_m) * lean
    assert root(joint) == pytest.approx(plain(joint) - extra / math.hypot(1.0, lean), abs=1e-12)
    assert root(joint + 0.05) == plain(joint + 0.05)


def test_a_root_tube_that_only_fits_a_thin_rib_is_never_planned(g1_kit_inputs):
    """A tube between the 1.5 mm and 3 mm depths breaks WR-01's web, so the plan must not pick it as the root."""
    layout, airfoil, spar, settings = g1_kit_inputs
    wing, joint = layout.wing, spar.joint_y_m
    plain = spar_depth_at(wing, airfoil, settings, root_doubler_m=None)(joint)
    root = spar_depth_at(wing, airfoil, settings, root_doubler_m=joint)(joint)
    od = 0.5 * (plain + root)
    tube = TubeStock(od_m=od, id_m=0.5 * od)
    rod = TubeStock(od_m=0.003, id_m=0.0)
    plan = SparPlan((SparSegment(tube, joint, joint + 0.001), SparSegment(rod, joint + 0.001, wing.semi_span_m)),
                    spar.socket, joint, joint, spar.centre_moment_nm, spar.pin)
    with pytest.raises(KitUnavailable, match="WR-01"):
        wing_ribs(layout, airfoil, plan, settings)
    only = settings.model_copy(update={"tubes": (tube,)})
    depth = spar_depth_at(wing, airfoil, only, root_doubler_m=joint)
    with pytest.raises(KitUnavailable, match=f"y = {joint:.3f}"):
        plan_spar(0.5, wing.semi_span_m, joint, depth, None, only)


def test_a_hole_at_exactly_the_minimum_web_keeps_it():
    """D9: keeps_web tolerates float round-off at an exact fit (a web short by 1e-12 m still passes)."""
    from src.tools.fabrication.rib_outline import Hole, keeps_web, rect_face

    face = rect_face(0.0, 0.0, 0.02, 0.01)
    exact = 2.0 * (0.005 - MIN_WEB_M) + 2e-12  # 1e-12 m over: the web is short by 1e-12 m
    assert keeps_web(face, Hole(0.01, 0.005, exact), [])
    assert not keeps_web(face, Hole(0.01, 0.005, exact + 2e-6), [])


STEP_PROBE_M = 0.001  # distance either side of the D-box step at which the rib thickness is probed


def _thickness_at(face: bd.Face, x_m: float) -> float:
    """Vertical extent of the (hole-free) outline along the line local x = x_m."""
    strip = face.intersect(bd.Face(bd.Wire.make_polygon(
        [(x_m, -1.0, 0.0), (x_m, 1.0, 0.0), (x_m + 1e-6, 1.0, 0.0), (x_m + 1e-6, -1.0, 0.0)], close=True)))
    box = bd.Compound(list(strip)).bounding_box()
    return box.max.Y - box.min.Y


def test_inset_rib_shrinks_only_ahead_of_the_web(g2_kit_inputs):
    # Arrange
    _, _, _, settings = g2_kit_inputs
    airfoil, chord, inset, end = load_airfoil("naca2412"), 0.25, 0.0015, 0.06
    plain = notched_profile(airfoil, chord, settings)
    # Act
    inset_face = notched_profile(airfoil, chord, settings, inset_m=inset, inset_end_x_m=end)
    # Assert: ahead of the web both faces shrink by about 2 t; aft of it the outline is identical
    assert _thickness_at(plain, 0.10 * chord) - _thickness_at(inset_face, 0.10 * chord) == pytest.approx(
        2.0 * inset, rel=0.1)
    assert _thickness_at(inset_face, 0.5 * chord) == pytest.approx(_thickness_at(plain, 0.5 * chord), abs=1e-5)
    assert plain.cut(inset_face).area > 0.0
    assert inset_face.cut(plain).area < 1e-9  # the inset outline lies inside the plain one


def test_every_g2_wing_rib_keeps_its_spar_web(g2_kit_inputs):
    # Arrange
    layout, _, spar, settings = g2_kit_inputs
    assert spar.dbox
    # Act
    ribs = wing_ribs(*g2_kit_inputs)
    # Assert: every spar hole (the first hole of WR-/WL- ribs) keeps MIN_WEB_M to the outline and other holes
    assert any(p.label.startswith("WL-") for p in ribs)
    _assert_minimum_webs([p for p in ribs if p.label[:3] in ("WR-", "WL-")])


def test_dbox_ribs_are_smaller_than_the_plain_ones(g2_kit_inputs):
    # Arrange
    layout, airfoil, spar, settings = g2_kit_inputs
    plain_spar = dataclasses.replace(spar, dbox=False)
    # Act
    boxed, plain = wing_ribs(*g2_kit_inputs), wing_ribs(layout, airfoil, plain_spar, settings)
    # Assert
    assert sum(p.area_m2 * p.quantity for p in boxed) < sum(p.area_m2 * p.quantity for p in plain)


def test_tail_ribs_are_not_inset(g2_kit_inputs, designs):
    # Arrange: a wing with a D-box; the tail must stay plain
    layout, _, spar, settings = g2_kit_inputs
    assert spar.dbox
    tail_airfoil = load_airfoil(TAIL_AIRFOIL_NAME)
    spars = tail_spar_tubes(designs["g2"][0], layout, tail_airfoil, settings)
    # Act
    root = next(p for p in tail_ribs(layout, tail_airfoil, spars, settings) if p.label == "HR-01")
    chord = layout.htail.chord_at(tail_root_radius_m(layout))
    plain = notched_profile(tail_airfoil, chord, settings)
    # Assert: an inset rib would be about 2 x the sheet thickness thinner near 0.10 c
    assert _thickness_at(root.outline, 0.10 * chord) == pytest.approx(_thickness_at(plain, 0.10 * chord), abs=1e-5)


def test_a_real_g2_wing_rib_steps_at_the_dbox_web(g2_kit_inputs):
    # Arrange: WR-02's own outline against the plain profile at its chord
    layout, airfoil, spar, settings = g2_kit_inputs
    rib = ribs_by_label(wing_ribs(*g2_kit_inputs))["WR-02"]
    y = rib.location.position.Y
    chord = layout.wing.chord_at(y)
    web_x = dbox_web_x_m(chord, spar_member_at(spar, y).od_m, settings)
    plain = notched_profile(airfoil, chord, settings)
    ahead, aft = web_x - STEP_PROBE_M, web_x + STEP_PROBE_M
    # Act / Assert: ahead of the step the rib is about 2 t thinner; just aft it is unchanged
    assert _thickness_at(plain, ahead) - _thickness_at(rib.outline, ahead) == pytest.approx(
        2.0 * settings.dbox_sheet_thickness_m, rel=0.15)
    assert _thickness_at(rib.outline, aft) == pytest.approx(_thickness_at(plain, aft), abs=1e-5)
