"""Task 3: kit model and wing/tail rib outlines (flat faces, metres, located into the aircraft frame)."""

import dataclasses
import itertools
import math

import build123d as bd
import pytest

from src.tools.aero_cad.airfoil_geometry import camber_at, load_airfoil
from src.tools.aero_cad.layout import AILERON_ETA, TAIL_AIRFOIL_NAME, TAIL_CONTROL_ETA
from src.tools.aero_cad.structure import RIB_PITCH_M
from src.tools.fabrication.kit import KitPart, KitUnavailable
from src.schemas.fabrication import TubeStock
from src.tools.fabrication.tubes import MIN_WEB_M
from src.tools.fabrication.wing_parts import (
    AILERON_HINGE_FRACTION,
    PIN_CHORD_FRACTION,
    TAIL_HINGE_FRACTION,
    tail_ribs,
    tail_spar_tube,
    wing_ribs,
)
from tests.cad_helpers import hole_centre, hole_diameters, part_solid, ribs_by_label

TOL_M = 1e-6


@pytest.fixture(scope="module")
def ribs(g1_kit_inputs):
    return wing_ribs(*g1_kit_inputs)


@pytest.fixture(scope="module")
def tails(g1_kit_inputs):
    layout, _, _, settings = g1_kit_inputs
    return tail_ribs(layout, load_airfoil(TAIL_AIRFOIL_NAME), settings)


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


def test_rib_hole_matches_the_tube_through_it(g1_kit_inputs, ribs):
    outer = ribs_by_label(ribs)["WR-05"]
    spar, settings = g1_kit_inputs[2], g1_kit_inputs[3]
    y = _station(outer, settings)
    tube = next(s.tube for s in spar.segments if s.y_start_m < y <= s.y_end_m)  # inboard member on a step
    assert hole_diameters(outer.outline)[0] == pytest.approx(tube.od_m + 0.0002, abs=1e-6)


@pytest.mark.parametrize("inputs", ["g1_kit_inputs", "g2_kit_inputs"])
def test_every_wing_rib_hole_is_its_spar_member_plus_clearance(inputs, request):
    """The socket stays inside the fuselage, so no rib carries a socket hole (R11)."""
    kit_inputs = request.getfixturevalue(inputs)
    _, _, spar, settings = kit_inputs
    ribs = wing_ribs(*kit_inputs)
    for rib in _main_ribs(ribs, "WR-"):
        y = _station(rib, settings)
        tube = next(s.tube for s in spar.segments if y <= s.y_end_m + 1e-12)  # step on a rib: inboard member
        assert hole_diameters(rib.outline)[0] == pytest.approx(tube.od_m + settings.fit_clearance_m, abs=TOL_M), rib.label


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
               if hole_centre(w).X == pytest.approx(PIN_CHORD_FRACTION * chord, abs=TOL_M))
    assert PIN_CHORD_FRACTION == pytest.approx(0.70)
    assert hole_centre(pin).Y == pytest.approx(camber_at(airfoil, 0.70) * chord, abs=TOL_M)
    assert 2 * pin.edges()[0].radius == pytest.approx(0.004 + settings.fit_clearance_m, abs=TOL_M)
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


def test_tail_spar_is_the_smallest_tube_that_fits(g1_kit_inputs, tails):
    layout, _, _, settings = g1_kit_inputs
    tube = tail_spar_tube(layout, load_airfoil(TAIL_AIRFOIL_NAME), settings)
    assert tube == settings.tubes[0]
    root = ribs_by_label(tails)["HR-01"]
    assert hole_diameters(root.outline)[0] == pytest.approx(tube.od_m + settings.fit_clearance_m, abs=TOL_M)
    assert ribs_by_label(tails)["TM-1"].family == "plate"


def test_tail_too_thin_for_any_tube_is_unavailable(g1_kit_inputs):
    layout, _, _, settings = g1_kit_inputs
    with pytest.raises(KitUnavailable, match="tail"):
        tail_spar_tube(layout, load_airfoil(TAIL_AIRFOIL_NAME), settings.model_copy(update={"tubes": (settings.tubes[-1],)}))


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


def test_fin_sits_on_a_mount_saddle_on_the_boom(g1_kit_inputs, tails):
    layout, _, _, settings = g1_kit_inputs
    by = ribs_by_label(tails)
    r_boom = layout.tail_boom.z_m + layout.tail_boom.diameter_m / 2
    t_plate = settings.sheet_for("plate").thickness_m
    mount = part_solid(by["TM-1"], settings).bounding_box()
    assert (mount.min.Z, mount.max.Z) == pytest.approx((r_boom, r_boom + t_plate), abs=1e-9)
    vt = layout.vtail
    assert (mount.min.X, mount.max.X) == pytest.approx((vt.x_le_root_m, vt.x_le_root_m + vt.root_chord_m), abs=1e-9)
    assert part_solid(by["V-01"], settings).bounding_box().min.Z == pytest.approx(r_boom + t_plate, abs=1e-9)


# ------------------------------------------------------------------ Task 5: fittings and bed splitting ---
from src.schemas.fabrication import DEFAULT_FABRICATION  # noqa: E402
from src.tools.aero_cad import components as cad_components  # noqa: E402
from src.tools.aero_cad.components import COMPONENT_GAP_M, PlacedComponent  # noqa: E402
from src.tools.fabrication.boom_parts import CLAMP_LABELS  # noqa: E402
from src.tools.fabrication.fittings import (  # noqa: E402
    DOVETAIL_DEPTH_M,
    DOVETAIL_FLARE_M,
    DOVETAIL_NECK_M,
    SPLIT_SEARCH_M,
    boom_parts,
    component_plates,
    cruise_motor_plate,
    join_notes,
    split_to_bed,
)
from src.tools.fabrication.fuselage_parts import firewall_station, fuselage_parts, socket_axis_m  # noqa: E402
from src.tools.fabrication.rib_outline import Hole, circle_face, rect_face  # noqa: E402
from src.tools.fabrication.tubes import FIT_EPS_M, boom_tube  # noqa: E402
import numpy as np  # noqa: E402
from src.tools.aero_cad.layout import fuselage_radius_at  # noqa: E402
from tests.test_fabrication_fuselage import (  # noqa: E402
    _assert_no_intersections,
    _boom_solid,
    _pin_rods,
    _real_components,
    _rod,
    _socket_tubes,
)

SPLIT_CLEAR_M = 0.003
USABLE_W_M = 0.590
USABLE_H_M = 0.390
MOTOR_SLOT_RADII_M = (0.008, 0.0125)  # 16 / 25 mm bolt circles bound the slots


def _plate(width_m: float, height_m: float, holes=(), label: str = "C-9", family: str = "crutch") -> KitPart:
    face = rect_face(0.0, 0.0, width_m, height_m)
    if holes:
        face = face.cut(*(circle_face(h) for h in holes)).faces()[0]
    return KitPart(label, family, face, bd.Location((0.1, 0.0, -0.0015)))


def _pieces(parts):
    return [p for p in parts if "-D" not in p.label]


def _doublers(parts):
    return [p for p in parts if "-D" in p.label]


def _fits_bed(face: bd.Face) -> bool:
    box = face.bounding_box()
    w, h = box.max.X - box.min.X, box.max.Y - box.min.Y
    return (w <= USABLE_W_M and h <= USABLE_H_M) or (h <= USABLE_W_M and w <= USABLE_H_M)


def test_dovetail_constants_are_the_brief_values():
    assert (SPLIT_SEARCH_M, DOVETAIL_NECK_M, DOVETAIL_FLARE_M, DOVETAIL_DEPTH_M) == (0.03, 0.010, 0.016, 0.012)


def test_part_that_fits_is_untouched():
    part = _plate(0.2, 0.1)
    assert split_to_bed(part, DEFAULT_FABRICATION) == (part,)


def test_long_crutch_splits_into_pieces_that_fit():
    parts = split_to_bed(_plate(1.4, 0.12), DEFAULT_FABRICATION)
    pieces = _pieces(parts)
    assert [p.label for p in pieces] == ["C-9a", "C-9b", "C-9c"]
    assert [p.label for p in _doublers(parts)] == ["C-9-D1", "C-9-D2"]
    assert all(_fits_bed(p.outline) for p in parts)
    assert all(p.family == "crutch" for p in parts)


def test_tall_part_splits_along_its_long_axis():
    pieces = _pieces(split_to_bed(_plate(0.12, 1.4), DEFAULT_FABRICATION))
    assert len(pieces) == 3 and all(_fits_bed(p.outline) for p in pieces)
    assert all(p.outline.bounding_box().max.Y - p.outline.bounding_box().min.Y < 0.6 for p in pieces)


def test_split_pieces_area_plus_dovetails_conserves_area():
    part = _plate(1.4, 0.12)
    pieces = _pieces(split_to_bed(part, DEFAULT_FABRICATION))
    assert sum(p.area_m2 for p in pieces) == pytest.approx(part.area_m2, abs=1e-9)
    assert all(p.location == part.location for p in pieces)  # pieces sit side by side as cut


def test_split_avoids_holes():
    holes = [Hole(1.4 / 3.0, 0.06, 0.01), Hole(2.8 / 3.0 + 0.012, 0.06, 0.01), Hole(0.7, 0.03, 0.008)]
    part = _plate(1.4, 0.12, holes)
    pieces = _pieces(split_to_bed(part, DEFAULT_FABRICATION))
    assert sum(len(p.outline.inner_wires()) for p in pieces) == len(holes)  # no hole was cut open
    for p in pieces:
        for hole in p.outline.inner_wires():
            assert hole.distance_to(p.outline.outer_wire()) >= SPLIT_CLEAR_M - TOL_M, p.label


def test_doubler_is_glued_on_one_face_across_the_joint():
    part = _plate(1.4, 0.12)
    parts = split_to_bed(part, DEFAULT_FABRICATION)
    pieces, doublers = _pieces(parts), _doublers(parts)
    t = DEFAULT_FABRICATION.sheet_for("crutch").thickness_m
    for k, d in enumerate(doublers):
        assert d.family == part.family
        assert d.location.position.Z == pytest.approx(part.location.position.Z + t, abs=1e-12)  # on the +z face
        for p in pieces[k:k + 2]:  # it laps over both pieces it joins
            common = p.outline.intersect(d.outline)
            assert sum(f.area for f in common.faces()) > 0.0
    solids = [(p.label, part_solid(p, DEFAULT_FABRICATION)) for p in parts]
    _assert_no_intersections(solids)


def test_part_wider_than_the_bed_is_unavailable():
    with pytest.raises(KitUnavailable):
        split_to_bed(_plate(0.5, 0.5), DEFAULT_FABRICATION)


def test_join_notes_name_both_pieces_and_the_station():
    parts = split_to_bed(_plate(1.4, 0.12), DEFAULT_FABRICATION)
    notes = join_notes(parts)
    assert len(notes) == 2
    assert notes[0].startswith("C-9a + C-9b") and "C-9-D1" in notes[0] and "x = " in notes[0]
    assert join_notes((_plate(0.2, 0.1),)) == ()


def test_part_with_no_clear_joint_is_unavailable():
    holes = [Hole(0.01 + 0.02 * k, 0.06, 0.008) for k in range(70)]  # a hole every 20 mm along the whole part
    with pytest.raises(KitUnavailable, match="C-9 cannot be split"):
        split_to_bed(_plate(1.4, 0.12, holes), DEFAULT_FABRICATION)


@pytest.mark.parametrize("case", ["g1", "g2"])
def test_split_crutch_stays_clear_of_every_fuselage_part(case, designs, request):
    """Both crutches exceed the 590 mm usable bed; g1's joint is hemmed in by the battery opening and a former slot,
    so its doubler gets the shorter lap."""
    layout, _, spar, settings = request.getfixturevalue(f"{case}_kit_inputs")
    parts = fuselage_parts(layout, spar, _real_components(designs, case, layout), settings)
    crutch = ribs_by_label(parts)["C-1"]
    split = split_to_bed(crutch, settings)
    assert len(_pieces(split)) >= 2
    assert sum(p.area_m2 for p in _pieces(split)) == pytest.approx(crutch.area_m2, abs=1e-9)
    others = [p for p in parts if p.label != "C-1"]
    placed = _real_components(designs, case, layout)
    trays = component_plates(placed, layout, settings, spar)
    extras = _socket_tubes(layout, spar) + _pin_rods(layout, spar, parts, settings) + [_boom_solid(layout, settings)]
    extras += [(c.name, c.solid()) for c in placed]
    _assert_no_intersections([(p.label, part_solid(p, settings)) for p in (*split, *others, *trays)] + extras)
    cover = settings.fuselage_cover_thickness_m
    for doubler in _doublers(split):  # its top corners stay inside the covering
        box = part_solid(doubler, settings).bounding_box()
        for x in (box.min.X, box.max.X):
            r = fuselage_radius_at(layout, x) - cover
            assert max(abs(box.min.Y), abs(box.max.Y)) <= math.sqrt(r * r - box.max.Z**2) + 1e-9, doubler.label


def _tail_vertices(face: bd.Face, tip_x: float, cut_x: float) -> list[tuple[float, float]]:
    return sorted((round(v.X, 9), round(v.Y, 9)) for v in face.outer_wire().vertices()
                  if abs(v.X - tip_x) < 1e-9 or (abs(v.X - cut_x) < 1e-9 and 0.0 < v.Y < 0.12))


def test_pieces_interlock_with_the_brief_dovetail():
    part = _plate(1.4, 0.12)
    a, b, *_ = _pieces(split_to_bed(part, DEFAULT_FABRICATION))
    tip = a.outline.bounding_box().max.X
    cut = tip - DOVETAIL_DEPTH_M
    assert abs(cut - 1.4 / 3.0) <= SPLIT_SEARCH_M + 1e-9
    tail = _tail_vertices(a.outline, tip, cut)
    neck = [y for x, y in tail if abs(x - cut) < 1e-9]
    flare = [y for x, y in tail if abs(x - tip) < 1e-9]
    assert len(neck) == 2 and len(flare) == 2
    assert neck[1] - neck[0] == pytest.approx(0.010, abs=1e-9)
    assert flare[1] - flare[0] == pytest.approx(0.016, abs=1e-9)
    assert sum(neck) / 2 == pytest.approx(sum(flare) / 2, abs=1e-9)  # symmetric tail
    socket = _tail_vertices(b.outline, tip, cut)
    flat = [c for v in tail for c in v]
    assert [c for v in socket for c in v] == pytest.approx(flat, abs=1e-9)  # b carries the same tail (no straight cut)


# ------------------------------------------------------------------------------------- component trays ---
def _avionics(layout) -> PlacedComponent:
    """An avionics box on the axis, just aft of the test battery."""
    x = socket_axis_m(layout, 0.0)[0] + 0.12
    return PlacedComponent("avionics_tray", "box", 0.3, (0.08, 0.06, 0.03), (x, 0.0, -0.01))


def test_one_tray_per_internal_component_directly_under_it(g1_kit_inputs, designs):
    layout, _, _, settings = g1_kit_inputs
    placed = (*_real_components(designs, "g1", layout), _avionics(layout))
    trays = ribs_by_label(component_plates(placed, layout, settings))
    internal = [c for c in placed if c.is_internal]
    assert sorted(trays) == sorted(f"P-{c.name}" for c in internal)
    t = settings.sheet_for("plate").thickness_m
    for c in internal:
        tray = trays[f"P-{c.name}"]
        box = part_solid(tray, settings).bounding_box()
        lx, ly, lz = c.size_m
        assert tray.family == "plate"
        assert box.max.Z == pytest.approx(c.center_m[2] - lz / 2, abs=1e-9)
        assert box.min.Z == pytest.approx(c.center_m[2] - lz / 2 - t, abs=1e-9)
        assert box.max.X - box.min.X == pytest.approx(lx + COMPONENT_GAP_M, abs=1e-9)
        assert box.max.Y - box.min.Y <= ly + COMPONENT_GAP_M + 1e-9


@pytest.mark.parametrize("case", ["g1", "g2"])
def test_trays_intersect_no_fuselage_part_tube_or_component(case, designs, request):
    layout, _, spar, settings = request.getfixturevalue(f"{case}_kit_inputs")
    placed = (*_real_components(designs, case, layout), _avionics(layout))
    parts = fuselage_parts(layout, spar, placed, settings)
    trays = component_plates(placed, layout, settings, spar)
    extras = _socket_tubes(layout, spar) + _pin_rods(layout, spar, parts, settings) + [_boom_solid(layout, settings)]
    extras += [(c.name, c.solid()) for c in placed]
    solids = [(p.label, part_solid(p, settings)) for p in (*parts, *trays)]
    for label, tray in ((p.label, part_solid(p, settings)) for p in trays):
        _assert_no_intersections([(label, tray)] + [s for s in solids if s[0] != label] + extras)


def test_tray_thicker_than_half_the_gap_is_unavailable(g1_kit_inputs, designs):
    layout, _, _, settings = g1_kit_inputs
    thick = [s.model_copy(update={"thickness_m": 0.006}) for s in settings.sheets]
    with pytest.raises(KitUnavailable, match="P-battery"):
        component_plates(_real_components(designs, "g1", layout), layout, settings.model_copy(update={"sheets": tuple(thick)}))


def test_tray_hitting_another_component_is_unavailable(g1_kit_inputs, designs):
    layout, _, _, settings = g1_kit_inputs
    battery = _real_components(designs, "g1", layout)[2]
    below = PlacedComponent("gps", "box", 0.05, (0.03, 0.03, 0.02),
                            (battery.center_m[0], 0.0, battery.center_m[2] - battery.size_m[2] / 2 - 0.011))
    with pytest.raises(KitUnavailable, match="P-battery.*gps"):
        component_plates((battery, below), layout, settings)


# ------------------------------------------------------------------------------------------ motor plates ---
def _slot_and_hole_counts(face: bd.Face) -> tuple[int, int]:
    wires = face.inner_wires()
    holes = [w for w in wires if len(w.edges()) == 1]
    return len(wires) - len(holes), len(holes)


def _assert_motor_pattern(plate: KitPart) -> None:
    slots, holes = _slot_and_hole_counts(plate.outline)
    assert (slots, holes) == (4, 1), plate.label
    centre = [w for w in plate.outline.inner_wires() if len(w.edges()) == 1][0]
    assert 2.0 * centre.edges()[0].radius == pytest.approx(0.003, abs=TOL_M)
    for w in plate.outline.inner_wires():
        if len(w.edges()) == 1:
            continue
        origin = hole_centre(centre)
        ends = [e.arc_center for e in w.edges() if e.geom_type == bd.GeomType.CIRCLE]  # bolt positions
        reach = sorted(round(math.hypot(p.X - origin.X, p.Y - origin.Y), 9) for p in ends)
        assert reach == pytest.approx(list(MOTOR_SLOT_RADII_M), abs=TOL_M)


def test_cruise_motor_plate_is_glued_to_the_firewall_front(g1_kit_inputs, designs):
    layout, _, spar, settings = g1_kit_inputs
    placed = _real_components(designs, "g1", layout)
    motor = next(c for c in placed if c.name == "cruise_motor")
    (plate,) = cruise_motor_plate(placed, layout, settings)
    assert plate.label == "MP-C" and plate.family == "plate"
    box = part_solid(plate, settings).bounding_box()
    station = firewall_station(layout, settings, placed)
    assert box.max.X == pytest.approx(station, abs=1e-9)  # aft face on the firewall's front face
    assert box.max.Y - box.min.Y >= motor.size_m[1] + 2 * MIN_WEB_M - 1e-9
    _assert_motor_pattern(plate)
    parts = fuselage_parts(layout, spar, placed, settings)
    _assert_no_intersections([(p.label, part_solid(p, settings)) for p in (*parts, plate)]
                             + [(c.name, c.solid()) for c in placed])


def test_no_cruise_motor_gives_no_motor_plate(g1_kit_inputs):
    layout, _, _, settings = g1_kit_inputs
    assert cruise_motor_plate((), layout, settings) == ()


# ----------------------------------------------------------------------------------- booms and clamps ---
def _lift_motors(designs, layout):
    return tuple(cad_components.vtol_motors(designs["g2"][1], layout))


@pytest.fixture(scope="module")
def g2_booms(g2_kit_inputs, designs):
    layout, _, spar, settings = g2_kit_inputs
    return boom_parts(layout, spar, settings, _lift_motors(designs, layout))


def test_fixed_wing_has_no_boom_clamps(g1_kit_inputs):
    layout, _, spar, settings = g1_kit_inputs
    parts, tubes = boom_parts(layout, spar, settings)
    assert not [p for p in parts if p.label.startswith(("BC-", "MP-"))]
    assert not [t for t in tubes if t.role == "lift_boom"]
    (tail,) = [t for t in tubes if t.role == "tail_boom"]
    assert tail.tube == boom_tube(layout.tail_boom.diameter_m, settings)
    assert tail.start_m == pytest.approx((layout.tail_boom.x_start_m, 0.0, layout.tail_boom.z_m))
    assert tail.length_m == pytest.approx(layout.tail_boom.length_m)


def test_boom_tube_is_the_r13_stock_tube(g2_kit_inputs, g2_booms):
    layout, _, _, settings = g2_kit_inputs
    _, tubes = g2_booms
    lift = [t for t in tubes if t.role == "lift_boom"]
    assert len(lift) == 2
    for tube, boom in zip(sorted(lift, key=lambda t: -t.start_m[1]), layout.vtol_booms):
        assert tube.tube in settings.tubes
        assert tube.tube == boom_tube(boom.diameter_m, settings)
        assert tube.start_m[:2] == pytest.approx((boom.x_start_m, boom.y_m))
        assert tube.axis == (1.0, 0.0, 0.0) and tube.length_m == pytest.approx(boom.length_m)


def _clamp_hole_centre(plate: KitPart) -> bd.Vector:
    (wire,) = plate.outline.inner_wires()
    return (plate.location * bd.Location(tuple(hole_centre(wire)))).position


def test_boom_clamps_carry_the_boom_hole_on_its_axis(g2_kit_inputs, g2_booms):
    _, _, _, settings = g2_kit_inputs
    parts, tubes = g2_booms
    by = ribs_by_label(parts)
    assert set(CLAMP_LABELS) <= set(by)
    for label in CLAMP_LABELS:
        plate = by[label]
        tube = next(t for t in tubes if t.role == "lift_boom" and (t.start_m[1] > 0) == ("R" in label))
        (diameter,) = hole_diameters(plate.outline)
        assert diameter == pytest.approx(tube.tube.od_m + settings.fit_clearance_m, abs=TOL_M)
        centre = _clamp_hole_centre(plate)
        assert (centre.Y, centre.Z) == pytest.approx(tube.start_m[1:], abs=1e-9)
        assert plate.family == "clamp"


def _spar_rod(layout, spar, side: float, y_m: float):
    seg = next(s for s in spar.segments if s.y_start_m <= y_m <= s.y_end_m)
    ends = [bd.Vector(socket_axis_m(layout, y)[0], side * y, socket_axis_m(layout, y)[1]) for y in (seg.y_start_m, seg.y_end_m)]
    return _rod(*ends, seg.tube.od_m)


def _tube_solid(tube):
    start = bd.Vector(*tube.start_m)
    return _rod(start, start + bd.Vector(*tube.axis) * tube.length_m, tube.tube.od_m)


def test_boom_assembly_intersects_nothing(g2_kit_inputs, g2_booms, designs):
    layout, airfoil, spar, settings = g2_kit_inputs
    parts, tubes = g2_booms
    y_boom = abs(layout.vtol_booms[0].y_m)
    solids = [(p.label, part_solid(p, settings)) for p in (*parts, *wing_ribs(layout, airfoil, spar, settings))]
    solids += [(t.label, _tube_solid(t)) for t in tubes if t.role == "lift_boom"]
    solids += [(f"spar{side:+.0f}", _spar_rod(layout, spar, side, y_boom)) for side in (1.0, -1.0)]
    solids += [(c.name, c.solid()) for c in _lift_motors(designs, layout)]
    _assert_no_intersections(solids)
    for part in parts:
        holes = part.outline.inner_wires()
        for i, hole in enumerate(holes):
            assert hole.distance_to(part.outline.outer_wire()) >= MIN_WEB_M - TOL_M, part.label
            for other in holes[i + 1:]:
                assert hole.distance_to(other) >= MIN_WEB_M - TOL_M, part.label


def test_clamp_saddles_sit_close_under_the_spar(g2_kit_inputs, g2_booms):
    """The saddle edge stays within the web of the spar member's underside (it really bears on the spar)."""
    layout, _, spar, settings = g2_kit_inputs
    parts, _ = g2_booms
    y_boom = abs(layout.vtol_booms[0].y_m)
    rod = _spar_rod(layout, spar, 1.0, y_boom)
    for plate in (p for p in parts if p.label.startswith("BC-R")):
        gap = part_solid(plate, settings).distance_to(rod)
        assert gap <= settings.fit_clearance_m + TOL_M, plate.label


def test_motor_plates_sit_on_the_boom_under_each_rotor_hub(g2_kit_inputs, g2_booms):
    layout, _, _, settings = g2_kit_inputs
    parts, tubes = g2_booms
    by = ribs_by_label(parts)
    for k, (x, y, z) in enumerate(layout.rotor_positions, start=1):
        plate = by[f"MP-{k}"]
        _assert_motor_pattern(plate)
        centre = (plate.location * bd.Location(tuple(hole_centre(
            next(w for w in plate.outline.inner_wires() if len(w.edges()) == 1))))).position
        assert (centre.X, centre.Y) == pytest.approx((x, y), abs=1e-9)
        tube = next(t for t in tubes if t.role == "lift_boom" and (t.start_m[1] > 0) == (y > 0))
        box = part_solid(plate, settings).bounding_box()
        assert box.min.Z == pytest.approx(tube.start_m[2] + tube.tube.od_m / 2, abs=1e-9)
        assert box.max.Z <= z + 1e-9  # under the motor


def _lowered_quadplane(layout, drop_m: float):
    """g2 with its lift booms and rotor hubs moved down, so the spar no longer forces the booms lower."""
    booms = tuple(dataclasses.replace(b, z_m=b.z_m - drop_m) for b in layout.vtol_booms)
    rotors = tuple((x, y, z - drop_m) for x, y, z in layout.rotor_positions)
    return dataclasses.replace(layout, vtol_booms=booms, rotor_positions=rotors)


@pytest.mark.parametrize("drop_m", [round(float(d), 6) for d in np.linspace(0.0045, 0.054, 25)])
def test_lift_boom_keeps_its_layout_height_when_the_spar_allows(drop_m, g2_kit_inputs, g2_booms, designs):
    """Every drop builds. The boom sits one plate under the hub unless the spar forces it lower (g2's forced height,
    unchanged by the drop; it stops forcing at about a 5 mm drop); then the motor plates carry the motors exactly."""
    layout, _, spar, settings = g2_kit_inputs
    forced_z = next(t_ for t_ in g2_booms[1] if t_.role == "lift_boom").start_m[2]
    low = _lowered_quadplane(layout, drop_m)
    parts, tubes = boom_parts(low, spar, settings, _lift_motors(designs, low))
    t = settings.sheet_for("plate").thickness_m
    od = boom_tube(low.vtol_booms[0].diameter_m, settings).od_m
    under_hub = low.vtol_booms[0].z_m + low.vtol_booms[0].diameter_m / 2 - t - od / 2
    for tube in (t_ for t_ in tubes if t_.role == "lift_boom"):
        assert tube.start_m[2] == pytest.approx(min(under_hub, forced_z), abs=1e-9)
    by = ribs_by_label(parts)
    for k, (_, _, z_hub) in enumerate(low.rotor_positions, start=1):
        top = part_solid(by[f"MP-{k}"], settings).bounding_box().max.Z
        assert top <= z_hub + FIT_EPS_M
        if under_hub <= forced_z:  # the spar no longer forces the boom: plates carry the motors exactly
            assert top == pytest.approx(z_hub, abs=FIT_EPS_M)


def test_g2_motor_plates_carry_their_motors(g2_kit_inputs, g2_booms):
    """Even where the spar forces the boom down, a plate never sits above its hub."""
    layout, _, _, settings = g2_kit_inputs
    by = ribs_by_label(g2_booms[0])
    for k, (_, _, z_hub) in enumerate(layout.rotor_positions, start=1):
        assert part_solid(by[f"MP-{k}"], settings).bounding_box().max.Z <= z_hub + 1e-9


def test_one_sided_boom_set_is_unavailable(g2_kit_inputs):
    layout, _, spar, settings = g2_kit_inputs
    one = dataclasses.replace(layout, vtol_booms=layout.vtol_booms[:1])
    with pytest.raises(KitUnavailable, match="MP-"):
        boom_parts(one, spar, settings)


def test_join_notes_ignore_labels_that_merely_contain_d():
    part = _plate(1.4, 0.12, label="X-Dummy")
    notes = join_notes(split_to_bed(part, DEFAULT_FABRICATION))
    assert len(notes) == 2 and notes[0].startswith("X-Dummya + X-Dummyb")


def test_narrow_neck_far_from_the_joint_still_splits():
    """A 3 mm neck near one end must not stop the doubler at the joint (only the band region is inset)."""
    face = rect_face(0.0, 0.0, 1.4, 0.12).cut(rect_face(0.05, 0.003, 0.07, 0.2)).faces()[0]
    part = KitPart("C-9", "crutch", face, bd.Location((0.1, 0.0, -0.0015)))
    parts = split_to_bed(part, DEFAULT_FABRICATION)
    assert len(_pieces(parts)) == 3 and len(_doublers(parts)) == 2
