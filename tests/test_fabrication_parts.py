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
    assert hole_diameters(outer.outline)[0] == pytest.approx(tube.od_m + 0.0002, abs=1e-6)  # outside insertion


def test_every_rib_hole_matches_its_segment(g1_kit_inputs, ribs):
    _, _, spar, settings = g1_kit_inputs
    for rib in _main_ribs(ribs, "WR-"):
        y = _station(rib, settings)
        if y < spar.joint_y_m + spar.insertion_m:
            continue
        tube = next(s.tube for s in spar.segments if s.y_start_m < y <= s.y_end_m)  # step on a rib: inboard member
        assert hole_diameters(rib.outline)[0] == pytest.approx(tube.od_m + settings.fit_clearance_m, abs=TOL_M)


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


def test_oversized_socket_breaking_the_web_is_unavailable(g1_kit_inputs):
    layout, airfoil, spar, settings = g1_kit_inputs
    fat = dataclasses.replace(spar, socket=TubeStock(od_m=0.016, id_m=0.014))  # inside the rib, web < 2 mm
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


def test_every_rib_keeps_a_minimum_web(g1_kit_inputs, ribs, tails):
    _, _, spar, _ = g1_kit_inputs
    for part in [*ribs, *tails]:
        if part.family not in ("rib", "doubler"):
            continue
        outer, holes = part.outline.outer_wire(), part.outline.inner_wires()
        for i, hole in enumerate(holes):
            assert hole.distance_to(outer) >= MIN_WEB_M - TOL_M, part.label
            for other in holes[i + 1:]:
                assert hole.distance_to(other) >= MIN_WEB_M - TOL_M, part.label


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


def test_ribs_over_the_socket_have_socket_holes(g1_kit_inputs, ribs):
    _, _, spar, settings = g1_kit_inputs
    socket_hole = spar.socket.od_m + settings.fit_clearance_m
    over = 0
    for rib in _main_ribs(ribs, "WR-"):
        is_over = _station(rib, settings) < spar.joint_y_m + spar.insertion_m
        over += is_over
        assert (hole_diameters(rib.outline)[0] == pytest.approx(socket_hole, abs=TOL_M)) == is_over, rib.label
    assert over >= 1


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
