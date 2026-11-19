"""Task 3: kit model and wing/tail rib outlines (flat faces, metres, located into the aircraft frame)."""

import math

import build123d as bd
import pytest

from src.tools.aero_cad.airfoil_geometry import camber_at, load_airfoil
from src.tools.aero_cad.layout import AILERON_ETA, TAIL_CONTROL_ETA
from src.tools.aero_cad.structure import RIB_PITCH_M
from src.tools.fabrication.kit import KitPart, KitUnavailable
from src.tools.fabrication.tubes import MIN_WEB_M
from src.tools.fabrication.wing_parts import (
    AILERON_HINGE_FRACTION,
    PIN_CHORD_FRACTION,
    TAIL_AIRFOIL_NAME,
    TAIL_HINGE_FRACTION,
    tail_ribs,
    tail_spar_tube,
    wing_ribs,
)
from tests.cad_helpers import hole_centre, hole_diameters, ribs_by_label

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


def _x_range(face: bd.Face) -> tuple[float, float]:
    box = face.bounding_box()
    return box.min.X, box.max.X


def test_rib_hole_matches_the_tube_through_it(g1_kit_inputs, ribs):
    outer = ribs_by_label(ribs)["WR-05"]
    spar = g1_kit_inputs[2]
    tube = next(s.tube for s in spar.segments if s.y_start_m <= outer.location.position.Y < s.y_end_m)
    assert hole_diameters(outer.outline)[0] == pytest.approx(tube.od_m + 0.0002, abs=1e-6)  # outside insertion


def test_every_rib_hole_matches_its_segment(g1_kit_inputs, ribs):
    _, _, spar, settings = g1_kit_inputs
    for rib in _main_ribs(ribs, "WR-"):
        y = rib.location.position.Y
        if y < spar.joint_y_m + spar.insertion_m:
            continue
        tube = next(s.tube for s in spar.segments if s.y_start_m < y <= s.y_end_m)  # step on a rib: inboard member
        assert hole_diameters(rib.outline)[0] == pytest.approx(tube.od_m + settings.fit_clearance_m, abs=TOL_M)


def test_left_ribs_mirror_right(ribs):
    by = ribs_by_label(ribs)
    assert by["WL-03"].location.position.Y == pytest.approx(-by["WR-03"].location.position.Y)
    for right in ribs:
        left = by[right.label.replace("WR-", "WL-", 1)] if right.label.startswith("WR-") else None
        if left is not None and right.family != "plate":
            assert left.outline.area == pytest.approx(right.outline.area, rel=1e-9)
            assert left.location.position.X == pytest.approx(right.location.position.X)
            assert left.location.position.Z == pytest.approx(right.location.position.Z)


def test_rib_location_maps_chord_aft_and_thickness_up(g1_kit_inputs, ribs):
    layout = g1_kit_inputs[0]
    rib = ribs_by_label(ribs)["WR-02"]
    y = rib.location.position.Y
    le = rib.location * bd.Location((0, 0, 0))
    up = rib.location * bd.Location((0, 1, 0))
    aft = rib.location * bd.Location((1, 0, 0))
    assert tuple(le.position) == pytest.approx((layout.wing.x_le_root_m, y, layout.wing.z_at(y)))
    assert tuple(up.position - le.position) == pytest.approx((0, 0, 1), abs=1e-9)
    assert tuple(aft.position - le.position) == pytest.approx((1, 0, 0), abs=1e-9)


def test_aileron_ribs_split_at_75_percent(g1_kit_inputs, ribs):
    layout = g1_kit_inputs[0]
    by = ribs_by_label(ribs)
    split_count = 0
    for rib in _main_ribs(ribs, "WR-"):
        y = rib.location.position.Y
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
    chord = layout.wing.chord_at(root.location.position.Y)
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
        is_over = rib.location.position.Y < spar.joint_y_m + spar.insertion_m
        over += is_over
        assert (hole_diameters(rib.outline)[0] == pytest.approx(socket_hole, abs=TOL_M)) == is_over, rib.label
    assert over >= 1


def test_rib_count_follows_the_20cm_pitch(g1_kit_inputs, ribs):
    layout, _, spar, _ = g1_kit_inputs
    expected = math.floor((layout.wing.semi_span_m - spar.joint_y_m) / RIB_PITCH_M + 1e-9) + 1
    stations = [r.location.position.Y for r in _main_ribs(ribs, "WR-")]
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
    layout = g1_kit_inputs[0]
    by = ribs_by_label(tails)
    assert TAIL_HINGE_FRACTION == pytest.approx(0.65)
    assert _main_ribs(tails, "HR-") and _main_ribs(tails, "HL-") and _main_ribs(tails, "V-")
    for prefix, surface, axis in (("HR-", layout.htail, "Y"), ("V-", layout.vtail, "Z")):
        for rib in _main_ribs(tails, prefix):
            span = getattr(rib.location.position, axis) - (surface.z_root_m if axis == "Z" else 0.0)
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


def test_wiring_hole_runs_from_the_servo_bay_to_the_root(ribs):
    by = ribs_by_label(ribs)
    servo_y = by["WR-SV"].location.position.Y
    for rib in _main_ribs(ribs, "WR-"):
        wiring = [d for d in hole_diameters(rib.outline) if d == pytest.approx(0.008, abs=TOL_M)]
        assert len(wiring) == (1 if rib.location.position.Y < servo_y else 0), rib.label
