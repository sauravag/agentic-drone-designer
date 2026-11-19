"""Tasks 3-4: kit model, wing/tail rib outlines and fuselage parts (flat faces, metres, located into the aircraft frame)."""

import dataclasses
import itertools
import math

import build123d as bd
import pytest

from src.tools.aero_cad.airfoil_geometry import camber_at, load_airfoil
from src.tools.aero_cad.components import PlacedComponent
from src.tools.aero_cad.layout import (
    AILERON_ETA,
    TAIL_AIRFOIL_NAME,
    TAIL_BOOM_START_FRACTION,
    TAIL_CONTROL_ETA,
    fuselage_radius_at,
)
from src.tools.aero_cad.structure import RIB_PITCH_M
from src.tools.fabrication.kit import KitPart, KitUnavailable
from src.schemas.fabrication import TubeStock
from src.tools.fabrication.fuselage_parts import (
    FIRST_FORMER_FRACTION,
    FORMER_PITCH_M,
    STATION_MERGE_M,
    former_stations,
    fuselage_parts,
    socket_axis_m,
)
from src.tools.fabrication.rib_outline import rect_face
from src.tools.fabrication.tubes import MIN_WEB_M, boom_tube
from src.tools.fabrication.wing_parts import (
    AILERON_HINGE_FRACTION,
    PIN_CHORD_FRACTION,
    PIN_ROD_DIAMETER_M,
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


# ------------------------------------------------------------------------------- Task 4: fuselage ---

COVER_M = 0.0015
AREA_TOL_M2 = 1e-10
OBLIQUE_ALLOWANCE_M = 0.0005  # a few-degree lean through 3 mm ply


def _test_battery(layout, length_m: float = 0.10, width_frac: float = 0.5) -> PlacedComponent:
    """A battery on the axis just aft of the wing-mount plates (clear of the socket, pin and tail boom)."""
    wing = layout.wing
    x_min = wing.x_le_root_m + PIN_CHORD_FRACTION * wing.root_chord_m + 0.02
    size = (length_m, width_frac * layout.fuselage_diameter_m, 0.35 * layout.fuselage_diameter_m)
    return PlacedComponent("battery", "box", 0.5, size, (x_min + length_m / 2, 0.0, 0.0))


def _fuselage(inputs, components=None):
    layout, _, spar, settings = inputs
    placed = (_test_battery(layout),) if components is None else components
    return fuselage_parts(layout, spar, placed, settings)


@pytest.fixture(scope="module")
def fuselage(g1_kit_inputs):
    return _fuselage(g1_kit_inputs)


def _formers(parts) -> list[KitPart]:
    return [p for p in parts if p.label.startswith("F-")]


def _slab_x(part: KitPart, settings) -> tuple[float, float]:
    box = part_solid(part, settings).bounding_box()
    return box.min.X, box.max.X


def _yz_box(c: PlacedComponent) -> bd.Face:
    _, ly, lz = c.size_m
    return rect_face(c.center_m[1] - ly / 2, c.center_m[2] - lz / 2, c.center_m[1] + ly / 2, c.center_m[2] + lz / 2)


def _overlap_area(a: bd.Face, b: bd.Face) -> float:
    common = a.intersect(b)
    if common is None:
        return 0.0
    return sum(f.area for shape in (common if isinstance(common, list) else [common]) for f in shape.faces())


def _socket_band(layout, spar) -> tuple[float, float]:
    """x range the two socket tubes sweep across the fuselage, grown by OD/2 + 10 mm."""
    extra = spar.socket.od_m / 2 + 0.010
    xs = [socket_axis_m(layout, y)[0] for y in (0.0, spar.joint_y_m)]
    return min(xs) - extra, max(xs) + extra


def test_former_stations_follow_the_pitch_and_clear_the_socket(g1_kit_inputs):
    layout, _, spar, settings = g1_kit_inputs
    stations = former_stations(layout, spar, settings)
    fore, aft = _socket_band(layout, spar)
    pin = layout.wing.x_le_root_m + PIN_CHORD_FRACTION * layout.wing.root_chord_m + 0.010
    assert stations[0] == pytest.approx(FIRST_FORMER_FRACTION * layout.fuselage_length_m)
    assert stations[-1] <= TAIL_BOOM_START_FRACTION * layout.fuselage_length_m + 1e-12
    assert FORMER_PITCH_M == 0.12 and STATION_MERGE_M == 0.020
    for mandatory in (fore, aft, pin):
        assert any(abs(x - mandatory) < 1e-12 for x in stations), mandatory
    assert list(stations) == sorted(stations)
    assert min(b - a for a, b in zip(stations, stations[1:])) >= STATION_MERGE_M - 1e-12
    assert not any(fore + 1e-12 < x < aft - 1e-12 for x in stations)  # nothing between the socket stations


def test_formers_follow_the_oml_minus_covering(g1_kit_inputs, fuselage):
    layout = g1_kit_inputs[0]
    for label in ("F-03U", "F-03L"):
        f = ribs_by_label(fuselage)[label]
        box = f.outline.bounding_box()
        radius = max(box.max.Y, -box.min.Y)  # local Y is aircraft z
        assert radius == pytest.approx(fuselage_radius_at(layout, f.location.position.X) - COVER_M, abs=1e-5)
        assert f.family == "former"


def test_component_crossing_a_former_gets_an_opening(g1_kit_inputs, fuselage):
    layout, settings = g1_kit_inputs[0], g1_kit_inputs[3]
    battery = _test_battery(layout)
    crossed = [f for f in _formers(fuselage) if _slab_x(f, settings)[1] > battery.x_min and _slab_x(f, settings)[0] < battery.x_max]
    assert len(crossed) >= 2  # an upper and a lower half
    for f in crossed:
        assert _overlap_area(f.outline, _yz_box(battery)) < AREA_TOL_M2, f.label


def test_crutch_has_an_opening_under_each_component(g1_kit_inputs, fuselage):
    layout, settings = g1_kit_inputs[0], g1_kit_inputs[3]
    crutch = ribs_by_label(fuselage)["C-1"]
    battery = _test_battery(layout)
    _, ly, _ = battery.size_m
    origin = crutch.location.position  # the crutch is a flat plate: local XY is aircraft XY shifted
    footprint = rect_face(battery.x_min - origin.X, -ly / 2, battery.x_max - origin.X, ly / 2)
    assert crutch.family == "crutch"
    assert _overlap_area(crutch.outline, footprint) < AREA_TOL_M2
    solid = part_solid(crutch, settings).bounding_box()
    assert (solid.min.Z, solid.max.Z) == pytest.approx((-0.0015, 0.0015), abs=1e-9)  # centred on z = 0


def _plate_holes(plate: KitPart) -> list[bd.Vector]:
    """Hole centres in the aircraft frame (on the plate's local z = 0 face), fore to aft."""
    return sorted(((plate.location * bd.Location(tuple(hole_centre(w)))).position for w in plate.outline.inner_wires()),
                  key=lambda v: v.X)


def _mid_y(plate: KitPart, settings) -> float:
    box = part_solid(plate, settings).bounding_box()
    return 0.5 * (box.min.Y + box.max.Y)


def test_wing_mount_plates_carry_socket_and_pin_holes(g1_kit_inputs, fuselage):
    layout, _, spar, settings = g1_kit_inputs
    for label in ("WM-L", "WM-R"):
        plate = ribs_by_label(fuselage)[label]
        socket_d, pin_d = hole_diameters(plate.outline)  # fore to aft
        # OD + clearance, plus a small allowance for the rod crossing the plate at the dihedral / sweep angle
        nominal_socket, nominal_pin = spar.socket.od_m + settings.fit_clearance_m, PIN_ROD_DIAMETER_M + settings.fit_clearance_m
        assert nominal_socket < socket_d < nominal_socket + OBLIQUE_ALLOWANCE_M
        assert nominal_pin < pin_d < nominal_pin + OBLIQUE_ALLOWANCE_M
        box = part_solid(plate, settings).bounding_box()
        assert box.max.Y - box.min.Y == pytest.approx(0.003, abs=1e-9)
        assert max(abs(box.min.Y), abs(box.max.Y)) <= spar.joint_y_m  # the socket passes through the plate
    left, right = (part_solid(ribs_by_label(fuselage)[k], settings).bounding_box() for k in ("WM-L", "WM-R"))
    assert (left.min.Y, left.max.Y) == pytest.approx((-right.max.Y, -right.min.Y), abs=1e-9)


@pytest.mark.parametrize("inputs", ["g1_kit_inputs", "g2_kit_inputs"])
def test_side_plate_holes_follow_the_dihedral_spar_axis(inputs, request):
    """R14: each socket tube is coaxial with its own wing's spar, so the holes sit on that axis at the plate's y."""
    kit_inputs = request.getfixturevalue(inputs)
    layout, settings = kit_inputs[0], kit_inputs[3]
    wing = layout.wing
    for plate in (p for p in _fuselage(kit_inputs) if p.label.startswith("WM-")):
        y = _mid_y(plate, settings)
        socket, pin = _plate_holes(plate)
        assert (socket.X, socket.Z) == pytest.approx(socket_axis_m(layout, y), abs=1e-9), plate.label
        assert (pin.X, pin.Z) == pytest.approx(socket_axis_m(layout, y, PIN_CHORD_FRACTION), abs=1e-9), plate.label
        assert socket.Z > wing.z_root_m + abs(y) * math.tan(math.radians(wing.dihedral_deg)) - 1e-12


def test_wing_mount_plates_stay_inside_the_covering(g1_kit_inputs, fuselage):
    layout, settings = g1_kit_inputs[0], g1_kit_inputs[3]
    for label in ("WM-L", "WM-R"):
        box = part_solid(ribs_by_label(fuselage)[label], settings).bounding_box()
        r = min(fuselage_radius_at(layout, box.min.X), fuselage_radius_at(layout, box.max.X)) - COVER_M
        assert math.hypot(max(abs(box.min.Y), abs(box.max.Y)), box.max.Z) <= r


def test_wing_box_plates_hang_between_two_formers_above_the_axis(g1_kit_inputs, fuselage):
    """R12: plates run from the forward socket former to the pin former, tabbed into both, clear of the crutch."""
    layout, _, spar, settings = g1_kit_inputs
    stations = former_stations(layout, spar, settings)
    fore = _socket_band(layout, spar)[0]
    pin = layout.wing.x_le_root_m + PIN_CHORD_FRACTION * layout.wing.root_chord_m + 0.010
    t = settings.sheet_for("former").thickness_m
    by = ribs_by_label(fuselage)
    for label in ("WM-L", "WM-R"):
        plate = by[label]
        box = part_solid(plate, settings).bounding_box()
        assert (box.min.X, box.max.X) == pytest.approx((fore, pin + t), abs=1e-9)  # tabs fill both former slots
        socket_z = socket_axis_m(layout, _mid_y(plate, settings))[1]
        socket_d = hole_diameters(plate.outline)[0]  # a 2 mm web under the socket hole
        assert box.min.Z == pytest.approx(socket_z - socket_d / 2 - MIN_WEB_M, abs=1e-9)
        assert box.min.Z > 0.0015  # nothing reaches the crutch
    for x in (fore, pin):
        index = [s for s in stations[1:]].index(next(s for s in stations[1:] if abs(s - x) < 1e-12)) + 1
        slots = [w.bounding_box() for w in by[f"F-{index:02d}U"].outline.inner_wires() if len(w.edges()) == 4]
        assert len(slots) == 2 and all(b.size.X == pytest.approx(0.003, abs=1e-9) for b in slots)


@pytest.mark.parametrize("inputs", ["g1_kit_inputs", "g2_kit_inputs"])
def test_battery_on_the_axis_passes_under_the_wing_box(inputs, request):
    kit_inputs = request.getfixturevalue(inputs)
    layout, _, spar, settings = kit_inputs
    x_s = socket_axis_m(layout, 0.0)[0]
    battery = _test_battery(layout).at_x(x_s)
    parts = _fuselage(kit_inputs, components=(battery,))
    solids = [(p.label, part_solid(p, settings)) for p in parts] + _socket_tubes(layout, spar)
    _assert_no_intersections(solids + [("battery", battery.solid())])


def test_firewall_is_two_solid_layers(g1_kit_inputs, fuselage):
    layout, settings = g1_kit_inputs[0], g1_kit_inputs[3]
    by = ribs_by_label(fuselage)
    a, b = by["FW-1a"], by["FW-1b"]
    for wall in (a, b):
        assert wall.outline.inner_wires() == [] and wall.family == "bulkhead"
        radius = fuselage_radius_at(layout, a.location.position.X) - COVER_M
        assert wall.outline.area == pytest.approx(math.pi * radius**2, rel=1e-3)
    assert a.location.position.X == pytest.approx(FIRST_FORMER_FRACTION * layout.fuselage_length_m)
    assert _slab_x(a, settings)[1] == pytest.approx(_slab_x(b, settings)[0], abs=1e-9)  # laminated face to face


def test_slot_width_equals_mating_thickness(g1_kit_inputs, fuselage):
    crutch = ribs_by_label(fuselage)["C-1"]
    slots = [w.bounding_box() for w in crutch.outline.inner_wires() if len(w.edges()) == 4]
    slots = [b for b in slots if min(b.size.X, b.size.Y) < 0.005]
    half_formers = len(_formers(fuselage)) // 2
    assert len(slots) == 2 * half_formers  # two tabs per former station; the wing-box plates sit in formers
    for box in slots:
        assert min(box.size.X, box.size.Y) == pytest.approx(0.003, abs=1e-9)


def test_tail_boom_sleeves_in_the_last_two_formers(g1_kit_inputs):
    layout, settings = g1_kit_inputs[0], g1_kit_inputs[3]
    fuselage = _fuselage(g1_kit_inputs, components=())  # no battery opening to clear the axis by accident
    boom = layout.tail_boom
    tube = boom_tube(boom.diameter_m, settings)
    sleeve = bd.Face(bd.Wire([bd.Edge.make_circle(tube.od_m / 2)])).moved(bd.Location((boom.y_m, boom.z_m, 0)))
    formers = _formers(fuselage)
    last_two = sorted({f.location.position.X for f in formers})[-2:]
    for f in formers:
        cleared = _overlap_area(f.outline, sleeve) < AREA_TOL_M2
        assert cleared == (f.location.position.X in last_two), f.label
        if f.location.position.X in last_two:  # the sleeve is a half-circle notch on the flat edge
            arcs = [e for e in f.outline.outer_wire().edges() if e.geom_type == bd.GeomType.CIRCLE]
            assert any(e.radius == pytest.approx((tube.od_m + settings.fit_clearance_m) / 2, abs=1e-9) for e in arcs), f.label


def test_empty_components_give_only_lightening_holes(g1_kit_inputs):
    layout, _, spar, settings = g1_kit_inputs
    parts = _fuselage(g1_kit_inputs, components=())
    formers = _formers(parts)
    assert any(f.outline.inner_wires() for f in formers)
    for f in formers:
        for wire in f.outline.inner_wires():
            edges = wire.edges()
            if len(edges) == 1:
                assert edges[0].geom_type == bd.GeomType.CIRCLE, f.label
            else:  # the wing-box plate passing through or tabbed into this former
                assert len(edges) == 4 and wire.bounding_box().size.X == pytest.approx(0.003, abs=1e-9), f.label


def test_oversized_component_breaking_a_former_rim_is_unavailable(g1_kit_inputs):
    layout = g1_kit_inputs[0]
    fat = _test_battery(layout, width_frac=0.95)
    with pytest.raises(KitUnavailable, match=r"F-\d\d[UL].*battery"):
        _fuselage(g1_kit_inputs, components=(fat,))


def _assert_no_intersections(solids) -> None:
    for (la, a), (lb, b) in itertools.combinations(solids, 2):
        if la.startswith("socket") and lb.startswith("socket"):
            continue  # the two socket tubes meet in a mitre at the centreline by design (R14)
        ba, bb = a.bounding_box(), b.bounding_box()
        if any(lo_a >= hi_b or lo_b >= hi_a for lo_a, hi_a, lo_b, hi_b in zip(ba.min, ba.max, bb.min, bb.max)):
            continue
        volume = _common_volume(a, b)
        assert volume < 1e-12, f"{la} intersects {lb}: {volume:.3e} m^3"


def _socket_tubes(layout, spar) -> list[tuple[str, bd.Solid]]:
    """R14: one socket tube per wing, on that wing's spar axis from the centreline to the skin joint."""
    tubes = []
    for side in (1.0, -1.0):
        x0, z0 = socket_axis_m(layout, 0.0)
        x1, z1 = socket_axis_m(layout, spar.joint_y_m)
        axis = bd.Vector(x1 - x0, side * spar.joint_y_m, z1 - z0)
        plane = bd.Plane(origin=(x0, 0.0, z0), z_dir=axis.normalized())
        tubes.append((f"socket{side:+.0f}", bd.Solid.make_cylinder(spar.socket.od_m / 2, axis.length, plane)))
    return tubes


@pytest.mark.parametrize("inputs", ["g1_kit_inputs", "g2_kit_inputs"])
def test_no_two_fuselage_parts_or_the_socket_intersect(inputs, request):
    kit_inputs = request.getfixturevalue(inputs)
    layout, _, spar, settings = kit_inputs
    parts = _fuselage(kit_inputs)
    _assert_no_intersections([(p.label, part_solid(p, settings)) for p in parts] + _socket_tubes(layout, spar))
    for part in parts:
        holes = part.outline.inner_wires()
        for i, hole in enumerate(holes):
            assert hole.distance_to(part.outline.outer_wire()) >= MIN_WEB_M - TOL_M, part.label
            for other in holes[i + 1:]:
                assert hole.distance_to(other) >= MIN_WEB_M - TOL_M, part.label
