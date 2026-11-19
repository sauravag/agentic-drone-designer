"""Task 4: fuselage parts (crutch, half-formers, firewall, wing-box plates, boom mounts), located into the
aircraft frame per `kit.py`."""

import dataclasses
import itertools
import math

import build123d as bd
import pytest

from src.tools.aero_cad import components
from src.tools.aero_cad.components import COMPONENT_GAP_M, PlacedComponent
from src.tools.aero_cad.layout import TAIL_BOOM_START_FRACTION, fuselage_radius_at
from src.tools.fabrication.kit import KitPart, KitUnavailable
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
from src.tools.fabrication.wing_parts import PIN_CHORD_FRACTION, PIN_ROD_DIAMETER_M
from tests.cad_helpers import hole_centre, hole_diameters, part_solid, ribs_by_label

TOL_M = 1e-6
COVER_M = 0.0015
AREA_TOL_M2 = 1e-10
OBLIQUE_ALLOWANCE_M = 0.0005  # a few-degree lean through 3 mm ply


def _common_volume(a: bd.Solid, b: bd.Solid) -> float:
    common = a.intersect(b)
    if common is None:
        return 0.0
    return sum(shape.volume for shape in (common if isinstance(common, list) else [common]))


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


def _boom_mounts(parts) -> list[KitPart]:
    return sorted((p for p in parts if p.label.startswith("BM-")), key=lambda p: p.label)


def test_tail_boom_sits_in_two_boom_mount_bulkheads_in_the_tail_cone(g1_kit_inputs):
    """R16: BM-1 just aft of the boom root, BM-2 as far aft as a 2 mm rim around the boom hole allows."""
    layout, settings = g1_kit_inputs[0], g1_kit_inputs[3]
    parts = _fuselage(g1_kit_inputs, components=())
    boom, t = layout.tail_boom, settings.sheet_for("bulkhead").thickness_m
    hole = boom_tube(boom.diameter_m, settings).od_m + settings.fit_clearance_m
    bm1, bm2 = _boom_mounts(parts)
    assert (bm1.label, bm2.label) == ("BM-1", "BM-2") and bm1.family == bm2.family == "bulkhead"
    assert bm1.location.position.X == pytest.approx(boom.x_start_m + t, abs=1e-9)
    assert bm2.location.position.X - bm1.location.position.X >= 0.030
    for bm in (bm1, bm2):
        assert hole_diameters(bm.outline) == [pytest.approx(hole, abs=1e-9)]
        rim = bm.outline.inner_wires()[0].distance_to(bm.outline.outer_wire())
        assert rim >= MIN_WEB_M - TOL_M, bm.label
        aft_face = bm.location.position.X + t  # the tail cone narrows aft: the aft face bounds the disc
        assert max(bm.outline.bounding_box().size.X, bm.outline.bounding_box().size.Y) / 2 == pytest.approx(
            fuselage_radius_at(layout, aft_face) - COVER_M, abs=1e-6)
    aft = bm2.location.position.X + 1e-4  # a little further aft leaves less than the 2 mm rim
    assert fuselage_radius_at(layout, aft + t) - COVER_M - hole / 2 < MIN_WEB_M
    assert not any(p.label.startswith("F-") and _overlap_area(p.outline, _disc_face(hole / 2)) < AREA_TOL_M2
                   for p in parts)  # no sleeve holes left in the formers


def _disc_face(radius_m: float) -> bd.Face:
    return bd.Face(bd.Wire([bd.Edge.make_circle(radius_m)]))


def test_boom_mounts_closer_than_30mm_are_unavailable(g1_kit_inputs):
    layout, airfoil, spar, settings = g1_kit_inputs
    late = dataclasses.replace(layout.tail_boom, x_start_m=0.96 * layout.fuselage_length_m)
    with pytest.raises(KitUnavailable, match="BM-"):
        fuselage_parts(dataclasses.replace(layout, tail_boom=late), spar, (), settings)


def _real_components(designs, case: str, layout) -> tuple[PlacedComponent, ...]:
    bom = designs[case][1]
    battery = _test_battery(layout).at_x(socket_axis_m(layout, 0.0)[0])
    return components.payload(bom, layout), components.cruise_motor(bom), battery


@pytest.mark.parametrize("case", ["g1", "g2"])
def test_real_components_never_intersect_a_fuselage_part(case, designs, request):
    """R15: every component gets an opening or notch in each part it crosses, external ones included."""
    kit_inputs = request.getfixturevalue(f"{case}_kit_inputs")
    layout, settings = kit_inputs[0], kit_inputs[3]
    placed = _real_components(designs, case, layout)
    parts = _fuselage(kit_inputs, components=placed)
    for part in parts:
        solid = part_solid(part, settings)
        for c in placed:
            volume = _common_volume(solid, c.solid())
            assert volume < 1e-12, f"{part.label} intersects {c.name}: {volume:.3e} m^3"


def test_firewall_moves_ahead_of_the_foremost_component_in_the_pod(designs, g1_kit_inputs):
    layout, settings = g1_kit_inputs[0], g1_kit_inputs[3]
    payload = components.payload(designs["g1"][1], layout)
    parts = ribs_by_label(_fuselage(g1_kit_inputs, components=_real_components(designs, "g1", layout)))
    t = settings.sheet_for("bulkhead").thickness_m
    station = min(FIRST_FORMER_FRACTION * layout.fuselage_length_m, payload.x_min - COMPONENT_GAP_M - 2 * t)
    assert parts["FW-1a"].location.position.X == pytest.approx(station, abs=1e-9)
    assert parts["C-1"].location.position.X == pytest.approx(station + 2 * t, abs=1e-9)  # crutch from its aft face


def test_colliding_mandatory_stations_are_unavailable(g1_kit_inputs):
    """A wing so far forward that its socket former lands 5 mm behind the firewall."""
    layout, _, spar, settings = g1_kit_inputs
    fore = _socket_band(layout, spar)[0]
    shift = FIRST_FORMER_FRACTION * layout.fuselage_length_m + 0.005 - fore
    wing = dataclasses.replace(layout.wing, x_le_root_m=layout.wing.x_le_root_m + shift)
    with pytest.raises(KitUnavailable, match="collide"):
        former_stations(dataclasses.replace(layout, wing=wing), spar, settings)


def test_component_crossing_a_wing_box_plate_gets_an_opening(g1_kit_inputs):
    layout, _, spar, settings = g1_kit_inputs
    plate = ribs_by_label(_fuselage(g1_kit_inputs, components=()))["WM-R"]
    box = part_solid(plate, settings).bounding_box()
    socket, pin = _plate_holes(plate)
    x_mid, z_mid = 0.5 * (socket.X + pin.X), 0.5 * (box.min.Z + box.max.Z)
    lead = PlacedComponent("servo_lead", "box", 0.01, (0.01, 0.02, 0.004), (x_mid, 0.5 * (box.min.Y + box.max.Y), z_mid))
    parts = _fuselage(g1_kit_inputs, components=(lead,))
    cut = ribs_by_label(parts)["WM-R"]
    assert len(cut.outline.inner_wires()) == 3  # socket, pin and the opening
    assert _common_volume(part_solid(cut, settings), lead.solid()) < 1e-12


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


def _rod(start: bd.Vector, end: bd.Vector, diameter_m: float) -> bd.Solid:
    axis = end - start
    return bd.Solid.make_cylinder(diameter_m / 2, axis.length, bd.Plane(origin=start, z_dir=axis.normalized()))


def _pin_rods(layout, spar, parts, settings) -> list[tuple[str, bd.Solid]]:
    """The 4 mm anti-rotation pins on each wing's 70 % chord line, from the plate's inner face to the skin joint."""
    rods = []
    for plate in (p for p in parts if p.label.startswith("WM-")):
        box = part_solid(plate, settings).bounding_box()
        side = 1.0 if plate.label == "WM-R" else -1.0
        inner = min(abs(box.min.Y), abs(box.max.Y))
        ends = [bd.Vector(socket_axis_m(layout, y, PIN_CHORD_FRACTION)[0], side * y,
                          socket_axis_m(layout, y, PIN_CHORD_FRACTION)[1]) for y in (inner, spar.joint_y_m)]
        rods.append((f"pin {plate.label}", _rod(*ends, PIN_ROD_DIAMETER_M)))
    return rods


def _boom_solid(layout, settings) -> tuple[str, bd.Solid]:
    boom = layout.tail_boom
    od = boom_tube(boom.diameter_m, settings).od_m
    return "tail boom", _rod(bd.Vector(boom.x_start_m, boom.y_m, boom.z_m), bd.Vector(boom.x_end_m, boom.y_m, boom.z_m), od)


@pytest.mark.parametrize("inputs", ["g1_kit_inputs", "g2_kit_inputs"])
def test_no_two_fuselage_parts_or_the_socket_intersect(inputs, request):
    kit_inputs = request.getfixturevalue(inputs)
    layout, _, spar, settings = kit_inputs
    parts = _fuselage(kit_inputs)
    extras = _socket_tubes(layout, spar) + _pin_rods(layout, spar, parts, settings) + [_boom_solid(layout, settings)]
    _assert_no_intersections([(p.label, part_solid(p, settings)) for p in parts] + extras)
    for part in parts:
        holes = part.outline.inner_wires()
        for i, hole in enumerate(holes):
            assert hole.distance_to(part.outline.outer_wire()) >= MIN_WEB_M - TOL_M, part.label
            for other in holes[i + 1:]:
                assert hole.distance_to(other) >= MIN_WEB_M - TOL_M, part.label
