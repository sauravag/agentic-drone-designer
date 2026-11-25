"""Task 5: fittings (component trays, the cruise-motor plate, boom clamps and lift-motor plates) and splitting parts
longer than the bed (dovetail pieces and doublers)."""

import dataclasses
import logging
import math
import re

import build123d as bd
import numpy as np
import pytest

from src.schemas.fabrication import DEFAULT_FABRICATION
from src.tools.aero_cad import components as cad_components
from src.tools.aero_cad.components import COMPONENT_GAP_M, PlacedComponent
from src.tools.aero_cad.layout import fuselage_radius_at
from src.tools.fabrication.boom_parts import CLAMP_LABELS
from src.tools.fabrication.fittings import (
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
from src.tools.fabrication.fuselage_parts import firewall_station, fuselage_parts
from src.tools.fabrication.kit import KitPart, KitUnavailable
from src.tools.fabrication.rib_outline import Hole, circle_face, rect_face
from src.tools.fabrication.tubes import FIT_EPS_M, MIN_WEB_M, NO_PLATE_LOADS, boom_tube
from src.tools.fabrication.build_kit import stiffened_spar_plan
from src.tools.fabrication.wing_parts import (MIN_RIB_SPACING_M, RIB_CLAMP_MARGIN_M, RibKeepOut, _servo_bay_y,
                                              clear_rib_stations, panel_rib_stations, socket_axis_m,
                                              wing_rib_stations, wing_ribs)
from tests.cad_helpers import hole_centre, hole_diameters, part_solid, ribs_by_label
from tests.test_fabrication_fuselage import (
    _assert_no_intersections,
    _boom_solid,
    _pin_rods,
    _real_components,
    _rod,
    _socket_tubes,
)

TOL_M = 1e-6
SPLIT_CLEAR_M = 0.003
USABLE_W_M, USABLE_H_M = DEFAULT_FABRICATION.usable_bed_m
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
    assert (SPLIT_SEARCH_M, DOVETAIL_NECK_M, DOVETAIL_FLARE_M, DOVETAIL_DEPTH_M) == (0.06, 0.010, 0.016, 0.012)


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
def test_split_crutch_stays_clear_of_every_fuselage_part(case, kit_designs, request):
    """Both crutches exceed the 590 mm usable bed; g1's joint is hemmed in by the battery opening and a former slot,
    so its doubler gets the shorter lap."""
    layout, _, spar, settings = request.getfixturevalue(f"{case}_kit_inputs")
    parts = fuselage_parts(layout, spar, _real_components(kit_designs, case, layout), settings, loads=NO_PLATE_LOADS)
    crutch = ribs_by_label(parts)["C-1"]
    split = split_to_bed(crutch, settings)
    assert len(_pieces(split)) >= 2
    assert sum(p.area_m2 for p in _pieces(split)) == pytest.approx(crutch.area_m2, abs=1e-9)
    others = [p for p in parts if p.label != "C-1"]
    placed = _real_components(kit_designs, case, layout)
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


def test_one_tray_per_internal_component_directly_under_it(g1_kit_inputs, kit_designs):
    layout, _, _, settings = g1_kit_inputs
    placed = (*_real_components(kit_designs, "g1", layout), _avionics(layout))
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
def test_trays_intersect_no_fuselage_part_tube_or_component(case, kit_designs, request):
    layout, _, spar, settings = request.getfixturevalue(f"{case}_kit_inputs")
    placed = (*_real_components(kit_designs, case, layout), _avionics(layout))
    parts = fuselage_parts(layout, spar, placed, settings, loads=NO_PLATE_LOADS)
    trays = component_plates(placed, layout, settings, spar)
    extras = _socket_tubes(layout, spar) + _pin_rods(layout, spar, parts, settings) + [_boom_solid(layout, settings)]
    extras += [(c.name, c.solid()) for c in placed]
    solids = [(p.label, part_solid(p, settings)) for p in (*parts, *trays)]
    for label, tray in ((p.label, part_solid(p, settings)) for p in trays):
        _assert_no_intersections([(label, tray)] + [s for s in solids if s[0] != label] + extras)


def test_tray_thicker_than_half_the_gap_is_unavailable(g1_kit_inputs, kit_designs):
    layout, _, _, settings = g1_kit_inputs
    thick = [s.model_copy(update={"thickness_m": 0.006}) for s in settings.sheets]
    with pytest.raises(KitUnavailable, match="P-battery"):
        component_plates(_real_components(kit_designs, "g1", layout), layout,
                         settings.model_copy(update={"sheets": tuple(thick)}))


def test_tray_hitting_another_component_is_unavailable(g1_kit_inputs, kit_designs):
    layout, _, _, settings = g1_kit_inputs
    battery = _real_components(kit_designs, "g1", layout)[2]
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


def test_cruise_motor_plate_is_glued_to_the_firewall_front(g1_kit_inputs, kit_designs):
    layout, _, spar, settings = g1_kit_inputs
    placed = _real_components(kit_designs, "g1", layout)
    motor = next(c for c in placed if c.name == "cruise_motor")
    (plate,) = cruise_motor_plate(placed, layout, settings)
    assert plate.label == "MP-C" and plate.family == "plate"
    box = part_solid(plate, settings).bounding_box()
    station = firewall_station(layout, settings, placed)
    assert box.max.X == pytest.approx(station, abs=1e-9)  # aft face on the firewall's front face
    assert box.max.Y - box.min.Y >= motor.size_m[1] + 2 * MIN_WEB_M - 1e-9
    _assert_motor_pattern(plate)
    parts = fuselage_parts(layout, spar, placed, settings, loads=NO_PLATE_LOADS)
    _assert_no_intersections([(p.label, part_solid(p, settings)) for p in (*parts, plate)]
                             + [(c.name, c.solid()) for c in placed])


def test_no_cruise_motor_gives_no_motor_plate(g1_kit_inputs):
    layout, _, _, settings = g1_kit_inputs
    assert cruise_motor_plate((), layout, settings) == ()


# ----------------------------------------------------------------------------------- booms and clamps ---
def _lift_motors(kit_designs, layout):
    return tuple(cad_components.vtol_motors(kit_designs["g2"][1], layout))


@pytest.fixture(scope="module")
def g2_booms(g2_kit_inputs, kit_designs):
    layout, _, spar, settings = g2_kit_inputs
    return boom_parts(layout, spar, settings, _lift_motors(kit_designs, layout), lift_thrust_n=_thrust(kit_designs))


def _thrust(kit_designs) -> float:
    return kit_designs["g2"][0].vtol_thrust_per_motor_n


@pytest.mark.parametrize("thrust", [None, 0.0, -5.0])
def test_quadplane_boom_parts_need_a_positive_lift_thrust(thrust, g2_kit_inputs):
    """D19: no thrust, zero thrust and negative thrust are all refused."""
    layout, _, spar, settings = g2_kit_inputs
    with pytest.raises(ValueError, match="lift_thrust_n"):
        boom_parts(layout, spar, settings, lift_thrust_n=thrust)


def test_fixed_wing_has_no_boom_clamps(g1_kit_inputs):
    layout, _, spar, settings = g1_kit_inputs
    parts, tubes = boom_parts(layout, spar, settings)
    assert not [p for p in parts if p.label.startswith(("BC-", "MP-"))]
    assert not [t for t in tubes if t.role == "lift_boom"]
    (tail,) = [t for t in tubes if t.role == "tail_boom"]
    assert tail.tube == boom_tube(layout.tail_boom.diameter_m, settings)
    assert tail.start_m == pytest.approx((layout.tail_boom.x_start_m, 0.0, layout.tail_boom.z_m))
    assert tail.length_m == pytest.approx(layout.tail_boom.length_m)


def test_lift_boom_is_the_r21_strength_sized_tube(g2_kit_inputs, g2_booms, kit_designs):
    """R21: sized for one motor's thrust at the farthest hub from the clamp; lighter than R13's widest tube."""
    from src.tools.fabrication.tubes import lift_boom_tube
    layout, _, _, settings = g2_kit_inputs
    _, tubes = g2_booms
    lift = [t for t in tubes if t.role == "lift_boom"]
    assert len(lift) == 2
    for tube, boom in zip(sorted(lift, key=lambda t: -t.start_m[1]), layout.vtol_booms):
        clamp_x = socket_axis_m(layout, abs(boom.y_m))[0]
        arm = max(abs(x - clamp_x) for x, y, _ in layout.rotor_positions if (y > 0) == (boom.y_m > 0))
        assert tube.tube in settings.tubes
        assert tube.tube == lift_boom_tube(_thrust(kit_designs), arm, boom.diameter_m, settings)
        assert tube.tube.area_m2 < boom_tube(boom.diameter_m, settings).area_m2
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


def test_boom_assembly_intersects_nothing(g2_kit_inputs, g2_booms, kit_designs):
    layout, airfoil, spar, settings = g2_kit_inputs
    parts, tubes = g2_booms
    y_boom = abs(layout.vtol_booms[0].y_m)
    solids = [(p.label, part_solid(p, settings)) for p in (*parts, *wing_ribs(layout, airfoil, spar, settings))]
    solids += [(t.label, _tube_solid(t)) for t in tubes if t.role == "lift_boom"]
    solids += [(f"spar{side:+.0f}", _spar_rod(layout, spar, side, y_boom)) for side in (1.0, -1.0)]
    solids += [(c.name, c.solid()) for c in _lift_motors(kit_designs, layout)]
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
def test_lift_boom_keeps_its_layout_height_when_the_spar_allows(drop_m, g2_kit_inputs, g2_booms, kit_designs):
    """Every drop builds. The boom sits one plate under the hub unless the spar forces it lower (g2's forced height,
    unchanged by the drop; it stops forcing at about a 5 mm drop); then the motor plates carry the motors exactly."""
    layout, _, spar, settings = g2_kit_inputs
    forced_z = next(t_ for t_ in g2_booms[1] if t_.role == "lift_boom").start_m[2]
    low = _lowered_quadplane(layout, drop_m)
    parts, tubes = boom_parts(low, spar, settings, _lift_motors(kit_designs, low), lift_thrust_n=_thrust(kit_designs))
    t = settings.sheet_for("plate").thickness_m
    od = next(t_ for t_ in tubes if t_.role == "lift_boom").tube.od_m
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


def test_one_sided_boom_set_is_unavailable(g2_kit_inputs, kit_designs):
    layout, _, spar, settings = g2_kit_inputs
    one = dataclasses.replace(layout, vtol_booms=layout.vtol_booms[:1])
    with pytest.raises(KitUnavailable, match="MP-"):
        boom_parts(one, spar, settings, lift_thrust_n=_thrust(kit_designs))


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


# ------------------------------------------------------------------- wing ribs clear the lift-boom clamps ---
def test_rib_inside_a_clamp_zone_moves_to_the_outboard_edge():
    moved = clear_rib_stations((0.1, 0.3, 0.5, 0.7), (RibKeepOut(0.49, 0.03),), semi_span_m=0.8)
    assert moved == pytest.approx((0.1, 0.3, 0.52 + RIB_CLAMP_MARGIN_M, 0.7))


def test_tip_rib_inside_a_clamp_zone_moves_inboard():
    moved = clear_rib_stations((0.2, 0.4, 0.6, 0.8), (RibKeepOut(0.79, 0.03),), semi_span_m=0.8)
    assert moved == pytest.approx((0.2, 0.4, 0.6, 0.76 - RIB_CLAMP_MARGIN_M))


def test_rib_moves_inboard_when_the_outboard_edge_is_in_another_zone():
    zones = (RibKeepOut(0.49, 0.03), RibKeepOut(0.56, 0.05))
    moved = clear_rib_stations((0.1, 0.3, 0.5, 0.7), zones, semi_span_m=0.8)
    assert moved == pytest.approx((0.1, 0.3, 0.46 - RIB_CLAMP_MARGIN_M, 0.7))


def test_rib_moves_inboard_when_outboard_would_crowd_the_next_rib():
    moved = clear_rib_stations((0.1, 0.3, 0.5, 0.7), (RibKeepOut(0.56, 0.065),), semi_span_m=0.8)
    assert moved == pytest.approx((0.1, 0.3, 0.4945, 0.7))
    assert min(b - a for a, b in zip(moved, moved[1:])) >= MIN_RIB_SPACING_M


def test_rib_with_no_clear_spaced_station_is_dropped_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING):
        moved = clear_rib_stations((0.1, 0.3, 0.5, 0.7), (RibKeepOut(0.5, 0.15),), semi_span_m=0.8)
    assert moved == (0.1, 0.3, 0.7)
    assert "rib dropped" in caplog.text


def test_a_dropped_rib_renumbers_the_ribs_and_places_the_servo_bay_from_the_rest(g1_kit_inputs):
    """Documented behaviour: labels stay contiguous (WR-01..WR-n-1) and the servo bay follows the remaining ribs."""
    layout, airfoil, spar, settings = g1_kit_inputs
    nominal = wing_rib_stations(spar, layout.wing.semi_span_m)
    reduced = (*nominal[:3], *nominal[4:])
    ribs = wing_ribs(layout, airfoil, dataclasses.replace(spar, rib_stations_m=reduced), settings)
    by = ribs_by_label(ribs)
    labels = sorted(label for label in by if re.fullmatch(r"WR-\d\d", label))
    assert labels == [f"WR-{i:02d}" for i in range(1, len(reduced) + 1)]
    offset = by["WR-02"].location.position.Y - reduced[1]
    assert by["WR-04"].location.position.Y - offset == pytest.approx(reduced[3], abs=1e-9)
    assert by["WR-SV"].location.position.Y == pytest.approx(_servo_bay_y(reduced, layout.wing), abs=1e-9)


def test_root_rib_never_moves():
    moved = clear_rib_stations((0.1, 0.3), (RibKeepOut(0.1, 0.03),), semi_span_m=0.8)
    assert moved[0] == 0.1


def _boom_beside_rib(layout, spar, offset_m: float):
    """`layout` with both lift booms (and their rotors) `offset_m` inboard of the third nominal wing rib station."""
    y_target = panel_rib_stations(spar.joint_y_m, layout.wing.semi_span_m)[2] - offset_m
    booms = tuple(dataclasses.replace(b, y_m=math.copysign(y_target, b.y_m)) for b in layout.vtol_booms)
    rotors = tuple((x, math.copysign(y_target, y), z) for x, y, z in layout.rotor_positions)
    return dataclasses.replace(layout, vtol_booms=booms, rotor_positions=rotors)


def test_lift_boom_inside_a_nominal_rib_clamp_zone_still_builds(g2_kit_inputs, kit_designs):
    layout, airfoil, spar, settings = g2_kit_inputs
    sizing, thrust = kit_designs["g2"][0], _thrust(kit_designs)
    shifted = _boom_beside_rib(layout, spar, offset_m=0.005)
    spar_shifted = stiffened_spar_plan(sizing, shifted, airfoil, settings)
    parts, tubes = boom_parts(shifted, spar_shifted, settings, _lift_motors(kit_designs, shifted), lift_thrust_n=thrust)
    assert {p.label for p in parts} >= set(CLAMP_LABELS[::2])
    nominal = panel_rib_stations(spar_shifted.joint_y_m, shifted.wing.semi_span_m)
    stations = wing_rib_stations(spar_shifted, shifted.wing.semi_span_m)
    assert len(stations) == len(nominal) and stations != nominal
    od = next(t for t in tubes if t.role == "lift_boom").tube.od_m
    t_rib = settings.sheet_for("rib").thickness_m
    y_boom = abs(shifted.vtol_booms[0].y_m)
    for y in stations[1:]:  # BC-R1 margin: hole half-width + web + rib gap + half a rib, from every rib centre
        assert abs(y - y_boom) >= (od + settings.fit_clearance_m) / 2 + MIN_WEB_M + 1.5 * t_rib - 1e-12
    ribs = wing_ribs(shifted, airfoil, spar_shifted, settings)
    by = ribs_by_label(ribs)  # the rib parts sit at the moved stations (same slab offset as their neighbour)
    gap = by["WR-03"].location.position.Y - by["WR-02"].location.position.Y
    assert gap == pytest.approx(stations[2] - stations[1], abs=1e-9)


def test_fixed_wing_rib_stations_are_the_nominal_pitch(g1_kit_inputs):
    layout, _, spar, _ = g1_kit_inputs
    assert wing_rib_stations(spar, layout.wing.semi_span_m) == panel_rib_stations(spar.joint_y_m,
                                                                                  layout.wing.semi_span_m)
