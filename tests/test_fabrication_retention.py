"""R36 wing retention: one M5 nylon bolt per side through the wing-box plate and the root doubler, into a nut held
captive in a nut plate NP-R / NP-L glued inside the wing root."""

import math

import build123d as bd
import pytest

from src.schemas.fabrication import DEFAULT_FABRICATION
from src.tools.aero_cad.components import PlacedComponent
from src.tools.aero_cad.structure import SPAR_CHORD_STATION
from src.tools.fabrication.kit import KitUnavailable
from src.tools.fabrication.kit_export import build_sheet_text, ply_solids
from src.tools.fabrication.kit_mass import kit_mass_items
from src.tools.fabrication.nesting import nest
from src.tools.fabrication.rib_outline import Hole, circle_face
from src.tools.fabrication.tubes import MIN_WEB_M
from src.tools.fabrication.wing_parts import PIN_CHORD_FRACTION, socket_axis_m
from src.tools.fabrication.wing_retention import (
    BOLT_CHORD_FRACTION,
    BOLT_HEAD_STACK_M,
    BOLT_LENGTH_STEP_M,
    BOLT_THREAD_MARGIN_M,
    BOLT_WASHER_DIAMETER_M,
    M5_CLEARANCE_HOLE_M,
    M5_NUT_ACROSS_FLATS_M,
    M5_NUT_HEIGHT_M,
    NUT_PLATE_DIAMETER_M,
    NUT_PLATE_LABELS,
    check_clear_of_holes,
    check_clear_of_outlines,
    check_head_clear,
    nut_plate_outline,
)
from tests.cad_helpers import hole_centre, part_solid, ribs_by_label, tube_solid
from tests.test_fabrication_kit import _build

S = DEFAULT_FABRICATION
WEB_TOL_M = 1e-7
CASES = ("g1", "g2")


@pytest.fixture(scope="module")
def kits(kit_designs, g1_kit_inputs, g2_kit_inputs):
    return {"g1": (_build(kit_designs, g1_kit_inputs, "g1"), g1_kit_inputs),
            "g2": (_build(kit_designs, g2_kit_inputs, "g2"), g2_kit_inputs)}


def _round_holes(face: bd.Face) -> list[bd.Wire]:
    return [w for w in face.inner_wires() if len(w.edges()) == 1 and w.edges()[0].geom_type == bd.GeomType.CIRCLE]


def _assert_every_hole_keeps_its_web(face: bd.Face, label: str) -> None:
    wires = [face.outer_wire(), *face.inner_wires()]
    for hole in _round_holes(face):
        for other in wires:
            if not other.is_same(hole):
                assert hole.distance_to(other) >= MIN_WEB_M - WEB_TOL_M, label


# ------------------------------------------------------------------------------------- constants ---
def test_the_bolt_sits_between_the_spar_and_the_pin():
    assert SPAR_CHORD_STATION < BOLT_CHORD_FRACTION < PIN_CHORD_FRACTION


def test_the_named_hardware_dimensions_are_m5():
    assert 0.005 < M5_CLEARANCE_HOLE_M < 0.006
    assert M5_NUT_ACROSS_FLATS_M == pytest.approx(0.008)
    assert NUT_PLATE_LABELS == {1: "NP-R", -1: "NP-L"}


def test_the_nut_plate_holds_the_hex_with_a_web_all_round():
    face = nut_plate_outline(0.1, 0.0, S, "NP-R")
    (hexagon,) = face.inner_wires()
    assert len(hexagon.edges()) == 6
    assert hexagon.bounding_box().size.Y == pytest.approx(M5_NUT_ACROSS_FLATS_M + S.fit_clearance_m, abs=1e-9)
    assert face.outer_wire().distance_to(hexagon) >= MIN_WEB_M - WEB_TOL_M
    assert face.outer_wire().bounding_box().size.X == pytest.approx(NUT_PLATE_DIAMETER_M, abs=1e-9)


# --------------------------------------------------------------------------------------- the kit ---
@pytest.mark.parametrize("case", CASES)
def test_nut_plates_are_doubler_parts_mirrored_left_and_right(case, kits):
    kit, _ = kits[case]
    by = ribs_by_label(kit.parts)
    right, left = by["NP-R"], by["NP-L"]
    assert right.family == left.family == "doubler"
    a, b = (part_solid(p, S).bounding_box() for p in (right, left))
    assert (b.min.Y, b.max.Y) == pytest.approx((-a.max.Y, -a.min.Y), abs=1e-9)
    t = S.sheet_for("doubler").thickness_m
    joint = kit.spar.joint_y_m
    assert (a.min.Y, a.max.Y) == pytest.approx((joint + t, joint + 2 * t), abs=1e-9)  # outboard of WR-01


@pytest.mark.parametrize("case", CASES)
def test_nut_plates_reach_the_mass_the_cut_sheets_and_the_solids(case, kits):
    kit, _ = kits[case]
    names = {item.name for item in kit_mass_items(kit, S)}
    placed = {p.part.label for ps in nest(kit.parts, S).values() for p in ps}
    assert {"NP-R", "NP-L"} <= names and {"NP-R", "NP-L"} <= placed
    assert len(ply_solids(kit, S)) == len(kit.parts)


@pytest.mark.parametrize("case", CASES)
def test_root_doublers_carry_the_bolt_hole_on_the_bolt_line(case, kits):
    kit, (layout, _, _, _) = kits[case]
    by = ribs_by_label(kit.parts)
    chord = layout.wing.chord_at(kit.spar.joint_y_m)
    xs = [hole_centre(w).X for w in _round_holes(by["WR-01"].outline)]
    assert any(x == pytest.approx(BOLT_CHORD_FRACTION * chord, abs=0.001) for x in xs)
    for label in ("WR-01", "WL-01"):
        _assert_every_hole_keeps_its_web(by[label].outline, label)


@pytest.mark.parametrize("case", CASES)
def test_wing_box_plates_carry_socket_bolt_and_pin_holes_with_their_webs(case, kits):
    kit, (layout, _, _, _) = kits[case]
    by = ribs_by_label(kit.parts)
    for label in ("WM-R", "WM-L"):
        plate = by[label]
        centres = sorted(((plate.location * bd.Location(tuple(hole_centre(w)))).position
                          for w in _round_holes(plate.outline)), key=lambda v: v.X)
        assert len(centres) == 3, label  # socket, bolt, pin (fore to aft)
        box = part_solid(plate, S).bounding_box()
        y = 0.5 * (box.min.Y + box.max.Y)
        assert (centres[1].X, centres[1].Z) == pytest.approx(socket_axis_m(layout, y, BOLT_CHORD_FRACTION), abs=1e-9)
        _assert_every_hole_keeps_its_web(plate.outline, label)


@pytest.mark.parametrize("case", CASES)
def test_nut_plates_sit_inside_the_root_rib_clear_of_the_pin_boss(case, kits):
    kit, _ = kits[case]
    by = ribs_by_label(kit.parts)
    nut = by["NP-R"].outline
    if "WR-PD" in by:
        assert nut.distance_to(by["WR-PD"].outline) >= MIN_WEB_M - WEB_TOL_M
    assert nut.cut(bd.Face(by["WR-01"].outline.outer_wire())).area < 1e-10


@pytest.mark.parametrize("case", CASES)
def test_the_wing_bolt_is_listed_with_length_location_and_order(case, kits):
    kit, _ = kits[case]
    bolt = kit.wing_bolt
    steps_of_length = bolt.length_m / BOLT_LENGTH_STEP_M
    assert bolt.length_m >= bolt.grip_m and math.isclose(steps_of_length, round(steps_of_length), abs_tol=1e-9)
    text = build_sheet_text(kit, S, nest(kit.parts, S))
    hardware = text.split("## Hardware", 1)[1]
    assert f"M5 × {bolt.length_m * 1e3:.0f} mm" in hardware
    assert f"{BOLT_CHORD_FRACTION:.1%}" in hardware
    for label in ("NP-R", "NP-L", "WM-R", "WR-01"):
        assert label in hardware
    assert sum(1 for line in hardware.splitlines() if line[:2] in ("1.", "2.", "3.", "4.")) >= 3


# ----------------------------------------------------------------------------------------- checks ---
def test_a_feature_too_close_to_the_nut_plate_is_unavailable():
    with pytest.raises(KitUnavailable, match="NP-R"):
        check_clear_of_holes(Hole(0.0, 0.0, NUT_PLATE_DIAMETER_M), [Hole(0.009, 0.0, 0.004)], "NP-R", "the pin")
    check_clear_of_holes(Hole(0.0, 0.0, NUT_PLATE_DIAMETER_M), [Hole(0.03, 0.0, 0.004)], "NP-R", "the pin")


def test_a_boss_under_the_washer_is_unavailable():
    washer = Hole(0.0, 0.0, 0.010)
    with pytest.raises(KitUnavailable, match="WM-R"):
        check_clear_of_outlines(washer, [circle_face(Hole(0.008, 0.0, 0.008))], "WM-R", "a boss")
    check_clear_of_outlines(washer, [circle_face(Hole(0.05, 0.0, 0.008))], "WM-R", "a boss")


def test_a_component_at_the_bolt_head_is_unavailable():
    head = ((0.0, -0.01, -0.005), (0.01, 0.0, 0.005))

    def battery(y_m: float) -> PlacedComponent:
        return PlacedComponent(name="battery", kind="box", mass_kg=1.0, size_m=(0.01, 0.01, 0.01),
                               center_m=(0.005, y_m, 0.0))

    with pytest.raises(KitUnavailable, match="battery"):
        check_head_clear([battery(-0.02)], head, "WM-R")
    check_head_clear([battery(-0.1)], head, "WM-R")


# ------------------------------------------------------------------------------------ bolt path (R38) ---
M5_SHANK_M = 0.005  # the bolt's own diameter: what must pass without touching any part


def _bolt_path(kit, layout) -> bd.Solid:
    """The right wing bolt as a 5 mm cylinder on its line, from the head (BOLT_HEAD_STACK_M inboard of the plate's
    inner face) to the nut's far end (M5_NUT_HEIGHT_M outboard of the root doubler), plus a thread margin."""
    plate = ribs_by_label(kit.parts)["WM-R"]
    t_plate, t_root = S.sheet_for("plate").thickness_m, S.sheet_for("doubler").thickness_m
    y0 = plate.location.position.Y - t_plate - BOLT_HEAD_STACK_M
    y1 = kit.spar.joint_y_m + t_root + M5_NUT_HEIGHT_M + BOLT_THREAD_MARGIN_M
    (x0, z0), (x1, z1) = (socket_axis_m(layout, y, BOLT_CHORD_FRACTION) for y in (y0, y1))
    start, end = bd.Vector(x0, y0, z0), bd.Vector(x1, y1, z1)
    plane = bd.Plane(origin=start, z_dir=end - start)
    return bd.Solid.make_cylinder(M5_SHANK_M / 2.0, (end - start).length, plane)


@pytest.mark.parametrize("case", CASES)
def test_the_wing_bolt_path_is_clear_of_every_kit_part_and_tube(case, kits):
    """R38: the bolt goes in as cut: its path from head to nut meets no part or tube (it passes their holes)."""
    kit, (layout, _, _, _) = kits[case]
    path = _bolt_path(kit, layout)
    solids = [(p.label, part_solid(p, S)) for p in kit.parts] + [(t.label, tube_solid(t)) for t in kit.tubes]
    hits = {label: round(_volume(path.intersect(solid)) * 1e9, 1) for label, solid in solids}
    assert {label: mm3 for label, mm3 in hits.items() if mm3 > 0.0} == {}


def _volume(shape) -> float:
    pieces = shape if isinstance(shape, list) else [shape]
    return sum(s.volume for piece in pieces if piece is not None for s in piece.solids())


@pytest.mark.parametrize("case", CASES)
def test_pod_sizing_counts_the_bolt_head_in_the_wing_box_floor(case, kits, monkeypatch):
    """R38: the wing-box floor that pod sizing seats the battery under includes the bolt head and washer, so pod
    sizing never accepts a pod whose battery the kit's head check then rejects."""
    from src.tools.fabrication import fuselage_parts as fp
    from src.tools.fabrication.tubes import PlateLoads

    kit, (layout, _, spar, _) = kits[case]
    loads = PlateLoads(shear_n=1.0, down_n=1.0)
    floor = fp.wing_box_floor_m(layout, spar, S, loads)
    plate = ribs_by_label(kit.parts)["WM-R"]
    y_inner = plate.location.position.Y - S.sheet_for("plate").thickness_m
    head_z = socket_axis_m(layout, y_inner - S.sheet_for("plate").thickness_m / 2.0, BOLT_CHORD_FRACTION)[1]
    assert floor <= head_z - BOLT_WASHER_DIAMETER_M / 2.0 + 1e-9
    monkeypatch.setattr(fp, "BOLT_WASHER_DIAMETER_M", 0.2)  # a washer reaching below everything else governs
    assert fp.wing_box_floor_m(layout, spar, S, loads) == pytest.approx(head_z - 0.1, abs=1e-9)
