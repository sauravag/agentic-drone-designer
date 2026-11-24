"""R23: kit-aware pod diameter, sliders seated low, crutch slots on steep tail cones, crutch split through openings."""

import math

import build123d as bd
import pytest

from src.schemas.fabrication import DEFAULT_FABRICATION
from src.tools.aero_cad import components
from src.tools.aero_cad.airfoil_geometry import load_airfoil
from src.tools.aero_cad.components import COMPONENT_GAP_M, PlacedComponent
from src.tools.aero_cad.layout import DEFAULT_WING_STATION, WING_STATION_SCAN, fuselage_radius_at, layout_from_sizing
from src.tools.fabrication.build_kit import kit_spar_plan
from src.tools.fabrication.fittings import split_to_bed
from src.tools.fabrication.fuselage_parts import fuselage_parts, wing_box_floor_m, wing_box_plate_gap_m
from src.tools.fabrication.kit import KitPart, KitUnavailable
from src.tools.fabrication.kit_pod import (
    STRUCTURE_CLEARANCE_M,
    fits_kit_formers,
    kit_fuselage_diameter_m,
    pod_fits_kit,
    seat_low,
)
from src.tools.fabrication.rib_outline import rect_face
from src.tools.fabrication.wing_box_bosses import BOSS_LABELS
from src.tools.fabrication.tubes import MIN_WEB_M, NO_PLATE_LOADS, plate_loads, size_overwrap

SETTINGS = DEFAULT_FABRICATION
COVER_M = SETTINGS.fuselage_cover_thickness_m
USABLE_LONG_M, USABLE_SHORT_M = sorted(DEFAULT_FABRICATION.usable_bed_m, reverse=True)
STEEP_CONE_DIAMETER_M = 0.33  # g2 pod this wide: tail-cone slope ~0.37, beyond the 1 mm a slot inset leaves
AREA_TOL_M2 = 1e-9


def _layout(designs, case: str, diameter_m: float, station: float = DEFAULT_WING_STATION):
    sizing, bom = designs[case]
    length = components.required_fuselage_length_m(sizing, bom, diameter_m)
    return layout_from_sizing(sizing, station, diameter_m, length)


def _floor(designs, case: str, diameter_m: float, station: float = DEFAULT_WING_STATION) -> float:
    layout = _layout(designs, case, diameter_m, station)
    plan = kit_spar_plan(designs[case][0], layout, load_airfoil(layout.airfoil), SETTINGS)
    return wing_box_floor_m(layout, plan, SETTINGS, plate_loads(designs[case][0], SETTINGS))


def _battery_top(designs, case: str, diameter_m: float) -> float:
    battery = seat_low(components.sliders(*designs[case])[0], diameter_m, SETTINGS)
    return battery.center_m[2] + battery.size_m[2] / 2


# ------------------------------------------------------------------------------------ pod diameter ---
def test_group_3_pod_is_unchanged(designs):
    sizing, bom = designs["g3"]
    assert kit_fuselage_diameter_m(sizing, bom, SETTINGS) == components.required_fuselage_diameter_m(sizing, bom)


@pytest.mark.parametrize("case", ["g1", "g2"])
def test_kit_pod_seats_the_battery_under_the_wing_box(case, designs):
    sizing, bom = designs[case]
    d = kit_fuselage_diameter_m(sizing, bom, SETTINGS)
    assert d > max(sizing.fuselage_diameter_m, components.required_fuselage_diameter_m(sizing, bom))
    assert _battery_top(designs, case, d) + STRUCTURE_CLEARANCE_M <= _floor(designs, case, d) + 1e-9
    assert pod_fits_kit(sizing, bom, d, SETTINGS)
    assert not pod_fits_kit(sizing, bom, d - 0.001, SETTINGS)  # the smallest such pod (to 1 mm)


@pytest.mark.parametrize("case", ["g1", "g2"])
@pytest.mark.parametrize("station", WING_STATION_SCAN)
def test_battery_clears_the_wing_box_at_every_station_the_balance_may_pick(case, station, designs):
    sizing, bom = designs[case]
    d = kit_fuselage_diameter_m(sizing, bom, SETTINGS)
    try:
        floor = _floor(designs, case, d, station)
    except KitUnavailable:
        pytest.skip("no wing box can be built at this station (no kit there)")
    assert _battery_top(designs, case, d) + STRUCTURE_CLEARANCE_M <= floor + 1e-9


# The hand-launched G1's pod before pod sizing checked the overwrap: wide enough for the battery under the wing box,
# but its wing-box plates stand 83 mm apart and the 22 mm socket's overwrap is 88 mm long (2 laps of 2 OD).
HAND_POD_BEFORE_OVERWRAP_RULE_M = 0.172


def _plan_and_loads(designs, case: str, diameter_m: float, station: float = DEFAULT_WING_STATION):
    layout = _layout(designs, case, diameter_m, station)
    sizing = designs[case][0]
    return layout, kit_spar_plan(sizing, layout, load_airfoil(layout.airfoil), SETTINGS), plate_loads(sizing, SETTINGS)


@pytest.mark.parametrize("case", ["g1", "g1_hand"])
@pytest.mark.parametrize("bossed", [True, False])
def test_plate_gap_matches_the_built_plates(case, bossed, designs):
    """The overwrap's plate gap is the clear width between the built plates, or their socket bosses when the plate
    loads need them (both reference designs do; unloaded plates have none)."""
    d = kit_fuselage_diameter_m(*designs[case], SETTINGS)
    layout, plan, design_loads = _plan_and_loads(designs, case, d)
    loads = design_loads if bossed else NO_PLATE_LOADS
    built = fuselage_parts(layout, plan, (), SETTINGS, loads=loads)
    t = SETTINGS.sheet_for("plate").thickness_m  # plates and bosses extrude toward -y from their origin
    measured = 2.0 * min(p.location.position.Y - t for p in built if p.label in ("WM-R", BOSS_LABELS[1]))
    assert wing_box_plate_gap_m(layout, plan, SETTINGS, loads) == pytest.approx(measured)


def test_wing_box_floor_refuses_an_overwrap_longer_than_the_plate_gap(designs):
    with pytest.raises(KitUnavailable, match="centreline overwrap .* does not fit between the wing-box plates"):
        _floor(designs, "g1_hand", HAND_POD_BEFORE_OVERWRAP_RULE_M)


def test_kit_pod_widens_until_the_overwrap_fits_between_the_plates(designs):
    sizing, bom = designs["g1_hand"]
    d = kit_fuselage_diameter_m(sizing, bom, SETTINGS)
    assert d > HAND_POD_BEFORE_OVERWRAP_RULE_M and pod_fits_kit(sizing, bom, d, SETTINGS)
    layout, plan, loads = _plan_and_loads(designs, "g1_hand", d)
    size_overwrap(plan.centre_moment_nm, plan.socket, wing_box_plate_gap_m(layout, plan, SETTINGS, loads), SETTINGS)


def test_structure_clearance_is_half_the_component_gap():
    assert STRUCTURE_CLEARANCE_M == COMPONENT_GAP_M / 2  # the kit cuts openings box + half the gap


# --------------------------------------------------------------------------------------- seat low ---
@pytest.mark.parametrize("case", ["g1", "g2"])
def test_sliders_seat_as_low_as_the_former_openings_allow(case, designs):
    sizing, bom = designs[case]
    d = kit_fuselage_diameter_m(sizing, bom, SETTINGS)
    layout = _layout(designs, case, d)
    for item in components.sliders(sizing, bom):
        low = seat_low(item, d, SETTINGS)
        assert low.center_m[2] <= 0.0 and low.center_m[:2] == item.center_m[:2]  # the pod-sizing slider sits at 0
        hy, hz = (low.size_m[1] + COMPONENT_GAP_M) / 2, (low.size_m[2] + COMPONENT_GAP_M) / 2
        corner = math.hypot(hy, abs(low.center_m[2]) + hz)
        assert corner + MIN_WEB_M + SETTINGS.stringer_side_m == pytest.approx(d / 2 - COVER_M, abs=1e-9)
        mid = low.at_x(0.5 * layout.fuselage_length_m)
        assert fits_kit_formers(mid, layout, SETTINGS)
        assert not fits_kit_formers(mid.at_x(low.size_m[0] / 2 + 0.001), layout, SETTINGS)  # nose cone


def test_seat_low_never_raises_a_part_that_cannot_drop():
    big = PlacedComponent("b", "box", 1.0, (0.1, 0.2, 0.1), (0.3, 0.0, 0.0))
    assert seat_low(big, 0.2, SETTINGS).center_m[2] == 0.0


# ---------------------------------------------------------------------------------- crutch slots ---
def test_crutch_slots_survive_a_steep_tail_cone(designs):
    layout = _layout(designs, "g2", STEEP_CONE_DIAMETER_M)
    plan = kit_spar_plan(designs["g2"][0], layout, load_airfoil(layout.airfoil), SETTINGS)
    crutch = next(p for p in fuselage_parts(layout, plan, (), SETTINGS, loads=NO_PLATE_LOADS) if p.label == "C-1")
    x0 = crutch.location.position.X
    for wire in crutch.outline.inner_wires():  # every tab slot keeps MIN_WEB_M to the edge at its aft face
        box = wire.bounding_box()
        edge = fuselage_radius_at(layout, x0 + box.max.X + MIN_WEB_M) - COVER_M
        assert max(abs(box.min.Y), abs(box.max.Y)) + MIN_WEB_M <= edge + 1e-9


# ------------------------------------------------------------------------ split through openings ---
def _crutch_with_opening(length_m: float = 1.2, width_m: float = 0.16, opening=(0.40, 0.75, 0.12)) -> KitPart:
    """A crutch-like plate with a long centred component opening (x0, x1, width) and former tab slots."""
    x0, x1, w = opening
    face = rect_face(0.0, -width_m / 2, length_m, width_m / 2).cut(rect_face(x0, -w / 2, x1, w / 2))
    slots = [rect_face(x, side * 0.068, x + 0.003, side * 0.074) if side > 0 else rect_face(x, -0.074, x + 0.003, -0.068)
             for x in (0.12, 0.24, 0.36, 0.84, 0.96, 1.08) for side in (1, -1)]
    face = face.cut(*slots).faces()[0]
    return KitPart("C-9", "crutch", face, bd.Location((0.1, 0.0, -0.0015)), openings=((x0, -w / 2, x1, w / 2),))


def _fits_bed(face: bd.Face) -> bool:
    box = face.bounding_box()
    w, h = box.max.X - box.min.X, box.max.Y - box.min.Y
    return max(w, h) <= USABLE_LONG_M + 1e-9 and min(w, h) <= USABLE_SHORT_M + 1e-9


def test_crutch_splits_through_a_long_battery_opening():
    part = _crutch_with_opening()
    parts = split_to_bed(part, SETTINGS)
    pieces = [p for p in parts if "-D" not in p.label]
    doublers = [p for p in parts if "-D" in p.label]
    assert len(pieces) >= 3 and all(_fits_bed(p.outline) for p in parts)
    assert sum(p.area_m2 for p in pieces) == pytest.approx(part.area_m2, abs=AREA_TOL_M2)
    for d in doublers:  # doublers lap the side rails only: MIN_WEB_M clear of the opening and of every slot
        assert all(d.outline.distance_to(w) >= MIN_WEB_M - 1e-6 for w in part.outline.inner_wires()), d.label


def test_a_cut_through_the_opening_gives_each_rail_its_own_dovetail_and_doubler():
    """A 0.9 m plate whose only legal cut lies inside a 0.5 m opening."""
    face = rect_face(0.0, -0.08, 0.9, 0.08).cut(rect_face(0.2, -0.055, 0.7, 0.055)).faces()[0]
    part = KitPart("C-9", "crutch", face, bd.Location((0.1, 0.0, -0.0015)), openings=((0.2, -0.055, 0.7, 0.055),))
    parts = split_to_bed(part, SETTINGS)
    pieces = [p for p in parts if "-D" not in p.label]
    doublers = sorted(p.label for p in parts if "-D" in p.label)
    assert len(pieces) == 2 and doublers == ["C-9-D1a", "C-9-D1b"]
    cut_x = max(v.X for v in pieces[0].outline.vertices())  # tail tips of the left piece
    assert 0.2 < cut_x < 0.7
    assert sum(p.area_m2 for p in pieces) == pytest.approx(part.area_m2, abs=AREA_TOL_M2)


def test_small_holes_are_never_cut_open():
    """A cut may cross a component opening but never a small (bolt / lightening) hole."""
    holes = [rect_face(0.30 + 0.02 * k, -0.01, 0.31 + 0.02 * k, 0.01) for k in range(16)]  # 10 mm holes every 20 mm
    face = rect_face(0.0, -0.08, 0.9, 0.08).cut(*holes).faces()[0]
    parts = split_to_bed(KitPart("C-9", "crutch", face, bd.Location()), SETTINGS)
    pieces = [p for p in parts if "-D" not in p.label]
    assert sum(len(p.outline.inner_wires()) for p in pieces) == len(holes)


def test_a_long_untagged_slot_is_never_cut_through():
    """The same 0.5 m hole, but not tagged as a component opening (a future long slot): no legal cut exists."""
    face = rect_face(0.0, -0.08, 0.9, 0.08).cut(rect_face(0.2, -0.055, 0.7, 0.055)).faces()[0]
    with pytest.raises(KitUnavailable, match="C-9 cannot be split"):
        split_to_bed(KitPart("C-9", "crutch", face, bd.Location((0.1, 0.0, -0.0015))), SETTINGS)


def test_tagged_openings_follow_a_part_split_along_y():
    """A tall part is turned onto x before splitting; its tagged opening turns with it."""
    face = rect_face(-0.08, 0.0, 0.08, 0.9).cut(rect_face(-0.055, 0.2, 0.055, 0.7)).faces()[0]
    part = KitPart("C-9", "crutch", face, bd.Location(), openings=((-0.055, 0.2, 0.055, 0.7),))
    pieces = [p for p in split_to_bed(part, SETTINGS) if "-D" not in p.label]
    assert len(pieces) == 2 and sum(p.area_m2 for p in pieces) == pytest.approx(part.area_m2, abs=AREA_TOL_M2)


def test_a_part_in_the_kerf_band_is_split_so_nesting_accepts_it():
    """M1: within the usable bed nominally but over it once kerfed: splitting must use the bed less the kerf."""
    from src.tools.fabrication.nesting import nest

    width = max(SETTINGS.usable_bed_m) - SETTINGS.kerf_m / 2.0
    parts = split_to_bed(KitPart("C-9", "crutch", rect_face(0.0, -0.05, width, 0.05), bd.Location()), SETTINGS)
    assert len([p for p in parts if "-D" not in p.label]) == 2
    assert nest(parts, SETTINGS)  # every piece fits the usable bed with its kerf
