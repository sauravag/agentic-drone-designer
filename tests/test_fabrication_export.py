"""Task 8: nest kit parts on bed-sized sheets and write kerf-compensated DXF cut files."""

import time
from itertools import combinations
from pathlib import Path

import build123d as bd
import ezdxf
import pytest
from ezdxf import bbox as dxf_bbox

from src.schemas.fabrication import DEFAULT_FABRICATION
from src.tools.fabrication.kit import KitPart, KitUnavailable
from src.tools.fabrication.kit_export import label_point, write_sheets
from src.tools.fabrication.nesting import Placement, kerf_outline, nest, placed_outline
from tests.cad_helpers import hole_diameters
from tests.test_fabrication_kit import _build

S = DEFAULT_FABRICATION
KERF_M = 0.00015
MM_PER_M = 1000.0
GAP_TOL_M = 1e-9
G2_NEST_BUDGET_S = 30.0
SLOT_W_M = 0.003


# ------------------------------------------------------------------------------------------- helpers ---
def plate_with_hole(width_m: float, height_m: float, hole_d: float) -> bd.Face:
    return (bd.Rectangle(width_m, height_m) - bd.Pos(width_m / 4, 0) * bd.Circle(hole_d / 2)).faces()[0]


def plate_with_slot(width_m: float, height_m: float, slot_w: float, slot_l: float) -> bd.Face:
    return (bd.Rectangle(width_m, height_m) - bd.Rectangle(slot_w, slot_l)).faces()[0]


def outer_width(face: bd.Face) -> float:
    return face.outer_wire().bounding_box().size.X


def _plate(label: str, w: float, h: float, quantity: int = 1, family: str = "plate") -> KitPart:
    return KitPart(label, family, bd.Rectangle(w, h).faces()[0], bd.Location(), quantity)


def _sheet_boxes(placements: tuple[Placement, ...], sheet: int) -> list[tuple[str, bd.BoundBox]]:
    return [(p.part.label, placed_outline(p, S.kerf_m).bounding_box()) for p in placements if p.sheet == sheet]


def _assert_on_bed_and_apart(placements: tuple[Placement, ...]) -> None:
    m = S.margin_m
    for sheet in sorted({p.sheet for p in placements}):
        boxes = _sheet_boxes(placements, sheet)
        for label, b in boxes:
            assert b.min.X >= m - GAP_TOL_M and b.min.Y >= m - GAP_TOL_M, label
            assert b.max.X <= S.bed_width_m - m + GAP_TOL_M, label
            assert b.max.Y <= S.bed_height_m - m + GAP_TOL_M, label
        for (la, a), (lb, b) in combinations(boxes, 2):
            gap_x = max(b.min.X - a.max.X, a.min.X - b.max.X)
            gap_y = max(b.min.Y - a.max.Y, a.min.Y - b.max.Y)
            assert max(gap_x, gap_y) >= m - GAP_TOL_M, (la, lb, sheet)


def _texts(paths: tuple[Path, ...]) -> list:
    return [e for path in paths for e in ezdxf.readfile(path).modelspace().query("TEXT")]


@pytest.fixture(scope="module")
def kit(designs, g1_kit_inputs):
    return _build(designs, g1_kit_inputs, "g1")


@pytest.fixture(scope="module")
def g2_kit(designs, g2_kit_inputs):
    return _build(designs, g2_kit_inputs, "g2")


# ------------------------------------------------------------------------------------------- nesting ---
def test_no_part_leaves_the_bed_or_overlaps(kit):
    sheets = nest(kit.parts, S)
    assert set(sheets) <= {s.name for s in S.sheets}
    for placements in sheets.values():
        _assert_on_bed_and_apart(placements)


def test_every_part_lands_on_the_sheet_of_its_family(kit):
    sheets = nest(kit.parts, S)
    for name, placements in sheets.items():
        assert all(S.sheet_for(p.part.family).name == name for p in placements)
    assert sum(len(v) for v in sheets.values()) == sum(p.quantity for p in kit.parts)


def test_g2_nests_without_overlap_quickly(g2_kit):
    start = time.perf_counter()
    sheets = nest(g2_kit.parts, S)
    elapsed = time.perf_counter() - start
    for placements in sheets.values():
        _assert_on_bed_and_apart(placements)
    assert sum(len(v) for v in sheets.values()) == sum(p.quantity for p in g2_kit.parts)
    assert elapsed < G2_NEST_BUDGET_S


def test_rotation_used_when_only_90_degrees_fits():
    (placement,) = nest([_plate("BIG", 0.35, 0.55)], S)["Birch ply 3 mm"]
    assert placement.rotated
    box = placed_outline(placement, S.kerf_m).bounding_box()
    assert box.size.X == pytest.approx(0.55 + S.kerf_m) and box.size.Y == pytest.approx(0.35 + S.kerf_m)


def test_lower_height_orientation_chosen_when_both_fit():
    (tall,) = nest([_plate("TALL", 0.05, 0.30)], S)["Birch ply 3 mm"]
    (wide,) = nest([_plate("WIDE", 0.30, 0.05)], S)["Birch ply 3 mm"]
    assert tall.rotated and not wide.rotated


def test_part_too_big_for_the_bed_is_unavailable():
    with pytest.raises(KitUnavailable, match="HUGE"):
        nest([_plate("HUGE", 0.45, 0.45)], S)


def test_quantity_places_copies_and_overflows_to_more_sheets():
    placements = nest([_plate("Q", 0.25, 0.18, quantity=6)], S)["Birch ply 3 mm"]
    assert len(placements) == 6
    assert {p.sheet for p in placements} == {1, 2}  # 2 x 2 per 600 x 400 sheet with margins
    _assert_on_bed_and_apart(placements)


def test_nesting_is_deterministic(kit):
    def key(sheets):
        return {k: [(p.part.label, p.sheet, p.x_m, p.y_m, p.rotated) for p in v] for k, v in sheets.items()}

    assert key(nest(kit.parts, S)) == key(nest(kit.parts, S))


# ---------------------------------------------------------------------------------------------- kerf ---
def test_kerf_grows_outlines_and_shrinks_holes():
    f = plate_with_hole(0.1, 0.05, hole_d=0.01)
    k = kerf_outline(f, KERF_M)
    assert outer_width(k) == pytest.approx(0.10015) and hole_diameters(k)[0] == pytest.approx(0.00985)


def test_kerf_narrows_slots_by_exactly_one_kerf():
    k = kerf_outline(plate_with_slot(0.1, 0.05, SLOT_W_M, 0.02), KERF_M)
    (slot,) = k.inner_wires()
    assert slot.bounding_box().size.X == pytest.approx(SLOT_W_M - KERF_M)
    assert slot.bounding_box().size.Y == pytest.approx(0.02 - KERF_M)


def test_zero_kerf_leaves_the_outline_unchanged():
    f = plate_with_hole(0.1, 0.05, hole_d=0.01)
    assert kerf_outline(f, 0.0).area == pytest.approx(f.area)


# ----------------------------------------------------------------------------------------------- DXF ---
def test_dxf_round_trip_layers_and_units(kit, tmp_path):
    doc = ezdxf.readfile(write_sheets(nest(kit.parts, S), S, tmp_path)[0])
    assert {"CUT_INNER", "CUT_OUTER", "ENGRAVE"} <= {l.dxf.name for l in doc.layers}
    assert doc.header["$INSUNITS"] == 4  # millimetres


def test_sheet_files_are_named_by_stock_slug(tmp_path):
    parts = [_plate("Q", 0.25, 0.18, quantity=6), _plate("R", 0.05, 0.02, family="rib")]
    paths = write_sheets(nest(parts, S), S, tmp_path)
    names = sorted(p.relative_to(tmp_path).as_posix() for p in paths)
    assert names == ["kit/sheets/birch-ply-1-5-mm-1.dxf", "kit/sheets/birch-ply-3-mm-1.dxf",
                     "kit/sheets/birch-ply-3-mm-2.dxf"]


def test_every_label_engraved_once(kit, tmp_path):
    texts = _texts(write_sheets(nest(kit.parts, S), S, tmp_path))
    assert len(texts) == sum(p.quantity for p in kit.parts)
    assert sorted(t.dxf.text for t in texts) == sorted(p.label for p in kit.parts for _ in range(p.quantity))
    assert all(t.dxf.layer == "ENGRAVE" and t.dxf.height == pytest.approx(4.0) for t in texts)


def test_dxf_geometry_is_kerf_compensated_in_millimetres(tmp_path):
    part = KitPart("H", "plate", plate_with_hole(0.1, 0.05, hole_d=0.01), bd.Location())
    (path,) = write_sheets(nest([part], S), S, tmp_path)
    msp = ezdxf.readfile(path).modelspace()
    (circle,) = msp.query("CIRCLE[layer=='CUT_INNER']")
    assert 2.0 * circle.dxf.radius == pytest.approx(9.85)
    outer = dxf_bbox.extents(msp.query("*[layer=='CUT_OUTER']"))
    assert outer.size.x == pytest.approx(100.15) and outer.size.y == pytest.approx(50.15)
    assert outer.extmin.x == pytest.approx(S.margin_m * MM_PER_M)


def test_label_sits_inside_the_material_of_a_ring():
    ring = (bd.Circle(0.05) - bd.Circle(0.035)).faces()[0]
    x, y, _ = label_point(ring)
    assert ring.is_inside(bd.Vector(x, y, 0)) and bd.Vector(x, y).length > 0.035


def test_dxf_writing_is_deterministic(kit, tmp_path):
    first = write_sheets(nest(kit.parts, S), S, tmp_path / "a")
    second = write_sheets(nest(kit.parts, S), S, tmp_path / "b")
    for a, b in zip(first, second):
        ents = [ezdxf.readfile(p).modelspace() for p in (a, b)]
        assert [e.dxftype() for e in ents[0]] == [e.dxftype() for e in ents[1]]
        assert dxf_bbox.extents(ents[0]).extmax.isclose(dxf_bbox.extents(ents[1]).extmax)


def test_label_runs_along_the_longer_side_and_reads_level_on_round_parts():
    assert label_point(bd.Rectangle(0.01, 0.1).faces()[0])[2] == pytest.approx(90.0)
    assert label_point(bd.Rectangle(0.1, 0.01).faces()[0])[2] == pytest.approx(0.0)
    assert label_point(bd.Circle(0.05).faces()[0])[2] == pytest.approx(0.0)


def test_label_falls_back_to_the_centre_when_no_grid_point_is_in_the_material():
    strip_m = 0.0002
    ell = (bd.Pos(0.05, strip_m / 2) * bd.Rectangle(0.1, strip_m)
           + bd.Pos(strip_m / 2, 0.05) * bd.Rectangle(strip_m, 0.1)).faces()[0]
    x, y, _ = label_point(ell)
    assert (x, y) == pytest.approx((ell.center().X, ell.center().Y))


def test_curves_other_than_arcs_are_flattened_into_the_outline(tmp_path):
    part = KitPart("E", "plate", bd.Ellipse(0.04, 0.02).faces()[0], bd.Location())
    (path,) = write_sheets(nest([part], S), S, tmp_path)
    (poly,) = ezdxf.readfile(path).modelspace().query("LWPOLYLINE[layer=='CUT_OUTER']")
    assert poly.closed and len(poly) > 4
    outer = dxf_bbox.extents([poly])
    assert outer.size.x == pytest.approx(80.15, abs=0.05) and outer.size.y == pytest.approx(40.15, abs=0.05)
