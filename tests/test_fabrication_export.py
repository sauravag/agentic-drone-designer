"""Task 8: nest kit parts on bed-sized sheets and write kerf-compensated DXF cut files.
Task 9: the build sheet, parts list and kit solids (`export_kit`, `kit_solids`)."""

import csv
import math
import re
import time
from itertools import combinations
from pathlib import Path

import build123d as bd
import ezdxf
import pytest
from ezdxf import bbox as dxf_bbox

from src.schemas.fabrication import DEFAULT_FABRICATION
from src.tools.fabrication.kit import KitPart, KitUnavailable
from src.tools.fabrication import nesting
from src.tools.fabrication.kit_export import (
    LABEL_HEIGHT_M,
    LABEL_MIN_HEIGHT_M,
    PARTS_CSV_COLUMNS,
    carbon_solids,
    export_kit,
    kit_solids,
    label_layout,
    ply_solids,
    write_sheets,
)
from src.tools.fabrication.kit_mass import kit_mass_items, kit_mass_kg, part_item
from src.tools.fabrication.nesting import MM_PER_M, Placement, kerf_outline, nest, placed_outline
from tests.cad_helpers import hole_diameters
from tests.test_fabrication_kit import _build

S = DEFAULT_FABRICATION
KERF_M = 0.00015
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


def test_short_parts_fill_the_strip_beside_a_tall_part():
    parts = [_plate("TALL", 0.40, 0.30), _plate("S", 0.15, 0.10, quantity=3)]
    placements = nest(parts, S)["Birch ply 3 mm"]
    assert {p.sheet for p in placements} == {1}
    _assert_on_bed_and_apart(placements)


def test_g1_three_mm_parts_fit_two_sheets(kit):
    assert max(p.sheet for p in nest(kit.parts, S)["Birch ply 3 mm"]) == 2


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


def test_kerf_narrows_open_notches_by_exactly_one_kerf():
    plate = (bd.Rectangle(0.1, 0.05) - bd.Pos(0, 0.025) * bd.Rectangle(SLOT_W_M, 0.02)).faces()[0]
    k = kerf_outline(plate, KERF_M)
    section = k.intersect(bd.Pos(0, 0.02) * bd.Rectangle(0.2, 1e-6).faces()[0])  # a thin band across the notch
    xs = sorted((e.bounding_box().min.X, e.bounding_box().max.X) for e in section.faces())
    assert len(xs) == 2 and xs[1][0] - xs[0][1] == pytest.approx(SLOT_W_M - KERF_M)


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
    assert all(t.dxf.layer == "ENGRAVE" for t in texts)
    assert all(LABEL_MIN_HEIGHT_M * MM_PER_M - 1e-9 <= t.dxf.height <= LABEL_HEIGHT_M * MM_PER_M + 1e-9 for t in texts)


def test_dxf_geometry_is_kerf_compensated_in_millimetres(tmp_path):
    part = KitPart("H", "plate", plate_with_hole(0.1, 0.05, hole_d=0.01), bd.Location())
    (path,) = write_sheets(nest([part], S), S, tmp_path)
    msp = ezdxf.readfile(path).modelspace()
    (circle,) = msp.query("CIRCLE[layer=='CUT_INNER']")
    assert 2.0 * circle.dxf.radius == pytest.approx(9.85)
    outer = dxf_bbox.extents(msp.query("*[layer=='CUT_OUTER']"))
    assert outer.size.x == pytest.approx(100.15) and outer.size.y == pytest.approx(50.15)
    assert outer.extmin.x == pytest.approx(S.margin_m * MM_PER_M)


def _assert_label_box_in_material(face: bd.Face, layout) -> None:
    box = bd.Pos(layout.x_m, layout.y_m) * bd.Rot(0, 0, layout.rotation_deg) * bd.Rectangle(layout.width_m,
                                                                                         layout.height_m)
    common = box.faces()[0].intersect(face)
    shapes = common if isinstance(common, list) else [common]
    assert sum(sh.area for sh in shapes if sh is not None) == pytest.approx(box.area, rel=1e-6)


def test_label_sits_inside_the_material_of_a_ring():
    ring = (bd.Circle(0.05) - bd.Circle(0.035)).faces()[0]
    layout = label_layout(ring, "BM-1")
    assert layout.fits and layout.height_m == pytest.approx(LABEL_HEIGHT_M)
    _assert_label_box_in_material(ring, layout)


def test_label_shrinks_to_fit_a_small_part():
    part = bd.Rectangle(0.025, 0.005).faces()[0]
    layout = label_layout(part, "WR-07A")
    assert layout.fits and LABEL_MIN_HEIGHT_M <= layout.height_m < LABEL_HEIGHT_M
    _assert_label_box_in_material(part, layout)


def test_label_keeps_clear_of_holes():
    plate = (bd.Rectangle(0.03, 0.012) - bd.Circle(0.0045)).faces()[0]
    layout = label_layout(plate, "P-1")
    assert layout.fits
    _assert_label_box_in_material(plate, layout)


def test_label_too_long_for_any_height_is_engraved_at_the_floor():
    layout = label_layout(bd.Rectangle(0.004, 0.003).faces()[0], "A-VERY-LONG-LABEL")
    assert not layout.fits and layout.height_m == pytest.approx(LABEL_MIN_HEIGHT_M)


def test_small_part_label_is_shrunk_in_the_dxf(tmp_path):
    part = KitPart("WR-07A", "rib", bd.Rectangle(0.025, 0.005).faces()[0], bd.Location())
    (text,) = _texts(write_sheets(nest([part], S), S, tmp_path))
    assert LABEL_MIN_HEIGHT_M * MM_PER_M <= text.dxf.height < LABEL_HEIGHT_M * MM_PER_M


def test_rewriting_removes_stale_sheets(tmp_path):
    first = write_sheets(nest([_plate("Q", 0.25, 0.18, quantity=6)], S), S, tmp_path)
    keep = tmp_path / "kit" / "sheets" / "notes.txt"
    keep.write_text("not a sheet")
    second = write_sheets(nest([_plate("Q", 0.25, 0.18)], S), S, tmp_path)
    assert len(first) == 2 and len(second) == 1
    assert sorted((tmp_path / "kit" / "sheets").glob("*.dxf")) == list(second) and keep.exists()


def test_kerf_offset_is_computed_once_per_part(tmp_path, monkeypatch):
    calls: list[str] = []
    placements = nest([_plate("Q", 0.05, 0.05, quantity=3)], S)
    real = nesting.kerf_outline
    monkeypatch.setattr(nesting, "kerf_outline", lambda face, kerf: calls.append("k") or real(face, kerf))
    write_sheets(placements, S, tmp_path)
    assert len(calls) == 1


def test_dxf_writing_is_deterministic(kit, tmp_path):
    first = write_sheets(nest(kit.parts, S), S, tmp_path / "a")
    second = write_sheets(nest(kit.parts, S), S, tmp_path / "b")
    for a, b in zip(first, second):
        ents = [ezdxf.readfile(p).modelspace() for p in (a, b)]
        assert [e.dxftype() for e in ents[0]] == [e.dxftype() for e in ents[1]]
        assert dxf_bbox.extents(ents[0]).extmax.isclose(dxf_bbox.extents(ents[1]).extmax)


def test_label_runs_along_the_longer_side_and_reads_level_on_round_parts():
    assert label_layout(bd.Rectangle(0.01, 0.1).faces()[0], "R").rotation_deg == pytest.approx(90.0)
    assert label_layout(bd.Rectangle(0.1, 0.01).faces()[0], "R").rotation_deg == pytest.approx(0.0)
    assert label_layout(bd.Circle(0.05).faces()[0], "R").rotation_deg == pytest.approx(0.0)


def test_label_falls_back_to_the_centre_when_no_grid_point_is_in_the_material():
    strip_m = 0.0002
    ell = (bd.Pos(0.05, strip_m / 2) * bd.Rectangle(0.1, strip_m)
           + bd.Pos(strip_m / 2, 0.05) * bd.Rectangle(strip_m, 0.1)).faces()[0]
    layout = label_layout(ell, "L")
    assert not layout.fits and (layout.x_m, layout.y_m) == pytest.approx((ell.center().X, ell.center().Y))


def test_curves_other_than_arcs_are_flattened_into_the_outline(tmp_path):
    part = KitPart("E", "plate", bd.Ellipse(0.04, 0.02).faces()[0], bd.Location())
    (path,) = write_sheets(nest([part], S), S, tmp_path)
    (poly,) = ezdxf.readfile(path).modelspace().query("LWPOLYLINE[layer=='CUT_OUTER']")
    assert poly.closed and len(poly) > 4
    outer = dxf_bbox.extents([poly])
    assert outer.size.x == pytest.approx(80.15, abs=0.05) and outer.size.y == pytest.approx(40.15, abs=0.05)


# ------------------------------------------------------------------------- label and kerf guards ---
def test_write_sheets_rejects_two_different_parts_with_one_label(tmp_path):
    a, b = _plate("P-1", 0.05, 0.05), _plate("P-1", 0.08, 0.05)
    placements = {"Birch ply 3 mm": (Placement(a, 1, 0.005, 0.005, False), Placement(b, 1, 0.1, 0.005, False))}
    with pytest.raises(KitUnavailable, match="P-1"):
        write_sheets(placements, S, tmp_path)


def test_write_sheets_reports_a_kerf_failure_like_nest(tmp_path, monkeypatch):
    part = _plate("P-2", 0.05, 0.05)
    placements = nest([part], S)

    def broken(face, kerf_m):
        raise ValueError("offset failed")

    monkeypatch.setattr("src.tools.fabrication.nesting.kerf_outline", broken)
    with pytest.raises(KitUnavailable, match="P-2"):
        write_sheets(placements, S, tmp_path)


def test_build_kit_rejects_duplicate_labels():
    from src.tools.fabrication.build_kit import _assert_unique_labels

    with pytest.raises(KitUnavailable, match="P-3"):
        _assert_unique_labels([_plate("P-3", 0.05, 0.05), _plate("P-3", 0.05, 0.05)], [])


# ---------------------------------------------------------------------- Task 9: build sheet, parts list ---
BUILD_SHEET_SECTIONS = ("Summary", "Parts", "Tubes", "Linear stock", "Joins", "Centreline joint", "Covering",
                        "Hardware")
MASS_TOL_KG = 1e-6
VOLUME_MASS_REL = 1e-6


@pytest.fixture(scope="module")
def kit_files(kit, tmp_path_factory):
    return export_kit(kit, S, tmp_path_factory.mktemp("kit_g1"))


@pytest.fixture(scope="module")
def g2_kit_files(g2_kit, tmp_path_factory):
    return export_kit(g2_kit, S, tmp_path_factory.mktemp("kit_g2"))


def _section(text: str, heading: str) -> str:
    """The body of `## heading` up to the next `## `."""
    match = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    assert match, f"no section {heading!r}"
    return match.group(1)


def test_export_kit_writes_the_sheets_build_sheet_and_parts_list(kit_files):
    assert kit_files.kit_dir.name == "kit" and kit_files.build_sheet == kit_files.kit_dir / "build_sheet.md"
    assert kit_files.parts_csv == kit_files.kit_dir / "parts.csv"
    assert kit_files.sheets and all(p.exists() and p.parent == kit_files.kit_dir / "sheets" for p in kit_files.sheets)
    assert kit_files.build_sheet.exists() and kit_files.parts_csv.exists()


def test_build_sheet_lists_every_label(kit, kit_files):
    text = kit_files.build_sheet.read_text()
    for label in [p.label for p in kit.parts] + [t.label for t in kit.tubes]:
        assert f"| {label} |" in text, label


def test_build_sheet_sections_come_in_order(kit_files):
    headings = re.findall(r"^## (.+)$", kit_files.build_sheet.read_text(), re.M)
    assert tuple(headings) == BUILD_SHEET_SECTIONS


def test_build_sheet_summary_carries_the_packing_kit_mass_and_sheet_counts(kit, kit_files):
    summary = _section(kit_files.build_sheet.read_text(), "Summary")
    total = float(re.search(r"Kit mass: \*\*([0-9.]+) kg\*\*", summary).group(1))
    assert total == pytest.approx(kit_mass_kg(kit, S), abs=5e-5)
    mass_table = summary.split("| Item | Mass (g) |")[1].split("\n\n")[0]
    grams = [float(g) for g in re.findall(r"^\| [^|]+ \| ([0-9.]+) \|$", mass_table, re.M)]
    assert len(grams) == 6
    assert sum(grams) / 1000 == pytest.approx(kit_mass_kg(kit, S), abs=1e-4)  # breakdown adds up, tubes included
    for stock, placements in nest(kit.parts, S).items():
        assert f"| {stock} | {len({p.sheet for p in placements})} |" in summary


def test_build_sheet_tubes_give_od_id_and_length_in_mm(kit, kit_files):
    tubes = _section(kit_files.build_sheet.read_text(), "Tubes")
    spar = next(t for t in kit.tubes if t.label == "SP-R1")
    assert f"| SP-R1 | spar | 12.0 × 10.0 | {spar.length_m * 1000:.1f} |" in tubes


def test_build_sheet_linear_stock_totals_each_kind(kit, kit_files):
    linear = _section(kit_files.build_sheet.read_text(), "Linear stock")
    stringers = sum(s.length_m for s in kit.linear if s.kind == "stringer")
    assert f"| stringer | 3.0 × 3.0 mm | {stringers * 1000:.0f} |" in linear


def test_build_sheet_joins_and_covering(kit, kit_files):
    text = kit_files.build_sheet.read_text()
    assert all(join in _section(text, "Joins") for join in kit.joins)
    covering = _section(text, "Covering")
    assert f"{kit.film_area_m2:.3f} m²" in covering and f"{kit.film_area_m2 * S.film_kg_m2 * 1000:.1f} g" in covering
    assert f"{kit.fuselage_cover_area_m2:.3f} m²" in covering


def test_centreline_joint_gives_the_overwrap_and_the_lay_up_order(kit, kit_files):
    joint = _section(kit_files.build_sheet.read_text(), "Centreline joint")
    wrap = kit.overwrap
    assert f"{wrap.layers} layers" in joint and "0.25 mm" in joint and "200 g/m² carbon" in joint
    assert f"{wrap.lap_m * 1000:.1f} mm per side" in joint and f"{wrap.od_m * 1000:.1f} mm" in joint
    assert f"{wrap.mass_kg * 1000:.1f} g" in joint
    steps = [joint.index(s) for s in ("Insert the sockets", "Glue the mitre", "Wrap between the plates")]
    assert steps == sorted(steps)
    assert "Do not glue the spar roots: the wings must stay detachable." in joint
    assert "| WM-RD |" in joint and "| WM-LD |" in joint  # g1's socket bosses


def test_build_sheet_cloth_comes_from_the_settings(kit, tmp_path):
    from src.tools.fabrication.kit_export import build_sheet_text

    heavy = S.model_copy(update={"overwrap_cloth_g_m2": 300.0})
    joint = _section(build_sheet_text(kit, heavy, nest(kit.parts, S)), "Centreline joint")
    assert "300 g/m² carbon" in joint and "200 g/m²" not in joint


def test_export_failure_leaves_no_partial_kit_files(kit, tmp_path, monkeypatch):
    from src.tools.fabrication import kit_export

    def broken(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(kit_export, "_write_parts_csv", broken)
    with pytest.raises(OSError, match="disk full"):
        export_kit(kit, S, tmp_path)
    kit_dir = tmp_path / "kit"
    assert not list((kit_dir / "sheets").glob("*.dxf"))
    assert not (kit_dir / "build_sheet.md").exists() and not (kit_dir / "parts.csv").exists()


def test_copies_of_a_part_are_one_solid_carrying_the_quantity(kit):
    """Copies (quantity > 1) have no placement of their own: one solid per part, mass still quantity-weighted."""
    import dataclasses

    plate = _plate("P-Q", 0.05, 0.04, quantity=3)
    twin = dataclasses.replace(kit, parts=(plate,))
    (solid,) = ply_solids(twin, S)
    sheet = S.sheet_for("plate")
    assert solid.volume * sheet.density_kg_m3 * plate.quantity == pytest.approx(part_item(plate, S).mass_kg,
                                                                               rel=VOLUME_MASS_REL)


def test_g2_build_sheet_lists_the_pin_bosses_and_its_pin_size(g2_kit, g2_kit_files):
    text = g2_kit_files.build_sheet.read_text()
    joint = _section(text, "Centreline joint")
    for label in ("WM-RPD", "WM-LPD", "WR-PD", "WL-PD"):
        assert f"| {label} |" in joint, label
    assert "2 × 5 mm anti-rotation pins" in _section(text, "Hardware")


def test_g1_hardware_lists_the_fixed_items_and_4_mm_pins(kit_files):
    hardware = _section(kit_files.build_sheet.read_text(), "Hardware")
    for item in ("2 nylon wing bolts M5 ×", "2 M5 nuts, one captive in each nut plate", "2 × 4 mm anti-rotation pins",
                 "2 wing connectors"):
        assert item in hardware


def test_parts_csv_mass_sums_to_part_mass(kit, kit_files):
    with kit_files.parts_csv.open(newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert tuple(reader.fieldnames) == PARTS_CSV_COLUMNS
    assert [r["label"] for r in rows] == [p.label for p in kit.parts]
    assert sum(float(r["mass_g"]) for r in rows) / 1000 == pytest.approx(
        sum(part_item(p, S).mass_kg for p in kit.parts), abs=MASS_TOL_KG)
    assert all(r["sheet"] for r in rows)
    rib = next(r for r in rows if r["family"] == "rib")
    assert rib["stock"] == "Birch ply 1.5 mm" and float(rib["thickness_mm"]) == 1.5


def test_parts_csv_includes_the_doublers(g2_kit, g2_kit_files):
    with g2_kit_files.parts_csv.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    labels = {r["label"] for r in rows}
    assert {"WM-RPD", "WM-LPD", "WR-PD", "WL-PD", "WM-RD", "WM-LD"} <= labels
    assert sum(float(r["mass_g"]) for r in rows) / 1000 == pytest.approx(
        sum(part_item(p, S).mass_kg for p in g2_kit.parts), abs=MASS_TOL_KG)


def test_kit_solid_volume_matches_part_mass(kit):
    ply = ply_solids(kit, S)
    carbon = carbon_solids(kit)
    tubes = carbon[:len(kit.tubes)]
    weights = [S.sheet_for(p.family).density_kg_m3 * p.quantity for p in kit.parts]  # one solid per part
    solid_mass = sum(s.volume * w for s, w in zip(ply, weights, strict=True))
    solid_mass += sum(s.volume for s in tubes) * S.carbon_density_kg_m3
    items = {i.name: i.mass_kg for i in kit_mass_items(kit, S)}
    expected = sum(items[p.label] for p in kit.parts) + sum(items[t.label] for t in kit.tubes)
    assert solid_mass == pytest.approx(expected, rel=VOLUME_MASS_REL)
    assert len(kit_solids(kit, S)) == len(ply) + len(carbon)


def test_overwrap_halves_are_solids_round_the_sockets(kit):
    halves = carbon_solids(kit)[len(kit.tubes):]
    wrap = kit.overwrap
    socket_od = next(t.tube.od_m for t in kit.tubes if t.role == "socket")
    annulus = math.pi / 4 * (wrap.od_m**2 - socket_od**2)
    assert len(halves) == 2
    assert sum(h.volume for h in halves) == pytest.approx(2 * annulus * wrap.lap_m, rel=VOLUME_MASS_REL)
