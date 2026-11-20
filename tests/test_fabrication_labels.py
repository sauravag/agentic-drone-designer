"""Engraved part labels: placed where they fit (`label_layout`) and drawn as single-stroke polylines on ENGRAVE
(`stroke_font`), one DXF group per label, with no TEXT entities (laser software drops DXF text)."""

import string
from pathlib import Path

import build123d as bd
import ezdxf
import pytest
from ezdxf import bbox as dxf_bbox

from src.schemas.fabrication import DEFAULT_FABRICATION
from src.tools.fabrication.kit import KitPart
from src.tools.fabrication.kit_export import write_sheets
from src.tools.fabrication.kit_labels import (
    LABEL_HEIGHT_M,
    LABEL_MIN_HEIGHT_M,
    label_layout,
    label_strokes,
)
from src.tools.fabrication.nesting import MM_PER_M, nest
from src.tools.fabrication.stroke_font import GLYPHS, text_strokes, text_width
from tests.test_fabrication_kit import _build

S = DEFAULT_FABRICATION
EXTENT_TOL_MM = 1e-6


@pytest.fixture(scope="module")
def kit(designs, g1_kit_inputs):
    return _build(designs, g1_kit_inputs, "g1")


def _groups(paths: tuple[Path, ...]) -> list[tuple[str, list]]:
    """(label, entities) of every DXF group in the sheets."""
    out = []
    for path in paths:
        for _, group in ezdxf.readfile(path).groups:
            out.append((group.dxf.description, list(group)))
    return out


def _height_mm(entities: list) -> float:
    """Cap height of an engraved label: its shorter extent (every kit label is wider than tall, turned or not)."""
    size = dxf_bbox.extents(entities).size
    return min(size.x, size.y)


# ------------------------------------------------------------------------------------------- font ---
def test_every_character_a_kit_label_uses_has_a_glyph():
    for char in string.ascii_uppercase + string.digits + "-":
        assert char in GLYPHS, char


def test_lowercase_letters_are_small_capitals():
    tall, small = text_strokes("A", 1.0), text_strokes("a", 1.0)
    top = max(y for stroke in tall for _, y in stroke)
    assert max(y for stroke in small for _, y in stroke) < top
    assert text_width("a", 1.0) < text_width("A", 1.0)


def test_an_unknown_character_draws_a_box_not_nothing():
    assert text_strokes("?", 1.0) and text_width("?", 1.0) == pytest.approx(text_width("A", 1.0))


def test_strokes_are_centred_and_span_the_text_width():
    strokes = text_strokes("WR-01", 0.004)
    xs = [x for stroke in strokes for x, _ in stroke]
    ys = [y for stroke in strokes for _, y in stroke]
    half = text_width("WR-01", 0.004) / 2
    assert min(xs) == pytest.approx(-half) and max(xs) <= half + 1e-12  # the "1" is narrower than its cell
    assert (min(ys), max(ys)) == pytest.approx((-0.002, 0.002))


# ------------------------------------------------------------------------------------------ layout ---
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


def test_label_strokes_stay_inside_the_laid_out_box():
    part = bd.Rectangle(0.01, 0.1).faces()[0]  # a turned label
    layout = label_layout(part, "WR-07A")
    pts = [p for stroke in label_strokes("WR-07A", layout) for p in stroke]
    half_w, half_h = layout.height_m / 2 + 1e-12, layout.width_m / 2 + 1e-12  # turned 90 deg
    assert all(abs(x - layout.x_m) <= half_w and abs(y - layout.y_m) <= half_h for x, y in pts)


# --------------------------------------------------------------------------------------------- DXF ---
def test_every_label_is_engraved_once_as_polylines(kit, tmp_path):
    paths = write_sheets(nest(kit.parts, S), S, tmp_path)
    groups = _groups(paths)
    assert sorted(label for label, _ in groups) == sorted(p.label for p in kit.parts for _ in range(p.quantity))
    for label, entities in groups:
        assert entities and all(e.dxftype() == "LWPOLYLINE" and e.dxf.layer == "ENGRAVE" for e in entities), label
        height = _height_mm(entities)
        assert LABEL_MIN_HEIGHT_M * MM_PER_M - EXTENT_TOL_MM <= height <= LABEL_HEIGHT_M * MM_PER_M + EXTENT_TOL_MM


def test_no_text_entities_are_written(kit, tmp_path):
    for path in write_sheets(nest(kit.parts, S), S, tmp_path):
        assert not ezdxf.readfile(path).modelspace().query("TEXT MTEXT")


def test_small_part_label_is_shrunk_in_the_dxf(tmp_path):
    part = KitPart("WR-07A", "rib", bd.Rectangle(0.025, 0.005).faces()[0], bd.Location())
    ((label, entities),) = _groups(write_sheets(nest([part], S), S, tmp_path))
    assert label == "WR-07A"
    assert LABEL_MIN_HEIGHT_M * MM_PER_M - EXTENT_TOL_MM <= _height_mm(entities) < LABEL_HEIGHT_M * MM_PER_M


def test_engraved_labels_are_deterministic(kit, tmp_path):
    def engraved(folder: Path) -> list:
        return [[tuple(v) for v in e.get_points("xy")]
                for path in write_sheets(nest(kit.parts, S), S, folder)
                for e in ezdxf.readfile(path).modelspace().query("LWPOLYLINE[layer=='ENGRAVE']")]

    assert engraved(tmp_path / "a") == engraved(tmp_path / "b")
