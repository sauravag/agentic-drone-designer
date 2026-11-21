"""Rib outline helpers: sliver cleanup."""

import build123d as bd
import pytest

from src.tools.fabrication.rib_outline import MIN_OUTLINE_EDGE_M, rect_face, without_slivers

SLIVER_M = 1e-5


def _polygon(points: list[tuple[float, float]]) -> bd.Face:
    return bd.Face(bd.Wire.make_polygon([(x, y, 0.0) for x, y in points], close=True))


def test_a_polygon_sliver_edge_is_closed_up():
    # Arrange: a 10 um edge on the right side of a 40 x 10 mm rectangle
    face = _polygon([(0.0, 0.0), (0.04, 0.0), (0.04, 0.005), (0.04, 0.005 + SLIVER_M), (0.04, 0.01), (0.0, 0.01)])
    assert min(e.length for e in face.outer_wire().edges()) < MIN_OUTLINE_EDGE_M
    # Act
    cleaned = without_slivers(face)
    # Assert
    edges = cleaned.outer_wire().edges()
    assert len(edges) == 5 and min(e.length for e in edges) >= MIN_OUTLINE_EDGE_M
    assert cleaned.area == pytest.approx(face.area, rel=1e-3)


def test_a_face_with_an_arc_is_returned_unchanged():
    # Arrange
    face = bd.Face(bd.Wire([bd.Edge.make_circle(0.01)]))
    # Act / Assert
    assert without_slivers(face) is face


def test_a_face_with_an_inner_wire_is_returned_unchanged():
    # Arrange: a rectangle with a sliver edge and a rectangular hole
    outer = rect_face(0.0, 0.0, 0.04, 0.01)
    face = outer.cut(rect_face(0.01, 0.002, 0.02, 0.008)).faces()[0]
    # Act / Assert
    assert face.inner_wires() and without_slivers(face) is face
