"""Engraved part labels for the DXF sheets (SI metres): where each label goes and the strokes that draw it.

`label_layout` picks the label's height, from LABEL_HEIGHT_M shrinking in steps down to LABEL_MIN_HEIGHT_M, until
its extent (the stroke font's exact text width by its height) plus LABEL_CLEARANCE_M fits in the material clear of
the outline and holes; it is centred at the roomiest grid point where it fits, along the part's longer side when it
fits that way. A label that fits nowhere even at the floor is engraved at the floor anyway (engraving cuts nothing).
`label_strokes` draws it in the single-stroke font (`stroke_font`) as open polylines, which laser software keeps
(it drops DXF TEXT). Deterministic: the same face and text give the same strokes.
"""

from dataclasses import dataclass
from typing import Final, Optional, Sequence

import build123d as bd
import numpy as np
from OCP.BRepTools import BRepTools_WireExplorer

from src.tools.fabrication.nesting import EPS_M, QUARTER_TURN_DEG
from src.tools.fabrication.stroke_font import Point, text_strokes, text_width

__all__ = ["CURVE_SAMPLES", "LABEL_HEIGHT_M", "LABEL_MIN_HEIGHT_M", "LabelLayout", "label_layout", "label_strokes",
           "ordered_edges"]

LABEL_HEIGHT_M: Final[float] = 0.004
LABEL_MIN_HEIGHT_M: Final[float] = 0.0015  # labels shrink to fit down to this, and are engraved at it regardless
LABEL_HEIGHT_STEP_M: Final[float] = 0.00025
LABEL_CLEARANCE_M: Final[float] = 0.001  # kept between the label's extent and any cut (no burn-through)
CURVE_SAMPLES: Final[int] = 24  # points per non-line edge when flattening (labels; non-circular DXF curves)
LABEL_GRID: Final[int] = 15  # label candidates per bounding-box side
MIN_SEGMENT_SQ_M2: Final[float] = 1e-30  # guards the projection onto a zero-length boundary segment


def _xy(v: bd.Vector) -> tuple[float, float]:
    return v.X, v.Y


def ordered_edges(wire: bd.Wire) -> list[bd.Edge]:
    """The wire's edges in traversal order and direction (OCCT's wire explorer: ~100x faster than
    `Wire.order_edges`, which dominated sheet-writing time)."""
    explorer = BRepTools_WireExplorer(wire.wrapped)
    edges: list[bd.Edge] = []
    while explorer.More():
        edges.append(bd.Edge(explorer.Current()))
        explorer.Next()
    return edges


def _ring(wire: bd.Wire) -> np.ndarray:
    """The wire as a closed polygon (n, 2), arcs and curves sampled."""
    pts = []
    for edge in ordered_edges(wire):
        ts = [0.0] if edge.geom_type == bd.GeomType.LINE else np.linspace(0.0, 1.0, CURVE_SAMPLES, endpoint=False)
        pts.extend(_xy(edge.position_at(t)) for t in ts)
    return np.array(pts)


def _segments(face: bd.Face) -> tuple[np.ndarray, np.ndarray]:
    rings = [_ring(w) for w in (face.outer_wire(), *face.inner_wires())]
    return np.vstack(rings), np.vstack([np.roll(r, -1, axis=0) for r in rings])


def _inside(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Even-odd test of points (n, 2) against all boundary segments a -> b (holes count as outside)."""
    px, py = pts[:, None, 0], pts[:, None, 1]
    ax, ay, bx, by = a[None, :, 0], a[None, :, 1], b[None, :, 0], b[None, :, 1]
    straddles = (ay > py) != (by > py)
    with np.errstate(divide="ignore", invalid="ignore"):  # level segments never straddle; their nan is masked
        x_cross = ax + (py - ay) * (bx - ax) / (by - ay)
    return np.count_nonzero(straddles & (px < x_cross), axis=1) % 2 == 1


def _clearance(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest boundary segment."""
    ab = b - a
    length_sq = np.maximum((ab * ab).sum(-1), MIN_SEGMENT_SQ_M2)
    t = np.clip(((pts[:, None, :] - a[None]) * ab[None]).sum(-1) / length_sq, 0.0, 1.0)
    nearest = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(pts[:, None, :] - nearest, axis=-1).min(axis=1)


@dataclass(frozen=True)
class LabelLayout:
    x_m: float  # centre of the label text
    y_m: float
    rotation_deg: float
    width_m: float  # the stroke font's text width at `height_m`
    height_m: float
    fits: bool  # the extent, plus LABEL_CLEARANCE_M, lies wholly in the material


def _box_crosses(centres: np.ndarray, half: tuple[float, float], a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Whether any boundary segment a -> b touches the axis-aligned box of half-extents `half` round each centre."""
    lo, hi = centres - np.array(half), centres + np.array(half)
    seg_lo, seg_hi = np.minimum(a, b), np.maximum(a, b)
    overlap = ((seg_lo[None] <= hi[:, None]) & (seg_hi[None] >= lo[:, None])).all(-1)  # (n, m) bbox overlap
    corners = np.stack([lo, np.column_stack([hi[:, 0], lo[:, 1]]), hi, np.column_stack([lo[:, 0], hi[:, 1]])], 1)
    ab = b - a
    rel = corners[:, None, :, :] - a[None, :, None, :]  # (n, m, 4, 2)
    side = ab[None, :, None, 0] * rel[..., 1] - ab[None, :, None, 1] * rel[..., 0]
    straddle = (side.min(-1) <= 0.0) & (side.max(-1) >= 0.0)
    return (overlap & straddle).any(axis=1)


def _extent(text: str, height_m: float) -> tuple[float, float]:
    """The label's exact extent in the stroke font: (width, cap height)."""
    return text_width(text, height_m), height_m


def _heights() -> np.ndarray:
    steps = round((LABEL_HEIGHT_M - LABEL_MIN_HEIGHT_M) / LABEL_HEIGHT_STEP_M)
    return np.linspace(LABEL_HEIGHT_M, LABEL_MIN_HEIGHT_M, steps + 1)


def _first_fit(text: str, candidates: np.ndarray, rotations: Sequence[float], a: np.ndarray,
               b: np.ndarray) -> Optional[LabelLayout]:
    """The tallest label (then the first rotation, then the roomiest candidate) whose padded box clears every cut."""
    for height_m in _heights():
        w_m, h_m = _extent(text, height_m)
        for rotation in rotations:
            span = (w_m, h_m) if rotation == 0.0 else (h_m, w_m)
            half = (span[0] / 2.0 + LABEL_CLEARANCE_M, span[1] / 2.0 + LABEL_CLEARANCE_M)
            clear = ~_box_crosses(candidates, half, a, b)
            if clear.any():
                x, y = candidates[int(np.argmax(clear))]
                return LabelLayout(float(x), float(y), rotation, w_m, h_m, True)
    return None


def label_layout(face: bd.Face, text: str) -> LabelLayout:
    """Where and how big to engrave `text` on `face`: the largest height from LABEL_HEIGHT_M down to
    LABEL_MIN_HEIGHT_M whose stroke-font extent (plus LABEL_CLEARANCE_M) fits in the material, clear of the
    outline and holes, at a grid point (roomiest first), along the longer side if it fits (level when the sides
    are equal). If nothing fits at the floor, the floor height at the roomiest point (`fits=False`), or at the
    face centre when no grid point lies in the material."""
    box = face.bounding_box()
    xs = np.linspace(box.min.X, box.max.X, LABEL_GRID + 2)[1:-1]
    ys = np.linspace(box.min.Y, box.max.Y, LABEL_GRID + 2)[1:-1]
    grid = np.array([(x, y) for y in ys for x in xs])
    a, b = _segments(face)
    inside = grid[_inside(grid, a, b)]
    along = QUARTER_TURN_DEG if box.size.Y > box.size.X + EPS_M else 0.0  # round parts read level
    w_m, h_m = _extent(text, LABEL_MIN_HEIGHT_M)
    if not len(inside):
        centre = face.center()
        return LabelLayout(centre.X, centre.Y, along, w_m, h_m, False)
    roomiest = inside[np.argsort(-_clearance(inside, a, b), kind="stable")]
    layout = _first_fit(text, roomiest, (along, QUARTER_TURN_DEG - along), a, b)
    if layout is not None:
        return layout
    return LabelLayout(float(roomiest[0][0]), float(roomiest[0][1]), along, w_m, h_m, False)


def label_strokes(text: str, layout: LabelLayout) -> list[list[Point]]:
    """The label's strokes (open polylines, metres) at `layout`'s height, turned by its rotation about its centre."""
    angle = np.radians(layout.rotation_deg)
    cos, sin = float(np.cos(angle)), float(np.sin(angle))
    return [[(layout.x_m + x * cos - y * sin, layout.y_m + x * sin + y * cos) for x, y in stroke]
            for stroke in text_strokes(text, layout.height_m)]
