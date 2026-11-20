"""DXF cut files for the nested laser-cut kit: one file per sheet, millimetres.

`write_sheets` writes `<out_dir>/kit/sheets/<stock-name-slug>-<n>.dxf` (n = 1-based sheet number) for every
sheet `nesting.nest` filled. Each placement's kerf-compensated outline is drawn as:
- outer wire on layer `CUT_OUTER`, inner wires (holes, slots) on `CUT_INNER`;
- a wire that is one full circle as a DXF `CIRCLE`; any other wire as one closed `LWPOLYLINE` with exact bulge
  arcs for circular edges (lines and arcs are all the kit outlines contain), and any other curve flattened to
  CURVE_SAMPLES points per edge. Closed polylines and circles are what laser software imports most reliably;
- the part label as `TEXT` on layer `ENGRAVE` (`label_layout`): LABEL_HEIGHT_M high, shrunk in steps down to
  LABEL_MIN_HEIGHT_M until its estimated extent fits in the material clear of the outline and holes, centred at
  the roomiest grid point where it fits, along the part's longer side when it fits that way. A label that fits
  nowhere even at the floor is engraved at the floor anyway (engraving cuts nothing).
- the kerf offset is computed once per part label, however many copies are placed.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping, Optional, Sequence

import build123d as bd
import ezdxf
import numpy as np
from ezdxf import colors, units
from ezdxf.enums import TextEntityAlignment
from ezdxf.layouts import Modelspace
from OCP.BRepTools import BRepTools_WireExplorer

from src.schemas.fabrication import FabricationSettings
from src.tools.fabrication.nesting import EPS_M, MM_PER_M, QUARTER_TURN_DEG, Placement, kerf_outline, place_kerfed

__all__ = ["LABEL_HEIGHT_M", "LABEL_MIN_HEIGHT_M", "LabelLayout", "label_layout", "write_sheets"]

LABEL_HEIGHT_M: Final[float] = 0.004
LABEL_MIN_HEIGHT_M: Final[float] = 0.0015  # labels shrink to fit down to this, and are engraved at it regardless
LABEL_HEIGHT_STEP_M: Final[float] = 0.00025
LABEL_CHAR_WIDTH: Final[float] = 0.8  # estimated character advance / text height (generous for laser fonts)
LABEL_CLEARANCE_M: Final[float] = 0.001  # kept between the label's estimated extent and any cut (no burn-through)
LAYER_CUT_OUTER: Final[str] = "CUT_OUTER"
LAYER_CUT_INNER: Final[str] = "CUT_INNER"
LAYER_ENGRAVE: Final[str] = "ENGRAVE"
LAYER_COLOURS: Final[dict[str, int]] = {LAYER_CUT_OUTER: colors.RED, LAYER_CUT_INNER: colors.BLUE,
                                        LAYER_ENGRAVE: colors.BLACK}
DXF_VERSION: Final[str] = "R2010"
METRIC_MEASUREMENT: Final[int] = 1  # $MEASUREMENT: 1 = metric
CURVE_SAMPLES: Final[int] = 24  # points per non-line edge when flattening (labels; non-circular DXF curves)
LABEL_GRID: Final[int] = 15  # label candidates per bounding-box side
MIN_SEGMENT_SQ_M2: Final[float] = 1e-30  # guards the projection onto a zero-length boundary segment
SHEETS_SUBDIR: Final[tuple[str, str]] = ("kit", "sheets")


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _mm(v: bd.Vector) -> tuple[float, float]:
    return v.X * MM_PER_M, v.Y * MM_PER_M


def _xy(v: bd.Vector) -> tuple[float, float]:
    return v.X, v.Y


def _ordered_edges(wire: bd.Wire) -> list[bd.Edge]:
    """The wire's edges in traversal order and direction (OCCT's wire explorer: ~100x faster than
    `Wire.order_edges`, which dominated sheet-writing time)."""
    explorer = BRepTools_WireExplorer(wire.wrapped)
    edges: list[bd.Edge] = []
    while explorer.More():
        edges.append(bd.Edge(explorer.Current()))
        explorer.Next()
    return edges


def _is_circle(edges: Sequence[bd.Edge]) -> bool:
    return len(edges) == 1 and edges[0].geom_type == bd.GeomType.CIRCLE


def _bulge(edge: bd.Edge) -> float:
    """DXF bulge (tan of a quarter of the signed sweep; + is counter-clockwise) of a circular arc edge."""
    start, mid, end = (np.array(_xy(edge.position_at(t))) for t in (0.0, 0.5, 1.0))
    chord = end - start
    cross = chord[0] * (mid - start)[1] - chord[1] * (mid - start)[0]
    sagitta = abs(cross) / np.linalg.norm(chord)
    return float(-np.sign(cross) * 2.0 * sagitta / np.linalg.norm(chord))


def _edge_points(edge: bd.Edge) -> list[tuple[bd.Vector, float]]:
    """(point, bulge) vertices from the edge's start up to (not including) its end."""
    if edge.geom_type == bd.GeomType.LINE:
        return [(edge.position_at(0.0), 0.0)]
    if edge.geom_type == bd.GeomType.CIRCLE:
        return [(edge.position_at(0.0), _bulge(edge))]
    return [(edge.position_at(t), 0.0) for t in np.linspace(0.0, 1.0, CURVE_SAMPLES, endpoint=False)]


def _add_wire(msp: Modelspace, wire: bd.Wire, layer: str) -> None:
    edges = _ordered_edges(wire)
    if _is_circle(edges):
        msp.add_circle(_mm(edges[0].arc_center), edges[0].radius * MM_PER_M, dxfattribs={"layer": layer})
        return
    points = [(*_mm(p), bulge) for edge in edges for p, bulge in _edge_points(edge)]
    msp.add_lwpolyline(points, format="xyb", close=True, dxfattribs={"layer": layer})


def _ring(wire: bd.Wire) -> np.ndarray:
    """The wire as a closed polygon (n, 2), arcs and curves sampled."""
    pts = []
    for edge in _ordered_edges(wire):
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
    width_m: float  # estimated text extent (LABEL_CHAR_WIDTH x height per character) by its height
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
    return len(text) * LABEL_CHAR_WIDTH * height_m, height_m


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
    LABEL_MIN_HEIGHT_M whose estimated extent (plus LABEL_CLEARANCE_M) fits in the material, clear of the
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


def _add_part(msp: Modelspace, placement: Placement, kerfed: bd.Face) -> None:
    face = place_kerfed(kerfed, placement)
    _add_wire(msp, face.outer_wire(), LAYER_CUT_OUTER)
    for wire in face.inner_wires():
        _add_wire(msp, wire, LAYER_CUT_INNER)
    label = label_layout(face, placement.part.label)
    text = msp.add_text(placement.part.label, height=label.height_m * MM_PER_M, rotation=label.rotation_deg,
                        dxfattribs={"layer": LAYER_ENGRAVE})
    text.set_placement((label.x_m * MM_PER_M, label.y_m * MM_PER_M), align=TextEntityAlignment.MIDDLE_CENTER)


def _write_sheet(placements: Sequence[Placement], kerfed: Mapping[str, bd.Face], path: Path) -> Path:
    doc = ezdxf.new(DXF_VERSION)
    doc.units = units.MM
    doc.header["$MEASUREMENT"] = METRIC_MEASUREMENT
    for name, colour in LAYER_COLOURS.items():
        doc.layers.add(name, color=colour)
    msp = doc.modelspace()
    for placement in placements:
        _add_part(msp, placement, kerfed[placement.part.label])
    doc.saveas(path)
    return path


def write_sheets(placements: Mapping[str, Sequence[Placement]], settings: FabricationSettings,
                 out_dir: Path) -> tuple[Path, ...]:
    """One DXF per nested sheet under `out_dir/kit/sheets/`; returns the paths in stock, then sheet order.
    Any `*.dxf` already in that folder (an earlier run's sheets) is deleted first; nothing else there is touched."""
    sheet_dir = out_dir.joinpath(*SHEETS_SUBDIR)
    sheet_dir.mkdir(parents=True, exist_ok=True)
    for stale in sorted(sheet_dir.glob("*.dxf")):
        stale.unlink()
    all_placements = [p for stock_placements in placements.values() for p in stock_placements]
    parts = {p.part.label: p.part for p in all_placements}
    kerfed = {label: kerf_outline(part.outline, settings.kerf_m) for label, part in parts.items()}
    paths: list[Path] = []
    for stock, stock_placements in placements.items():
        for n in sorted({p.sheet for p in stock_placements}):
            on_sheet = [p for p in stock_placements if p.sheet == n]
            paths.append(_write_sheet(on_sheet, kerfed, sheet_dir / f"{_slug(stock)}-{n}.dxf"))
    return tuple(paths)
