"""DXF cut files for the nested laser-cut kit: one file per sheet, millimetres.

`write_sheets` writes `<out_dir>/kit/sheets/<stock-name-slug>-<n>.dxf` (n = 1-based sheet number) for every
sheet `nesting.nest` filled. Each placement's kerf-compensated outline is drawn as:
- outer wire on layer `CUT_OUTER`, inner wires (holes, slots) on `CUT_INNER`;
- a wire that is one full circle as a DXF `CIRCLE`; any other wire as one closed `LWPOLYLINE` with exact bulge
  arcs for circular edges (lines and arcs are all the kit outlines contain), and any other curve flattened to
  CURVE_SAMPLES points per edge. Closed polylines and circles are what laser software imports most reliably;
- the part label as `TEXT` on layer `ENGRAVE`, LABEL_HEIGHT_M high, centred at the grid point inside the
  material farthest from the outline (`label_point`), along the part's longer side.
"""

import re
from pathlib import Path
from typing import Final, Mapping, Sequence

import build123d as bd
import ezdxf
import numpy as np
from ezdxf import colors, units
from ezdxf.enums import TextEntityAlignment
from ezdxf.layouts import Modelspace

from src.schemas.fabrication import FabricationSettings
from src.tools.fabrication.nesting import EPS_M, QUARTER_TURN_DEG, Placement, placed_outline

__all__ = ["LABEL_HEIGHT_M", "label_point", "write_sheets"]

LABEL_HEIGHT_M: Final[float] = 0.004
MM_PER_M: Final[float] = 1000.0
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
    edges = wire.order_edges()
    if _is_circle(edges):
        msp.add_circle(_mm(edges[0].arc_center), edges[0].radius * MM_PER_M, dxfattribs={"layer": layer})
        return
    points = [(*_mm(p), bulge) for edge in edges for p, bulge in _edge_points(edge)]
    msp.add_lwpolyline(points, format="xyb", close=True, dxfattribs={"layer": layer})


def _ring(wire: bd.Wire) -> np.ndarray:
    """The wire as a closed polygon (n, 2), arcs and curves sampled."""
    pts = []
    for edge in wire.order_edges():
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


def label_point(face: bd.Face) -> tuple[float, float, float]:
    """(x, y, rotation_deg) for the label: inside the material, as far from the outline as the grid allows,
    along the face's longer side (level when the sides are equal). Falls back to the face centre if no grid
    point lies in the material (only a part thinner than the grid pitch everywhere)."""
    box = face.bounding_box()
    xs = np.linspace(box.min.X, box.max.X, LABEL_GRID + 2)[1:-1]
    ys = np.linspace(box.min.Y, box.max.Y, LABEL_GRID + 2)[1:-1]
    grid = np.array([(x, y) for y in ys for x in xs])
    a, b = _segments(face)
    inside = _inside(grid, a, b)
    rotation = QUARTER_TURN_DEG if box.size.Y > box.size.X + EPS_M else 0.0  # round parts read level
    if not inside.any():
        centre = face.center()
        return centre.X, centre.Y, rotation
    candidates = grid[inside]
    best = candidates[int(np.argmax(_clearance(candidates, a, b)))]
    return float(best[0]), float(best[1]), rotation


def _add_part(msp: Modelspace, placement: Placement, kerf_m: float) -> None:
    face = placed_outline(placement, kerf_m)
    _add_wire(msp, face.outer_wire(), LAYER_CUT_OUTER)
    for wire in face.inner_wires():
        _add_wire(msp, wire, LAYER_CUT_INNER)
    x, y, rotation = label_point(face)
    text = msp.add_text(placement.part.label, height=LABEL_HEIGHT_M * MM_PER_M, rotation=rotation,
                        dxfattribs={"layer": LAYER_ENGRAVE})
    text.set_placement((x * MM_PER_M, y * MM_PER_M), align=TextEntityAlignment.MIDDLE_CENTER)


def _write_sheet(placements: Sequence[Placement], kerf_m: float, path: Path) -> Path:
    doc = ezdxf.new(DXF_VERSION)
    doc.units = units.MM
    doc.header["$MEASUREMENT"] = METRIC_MEASUREMENT
    for name, colour in LAYER_COLOURS.items():
        doc.layers.add(name, color=colour)
    msp = doc.modelspace()
    for placement in placements:
        _add_part(msp, placement, kerf_m)
    doc.saveas(path)
    return path


def write_sheets(placements: Mapping[str, Sequence[Placement]], settings: FabricationSettings,
                 out_dir: Path) -> tuple[Path, ...]:
    """One DXF per nested sheet under `out_dir/kit/sheets/`; returns the paths in stock, then sheet order."""
    sheet_dir = out_dir.joinpath(*SHEETS_SUBDIR)
    sheet_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for stock, stock_placements in placements.items():
        for n in sorted({p.sheet for p in stock_placements}):
            on_sheet = [p for p in stock_placements if p.sheet == n]
            paths.append(_write_sheet(on_sheet, settings.kerf_m, sheet_dir / f"{_slug(stock)}-{n}.dxf"))
    return tuple(paths)
