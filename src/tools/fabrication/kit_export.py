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
- the kerf offset is computed once per part label, however many copies are placed; two different parts sharing a
  label raise KitUnavailable, as does a failed offset (naming the part, as `nesting.nest` does).

`export_kit` nests the kit, writes those sheets and, beside them, `kit/build_sheet.md` (sections in
BUILD_SHEET_SECTIONS order; its Summary kit mass is `kit_mass_kg`, tubes and overwrap included) and `kit/parts.csv`
(one row per sheet part, PARTS_CSV_COLUMNS; its masses sum to the parts' `part_item` masses). `kit_solids` gives the
kit as 3D solids in the aircraft frame: every sheet part (`ply_solids`, outline extruded by its sheet thickness, moved
by its location; one solid per copy), then every tube and the two overwrap halves (`carbon_solids`, hollow cylinders;
a rod is solid).
"""

import csv
import re
from collections import defaultdict
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
from src.tools.fabrication.fuselage_parts import BOSS_LABELS, PIN_BOSS_LABELS
from src.tools.fabrication.kit import OVERWRAP_LABEL, Kit, KitPart, KitUnavailable, LinearKind, TubePart
from src.tools.fabrication.kit_mass import GLUE_NAME, kit_mass_items, part_item
from src.tools.fabrication.nesting import EPS_M, MM_PER_M, QUARTER_TURN_DEG, Placement, kerfed_part, nest, place_kerfed
from src.tools.fabrication.wing_parts import ROOT_PIN_BOSS_LABEL

__all__ = ["BUILD_SHEET_SECTIONS", "KitFiles", "LABEL_HEIGHT_M", "LABEL_MIN_HEIGHT_M", "LabelLayout",
           "PARTS_CSV_COLUMNS", "build_sheet_text", "carbon_solids", "export_kit", "kit_solids", "label_layout",
           "ply_solids", "remove_kit_files", "write_sheets"]

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
KIT_SUBDIR: Final[str] = "kit"
SHEETS_SUBDIR: Final[tuple[str, str]] = (KIT_SUBDIR, "sheets")
BUILD_SHEET_FILE: Final[str] = "build_sheet.md"
PARTS_CSV_FILE: Final[str] = "parts.csv"
BUILD_SHEET_SECTIONS: Final[tuple[str, ...]] = ("Summary", "Parts", "Tubes", "Linear stock", "Joins",
                                                "Centreline joint", "Covering", "Hardware")
PARTS_CSV_COLUMNS: Final[tuple[str, ...]] = ("label", "family", "stock", "thickness_mm", "quantity", "area_mm2",
                                             "mass_g", "sheet")
G_PER_KG: Final[float] = 1000.0
MM2_PER_M2: Final[float] = 1e6
CSV_MASS_DECIMALS: Final[int] = 6  # grams: rounding stays far below the 1e-6 kg mass identity
CSV_AREA_DECIMALS: Final[int] = 3
SHEET_LIST_SEPARATOR: Final[str] = ";"
OVERWRAP_CLOTH: Final[str] = "200 g/m² carbon"  # the cloth `overwrap_layer_thickness_m` describes
LAY_UP_STEPS: Final[tuple[str, ...]] = (
    "Insert the sockets through the side plates (`WM-R`, `WM-L`) so they meet at the centreline.",
    "Glue the mitre where the two sockets meet.",
    "Wrap between the plates: {layers} layers of cloth over the mitre, lapping {lap_mm:.1f} mm onto each socket.",
)
LEFT_ROOT_PIN_BOSS_LABEL: Final[str] = ROOT_PIN_BOSS_LABEL.replace("WR-", "WL-")  # the mirrored panel's boss
BOSS_ROLES: Final[dict[str, str]] = {
    BOSS_LABELS[1]: "socket boss, inner face of `WM-R`",
    BOSS_LABELS[-1]: "socket boss, inner face of `WM-L`",
    PIN_BOSS_LABELS[0]: "pin boss, inner face of `WM-R`",
    PIN_BOSS_LABELS[1]: "pin boss, inner face of `WM-L`",
    ROOT_PIN_BOSS_LABEL: "pin boss, outboard face of the right root rib",
    LEFT_ROOT_PIN_BOSS_LABEL: "pin boss, outboard face of the left root rib",
}
FIXED_HARDWARE: Final[tuple[str, ...]] = ("2 nylon wing bolts M5", "2 captive-nut plates")
WING_CONNECTORS: Final[str] = "2 wing connectors"
LINEAR_KINDS: Final[tuple[LinearKind, ...]] = ("stringer", "le", "te")


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


def _parts_by_label(placements: Sequence[Placement]) -> dict[str, KitPart]:
    """Each label's part (copies share one); the kerf cache is keyed by label, so a label must name one part."""
    parts: dict[str, KitPart] = {}
    for placement in placements:
        known = parts.setdefault(placement.part.label, placement.part)
        if known is not placement.part and known != placement.part:
            raise KitUnavailable(f"two different kit parts share the label {placement.part.label!r}")
    return parts


def write_sheets(placements: Mapping[str, Sequence[Placement]], settings: FabricationSettings,
                 out_dir: Path) -> tuple[Path, ...]:
    """One DXF per nested sheet under `out_dir/kit/sheets/`; returns the paths in stock, then sheet order.
    Any `*.dxf` already in that folder (an earlier run's sheets) is deleted first; nothing else there is touched."""
    sheet_dir = out_dir.joinpath(*SHEETS_SUBDIR)
    sheet_dir.mkdir(parents=True, exist_ok=True)
    for stale in sorted(sheet_dir.glob("*.dxf")):
        stale.unlink()
    parts = _parts_by_label([p for stock_placements in placements.values() for p in stock_placements])
    kerfed = {label: kerfed_part(part, settings.kerf_m) for label, part in parts.items()}
    paths: list[Path] = []
    for stock, stock_placements in placements.items():
        for n in sorted({p.sheet for p in stock_placements}):
            on_sheet = [p for p in stock_placements if p.sheet == n]
            paths.append(_write_sheet(on_sheet, kerfed, sheet_dir / f"{_slug(stock)}-{n}.dxf"))
    return tuple(paths)


# ------------------------------------------------------------------------------------- kit solids ---
def _part_solid(part: KitPart, settings: FabricationSettings) -> bd.Solid:
    """The outline extruded along local +Z by its sheet thickness, moved into the aircraft frame (`kit.py`)."""
    thickness = settings.sheet_for(part.family).thickness_m
    return bd.Solid.extrude(part.outline, bd.Vector(0.0, 0.0, thickness)).moved(part.location)


def _tube_solid(tube: TubePart) -> bd.Solid:
    """A hollow cylinder (a solid one for a rod) from `start_m` along `axis` for `length_m`."""
    plane = bd.Plane(origin=tube.start_m, z_dir=tube.axis)
    outer = bd.Solid.make_cylinder(tube.tube.od_m / 2.0, tube.length_m, plane)
    if tube.tube.id_m <= 0.0:
        return outer
    return outer.cut(bd.Solid.make_cylinder(tube.tube.id_m / 2.0, tube.length_m, plane)).solids()[0]


def ply_solids(kit: Kit, settings: FabricationSettings) -> list[bd.Solid]:
    """Every sheet part as built, in kit order, one solid per copy (copies share their part's location)."""
    solids: list[bd.Solid] = []
    for part in kit.parts:
        solid = _part_solid(part, settings)
        solids.extend(solid for _ in range(part.quantity))
    return solids


def carbon_solids(kit: Kit) -> list[bd.Solid]:
    """Every stock tube and rod in kit order, then the overwrap halves (`Kit.overwrap_halves`)."""
    return [_tube_solid(t) for t in (*kit.tubes, *kit.overwrap_halves())]


def kit_solids(kit: Kit, settings: FabricationSettings) -> list[bd.Solid]:
    """The whole kit in the aircraft frame: `ply_solids` then `carbon_solids`."""
    return [*ply_solids(kit, settings), *carbon_solids(kit)]


# ----------------------------------------------------------------------------- build sheet helpers ---
@dataclass(frozen=True)
class KitFiles:
    kit_dir: Path
    build_sheet: Path
    parts_csv: Path
    sheets: tuple[Path, ...]


def _to_mm(m: float) -> float:
    return m * MM_PER_M


def _g(kg: float) -> float:
    return kg * G_PER_KG


def _sheets_by_label(placements: Mapping[str, Sequence[Placement]]) -> dict[str, tuple[int, ...]]:
    """The sheet numbers (within its stock) every part label's copies landed on."""
    sheets: dict[str, set[int]] = defaultdict(set)
    for stock_placements in placements.values():
        for placement in stock_placements:
            sheets[placement.part.label].add(placement.sheet)
    return {label: tuple(sorted(numbers)) for label, numbers in sheets.items()}


def _table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    lines = [f"| {' | '.join(header)} |", f"|{'|'.join('---' for _ in header)}|"]
    return lines + [f"| {' | '.join(str(c) for c in row)} |" for row in rows]


def _mass_breakdown_kg(kit: Kit, settings: FabricationSettings) -> list[tuple[str, float]]:
    """`kit_mass_items` summed by kind; the rows add up to `kit_mass_kg`."""
    kinds = {**{p.label: "Sheet parts" for p in kit.parts}, **{t.label: "Tubes and rods" for t in kit.tubes},
             **{s.label: "Linear stock" for s in kit.linear}, **{c.label: "Covering" for c in kit.covering},
             OVERWRAP_LABEL: "Centreline overwrap", GLUE_NAME: "Glue and hardware allowance"}
    totals: dict[str, float] = {}
    for item in kit_mass_items(kit, settings):
        totals[kinds[item.name]] = totals.get(kinds[item.name], 0.0) + item.mass_kg
    return list(totals.items())


# -------------------------------------------------------------------------------- build sheet sections ---
def _summary(kit: Kit, settings: FabricationSettings, placements: Mapping[str, Sequence[Placement]]) -> list[str]:
    breakdown = _mass_breakdown_kg(kit, settings)
    total = sum(kg for _, kg in breakdown)
    sheets = [(stock, len({p.sheet for p in ps})) for stock, ps in placements.items()]
    return [f"Kit mass: **{total:.4f} kg** (sheet parts, tubes and rods, linear stock, the centreline overwrap, "
            "covering, and the glue and hardware allowance).", "",
            *_table(("Item", "Mass (g)"), [(name, f"{_g(kg):.2f}") for name, kg in breakdown]), "",
            *_table(("Stock", "Sheets"), sheets)]


def _parts(kit: Kit, settings: FabricationSettings) -> list[str]:
    """One row per sheet part; the station is the part's centroid in the aircraft frame (x aft, y right, z up)."""
    rows = []
    for part in kit.parts:
        x, y, z = (_to_mm(c) for c in part_item(part, settings).cg_m)
        rows.append((part.label, part.family, settings.sheet_for(part.family).name, part.quantity,
                     f"{x:.0f}, {y:.0f}, {z:.0f}"))
    return _table(("Label", "Family", "Stock", "Qty", "Station x, y, z (mm)"), rows)


def _tubes(kit: Kit) -> list[str]:
    rows = [(t.label, t.role, f"{_to_mm(t.tube.od_m):.1f} × {_to_mm(t.tube.id_m):.1f}",
             f"{_to_mm(t.length_m):.1f}") for t in kit.tubes]
    return ["Carbon tube and rod stock (an inner diameter of 0 is a solid rod).", "",
            *_table(("Label", "Role", "OD × ID (mm)", "Length (mm)"), rows)]


def _section_text(kind: LinearKind, settings: FabricationSettings) -> str:
    if kind == "stringer":
        side = _to_mm(settings.stringer_side_m)
        return f"{side:.1f} × {side:.1f} mm"
    if kind == "le":
        side = _to_mm(settings.le_side_m)
        return f"{side:.1f} × {side:.1f} mm"
    return f"{_to_mm(settings.te_height_m):.1f} × {_to_mm(settings.te_chord_m):.1f} mm"


def _linear(kit: Kit, settings: FabricationSettings) -> list[str]:
    rows = []
    for kind in LINEAR_KINDS:
        strips = [s for s in kit.linear if s.kind == kind]
        if strips:
            mass = sum(s.section_area_m2 * s.length_m * s.density_kg_m3 for s in strips)
            rows.append((kind, _section_text(kind, settings), f"{_to_mm(sum(s.length_m for s in strips)):.0f}",
                         len(strips), f"{_g(mass):.1f}"))
    return _table(("Kind", "Section", "Total length (mm)", "Pieces", "Mass (g)"), rows)


def _joins(kit: Kit) -> list[str]:
    return [f"- {join}" for join in kit.joins] if kit.joins else ["No glued joins beyond the slot-and-tab fits."]


def _centreline_joint(kit: Kit, settings: FabricationSettings) -> list[str]:
    wrap = kit.overwrap
    lap_mm = _to_mm(wrap.lap_m)
    values = [("Cloth", OVERWRAP_CLOTH), ("Layers", f"{wrap.layers} layers"),
              ("Layer thickness", f"{_to_mm(wrap.layer_thickness_m):.2f} mm"), ("Lap", f"{lap_mm:.1f} mm per side"),
              ("Wrapped OD", f"{_to_mm(wrap.od_m):.1f} mm"), ("Mass", f"{_g(wrap.mass_kg):.1f} g")]
    steps = [f"{n}. {step.format(layers=wrap.layers, lap_mm=lap_mm)}" for n, step in enumerate(LAY_UP_STEPS, 1)]
    labels = {p.label for p in kit.parts}
    bosses = [(label, role) for label, role in BOSS_ROLES.items() if label in labels]
    doublers = (_table(("Label", "Doubler"), bosses) if bosses
                else ["No bosses: the plates and root ribs bear the socket and pin loads alone."])
    return ["Wet-laid carbon overwrap over the socket mitre at the centreline.", "",
            *_table(("Item", "Value"), values), "", "Lay-up order:", "", *steps, "",
            "Bosses (glue before inserting the sockets and pins):", "", *doublers]


def _covering(kit: Kit, settings: FabricationSettings) -> list[str]:
    film, cover = kit.film_area_m2, kit.fuselage_cover_area_m2
    return [f"- Film (both faces of the wing and tails): {film:.3f} m², {_g(film * settings.film_kg_m2):.1f} g",
            f"- Fuselage cover: {cover:.3f} m², {_g(cover * settings.fuselage_cover_kg_m2):.1f} g"]


def _hardware(kit: Kit) -> list[str]:
    pins = f"2 × {round(_to_mm(kit.spar.pin.rod_m), 1):g} mm anti-rotation pins"
    return [f"- {item}" for item in (*FIXED_HARDWARE, pins, WING_CONNECTORS)]


def build_sheet_text(kit: Kit, settings: FabricationSettings, placements: Mapping[str, Sequence[Placement]]) -> str:
    """The build sheet as Markdown, sections in BUILD_SHEET_SECTIONS order."""
    bodies = (_summary(kit, settings, placements), _parts(kit, settings), _tubes(kit), _linear(kit, settings),
              _joins(kit), _centreline_joint(kit, settings), _covering(kit, settings), _hardware(kit))
    lines = ["# Laser-cut kit build sheet", ""]
    for heading, body in zip(BUILD_SHEET_SECTIONS, bodies, strict=True):
        lines += [f"## {heading}", "", *body, ""]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------- export ---
def _write_parts_csv(kit: Kit, settings: FabricationSettings, placements: Mapping[str, Sequence[Placement]],
                     path: Path) -> Path:
    sheets = _sheets_by_label(placements)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(PARTS_CSV_COLUMNS)
        for part in kit.parts:
            sheet = settings.sheet_for(part.family)
            writer.writerow((part.label, part.family, sheet.name, f"{_to_mm(sheet.thickness_m):g}", part.quantity,
                             f"{part.area_m2 * MM2_PER_M2:.{CSV_AREA_DECIMALS}f}",
                             f"{_g(part_item(part, settings).mass_kg):.{CSV_MASS_DECIMALS}f}",
                             SHEET_LIST_SEPARATOR.join(str(n) for n in sheets.get(part.label, ()))))
    return path


def remove_kit_files(out_dir: Path) -> None:
    """Delete an earlier run's build sheet, parts list and sheet DXFs, so a failed or kit-less run leaves none."""
    kit_dir = out_dir / KIT_SUBDIR
    stale_sheets = sorted(out_dir.joinpath(*SHEETS_SUBDIR).glob("*.dxf"))
    for path in (kit_dir / BUILD_SHEET_FILE, kit_dir / PARTS_CSV_FILE, *stale_sheets):
        path.unlink(missing_ok=True)


def export_kit(kit: Kit, settings: FabricationSettings, out_dir: Path) -> KitFiles:
    """Nest the kit and write its DXF sheets, `kit/build_sheet.md` and `kit/parts.csv` under `out_dir`. Any earlier
    kit files are removed first; a nesting or DXF failure raises (KitUnavailable for the known ones)."""
    remove_kit_files(out_dir)
    placements = nest(kit.parts, settings)
    sheets = write_sheets(placements, settings, out_dir)
    kit_dir = out_dir / KIT_SUBDIR
    build_sheet = kit_dir / BUILD_SHEET_FILE
    build_sheet.write_text(build_sheet_text(kit, settings, placements), encoding="utf-8")
    parts_csv = _write_parts_csv(kit, settings, placements, kit_dir / PARTS_CSV_FILE)
    return KitFiles(kit_dir, build_sheet, parts_csv, sheets)
