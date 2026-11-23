"""DXF cut files for the nested laser-cut kit: one file per sheet, millimetres.

`write_sheets` writes `<out_dir>/kit/sheets/<stock-name-slug>-<n>.dxf` (n = 1-based sheet number) for every
sheet `nesting.nest` filled. Each placement's kerf-compensated outline is drawn as:
- outer wire on layer `CUT_OUTER`, inner wires (holes, slots) on `CUT_INNER`;
- a wire that is one full circle as a DXF `CIRCLE`; any other wire as one closed `LWPOLYLINE` with exact bulge
  arcs for circular edges (lines and arcs are all the kit outlines contain), and any other curve flattened to
  CURVE_SAMPLES points per edge. Closed polylines and circles are what laser software imports most reliably;
- the part label as open single-stroke polylines on layer `ENGRAVE`, gathered in one DXF group whose description
  is the label (`kit_labels`: placed and shrunk to fit by `label_layout`, drawn by `label_strokes`). No TEXT
  entities: laser software drops them on import.
- the kerf offset is computed once per part label, however many copies are placed; two different parts sharing a
  label raise KitUnavailable, as does a failed offset (naming the part, as `nesting.nest` does).

`export_kit` nests the kit, writes those sheets and, beside them, `kit/build_sheet.md` (sections in
BUILD_SHEET_SECTIONS order; its Summary kit mass is `kit_mass_kg`, tubes and overwrap included, then any build notes)
and `kit/parts.csv` (one row per sheet part, PARTS_CSV_COLUMNS; its masses sum to the parts' `part_item` masses).
`kit_solids` gives the kit as 3D solids in the aircraft frame: every sheet part (`ply_solids`, outline extruded by its
sheet thickness, moved by its location; one solid per part), then every tube and the two overwrap halves
(`carbon_solids`, hollow cylinders; a rod is solid); `kit_solid_sets` keeps the two apart so one export builds them
once for STEP and the viewer.
"""

import csv
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping, Optional, Sequence

import build123d as bd
import ezdxf
import numpy as np
from ezdxf import colors, units
from ezdxf.document import Drawing
from ezdxf.layouts import Modelspace

from src.schemas.fabrication import G_PER_KG, FabricationSettings
from src.schemas.kit_files import (
    BUILD_SHEET_FILE,
    KIT_SUBDIR,
    PARTS_CSV_FILE,
    SHEET_GLOB,
    SHEETS_SUBDIR,
    sheet_file_name,
)
from src.tools.fabrication.kit import OVERWRAP_LABEL, Kit, KitPart, KitUnavailable, LinearKind, TubePart, part_solid, tube_solid
from src.tools.fabrication.kit_labels import CURVE_SAMPLES, label_layout, label_strokes, ordered_edges
from src.tools.aero_cad import mass_properties as mp
from src.tools.fabrication.kit_mass import GLUE_NAME, areal_density_kg_m2, kit_mass_items
from src.tools.fabrication.nesting import MM_PER_M, Placement, kerfed_part, nest, place_kerfed
from src.tools.fabrication.wing_box_bosses import BOSS_LABELS, PIN_BOSS_LABELS
from src.tools.fabrication.wing_parts import ROOT_PIN_BOSS_LABEL
from src.tools.fabrication.wing_retention import BOLT_CHORD_FRACTION
from src.tools.fabrication.wing_stiffness import REVERSAL_MARGIN, TIP_DEFLECTION_LOAD_FACTOR

__all__ = ["BUILD_SHEET_SECTIONS", "KitFiles", "KitSolids", "PARTS_CSV_COLUMNS", "build_sheet_text", "carbon_solids",
           "export_kit", "kit_solid_sets", "kit_solids", "ply_solids", "remove_kit_files", "write_sheets"]

logger = logging.getLogger(__name__)

LAYER_CUT_OUTER: Final[str] = "CUT_OUTER"
LAYER_CUT_INNER: Final[str] = "CUT_INNER"
LAYER_ENGRAVE: Final[str] = "ENGRAVE"
LAYER_COLOURS: Final[dict[str, int]] = {LAYER_CUT_OUTER: colors.RED, LAYER_CUT_INNER: colors.BLUE,
                                        LAYER_ENGRAVE: colors.BLACK}
DXF_VERSION: Final[str] = "R2010"
METRIC_MEASUREMENT: Final[int] = 1  # $MEASUREMENT: 1 = metric
BUILD_SHEET_SECTIONS: Final[tuple[str, ...]] = ("Summary", "Parts", "Tubes", "Linear stock", "Joins",
                                                "Centreline joint", "Covering", "Wing stiffness", "Hardware")
PARTS_CSV_COLUMNS: Final[tuple[str, ...]] = ("label", "family", "stock", "thickness_mm", "quantity", "area_mm2",
                                             "mass_g", "sheet")
MM2_PER_M2: Final[float] = 1e6
CSV_MASS_DECIMALS: Final[int] = 6  # grams: rounding stays far below the 1e-6 kg mass identity
CSV_AREA_DECIMALS: Final[int] = 3
SHEET_LIST_SEPARATOR: Final[str] = ";"
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
WING_NUTS: Final[str] = "2 M5 nuts, one captive in each nut plate (`NP-R`, `NP-L`)"
WING_CONNECTORS: Final[str] = "2 wing connectors"
RETENTION_STEPS: Final[tuple[str, ...]] = (
    "Glue `NP-R` to the outboard face of `WR-01` (and `NP-L` to `WL-01`) over the bolt hole, press an M5 nut into "
    "its hexagon and fix it with a drop of epoxy: the nut is captive.",
    "Drill the fuselage cover through the bolt holes of `WM-R` and `WM-L` once the cover is on.",
    "Plug each wing's spar into its socket with the pin in its holes.",
    "From inside the fuselage, put a washer on the bolt and drive it through `WM-R` (or `WM-L`), the cover and "
    "`WR-01` (or `WL-01`) into the captive nut; tighten by hand, it is nylon.",
)
DBOX_NOTE: Final[str] = (
    "Leading-edge D-box (R45): glue {t} mm balsa sheet, grain spanwise, over the inset rib noses on the top and "
    "bottom of each panel, from the leading-edge stock back to the step in the ribs ahead of the spar. Butt the "
    "sheets onto the leading-edge stock, which closes the box. Glue a {t} mm balsa shear web, grain vertical, across "
    "each rib bay at that step. Sand the leading-edge stock and the sheet edges to blend before covering."
)
LINEAR_KINDS: Final[tuple[LinearKind, ...]] = ("stringer", "le", "te")


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
    edges = ordered_edges(wire)
    if _is_circle(edges):
        msp.add_circle(_mm(edges[0].arc_center), edges[0].radius * MM_PER_M, dxfattribs={"layer": layer})
        return
    points = [(*_mm(p), bulge) for edge in edges for p, bulge in _edge_points(edge)]
    msp.add_lwpolyline(points, format="xyb", close=True, dxfattribs={"layer": layer})


def _add_label(doc: Drawing, face: bd.Face, text: str) -> None:
    """The label as open single-stroke polylines on ENGRAVE, gathered in one DXF group described by the label."""
    msp = doc.modelspace()
    strokes = label_strokes(text, label_layout(face, text))
    entities = [msp.add_lwpolyline([(x * MM_PER_M, y * MM_PER_M) for x, y in stroke], close=False,
                                   dxfattribs={"layer": LAYER_ENGRAVE}) for stroke in strokes]
    doc.groups.new(description=text).extend(entities)


def _add_part(doc: Drawing, placement: Placement, kerfed: bd.Face) -> None:
    msp = doc.modelspace()
    face = place_kerfed(kerfed, placement)
    _add_wire(msp, face.outer_wire(), LAYER_CUT_OUTER)
    for wire in face.inner_wires():
        _add_wire(msp, wire, LAYER_CUT_INNER)
    _add_label(doc, face, placement.part.label)


def _write_sheet(placements: Sequence[Placement], kerfed: Mapping[str, bd.Face], path: Path) -> Path:
    doc = ezdxf.new(DXF_VERSION)
    doc.units = units.MM
    doc.header["$MEASUREMENT"] = METRIC_MEASUREMENT
    for name, colour in LAYER_COLOURS.items():
        doc.layers.add(name, color=colour)
    for placement in placements:
        _add_part(doc, placement, kerfed[placement.part.label])
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
    for stale in sorted(sheet_dir.glob(SHEET_GLOB)):
        stale.unlink()
    parts = _parts_by_label([p for stock_placements in placements.values() for p in stock_placements])
    kerfed = {label: kerfed_part(part, settings.kerf_m) for label, part in parts.items()}
    paths: list[Path] = []
    for stock, stock_placements in placements.items():
        for n in sorted({p.sheet for p in stock_placements}):
            on_sheet = [p for p in stock_placements if p.sheet == n]
            paths.append(_write_sheet(on_sheet, kerfed, sheet_dir / sheet_file_name(stock, n)))
    return tuple(paths)


# ------------------------------------------------------------------------------------- kit solids ---
def _part_solid(part: KitPart, settings: FabricationSettings) -> bd.Solid:
    """The outline extruded along local +Z by its sheet thickness, moved into the aircraft frame (`kit.py`)."""
    return part_solid(part, settings)


def _tube_solid(tube: TubePart) -> bd.Solid:
    """A hollow cylinder (a solid one for a rod) from `start_m` along `axis` for `length_m` (`kit.py`)."""
    return tube_solid(tube)


def ply_solids(kit: Kit, settings: FabricationSettings) -> list[bd.Solid]:
    """Every sheet part as built, one solid per part in kit order. A part's copies (`quantity` > 1) have no location of
    their own, so they are not drawn on top of it: its mass (`part_item`) carries the quantity instead."""
    return [_part_solid(part, settings) for part in kit.parts]


def carbon_solids(kit: Kit) -> list[bd.Solid]:
    """Every stock tube and rod in kit order, then the overwrap halves (`Kit.overwrap_halves`)."""
    return [_tube_solid(t) for t in (*kit.tubes, *kit.overwrap_halves())]


@dataclass(frozen=True)
class KitSolids:
    """The kit's solids, built once per export for both the STEP assembly and the viewer."""

    ply: tuple[bd.Solid, ...]
    carbon: tuple[bd.Solid, ...]

    @property
    def all(self) -> list[bd.Solid]:
        return [*self.ply, *self.carbon]


def kit_solid_sets(kit: Kit, settings: FabricationSettings) -> KitSolids:
    """`ply_solids` and `carbon_solids`, kept apart (the viewer shows them as two parts)."""
    return KitSolids(ply=tuple(ply_solids(kit, settings)), carbon=tuple(carbon_solids(kit)))


def kit_solids(kit: Kit, settings: FabricationSettings) -> list[bd.Solid]:
    """The whole kit in the aircraft frame: `ply_solids` then `carbon_solids`."""
    return kit_solid_sets(kit, settings).all


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


def _items_by_name(kit: Kit, settings: FabricationSettings,
                   items: Optional[Sequence[mp.MassItem]]) -> dict[str, mp.MassItem]:
    """The kit's mass items by name (labels are unique), `kit_mass_items` only when the caller has none (M6)."""
    return {item.name: item for item in (kit_mass_items(kit, settings) if items is None else items)}


def _mass_breakdown_kg(kit: Kit, items: Mapping[str, mp.MassItem]) -> list[tuple[str, float]]:
    """The kit's mass items summed by kind; the rows add up to `kit_mass_kg`."""
    kinds = {**{p.label: "Sheet parts" for p in kit.parts}, **{t.label: "Tubes and rods" for t in kit.tubes},
             **{s.label: "Linear stock" for s in kit.linear}, **{c.label: "Covering" for c in kit.covering},
             OVERWRAP_LABEL: "Centreline overwrap", GLUE_NAME: "Glue and hardware allowance"}
    totals: dict[str, float] = {}
    for item in items.values():
        totals[kinds[item.name]] = totals.get(kinds[item.name], 0.0) + item.mass_kg
    return list(totals.items())


# -------------------------------------------------------------------------------- build sheet sections ---
def _summary(kit: Kit, items: Mapping[str, mp.MassItem], placements: Mapping[str, Sequence[Placement]]) -> list[str]:
    breakdown = _mass_breakdown_kg(kit, items)
    total = sum(kg for _, kg in breakdown)
    sheets = [(stock, len({p.sheet for p in ps})) for stock, ps in placements.items()]
    return [f"Kit mass: **{total:.4f} kg** (sheet parts, tubes and rods, linear stock, the centreline overwrap, "
            "covering, and the glue and hardware allowance).", "",
            *_table(("Item", "Mass (g)"), [(name, f"{_g(kg):.2f}") for name, kg in breakdown]), "",
            *_table(("Stock", "Sheets"), sheets), *_notes(kit)]


def _notes(kit: Kit) -> list[str]:
    """The kit's build notes (boom-end motor wiring, folding propeller) under the summary; none for most kits."""
    return ["", *(f"**Note:** {note}." for note in kit.notes)] if kit.notes else []


def _parts(kit: Kit, settings: FabricationSettings, items: Mapping[str, mp.MassItem]) -> list[str]:
    """One row per sheet part; the station is the part's centroid in the aircraft frame (x aft, y right, z up)."""
    rows = []
    for part in kit.parts:
        x, y, z = (_to_mm(c) for c in items[part.label].cg_m)
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
    values = [("Cloth", f"{settings.overwrap_cloth_g_m2:g} g/m² carbon"), ("Layers", f"{wrap.layers} layers"),
              ("Layer thickness", f"{_to_mm(wrap.layer_thickness_m):.2f} mm"), ("Lap", f"{lap_mm:.1f} mm per side"),
              ("Wrapped OD", f"{_to_mm(wrap.od_m):.1f} mm"), ("Mass", f"{_g(wrap.mass_kg):.1f} g")]
    steps = [f"{n}. {step.format(layers=wrap.layers, lap_mm=lap_mm)}" for n, step in enumerate(LAY_UP_STEPS, 1)]
    labels = {p.label for p in kit.parts}
    bosses = [(label, role) for label, role in BOSS_ROLES.items() if label in labels]
    doublers = (_table(("Label", "Doubler"), bosses) if bosses
                else ["No bosses: the plates and root ribs bear the socket and pin loads alone."])
    return ["Wet-laid carbon overwrap over the socket mitre at the centreline.", "",
            *_table(("Item", "Value"), values), "", "Lay-up order:", "", *steps, "",
            "Do not glue the spar roots: the wings must stay detachable.", "",
            "Bosses (glue before inserting the sockets and pins):", "", *doublers]


def _dbox_mass_g(kit: Kit, settings: FabricationSettings) -> float:
    return _g(kit.dbox_area_m2 * areal_density_kg_m2("dbox", settings))


def _covering(kit: Kit, settings: FabricationSettings) -> list[str]:
    film, cover = kit.film_area_m2, kit.fuselage_cover_area_m2
    lines = [f"- Film (both faces of the wing and tails, outside the fuselage): {film:.3f} m², "
             f"{_g(film * settings.film_kg_m2):.1f} g",
             f"- Fuselage cover: {cover:.3f} m², {_g(cover * settings.fuselage_cover_kg_m2):.1f} g"]
    if kit.dbox_area_m2 > 0.0:
        lines.append(f"- Leading-edge D-box (balsa): {kit.dbox_area_m2:.3f} m², {_dbox_mass_g(kit, settings):.1f} g")
    return lines


def _wing_stiffness(kit: Kit, settings: FabricationSettings) -> list[str]:
    """R43-R45: tip deflection and aileron reversal against their limits, and the D-box with its build note."""
    report = kit.spar.stiffness
    if report is None:
        return ["Wing stiffness: not checked"]
    has_dbox = kit.dbox_area_m2 > 0.0  # the covering entry is the one source of truth for area, mass and yes/no
    lines = [f"- Tip deflection at {TIP_DEFLECTION_LOAD_FACTOR:g} g: {_to_mm(report.tip_deflection_m):.1f} mm "
             f"(limit {_to_mm(report.tip_deflection_limit_m):.1f} mm)",
             f"- Aileron reversal speed: {report.reversal_speed_ms:.1f} m/s "
             f"(needs above {report.reversal_required_ms:.1f} m/s = {REVERSAL_MARGIN:.2f} x V_NE)",
             f"- D-box: {'yes' if has_dbox else 'no'}"]
    if has_dbox:
        t = f"{_to_mm(settings.dbox_sheet_thickness_m):g}"
        lines += [f"- D-box area: {kit.dbox_area_m2:.3f} m²", f"- D-box mass: {_dbox_mass_g(kit, settings):.1f} g",
                  "", DBOX_NOTE.format(t=t)]
    return lines


def _hardware(kit: Kit) -> list[str]:
    bolt = kit.wing_bolt
    bolts = f"2 nylon wing bolts M5 × {_to_mm(bolt.length_m):.0f} mm with washers (grip {_to_mm(bolt.grip_m):.1f} mm)"
    pins = f"2 × {round(_to_mm(kit.spar.pin.rod_m), 1):g} mm anti-rotation pins"
    where = (f"Wing bolt (R36), one per side: spanwise on the wing's chord line at {BOLT_CHORD_FRACTION:.1%} chord, "
             f"{_to_mm(bolt.aft_of_spar_m):.0f} mm aft of the spar axis at the root; it crosses `WM-R` at "
             f"x = {_to_mm(bolt.plate_x_m):.0f} mm, z = {_to_mm(bolt.plate_z_m):.0f} mm (`WM-L` mirrored).")
    steps = [f"{n}. {step}" for n, step in enumerate(RETENTION_STEPS, 1)]
    return [*(f"- {item}" for item in (bolts, WING_NUTS, pins, WING_CONNECTORS)), "", where, "", *steps]


def build_sheet_text(kit: Kit, settings: FabricationSettings, placements: Mapping[str, Sequence[Placement]],
                     items: Optional[Sequence[mp.MassItem]] = None) -> str:
    """The build sheet as Markdown, sections in BUILD_SHEET_SECTIONS order; `items` are the kit's mass items when the
    caller already has them (the packing's), else they are computed once here."""
    by_name = _items_by_name(kit, settings, items)
    bodies = (_summary(kit, by_name, placements), _parts(kit, settings, by_name), _tubes(kit), _linear(kit, settings),
              _joins(kit), _centreline_joint(kit, settings), _covering(kit, settings),
              _wing_stiffness(kit, settings), _hardware(kit))
    lines = ["# Laser-cut kit build sheet", ""]
    for heading, body in zip(BUILD_SHEET_SECTIONS, bodies, strict=True):
        lines += [f"## {heading}", "", *body, ""]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------- export ---
def _write_parts_csv(kit: Kit, settings: FabricationSettings, placements: Mapping[str, Sequence[Placement]],
                     items: Mapping[str, mp.MassItem], path: Path) -> Path:
    sheets = _sheets_by_label(placements)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(PARTS_CSV_COLUMNS)
        for part in kit.parts:
            sheet = settings.sheet_for(part.family)
            writer.writerow((part.label, part.family, sheet.name, f"{_to_mm(sheet.thickness_m):g}", part.quantity,
                             f"{part.area_m2 * MM2_PER_M2:.{CSV_AREA_DECIMALS}f}",
                             f"{_g(items[part.label].mass_kg):.{CSV_MASS_DECIMALS}f}",
                             SHEET_LIST_SEPARATOR.join(str(n) for n in sheets.get(part.label, ()))))
    return path


def remove_kit_files(out_dir: Path) -> None:
    """Delete an earlier run's build sheet, parts list and sheet DXFs, so a failed or kit-less run leaves none."""
    kit_dir = out_dir / KIT_SUBDIR
    stale_sheets = sorted(out_dir.joinpath(*SHEETS_SUBDIR).glob(SHEET_GLOB))
    for path in (kit_dir / BUILD_SHEET_FILE, kit_dir / PARTS_CSV_FILE, *stale_sheets):
        path.unlink(missing_ok=True)


def export_kit(kit: Kit, settings: FabricationSettings, out_dir: Path,
               items: Optional[Sequence[mp.MassItem]] = None) -> KitFiles:
    """Nest the kit and write its DXF sheets, `kit/build_sheet.md` and `kit/parts.csv` under `out_dir`, with the
    kit's mass items `items` (the packing's; computed once when omitted). Any earlier kit files are removed first; on
    any failure the files written so far are removed too, then the error is raised (KitUnavailable for the known
    nesting and DXF ones)."""
    remove_kit_files(out_dir)
    try:
        return _write_kit_files(kit, settings, out_dir, _items_by_name(kit, settings, items))
    except Exception:
        _discard_partial_files(out_dir)
        raise


def _write_kit_files(kit: Kit, settings: FabricationSettings, out_dir: Path,
                     items: Mapping[str, mp.MassItem]) -> KitFiles:
    placements = nest(kit.parts, settings)
    sheets = write_sheets(placements, settings, out_dir)
    kit_dir = out_dir / KIT_SUBDIR
    build_sheet = kit_dir / BUILD_SHEET_FILE
    build_sheet.write_text(build_sheet_text(kit, settings, placements, tuple(items.values())), encoding="utf-8")
    parts_csv = _write_parts_csv(kit, settings, placements, items, kit_dir / PARTS_CSV_FILE)
    return KitFiles(kit_dir, build_sheet, parts_csv, sheets)


def _discard_partial_files(out_dir: Path) -> None:
    """Best effort after a failed export: remove whatever was written, so no orphaned sheet set is left. A removal
    failure is logged; the export's own error is what the caller sees."""
    try:
        remove_kit_files(out_dir)
    except OSError:
        logger.exception("could not remove the partial kit files under %s", out_dir / KIT_SUBDIR)
