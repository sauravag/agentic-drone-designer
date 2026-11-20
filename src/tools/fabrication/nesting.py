"""Nest kit parts on bed-sized sheets, kerf-compensated (SI units, metres).

- `kerf_outline`: the laser's centre path. The material region is dilated by kerf/2, so outer outlines grow by
  kerf/2 per side and holes and slots shrink by kerf/2 per side: a slot cut at nominal width ends up exactly one
  kerf narrower, which (with the kerf the beam removes) gives the nominal slot. Convex outer corners become
  kerf/2 arcs (`Kind.ARC`), the true centre path of a round beam; it is also the offset OCCT computes reliably on
  every kit outline (`Kind.INTERSECTION` fails on some tail ribs and crutches).
- `nest`: first-fit decreasing-height shelf packing, per sheet stock, on the kerf-compensated bounding boxes.
  The usable bed is the bed less `margin_m` on every edge, and `margin_m` is also kept between parts. Each part
  is laid at 0 deg or turned 90 deg about its local Z, whichever fits the usable bed with the lower height (0 deg
  on a tie); a part with `quantity` n is placed n times. Deterministic: the same parts give the same sheets.
- `Placement.(x_m, y_m)` is the lower-left corner of the placed kerf-compensated outline's bounding box on the
  sheet (sheet origin at the bed's lower-left corner); `placed_outline` rebuilds that face.
"""

from dataclasses import dataclass, replace
from typing import Final, Sequence

import build123d as bd

from src.schemas.fabrication import FabricationSettings
from src.tools.fabrication.kit import KitPart, KitUnavailable

__all__ = ["Placement", "kerf_outline", "nest", "placed_outline"]

EPS_M: Final[float] = 1e-9  # float slack on bed and shelf fits
QUARTER_TURN_DEG: Final[float] = 90.0
MM_PER_M: Final[float] = 1000.0


@dataclass(frozen=True)
class Placement:
    part: KitPart
    sheet: int  # 1-based sheet number within the part's sheet stock
    x_m: float  # lower-left corner of the kerf-compensated outline's bounding box on the sheet
    y_m: float
    rotated: bool  # turned 90 deg about the part's local Z before placing


@dataclass(frozen=True)
class _Item:
    part: KitPart
    copy: int
    w_m: float  # kerf-compensated footprint in the chosen orientation
    h_m: float
    rotated: bool


@dataclass(frozen=True)
class _Shelf:
    sheet: int
    y_m: float
    h_m: float
    x_next_m: float  # where the next part on this shelf starts


def kerf_outline(face: bd.Face, kerf_m: float) -> bd.Face:
    """The cut path: outer wire offset +kerf/2, inner wires (holes, slots) -kerf/2."""
    if kerf_m <= 0.0:
        return face
    faces = bd.offset(face, kerf_m / 2.0, kind=bd.Kind.ARC).faces()
    if len(faces) != 1:
        raise ValueError(f"kerf offset produced {len(faces)} faces, expected 1")
    return faces[0]


def _oriented(face: bd.Face, rotated: bool) -> bd.Face:
    return bd.Rot(0, 0, QUARTER_TURN_DEG) * face if rotated else face


def _at_corner(face: bd.Face, x_m: float, y_m: float) -> bd.Face:
    box = face.bounding_box()
    return bd.Pos(x_m - box.min.X, y_m - box.min.Y, -box.min.Z) * face


def placed_outline(placement: Placement, kerf_m: float) -> bd.Face:
    """The placement's kerf-compensated outline in sheet coordinates (m)."""
    face = _oriented(kerf_outline(placement.part.outline, kerf_m), placement.rotated)
    return _at_corner(face, placement.x_m, placement.y_m)


def _usable_bed(settings: FabricationSettings) -> tuple[float, float]:
    return settings.bed_width_m - 2.0 * settings.margin_m, settings.bed_height_m - 2.0 * settings.margin_m


def _kerfed(part: KitPart, kerf_m: float) -> bd.Face:
    try:
        return kerf_outline(part.outline, kerf_m)
    except (RuntimeError, ValueError) as exc:
        raise KitUnavailable(f"part {part.label}: cannot kerf-compensate its outline ({exc})") from exc


def _orientation(part: KitPart, settings: FabricationSettings) -> tuple[float, float, bool]:
    """(width, height, rotated) of the kerf-compensated part: the fitting orientation with the lower height."""
    size = _kerfed(part, settings.kerf_m).bounding_box().size
    usable_w, usable_h = _usable_bed(settings)
    options = ((size.X, size.Y, False), (size.Y, size.X, True))
    fitting = [o for o in options if o[0] <= usable_w + EPS_M and o[1] <= usable_h + EPS_M]
    if not fitting:
        raise KitUnavailable(
            f"part {part.label} ({size.X * MM_PER_M:.1f} x {size.Y * MM_PER_M:.1f} mm with kerf) does not fit "
            f"the usable bed ({usable_w * MM_PER_M:.0f} x {usable_h * MM_PER_M:.0f} mm) in either orientation"
        )
    return min(fitting, key=lambda o: (o[1], o[2]))


def _items(parts: Sequence[KitPart], settings: FabricationSettings) -> list[_Item]:
    """Every copy of every part, tallest first (ties: wider, then label, then copy, so the order is fixed)."""
    items: list[_Item] = []
    for part in parts:
        w_m, h_m, rotated = _orientation(part, settings)
        items.extend(_Item(part, copy, w_m, h_m, rotated) for copy in range(part.quantity))
    return sorted(items, key=lambda i: (-i.h_m, -i.w_m, i.part.label, i.copy))


def _new_shelf(item: _Item, shelves: Sequence[_Shelf], settings: FabricationSettings) -> _Shelf:
    """A shelf for `item` above the last shelf of the first sheet with room, else on a fresh sheet."""
    m, top_limit = settings.margin_m, settings.bed_height_m - settings.margin_m
    sheets = sorted({s.sheet for s in shelves})
    for sheet in sheets:
        y_m = max(s.y_m + s.h_m for s in shelves if s.sheet == sheet) + m
        if y_m + item.h_m <= top_limit + EPS_M:
            return _Shelf(sheet, y_m, item.h_m, m)
    return _Shelf(sheets[-1] + 1 if sheets else 1, m, item.h_m, m)


def _place(item: _Item, shelves: tuple[_Shelf, ...], settings: FabricationSettings
           ) -> tuple[tuple[_Shelf, ...], Placement]:
    """First shelf (by sheet, then height on the sheet) with room for `item`; else a new shelf."""
    right_limit = settings.bed_width_m - settings.margin_m
    ordered = sorted(shelves, key=lambda s: (s.sheet, s.y_m))
    shelf = next((s for s in ordered if item.h_m <= s.h_m + EPS_M and s.x_next_m + item.w_m <= right_limit + EPS_M),
                 None)
    if shelf is None:
        shelf = _new_shelf(item, shelves, settings)
        ordered = [*ordered, shelf]
    placement = Placement(item.part, shelf.sheet, shelf.x_next_m, shelf.y_m, item.rotated)
    grown = replace(shelf, x_next_m=shelf.x_next_m + item.w_m + settings.margin_m)
    return tuple(grown if s is shelf else s for s in ordered), placement


def _pack(items: Sequence[_Item], settings: FabricationSettings) -> tuple[Placement, ...]:
    shelves: tuple[_Shelf, ...] = ()
    placements: list[Placement] = []
    for item in items:
        shelves, placement = _place(item, shelves, settings)
        placements.append(placement)
    return tuple(placements)


def nest(parts: Sequence[KitPart], settings: FabricationSettings) -> dict[str, tuple[Placement, ...]]:
    """Lay every part copy on sheets of its stock; keyed by sheet stock name, in `settings.sheets` order."""
    out: dict[str, tuple[Placement, ...]] = {}
    for stock in settings.sheets:
        stock_parts = [p for p in parts if p.family in stock.families]
        if stock_parts:
            out[stock.name] = _pack(_items(stock_parts, settings), settings)
    return out
