"""Nest kit parts on bed-sized sheets, kerf-compensated (SI units, metres).

- `kerf_outline`: the laser's centre path. The material region is dilated by kerf/2, so outer outlines grow by
  kerf/2 per side and holes and slots shrink by kerf/2 per side: a slot cut at nominal width ends up exactly one
  kerf narrower, which (with the kerf the beam removes) gives the nominal slot. Convex outer corners become
  kerf/2 arcs (`Kind.ARC`), the true centre path of a round beam; it is also the offset OCCT computes reliably on
  every kit outline (`Kind.INTERSECTION` fails on some tail ribs and crutches).
- `nest`: MaxRects packing (bottom-left rule), per sheet stock, on the kerf-compensated bounding boxes, parts
  taken tallest first. Unlike shelf packing it fills the strip beside a tall part (g1's 3 mm stock: 2 sheets,
  not 3). The usable bed is the bed less `margin_m` on every edge, and `margin_m` is also kept between parts.
  Each part is laid at 0 deg or turned 90 deg about its local Z: a part opening a sheet takes the orientation
  that fits the usable bed with the lower height (0 deg on a tie); on a part-filled sheet either orientation may
  take the free spot with the lowest top edge. A part with `quantity` n is placed n times. Deterministic: the
  same parts give the same sheets.
- `Placement.(x_m, y_m)` is the lower-left corner of the placed kerf-compensated outline's bounding box on the
  sheet (sheet origin at the bed's lower-left corner); `placed_outline` rebuilds that face.
"""

from dataclasses import dataclass
from typing import Final, Optional, Sequence

import build123d as bd

from src.schemas.fabrication import FabricationSettings
from src.tools.fabrication.kit import KitPart, KitUnavailable

__all__ = ["MM_PER_M", "Placement", "kerf_outline", "kerfed_part", "nest", "place_kerfed", "placed_outline"]

EPS_M: Final[float] = 1e-9  # float slack on bed and free-rectangle fits
QUARTER_TURN_DEG: Final[float] = 90.0
MM_PER_M: Final[float] = 1000.0

Rect = tuple[float, float, float, float]  # (x, y, width, height) on a sheet (m)


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


def place_kerfed(kerfed: bd.Face, placement: Placement) -> bd.Face:
    """A part's kerf-compensated outline (from `kerf_outline`) moved to its placement, in sheet coordinates (m)."""
    return _at_corner(_oriented(kerfed, placement.rotated), placement.x_m, placement.y_m)


def placed_outline(placement: Placement, kerf_m: float) -> bd.Face:
    """The placement's kerf-compensated outline in sheet coordinates (m)."""
    return place_kerfed(kerf_outline(placement.part.outline, kerf_m), placement)


def kerfed_part(part: KitPart, kerf_m: float) -> bd.Face:
    """The part's kerf-compensated outline; a failed offset raises KitUnavailable naming the part."""
    try:
        return kerf_outline(part.outline, kerf_m)
    except (RuntimeError, ValueError) as exc:
        raise KitUnavailable(f"part {part.label}: cannot kerf-compensate its outline ({exc})") from exc


def _orientation(part: KitPart, settings: FabricationSettings) -> tuple[float, float, bool]:
    """(width, height, rotated) of the kerf-compensated part: the fitting orientation with the lower height."""
    size = kerfed_part(part, settings.kerf_m).bounding_box().size
    usable_w, usable_h = settings.usable_bed_m
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


def _overlaps(free: Rect, used: Rect) -> bool:
    fx, fy, fw, fh = free
    ux, uy, uw, uh = used
    return ux < fx + fw - EPS_M and ux + uw > fx + EPS_M and uy < fy + fh - EPS_M and uy + uh > fy + EPS_M


def _split(free: Rect, used: Rect) -> list[Rect]:
    """The maximal parts of `free` left uncovered by `used` (MaxRects split)."""
    if not _overlaps(free, used):
        return [free]
    fx, fy, fw, fh = free
    ux, uy, uw, uh = used
    pieces = [(fx, fy, ux - fx, fh), (ux + uw, fy, fx + fw - ux - uw, fh),
              (fx, fy, fw, uy - fy), (fx, uy + uh, fw, fy + fh - uy - uh)]
    return [r for r in pieces if r[2] > EPS_M and r[3] > EPS_M]


def _contains(outer: Rect, inner: Rect) -> bool:
    return (inner[0] >= outer[0] - EPS_M and inner[1] >= outer[1] - EPS_M
            and inner[0] + inner[2] <= outer[0] + outer[2] + EPS_M
            and inner[1] + inner[3] <= outer[1] + outer[3] + EPS_M)


def _occupy(free: tuple[Rect, ...], used: Rect) -> tuple[Rect, ...]:
    """Free rectangles after placing `used`: split every overlapped one, then drop those inside another (of two
    equal within EPS_M, the first is kept)."""
    pieces = sorted(set(r for f in free for r in _split(f, used)))

    def redundant(i: int, r: Rect) -> bool:
        return any(j != i and _contains(o, r) and (j < i or not _contains(r, o)) for j, o in enumerate(pieces))

    return tuple(r for i, r in enumerate(pieces) if not redundant(i, r))


def _fresh_sheet(settings: FabricationSettings) -> tuple[Rect, ...]:
    """The bed above and right of the first margin. Parts are inflated by `margin_m` (right and top), so this
    keeps `margin_m` to every bed edge and between parts."""
    m = settings.margin_m
    return ((m, m, settings.bed_width_m - m, settings.bed_height_m - m),)


def _best_spot(item: _Item, sheets: Sequence[tuple[Rect, ...]], m: float) -> Optional[tuple[int, Rect, bool]]:
    """(sheet index, inflated footprint, rotated): the lowest top edge, then leftmost, on the earliest sheet;
    the item's preferred orientation wins a tie."""
    best: Optional[tuple[tuple, tuple[int, Rect, bool]]] = None
    turns = ((item.w_m, item.h_m, item.rotated), (item.h_m, item.w_m, not item.rotated))
    for index, free in enumerate(sheets):
        for fx, fy, fw, fh in free:
            for w_m, h_m, rotated in turns:
                if w_m + m <= fw + EPS_M and h_m + m <= fh + EPS_M:
                    key = (index, fy + h_m, fx, rotated != item.rotated)
                    if best is None or key < best[0]:
                        best = (key, (index, (fx, fy, w_m + m, h_m + m), rotated))
        if best is not None:
            return best[1]
    return None


def _pack(items: Sequence[_Item], settings: FabricationSettings) -> tuple[Placement, ...]:
    """MaxRects, bottom-left rule, over sheets opened as needed (first fit by sheet)."""
    m = settings.margin_m
    sheets: tuple[tuple[Rect, ...], ...] = ()
    placements: list[Placement] = []
    for item in items:
        spot = _best_spot(item, sheets, m)
        if spot is None:
            sheets = (*sheets, _fresh_sheet(settings))
            spot = (len(sheets) - 1, (m, m, item.w_m + m, item.h_m + m), item.rotated)
        index, used, rotated = spot
        sheets = tuple(_occupy(free, used) if i == index else free for i, free in enumerate(sheets))
        placements.append(Placement(item.part, index + 1, used[0], used[1], rotated))
    return tuple(placements)


def nest(parts: Sequence[KitPart], settings: FabricationSettings) -> dict[str, tuple[Placement, ...]]:
    """Lay every part copy on sheets of its stock; keyed by sheet stock name, in `settings.sheets` order."""
    out: dict[str, tuple[Placement, ...]] = {}
    for stock in settings.sheets:
        stock_parts = [p for p in parts if p.family in stock.families]
        if stock_parts:
            out[stock.name] = _pack(_items(stock_parts, settings), settings)
    return out
