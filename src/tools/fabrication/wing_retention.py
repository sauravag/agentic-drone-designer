"""Wing retention (R36): one M5 nylon bolt per side holds each detachable wing on its spar (SI, metres).

The bolt runs spanwise along the wing's chord line at BOLT_CHORD_FRACTION, between the spar (25 % chord) and the
anti-rotation pin (70 %). It is driven from inside the fuselage: its head and washer bear on the wing-box plate's inner
face (`WM-R` / `WM-L`), and it passes through the plate, the fuselage cover and the root doubler (`WR-01` / `WL-01`)
into an M5 nut held captive in a nut plate `NP-R` / `NP-L` (doubler stock, a disc with a hexagonal hole for the nut)
glued to the root doubler's outboard face, inside the wing root.

This module holds the named dimensions and the checks shared by the root rib (`wing_parts`) and the wing-box plates
(`fuselage_parts`); it imports neither, so both can use it. Every check raises `KitUnavailable` naming the part.
"""

import math
from dataclasses import dataclass
from typing import Final, Sequence

import build123d as bd

from src.schemas.fabrication import FabricationSettings
from src.tools.aero_cad.components import COMPONENT_GAP_M, PlacedComponent
from src.tools.fabrication.rib_outline import Hole, circle_face
from src.tools.fabrication.tubes import FIT_EPS_M, MIN_WEB_M, KitUnavailable

BOLT_CHORD_FRACTION: Final[float] = 0.475  # midway between the spar (25 % chord) and the pin (70 % chord)
M5_CLEARANCE_HOLE_M: Final[float] = 0.0055  # ISO 273 medium clearance hole for M5
M5_NUT_ACROSS_FLATS_M: Final[float] = 0.008  # ISO 4032 M5 hex nut
M5_NUT_HEIGHT_M: Final[float] = 0.004
NUT_PLATE_DIAMETER_M: Final[float] = 0.015  # the nut plate disc round the hexagonal nut hole
BOLT_WASHER_DIAMETER_M: Final[float] = 0.010  # M5 washer under the bolt head (on the plate's inner face)
BOLT_WASHER_THICKNESS_M: Final[float] = 0.001
BOLT_HEAD_STACK_M: Final[float] = 0.0045  # head plus washer, standing inboard of the plate's inner face
BOLT_THREAD_MARGIN_M: Final[float] = 0.002  # thread left beyond the nut
BOLT_LENGTH_STEP_M: Final[float] = 0.005  # bolts are sold in 5 mm length steps
NUT_PLATE_LABELS: Final[dict[int, str]] = {1: "NP-R", -1: "NP-L"}
HEX_SIDES: Final[int] = 6
AREA_EPS_M2: Final[float] = 1e-10
WEB_TEXT: Final[str] = f"{MIN_WEB_M * 1e3:.0f} mm"

Box3 = tuple[tuple[float, float, float], tuple[float, float, float]]  # (min corner, max corner)


@dataclass(frozen=True)
class WingBolt:
    """One side's wing bolt for the build sheet: length (a stock step), the grip it must span (plate inner face to
    the far side of the captive nut, plus washer and thread margin) and where it crosses the right wing-box plate."""

    length_m: float
    grip_m: float
    plate_x_m: float
    plate_z_m: float
    aft_of_spar_m: float  # along the root chord, from the spar axis


def bolt_length_m(grip_m: float) -> float:
    """The shortest stock bolt (BOLT_LENGTH_STEP_M steps) at least `grip_m` long."""
    return math.ceil(grip_m / BOLT_LENGTH_STEP_M - FIT_EPS_M) * BOLT_LENGTH_STEP_M


def hex_face(x_m: float, y_m: float, across_flats_m: float) -> bd.Face:
    """A regular hexagon with flats top and bottom (across them `across_flats_m`), centred at (x, y)."""
    radius = across_flats_m / 2.0 / math.cos(math.pi / HEX_SIDES)
    corners = [(x_m + radius * math.cos(2.0 * math.pi * k / HEX_SIDES),
                y_m + radius * math.sin(2.0 * math.pi * k / HEX_SIDES), 0.0) for k in range(HEX_SIDES)]
    return bd.Face(bd.Wire.make_polygon(corners, close=True))


def _area(shape: object) -> float:
    pieces = shape if isinstance(shape, list) else [shape]
    return sum(f.area for piece in pieces if piece is not None for f in piece.faces())


def nut_plate_outline(x_m: float, y_m: float, settings: FabricationSettings, label: str) -> bd.Face:
    """The nut plate: a NUT_PLATE_DIAMETER_M disc at (x, y) with the nut's hexagon (across flats plus the fit
    clearance) cut out, which must keep a MIN_WEB_M web to the rim."""
    disc = circle_face(Hole(x_m, y_m, NUT_PLATE_DIAMETER_M))
    nut = hex_face(x_m, y_m, M5_NUT_ACROSS_FLATS_M + settings.fit_clearance_m)
    if _area(bd.offset(nut, amount=MIN_WEB_M).cut(disc)) > AREA_EPS_M2:
        raise KitUnavailable(f"wing bolt: the nut hole leaves less than {WEB_TEXT} of {label}")
    return disc.cut(nut).faces()[0]


def check_clear_of_holes(disc: Hole, holes: Sequence[Hole], label: str, what: str) -> None:
    """The circle `disc` keeps MIN_WEB_M from every circle in `holes` (rods, rings, holes)."""
    for hole in holes:
        gap = math.hypot(disc.x_m - hole.x_m, disc.y_m - hole.y_m) - (disc.diameter_m + hole.diameter_m) / 2.0
        if gap < MIN_WEB_M - FIT_EPS_M:
            raise KitUnavailable(f"wing bolt: {label} comes within {WEB_TEXT} of {what}")


def check_clear_of_outlines(disc: Hole, outlines: Sequence[bd.Face], label: str, what: str) -> None:
    """The circle `disc` (the washer under the bolt head) keeps MIN_WEB_M from every face in `outlines` (bosses)."""
    zone = circle_face(Hole(disc.x_m, disc.y_m, disc.diameter_m + 2.0 * MIN_WEB_M))
    if any(_area(zone.intersect(face)) > AREA_EPS_M2 for face in outlines):
        raise KitUnavailable(f"wing bolt: the bolt seat on {label} comes within {WEB_TEXT} of {what}")


def check_head_clear(components: Sequence[PlacedComponent], head: Box3, label: str) -> None:
    """No component box comes within half the component gap of the bolt head's box (inside the fuselage)."""
    gap = COMPONENT_GAP_M / 2.0
    lo, hi = head
    for c in components:
        c_lo = [c.center_m[k] - c.size_m[k] / 2.0 - gap for k in range(3)]
        c_hi = [c.center_m[k] + c.size_m[k] / 2.0 + gap for k in range(3)]
        if all(c_lo[k] < hi[k] and c_hi[k] > lo[k] for k in range(3)):
            raise KitUnavailable(f"wing bolt: the {c.name} blocks the bolt head inside {label}")
