"""Flat rib outlines for the laser-cut kit: airfoil, LE/TE stock notches, holes and lightening holes.

Outlines live in the rib's local XY plane (metres): x aft along the chord from the leading edge,
y in the airfoil thickness direction. Outlines are nominal (no kerf)
and follow the airfoil exactly: the iron-on film covering adds no thickness.
"""

import math
from dataclasses import dataclass
from typing import Final, Iterable, Optional, Sequence

import build123d as bd

from src.schemas.fabrication import FabricationSettings
from src.tools.aero_cad.airfoil_geometry import AirfoilShape, camber_at
from src.tools.fabrication.tubes import FIT_EPS_M, MIN_WEB_M, KitUnavailable

MIN_LIGHTENING_DIAMETER_M: Final[float] = 0.006  # smaller holes save too little to be worth cutting
CUTTER_REACH_FACTOR: Final[float] = 2.0  # cutting rectangles extend this many chords past the rib


@dataclass(frozen=True)
class Hole:
    x_m: float
    y_m: float
    diameter_m: float


def rect_face(x0: float, y0: float, x1: float, y1: float) -> bd.Face:
    corners = [(x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0)]
    return bd.Face(bd.Wire.make_polygon(corners, close=True))


def circle_face(hole: Hole) -> bd.Face:
    circle = bd.Face(bd.Wire([bd.Edge.make_circle(hole.diameter_m / 2.0)]))
    return circle.moved(bd.Location((hole.x_m, hole.y_m, 0.0)))


def largest_face(shape: bd.Shape) -> bd.Face:
    """The biggest face of a boolean result (slivers cut free by a notch are dropped)."""
    faces = shape.faces()
    if not faces:
        raise KitUnavailable("a rib outline was cut away entirely")
    return max(faces, key=lambda f: f.area)


def camber_point(airfoil: AirfoilShape, chord_m: float, x_over_c: float) -> tuple[float, float]:
    """Local (x, y) of the mean line at chordwise fraction `x_over_c`."""
    return x_over_c * chord_m, camber_at(airfoil, x_over_c) * chord_m


def airfoil_face(airfoil: AirfoilShape, chord_m: float) -> bd.Face:
    pts = [(chord_m * x, chord_m * z, 0.0) for x, z in airfoil.coordinates]
    if pts[0] == pts[-1]:
        pts = pts[:-1]  # make_polygon closes the loop itself
    return bd.Face(bd.Wire.make_polygon(pts, close=True))


def notched_profile(airfoil: AirfoilShape, chord_m: float, settings: FabricationSettings) -> bd.Face:
    """Airfoil with a square LE-stock notch at the nose and cut back to the TE stock."""
    reach = CUTTER_REACH_FACTOR * chord_m
    le = settings.le_side_m
    le_y = camber_at(airfoil, 0.0) * chord_m
    le_notch = rect_face(-reach, le_y - le / 2.0, le, le_y + le / 2.0)
    te_cut = rect_face(chord_m - settings.te_chord_m, -reach, chord_m + reach, reach)
    if chord_m - settings.te_chord_m <= le:
        raise KitUnavailable(f"a {chord_m * 1e3:.0f} mm rib is too short for its LE and TE stock")
    return largest_face(airfoil_face(airfoil, chord_m).cut(le_notch, te_cut))


def split_at(face: bd.Face, x_m: float) -> tuple[bd.Face, bd.Face]:
    """(fore, aft) pieces of `face` either side of the line local x = `x_m` (a control-surface hinge)."""
    box = face.bounding_box()
    if not box.min.X < x_m < box.max.X:
        raise KitUnavailable(f"hinge at x = {x_m * 1e3:.1f} mm misses the rib")
    reach = CUTTER_REACH_FACTOR * (box.max.X - box.min.X)
    fore = largest_face(face.cut(rect_face(x_m, -reach, box.max.X + reach, reach)))
    aft = largest_face(face.cut(rect_face(box.min.X - reach, -reach, x_m, reach)))
    return fore, aft


def contains(face: bd.Face, hole: Hole) -> bool:
    return face.is_inside(bd.Vector(hole.x_m, hole.y_m, 0.0))


def clearance_m(face: bd.Face, x_m: float, y_m: float, holes: Sequence[Hole]) -> float:
    """Distance from a point inside `face` to its outline, its existing openings and the edge of every hole
    in `holes` (negative inside one)."""
    point, vertex = bd.Vector(x_m, y_m, 0.0), bd.Vertex(x_m, y_m, 0.0)
    gaps = [vertex.distance_to(wire) for wire in (face.outer_wire(), *face.inner_wires())]
    gaps += [(point - bd.Vector(h.x_m, h.y_m, 0.0)).length - h.diameter_m / 2.0 for h in holes]
    return min(gaps)


def keeps_web(face: bd.Face, hole: Hole, others: Sequence[Hole]) -> bool:
    """True when `hole` sits inside `face` with MIN_WEB_M to the outline and to every other hole (to FIT_EPS_M: an
    exact fit passes despite float round-off, D9)."""
    if not contains(face, hole):
        return False
    return clearance_m(face, hole.x_m, hole.y_m, others) - hole.diameter_m / 2.0 >= MIN_WEB_M - FIT_EPS_M


def lightening_holes(
    face: bd.Face, fixed: Sequence[Hole], centres: Iterable[tuple[float, float]]
) -> tuple[Hole, ...]:
    """Largest circle at each candidate centre that keeps MIN_WEB_M to the outline and every hole.

    Candidates outside the face, inside another hole, or with room only for a hole smaller than
    MIN_LIGHTENING_DIAMETER_M are skipped (small tip ribs simply get none).
    """
    placed: list[Hole] = []
    for x, y in centres:
        if not face.is_inside(bd.Vector(x, y, 0.0)):
            continue
        diameter = 2.0 * (clearance_m(face, x, y, (*fixed, *placed)) - MIN_WEB_M)
        if diameter >= MIN_LIGHTENING_DIAMETER_M:
            placed.append(Hole(x, y, diameter))
    return tuple(placed)


def drill(face: bd.Face, holes: Sequence[Hole], label: str) -> bd.Face:
    """Cut `holes` from `face`; each must stay a closed hole inside the outline."""
    if not holes:
        return face
    result = face.cut(*(circle_face(h) for h in holes)).faces()
    if len(result) != 1 or len(result[0].inner_wires()) != len(face.inner_wires()) + len(holes):
        raise KitUnavailable(f"a hole breaks out of rib {label}")
    return result[0]


def oblique_diameter_m(od_m: float, thickness_m: float, lean: float) -> float:
    """Round hole (before fit clearance) that clears a rod of `od_m` crossing a slab of `thickness_m` along a line
    leaning `lean` (off-normal travel per unit along the normal): the section is an ellipse od / cos long that also
    walks t tan across the thickness (R19)."""
    return od_m * math.hypot(1.0, lean) + thickness_m * lean


# ------------------------------------------------------------------------- face booleans ---
def shape_area(shape: Optional[bd.Shape]) -> float:
    return 0.0 if shape is None else sum(f.area for f in shape.faces())


def common(a: bd.Shape, b: bd.Shape) -> Optional[bd.Shape]:
    """Boolean intersection as one shape (Build123d may return a list of pieces), or None when empty."""
    result = a.intersect(b)
    pieces = result if isinstance(result, list) else ([] if result is None else [result])
    faces = [f for piece in pieces for f in piece.faces()]
    return bd.Compound(faces) if faces else None


def single_face(shape: bd.Shape, label: str) -> bd.Face:
    faces = shape.faces()
    if len(faces) != 1:
        raise KitUnavailable(f"part {label} falls apart into {len(faces)} pieces")
    return faces[0]
