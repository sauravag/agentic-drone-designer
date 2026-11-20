"""Fittings and bed-size splitting for the laser-cut kit (frame convention: see `kit.py`).

- Component trays `P-<name>`: a flat 3 mm plate directly under each internal component, footprint = the component's
  x/y size + COMPONENT_GAP_M (half each side), clipped to the skin inside the covering. The tray lies inside the
  half-gap envelope round the component that every former, the crutch and the wing-box plates already open around,
  so it needs no slots and is never split at a former; that holds while the plate sheet is no thicker than half the
  gap (else `KitUnavailable`). A tray that would hit another component, a boom, the wing socket or a pin is
  unavailable.
- Cruise-motor plate `MP-C` (R17): a disc glued to the firewall's front face, motor diameter + 2 MIN_WEB_M, centre
  hole and slotted bolt cross as the lift-motor plates. It may overhang a small firewall on a pointed nose.
- Boom clamps, lift-motor plates and boom tubes live in `boom_parts.py` (re-exported here).
- `split_to_bed`: a part too long for the usable bed (bed - 2 margin each way) is cut across its long axis into the
  fewest equal pieces that fit, each cut moved up to SPLIT_SEARCH_M so the joint keeps SPLIT_CLEARANCE_M from every
  hole and slot. Pieces `<label>a, b, …` keep the part's location (they sit side by side as cut) and interlock with a
  dovetail; a doubler `<label>-D<k>` of the same family is glued on the part's +Z face over joint k, lapping
  DOUBLER_LAP_M either side of the dovetail (down to DOUBLER_MIN_LAP_M where holes crowd it). Its band stays clear of
  the long-axis extent of every hole and slot (+ MIN_WEB_M), so it misses whatever passes through them (formers
  stand in the crutch's tab slots and span its width). Its edges are inset DOUBLER_EDGE_INSET_M from the part's
  outline: on a curved skin the doubler's outer face sits one sheet off the part's plane, where the skin is narrower
  by r - sqrt(r^2 - z^2); 2 mm bounds that for every radius above 6 mm at the 4.5 mm top of a crutch doubler.
  The dovetail is cut at nominal size: its press fit comes from the kerf (like every tab and slot).
  R23: a cut may pass straight through a component opening tagged in `KitPart.openings` (the whole dovetail core
  SPLIT_CLEARANCE_M inside its ends; never an untagged slot or hole, whatever its size). Each side rail the line crosses then gets its
  own dovetail (MIN_WEB_M rim) and its own doubler `<label>-D<k>a`, `-D<k>b`, … lapping that rail only.
"""

import math
import re
from dataclasses import dataclass
from typing import Final, Optional, Sequence

import build123d as bd

from src.schemas.fabrication import FabricationSettings
from src.tools.aero_cad.components import COMPONENT_GAP_M, PlacedComponent, boom_obstacles, overlap_volume_m3
from src.tools.aero_cad.layout import AircraftLayout, fuselage_radius_at
from src.tools.fabrication.boom_parts import boom_parts, motor_plate_diameter_m, motor_plate_outline
from src.tools.fabrication.fuselage_parts import (
    firewall_station,
    material_spans,
    profile_corners,
    socket_axis_m,
    station_location,
)
from src.tools.fabrication.kit import Box2, KitPart
from src.tools.fabrication.rib_outline import rect_face
from src.tools.fabrication.tubes import MIN_WEB_M, KitUnavailable, SparPlan
from src.tools.fabrication.wing_parts import PIN_CHORD_FRACTION

__all__ = ["boom_parts", "component_plates", "cruise_motor_plate", "join_notes", "split_to_bed"]

SPLIT_SEARCH_M: Final[float] = 0.03
DOVETAIL_NECK_M: Final[float] = 0.010
DOVETAIL_FLARE_M: Final[float] = 0.016
DOVETAIL_DEPTH_M: Final[float] = 0.012
SPLIT_CLEARANCE_M: Final[float] = 0.003  # joint path to any hole or slot (>= MIN_WEB_M, room for kerf)
SPLIT_SEARCH_STEP_M: Final[float] = 0.001
DOUBLER_LAP_M: Final[float] = 0.020  # doubler overlap either side of the dovetail, when there is room
DOUBLER_MIN_LAP_M: Final[float] = 0.010  # shortest overlap accepted between holes or slots
DOUBLER_EDGE_INSET_M: Final[float] = 0.002  # doubler edges stay this far inside the part's outline
DOUBLER_LABEL: Final[re.Pattern[str]] = re.compile(r"^(?P<base>.+)-D(?P<k>\d+)(?P<rail>[a-z])?$")
MAX_SPLIT_PIECES: Final[int] = 8
SPAN_EPS_M: Final[float] = 1e-6  # a hole edge on a rail edge does not count as beside the rail
CRUISE_MOTOR_NAME: Final[str] = "cruise_motor"
AREA_EPS_M2: Final[float] = 1e-10
VOLUME_EPS_M3: Final[float] = 1e-12
REACH_M: Final[float] = 10.0  # cutter half-planes extend this far past any part
ROTATE_TO_X_DEG: Final[float] = -90.0  # turns a part whose long axis is local Y onto X


# -------------------------------------------------------------------------------------- helpers ---
def _pieces_of(shape: object) -> list[bd.Shape]:
    """A boolean result as a list (Build123d may return a shape, a list or None)."""
    if shape is None:
        return []
    if isinstance(shape, list):
        return list(shape)
    return [shape] if isinstance(shape, bd.Shape) else []


def _faces(shape: object) -> list[bd.Face]:
    return [f for piece in _pieces_of(shape) for f in piece.faces()]


def _rod(start: bd.Vector, end: bd.Vector, diameter_m: float) -> bd.Solid:
    axis = end - start
    return bd.Solid.make_cylinder(diameter_m / 2.0, axis.length, bd.Plane(origin=start, z_dir=axis.normalized()))


# ---------------------------------------------------------------------------------- component trays ---
def _skin_section(layout: AircraftLayout, settings: FabricationSettings, x0: float, x1: float, z_m: float,
                  label: str) -> bd.Face:
    """Plan-view region inside the covering at height z (chords between profile corners stay inside the skin)."""
    xs = sorted({x0, x1, *profile_corners(layout, x0, x1)})
    cover = settings.fuselage_cover_thickness_m
    half = [math.sqrt(max((fuselage_radius_at(layout, x) - cover) ** 2 - z_m**2, 0.0)) for x in xs]
    if min(half) <= 0.0:
        raise KitUnavailable(f"{label}: the tray reaches outside the fuselage")
    upper = [(x, h, 0.0) for x, h in zip(xs, half)]
    lower = [(x, -h, 0.0) for x, h in reversed(list(zip(xs, half)))]
    return bd.Face(bd.Wire.make_polygon(upper + lower, close=True))


def _tray_outline(c: PlacedComponent, layout: AircraftLayout, settings: FabricationSettings, label: str) -> bd.Face:
    """Footprint + gap round the component centre (local origin), clipped to the skin at the tray's worse face."""
    lx, ly, lz = c.size_m
    hx, hy = (lx + COMPONENT_GAP_M) / 2.0, (ly + COMPONENT_GAP_M) / 2.0
    top = c.center_m[2] - lz / 2.0
    z = max(abs(top), abs(top - settings.sheet_for("plate").thickness_m))
    cx, cy = c.center_m[0], c.center_m[1]
    skin = _skin_section(layout, settings, cx - hx, cx + hx, z, label).moved(bd.Location((-cx, -cy, 0.0)))
    faces = _faces(rect_face(-hx, -hy, hx, hy).intersect(skin))
    if len(faces) != 1:
        raise KitUnavailable(f"{label}: the tray under the {c.name} does not fit inside the fuselage")
    return faces[0]


def _wing_rods(layout: AircraftLayout, spar: SparPlan) -> list[tuple[str, bd.Solid]]:
    """Each wing's socket tube and anti-rotation pin, from the centreline to the skin joint (the pin really starts
    at the wing-box plate, so this is conservative)."""
    rods = []
    for side in (1.0, -1.0):
        for name, fraction, diameter in (("socket", None, spar.socket.od_m),
                                         ("pin", PIN_CHORD_FRACTION, spar.pin.rod_m)):
            ends = []
            for y in (0.0, spar.joint_y_m):
                x, z = socket_axis_m(layout, y) if fraction is None else socket_axis_m(layout, y, fraction)
                ends.append(bd.Vector(x, side * y, z))
            rods.append((f"wing {name}", _rod(ends[0], ends[1], diameter)))
    return rods


def _structure_obstacles(layout: AircraftLayout, spar: Optional[SparPlan]) -> list[tuple[str, bd.Solid]]:
    """Booms (layout diameter, never thinner than the stock tube) and, with a spar plan, the socket and pins."""
    solids = [(b.name.replace("_", " "), b.solid()) for b in boom_obstacles(layout)]
    return solids if spar is None else solids + _wing_rods(layout, spar)


def _tray(c: PlacedComponent, components: Sequence[PlacedComponent], obstacles: Sequence[tuple[str, bd.Solid]],
          layout: AircraftLayout, settings: FabricationSettings) -> KitPart:
    label, t = f"P-{c.name}", settings.sheet_for("plate").thickness_m
    if t > COMPONENT_GAP_M / 2.0:
        raise KitUnavailable(f"{label}: a {t * 1e3:.1f} mm tray does not fit the "
                             f"{COMPONENT_GAP_M / 2.0 * 1e3:.0f} mm gap under the {c.name}")
    z0 = c.center_m[2] - c.size_m[2] / 2.0 - t
    outline = _tray_outline(c, layout, settings, label)
    part = KitPart(label, "plate", outline, bd.Location((c.center_m[0], c.center_m[1], z0)))
    solid = bd.extrude(part.outline, amount=t, dir=(0, 0, 1)).moved(part.location)
    for name, other in [(o.name, o.solid()) for o in components if o is not c] + list(obstacles):
        if overlap_volume_m3(solid, other) > VOLUME_EPS_M3:
            raise KitUnavailable(f"{label}: the tray under the {c.name} hits the {name}")
    return part


def component_plates(
    components: Sequence[PlacedComponent], layout: AircraftLayout, settings: FabricationSettings,
    spar: Optional[SparPlan] = None,
) -> tuple[KitPart, ...]:
    """One tray `P-<name>` directly under each internal component (see the module docstring). With `spar`, trays
    are also checked against the wing socket tubes and pins."""
    obstacles = _structure_obstacles(layout, spar)
    return tuple(_tray(c, components, obstacles, layout, settings) for c in components if c.is_internal)


def cruise_motor_plate(
    components: Sequence[PlacedComponent], layout: AircraftLayout, settings: FabricationSettings
) -> tuple[KitPart, ...]:
    """`MP-C` on the firewall's front face, centred on the cruise motor's axis (none without a cruise motor)."""
    motor = next((c for c in components if c.name == CRUISE_MOTOR_NAME), None)
    if motor is None:
        return ()
    front = firewall_station(layout, settings, components) - settings.sheet_for("plate").thickness_m
    if front < motor.x_max:
        raise KitUnavailable("MP-C: the cruise motor reaches back into its mount plate")
    outline = motor_plate_outline(motor_plate_diameter_m(motor.size_m[1]))
    outline = outline.moved(bd.Location((motor.center_m[1], motor.center_m[2], 0.0)))
    return (KitPart("MP-C", "plate", outline, station_location(front)),)


# ------------------------------------------------------------------------------------ bed splitting ---
@dataclass(frozen=True)
class _Rail:
    """One stretch of material the cut line crosses (the whole width, or a side rail beside an opening)."""

    y_lo_m: float
    y_hi_m: float
    lap_m: float = DOUBLER_LAP_M  # doubler overlap either side of the dovetail

    @property
    def y_m(self) -> float:  # tail centre
        return 0.5 * (self.y_lo_m + self.y_hi_m)


@dataclass(frozen=True)
class _Joint:
    x_m: float  # cut line (local x); each tail reaches DOVETAIL_DEPTH_M beyond it
    rails: tuple[_Rail, ...]  # fore to aft in y; one dovetail and one doubler per rail


def _usable_bed(settings: FabricationSettings) -> tuple[float, float]:
    """(long, short) usable bed sides."""
    sides = (settings.bed_width_m - 2.0 * settings.margin_m, settings.bed_height_m - 2.0 * settings.margin_m)
    return max(sides), min(sides)


def _size(face: bd.Face) -> tuple[float, float]:
    box = face.bounding_box()
    return box.max.X - box.min.X, box.max.Y - box.min.Y


def _fits(face: bd.Face, settings: FabricationSettings) -> bool:
    (w, h), (long, short) = _size(face), _usable_bed(settings)
    return max(w, h) <= long and min(w, h) <= short


def _tail_points(x_m: float, y_m: float) -> list[tuple[float, float]]:
    """Neck, flare, flare, neck of one dovetail bulging +x round y_m."""
    d = DOVETAIL_DEPTH_M
    return [(x_m, y_m - DOVETAIL_NECK_M / 2.0), (x_m + d, y_m - DOVETAIL_FLARE_M / 2.0),
            (x_m + d, y_m + DOVETAIL_FLARE_M / 2.0), (x_m, y_m + DOVETAIL_NECK_M / 2.0)]


def _dovetail_points(joint: _Joint) -> list[tuple[float, float]]:
    """The joint path from far below to far above: straight cut, with a tail on every rail."""
    tails = [pt for rail in joint.rails for pt in _tail_points(joint.x_m, rail.y_m)]
    return [(joint.x_m, -REACH_M), *tails, (joint.x_m, REACH_M)]


def _left_of(joint: _Joint) -> bd.Face:
    """Half-plane left of the joint path; the tails belong to the left piece."""
    pts = [(-REACH_M, -REACH_M), *_dovetail_points(joint), (-REACH_M, REACH_M)]
    return bd.Face(bd.Wire.make_polygon([(px, py, 0.0) for px, py in pts], close=True))


def _doubler_band(joint: _Joint, rail: _Rail) -> tuple[float, float]:
    return joint.x_m - rail.lap_m, joint.x_m + DOVETAIL_DEPTH_M + rail.lap_m


def _beside(box: bd.BoundBox, y_lo_m: float, y_hi_m: float) -> bool:
    """The box's y extent overlaps the open interval (y_lo, y_hi)."""
    return box.min.Y < y_hi_m - SPAN_EPS_M and box.max.Y > y_lo_m + SPAN_EPS_M


def _doubler_lap(face: bd.Face, x_m: float, span: tuple[float, float]) -> Optional[float]:
    """Longest lap (<= DOUBLER_LAP_M) whose band, on this rail, misses the x extent + MIN_WEB_M of every hole and
    slot beside the rail; None when shorter than DOUBLER_MIN_LAP_M or when one sits over the dovetail itself.
    An opening the rail runs alongside (not beside it in y) never meets the rail's doubler."""
    lap, core = DOUBLER_LAP_M, (x_m, x_m + DOVETAIL_DEPTH_M)
    for wire in face.inner_wires():
        box = wire.bounding_box()
        if not _beside(box, *span):
            continue
        lo, hi = box.min.X - MIN_WEB_M, box.max.X + MIN_WEB_M
        if lo < core[1] and hi > core[0]:
            return None
        lap = min(lap, core[0] - hi if hi <= core[0] else lo - core[1])
    return lap if lap >= DOUBLER_MIN_LAP_M else None


def _inside(box: bd.BoundBox, opening: Box2) -> bool:
    x0, y0, x1, y1 = opening
    e = SPAN_EPS_M
    return x0 - e <= box.min.X and box.max.X <= x1 + e and y0 - e <= box.min.Y and box.max.Y <= y1 + e


def _crosses_opening(wire: bd.Wire, x_m: float, openings: Sequence[Box2]) -> bool:
    """The cut may pass through this hole: it is one of the part's tagged component openings (`KitPart.openings`;
    never a slot or a bolt hole, whatever its size), and the whole dovetail core keeps SPLIT_CLEARANCE_M from its
    ends."""
    box = wire.bounding_box()
    return (any(_inside(box, o) for o in openings) and box.min.X + SPLIT_CLEARANCE_M <= x_m
            and x_m + DOVETAIL_DEPTH_M + SPLIT_CLEARANCE_M <= box.max.X)


def _tails_clear(face: bd.Face, joint: _Joint) -> bool:
    """Every tail sits in material with a MIN_WEB_M rim."""
    w, half = MIN_WEB_M, DOVETAIL_FLARE_M / 2.0
    for rail in joint.rails:
        zone = rect_face(joint.x_m - w, rail.y_m - half - w, joint.x_m + DOVETAIL_DEPTH_M + w, rail.y_m + half + w)
        if sum(f.area for f in _faces(zone.cut(face))) > AREA_EPS_M2:
            return False
    return True


def _joint_is_clear(face: bd.Face, joint: _Joint, openings: Sequence[Box2]) -> bool:
    """Tails in material with a MIN_WEB_M rim, and the path keeps SPLIT_CLEARANCE_M from every hole and slot except
    a component opening it passes straight through: beside that, the tails' MIN_WEB_M rim is the web (R23)."""
    if not _tails_clear(face, joint):
        return False
    path = bd.Wire.make_polygon([(px, py, 0.0) for px, py in _dovetail_points(joint)], close=False)
    return all(_crosses_opening(wire, joint.x_m, openings) or path.distance_to(wire) >= SPLIT_CLEARANCE_M
               for wire in face.inner_wires())


def _joint_at(face: bd.Face, x_m: float) -> Optional[_Joint]:
    """A joint on the line x = x_m with one dovetail per rail, or None when a rail has no doubler room."""
    rails: list[_Rail] = []
    for lo, hi in material_spans(face, x_m):
        lap = _doubler_lap(face, x_m, (lo, hi))
        if lap is None:
            return None
        rails.append(_Rail(lo, hi, lap))
    return _Joint(x_m, tuple(rails)) if rails else None


def _find_joint(face: bd.Face, nominal_m: float, openings: Sequence[Box2]) -> Optional[_Joint]:
    """The clear joint nearest the nominal cut, within SPLIT_SEARCH_M of it."""
    steps = round(SPLIT_SEARCH_M / SPLIT_SEARCH_STEP_M)
    for k in sorted(range(-steps, steps + 1), key=abs):
        joint = _joint_at(face, nominal_m + k * SPLIT_SEARCH_STEP_M)
        if joint is not None and _joint_is_clear(face, joint, openings):
            return joint
    return None


def _cut(face: bd.Face, joints: Sequence[_Joint]) -> Optional[list[bd.Face]]:
    """Pieces left to right, or None when a cut leaves more than one piece on either side."""
    pieces, rest = [], face
    for joint in joints:
        left, right = _faces(rest.intersect(_left_of(joint))), _faces(rest.cut(_left_of(joint)))
        if len(left) != 1 or len(right) != 1:
            return None
        pieces.append(left[0])
        rest = right[0]
    return pieces + [rest]


def _rail_strip(joint: _Joint, k: int) -> tuple[float, float]:
    """y strip owning rail k: halfway to the neighbouring rails (the far ends reach out)."""
    rails = joint.rails
    lo = -REACH_M if k == 0 else 0.5 * (rails[k - 1].y_hi_m + rails[k].y_lo_m)
    hi = REACH_M if k == len(rails) - 1 else 0.5 * (rails[k].y_hi_m + rails[k + 1].y_lo_m)
    return lo, hi


def _doubler(face: bd.Face, joint: _Joint, k: int) -> Optional[bd.Face]:
    """Rail k's material inside its doubler band, edges inset DOUBLER_EDGE_INSET_M. Only a slightly wider band is
    inset (then trimmed back to the band), so the band ends keep their full lap and narrow necks elsewhere don't
    matter."""
    lo, hi = _doubler_band(joint, joint.rails[k])
    y_lo, y_hi = _rail_strip(joint, k)
    pad = DOUBLER_EDGE_INSET_M
    region = _faces(face.intersect(rect_face(lo - pad, y_lo, hi + pad, y_hi)))
    if len(region) != 1:
        return None
    try:
        inset = _faces(bd.offset(region[0], amount=-DOUBLER_EDGE_INSET_M))
    except RuntimeError:  # OCCT cannot offset this region: no doubler fits here
        return None
    if len(inset) != 1:
        return None
    faces = _faces(inset[0].intersect(rect_face(lo, -REACH_M, hi, REACH_M)))
    return faces[0] if len(faces) == 1 else None


def _joint_doublers(face: bd.Face, joint: _Joint) -> Optional[list[bd.Face]]:
    doublers: list[bd.Face] = []
    for k in range(len(joint.rails)):
        doubler = _doubler(face, joint, k)
        if doubler is None:
            return None
        doublers.append(doubler)
    return doublers


Split = tuple[list[bd.Face], list[list[bd.Face]]]  # pieces left to right; doublers per joint, one per rail


def _joints(face: bd.Face, count: int, openings: Sequence[Box2]) -> Optional[list[_Joint]]:
    """A clear joint near each of the `count - 1` equal-pitch cuts, or None when one finds no clear spot."""
    box = face.bounding_box()
    pitch = (box.max.X - box.min.X) / count
    joints: list[_Joint] = []
    for k in range(1, count):
        joint = _find_joint(face, box.min.X + k * pitch, openings)
        if joint is None:
            return None
        joints.append(joint)
    return joints


def _try_split(face: bd.Face, count: int, settings: FabricationSettings, openings: Sequence[Box2]) -> Optional[Split]:
    """(pieces, doublers) for `count` equal pieces, or None when a joint finds no clear spot or a piece won't fit."""
    joints = _joints(face, count, openings)
    pieces = None if joints is None else _cut(face, joints)
    if joints is None or pieces is None:
        return None
    doublers: list[list[bd.Face]] = []
    for joint in joints:
        found = _joint_doublers(face, joint)
        if found is None:
            return None
        doublers.append(found)
    if not all(_fits(f, settings) for f in (*pieces, *(d for ds in doublers for d in ds))):
        return None
    return pieces, doublers


def _split_along_x(face: bd.Face, label: str, settings: FabricationSettings, openings: Sequence[Box2]) -> Split:
    long = _usable_bed(settings)[0]
    for count in range(max(2, math.ceil(_size(face)[0] / long)), MAX_SPLIT_PIECES + 1):
        result = _try_split(face, count, settings, openings)
        if result is not None:
            return result
    raise KitUnavailable(f"{label} cannot be split to fit the {settings.bed_width_m * 1e3:.0f} x "
                         f"{settings.bed_height_m * 1e3:.0f} mm bed clear of its holes and slots")


def _doubler_labels(label: str, doublers: Sequence[Sequence[bd.Face]]) -> list[tuple[str, bd.Face]]:
    """`<label>-D<k>` for joint k, with a rail letter (`-D<k>a`, `-D<k>b`) when the joint has several rails."""
    named = []
    for k, rail_doublers in enumerate(doublers, start=1):
        many = len(rail_doublers) > 1
        named += [(f"{label}-D{k}{chr(ord('a') + i) if many else ''}", f) for i, f in enumerate(rail_doublers)]
    return named


def _turned(opening: Box2) -> Box2:
    """An opening box after the ROTATE_TO_X_DEG (-90 deg) turn: (x, y) -> (y, -x)."""
    x0, y0, x1, y1 = opening
    return y0, -x1, y1, -x0


def split_to_bed(part: KitPart, settings: FabricationSettings) -> tuple[KitPart, ...]:
    """The part unchanged if it fits the usable bed, else its pieces then their doublers (module docstring).

    Pieces are nominal geometry: the dovetail's tail and socket are the same line, and the joint gets its press fit
    from the laser kerf (applied at DXF export), as tabs and slots do."""
    if _fits(part.outline, settings):
        return (part,)
    (w, h), short = _size(part.outline), _usable_bed(settings)[1]
    if min(w, h) > short:
        raise KitUnavailable(f"{part.label} is wider than the {short * 1e3:.0f} mm usable bed")
    along_y = h > w
    face = part.outline.rotate(bd.Axis.Z, ROTATE_TO_X_DEG) if along_y else part.outline
    openings = tuple(_turned(o) for o in part.openings) if along_y else part.openings
    pieces, doublers = _split_along_x(face, part.label, settings, openings)
    named_doublers = _doubler_labels(part.label, doublers)
    if along_y:
        pieces = [f.rotate(bd.Axis.Z, -ROTATE_TO_X_DEG) for f in pieces]
        named_doublers = [(n, f.rotate(bd.Axis.Z, -ROTATE_TO_X_DEG)) for n, f in named_doublers]
    on_face = part.location * bd.Location((0.0, 0.0, settings.sheet_for(part.family).thickness_m))
    named = [KitPart(f"{part.label}{chr(ord('a') + i)}", part.family, f, part.location) for i, f in enumerate(pieces)]
    named += [KitPart(n, part.family, f, on_face) for n, f in named_doublers]
    return tuple(named)


def _world_centre(part: KitPart) -> bd.Vector:
    return (part.location * bd.Location(tuple(part.outline.center()))).position


def join_notes(pieces: Sequence[KitPart]) -> tuple[str, ...]:
    """'C-1a + C-1b: dovetail + doubler C-1-D1 at x = 0.512 m' for every doubler among `pieces`."""
    by_label, notes = {p.label: p for p in pieces}, []
    for doubler in pieces:
        match = DOUBLER_LABEL.match(doubler.label)
        if match is None:
            continue
        base, k = match["base"], int(match["k"])
        a, b = (by_label[f"{base}{chr(ord('a') + k - 1 + i)}"] for i in (0, 1))
        direction = tuple(_world_centre(b) - _world_centre(a))
        axis = max(range(3), key=lambda i: abs(direction[i]))
        coord = tuple(_world_centre(doubler))[axis]
        notes.append(f"{a.label} + {b.label}: dovetail + doubler {doubler.label} at {'xyz'[axis]} = {coord:.3f} m")
    return tuple(notes)
