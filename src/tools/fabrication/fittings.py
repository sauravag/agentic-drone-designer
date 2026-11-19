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
  stand in the crutch's tab slots and span its width).
"""

import math
from dataclasses import dataclass
from typing import Final, Optional, Sequence

import build123d as bd

from src.schemas.fabrication import FabricationSettings
from src.tools.aero_cad.components import COMPONENT_GAP_M, PlacedComponent, boom_obstacles, overlap_volume_m3
from src.tools.aero_cad.layout import AircraftLayout, fuselage_radius_at
from src.tools.fabrication.boom_parts import boom_parts, motor_plate_diameter_m, motor_plate_outline
from src.tools.fabrication.fuselage_parts import firewall_station, profile_corners, socket_axis_m, station_location
from src.tools.fabrication.kit import KitPart
from src.tools.fabrication.rib_outline import rect_face
from src.tools.fabrication.tubes import MIN_WEB_M, KitUnavailable, SparPlan
from src.tools.fabrication.wing_parts import PIN_CHORD_FRACTION, PIN_ROD_DIAMETER_M

__all__ = ["boom_parts", "component_plates", "cruise_motor_plate", "join_notes", "split_to_bed"]

SPLIT_SEARCH_M: Final[float] = 0.03
DOVETAIL_NECK_M: Final[float] = 0.010
DOVETAIL_FLARE_M: Final[float] = 0.016
DOVETAIL_DEPTH_M: Final[float] = 0.012
SPLIT_CLEARANCE_M: Final[float] = 0.003  # joint path to any hole or slot (>= MIN_WEB_M, room for kerf)
SPLIT_SEARCH_STEP_M: Final[float] = 0.001
DOUBLER_LAP_M: Final[float] = 0.020  # doubler overlap either side of the dovetail, when there is room
DOUBLER_MIN_LAP_M: Final[float] = 0.010  # shortest overlap accepted between holes or slots
MAX_SPLIT_PIECES: Final[int] = 8
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
    return list(shape) if isinstance(shape, list) else [shape]  # type: ignore[list-item]


def _faces(shape: object) -> list[bd.Face]:
    return [f for piece in _pieces_of(shape) for f in piece.faces()]


def _edges(shape: object) -> list[bd.Edge]:
    return [e for piece in _pieces_of(shape) for e in piece.edges()]


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
                                         ("pin", PIN_CHORD_FRACTION, PIN_ROD_DIAMETER_M)):
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
class _Joint:
    x_m: float  # cut line (local x); the tail reaches DOVETAIL_DEPTH_M beyond it
    y_m: float  # tail centre
    lap_m: float = DOUBLER_LAP_M  # doubler overlap either side of the dovetail


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


def _dovetail_points(joint: _Joint) -> list[tuple[float, float]]:
    """The joint path from far below to far above: straight cut, with the tail bulging +x round y_m."""
    c, y, d = joint.x_m, joint.y_m, DOVETAIL_DEPTH_M
    return [(c, -REACH_M), (c, y - DOVETAIL_NECK_M / 2.0), (c + d, y - DOVETAIL_FLARE_M / 2.0),
            (c + d, y + DOVETAIL_FLARE_M / 2.0), (c, y + DOVETAIL_NECK_M / 2.0), (c, REACH_M)]


def _left_of(joint: _Joint) -> bd.Face:
    """Half-plane left of the joint path; the tail belongs to the left piece."""
    pts = [(-REACH_M, -REACH_M), *_dovetail_points(joint), (-REACH_M, REACH_M)]
    return bd.Face(bd.Wire.make_polygon([(px, py, 0.0) for px, py in pts], close=True))


def _doubler_band(joint: _Joint) -> tuple[float, float]:
    return joint.x_m - joint.lap_m, joint.x_m + DOVETAIL_DEPTH_M + joint.lap_m


def _doubler_lap(face: bd.Face, x_m: float) -> Optional[float]:
    """Longest lap (<= DOUBLER_LAP_M) whose band misses every hole's and slot's x extent + MIN_WEB_M; None when
    shorter than DOUBLER_MIN_LAP_M or when one sits over the dovetail itself."""
    lap, core = DOUBLER_LAP_M, (x_m, x_m + DOVETAIL_DEPTH_M)
    for wire in face.inner_wires():
        box = wire.bounding_box()
        lo, hi = box.min.X - MIN_WEB_M, box.max.X + MIN_WEB_M
        if lo < core[1] and hi > core[0]:
            return None
        lap = min(lap, core[0] - hi if hi <= core[0] else lo - core[1])
    return lap if lap >= DOUBLER_MIN_LAP_M else None


def _material_span(face: bd.Face, x_m: float) -> Optional[tuple[float, float]]:
    """The single stretch of material on the line local x = x_m (None when the line meets it more than once)."""
    edges = _edges(face.intersect(bd.Edge.make_line((x_m, -REACH_M, 0.0), (x_m, REACH_M, 0.0))))
    if len(edges) != 1:
        return None
    ys = sorted((edges[0].start_point().Y, edges[0].end_point().Y))
    return ys[0], ys[1]


def _joint_is_clear(face: bd.Face, joint: _Joint) -> bool:
    """The tail sits in material with a MIN_WEB_M rim and the path keeps SPLIT_CLEARANCE_M from every hole and
    slot."""
    w, half = MIN_WEB_M, DOVETAIL_FLARE_M / 2.0
    zone = rect_face(joint.x_m - w, joint.y_m - half - w, joint.x_m + DOVETAIL_DEPTH_M + w, joint.y_m + half + w)
    if sum(f.area for f in _faces(zone.cut(face))) > AREA_EPS_M2:
        return False
    path = bd.Wire.make_polygon([(px, py, 0.0) for px, py in _dovetail_points(joint)], close=False)
    return all(path.distance_to(wire) >= SPLIT_CLEARANCE_M for wire in face.inner_wires())


def _find_joint(face: bd.Face, nominal_m: float) -> Optional[_Joint]:
    """The clear joint nearest the nominal cut, within SPLIT_SEARCH_M of it."""
    steps = round(SPLIT_SEARCH_M / SPLIT_SEARCH_STEP_M)
    for k in sorted(range(-steps, steps + 1), key=abs):
        x = nominal_m + k * SPLIT_SEARCH_STEP_M
        lap = _doubler_lap(face, x)
        span = None if lap is None else _material_span(face, x)
        if lap is None or span is None:
            continue
        joint = _Joint(x, (span[0] + span[1]) / 2.0, lap)
        if _joint_is_clear(face, joint):
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


def _doubler(face: bd.Face, joint: _Joint) -> Optional[bd.Face]:
    lo, hi = _doubler_band(joint)
    faces = _faces(face.intersect(rect_face(lo, -REACH_M, hi, REACH_M)))
    return faces[0] if len(faces) == 1 else None


def _try_split(face: bd.Face, count: int, settings: FabricationSettings) -> Optional[tuple[list[bd.Face], list[bd.Face]]]:
    """(pieces, doublers) for `count` equal pieces, or None when a joint finds no clear spot or a piece won't fit."""
    box = face.bounding_box()
    pitch = (box.max.X - box.min.X) / count
    found = [_find_joint(face, box.min.X + k * pitch) for k in range(1, count)]
    joints = [j for j in found if j is not None]
    if len(joints) != len(found):
        return None
    pieces = _cut(face, joints)
    doublers = [d for d in (_doubler(face, j) for j in joints) if d is not None]
    if pieces is None or len(doublers) != len(joints):
        return None
    if not all(_fits(f, settings) for f in (*pieces, *doublers)):
        return None
    return pieces, doublers


def _split_along_x(face: bd.Face, label: str, settings: FabricationSettings) -> tuple[list[bd.Face], list[bd.Face]]:
    long = _usable_bed(settings)[0]
    for count in range(max(2, math.ceil(_size(face)[0] / long)), MAX_SPLIT_PIECES + 1):
        result = _try_split(face, count, settings)
        if result is not None:
            return result
    raise KitUnavailable(f"{label} cannot be split to fit the {settings.bed_width_m * 1e3:.0f} x "
                         f"{settings.bed_height_m * 1e3:.0f} mm bed clear of its holes and slots")


def split_to_bed(part: KitPart, settings: FabricationSettings) -> tuple[KitPart, ...]:
    """The part unchanged if it fits the usable bed, else its pieces then their doublers (module docstring)."""
    if _fits(part.outline, settings):
        return (part,)
    (w, h), short = _size(part.outline), _usable_bed(settings)[1]
    if min(w, h) > short:
        raise KitUnavailable(f"{part.label} is wider than the {short * 1e3:.0f} mm usable bed")
    along_y = h > w
    face = part.outline.rotate(bd.Axis.Z, ROTATE_TO_X_DEG) if along_y else part.outline
    pieces, doublers = _split_along_x(face, part.label, settings)
    if along_y:
        pieces = [f.rotate(bd.Axis.Z, -ROTATE_TO_X_DEG) for f in pieces]
        doublers = [f.rotate(bd.Axis.Z, -ROTATE_TO_X_DEG) for f in doublers]
    on_face = part.location * bd.Location((0.0, 0.0, settings.sheet_for(part.family).thickness_m))
    named = [KitPart(f"{part.label}{chr(ord('a') + i)}", part.family, f, part.location) for i, f in enumerate(pieces)]
    named += [KitPart(f"{part.label}-D{k}", part.family, f, on_face) for k, f in enumerate(doublers, start=1)]
    return tuple(named)


def _world_centre(part: KitPart) -> bd.Vector:
    return (part.location * bd.Location(tuple(part.outline.center()))).position


def join_notes(pieces: Sequence[KitPart]) -> tuple[str, ...]:
    """'C-1a + C-1b: dovetail + doubler C-1-D1 at x = 0.512 m' for every doubler among `pieces`."""
    by_label, notes = {p.label: p for p in pieces}, []
    for doubler in (p for p in pieces if "-D" in p.label):
        base, k = doubler.label.rsplit("-D", 1)
        a, b = (by_label[f"{base}{chr(ord('a') + int(k) - 1 + i)}"] for i in (0, 1))
        direction = tuple(_world_centre(b) - _world_centre(a))
        axis = max(range(3), key=lambda i: abs(direction[i]))
        coord = tuple(_world_centre(doubler))[axis]
        notes.append(f"{a.label} + {b.label}: dovetail + doubler {doubler.label} at {'xyz'[axis]} = {coord:.3f} m")
    return tuple(notes)
