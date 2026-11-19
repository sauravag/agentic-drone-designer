"""Fuselage parts for the laser-cut kit: crutch, half-formers, firewall and wing-mount side plates
(frame convention: see `kit.py`).

- Crutch `C-1`: a horizontal 3 mm plate centred on z = 0 (it spans z = +-t/2), as wide as the fuselage inside
  the covering, from the firewall's aft face to just aft of the last former. Its outline is in aircraft x, y,
  shifted by the plate origin x.
- Half-formers `F-nnU` / `F-nnL`: semi-discs of r(x) - cover above and below the crutch. Each has two tabs
  that drop into crutch slots (U and L tabs meet at z = 0), stringer notches, openings where components or the
  tail boom pass, and lightening holes. A former at station x lies in the yz plane and extrudes aft
  (local X -> y, local Y -> z, local Z -> +x), so its location X is the station and its outline radius is
  r(station) - cover.
- Firewall `FW-1a`, `FW-1b`: two solid discs laminated face to face at the first station. The cruise motor is a
  tractor ahead of the nose (outside the pod), so the firewall takes the former station nearest to it.
- Wing-box side plates `WM-L`, `WM-R`: vertical plates (xz plane) in the upper fuselage, from the forward socket
  former to a former just aft of the root pin, tabbed into both; formers in between get the plate's section cut
  out. Each carries its wing's socket tube (coaxial with that wing's spar, so following the dihedral; the two
  meet at the centreline) and the anti-rotation pin. They stop MIN_WEB_M under the socket hole, so components
  on the axis pass underneath. Local X -> x, local Y -> z; WM-R extrudes toward -y from its outer face, WM-L is
  its mirror.

Openings, slots and notches keep a `MIN_WEB_M` web; anything that would break a part raises `KitUnavailable`.
Slots are cut at the mating stock's nominal thickness; kerf is applied at DXF export.
"""

import math
from dataclasses import dataclass
from typing import Final, Optional, Sequence

import build123d as bd

from src.schemas.fabrication import FabricationSettings
from src.tools.aero_cad.airfoil_geometry import load_airfoil
from src.tools.aero_cad.components import COMPONENT_GAP_M, PlacedComponent
from src.tools.aero_cad.layout import (
    FUSELAGE_PROFILE,
    TAIL_BOOM_START_FRACTION,
    AircraftLayout,
    fuselage_radius_at,
)
from src.tools.aero_cad.structure import SPAR_CHORD_STATION, STATION_EPS
from src.tools.fabrication.kit import KitPart, Vec3
from src.tools.fabrication.rib_outline import (
    Hole,
    camber_point,
    circle_face,
    drill,
    keeps_web,
    lightening_holes,
    rect_face,
)
from src.tools.fabrication.tubes import BISECTION_ITERATIONS, MIN_WEB_M, KitUnavailable, SparPlan, boom_tube
from src.tools.fabrication.wing_parts import PIN_CHORD_FRACTION, pin_hole_m

FORMER_PITCH_M: Final[float] = 0.12
FIRST_FORMER_FRACTION: Final[float] = 0.10  # of fuselage length
STATION_MERGE_M: Final[float] = 0.020  # a regular station this close to a socket station is dropped
SOCKET_FORMER_GAP_M: Final[float] = 0.010  # socket stations sit this far beyond the socket's outer surface
STRINGER_COUNT: Final[int] = 8
FULL_TURN_DEG: Final[float] = 360.0
STRINGER_START_DEG: Final[float] = FULL_TURN_DEG / STRINGER_COUNT / 2.0  # none on the crutch plane
FORMER_TAB_WIDTH_M: Final[float] = 0.006
FORMER_TAB_INSET_M: Final[float] = 0.003  # tab outer edge inboard of the former rim / crutch edge
CRUTCH_TAIL_MARGIN_M: Final[float] = 0.010  # crutch runs this far past the last former's aft face
BOOM_SLOT_LEAD_M: Final[float] = 0.010  # crutch slot for the tail boom starts this far ahead of its first sleeve
BOOM_SLEEVE_FORMERS: Final[int] = 2
PIN_FORMER_GAP_M: Final[float] = 0.010  # the wing-box's aft former sits this far aft of the root pin
MOUNT_TAB_INSET_M: Final[float] = 0.004  # wing-box tabs are this much shorter than the plate, top and bottom
MOUNT_TAB_MIN_HEIGHT_M: Final[float] = 0.004
LIGHTENING_POLAR: Final[tuple[tuple[float, float], ...]] = ((0.50, 90.0), (0.62, 35.0), (0.62, 145.0))  # (r/R, deg)
AREA_EPS_M2: Final[float] = 1e-10
REACH_FACTOR: Final[float] = 2.0  # half-plane cutters extend this many radii
FAR_M: Final[float] = 1.0  # open-ended cutters run this far past the part
UP: Final[int] = 1
DOWN: Final[int] = -1


@dataclass(frozen=True)
class _Mount:
    """Wing-box side plate (right side; the left is its mirror). x runs from the forward socket former's front face
    to the pin former's aft face: the plate body spans the gap between the two, its tabs fill their slots."""

    x_min_m: float
    x_max_m: float
    body_x_m: tuple[float, float]
    z_m: tuple[float, float]  # plate bottom, top
    tab_z_m: tuple[float, float]
    y_inner_m: float
    y_outer_m: float
    holes: tuple[Hole, Hole]  # socket, pin, in (x, z)


@dataclass(frozen=True)
class _Frame:
    """Everything the individual parts need, computed once."""

    layout: AircraftLayout
    settings: FabricationSettings
    components: tuple[PlacedComponent, ...]
    stations: tuple[float, ...]  # firewall first, then the half-formers
    mount: _Mount
    crutch_t_m: float
    former_t_m: float
    firewall_t_m: float

    def inner_radius(self, x_m: float) -> float:
        return fuselage_radius_at(self.layout, x_m) - self.settings.fuselage_cover_thickness_m

    @property
    def former_stations(self) -> tuple[float, ...]:
        return self.stations[1:]

    @property
    def crutch_x(self) -> tuple[float, float]:
        return self.stations[0] + 2.0 * self.firewall_t_m, self.stations[-1] + self.former_t_m + CRUTCH_TAIL_MARGIN_M


# ------------------------------------------------------------------------------------- geometry ---
def _area(shape: Optional[bd.Shape]) -> float:
    return 0.0 if shape is None else sum(f.area for f in shape.faces())


def _common(a: bd.Shape, b: bd.Shape) -> Optional[bd.Shape]:
    """Boolean intersection as one shape (Build123d may return a list of pieces), or None when empty."""
    result = a.intersect(b)
    pieces = result if isinstance(result, list) else ([] if result is None else [result])
    faces = [f for piece in pieces for f in piece.faces()]
    return bd.Compound(faces) if faces else None


def _disc(radius_m: float, y_m: float = 0.0, z_m: float = 0.0) -> bd.Face:
    return circle_face(Hole(y_m, z_m, 2.0 * radius_m))


def _square_at(side_m: float, radius_m: float, angle_deg: float) -> bd.Face:
    """Square of `side_m` centred at `radius_m` along `angle_deg`, edges aligned with that radius."""
    half, rad = side_m / 2.0, math.radians(angle_deg)
    centre = (radius_m * math.cos(rad), radius_m * math.sin(rad), 0.0)
    return rect_face(-half, -half, half, half).moved(bd.Location(centre, (0.0, 0.0, angle_deg)))


def _assert_web(result: bd.Shape, cut: bd.Face, voids: Sequence[bd.Face], region: Optional[bd.Face],
                message: str) -> None:
    """The band of MIN_WEB_M around `cut` (outside every void, inside `region` if given) must be all material."""
    ring = bd.offset(cut, amount=MIN_WEB_M).cut(*voids)
    if region is not None:
        ring = _common(ring, region)
    if ring is not None and _area(ring.cut(result)) > AREA_EPS_M2:
        raise KitUnavailable(message)


def _single_face(shape: bd.Shape, label: str) -> bd.Face:
    faces = shape.faces()
    if len(faces) != 1:
        raise KitUnavailable(f"part {label} falls apart into {len(faces)} pieces")
    return faces[0]


def _profile_corners(layout: AircraftLayout, x0_m: float, x1_m: float) -> list[float]:
    """Fuselage-profile break stations strictly inside (x0, x1): r(x) is linear between them."""
    length = layout.fuselage_length_m
    return [f * length for f, _ in FUSELAGE_PROFILE if x0_m < f * length < x1_m]


def _plane_location(origin: Vec3, x_dir: Vec3, z_dir: Vec3) -> bd.Location:
    return bd.Location(bd.Plane(origin=origin, x_dir=x_dir, z_dir=z_dir))


def _station_location(x_m: float) -> bd.Location:
    """yz-plane part at station x: local X -> y, local Y -> z, extruded aft (+x)."""
    return _plane_location((x_m, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))


# ------------------------------------------------------------------------------------- stations ---
def socket_axis_m(layout: AircraftLayout, y_m: float = 0.0, fraction: float = SPAR_CHORD_STATION) -> tuple[float, float]:
    """(x, z) where a wing's spar axis (or, with `fraction`, its chord line at that fraction on the camber line)
    crosses span station y. Each socket tube is coaxial with its own wing's spar, so it follows the dihedral and
    meets the other at the centreline (y = 0)."""
    wing = layout.wing
    chord = wing.chord_at(y_m)
    dx, dz = camber_point(load_airfoil(layout.airfoil), chord, fraction)
    return wing.x_le_root_m + dx, wing.z_at(y_m) + dz


def _mount_stations(layout: AircraftLayout, spar: SparPlan) -> tuple[float, float, float]:
    """(fore, aft, pin) former stations: either side of the x band the two socket tubes sweep (OD/2 + 10 mm
    clear), and just aft of the root pin, where the wing-box plates end."""
    xs = [socket_axis_m(layout, y)[0] for y in (0.0, spar.joint_y_m)]
    offset = spar.socket.od_m / 2.0 + SOCKET_FORMER_GAP_M
    pin = socket_axis_m(layout, 0.0, PIN_CHORD_FRACTION)[0] + PIN_FORMER_GAP_M  # the root is the widest chord
    return min(xs) - offset, max(xs) + offset, pin


def former_stations(layout: AircraftLayout, spar: SparPlan, settings: FabricationSettings) -> tuple[float, ...]:
    """Every FORMER_PITCH_M from FIRST_FORMER_FRACTION to TAIL_BOOM_START_FRACTION of the length, plus the two
    socket stations and the pin station (`_mount_stations`). Regular stations within STATION_MERGE_M of those,
    or between the two socket stations, are dropped so no former meets the socket. (`settings` keeps the
    part-builder signature uniform; stations do not depend on it.)"""
    length = layout.fuselage_length_m
    first, last = FIRST_FORMER_FRACTION * length, TAIL_BOOM_START_FRACTION * length
    count = math.floor((last - first) / FORMER_PITCH_M + STATION_EPS) + 1
    fore, aft, pin = _mount_stations(layout, spar)

    def is_kept(x: float) -> bool:
        near = any(abs(x - m) < STATION_MERGE_M for m in (fore, aft, pin))
        return not (near or fore < x < aft)

    regular = (first + k * FORMER_PITCH_M for k in range(count))
    return tuple(sorted((*(x for x in regular if is_kept(x)), fore, aft, pin)))


# ------------------------------------------------------------------------------- wing-mount plates ---
def _oblique_diameter_m(layout: AircraftLayout, fraction: float, od_m: float, thickness_m: float) -> float:
    """Round hole that clears a rod of `od_m` crossing a plate of `thickness_m` along that wing's line at
    `fraction` chord: the line leans off the plate normal (y) by dihedral and taper sweep, so its section is a
    slightly longer ellipse (od / cos) that also walks across the thickness (t tan)."""
    (x0, z0), (x1, z1) = socket_axis_m(layout, 0.0, fraction), socket_axis_m(layout, 1.0, fraction)
    lean = math.hypot(x1 - x0, z1 - z0)  # off-normal travel per metre of span
    cos_lean = 1.0 / math.hypot(1.0, lean)
    return od_m / cos_lean + thickness_m * lean


def _plate_holes(layout: AircraftLayout, spar: SparPlan, settings: FabricationSettings, y_m: float) -> tuple[Hole, Hole]:
    """Socket and pin holes (x, z) where each wing's spar axis / pin line crosses the plate mid-plane at y, sized
    for the rod's oblique crossing plus the fit clearance."""
    t = settings.sheet_for("plate").thickness_m
    socket_d = _oblique_diameter_m(layout, SPAR_CHORD_STATION, spar.socket.od_m, t) + settings.fit_clearance_m
    pin_d = _oblique_diameter_m(layout, PIN_CHORD_FRACTION, pin_hole_m(settings) - settings.fit_clearance_m, t)
    socket = Hole(*socket_axis_m(layout, y_m), socket_d)
    pin = Hole(*socket_axis_m(layout, y_m, PIN_CHORD_FRACTION), pin_d + settings.fit_clearance_m)
    return socket, pin


def _plate_outer_y(layout: AircraftLayout, spar: SparPlan, settings: FabricationSettings, rim_m: float) -> float:
    """Outer face y: as far out as possible (never past the skin joint) while both holes keep MIN_WEB_M below the
    rim line sqrt(rim^2 - y^2), which keeps a MIN_WEB_M web on every former the plate passes through."""
    t = settings.sheet_for("plate").thickness_m

    def fits(y: float) -> bool:
        top = max(h.y_m + h.diameter_m / 2.0 for h in _plate_holes(layout, spar, settings, y - t / 2.0)) + MIN_WEB_M
        return y < rim_m and math.sqrt(rim_m**2 - y**2) >= top

    lo, hi = t, spar.joint_y_m
    if fits(hi):
        return hi
    if not fits(lo):
        raise KitUnavailable(f"the wing-box plates do not fit under the skin at the socket ({rim_m * 1e3:.0f} mm rim)")
    for _ in range(BISECTION_ITERATIONS):
        lo, hi = ((lo + hi) / 2.0, hi) if fits((lo + hi) / 2.0) else (lo, (lo + hi) / 2.0)
    return lo


def _mount(layout: AircraftLayout, spar: SparPlan, settings: FabricationSettings) -> _Mount:
    """R12 wing-box plate: from the forward socket former to the pin former (tabbed into both), z from below the
    socket hole (MIN_WEB_M under it) up to the rim line, so components on the axis pass underneath."""
    fore, _, pin_station = _mount_stations(layout, spar)
    t_former, t_plate = settings.sheet_for("former").thickness_m, settings.sheet_for("plate").thickness_m
    x_max = pin_station + t_former
    corners = _profile_corners(layout, fore, x_max)
    rim = min(fuselage_radius_at(layout, x) for x in (fore, x_max, *corners)) - settings.fuselage_cover_thickness_m - MIN_WEB_M
    y_outer = _plate_outer_y(layout, spar, settings, rim)
    socket, pin = _plate_holes(layout, spar, settings, y_outer - t_plate / 2.0)
    z_bottom, z_top = socket.y_m - socket.diameter_m / 2.0 - MIN_WEB_M, math.sqrt(rim**2 - y_outer**2)
    if z_bottom <= settings.sheet_for("crutch").thickness_m / 2.0:
        raise KitUnavailable("the wing-box plates would reach down to the crutch")
    tab_z = (z_bottom + MOUNT_TAB_INSET_M, z_top - MOUNT_TAB_INSET_M)
    if tab_z[1] - tab_z[0] < MOUNT_TAB_MIN_HEIGHT_M:
        raise KitUnavailable(f"the wing-box plates are too shallow ({(z_top - z_bottom) * 1e3:.1f} mm) for their tabs")
    return _Mount(fore, x_max, (fore + t_former, pin_station), (z_bottom, z_top), tab_z,
                  y_outer - t_plate, y_outer, (socket, pin))


def _mount_plates(frame: _Frame) -> tuple[KitPart, KitPart]:
    m = frame.mount
    body = rect_face(m.body_x_m[0], m.z_m[0], m.body_x_m[1], m.z_m[1])
    tabs = (rect_face(m.x_min_m, m.tab_z_m[0], m.body_x_m[0], m.tab_z_m[1]),
            rect_face(m.body_x_m[1], m.tab_z_m[0], m.x_max_m, m.tab_z_m[1]))
    outline = _single_face(body.fuse(*tabs).clean(), "WM-R")
    socket, pin = m.holes
    for hole, others, name in ((socket, [], "socket"), (pin, [socket], "pin")):
        if not keeps_web(outline, hole, others):
            raise KitUnavailable(f"no room for the {name} hole with a {MIN_WEB_M * 1e3:.0f} mm web in WM-R")
    outline = drill(outline, m.holes, "WM-R")
    right = _plane_location((0.0, m.y_outer_m, 0.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0))
    left = _plane_location((0.0, -m.y_inner_m, 0.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0))
    return KitPart("WM-L", "plate", outline, left), KitPart("WM-R", "plate", outline, right)


# ---------------------------------------------------------------------------------- half-formers ---
def _tab_span(radius_m: float) -> tuple[float, float]:
    """y range (right side) of a former tab and its crutch slot."""
    outer = radius_m - FORMER_TAB_INSET_M
    return outer - FORMER_TAB_WIDTH_M, outer


def _half_plane(radius_m: float, sign: int, half_t_m: float) -> bd.Face:
    reach = REACH_FACTOR * radius_m
    return rect_face(-reach, half_t_m, reach, reach) if sign == UP else rect_face(-reach, -reach, reach, -half_t_m)


def _tabs(radius_m: float, sign: int, half_t_m: float) -> list[bd.Face]:
    y0, y1 = _tab_span(radius_m)
    z0, z1 = sorted((0.0, sign * half_t_m))
    return [rect_face(y0, z0, y1, z1), rect_face(-y1, z0, -y0, z1)]


def _stringer_notches(radius_m: float, sign: int, side_m: float) -> list[bd.Face]:
    pitch = FULL_TURN_DEG / STRINGER_COUNT
    angles = [STRINGER_START_DEG + k * pitch for k in range(STRINGER_COUNT // 2)]
    return [_square_at(side_m, radius_m - side_m / 2.0, sign * a) for a in angles]


def _crosses(component: PlacedComponent, x0_m: float, x1_m: float) -> bool:
    gap = COMPONENT_GAP_M / 2.0
    return component.x_min - gap < x1_m and component.x_max + gap > x0_m


def _component_opening(c: PlacedComponent) -> bd.Face:
    """yz bounding box grown by COMPONENT_GAP_M (half each side)."""
    _, ly, lz = c.size_m
    hy, hz = (ly + COMPONENT_GAP_M) / 2.0, (lz + COMPONENT_GAP_M) / 2.0
    return rect_face(c.center_m[1] - hy, c.center_m[2] - hz, c.center_m[1] + hy, c.center_m[2] + hz)


def _former_cuts(frame: _Frame, x_m: float, sign: int, has_sleeve: bool) -> list[tuple[str, bd.Face]]:
    """(what, cutter) for every component, tail-boom sleeve and wing-mount plate passing this half-former."""
    x1 = x_m + frame.former_t_m
    cuts = [(c.name, _component_opening(c)) for c in frame.components if c.is_internal and _crosses(c, x_m, x1)]
    boom = frame.layout.tail_boom
    if has_sleeve and boom is not None:
        sleeve = _disc(_sleeve_diameter_m(frame) / 2.0, boom.y_m, boom.z_m)
        cuts.append(("tail boom", sleeve))
    return cuts + _plate_cuts(frame, x_m, sign)


def _plate_cuts(frame: _Frame, x_m: float, sign: int) -> list[tuple[str, bd.Face]]:
    """Wing-box plate section where the plate passes this upper former, or its tab slot where it ends in it."""
    m, x1 = frame.mount, x_m + frame.former_t_m
    if sign != UP or not (m.x_min_m < x1 and m.x_max_m > x_m):
        return []
    passes = m.body_x_m[0] < x1 and m.body_x_m[1] > x_m
    z0, z1 = m.z_m if passes else m.tab_z_m
    sides = ((m.y_inner_m, m.y_outer_m), (-m.y_outer_m, -m.y_inner_m))
    return [("wing-box plate", rect_face(y0, z0, y1, z1)) for y0, y1 in sides]


def _sleeve_diameter_m(frame: _Frame) -> float:
    """Tail-boom sleeve: the stock boom tube (R13) plus the fit clearance."""
    boom = frame.layout.tail_boom
    if boom is None:
        raise ValueError("tail-boom sleeve requested on a layout without a tail boom")
    return boom_tube(boom.diameter_m, frame.settings).od_m + frame.settings.fit_clearance_m


def _lightening_centres(radius_m: float, sign: int, half_t_m: float) -> list[tuple[float, float]]:
    return [(f * radius_m * math.cos(math.radians(a)), sign * (half_t_m + f * radius_m * math.sin(math.radians(a))))
            for f, a in LIGHTENING_POLAR]


def _half_former(frame: _Frame, index: int, x_m: float, sign: int, has_sleeve: bool) -> KitPart:
    label = f"F-{index:02d}{'U' if sign == UP else 'L'}"
    radius, half_t = frame.inner_radius(x_m), frame.crutch_t_m / 2.0
    half_plane, tabs = _half_plane(radius, sign, half_t), _tabs(radius, sign, half_t)
    semi = _single_face(_common(_disc(radius), half_plane), label)
    base = semi.fuse(*tabs).clean()
    notched = base.cut(*_stringer_notches(radius, sign, frame.settings.stringer_side_m))
    cuts = _former_cuts(frame, x_m, sign, has_sleeve)
    voids = [face for _, face in cuts]
    result = notched.cut(*voids) if voids else notched
    region = half_plane.fuse(*tabs)
    for what, face in cuts:
        _assert_web(result, face, voids, region,
                    f"{label}: the opening for the {what} leaves less than {MIN_WEB_M * 1e3:.0f} mm of the former")
    face = _single_face(result, label)
    face = drill(face, lightening_holes(face, (), _lightening_centres(radius, sign, half_t)), label)
    return KitPart(label, "former", face, _station_location(x_m))


def _half_formers(frame: _Frame) -> list[KitPart]:
    stations = frame.former_stations
    sleeved = set(stations[-BOOM_SLEEVE_FORMERS:])
    return [_half_former(frame, i, x, sign, x in sleeved)
            for i, x in enumerate(stations, start=1) for sign in (UP, DOWN)]


# ------------------------------------------------------------------------------------- firewall ---
def _firewall(frame: _Frame) -> tuple[KitPart, KitPart]:
    x = frame.stations[0]
    disc = _disc(frame.inner_radius(x))
    return (KitPart("FW-1a", "bulkhead", disc, _station_location(x)),
            KitPart("FW-1b", "bulkhead", disc, _station_location(x + frame.firewall_t_m)))


# --------------------------------------------------------------------------------------- crutch ---
def _crutch_outline(frame: _Frame) -> bd.Face:
    x0, x1 = frame.crutch_x
    xs = sorted({x0, x1, *_profile_corners(frame.layout, x0, x1)})
    upper = [(x, frame.inner_radius(x), 0.0) for x in xs]
    lower = [(x, -y, 0.0) for x, y, _ in reversed(upper)]
    return bd.Face(bd.Wire.make_polygon(upper + lower, close=True))


def _crutch_voids(frame: _Frame) -> tuple[list[tuple[str, bd.Face]], list[bd.Face]]:
    """(closed component openings, open tail-boom slot) in aircraft x, y."""
    x0, x1 = frame.crutch_x
    gap = COMPONENT_GAP_M / 2.0
    openings = []
    for c in frame.components:
        if not (c.is_internal and _crosses(c, x0, x1)):
            continue
        hy = (c.size_m[1] + COMPONENT_GAP_M) / 2.0
        openings.append((c.name, rect_face(c.x_min - gap, c.center_m[1] - hy, c.x_max + gap, c.center_m[1] + hy)))
    boom, open_slots = frame.layout.tail_boom, []
    if boom is not None:
        start = frame.former_stations[-BOOM_SLEEVE_FORMERS] - BOOM_SLOT_LEAD_M
        half = _sleeve_diameter_m(frame) / 2.0
        open_slots.append(rect_face(start, boom.y_m - half, x1 + FAR_M, boom.y_m + half))
    return openings, open_slots


def _crutch_slots(frame: _Frame) -> list[tuple[str, bd.Face]]:
    """Slots for the former tabs, each as wide as the former sheet (the wing-box plates sit in the formers)."""
    slots = []
    for i, x in enumerate(frame.former_stations, start=1):
        y0, y1 = _tab_span(frame.inner_radius(x))
        slots += [(f"F-{i:02d}", rect_face(x, a, x + frame.former_t_m, b)) for a, b in ((y0, y1), (-y1, -y0))]
    return slots


def _crutch(frame: _Frame) -> KitPart:
    openings, open_slots = _crutch_voids(frame)
    slots = _crutch_slots(frame)
    voids = [f for _, f in openings] + open_slots
    result = _crutch_outline(frame).cut(*voids, *(f for _, f in slots))
    web = f"{MIN_WEB_M * 1e3:.0f} mm"
    for name, face in openings:
        _assert_web(result, face, voids, None, f"C-1: the opening for the {name} leaves less than {web} of crutch")
    for name, face in slots:
        _assert_web(result, face, [face], None, f"C-1: the slot for {name} leaves less than {web} of crutch")
    x0 = frame.crutch_x[0]
    face = _single_face(result, "C-1").moved(bd.Location((-x0, 0.0, 0.0)))
    return KitPart("C-1", "crutch", face, bd.Location((x0, 0.0, -frame.crutch_t_m / 2.0)))


# ------------------------------------------------------------------------------------------ API ---
def fuselage_parts(
    layout: AircraftLayout, spar: SparPlan, components: Sequence[PlacedComponent], settings: FabricationSettings
) -> tuple[KitPart, ...]:
    """Crutch `C-1`, half-formers `F-nnU`/`F-nnL`, firewall `FW-1a`/`FW-1b` and wing-mount plates `WM-L`/`WM-R`.

    Openings are cut for every internal component; with no components the formers carry only lightening holes
    (plus stringer notches, the tail-boom sleeves and the wing-mount notches, which are part of the outline).
    """
    frame = _Frame(
        layout=layout, settings=settings, components=tuple(components),
        stations=former_stations(layout, spar, settings), mount=_mount(layout, spar, settings),
        crutch_t_m=settings.sheet_for("crutch").thickness_m, former_t_m=settings.sheet_for("former").thickness_m,
        firewall_t_m=settings.sheet_for("bulkhead").thickness_m,
    )
    formers = _half_formers(frame)  # first, so a too-big component is reported against the former it breaks
    return (_crutch(frame), *formers, *_firewall(frame), *_mount_plates(frame))
