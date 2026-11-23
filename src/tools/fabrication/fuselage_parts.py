"""Fuselage parts for the laser-cut kit: crutch, half-formers, firewall, wing-box side plates and tail-boom mounts
(frame convention: see `kit.py`).

- Firewall `FW-1a`, `FW-1b`: two solid discs laminated face to face. The cruise motor is a tractor ahead of the
  nose, so the firewall takes the first station (FIRST_FORMER_FRACTION of the length), moved forward when a
  component in the pod starts earlier (a chin payload): it then ends COMPONENT_GAP_M ahead of that component.
- Crutch `C-1`: a horizontal 3 mm plate centred on z = 0 (it spans z = +-t/2), as wide as the fuselage inside
  the covering, from the firewall's aft face to just aft of the last former (never into the boom mounts). Its
  outline is in aircraft x, y, shifted by the plate origin x.
- Half-formers `F-nnU` / `F-nnL`: semi-discs of r(x) - cover above and below the crutch, with two tabs that drop
  into crutch slots (U and L tabs meet at z = 0), stringer notches and lightening holes. A former lies in the yz
  plane and extrudes aft (local X -> y, local Y -> z, local Z -> +x), so its location X is the station and its
  outline radius is r(station) - cover. A component cutting a half-former in two gives pieces `F-nnLa`, `F-nnLb`,
  each held by its own tab.
- Wing-box side plates `WM-L`, `WM-R`: vertical plates (xz plane) in the upper fuselage, from the forward socket
  former to a former just aft of the root pin, tabbed into both; formers in between get the plate's section cut
  out. Each carries its wing's socket tube (coaxial with that wing's spar, so following the dihedral; the two
  meet at the centreline) and the anti-rotation pin. They stop MIN_WEB_M under the socket hole, so components
  on the axis pass underneath. Local X -> x, local Y -> z; WM-R extrudes toward -y from its outer face, WM-L is
  its mirror. When the plain plate cannot bear its wing's shear on the socket (R27: sigma_b = F / (OD t) <=
  ply_bearing_allow_pa / FS) or its web under the socket shears out under the downward load (R32/R33:
  F / (2 w t) <= ply_shear_allow_pa / FS), a socket-boss doubler `WM-RD` / `WM-LD` (an annulus SOCKET_BOSS_WIDTH_M
  wide plus a skirt below the hole grown until the web passes, cut from plate stock) is glued round the socket hole
  on each plate's inner face; none is cut when the plate suffices. R31 pin bosses `WM-RPD` / `WM-LPD` likewise.
  Each plate also carries the R36 wing bolt's M5 clearance hole between the socket and the pin; the bolt's washer
  seats on the inner face clear of the bosses, and its head clear of every component (`wing_retention`).
- Boom mounts `BM-1`, `BM-2`: 3 mm bulkheads in the tail cone carrying the stock tail-boom tube, one just aft of
  the boom root, one as far aft as a MIN_WEB_M rim around the boom hole allows (at least 30 mm apart).

Every component whose box crosses a former, the crutch or a plate gets an opening (or a notch where it breaks
through the outline); openings, slots and notches keep a MIN_WEB_M web, and anything that would break a part
raises `KitUnavailable`. Every crutch side rail beside a component opening keeps at least `crutch_rail_min_m`
(R28: the rails carry fuselage bending). Slots are cut at the mating stock's nominal thickness; kerf is applied at
DXF export.
"""

import math
from dataclasses import dataclass
from typing import Final, NamedTuple, Optional, Sequence

import build123d as bd

from src.schemas.fabrication import FabricationSettings
from src.tools.aero_cad.airfoil_geometry import load_airfoil
from src.tools.aero_cad.components import COMPONENT_GAP_M, PlacedComponent
from src.tools.aero_cad.layout import (
    TAIL_BOOM_START_FRACTION,
    AircraftLayout,
    fuselage_radius_at,
)
from src.tools.aero_cad.structure import SPAR_CHORD_STATION, STATION_EPS
from src.tools.fabrication.kit import KitPart, Vec3
from src.tools.fabrication.rib_outline import (
    Hole,
    circle_face,
    common,
    drill,
    keeps_web,
    lightening_holes,
    oblique_diameter_m,
    rect_face,
    shape_area,
    single_face,
)
from src.tools.fabrication.skin import max_skin_radius_m, min_skin_radius_m, profile_corners
from src.tools.fabrication.tubes import (
    BISECTION_ITERATIONS,
    FIT_EPS_M,
    MIN_WEB_M,
    KitUnavailable,
    PlateLoads,
    SparPlan,
    boom_tube,
    overwrap_layers,
    size_overwrap,
)
from src.tools.fabrication.wing_parts import (
    PIN_CHORD_FRACTION,
    line_lean,
    pin_hole_m,
    socket_axis_m,
)
from src.tools.fabrication.wing_box_bosses import (
    BOSS_LABELS,
    PIN_BOSS_LABELS,
    check_skirt_clears_crutch,
    pin_boss_blank,
    socket_boss_depth_m,
    socket_boss_outline,
)
from src.tools.fabrication.wing_retention import (
    BOLT_CHORD_FRACTION,
    BOLT_HEAD_STACK_M,
    BOLT_THREAD_MARGIN_M,
    BOLT_WASHER_DIAMETER_M,
    M5_CLEARANCE_HOLE_M,
    M5_DIAMETER_M,
    M5_NUT_HEIGHT_M,
    check_clear_of_holes,
    check_clear_of_outlines,
    check_head_clear,
)

FORMER_PITCH_M: Final[float] = 0.12
FIRST_FORMER_FRACTION: Final[float] = 0.10  # of fuselage length
STATION_MERGE_M: Final[float] = 0.020  # a regular station this close to a mandatory one is dropped
SOCKET_FORMER_GAP_M: Final[float] = 0.010  # socket stations sit this far beyond the socket's outer surface
PIN_FORMER_GAP_M: Final[float] = 0.010  # the wing-box's aft former sits this far aft of the root pin
STRINGER_COUNT: Final[int] = 8
FULL_TURN_DEG: Final[float] = 360.0
STRINGER_START_DEG: Final[float] = FULL_TURN_DEG / STRINGER_COUNT / 2.0  # none on the crutch plane
FORMER_TAB_WIDTH_M: Final[float] = 0.006
FORMER_TAB_INSET_M: Final[float] = 0.003  # tab outer edge inboard of the former rim / crutch edge
CRUTCH_TAIL_MARGIN_M: Final[float] = 0.010  # crutch runs this far past the last former's aft face
MOUNT_TAB_INSET_M: Final[float] = 0.004  # wing-box tabs are this much shorter than the plate, top and bottom
MOUNT_TAB_MIN_HEIGHT_M: Final[float] = 0.004
BOOM_MOUNT_MIN_SPACING_M: Final[float] = 0.030
LIGHTENING_POLAR: Final[tuple[tuple[float, float], ...]] = ((0.50, 90.0), (0.62, 35.0), (0.62, 145.0))  # (r/R, deg)
AREA_EPS_M2: Final[float] = 1e-10
REACH_FACTOR: Final[float] = 2.0  # half-plane cutters extend this many radii
UP: Final[int] = 1
DOWN: Final[int] = -1
WEB_TEXT: Final[str] = f"{MIN_WEB_M * 1e3:.0f} mm"
SLIVER_MAX_RUN_M: Final[float] = MIN_WEB_M  # D15: a sub-web band along the rim may run no further than this
RAIL_PROBE_INSET_M: Final[float] = 1e-4  # crutch rails are measured just inside each opening's fore and aft ends
SPAN_REACH_M: Final[float] = 10.0  # probe lines run this far either side of any part

Cut = tuple[str, bd.Face]  # (what passes through, cutter in the part's local XY)


class PlateHoles(NamedTuple):
    """A wing-box plate's holes in its local (x, z): the socket, the anti-rotation pin and the R36 wing bolt."""

    socket: Hole
    pin: Hole
    bolt: Hole


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
    holes: "PlateHoles"  # in (x, z)


@dataclass(frozen=True)
class _Frame:
    """Everything the individual parts need, computed once."""

    layout: AircraftLayout
    settings: FabricationSettings
    components: tuple[PlacedComponent, ...]
    stations: tuple[float, ...]  # firewall first, then the half-formers
    boom_mounts: tuple[float, ...]  # BM-1, BM-2 stations (empty without a tail boom)
    mount: _Mount
    crutch_t_m: float
    former_t_m: float
    bulkhead_t_m: float

    def inner_radius(self, x_m: float) -> float:
        return fuselage_radius_at(self.layout, x_m) - self.settings.fuselage_cover_thickness_m

    def former_radius(self, x_m: float) -> float:
        """D12: a half-former's outline radius, at the smaller of its two faces (the aft face on the tail cone), so
        neither face reaches into the covering."""
        return min(self.inner_radius(x_m), self.inner_radius(x_m + self.former_t_m))

    def tab_radius(self, x_m: float) -> float:
        """Radius the tabs and crutch slots of the former at station x are inset from: the smaller of the former's
        front face and a web beyond its aft face, so a slot keeps MIN_WEB_M to the crutch edge on a steep cone."""
        return min(self.inner_radius(x_m), self.inner_radius(x_m + self.former_t_m + MIN_WEB_M))

    @property
    def former_stations(self) -> tuple[float, ...]:
        return self.stations[1:]

    @property
    def crutch_x(self) -> tuple[float, float]:
        end = self.stations[-1] + self.former_t_m + CRUTCH_TAIL_MARGIN_M
        boom = self.layout.tail_boom
        return self.stations[0] + 2.0 * self.bulkhead_t_m, end if boom is None else min(end, boom.x_start_m)


# ------------------------------------------------------------------------------------- geometry ---
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
        ring = common(ring, region)
    if ring is not None and shape_area(ring.cut(result)) > AREA_EPS_M2:
        raise KitUnavailable(message)


def _cut_openings(base: bd.Face, rim: bd.Face, cuts: Sequence[Cut], region: Optional[bd.Face],
                  protected: Sequence[bd.Face], label: str, part: str) -> list[bd.Face]:
    """Cut every opening from `base`; return the resulting pieces.

    An opening inside `rim` is a closed hole and must keep a MIN_WEB_M band of material (within `region`) all
    round. An opening breaking through `rim` is a notch. Every opening must stay MIN_WEB_M clear of each
    `protected` feature (tabs). Openings may merge with each other."""
    if not cuts:
        return [base]
    voids = [face for _, face in cuts]
    result = base.cut(*voids)
    for name, cut in cuts:
        message = f"{label}: the opening for the {name} leaves less than {WEB_TEXT} of the {part}"
        if any(shape_area(common(cut, bd.offset(p, amount=MIN_WEB_M))) > AREA_EPS_M2 for p in protected):
            raise KitUnavailable(message)
        if shape_area(cut.cut(rim)) <= AREA_EPS_M2:  # a closed hole; a notch through the rim needs only the tab check
            _assert_web(result, cut, voids, region, message)
    return list(result.faces())


def _plane_location(origin: Vec3, x_dir: Vec3, z_dir: Vec3) -> bd.Location:
    return bd.Location(bd.Plane(origin=origin, x_dir=x_dir, z_dir=z_dir))


def station_location(x_m: float) -> bd.Location:
    """yz-plane part at station x: local X -> y, local Y -> z, extruded aft (+x)."""
    return _plane_location((x_m, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))


def _overlaps(c: PlacedComponent, axis: int, lo_m: float, hi_m: float, grow_m: float = 0.0) -> bool:
    """The component's box (grown by `grow_m` each side) overlaps [lo, hi] along aircraft axis 0/1/2."""
    half = c.size_m[axis] / 2.0 + grow_m
    return c.center_m[axis] - half < hi_m and c.center_m[axis] + half > lo_m


def _crossing(frame: _Frame, x0_m: float, x1_m: float) -> list[PlacedComponent]:
    """Components whose box, plus half the component gap, reaches into the slab x0..x1."""
    return [c for c in frame.components if _overlaps(c, 0, x0_m, x1_m, COMPONENT_GAP_M / 2.0)]


def _yz_opening(c: PlacedComponent) -> bd.Face:
    """yz bounding box grown by COMPONENT_GAP_M (half each side)."""
    _, ly, lz = c.size_m
    hy, hz = (ly + COMPONENT_GAP_M) / 2.0, (lz + COMPONENT_GAP_M) / 2.0
    return rect_face(c.center_m[1] - hy, c.center_m[2] - hz, c.center_m[1] + hy, c.center_m[2] + hz)


def _assert_clear(frame: _Frame, label: str, x_m: float, outline: bd.Face) -> None:
    """Solid yz parts (firewall, boom mounts) take no openings: no component may cross them."""
    for c in _crossing(frame, x_m, x_m + frame.bulkhead_t_m):
        if shape_area(common(outline, _yz_opening(c))) > AREA_EPS_M2:
            raise KitUnavailable(f"{label}: the {c.name} crosses this bulkhead")


# ------------------------------------------------------------------------------------- stations ---
def _mount_stations(layout: AircraftLayout, spar: SparPlan) -> tuple[float, float, float]:
    """(fore, aft, pin) former stations: either side of the x band the two socket tubes sweep (OD/2 + 10 mm
    clear), and just aft of the root pin, where the wing-box plates end."""
    xs = [socket_axis_m(layout, y)[0] for y in (0.0, spar.joint_y_m)]
    offset = spar.socket.od_m / 2.0 + SOCKET_FORMER_GAP_M
    pin = socket_axis_m(layout, 0.0, PIN_CHORD_FRACTION)[0] + PIN_FORMER_GAP_M  # the root is the widest chord
    return min(xs) - offset, max(xs) + offset, pin


def _in_pod(layout: AircraftLayout, c: PlacedComponent) -> bool:
    """The component's box reaches inside the fuselage skin: within the pod's length, and its yz box overlaps the
    largest skin section over its x range (r is piecewise linear, so ends and profile corners bound it)."""
    x0, x1 = max(c.x_min, 0.0), min(c.x_max, layout.fuselage_length_m)
    if x0 >= x1:
        return False
    radius = max_skin_radius_m(layout, x0, x1)
    _, ly, lz = c.size_m
    box = rect_face(c.center_m[1] - ly / 2.0, c.center_m[2] - lz / 2.0, c.center_m[1] + ly / 2.0, c.center_m[2] + lz / 2.0)
    return shape_area(common(box, _disc(radius))) > AREA_EPS_M2


def firewall_station(layout: AircraftLayout, settings: FabricationSettings,
                      components: Sequence[PlacedComponent]) -> float:
    """FIRST_FORMER_FRACTION of the length, or earlier so the firewall pair ends COMPONENT_GAP_M ahead of the
    foremost component in the pod (`_in_pod`: the tractor motor ahead of the nose and lift rotors out on the
    booms don't count)."""
    station = FIRST_FORMER_FRACTION * layout.fuselage_length_m
    in_pod = [c.x_min for c in components if _in_pod(layout, c)]
    if in_pod:
        station = min(station, min(in_pod) - COMPONENT_GAP_M - 2.0 * settings.sheet_for("bulkhead").thickness_m)
    if fuselage_radius_at(layout, station) <= settings.fuselage_cover_thickness_m:
        raise KitUnavailable(f"no room for a firewall ahead of the components (x = {station:.3f} m)")
    return station


def _check_no_collision(mandatory: Sequence[float]) -> None:
    ordered = sorted(mandatory)
    for a, b in zip(ordered, ordered[1:]):
        if b - a < STATION_MERGE_M:
            raise KitUnavailable(
                f"former stations collide: x = {a:.3f} m and {b:.3f} m (firewall / wing box) are closer than "
                f"{STATION_MERGE_M * 1e3:.0f} mm"
            )


def _check_layout_order(layout: AircraftLayout, firewall_m: float, fore_m: float, pin_aft_m: float) -> None:
    """D13: the firewall comes first, and the wing box (socket former to the pin former's aft face) lies inside the
    pod, ahead of the tail-boom root."""
    if fore_m <= firewall_m:
        raise KitUnavailable(f"the wing box (socket former at x = {fore_m:.3f} m) lies ahead of the firewall "
                             f"(x = {firewall_m:.3f} m)")
    end = layout.fuselage_length_m if layout.tail_boom is None else layout.tail_boom.x_start_m
    if pin_aft_m > end:
        raise KitUnavailable(f"the wing box (pin former ending at x = {pin_aft_m:.3f} m) lies aft of the "
                             f"{'pod' if layout.tail_boom is None else 'tail-boom root'} (x = {end:.3f} m)")


def _bolt_band_m(layout: AircraftLayout, spar: SparPlan, settings: FabricationSettings) -> tuple[float, float]:
    """R38: the x band no former may stand in: where the wing bolt line runs from the centreline out past its nut
    (it leans with taper), widened by the bolt's radius, the fit clearance and MIN_WEB_M."""
    reach = spar.joint_y_m + settings.sheet_for("doubler").thickness_m + M5_NUT_HEIGHT_M + BOLT_THREAD_MARGIN_M
    xs = [socket_axis_m(layout, y, BOLT_CHORD_FRACTION)[0] for y in (0.0, reach)]
    band = M5_DIAMETER_M / 2.0 + settings.fit_clearance_m + MIN_WEB_M
    return min(xs) - band, max(xs) + band


def former_stations(layout: AircraftLayout, spar: SparPlan, settings: FabricationSettings,
                    components: Sequence[PlacedComponent] = ()) -> tuple[float, ...]:
    """Firewall station first (`firewall_station`, taking the place of the first regular station), then every
    FORMER_PITCH_M from FIRST_FORMER_FRACTION up to TAIL_BOOM_START_FRACTION of the length (and clear of the boom
    root), plus the two socket stations and the pin station (`_mount_stations`). Regular stations within
    STATION_MERGE_M of those, between the two socket stations, or whose slab meets the wing bolt's band
    (`_bolt_band_m`, R38) are dropped, so no former meets the socket or the bolt.
    Mandatory stations closer than STATION_MERGE_M, a wing box ahead of the firewall or aft of the boom root (or the
    pod's end) raise `KitUnavailable`."""
    length, t = layout.fuselage_length_m, settings.sheet_for("former").thickness_m
    first, last = FIRST_FORMER_FRACTION * length, TAIL_BOOM_START_FRACTION * length
    if layout.tail_boom is not None:
        last = min(last, layout.tail_boom.x_start_m - t)  # a former slab never reaches the boom root
    count = math.floor((last - first) / FORMER_PITCH_M + STATION_EPS) + 1
    firewall = firewall_station(layout, settings, components)
    fore, aft, pin = _mount_stations(layout, spar)
    _check_layout_order(layout, firewall, fore, pin + t)
    _check_no_collision((firewall, fore, aft, pin))

    bolt_lo, bolt_hi = _bolt_band_m(layout, spar, settings)

    def is_kept(x: float) -> bool:
        near = any(abs(x - m) < STATION_MERGE_M for m in (firewall, fore, aft, pin))
        on_bolt = x < bolt_hi and x + t > bolt_lo  # R38: the slab would stand in the wing bolt's path
        return not (near or fore < x < aft or on_bolt)

    regular = (first + k * FORMER_PITCH_M for k in range(1, count))
    return (firewall, *sorted((*(x for x in regular if is_kept(x)), fore, aft, pin)))


def _boom_mount_stations(layout: AircraftLayout, settings: FabricationSettings) -> tuple[float, ...]:
    """BM-1 at the boom root + t, BM-2 at the aft-most station whose aft face keeps a MIN_WEB_M rim round the
    stock boom tube hole (the tail cone narrows aft)."""
    boom = layout.tail_boom
    if boom is None:
        return ()
    t, cover = settings.sheet_for("bulkhead").thickness_m, settings.fuselage_cover_thickness_m
    hole = boom_tube(boom.diameter_m, settings).od_m + settings.fit_clearance_m

    def has_rim(x: float) -> bool:
        return fuselage_radius_at(layout, x + t) - cover - hole / 2.0 - abs(boom.z_m) - abs(boom.y_m) >= MIN_WEB_M

    first = boom.x_start_m + t
    if not has_rim(first):
        raise KitUnavailable(f"BM-1: the tail cone is too thin for the {hole * 1e3:.0f} mm boom hole")
    lo, hi = first, layout.fuselage_length_m - t
    for _ in range(BISECTION_ITERATIONS):
        lo, hi = ((lo + hi) / 2.0, hi) if has_rim((lo + hi) / 2.0) else (lo, (lo + hi) / 2.0)
    if lo - first < BOOM_MOUNT_MIN_SPACING_M:
        raise KitUnavailable(f"BM-1 and BM-2 would be only {(lo - first) * 1e3:.0f} mm apart "
                             f"(need {BOOM_MOUNT_MIN_SPACING_M * 1e3:.0f} mm)")
    return first, lo


# ------------------------------------------------------------------------------- wing-box plates ---
def _oblique_diameter_m(layout: AircraftLayout, fraction: float, od_m: float, thickness_m: float) -> float:
    """Round hole that clears a rod of `od_m` crossing a plate of `thickness_m` along that wing's line at
    `fraction` chord: the line leans off the plate normal (y) by dihedral and taper sweep (`line_lean`), so its
    section is a slightly longer ellipse (od / cos) that also walks across the thickness (t tan)."""
    lean = line_lean(layout.wing, load_airfoil(layout.airfoil), fraction)
    return oblique_diameter_m(od_m, thickness_m, lean)


def _plate_holes(layout: AircraftLayout, spar: SparPlan, settings: FabricationSettings, y_m: float) -> PlateHoles:
    """Socket, pin and wing-bolt holes (x, z) where each wing's spar axis / pin line / bolt line crosses the plate
    slab whose mid-plane is at y, sized for the rod's oblique crossing plus the fit clearance (the bolt's M5
    clearance hole needs none)."""
    t, clearance = settings.sheet_for("plate").thickness_m, settings.fit_clearance_m
    socket_d = _oblique_diameter_m(layout, SPAR_CHORD_STATION, spar.socket.od_m, t) + clearance
    pin_nominal = pin_hole_m(spar.pin.rod_m, settings) - clearance
    pin_d = _oblique_diameter_m(layout, PIN_CHORD_FRACTION, pin_nominal, t) + clearance
    bolt_d = _oblique_diameter_m(layout, BOLT_CHORD_FRACTION, M5_CLEARANCE_HOLE_M, t)
    return PlateHoles(Hole(*socket_axis_m(layout, y_m), socket_d),
                      Hole(*socket_axis_m(layout, y_m, PIN_CHORD_FRACTION), pin_d),
                      Hole(*socket_axis_m(layout, y_m, BOLT_CHORD_FRACTION), bolt_d))


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
    skin = min_skin_radius_m(layout, fore, x_max)
    rim = skin - settings.fuselage_cover_thickness_m - MIN_WEB_M
    y_outer = _plate_outer_y(layout, spar, settings, rim)
    holes = _plate_holes(layout, spar, settings, y_outer - t_plate / 2.0)
    z_bottom = holes.socket.y_m - holes.socket.diameter_m / 2.0 - MIN_WEB_M
    z_top = math.sqrt(rim**2 - y_outer**2)
    if z_bottom <= settings.sheet_for("crutch").thickness_m / 2.0:
        raise KitUnavailable("the wing-box plates would reach down to the crutch")
    tab_z = (z_bottom + MOUNT_TAB_INSET_M, z_top - MOUNT_TAB_INSET_M)
    if tab_z[1] - tab_z[0] < MOUNT_TAB_MIN_HEIGHT_M:
        raise KitUnavailable(f"the wing-box plates are too shallow ({(z_top - z_bottom) * 1e3:.1f} mm) for their tabs")
    return _Mount(fore, x_max, (fore + t_former, pin_station), (z_bottom, z_top), tab_z,
                  y_outer - t_plate, y_outer, holes)


def _plate_gap_m(mount: _Mount, depth_m: Optional[float], settings: FabricationSettings) -> float:
    """Clear width between the wing-box plates' inner faces, or the socket bosses' (plate stock) when fitted."""
    boss_m = 0.0 if depth_m is None else settings.sheet_for("plate").thickness_m
    return 2.0 * (mount.y_inner_m - boss_m)


def wing_box_plate_gap_m(layout: AircraftLayout, spar: SparPlan, settings: FabricationSettings,
                         loads: PlateLoads) -> float:
    """The clear width the centreline overwrap must fit in (R24), as the kit builds the plates and socket bosses."""
    return _plate_gap_m(_mount(layout, spar, settings), socket_boss_depth_m(spar, settings, loads), settings)


def wing_box_floor_m(layout: AircraftLayout, spar: SparPlan, settings: FabricationSettings,
                     loads: PlateLoads) -> float:
    """Lowest z of the wing-box structure a component must pass beneath: the plates' bottom edge, the wing bolt's
    head and washer inside the plates (R38), the socket bosses' bottom when `loads` need them (R32), or the
    underside of the socket tubes' centreline overwrap (R24, at its
    wrapped OD), where they sit lowest (the dihedral lifts them outboard). Pin bosses stay within the plates. Raises
    `KitUnavailable` when no overwrap within the layer cap carries the centreline moment, the overwrap is longer than
    the plate gap, no boss suffices, or the boss's skirt would reach the crutch (the kit's own checks, so pod sizing
    never accepts a pod the kit rejects)."""
    wrap_m = overwrap_layers(spar.centre_moment_nm, spar.socket, settings) * settings.overwrap_layer_thickness_m
    mount = _mount(layout, spar, settings)
    depth = socket_boss_depth_m(spar, settings, loads)
    size_overwrap(spar.centre_moment_nm, spar.socket, _plate_gap_m(mount, depth, settings), settings)  # R24 fit
    bolt = _inner_face_holes(layout, spar, settings, mount).bolt  # R38: the head and washer stand inboard of it
    floors = [mount.z_m[0], socket_axis_m(layout, 0.0)[1] - spar.socket.od_m / 2.0 - wrap_m,
              bolt.y_m - BOLT_WASHER_DIAMETER_M / 2.0]
    if depth is not None:
        hole = _inner_face_holes(layout, spar, settings, mount).socket
        check_skirt_clears_crutch(hole, depth, settings.sheet_for("crutch").thickness_m)  # as the kit will (M7)
        floors.append(hole.y_m - hole.diameter_m / 2.0 - depth)
    return min(floors)


def _plate_component_cuts(frame: _Frame, side: int) -> list[Cut]:
    """xz openings for components whose box (grown by half the gap) reaches into this side's plate slab."""
    m, gap = frame.mount, COMPONENT_GAP_M / 2.0
    y0, y1 = sorted((side * m.y_inner_m, side * m.y_outer_m))
    hits = [c for c in frame.components if _overlaps(c, 0, m.x_min_m, m.x_max_m, gap)
            and _overlaps(c, 1, y0, y1, gap) and _overlaps(c, 2, m.z_m[0], m.z_m[1], gap)]
    return [(c.name, rect_face(c.x_min - gap, c.center_m[2] - c.size_m[2] / 2.0 - gap,
                               c.x_max + gap, c.center_m[2] + c.size_m[2] / 2.0 + gap)) for c in hits]


def _mount_plate(frame: _Frame, side: int) -> KitPart:
    m = frame.mount
    label = "WM-R" if side > 0 else "WM-L"
    body = rect_face(m.body_x_m[0], m.z_m[0], m.body_x_m[1], m.z_m[1])
    tabs = (rect_face(m.x_min_m, m.tab_z_m[0], m.body_x_m[0], m.tab_z_m[1]),
            rect_face(m.body_x_m[1], m.tab_z_m[0], m.x_max_m, m.tab_z_m[1]))
    plain = single_face(body.fuse(*tabs).clean(), label)
    pieces = _cut_openings(plain, plain, _plate_component_cuts(frame, side), None, tabs, label, "plate")
    outline = single_face(bd.Compound(pieces), label)
    socket, pin, bolt = m.holes
    for hole, others, name in ((socket, [], "socket"), (pin, [socket], "pin"), (bolt, [socket, pin], "wing bolt")):
        if not keeps_web(outline, hole, others):
            raise KitUnavailable(f"no room for the {name} hole with a {WEB_TEXT} web in {label}")
    outline = drill(outline, m.holes, label)
    origin_y = m.y_outer_m if side > 0 else -m.y_inner_m  # both extrude toward -y
    return KitPart(label, "plate", outline, _plane_location((0.0, origin_y, 0.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0)))


def _inner_face_holes(layout: AircraftLayout, spar: SparPlan, settings: FabricationSettings,
                      mount: _Mount) -> PlateHoles:
    """Socket, pin and bolt holes where the lines cross a slab on the plate's inner face (a boss, or the bolt's
    washer)."""
    t = settings.sheet_for("plate").thickness_m
    return _plate_holes(layout, spar, settings, mount.y_inner_m - t / 2.0)


def _bosses(frame: _Frame, hole: Hole, blanks: dict[int, bd.Face], labels: dict[int, str]) -> list[KitPart]:
    """A boss doubler of plate stock on each plate's inner face: that side's blank drilled with `hole`."""
    t, m = frame.settings.sheet_for("plate").thickness_m, frame.mount
    origins = {1: m.y_inner_m, -1: -m.y_inner_m + t}  # both extrude toward -y, onto the plate's inner face
    return [KitPart(label, "plate", drill(blanks[side], (hole,), label),
                    _plane_location((0.0, origins[side], 0.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0)))
            for side, label in sorted(labels.items())]


def _wing_box_bosses(frame: _Frame, spar: SparPlan, loads: PlateLoads,
                     plates: dict[int, KitPart]) -> list[KitPart]:
    """R27/R32/R33 socket bosses (annulus + skirt below) and R31 pin bosses (an annulus clipped to each plate's real
    material, so it adds no height), each only when needed."""
    socket, pin, _ = _inner_face_holes(frame.layout, spar, frame.settings, frame.mount)
    depth = socket_boss_depth_m(spar, frame.settings, loads)
    parts: list[KitPart] = []
    if depth is not None:
        check_skirt_clears_crutch(socket, depth, frame.crutch_t_m)
        blank = socket_boss_outline(socket, depth)
        parts += _bosses(frame, socket, {1: blank, -1: blank}, BOSS_LABELS)
    if spar.pin.plate_boss:
        labels = dict(zip((1, -1), PIN_BOSS_LABELS))
        plate_pin = frame.mount.holes.pin
        blanks = {side: pin_boss_blank(plates[side].outline, plate_pin, pin, labels[side]) for side in labels}
        parts += _bosses(frame, pin, blanks, labels)
    return parts


def _check_bolt_seats(frame: _Frame, spar: SparPlan, bosses: Sequence[KitPart]) -> None:
    """R36: the bolt's washer seats on each plate's inner face MIN_WEB_M clear of the socket, the pin and every boss,
    and its head (BOLT_HEAD_STACK_M proud of that face) is clear of every component."""
    m = frame.mount
    socket, pin, bolt = _inner_face_holes(frame.layout, spar, frame.settings, m)
    washer = Hole(bolt.x_m, bolt.y_m, BOLT_WASHER_DIAMETER_M)
    half = BOLT_WASHER_DIAMETER_M / 2.0
    for side, label in ((1, "WM-R"), (-1, "WM-L")):
        check_clear_of_holes(washer, (socket, pin), label, "the socket or the pin")
        check_clear_of_outlines(washer, [b.outline for b in bosses], label, "a boss")
        y0, y1 = sorted((side * m.y_inner_m, side * (m.y_inner_m - BOLT_HEAD_STACK_M)))
        head = ((bolt.x_m - half, y0, bolt.y_m - half), (bolt.x_m + half, y1, bolt.y_m + half))
        check_head_clear(frame.components, head, label)


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


def _former_cuts(frame: _Frame, x_m: float, sign: int) -> list[Cut]:
    """(what, cutter) for every component and wing-box plate passing this half-former."""
    cuts = [(c.name, _yz_opening(c)) for c in _crossing(frame, x_m, x_m + frame.former_t_m)]
    return cuts + _plate_cuts(frame, x_m, sign)


def _plate_cuts(frame: _Frame, x_m: float, sign: int) -> list[Cut]:
    """Wing-box plate section where the plate passes this upper former, or its tab slot where it ends in it."""
    m, x1 = frame.mount, x_m + frame.former_t_m
    if sign != UP or not (m.x_min_m < x1 and m.x_max_m > x_m):
        return []
    passes = m.body_x_m[0] < x1 and m.body_x_m[1] > x_m
    z0, z1 = m.z_m if passes else m.tab_z_m
    sides = ((m.y_inner_m, m.y_outer_m), (-m.y_outer_m, -m.y_inner_m))
    return [("wing-box plate", rect_face(y0, z0, y1, z1)) for y0, y1 in sides]


def _lightening_centres(radius_m: float, sign: int, half_t_m: float) -> list[tuple[float, float]]:
    return [(f * radius_m * math.cos(math.radians(a)), sign * (half_t_m + f * radius_m * math.sin(math.radians(a))))
            for f, a in LIGHTENING_POLAR]


def _held_by_a_tab(piece: bd.Face, tabs: Sequence[bd.Face]) -> bool:
    return any(abs(shape_area(common(piece, tab)) - tab.area) <= AREA_EPS_M2 for tab in tabs)


def _thin_run_m(offset_m: float, lo_m: float, hi_m: float, radius_m: float) -> float:
    """Along an opening's side at `offset_m` from the axis (material beyond it, away from the opening) spanning
    [lo, hi]: how far the material between the side and a rim of `radius_m` is present but under MIN_WEB_M wide."""
    wide = offset_m + MIN_WEB_M
    if offset_m >= radius_m or wide <= 0.0:
        return 0.0  # no material beyond the side, or at least a web of it everywhere
    present = radius_m if offset_m <= 0.0 else math.sqrt(radius_m**2 - offset_m**2)
    thin_from = 0.0 if wide >= radius_m else math.sqrt(radius_m**2 - wide**2)
    runs = ((thin_from, present), (-present, -thin_from))
    return sum(max(0.0, min(hi_m, b) - max(lo_m, a)) for a, b in runs)


def notch_sliver_m(radius_m: float, box: tuple[float, float, float, float]) -> float:
    """D15: the longest run along a side of the rectangular opening `box` (x0, y0, x1, y1, local) over which the
    former between that side and its rim of `radius_m` is thinner than MIN_WEB_M (sides mirrored onto +offset)."""
    x0, y0, x1, y1 = box
    sides = ((x1, y0, y1), (-x0, y0, y1), (y1, x0, x1), (-y0, x0, x1))
    return max(_thin_run_m(offset, lo, hi, radius_m) for offset, lo, hi in sides)


def _check_slivers(radius_m: float, rim: bd.Face, cuts: Sequence[Cut], label: str) -> None:
    """A notch (an opening through the rim) must not leave a sliver thinner than MIN_WEB_M along the rim for more
    than SLIVER_MAX_RUN_M; closed holes keep their web by `_cut_openings`."""
    for name, cut in cuts:
        if shape_area(cut.cut(rim)) <= AREA_EPS_M2:
            continue
        b = cut.bounding_box()
        if notch_sliver_m(radius_m, (b.min.X, b.min.Y, b.max.X, b.max.Y)) > SLIVER_MAX_RUN_M:
            raise KitUnavailable(f"{label}: the opening for the {name} leaves a sliver of former thinner than "
                                 f"{WEB_TEXT} along the rim")


def _half_former(frame: _Frame, index: int, x_m: float, sign: int) -> list[KitPart]:
    label = f"F-{index:02d}{'U' if sign == UP else 'L'}"
    radius, half_t = frame.former_radius(x_m), frame.crutch_t_m / 2.0
    half_plane, tabs = _half_plane(radius, sign, half_t), _tabs(frame.tab_radius(x_m), sign, half_t)
    rim = _disc(radius)
    base = single_face(single_face(common(rim, half_plane), label).fuse(*tabs).clean(), label)
    base = single_face(base.cut(*_stringer_notches(radius, sign, frame.settings.stringer_side_m)), label)
    cuts = _former_cuts(frame, x_m, sign)
    _check_slivers(radius, rim, cuts, label)
    pieces = _cut_openings(base, rim, cuts, half_plane.fuse(*tabs), tabs, label, "former")
    if not all(_held_by_a_tab(p, tabs) for p in pieces):
        names = ", ".join(sorted({name for name, _ in cuts}))
        raise KitUnavailable(f"{label}: the opening for the {names} cuts off a piece of the former with no tab")
    pieces.sort(key=lambda p: p.center().X)
    names = [label] if len(pieces) == 1 else [f"{label}{chr(ord('a') + i)}" for i in range(len(pieces))]
    centres = _lightening_centres(radius, sign, half_t)
    return [KitPart(name, "former", drill(piece, lightening_holes(piece, (), centres), name), station_location(x_m))
            for name, piece in zip(names, pieces)]


def _half_formers(frame: _Frame) -> list[KitPart]:
    return [part for i, x in enumerate(frame.former_stations, start=1) for sign in (UP, DOWN)
            for part in _half_former(frame, i, x, sign)]


# ------------------------------------------------------------------------------- solid bulkheads ---
def _firewall(frame: _Frame) -> tuple[KitPart, KitPart]:
    x, t = frame.stations[0], frame.bulkhead_t_m
    disc = _disc(frame.inner_radius(x))  # the nose widens aft, so the front face bounds the pair
    for k in range(2):
        _assert_clear(frame, "FW-1", x + k * t, disc)
    return (KitPart("FW-1a", "bulkhead", disc, station_location(x)),
            KitPart("FW-1b", "bulkhead", disc, station_location(x + t)))


def _boom_mount_parts(frame: _Frame) -> list[KitPart]:
    boom = frame.layout.tail_boom
    if boom is None:
        return []
    hole = Hole(boom.y_m, boom.z_m, boom_tube(boom.diameter_m, frame.settings).od_m + frame.settings.fit_clearance_m)
    parts = []
    for k, x in enumerate(frame.boom_mounts, start=1):
        label = f"BM-{k}"
        disc = _disc(frame.inner_radius(x + frame.bulkhead_t_m))  # the tail cone narrows aft: the aft face bounds it
        _assert_clear(frame, label, x, disc)
        parts.append(KitPart(label, "bulkhead", drill(disc, (hole,), label), station_location(x)))
    return parts


# --------------------------------------------------------------------------------------- crutch ---
def _crutch_outline(frame: _Frame) -> bd.Face:
    x0, x1 = frame.crutch_x
    xs = sorted({x0, x1, *profile_corners(frame.layout, x0, x1)})
    upper = [(x, frame.inner_radius(x), 0.0) for x in xs]
    lower = [(x, -y, 0.0) for x, y, _ in reversed(upper)]
    return bd.Face(bd.Wire.make_polygon(upper + lower, close=True))


def _crutch_openings(frame: _Frame) -> list[Cut]:
    """Footprint + COMPONENT_GAP_M (half each side) under each internal component and each component crossing
    the crutch plane, in aircraft x, y."""
    x0, x1 = frame.crutch_x
    gap, half_t = COMPONENT_GAP_M / 2.0, frame.crutch_t_m / 2.0
    hits = [c for c in frame.components if _overlaps(c, 0, x0, x1, gap)
            and (c.is_internal or _overlaps(c, 2, -half_t, half_t, gap))]
    return [(c.name, rect_face(c.x_min - gap, c.center_m[1] - (c.size_m[1] + COMPONENT_GAP_M) / 2.0,
                               c.x_max + gap, c.center_m[1] + (c.size_m[1] + COMPONENT_GAP_M) / 2.0)) for c in hits]


def _crutch_slots(frame: _Frame) -> list[Cut]:
    """Slots for the former tabs, each as wide as the former sheet (the wing-box plates sit in the formers)."""
    slots = []
    for i, x in enumerate(frame.former_stations, start=1):
        y0, y1 = _tab_span(frame.tab_radius(x))
        slots += [(f"F-{i:02d}", rect_face(x, a, x + frame.former_t_m, b)) for a, b in ((y0, y1), (-y1, -y0))]
    return slots


def _crutch_body(frame: _Frame, outline: bd.Face, openings: Sequence[Cut]) -> bd.Face:
    """The crutch after its component openings. A component spanning the whole width near the nose (a chin
    payload) leaves only scraps ahead of it; those are dropped, so the crutch then starts behind it."""
    pieces = sorted(_cut_openings(outline, outline, openings, None, (), "C-1", "crutch"), key=lambda f: f.area)
    body = pieces[-1]
    start = body.bounding_box().min.X
    if any(p.bounding_box().max.X > start + STATION_EPS for p in pieces[:-1]):
        names = ", ".join(name for name, _ in openings)
        raise KitUnavailable(f"C-1: the openings for the {names} cut the crutch apart")
    return body


def material_spans(face: bd.Face, x_m: float) -> list[tuple[float, float]]:
    """Every stretch of material on the line local x = x_m, in y order."""
    hit = face.intersect(bd.Edge.make_line((x_m, -SPAN_REACH_M, 0.0), (x_m, SPAN_REACH_M, 0.0)))
    pieces = hit if isinstance(hit, list) else ([] if hit is None else [hit])
    ends = [(e.start_point().Y, e.end_point().Y) for piece in pieces for e in piece.edges()]
    return sorted((min(a, b), max(a, b)) for a, b in ends)


def _check_rails(frame: _Frame, body: bd.Face, openings: Sequence[Cut]) -> None:
    """R28: every stretch of crutch beside a component opening (probed at the opening's ends and at the profile
    corners between them, where the piecewise-linear width is least) is at least crutch_rail_min_m wide."""
    least = frame.settings.crutch_rail_min_m
    for name, cut in openings:
        box = cut.bounding_box()
        lo, hi = box.min.X + RAIL_PROBE_INSET_M, box.max.X - RAIL_PROBE_INSET_M
        for x in (lo, hi, *profile_corners(frame.layout, lo, hi)):
            narrow = [b - a for a, b in material_spans(body, x) if b - a < least - FIT_EPS_M]
            if narrow:
                raise KitUnavailable(f"C-1: a side rail beside the {name} is {min(narrow) * 1e3:.1f} mm wide, under "
                                     f"the {least * 1e3:.0f} mm crutch rail minimum")


def _crutch(frame: _Frame) -> KitPart:
    openings, slots = _crutch_openings(frame), _crutch_slots(frame)
    body = _crutch_body(frame, _crutch_outline(frame), openings)
    _check_rails(frame, body, openings)
    result = body.cut(*(face for _, face in slots))
    for name, face in slots:
        _assert_web(result, face, [face], None, f"C-1: the slot for {name} leaves less than {WEB_TEXT} of crutch")
    x0 = frame.crutch_x[0]
    face = single_face(result, "C-1").moved(bd.Location((-x0, 0.0, 0.0)))
    boxes = [cut.bounding_box() for _, cut in openings]
    tagged = tuple((b.min.X - x0, b.min.Y, b.max.X - x0, b.max.Y) for b in boxes)  # the split may cut through these
    return KitPart("C-1", "crutch", face, bd.Location((x0, 0.0, -frame.crutch_t_m / 2.0)), openings=tagged)


# ------------------------------------------------------------------------------------------ API ---
def fuselage_parts(
    layout: AircraftLayout, spar: SparPlan, components: Sequence[PlacedComponent], settings: FabricationSettings,
    *, loads: PlateLoads,
) -> tuple[KitPart, ...]:
    """Crutch `C-1`, half-formers `F-nnU`/`F-nnL`, firewall `FW-1a`/`FW-1b`, wing-box plates `WM-L`/`WM-R` (plus
    socket bosses `WM-LD`/`WM-RD` when the plates alone cannot bear `loads`, R27/R32, and pin bosses `WM-LPD`/`WM-RPD`
    when the pin needs them, R31) and boom mounts `BM-1`/`BM-2`.

    Every component crossing a former, the crutch or a plate gets an opening there; with no components the
    formers carry only lightening holes (plus stringer notches and the wing-box slots).
    """
    frame = _Frame(
        layout=layout, settings=settings, components=tuple(components),
        stations=former_stations(layout, spar, settings, components),
        boom_mounts=_boom_mount_stations(layout, settings), mount=_mount(layout, spar, settings),
        crutch_t_m=settings.sheet_for("crutch").thickness_m, former_t_m=settings.sheet_for("former").thickness_m,
        bulkhead_t_m=settings.sheet_for("bulkhead").thickness_m,
    )
    formers = _half_formers(frame)  # first, so a too-big component is reported against the former it breaks
    plates = {side: _mount_plate(frame, side) for side in (-1, 1)}
    bosses = _wing_box_bosses(frame, spar, loads, plates)
    _check_bolt_seats(frame, spar, bosses)
    return (_crutch(frame), *formers, *_firewall(frame), plates[-1], plates[1], *bosses, *_boom_mount_parts(frame))
