"""Assemble the laser-cut airframe kit: every ply part, stock tube, linear strip and covering (frame: see `kit.py`).

Tubes (`TubePart.start_m` + `length_m` along the unit `axis`):
- `SP-R1…`, `SP-L1…` spar members, one per `SparPlan` segment. The root member `SP-x1` runs from the centreline
  (inside its socket) to the first step; each inner member reaches min_insertion_diameters x its own OD inboard
  of its outer member's actual end (R18 bonded overlap; that end may be trimmed inboard of the step, see below).
- `SK-R`, `SK-L` socket tubes on each wing's spar axis from the centreline to the skin joint (R14). They, and the
  spar roots inside them, meet the other side's pair in a mitre at the centreline (`MITRE_LABELS`).
- `PN-R`, `PN-L` 4 mm anti-rotation pins on the 70 % chord line, from the wing-box plate's inner face to
  PIN_REACH_M outboard of the root doubler (R18).
- `TB-1`, `LB-R`, `LB-L` booms as `boom_parts` places them.
- `TS-HR`, `TS-HL` horizontal-tail spars from the tail root (boom or fuselage skin, R20) and `TS-V` fin spar from
  the fin start (top of `TM-1`, or the skin), each out
  to the tip or to where the tail's depth stops admitting it, short of the first rib too thin for its hole (the
  tail ribs' own rule). A tube's end faces are square to its axis, which
  leans (taper), so a tail spar starts just far enough off its root surface for its rim to clear it.
Linear stock (balsa, `stringer_density_kg_m3`): `ST-1…` stringers in the former notches from the first former
after the firewall to the aft face of the last former; `LE-…` / `TE-…` strips per wing panel and tail surface.
Covering: film over both faces of the wing and tails (2 x planform), fuselage cover over the pod's lateral area.
"""

import math
from typing import Callable, Final, Optional, Sequence

import build123d as bd

from src.schemas.fabrication import FabricationSettings, TubeStock
from src.schemas.state import SizingOutput
from src.tools.aero_cad.airfoil_geometry import AirfoilShape, load_airfoil
from src.tools.aero_cad.components import PlacedComponent
from src.tools.aero_cad.layout import (
    FUSELAGE_PROFILE,
    TAIL_AIRFOIL_NAME,
    WING_Z_FRACTION,
    AircraftLayout,
    SurfacePlanform,
    fuselage_radius_at,
)
from src.tools.aero_cad.structure import SPAR_CHORD_STATION
from src.tools.fabrication.fittings import boom_parts, component_plates, cruise_motor_plate, join_notes, split_to_bed
from src.tools.fabrication.fuselage_parts import (
    FULL_TURN_DEG,
    STRINGER_COUNT,
    STRINGER_START_DEG,
    former_stations,
    fuselage_parts,
    socket_axis_m,
)
from src.tools.fabrication.kit import (
    CoverSurface,
    Kit,
    KitPart,
    KitUnavailable,
    LinearKind,
    LinearStock,
    TubePart,
    TubeRole,
    Vec3,
)
from src.tools.fabrication.rib_outline import camber_point, keeps_web, notched_profile
from src.tools.fabrication.tubes import FIT_EPS_M, SparPlan, depth_limit_m, joint_y_m, plan_spar
from src.tools.fabrication.wing_parts import (
    PIN_CHORD_FRACTION,
    PIN_ROD_DIAMETER_M,
    fin_start_m,
    spar_depth_at,
    tail_ribs,
    tail_root_radius_m,
    tail_spar_hole,
    tail_spar_tube,
    wing_rib_stations,
    wing_ribs,
)

MAX_DOD_GROUP: Final[int] = 2
PIN_REACH_M: Final[float] = 0.020  # R18: pin end this far outboard of the root doubler
MITRE_LABELS: Final[tuple[str, ...]] = ("SK-R", "SK-L", "SP-R1", "SP-L1")
SIDES: Final[tuple[tuple[float, str], ...]] = ((1.0, "R"), (-1.0, "L"))
LE_FRACTION: Final[float] = 0.0
TE_FRACTION: Final[float] = 1.0

Line = Callable[[float], Vec3]  # a straight 3D line by its span (or fin height) parameter


# ------------------------------------------------------------------------------------------- lines ---
def _wing_line(layout: AircraftLayout, side: float, fraction: float) -> Line:
    """Wing line at `fraction` chord on the camber line, by span station y (straight: chord and z are linear)."""
    def point(y: float) -> Vec3:
        x, z = socket_axis_m(layout, y, fraction)
        return x, side * y, z
    return point


def _surface_line(surface: SurfacePlanform, airfoil: AirfoilShape, side: float, fraction: float) -> Line:
    """Line at `fraction` chord on the camber line of a surface; the fin's parameter is height above its root."""
    def point(s: float) -> Vec3:
        dx, dz = camber_point(airfoil, surface.chord_at(s), fraction)
        x = surface.x_le_root_m + dx
        if not surface.is_symmetric:
            return x, dz, surface.z_root_m + s
        return x, side * s, surface.z_at(s) + dz
    return point


def _tube_on(label: str, role: TubeRole, tube: TubeStock, line: Line, s0: float, s1: float,
             extend_m: float = 0.0) -> TubePart:
    """Tube along `line` from s0 to s1, reaching `extend_m` further back past s0."""
    a, b = line(s0), line(s1)
    length = math.dist(a, b)
    axis = ((b[0] - a[0]) / length, (b[1] - a[1]) / length, (b[2] - a[2]) / length)
    start = (a[0] - extend_m * axis[0], a[1] - extend_m * axis[1], a[2] - extend_m * axis[2])
    return TubePart(label, role, tube, length + extend_m, start, axis)


def _rim_clearance(line: Line, s0: float, s1: float, radius_m: float) -> float:
    """Parameter offset that keeps a square-cut tube end off the plane s = s0: the end disk leans with the axis."""
    along = abs(s1 - s0) / math.dist(line(s0), line(s1))  # axis component along the parameter direction
    return radius_m * math.sqrt(max(1.0 - along**2, 0.0))


# ------------------------------------------------------------------------------------------- tubes ---
def _member_end(line: Line, spar: SparPlan, k: int, settings: FabricationSettings) -> float:
    """Span station where member k ends: its step (or the tip), pulled inboard so its square-cut rim stays clear
    of the first rib outboard of the step, whose hole is cut for the next, smaller member."""
    seg = spar.segments[k]
    half_t = settings.sheet_for("rib").thickness_m / 2.0
    face = next((y - half_t for y in wing_rib_stations(spar.joint_y_m, spar.segments[-1].y_end_m)
                 if y > seg.y_end_m + FIT_EPS_M), None)
    if face is None:  # the tip member, or no rib outboard of this step
        return seg.y_end_m
    rim = _rim_clearance(line, seg.y_start_m, seg.y_end_m, seg.tube.od_m / 2.0)
    return min(seg.y_end_m, face - rim)


def _spar_tubes(layout: AircraftLayout, spar: SparPlan, settings: FabricationSettings) -> list[TubePart]:
    tubes: list[TubePart] = []
    for side, name in SIDES:
        line = _wing_line(layout, side, SPAR_CHORD_STATION)
        ends = [_member_end(line, spar, k, settings) for k in range(len(spar.segments))]
        for k, seg in enumerate(spar.segments):
            # R18: an inner member reaches its overlap past the outer member's actual (possibly trimmed) end
            inboard = 0.0 if k == 0 else ends[k - 1]
            overlap = 0.0 if k == 0 else settings.min_insertion_diameters * seg.tube.od_m
            tubes.append(_tube_on(f"SP-{name}{k + 1}", "spar", seg.tube, line, inboard, ends[k], overlap))
    return tubes


def socket_end_m(layout: AircraftLayout, spar: SparPlan) -> float:
    """Span station where each socket tube ends: its square-cut rim just reaches the skin joint plane, so it stays
    out of the root doubler (the spar axis leans with taper and dihedral)."""
    line = _wing_line(layout, 1.0, SPAR_CHORD_STATION)
    return spar.joint_y_m - _rim_clearance(line, 0.0, spar.joint_y_m, spar.socket.od_m / 2.0)


def _socket_and_pins(layout: AircraftLayout, spar: SparPlan, fuselage: Sequence[KitPart],
                     settings: FabricationSettings) -> list[TubePart]:
    plate = next(p for p in fuselage if p.label == "WM-R")
    inner = plate.location.position.Y - settings.sheet_for("plate").thickness_m  # WM-R extrudes toward -y
    outer = spar.joint_y_m + settings.sheet_for("doubler").thickness_m + PIN_REACH_M
    pin = TubeStock(od_m=PIN_ROD_DIAMETER_M, id_m=0.0)
    end = socket_end_m(layout, spar)
    sockets = [_tube_on(f"SK-{name}", "socket", spar.socket, _wing_line(layout, side, SPAR_CHORD_STATION),
                        0.0, end) for side, name in SIDES]
    pins = [_tube_on(f"PN-{name}", "pin", pin, _wing_line(layout, side, PIN_CHORD_FRACTION), inner, outer)
            for side, name in SIDES]
    return sockets + pins


def _first_rib_without_hole(surface: SurfacePlanform, start_m: float, airfoil: AirfoilShape, tube: TubeStock,
                            settings: FabricationSettings) -> Optional[float]:
    """Station of the first rib too thin to carry the spar hole with the minimum web (the rule `tail_ribs`
    drills by), or None when every rib carries it."""
    for index, s in enumerate(wing_rib_stations(start_m, surface.semi_span_m)):
        hole = tail_spar_hole(surface, airfoil, tube, index, s, settings)
        if not keeps_web(notched_profile(airfoil, surface.chord_at(s), settings), hole, []):
            return s
    return None


def _tail_spar(label: str, surface: SurfacePlanform, line: Line, root: float, below_m: float, tube: TubeStock,
               airfoil: AirfoilShape, settings: FabricationSettings) -> TubePart:
    """From just off the root surface (rim clear) out to the tip or where the depth stops admitting the tube, and
    short of the first rib without a hole (rim clear of its inner face, `below_m` inboard of its station)."""
    depth, tip, radius = spar_depth_at(surface, airfoil, settings, root_doubler_m=None), surface.semi_span_m, tube.od_m / 2.0
    end = tip if depth(tip) >= tube.od_m - FIT_EPS_M else depth_limit_m(depth, tube.od_m, root, tip)
    blocked = _first_rib_without_hole(surface, root, airfoil, tube, settings)
    if blocked is not None:
        end = min(end, blocked - below_m - _rim_clearance(line, root, end, radius))
    return _tube_on(label, "tail_spar", tube, line, root + _rim_clearance(line, root, end, radius), end)


def _tail_spars(layout: AircraftLayout, airfoil: AirfoilShape, settings: FabricationSettings) -> list[TubePart]:
    tube, t_rib = tail_spar_tube(layout, airfoil, settings), settings.sheet_for("rib").thickness_m
    ht, vt, root = layout.htail, layout.vtail, tail_root_radius_m(layout)
    spars = [_tail_spar(f"TS-H{name}", ht, _surface_line(ht, airfoil, side, SPAR_CHORD_STATION), root, t_rib / 2.0,
                        tube, airfoil, settings) for side, name in SIDES]  # tail ribs are centred on their station
    fin_line = _surface_line(vt, airfoil, 1.0, SPAR_CHORD_STATION)
    fin = _tail_spar("TS-V", vt, fin_line, fin_start_m(layout, settings), 0.0, tube, airfoil, settings)  # fin ribs extrude up
    return spars + [fin]


# ------------------------------------------------------------------------------------------ linear ---
def _strip(label: str, kind: LinearKind, section_m2: float, line: Line, s0: float, s1: float,
           settings: FabricationSettings) -> LinearStock:
    a, b = line(s0), line(s1)
    return LinearStock(label, kind, section_m2, settings.stringer_density_kg_m3, math.dist(a, b), a, b)


def _stringer_line(layout: AircraftLayout, settings: FabricationSettings, angle_deg: float) -> Line:
    """Centre of the stringer notch at `angle_deg` (r_in - side/2 from the axis), by station x."""
    angle = math.radians(angle_deg)

    def point(x: float) -> Vec3:
        r = fuselage_radius_at(layout, x) - settings.fuselage_cover_thickness_m - settings.stringer_side_m / 2.0
        return x, r * math.cos(angle), r * math.sin(angle)
    return point


def _stringers(layout: AircraftLayout, stations: Sequence[float], settings: FabricationSettings) -> list[LinearStock]:
    """In the former notches, from the first former after the firewall to the last former's aft face."""
    x0, x1 = stations[1], stations[-1] + settings.sheet_for("former").thickness_m
    pitch = FULL_TURN_DEG / STRINGER_COUNT
    return [_strip(f"ST-{k + 1}", "stringer", settings.stringer_side_m**2,
                   _stringer_line(layout, settings, STRINGER_START_DEG + k * pitch), x0, x1, settings)
            for k in range(STRINGER_COUNT)]


def _edge_strips(layout: AircraftLayout, spar: SparPlan, airfoil: AirfoilShape, tail: AirfoilShape,
                 settings: FabricationSettings) -> list[LinearStock]:
    """LE and TE strips per wing panel and tail surface, root rib to tip."""
    root = tail_root_radius_m(layout)
    surfaces = [(f"W{n}", layout.wing, airfoil, s, spar.joint_y_m) for s, n in SIDES]
    surfaces += [(f"H{n}", layout.htail, tail, s, root) for s, n in SIDES]
    surfaces.append(("V", layout.vtail, tail, 1.0, fin_start_m(layout, settings)))
    edges = (("le", LE_FRACTION, settings.le_side_m**2), ("te", TE_FRACTION, settings.te_height_m * settings.te_chord_m))
    return [_strip(f"{kind.upper()}-{name}", kind, section, _surface_line(surface, foil, side, fraction), start,
                   surface.semi_span_m, settings)
            for kind, fraction, section in edges for name, surface, foil, side, start in surfaces]


# ---------------------------------------------------------------------------------------- covering ---
def _planform_face(surface: SurfacePlanform, side: float) -> bd.Face:
    """One panel's planform (centreline or fin root to tip), in the panel's dihedral plane."""
    def at(s: float, fraction: float) -> Vec3:
        x = surface.x_le_root_m + fraction * surface.chord_at(s)
        return (x, side * s, surface.z_at(s)) if surface.is_symmetric else (x, 0.0, surface.z_root_m + s)
    span = surface.semi_span_m
    corners = [at(0.0, LE_FRACTION), at(0.0, TE_FRACTION), at(span, TE_FRACTION), at(span, LE_FRACTION)]
    return bd.Face(bd.Wire.make_polygon(corners, close=True))


def _film(label: str, surface: SurfacePlanform) -> CoverSurface:
    sides = (1.0, -1.0) if surface.is_symmetric else (1.0,)
    faces = bd.Compound([_planform_face(surface, s) for s in sides])
    return CoverSurface(label, "film", faces, 2.0 * surface.area_m2)  # film on both faces


def _fuselage_cover(layout: AircraftLayout) -> CoverSurface:
    """The pod's skin: the piecewise-linear radius profile revolved about the x axis (lateral area only)."""
    length = layout.fuselage_length_m
    xs = [0.0, *(f * length for f, _ in FUSELAGE_PROFILE), length]
    profile = bd.Wire.make_polygon([(x, 0.0, fuselage_radius_at(layout, x)) for x in xs], close=False)
    shell = bd.Shell.revolve(profile, FULL_TURN_DEG, bd.Axis.X)
    return CoverSurface("fuselage_cover", "fuselage", shell, shell.area)


def _covering(layout: AircraftLayout) -> tuple[CoverSurface, ...]:
    return (_film("film_wing", layout.wing), _film("film_htail", layout.htail), _film("film_vtail", layout.vtail),
            _fuselage_cover(layout))


# -------------------------------------------------------------------------------------------- API ---
def _plan(sizing: SizingOutput, layout: AircraftLayout, airfoil: AirfoilShape,
          settings: FabricationSettings) -> SparPlan:
    """Spar plan with the joint where the spar meets the skin at wing height (as the spar tests build it)."""
    radius = fuselage_radius_at(layout, socket_axis_m(layout, 0.0)[0])
    joint = joint_y_m(radius, WING_Z_FRACTION * layout.fuselage_diameter_m)
    boom_y = max((abs(b.y_m) for b in layout.vtol_booms), default=None)
    depth = spar_depth_at(layout.wing, airfoil, settings, root_doubler_m=joint)
    return plan_spar(sizing.mtow_kg, layout.wing.semi_span_m, joint, depth, boom_y, settings)


def _split_all(parts: Sequence[KitPart], settings: FabricationSettings) -> tuple[KitPart, ...]:
    return tuple(piece for part in parts for piece in split_to_bed(part, settings))


def build_kit(sizing: SizingOutput, layout: AircraftLayout, airfoil: AirfoilShape,
              components: Sequence[PlacedComponent], settings: FabricationSettings) -> Kit:
    """The whole kit for a Group 1-2 design; raises `KitUnavailable` when any part cannot be made from stock."""
    if sizing.dod_group > MAX_DOD_GROUP:
        raise KitUnavailable("laser-cut construction covers Group 1–2 designs only")
    spar = _plan(sizing, layout, airfoil, settings)
    tail = load_airfoil(TAIL_AIRFOIL_NAME)
    ribs = (*wing_ribs(layout, airfoil, spar, settings), *tail_ribs(layout, tail, settings))
    fuselage = fuselage_parts(layout, spar, components, settings)
    trays = component_plates(components, layout, settings, spar=spar)
    booms, boom_tubes = boom_parts(layout, spar, settings, components=components,
                                   lift_thrust_n=sizing.vtol_thrust_per_motor_n)
    motor = cruise_motor_plate(components, layout, settings)
    parts = _split_all((*ribs, *fuselage, *trays, *booms, *motor), settings)
    tubes = (*_spar_tubes(layout, spar, settings), *_socket_and_pins(layout, spar, fuselage, settings),
             *boom_tubes, *_tail_spars(layout, tail, settings))
    stations = former_stations(layout, spar, settings, components)
    linear = (*_stringers(layout, stations, settings), *_edge_strips(layout, spar, airfoil, tail, settings))
    covering = _covering(layout)
    return Kit(parts=parts, tubes=tubes, linear=linear, covering=covering, spar=spar, joins=join_notes(parts))
