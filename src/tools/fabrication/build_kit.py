"""Assemble the laser-cut airframe kit: every ply part, stock tube, linear strip and covering (frame: see `kit.py`).

Tubes (`TubePart.start_m` + `length_m` along the unit `axis`):
- `SP-R1…`, `SP-L1…` spar members, one per `SparPlan` segment. The root member `SP-x1` runs from the centreline
  (inside its socket) to the first step; each inner member reaches min_insertion_diameters x its own OD inboard
  of its outer member's actual end (R18 bonded overlap; that end may be trimmed inboard of the step, see below).
- `SK-R`, `SK-L` socket tubes on each wing's spar axis from the centreline to the skin joint (R14). They, and the
  spar roots inside them, meet the other side's pair in a mitre at the centreline (`MITRE_LABELS`).
- R36 wing retention (`Kit.wing_bolt`): one M5 bolt per side from the wing-box plate's inner face into the nut
  captive in `NP-R` / `NP-L`; the bolt is hardware (glue and hardware allowance), the nut plates are kit parts.
- `PN-R`, `PN-L` anti-rotation pins (`SparPlan.pin`: 4 mm, or larger per R31) on the 70 % chord line, from the
  wing-box plate's inner face to PIN_REACH_M outboard of the root doubler (R18).
- `TB-1`, `LB-R`, `LB-L` booms as `boom_parts` places them.
- One spar per tail panel, in the panel's own frame (`wing_parts.panel_point`): `TS-HR`, `TS-HL` horizontal-tail
  spars from the tail root (boom or fuselage skin, R20), `TS-V` fin spar from the fin start (top of `TM-1`, or the
  skin), `TS-VR`, `TS-VL` V-tail spars butting the root block `VR-1`'s faces, along each panel's dihedral; each the
  surface's strength-sized stock (R35, `tail_spar_tubes`), out to the tip or to where the tail's depth stops admitting
  it, short of the first rib too thin for its hole (the tail ribs' own rule). A tube's end faces are square to its
  axis, which leans (taper), so a tail spar starts just far enough off its root surface for its rim to clear it.
Linear stock (balsa, `stringer_density_kg_m3`): `ST-1…` stringers in the former notches from the first former
after the firewall to the aft face of the last former; `LE-…` / `TE-…` strips per wing panel and tail panel.
Covering: film over both faces of the wing and of each tail surface outside the fuselage, boom or root block
(`film_<surface name>`, 2 x exposed planform), fuselage cover over the pod's lateral area.
Boom-end pusher: the motor mount `MP-B` + `MC-B` replaces the nose `MP-C`, the tail boom check carries the motor's
weight at n_limit g at the boom end (`BoomTipMass`), and the build sheet notes the motor wiring along the boom.
Structural checks (R24-R35): the spar is sized against bending and torsion (`kit_spar_torsion`) and its pin for the
root torque, the tail boom is checked and the tail spars sized for each panel's load (R35), the plates get socket
bosses when bearing or the downward web load needs them (and pin bosses when the pin needs them), and the centreline
mitre gets its overwrap (`Kit.overwrap`), which must fit between the plates' (or bosses') inner faces. Part and tube
labels are unique.
"""

import dataclasses
import math
from collections import Counter
from typing import Callable, Final, Optional, Sequence

import build123d as bd

from src.schemas.fabrication import FabricationSettings, TubeStock
from src.schemas.requirements import LAYOUT_LABELS
from src.schemas.state import SizingOutput
from src.tools.aero_cad.airfoil_geometry import AirfoilShape, load_airfoil
from src.tools.aero_cad.components import PlacedComponent
from src.tools.aero_cad.layout import (
    FUSELAGE_PROFILE,
    TAIL_AIRFOIL_NAME,
    WING_Z_FRACTION,
    AircraftLayout,
    PlacedTailSurface,
    SurfacePlanform,
    fuselage_radius_at,
)
from src.tools.aero_cad.structure import SPAR_CHORD_STATION
from src.tools.fabrication.boom_parts import lift_boom_keep_out, tail_root_blocks
from src.tools.fabrication.fittings import (
    CRUISE_MOTOR_NAME,
    boom_parts,
    component_plates,
    cruise_motor_plate,
    join_notes,
    split_to_bed,
)
from src.tools.fabrication.fuselage_parts import (
    FULL_TURN_DEG,
    STRINGER_COUNT,
    STRINGER_START_DEG,
    former_stations,
    fuselage_parts,
    socket_axis_m,
)
from src.tools.fabrication.kit import (
    BOOM_MOTOR_WIRING_NOTE,
    TAIL_CONE_SLOT_NOTE,
    CoverSurface,
    Kit,
    KitPart,
    KitUnavailable,
    LinearKind,
    LinearStock,
    Overwrap,
    TubePart,
    TubeRole,
    Vec3,
    WingBolt,
    part_solid,
)
from src.tools.fabrication.rib_outline import camber_point, keeps_web, notched_profile
from src.tools.fabrication.tubes import (
    FIT_EPS_M,
    BoomTipMass,
    SparPlan,
    SparTorsion,
    aero_torque_nm,
    boom_tube,
    depth_limit_m,
    hover_pitch_torque_nm,
    joint_y_m,
    flight_dynamic_pressure_pa,
    plan_spar,
    size_overwrap,
    plate_loads,
    tail_boom_check,
)
from src.tools.fabrication.wing_box_bosses import BOSS_LABELS
from src.tools.fabrication.wing_parts import (
    PIN_CHORD_FRACTION,
    panel_planform,
    panel_point,
    panel_start_m,
    spar_depth_at,
    tail_panels,
    tail_ribs,
    TailSpars,
    tail_spar_hole,
    tail_spar_tubes,
    clear_rib_stations,
    panel_rib_stations,
    wing_rib_stations,
    spar_member_at,
    wing_ribs,
)
from src.tools.fabrication.wing_retention import (
    BOLT_CHORD_FRACTION,
    BOLT_THREAD_MARGIN_M,
    BOLT_WASHER_THICKNESS_M,
    M5_NUT_HEIGHT_M,
    bolt_length_m,
)
from src.tools.fabrication.wing_stiffness import (
    WingStiffness,
    dbox_material_m2,
    dbox_web_x_m,
    stiffen_spar,
    wing_stiffness,
)
from src.tools.sizing.layouts import layout_spec

MAX_DOD_GROUP: Final[int] = 2
# (tail, propeller, booms) LayoutSpec combinations the kit has parts and checks for; any other is refused whole
BUILT_LAYOUT_PARTS: Final[frozenset[tuple[str, str, str]]] = frozenset({("conventional", "tractor_nose", "centre"),
                                                                       ("v", "pusher_boom_end", "centre")})
BOOM_END_PROPELLER: Final[str] = "pusher_boom_end"  # LayoutSpec.propeller of a motor at the tail boom's end
CLASH_VOLUME_M3: Final[float] = 1e-12  # solids sharing more than this volume intersect
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
    """Line at `fraction` chord on the camber line of a mirrored surface (the wing) by span station y."""
    def point(s: float) -> Vec3:
        dx, dz = camber_point(airfoil, surface.chord_at(s), fraction)
        return surface.x_le_root_m + dx, side * s, surface.z_at(s) + dz
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
    face = next((y - half_t for y in wing_rib_stations(spar, spar.segments[-1].y_end_m)
                 if y > seg.y_end_m + FIT_EPS_M), None)
    if face is None:  # the tip member, or no rib outboard of this step
        return seg.y_end_m
    rim = _rim_clearance(line, seg.y_start_m, seg.y_end_m, seg.tube.od_m / 2.0)
    return min(seg.y_end_m, face - rim)


def spar_tubes(layout: AircraftLayout, spar: SparPlan, settings: FabricationSettings) -> list[TubePart]:
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
    pin = TubeStock(od_m=spar.pin.rod_m, id_m=0.0)  # R31: the rod the root torque needs
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
    for index, s in enumerate(panel_rib_stations(start_m, surface.semi_span_m)):
        hole = tail_spar_hole(surface, airfoil, tube, index, s, settings)
        if not keeps_web(notched_profile(airfoil, surface.chord_at(s), settings), hole, []):
            return s
    return None


def _tail_spar(label: str, surface: SurfacePlanform, line: Line, root: float, below_m: float, tube: TubeStock,
               airfoil: AirfoilShape, settings: FabricationSettings) -> TubePart:
    """From just off the root surface (rim clear) out to the tip or where the depth stops admitting the tube, and
    short of the first rib without a hole (rim clear of its inner face, `below_m` inboard of its station)."""
    depth = spar_depth_at(surface, airfoil, settings, root_doubler_m=None)
    tip, radius = surface.semi_span_m, tube.od_m / 2.0
    end = tip if depth(tip) >= tube.od_m - FIT_EPS_M else depth_limit_m(depth, tube.od_m, root, tip)
    blocked = _first_rib_without_hole(surface, root, airfoil, tube, settings)
    if blocked is not None:
        end = min(end, blocked - below_m - _rim_clearance(line, root, end, radius))
    return _tube_on(label, "tail_spar", tube, line, root + _rim_clearance(line, root, end, radius), end)


def _tail_line(surface: PlacedTailSurface, airfoil: AirfoilShape, side: float, fraction: float) -> Line:
    """Line at `fraction` chord on the camber line of one tail panel, by its panel station (`panel_point`)."""
    plan = panel_planform(surface)

    def point(s: float) -> Vec3:
        dx, dw = camber_point(airfoil, plan.chord_at(s), fraction)
        return panel_point(surface, side, s, plan.x_le_root_m + dx, dw)
    return point


def _tail_spars(layout: AircraftLayout, airfoil: AirfoilShape, tails: TailSpars,
                settings: FabricationSettings) -> list[TubePart]:
    """`TS-<panel>` per tail panel from its root station (`_tail_spar`); the first rib without a hole has its inner
    face half a rib sheet inboard of its station on a mirrored panel (ribs centred on it), at it on a fin (ribs
    extrude up)."""
    t_rib = settings.sheet_for("rib").thickness_m
    spars: list[TubePart] = []
    for surface in layout.tail_surfaces:
        plan, start = panel_planform(surface), panel_start_m(layout, surface, settings)
        below = t_rib / 2.0 if plan.is_symmetric else 0.0
        spars += [_tail_spar(f"TS-{name}", plan, _tail_line(surface, airfoil, side, SPAR_CHORD_STATION), start,
                             below, tails[surface.name], airfoil, settings) for side, name in tail_panels(surface)]
    return spars


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
    """LE and TE strips per wing panel and tail panel, root rib to tip."""
    edges = (("le", LE_FRACTION, settings.le_side_m**2),
             ("te", TE_FRACTION, settings.te_height_m * settings.te_chord_m))
    strips: list[LinearStock] = []
    for kind, fraction, section in edges:
        strips += [_strip(f"{kind.upper()}-W{name}", kind, section, _surface_line(layout.wing, airfoil, side, fraction),
                          spar.joint_y_m, layout.wing.semi_span_m, settings) for side, name in SIDES]
        for surface in layout.tail_surfaces:
            start = panel_start_m(layout, surface, settings)
            strips += [_strip(f"{kind.upper()}-{name}", kind, section, _tail_line(surface, tail, side, fraction),
                              start, surface.planform.semi_span_m, settings) for side, name in tail_panels(surface)]
    return strips


# ---------------------------------------------------------------------------------------- covering ---
def _planform_face(surface: SurfacePlanform, side: float, start_m: float, fore: float = LE_FRACTION,
                   aft: float = TE_FRACTION) -> bd.Face:
    """One wing panel's exposed planform between the chord fractions `fore` and `aft`, from span station `start_m`
    to the tip, in its dihedral plane."""
    def at(s: float, fraction: float) -> Vec3:
        return surface.x_le_root_m + fraction * surface.chord_at(s), side * s, surface.z_at(s)
    span = surface.semi_span_m
    corners = [at(start_m, fore), at(start_m, aft), at(span, aft), at(span, fore)]
    return bd.Face(bd.Wire.make_polygon(corners, close=True))


def _panel_face(surface: PlacedTailSurface, side: float, start_m: float) -> bd.Face:
    """One tail panel's exposed planform from `start_m` (its panel station) to the tip, in the panel's plane."""
    plan = surface.planform

    def at(s: float, fraction: float) -> Vec3:
        return panel_point(surface, side, s, plan.x_le_root_m + fraction * plan.chord_at(s), 0.0)
    span = plan.semi_span_m
    corners = [at(start_m, LE_FRACTION), at(start_m, TE_FRACTION), at(span, TE_FRACTION), at(span, LE_FRACTION)]
    return bd.Face(bd.Wire.make_polygon(corners, close=True))


def _film(label: str, faces: bd.Compound) -> CoverSurface:
    """Film over both faces of an exposed planform (M3: none inside the fuselage or over the boom)."""
    return CoverSurface(label, "film", faces, 2.0 * faces.area)  # film on both faces


def _tail_film(layout: AircraftLayout, surface: PlacedTailSurface, settings: FabricationSettings) -> CoverSurface:
    start = panel_start_m(layout, surface, settings)
    faces = bd.Compound([_panel_face(surface, side, start) for side, _ in tail_panels(surface)])
    return _film(f"film_{surface.name}", faces)


def _fuselage_cover(layout: AircraftLayout) -> CoverSurface:
    """The pod's skin: the piecewise-linear radius profile revolved about the x axis (lateral area only)."""
    length = layout.fuselage_length_m
    xs = [0.0, *(f * length for f, _ in FUSELAGE_PROFILE), length]
    profile = bd.Wire.make_polygon([(x, 0.0, fuselage_radius_at(layout, x)) for x in xs], close=False)
    shell = bd.Shell.revolve(profile, FULL_TURN_DEG, bd.Axis.X)
    return CoverSurface("fuselage_cover", "fuselage", shell, shell.area)


def _dbox(layout: AircraftLayout, airfoil: AirfoilShape, spar: SparPlan, settings: FabricationSettings
          ) -> CoverSurface:
    """The leading-edge D-box (R45): the planform strip from the leading edge back to the web, on both panels."""
    wing, root = layout.wing, spar.joint_y_m
    chord = wing.chord_at(root)
    web_fraction = dbox_web_x_m(chord, spar_member_at(spar, root).od_m, settings) / chord
    faces = bd.Compound([_planform_face(wing, side, root, LE_FRACTION, web_fraction) for side in (1.0, -1.0)])
    area_m2 = dbox_material_m2(spar, airfoil, wing.semi_span_m, wing.chord_at, settings)
    return CoverSurface("dbox_wing", "dbox", faces, area_m2)


def _covering(layout: AircraftLayout, airfoil: AirfoilShape, spar: SparPlan, settings: FabricationSettings
              ) -> tuple[CoverSurface, ...]:
    """Film over the exposed wing (from the skin joint) and each tail surface (from its panels' root station: the
    tail root, R20, the fin start, or the V-tail root block), the fuselage cover, and the balsa D-box when the spar
    has one."""
    wing = bd.Compound([_planform_face(layout.wing, side, spar.joint_y_m) for side in (1.0, -1.0)])
    covering = (_film("film_wing", wing), *(_tail_film(layout, t, settings) for t in layout.tail_surfaces),
                _fuselage_cover(layout))
    return (*covering, _dbox(layout, airfoil, spar, settings)) if spar.dbox else covering


# -------------------------------------------------------------------------------------------- API ---
def kit_spar_torsion(sizing: SizingOutput, layout: AircraftLayout, settings: FabricationSettings) -> SparTorsion:
    """R25 torque sources: the flight torque at the flight dynamic pressure (R30) with the default airfoil Cm0,
    and hover pitch control, one lift rotor at full thrust times the longer fore/aft arm from the spar at the boom
    station (none without lift booms)."""
    wing, q = layout.wing, flight_dynamic_pressure_pa(sizing, settings)

    def flight(y: float) -> float:
        return aero_torque_nm(sizing.mtow_kg, wing.semi_span_m, wing.chord_at, q, y, settings)

    boom_y = max((abs(b.y_m) for b in layout.vtol_booms), default=None)
    if boom_y is None:
        return SparTorsion(flight, hover_torque_nm=0.0, hover_thrust_n=0.0)
    spar_x = socket_axis_m(layout, boom_y)[0]
    arm = max(abs(x - spar_x) for x, _, _ in layout.rotor_positions)
    thrust = sizing.vtol_thrust_per_motor_n
    return SparTorsion(flight, hover_torque_nm=hover_pitch_torque_nm(thrust, arm), hover_thrust_n=thrust)


def kit_spar_plan(sizing: SizingOutput, layout: AircraftLayout, airfoil: AirfoilShape,
          settings: FabricationSettings) -> SparPlan:
    """Spar plan with the joint where the spar meets the skin at wing height (as the spar tests build it), sized
    against bending and torsion, with its anti-rotation pin sized for the root torque at the pin's chordwise arm
    from the spar axis at the joint (R31)."""
    radius = fuselage_radius_at(layout, socket_axis_m(layout, 0.0)[0])
    joint = joint_y_m(radius, WING_Z_FRACTION * layout.fuselage_diameter_m)
    boom_y = max((abs(b.y_m) for b in layout.vtol_booms), default=None)
    depth = spar_depth_at(layout.wing, airfoil, settings, root_doubler_m=joint)
    arm = (PIN_CHORD_FRACTION - SPAR_CHORD_STATION) * layout.wing.chord_at(joint)
    return plan_spar(sizing.mtow_kg, layout.wing.semi_span_m, joint, depth, boom_y, settings,
                     torsion=kit_spar_torsion(sizing, layout, settings), pin_arm_m=arm)


def stiffened_spar_plan(sizing: SizingOutput, layout: AircraftLayout, airfoil: AirfoilShape,
                        settings: FabricationSettings) -> SparPlan:
    """`kit_spar_plan` plus the wing stiffness checks (R43-R45): the spar as `build_kit` plans it. Refused when the
    design has no cruise speed. `kit_spar_plan` stays strength-only: the pod search relies on it."""
    if sizing.v_cruise_ms is None:
        raise KitUnavailable("the design has no cruise speed (it predates the wing stiffness checks); "
                             "re-run it to build a kit")
    v_cruise_ms = sizing.v_cruise_ms

    def check(spar: SparPlan, with_dbox: bool) -> WingStiffness:
        return wing_stiffness(spar, airfoil, with_dbox, sizing.mtow_kg, layout.wing.semi_span_m,
                              layout.wing.chord_at, sizing.aspect_ratio, v_cruise_ms, settings)

    spar = stiffen_spar(kit_spar_plan(sizing, layout, airfoil, settings), check, settings)
    keep_out = lift_boom_keep_out(layout, settings, sizing.vtol_thrust_per_motor_n if layout.vtol_booms else None)
    nominal = panel_rib_stations(spar.joint_y_m, layout.wing.semi_span_m)
    return dataclasses.replace(spar, rib_stations_m=clear_rib_stations(nominal, keep_out, layout.wing.semi_span_m))


def _plate_gap_m(fuselage: Sequence[KitPart], settings: FabricationSettings) -> float:
    """Clear width between the wing-box plates' innermost faces (the socket bosses' when fitted)."""
    t = settings.sheet_for("plate").thickness_m  # plates and bosses extrude toward -y from their origin
    return 2.0 * min(p.location.position.Y - t for p in fuselage if p.label in ("WM-R", BOSS_LABELS[1]))


def _overwrap(spar: SparPlan, fuselage: Sequence[KitPart], settings: FabricationSettings) -> Overwrap:
    return size_overwrap(spar.centre_moment_nm, spar.socket, _plate_gap_m(fuselage, settings), settings)


def _wing_bolt(layout: AircraftLayout, spar: SparPlan, fuselage: Sequence[KitPart],
               settings: FabricationSettings) -> WingBolt:
    """R36: the bolt grips from the right plate's inner face (washer under the head) through the root doubler and
    the captive nut (seated in `NP-R`, standing M5_NUT_HEIGHT_M outboard of the doubler), plus a thread margin."""
    plate = next(p for p in fuselage if p.label == "WM-R")
    t_plate, t_root = settings.sheet_for("plate").thickness_m, settings.sheet_for("doubler").thickness_m
    outer = plate.location.position.Y
    grip = (spar.joint_y_m + t_root + M5_NUT_HEIGHT_M - (outer - t_plate) + BOLT_WASHER_THICKNESS_M
            + BOLT_THREAD_MARGIN_M)
    x, z = socket_axis_m(layout, outer - t_plate / 2.0, BOLT_CHORD_FRACTION)
    aft = (BOLT_CHORD_FRACTION - SPAR_CHORD_STATION) * layout.wing.chord_at(spar.joint_y_m)
    return WingBolt(length_m=bolt_length_m(grip), grip_m=grip, plate_x_m=x, plate_z_m=z, aft_of_spar_m=aft)


def assert_unique_labels(parts: Sequence[KitPart], tubes: Sequence[TubePart]) -> None:
    """Labels identify parts on the sheets, in the build sheet and in the DXF kerf cache: never two alike."""
    counts = Counter([*(p.label for p in parts), *(t.label for t in tubes)])
    duplicates = sorted(label for label, n in counts.items() if n > 1)
    if duplicates:
        raise KitUnavailable(f"duplicate kit labels: {', '.join(duplicates)}")


def _boom_tip_mass(components: Sequence[PlacedComponent], boom_end: bool) -> Optional[BoomTipMass]:
    """A boom-end cruise motor (and propeller), which the tail boom carries (none for a nose or pod motor)."""
    motor = next((c for c in components if c.name == CRUISE_MOTOR_NAME), None)
    if motor is None or not boom_end:
        return None
    return BoomTipMass(mass_kg=motor.mass_kg, x_m=motor.center_m[0])


def _check_tail_boom(sizing: SizingOutput, layout: AircraftLayout, components: Sequence[PlacedComponent],
                     boom_end: bool, settings: FabricationSettings) -> None:
    if layout.tail_boom is not None:
        tail_boom_check(boom_tube(layout.tail_boom.diameter_m, settings), sizing, layout, settings,
                        _boom_tip_mass(components, boom_end))


def _root_block_in_pod(layout: AircraftLayout, blocks: Sequence[KitPart]) -> bool:
    """A V-tail root block whose plate starts ahead of the pod's tail lies inside the tail cone (a blended root)."""
    return any(b.location.position.X < layout.fuselage_length_m for b in blocks)


def _boxes_overlap(a: bd.Shape, b: bd.Shape) -> bool:
    ba, bb = a.bounding_box(), b.bounding_box()
    return all(lo_a < hi_b and lo_b < hi_a for lo_a, hi_a, lo_b, hi_b in zip(ba.min, ba.max, bb.min, bb.max))


def _common_volume_m3(a: bd.Shape, b: bd.Shape) -> float:
    """Volume two solids share (`intersect` may give one shape, a list of them, or nothing)."""
    common = a.intersect(b)
    pieces = common if isinstance(common, list) else ([] if common is None else [common])
    return sum(piece.volume for piece in pieces)


def _check_root_block_clear(blocks: Sequence[KitPart], fuselage: Sequence[KitPart],
                            settings: FabricationSettings) -> None:
    """A root block inside the tail cone must clear every fuselage part (formers, crutch, bulkheads, plates): the kit
    has no relief for it, so a clash refuses the kit, naming the part."""
    for block in blocks:
        solid = part_solid(block, settings)
        for part in fuselage:
            other = part_solid(part, settings)
            if _boxes_overlap(solid, other) and _common_volume_m3(solid, other) > CLASH_VOLUME_M3:
                raise KitUnavailable(f"{block.label}: the V-tail root block inside the tail cone runs into "
                                     f"{part.label}")


def _split_all(parts: Sequence[KitPart], settings: FabricationSettings) -> tuple[KitPart, ...]:
    return tuple(piece for part in parts for piece in split_to_bed(part, settings))


def build_kit(sizing: SizingOutput, layout: AircraftLayout, airfoil: AirfoilShape,
              components: Sequence[PlacedComponent], settings: FabricationSettings) -> Kit:
    """The whole kit for a Group 1-2 design; raises `KitUnavailable` when any part cannot be made from stock."""
    if sizing.dod_group > MAX_DOD_GROUP:
        raise KitUnavailable("laser-cut construction covers Group 1–2 designs only")
    spec = layout_spec(sizing.airframe_layout)
    if (spec.tail, spec.propeller, spec.booms) not in BUILT_LAYOUT_PARTS:
        label: str = LAYOUT_LABELS.get(sizing.airframe_layout, sizing.airframe_layout)
        raise KitUnavailable(f"the laser-cut kit has no parts for the {label} layout yet ({spec.tail} tail, "
                             f"{spec.propeller} propeller, {spec.booms} booms)")
    boom_end = spec.propeller == BOOM_END_PROPELLER
    spar = stiffened_spar_plan(sizing, layout, airfoil, settings)
    _check_tail_boom(sizing, layout, components, boom_end, settings)
    tail = load_airfoil(TAIL_AIRFOIL_NAME)
    tails = tail_spar_tubes(sizing, layout, tail, settings)
    ribs = (*wing_ribs(layout, airfoil, spar, settings), *tail_ribs(layout, tail, tails, settings))
    fuselage = fuselage_parts(layout, spar, components, settings, loads=plate_loads(sizing, settings))
    overwrap = _overwrap(spar, fuselage, settings)
    trays = component_plates(components, layout, settings, spar=spar)
    booms, boom_tubes = boom_parts(layout, spar, settings, components=components,
                                   lift_thrust_n=sizing.vtol_thrust_per_motor_n)
    blocks = tail_root_blocks(layout, tails, settings)
    in_pod = _root_block_in_pod(layout, blocks)
    if in_pod:
        _check_root_block_clear(blocks, fuselage, settings)
    booms = (*booms, *blocks)
    motor = cruise_motor_plate(components, layout, settings, at_boom_end=boom_end)
    parts = _split_all((*ribs, *fuselage, *trays, *booms, *motor), settings)
    tubes = (*spar_tubes(layout, spar, settings), *_socket_and_pins(layout, spar, fuselage, settings),
             *boom_tubes, *_tail_spars(layout, tail, tails, settings))
    stations = former_stations(layout, spar, settings, components)
    linear = (*_stringers(layout, stations, settings), *_edge_strips(layout, spar, airfoil, tail, settings))
    assert_unique_labels(parts, tubes)
    notes = (*((BOOM_MOTOR_WIRING_NOTE,) if boom_end else ()), *((TAIL_CONE_SLOT_NOTE,) if in_pod else ()))
    return Kit(parts=parts, tubes=tubes, linear=linear, covering=_covering(layout, airfoil, spar, settings), spar=spar,
               joins=join_notes(parts), overwrap=overwrap, wing_bolt=_wing_bolt(layout, spar, fuselage, settings),
               notes=notes)
