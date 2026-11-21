"""Wing and tail rib parts for the laser-cut kit (frame convention: see `kit.py`).

Wing ribs start at the spar joint on the fuselage skin and repeat every `RIB_PITCH_M` to the tip. Each rib
has the spar hole sized to the spar tube at its station, a wiring hole inboard of the aileron servo,
lightening holes, and is split at the hinge inside the aileron span. The root rib is a 3 mm doubler with
an anti-rotation pin hole (the R31 rod); when the pin needs one, a pin boss doubler `WR-PD` / `WL-PD`
(an annulus PIN_BOSS_WIDTH_M wide round the pin hole, doubler stock) is glued to the root rib's outboard face. The
root rib also carries the R36 wing bolt hole, and the captive-nut plate `NP-R` / `NP-L` is glued to its outboard face
(`wing_retention`).
Each tail surface's ribs carry its own stock spar, strength-sized at the root for the R35 tail load
(`tail_spar_tubes`), wherever it fits the local depth.
"""

import dataclasses
import math
from dataclasses import dataclass
from typing import Callable, Final, Optional, Sequence

import build123d as bd

from src.schemas.fabrication import FabricationSettings, PartFamily, TubeStock
from src.schemas.state import SizingOutput
from src.tools.aero_cad.airfoil_geometry import AirfoilShape, load_airfoil
from src.tools.aero_cad.layout import (
    AILERON_CHORD_FRACTION,
    AILERON_ETA,
    TAIL_CONTROL_CHORD_FRACTION,
    TAIL_CONTROL_ETA,
    AircraftLayout,
    SurfacePlanform,
)
from src.tools.aero_cad.structure import RIB_PITCH_M, SPAR_CHORD_STATION, STATION_EPS
from src.tools.fabrication.kit import KitPart
from src.tools.fabrication.rib_outline import (
    Hole,
    airfoil_face,
    camber_point,
    circle_face,
    contains,
    drill,
    keeps_web,
    lightening_holes,
    notched_profile,
    oblique_diameter_m,
    rect_face,
    split_at,
)
from src.tools.fabrication.skin import max_skin_radius_m
from src.tools.fabrication.tubes import (
    FIT_EPS_M,
    MIN_WEB_M,
    KitUnavailable,
    SparPlan,
    elliptic_root_moment_nm,
    smallest_passing,
    spar_member_at,
    tail_surface_load_n,
    usable_depth_m,
)
from src.tools.fabrication.wing_retention import (
    BOLT_CHORD_FRACTION,
    M5_CLEARANCE_HOLE_M,
    NUT_PLATE_DIAMETER_M,
    NUT_PLATE_LABELS,
    check_clear_of_holes,
    nut_plate_outline,
)
from src.tools.fabrication.wing_stiffness import dbox_web_x_m

AILERON_HINGE_FRACTION: Final[float] = 1.0 - AILERON_CHORD_FRACTION
TAIL_HINGE_FRACTION: Final[float] = 1.0 - TAIL_CONTROL_CHORD_FRACTION
WIRING_CHORD_FRACTIONS: Final[tuple[float, ...]] = (0.50, 0.45, 0.40, 0.35)  # 50 % chord, forward if too thin
# (or, on the root rib, if the R36 wing bolt and its nut plate are in the way)
WIRING_HOLE_DIAMETER_M: Final[float] = 0.008
PIN_CHORD_FRACTION: Final[float] = 0.70
PIN_BOSS_WIDTH_M: Final[float] = 0.003  # R31 pin boss annulus radial width
ROOT_PIN_BOSS_LABEL: Final[str] = "WR-PD"
ROOT_BOSS_MID_SLABS: Final[float] = 1.5  # the root pin boss slab [y + t, y + 2t] has its mid-plane 1.5 t outboard
AREA_EPS_M2: Final[float] = 1e-10
LIGHTENING_CHORD_FRACTIONS: Final[tuple[float, ...]] = (0.38, 0.62)  # either side of the wiring hole
SERVO_PLATE_CHORD_M: Final[float] = 0.040
SERVO_PLATE_SPAN_M: Final[float] = 0.050
SERVO_CUTOUT_CHORD_M: Final[float] = 0.013  # 9 g micro servo body, lying spanwise
SERVO_CUTOUT_SPAN_M: Final[float] = 0.024
SERVO_CHORD_FRACTION: Final[float] = 0.45  # plate centre, ahead of the hinge
TAIL_MOUNT_MIN_WIDTH_M: Final[float] = 0.020
RIB_ROTATION_DEG: Final[tuple[float, float, float]] = (90.0, 0.0, 0.0)  # local y -> up, local z -> -y
FLAT_ROTATION_DEG: Final[tuple[float, float, float]] = (0.0, 0.0, 0.0)


def spar_depth_at(surface: SurfacePlanform, airfoil: AirfoilShape, settings: FabricationSettings, *,
                  root_doubler_m: Optional[float]) -> Callable[[float], float]:
    """Usable tube depth at span station y, measured the way the rib's web check measures it.

    The local "thickness" is the diameter of the largest circle centred on the camber line at the spar
    station (twice its distance to the outline). That is slightly less than the vertical thickness, because
    the surfaces slope, so a tube sized to it always keeps the minimum web around its hole. The spar line leans
    (taper, dihedral), so the depth is also reduced to the tube whose R19 oblique rib hole still fits. At
    `root_doubler_m` (the wing's root station, None for tails) the rib is the thicker root doubler.
    """
    centre = bd.Vertex(*camber_point(airfoil, 1.0, SPAR_CHORD_STATION), 0.0)
    ratio = 2.0 * centre.distance_to(airfoil_face(airfoil, 1.0).outer_wire())  # unit chord; scales with chord
    lean = line_lean(surface, airfoil, SPAR_CHORD_STATION)
    t_rib, t_root = settings.sheet_for("rib").thickness_m, settings.sheet_for("doubler").thickness_m

    def depth(y: float) -> float:
        on_root = root_doubler_m is not None and abs(y - root_doubler_m) <= FIT_EPS_M
        walk = (t_root if on_root else t_rib) * lean  # R19: the rib hole grows to OD/cos + t tan
        return (usable_depth_m(ratio * surface.chord_at(y), settings) - walk) / math.hypot(1.0, lean)
    return depth


def panel_rib_stations(start_m: float, semi_span_m: float) -> tuple[float, ...]:
    """Rib stations of one panel of any surface (wing, horizontal tail, fin): `start_m` and every RIB_PITCH_M
    outboard of it, up to the semi-span (fin: height)."""
    count = math.floor((semi_span_m - start_m) / RIB_PITCH_M + STATION_EPS) + 1
    return tuple(start_m + k * RIB_PITCH_M for k in range(count))


def socket_axis_m(layout: AircraftLayout, y_m: float = 0.0,
                  fraction: float = SPAR_CHORD_STATION) -> tuple[float, float]:
    """(x, z) where a wing's spar axis (or, with `fraction`, its chord line at that fraction on the camber line)
    crosses span station y. Each socket tube is coaxial with its own wing's spar, so it follows the dihedral and
    meets the other at the centreline (y = 0)."""
    wing = layout.wing
    dx, dz = camber_point(load_airfoil(layout.airfoil), wing.chord_at(y_m), fraction)
    return wing.x_le_root_m + dx, wing.z_at(y_m) + dz


def _label(prefix: str, index: int) -> str:
    return f"{prefix}{index + 1:02d}"


def _rib_pieces(
    label: str, family: PartFamily, profile: bd.Face, chord: float, holes: Sequence[Hole],
    hinge_x_m: Optional[float], airfoil: AirfoilShape, location: bd.Location, keep_out: Sequence[Hole] = (),
) -> tuple[KitPart, ...]:
    """One notched rib profile (or its fore + aft pieces at the hinge), drilled and lightened. Lightening holes
    also keep MIN_WEB_M from each `keep_out` circle (a part glued to the rib, not drilled).

    All pieces share `location`."""
    faces = (profile,) if hinge_x_m is None else split_at(profile, hinge_x_m)
    names = (label, label + "A")[: len(faces)]
    owned = [tuple(h for h in holes if contains(face, h)) for face in faces]
    if sum(len(o) for o in owned) != len(holes):
        raise KitUnavailable(f"a hole breaks out of rib {label}")
    centres = [camber_point(airfoil, chord, f) for f in LIGHTENING_CHORD_FRACTIONS]
    parts: list[KitPart] = []
    for name, face, own in zip(names, faces, owned):
        light = lightening_holes(face, (*own, *keep_out), centres)
        parts.append(KitPart(name, family, drill(face, own + light, name), location))
    return tuple(parts)


def _rib_location(x_m: float, y_station_m: float, z_m: float, thickness_m: float, is_root: bool) -> bd.Location:
    """Right-panel rib location: the slab (extruded toward -y) is centred on its station, except a panel's
    root rib, which lies just outboard of the root station so it stays outside the skin or boom."""
    top = y_station_m + (thickness_m if is_root else thickness_m / 2.0)
    return bd.Location((x_m, top, z_m), RIB_ROTATION_DEG)


def _mirrored(part: KitPart, side_from: str, side_to: str, settings: FabricationSettings) -> KitPart:
    """Left-panel copy: same outline, placed so the extruded solid is the exact y-mirror of the right one."""
    pos = part.location.position
    normal_y = (part.location * bd.Location((0.0, 0.0, 1.0))).position.Y - pos.Y  # extrusion direction in y
    mirrored_y = -pos.Y - normal_y * settings.sheet_for(part.family).thickness_m
    location = bd.Location((pos.X, mirrored_y, pos.Z), tuple(part.location.orientation))
    return dataclasses.replace(part, label=part.label.replace(side_from, side_to, 1), location=location)


def spar_hole_m(spar: SparPlan, y_m: float, settings: FabricationSettings) -> float:
    """Nominal (square-crossing) hole for the spar member at y."""
    return spar_member_at(spar, y_m).od_m + settings.fit_clearance_m


def _in_span(eta: float, span: tuple[float, float]) -> bool:
    return span[0] <= eta <= span[1]


def _servo_bay_y(stations: Sequence[float], wing: SurfacePlanform) -> float:
    """Middle of the first aileron bay (or of the aileron span if it holds fewer than two ribs)."""
    inside = [y for y in stations if _in_span(y / wing.semi_span_m, AILERON_ETA)]
    if len(inside) >= 2:
        return 0.5 * (inside[0] + inside[1])
    return 0.5 * sum(AILERON_ETA) * wing.semi_span_m


def _wing_rib_profile(airfoil: AirfoilShape, chord: float, member: TubeStock, spar: SparPlan,
                      settings: FabricationSettings) -> bd.Face:
    """Rib outline; with the D-box its skin is inset by the sheet thickness ahead of the D-box web (R45)."""
    if not spar.dbox:
        return notched_profile(airfoil, chord, settings)
    return notched_profile(airfoil, chord, settings, inset_m=settings.dbox_sheet_thickness_m,
                           inset_end_x_m=dbox_web_x_m(chord, member.od_m, settings))


def _wing_rib(
    index: int, y: float, servo_y: float, layout: AircraftLayout, airfoil: AirfoilShape,
    spar: SparPlan, settings: FabricationSettings,
) -> tuple[KitPart, ...]:
    wing = layout.wing
    chord = wing.chord_at(y)
    label, is_root = _label("WR-", index), index == 0
    family: PartFamily = "doubler" if is_root else "rib"
    t = settings.sheet_for(family).thickness_m
    mid = y + t / 2.0 if is_root else y  # slab mid-plane (R8)
    member = spar_member_at(spar, y)
    profile = _wing_rib_profile(airfoil, chord, member, spar, settings)
    holes = [_checked(profile, oblique_hole(wing, airfoil, SPAR_CHORD_STATION, y, mid, member.od_m, t, settings), [], label)]
    keep_out = _root_holes(profile, layout, airfoil, spar, settings, holes, label) if is_root else []
    if y < servo_y:  # aileron servo leads run inboard from the servo bay, clear of the nut plate
        holes.append(_placed_hole(profile, airfoil, chord, WIRING_CHORD_FRACTIONS, WIRING_HOLE_DIAMETER_M,
                                  [*holes, *keep_out], label))
    hinge = AILERON_HINGE_FRACTION * chord if _in_span(y / wing.semi_span_m, AILERON_ETA) else None
    location = _rib_location(wing.x_le_root_m, y, wing.z_at(y), t, is_root)
    return _rib_pieces(label, family, profile, chord, holes, hinge, airfoil, location, keep_out)


def _root_holes(profile: bd.Face, layout: AircraftLayout, airfoil: AirfoilShape, spar: SparPlan,
                settings: FabricationSettings, holes: list[Hole], label: str) -> list[Hole]:
    """Adds the root doubler's anti-rotation pin hole (R31) and wing bolt hole (R36) to `holes`, each with its web
    checked; returns the nut plate's footprint, which the wiring and lightening holes keep clear of."""
    wing, y, t = layout.wing, spar.joint_y_m, settings.sheet_for("doubler").thickness_m
    mid = y + t / 2.0
    pin_hole_m(spar.pin.rod_m, settings)  # the pin rod must be in stock
    holes.append(_checked(profile, oblique_hole(wing, airfoil, PIN_CHORD_FRACTION, y, mid, spar.pin.rod_m, t,
                                                settings), holes, label))
    holes.append(_checked(profile, oblique_hole(wing, airfoil, BOLT_CHORD_FRACTION, y, mid, M5_CLEARANCE_HOLE_M,
                                                t, settings, clearance_m=0.0), holes, label))
    centre = _nut_plate_centre(layout, airfoil, spar, settings)
    return [Hole(centre.x_m, centre.y_m, NUT_PLATE_DIAMETER_M)]


def line_lean(surface: SurfacePlanform, airfoil: AirfoilShape, fraction: float) -> float:
    """Off-normal travel per unit span (fin: per unit height) of the surface's line at `fraction` chord on the
    camber line; the line is straight because chord and dihedral height are linear in span."""
    span = surface.semi_span_m
    (x0, v0), (x1, v1) = (camber_point(airfoil, surface.chord_at(s), fraction) for s in (0.0, span))
    rise = surface.z_at(span) - surface.z_at(0.0) if surface.is_symmetric else 0.0
    return math.hypot(x1 - x0, v1 - v0 + rise) / span


def oblique_hole(surface: SurfacePlanform, airfoil: AirfoilShape, fraction: float, station_m: float, mid_m: float,
                 rod_od_m: float, thickness_m: float, settings: FabricationSettings,
                 clearance_m: Optional[float] = None) -> Hole:
    """R19: rib-local hole for a rod on the surface's `fraction` line through a rib at `station_m` whose slab
    mid-plane is at `mid_m`: centred where the line crosses the mid-plane, OD/cos + t tan + clearance across (the
    settings' fit clearance unless `clearance_m` is given, e.g. 0 for a bolt's own clearance hole)."""
    dx, dv = camber_point(airfoil, surface.chord_at(mid_m), fraction)
    if surface.is_symmetric:  # the rib is placed at its station's dihedral height
        dv += surface.z_at(mid_m) - surface.z_at(station_m)
    lean = line_lean(surface, airfoil, fraction)
    clearance = settings.fit_clearance_m if clearance_m is None else clearance_m
    return Hole(dx, dv, oblique_diameter_m(rod_od_m, thickness_m, lean) + clearance)


def _checked(profile: bd.Face, hole: Hole, others: Sequence[Hole], label: str) -> Hole:
    if not keeps_web(profile, hole, others):
        raise KitUnavailable(
            f"no room for a {hole.diameter_m * 1e3:.1f} mm hole with a {MIN_WEB_M * 1e3:.0f} mm web in rib {label}"
        )
    return hole


def _placed_hole(
    profile: bd.Face, airfoil: AirfoilShape, chord: float, fractions: Sequence[float], diameter_m: float,
    others: Sequence[Hole], label: str,
) -> Hole:
    """First camber-line station in `fractions` where the hole keeps the minimum web."""
    for fraction in fractions:
        hole = Hole(*camber_point(airfoil, chord, fraction), diameter_m)
        if keeps_web(profile, hole, others):
            return hole
    raise KitUnavailable(
        f"no room for a {diameter_m * 1e3:.1f} mm hole with a {MIN_WEB_M * 1e3:.0f} mm web in rib {label}"
    )


def pin_hole_m(rod_m: float, settings: FabricationSettings) -> float:
    """Nominal hole for the anti-rotation pin rod, which must be in stock."""
    if not any(abs(r - rod_m) <= FIT_EPS_M for r in settings.rods_m):
        raise KitUnavailable(f"no {rod_m * 1e3:.0f} mm rod in stock for the wing pin")
    return rod_m + settings.fit_clearance_m


def _root_pin_boss(layout: AircraftLayout, airfoil: AirfoilShape, spar: SparPlan,
                   settings: FabricationSettings) -> tuple[KitPart, ...]:
    """R31: when the root doubler alone cannot bear the pin load, a pin boss of doubler stock on its outboard face
    (the next slab outboard), an annulus PIN_BOSS_WIDTH_M wide round the pin hole, inside the root profile."""
    if not spar.pin.root_boss:
        return ()
    wing, y, label = layout.wing, spar.joint_y_m, ROOT_PIN_BOSS_LABEL
    t = settings.sheet_for("doubler").thickness_m
    hole = oblique_hole(wing, airfoil, PIN_CHORD_FRACTION, y, y + ROOT_BOSS_MID_SLABS * t, spar.pin.rod_m, t, settings)
    ring = circle_face(Hole(hole.x_m, hole.y_m, hole.diameter_m + 2.0 * PIN_BOSS_WIDTH_M))
    if ring.cut(notched_profile(airfoil, wing.chord_at(y), settings)).area > AREA_EPS_M2:
        raise KitUnavailable(f"anti-rotation pin: the pin boss {label} does not fit inside the root rib")
    location = _rib_location(wing.x_le_root_m, y + t, wing.z_at(y), t, True)  # slab [y + t, y + 2t]
    return (KitPart(label, "doubler", drill(ring, (hole,), label), location),)


def _nut_plate_centre(layout: AircraftLayout, airfoil: AirfoilShape, spar: SparPlan,
                      settings: FabricationSettings) -> Hole:
    """Where the bolt line crosses the nut plate's slab [y + t, y + 2t] (outboard of the root doubler), rib-local."""
    y, t = spar.joint_y_m, settings.sheet_for("doubler").thickness_m
    return oblique_hole(layout.wing, airfoil, BOLT_CHORD_FRACTION, y, y + ROOT_BOSS_MID_SLABS * t,
                        M5_CLEARANCE_HOLE_M, t, settings, clearance_m=0.0)


def _nut_plate(layout: AircraftLayout, airfoil: AirfoilShape, spar: SparPlan,
               settings: FabricationSettings) -> KitPart:
    """R36 `NP-R`: the captive-nut plate on the root doubler's outboard face (slab [y + t, y + 2t], beside any pin
    boss), inside the root profile and MIN_WEB_M clear of the spar, the pin and the pin boss."""
    wing, y, label = layout.wing, spar.joint_y_m, NUT_PLATE_LABELS[1]
    t = settings.sheet_for("doubler").thickness_m
    mid, centre = y + ROOT_BOSS_MID_SLABS * t, _nut_plate_centre(layout, airfoil, spar, settings)
    outline = nut_plate_outline(centre.x_m, centre.y_m, settings, label)
    if outline.cut(notched_profile(airfoil, wing.chord_at(y), settings)).area > AREA_EPS_M2:
        raise KitUnavailable(f"wing bolt: the nut plate {label} does not fit inside the root rib")
    spar_rod = oblique_hole(wing, airfoil, SPAR_CHORD_STATION, y, mid, spar_member_at(spar, y).od_m, t, settings)
    pin = oblique_hole(wing, airfoil, PIN_CHORD_FRACTION, y, mid, spar.pin.rod_m, t, settings)
    ring = Hole(pin.x_m, pin.y_m, pin.diameter_m + 2.0 * PIN_BOSS_WIDTH_M) if spar.pin.root_boss else pin
    disc = Hole(centre.x_m, centre.y_m, NUT_PLATE_DIAMETER_M)
    check_clear_of_holes(disc, (spar_rod, ring), label, "the spar, the pin or its boss")
    location = _rib_location(wing.x_le_root_m, y + t, wing.z_at(y), t, True)  # slab [y + t, y + 2t]
    return KitPart(label, "doubler", outline, location)


def _servo_plate(wing: SurfacePlanform, y: float) -> KitPart:
    """Flat plate in the chord plane of the first aileron bay, with a cut-out for the servo body."""
    half_c, half_s = SERVO_PLATE_CHORD_M / 2.0, SERVO_PLATE_SPAN_M / 2.0
    cut_c, cut_s = SERVO_CUTOUT_CHORD_M / 2.0, SERVO_CUTOUT_SPAN_M / 2.0
    outline = rect_face(-half_c, -half_s, half_c, half_s).cut(rect_face(-cut_c, -cut_s, cut_c, cut_s)).faces()[0]
    x = wing.x_le_root_m + SERVO_CHORD_FRACTION * wing.chord_at(y)
    return KitPart("WR-SV", "plate", outline, bd.Location((x, y, wing.z_at(y)), FLAT_ROTATION_DEG))


def wing_ribs(
    layout: AircraftLayout, airfoil: AirfoilShape, spar: SparPlan, settings: FabricationSettings
) -> tuple[KitPart, ...]:
    """Right-panel ribs `WR-01…` (+ aileron pieces `WR-kA`, servo plate `WR-SV`, pin boss `WR-PD` when needed),
    then the left panel `WL-…`, then the R36 captive-nut plates `NP-R`, `NP-L`."""
    stations = panel_rib_stations(spar.joint_y_m, layout.wing.semi_span_m)
    servo_y = _servo_bay_y(stations, layout.wing)
    right = [p for i, y in enumerate(stations) for p in _wing_rib(i, y, servo_y, layout, airfoil, spar, settings)]
    right.append(_servo_plate(layout.wing, servo_y))
    right.extend(_root_pin_boss(layout, airfoil, spar, settings))
    nut = _nut_plate(layout, airfoil, spar, settings)
    nuts = (nut, _mirrored(nut, NUT_PLATE_LABELS[1], NUT_PLATE_LABELS[-1], settings))
    return tuple(right) + tuple(_mirrored(p, "WR-", "WL-", settings) for p in right) + nuts


def _over_root_chord(layout: AircraftLayout, surface: SurfacePlanform) -> float:
    return max_skin_radius_m(layout, surface.x_le_root_m, surface.x_le_root_m + surface.root_chord_m)


def tail_root_radius_m(layout: AircraftLayout) -> float:
    """R20: the horizontal-tail root ribs start outside both the boom and the fuselage skin over the whole root
    chord (as the wing starts at the skin joint)."""
    boom = layout.tail_boom.diameter_m / 2.0 if layout.tail_boom is not None else 0.0
    return max(boom, _over_root_chord(layout, layout.htail))


def _boom_top_m(layout: AircraftLayout) -> Optional[float]:
    boom = layout.tail_boom
    return None if boom is None else boom.z_m + boom.diameter_m / 2.0


def has_fin_mount(layout: AircraftLayout) -> bool:
    """R20: `TM-1` exists only where the boom, not the fuselage skin, limits the fin root."""
    top = _boom_top_m(layout)
    return top is not None and top >= _over_root_chord(layout, layout.vtail)


def fin_start_m(layout: AircraftLayout, settings: FabricationSettings) -> float:
    """Height of the first fin rib above the fin root: on the fin mount `TM-1` on the boom (R9) where the boom is
    the limit, else on the fuselage skin over the root chord, bonded to the covering (R20)."""
    skin, boom = _over_root_chord(layout, layout.vtail), _boom_top_m(layout)
    if boom is None:
        return skin - layout.vtail.z_root_m
    if boom >= skin:  # `has_fin_mount`: the boom limits the fin root, which sits on TM-1
        return boom + settings.sheet_for("plate").thickness_m - layout.vtail.z_root_m
    return skin - layout.vtail.z_root_m


@dataclass(frozen=True)
class TailSpars:
    """Stock spar of each tail surface (R35): both horizontal-tail panels share `htail`; the fin has `fin`."""

    htail: TubeStock
    fin: TubeStock


@dataclass(frozen=True)
class TailSparLoads:
    """Bending moment at each tail spar's root (N m): one horizontal-tail panel, and the fin."""

    htail_nm: float
    fin_nm: float


def tail_spar_loads(sizing: SizingOutput, layout: AircraftLayout, settings: FabricationSettings) -> TailSparLoads:
    """R35: each surface lifts q_max S CL_max (`tail_surface_load_n`), elliptically over its semi-span (each
    horizontal-tail panel half of S_h, the fin all of S_v over its height); the spar's moment is taken where it
    starts, at the tail root (R20) or the fin start."""
    half_h = tail_surface_load_n(sizing, sizing.horizontal_tail_area_m2, settings) / 2.0
    fin = tail_surface_load_n(sizing, sizing.vertical_tail_area_m2, settings)
    ht, vt = layout.htail, layout.vtail
    return TailSparLoads(htail_nm=elliptic_root_moment_nm(half_h, ht.semi_span_m, tail_root_radius_m(layout)),
                         fin_nm=elliptic_root_moment_nm(fin, vt.semi_span_m, fin_start_m(layout, settings)))


def _tail_spar_tube(name: str, station: str, surface: SurfacePlanform, root_m: float, moment_nm: float,
                    airfoil: AirfoilShape, settings: FabricationSettings) -> TubeStock:
    """The smallest stock tube or rod that carries `moment_nm` at the root and fits the root's usable depth."""
    depth = spar_depth_at(surface, airfoil, settings, root_doubler_m=None)(root_m)
    tube = smallest_passing(((moment_nm, 0.0),), depth, settings)
    if tube is None:
        raise KitUnavailable(
            f"tail spar: no stock tube or rod carries the {name} load ({moment_nm:.1f} N·m) within its "
            f"{depth * 1e3:.1f} mm root depth at {station} = {root_m:.3f} m"
        )
    return tube


def tail_spar_tubes(sizing: SizingOutput, layout: AircraftLayout, airfoil: AirfoilShape,
                    settings: FabricationSettings) -> TailSpars:
    """R35: each tail surface's spar, strength-sized at its root (`tail_spar_loads`). A spar runs out to where the
    depth stops admitting it (`_tail_surface_ribs`); the moment falls outboard, so the root check covers it."""
    loads = tail_spar_loads(sizing, layout, settings)
    htail = _tail_spar_tube("horizontal tail", "y", layout.htail, tail_root_radius_m(layout), loads.htail_nm,
                            airfoil, settings)
    fin = _tail_spar_tube("fin", "height", layout.vtail, fin_start_m(layout, settings), loads.fin_nm,
                          airfoil, settings)
    return TailSpars(htail=htail, fin=fin)


def _tail_surface_ribs(
    prefix: str, surface: SurfacePlanform, place: Callable[[int, float], bd.Location], start_m: float,
    airfoil: AirfoilShape, tube: TubeStock, settings: FabricationSettings,
) -> list[KitPart]:
    """Ribs of one tail surface; the spar hole appears only where it keeps the minimum web (the spar ends
    inboard of thinner ribs), and the root rib must carry it."""
    parts: list[KitPart] = []
    for index, span in enumerate(panel_rib_stations(start_m, surface.semi_span_m)):
        label, chord = _label(prefix, index), surface.chord_at(span)
        profile = notched_profile(airfoil, chord, settings)
        hole = tail_spar_hole(surface, airfoil, tube, index, span, settings)
        holes = [hole] if keeps_web(profile, hole, []) else []
        if index == 0 and not holes:
            raise KitUnavailable(f"the tail spar does not fit rib {label} with a {MIN_WEB_M * 1e3:.0f} mm web")
        hinge = TAIL_HINGE_FRACTION * chord if _in_span(span / surface.semi_span_m, TAIL_CONTROL_ETA) else None
        parts += _rib_pieces(label, "rib", profile, chord, holes, hinge, airfoil, place(index, span))
    return parts


def tail_rib_mid_m(surface: SurfacePlanform, index: int, span_m: float, thickness_m: float) -> float:
    """Slab mid-plane of a tail rib: fin ribs extrude up from their station, a horizontal-tail root rib lies
    outboard of its station, the other horizontal-tail ribs are centred on it (R8)."""
    if not surface.is_symmetric or index == 0:
        return span_m + thickness_m / 2.0
    return span_m


def tail_spar_hole(surface: SurfacePlanform, airfoil: AirfoilShape, tube: TubeStock, index: int, span_m: float,
                   settings: FabricationSettings) -> Hole:
    """R19 spar hole in tail rib `index` at `span_m`."""
    t = settings.sheet_for("rib").thickness_m
    mid = tail_rib_mid_m(surface, index, span_m, t)
    return oblique_hole(surface, airfoil, SPAR_CHORD_STATION, span_m, mid, tube.od_m, t, settings)


def _fin_mount(layout: AircraftLayout, settings: FabricationSettings) -> tuple[KitPart, ...]:
    """Ply saddle on top of the tail boom under the fin root (only where the boom is the limit, R20)."""
    boom, vt = layout.tail_boom, layout.vtail
    if boom is None or not has_fin_mount(layout):
        return ()
    width = max(boom.diameter_m, TAIL_MOUNT_MIN_WIDTH_M)
    outline = rect_face(0.0, -width / 2.0, vt.root_chord_m, width / 2.0)
    location = bd.Location((vt.x_le_root_m, 0.0, boom.z_m + boom.diameter_m / 2.0), FLAT_ROTATION_DEG)
    return (KitPart("TM-1", "plate", outline, location),)


def tail_ribs(layout: AircraftLayout, airfoil: AirfoilShape, spars: TailSpars,
              settings: FabricationSettings) -> tuple[KitPart, ...]:
    """Horizontal-tail ribs `HR-…`/`HL-…`, fin ribs `V-…` (split at the hinge) and, where the boom limits the fin
    root, the fin mount `TM-1` (R20). Each surface's ribs carry its own spar from `spars` (`tail_spar_tubes`).

    `airfoil` is the tail section (`layout.TAIL_AIRFOIL_NAME` for OpenVSP-built tails). Fin ribs start on top of
    the mount plate, or on the fuselage skin.
    """
    radius = tail_root_radius_m(layout)
    ht, vt = layout.htail, layout.vtail
    t_rib = settings.sheet_for("rib").thickness_m
    fin_start = fin_start_m(layout, settings)

    def htail_at(index: int, y: float) -> bd.Location:
        return _rib_location(ht.x_le_root_m, y, ht.z_at(y), t_rib, index == 0)

    def fin_at(index: int, h: float) -> bd.Location:
        return bd.Location((vt.x_le_root_m, 0.0, vt.z_root_m + h), FLAT_ROTATION_DEG)

    right = _tail_surface_ribs("HR-", ht, htail_at, radius, airfoil, spars.htail, settings)
    fin = _tail_surface_ribs("V-", vt, fin_at, fin_start, airfoil, spars.fin, settings)
    left = [_mirrored(p, "HR-", "HL-", settings) for p in right]
    return (*right, *left, *fin, *_fin_mount(layout, settings))
