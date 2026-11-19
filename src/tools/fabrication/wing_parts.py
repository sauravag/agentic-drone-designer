"""Wing and tail rib parts for the laser-cut kit (frame convention: see `kit.py`).

Wing ribs start at the spar joint on the fuselage skin and repeat every `RIB_PITCH_M` to the tip. Each rib
has the spar hole (socket-sized over the socket insertion), a wiring hole inboard of the aileron servo,
lightening holes, and is split at the hinge inside the aileron span. The root rib is a 3 mm doubler with
an anti-rotation pin hole. Tail ribs carry one stock tube spar wherever it fits the local depth.
"""

import dataclasses
import math
from typing import Callable, Final, Optional, Sequence

import build123d as bd

from src.schemas.fabrication import FabricationSettings, PartFamily, TubeStock
from src.tools.aero_cad.airfoil_geometry import AirfoilShape
from src.tools.aero_cad.layout import (
    AILERON_CHORD_FRACTION,
    AILERON_ETA,
    TAIL_CONTROL_CHORD_FRACTION,
    TAIL_CONTROL_ETA,
    AircraftLayout,
    SurfacePlanform,
    fuselage_radius_at,
)
from src.tools.aero_cad.structure import RIB_PITCH_M, SPAR_CHORD_STATION, STATION_EPS
from src.tools.fabrication.kit import KitPart
from src.tools.fabrication.rib_outline import (
    Hole,
    camber_point,
    contains,
    drill,
    keeps_web,
    lightening_holes,
    notched_profile,
    rect_face,
    split_at,
)
from src.tools.fabrication.tubes import FIT_EPS_M, KitUnavailable, SparPlan, usable_depth_m

AILERON_HINGE_FRACTION: Final[float] = 1.0 - AILERON_CHORD_FRACTION
TAIL_HINGE_FRACTION: Final[float] = 1.0 - TAIL_CONTROL_CHORD_FRACTION
WIRING_CHORD_FRACTIONS: Final[tuple[float, ...]] = (0.50, 0.45, 0.40)  # 50 % chord, forward if too thin
WIRING_HOLE_DIAMETER_M: Final[float] = 0.008
PIN_CHORD_FRACTION: Final[float] = 0.70
PIN_ROD_DIAMETER_M: Final[float] = 0.004
LIGHTENING_CHORD_FRACTIONS: Final[tuple[float, ...]] = (0.38, 0.62)  # either side of the wiring hole
TAIL_AIRFOIL_NAME: Final[str] = "naca0010"  # OpenVSP's default WING section, which the tail surfaces use
SERVO_PLATE_CHORD_M: Final[float] = 0.040
SERVO_PLATE_SPAN_M: Final[float] = 0.050
SERVO_CUTOUT_CHORD_M: Final[float] = 0.013  # 9 g micro servo body, lying spanwise
SERVO_CUTOUT_SPAN_M: Final[float] = 0.024
SERVO_CHORD_FRACTION: Final[float] = 0.45  # plate centre, ahead of the hinge
TAIL_MOUNT_MIN_WIDTH_M: Final[float] = 0.020
RIB_ROTATION_DEG: Final[tuple[float, float, float]] = (90.0, 0.0, 0.0)  # local y -> up, local z -> -y
FLAT_ROTATION_DEG: Final[tuple[float, float, float]] = (0.0, 0.0, 0.0)


def spar_depth_at(surface: SurfacePlanform, airfoil: AirfoilShape, settings: FabricationSettings) -> Callable[[float], float]:
    """Usable tube depth at span station y: airfoil thickness at the spar chord station minus the webs."""
    ratio = float(airfoil.to_asb().local_thickness(SPAR_CHORD_STATION))
    return lambda y: usable_depth_m(ratio * surface.chord_at(y), settings)


def rib_stations(start_m: float, semi_span_m: float) -> tuple[float, ...]:
    """`start_m` and every RIB_PITCH_M outboard of it, up to the semi-span."""
    count = math.floor((semi_span_m - start_m) / RIB_PITCH_M + STATION_EPS) + 1
    return tuple(start_m + k * RIB_PITCH_M for k in range(count))


def _label(prefix: str, index: int) -> str:
    return f"{prefix}{index + 1:02d}"


def _rib_pieces(
    label: str, family: PartFamily, profile: bd.Face, chord: float, holes: Sequence[Hole],
    hinge_x_m: Optional[float], airfoil: AirfoilShape, location: bd.Location,
) -> tuple[KitPart, ...]:
    """One notched rib profile (or its fore + aft pieces at the hinge), drilled and lightened.

    All pieces share `location`."""
    faces = (profile,) if hinge_x_m is None else split_at(profile, hinge_x_m)
    names = (label, label + "A")[: len(faces)]
    owned = [tuple(h for h in holes if contains(face, h)) for face in faces]
    if sum(len(o) for o in owned) != len(holes):
        raise KitUnavailable(f"a hole breaks out of rib {label}")
    centres = [camber_point(airfoil, chord, f) for f in LIGHTENING_CHORD_FRACTIONS]
    parts: list[KitPart] = []
    for name, face, own in zip(names, faces, owned):
        light = lightening_holes(face, own, centres)
        parts.append(KitPart(name, family, drill(face, own + light, name), location))
    return tuple(parts)


def _mirrored(part: KitPart, side_from: str, side_to: str) -> KitPart:
    """Left-panel copy: same outline, location reflected in y (frame convention in kit.py)."""
    pos = part.location.position
    location = bd.Location((pos.X, -pos.Y, pos.Z), tuple(part.location.orientation))
    return dataclasses.replace(part, label=part.label.replace(side_from, side_to, 1), location=location)


def _spar_hole_m(spar: SparPlan, y_m: float, settings: FabricationSettings) -> float:
    """Socket over the insertion length; elsewhere the segment at y (the inboard one on a step)."""
    if y_m < spar.joint_y_m + spar.insertion_m - FIT_EPS_M:
        return spar.socket.od_m + settings.fit_clearance_m
    seg = next(s for s in spar.segments if s.y_start_m - FIT_EPS_M <= y_m <= s.y_end_m + FIT_EPS_M)
    return seg.tube.od_m + settings.fit_clearance_m


def _in_span(eta: float, span: tuple[float, float]) -> bool:
    return span[0] <= eta <= span[1]


def _servo_bay_y(stations: Sequence[float], wing: SurfacePlanform) -> float:
    """Middle of the first aileron bay (or of the aileron span if it holds fewer than two ribs)."""
    inside = [y for y in stations if _in_span(y / wing.semi_span_m, AILERON_ETA)]
    if len(inside) >= 2:
        return 0.5 * (inside[0] + inside[1])
    return 0.5 * sum(AILERON_ETA) * wing.semi_span_m


def _wing_rib(
    index: int, y: float, servo_y: float, layout: AircraftLayout, airfoil: AirfoilShape,
    spar: SparPlan, settings: FabricationSettings,
) -> tuple[KitPart, ...]:
    wing = layout.wing
    chord = wing.chord_at(y)
    label, is_root = _label("WR-", index), index == 0
    profile = notched_profile(airfoil, chord, settings)
    holes = [Hole(*camber_point(airfoil, chord, SPAR_CHORD_STATION), _spar_hole_m(spar, y, settings))]
    if is_root:
        holes.append(_placed_hole(profile, airfoil, chord, (PIN_CHORD_FRACTION,), _pin_hole_m(settings), holes, label))
    if y < servo_y:  # aileron servo leads run inboard from the servo bay
        holes.append(_placed_hole(profile, airfoil, chord, WIRING_CHORD_FRACTIONS, WIRING_HOLE_DIAMETER_M, holes, label))
    hinge = AILERON_HINGE_FRACTION * chord if _in_span(y / wing.semi_span_m, AILERON_ETA) else None
    location = bd.Location((wing.x_le_root_m, y, wing.z_at(y)), RIB_ROTATION_DEG)
    family: PartFamily = "doubler" if is_root else "rib"
    return _rib_pieces(label, family, profile, chord, holes, hinge, airfoil, location)


def _placed_hole(
    profile: bd.Face, airfoil: AirfoilShape, chord: float, fractions: Sequence[float], diameter_m: float,
    others: Sequence[Hole], label: str,
) -> Hole:
    """First camber-line station in `fractions` where the hole keeps the minimum web."""
    for fraction in fractions:
        hole = Hole(*camber_point(airfoil, chord, fraction), diameter_m)
        if keeps_web(profile, hole, others):
            return hole
    raise KitUnavailable(f"no room for a {diameter_m * 1e3:.1f} mm hole in rib {label}")


def _pin_hole_m(settings: FabricationSettings) -> float:
    if not any(abs(r - PIN_ROD_DIAMETER_M) <= FIT_EPS_M for r in settings.rods_m):
        raise KitUnavailable(f"no {PIN_ROD_DIAMETER_M * 1e3:.0f} mm rod in stock for the wing pin")
    return PIN_ROD_DIAMETER_M + settings.fit_clearance_m


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
    """Right-panel ribs `WR-01…` (+ aileron pieces `WR-kA`, servo plate `WR-SV`), then the left panel `WL-…`."""
    stations = rib_stations(spar.joint_y_m, layout.wing.semi_span_m)
    servo_y = _servo_bay_y(stations, layout.wing)
    right = [p for i, y in enumerate(stations) for p in _wing_rib(i, y, servo_y, layout, airfoil, spar, settings)]
    right.append(_servo_plate(layout.wing, servo_y))
    return tuple(right) + tuple(_mirrored(p, "WR-", "WL-") for p in right)


def _tail_root_m(layout: AircraftLayout) -> float:
    """Tail surfaces start at the boom skin (or the fuselage skin under the tail)."""
    if layout.tail_boom is not None:
        return layout.tail_boom.diameter_m / 2.0
    x = layout.htail.x_le_root_m + SPAR_CHORD_STATION * layout.htail.root_chord_m
    return fuselage_radius_at(layout, x)


def tail_spar_tube(layout: AircraftLayout, airfoil: AirfoilShape, settings: FabricationSettings) -> TubeStock:
    """Smallest stock tube that fits the thinner tail root (no stress check: tail loads are small)."""
    root = _tail_root_m(layout)
    depth = min(spar_depth_at(s, airfoil, settings)(root) for s in (layout.htail, layout.vtail))
    tube = next((t for t in settings.tubes if t.od_m <= depth + FIT_EPS_M), None)
    if tube is None:
        raise KitUnavailable(f"no stock tube fits the tail root ({depth * 1e3:.1f} mm usable depth)")
    return tube


def _tail_surface_ribs(
    prefix: str, surface: SurfacePlanform, place: Callable[[float], bd.Location], start_m: float,
    airfoil: AirfoilShape, tube: TubeStock, settings: FabricationSettings,
) -> list[KitPart]:
    """Ribs of one tail surface; the spar hole appears only where the tube fits the local depth."""
    depth = spar_depth_at(surface, airfoil, settings)
    parts: list[KitPart] = []
    for index, span in enumerate(rib_stations(start_m, surface.semi_span_m)):
        chord = surface.chord_at(span)
        fits = tube.od_m <= depth(span) + FIT_EPS_M
        holes = [Hole(*camber_point(airfoil, chord, SPAR_CHORD_STATION), tube.od_m + settings.fit_clearance_m)] if fits else []
        hinge = TAIL_HINGE_FRACTION * chord if _in_span(span / surface.semi_span_m, TAIL_CONTROL_ETA) else None
        profile = notched_profile(airfoil, chord, settings)
        parts += _rib_pieces(_label(prefix, index), "rib", profile, chord, holes, hinge, airfoil, place(span))
    return parts


def _tail_mount(layout: AircraftLayout, root_m: float) -> KitPart:
    """Ply saddle on top of the tail boom under the tail roots."""
    width = max(2.0 * root_m, TAIL_MOUNT_MIN_WIDTH_M)
    outline = rect_face(0.0, -width / 2.0, layout.htail.root_chord_m, width / 2.0)
    z = (layout.tail_boom.z_m if layout.tail_boom is not None else layout.htail.z_root_m) + root_m
    return KitPart("TM-1", "plate", outline, bd.Location((layout.htail.x_le_root_m, 0.0, z), FLAT_ROTATION_DEG))


def tail_ribs(layout: AircraftLayout, airfoil: AirfoilShape, settings: FabricationSettings) -> tuple[KitPart, ...]:
    """Horizontal-tail ribs `HR-…`/`HL-…`, fin ribs `V-…` (split at the hinge) and the mount plate `TM-1`.

    `airfoil` is the tail section (`TAIL_AIRFOIL_NAME` for OpenVSP-built tails).
    """
    root = _tail_root_m(layout)
    tube = tail_spar_tube(layout, airfoil, settings)
    ht, vt = layout.htail, layout.vtail
    right = _tail_surface_ribs(
        "HR-", ht, lambda y: bd.Location((ht.x_le_root_m, y, ht.z_at(y)), RIB_ROTATION_DEG), root, airfoil, tube, settings
    )
    fin = _tail_surface_ribs(
        "V-", vt, lambda h: bd.Location((vt.x_le_root_m, 0.0, vt.z_root_m + h), FLAT_ROTATION_DEG), root, airfoil, tube, settings
    )
    left = [_mirrored(p, "HR-", "HL-") for p in right]
    return (*right, *left, *fin, _tail_mount(layout, root))
