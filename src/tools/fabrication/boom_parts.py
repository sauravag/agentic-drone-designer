"""Boom tubes, boom clamps and motor mount plates for the laser-cut kit (frame convention: see `kit.py`).

- Boom tubes (R13): `boom_tube(layout diameter)` stock. The tail boom `TB-1` sits where the layout puts it. Each lift
  boom `LB-R` / `LB-L` keeps its layout x, y and length; its top sits one plate thickness under the rotor hub plane
  (the layout boom's top), so the motor plates carry the motors at the layout height. Where the spar member is too
  low for that (the clamp needs a MIN_WEB_M web between the two tubes), the boom drops further and the motor plates
  sit under the hub plane by the difference.
- Boom clamps `BC-R1`/`BC-R2` (`BC-L…` mirrored), quadplane only: a laminated pair of 3 mm plates in the boom's
  yz-plane at the spar's x, BC-x1 just ahead of the spar axis and BC-x2 just aft of it (local X -> y, local Y -> z,
  extruded aft like a former). Each plate rings the boom tube (hole = boom OD + fit clearance) and its top edge is the
  saddle the spar member bears on: the member's underside ((OD + clearance) / 2 below its axis, stretched for the
  axis' slope) along the spar axis incl. dihedral. The boom axis sits MIN_WEB_M + hole radius below the lowest point
  of that saddle or lower (above): the plate keeps at least a MIN_WEB_M web between the two tubes and grows taller
  when the boom sits lower. A plate side that would reach a wing rib
  stops one rib-sheet thickness clear of it; the boom hole must still keep MIN_WEB_M there.
- Motor plates `MP-1…4`: discs on top of each lift boom under every rotor hub, with a centre mark hole and a slotted
  cross for 16 / 19 / 25 mm bolt circles; they sit under the motor (`vtol_motor_k` when given).
"""

import math
from dataclasses import dataclass
from typing import Final, Optional, Sequence

import build123d as bd

from src.schemas.fabrication import FabricationSettings
from src.tools.aero_cad.components import PlacedComponent
from src.tools.aero_cad.layout import AircraftLayout, Boom
from src.tools.fabrication.fuselage_parts import socket_axis_m, station_location
from src.tools.fabrication.kit import KitPart, TubePart, TubeRole
from src.tools.fabrication.rib_outline import Hole, circle_face, drill, keeps_web, largest_face, rect_face
from src.tools.fabrication.tubes import MIN_WEB_M, KitUnavailable, SparPlan, boom_tube
from src.tools.fabrication.wing_parts import spar_hole_m, wing_rib_stations

CLAMP_LABELS: Final[tuple[str, ...]] = ("BC-R1", "BC-R2", "BC-L1", "BC-L2")
CLAMP_CHEEK_M: Final[float] = 0.005  # plate material beside and below the boom hole
MOTOR_CENTRE_HOLE_M: Final[float] = 0.003  # centre mark hole
MOTOR_BOLT_CIRCLES_M: Final[tuple[float, ...]] = (0.016, 0.019, 0.025)  # common motor bolt patterns
MOTOR_SLOT_WIDTH_M: Final[float] = 0.0032  # M3 clearance
MOTOR_SLOT_ANGLES_DEG: Final[tuple[float, ...]] = (45.0, 135.0, 225.0, 315.0)  # a cross, diagonal to the boom
LIFT_MOTOR_PREFIX: Final[str] = "vtol_motor_"
X_AXIS: Final[tuple[float, float, float]] = (1.0, 0.0, 0.0)
WEB_EPS_M: Final[float] = 1e-12
RIGHT: Final[int] = 1
LEFT: Final[int] = -1


@dataclass(frozen=True)
class _Clamp:
    """One side's clamp geometry, in |y| (outboard positive) and aircraft z."""

    side: int
    x_m: float  # spar axis x at the boom: BC-x1 ends here, BC-x2 starts here
    y_span_m: tuple[float, float]  # plate |y| range
    seat_z_m: tuple[float, float]  # saddle (top edge) z at each end of the span
    bottom_z_m: float
    boom: Hole  # (|y|, z, hole diameter)


# ------------------------------------------------------------------------------------- motor plates ---
def motor_plate_diameter_m(motor_diameter_m: float) -> float:
    """Room for the motor's base and for the outer end of the bolt slots, each with a MIN_WEB_M rim."""
    slots = max(MOTOR_BOLT_CIRCLES_M) + MOTOR_SLOT_WIDTH_M + 2.0 * MIN_WEB_M
    return max(motor_diameter_m + 2.0 * MIN_WEB_M, slots)


def _slot(angle_deg: float) -> bd.Face:
    """Stadium from the smallest to the largest bolt circle along `angle_deg`."""
    r0, r1 = min(MOTOR_BOLT_CIRCLES_M) / 2.0, max(MOTOR_BOLT_CIRCLES_M) / 2.0
    half = MOTOR_SLOT_WIDTH_M / 2.0
    ends = [circle_face(Hole(r, 0.0, MOTOR_SLOT_WIDTH_M)) for r in (r0, r1)]
    stadium = largest_face(rect_face(r0, -half, r1, half).fuse(*ends).clean())
    return stadium.rotate(bd.Axis.Z, angle_deg)


def motor_plate_outline(diameter_m: float) -> bd.Face:
    """Disc centred on the local origin: centre mark hole plus a slotted cross for every bolt circle."""
    disc = circle_face(Hole(0.0, 0.0, diameter_m))
    cutters = [circle_face(Hole(0.0, 0.0, MOTOR_CENTRE_HOLE_M)), *(_slot(a) for a in MOTOR_SLOT_ANGLES_DEG)]
    return largest_face(disc.cut(*cutters))


def _lift_motor_diameter_m(components: Sequence[PlacedComponent], index: int) -> float:
    motor = next((c for c in components if c.name == f"{LIFT_MOTOR_PREFIX}{index}"), None)
    return 0.0 if motor is None else motor.size_m[0]


def _motor_plates(layout: AircraftLayout, boom_z: dict[int, float], tube_od_m: float,
                  components: Sequence[PlacedComponent], settings: FabricationSettings) -> list[KitPart]:
    """`MP-k` on top of its boom under rotor hub k (hub z is the motor's underside)."""
    t, plates = settings.sheet_for("plate").thickness_m, []
    for k, (x, y, z_hub) in enumerate(layout.rotor_positions, start=1):
        side = RIGHT if y > 0 else LEFT
        if side not in boom_z:
            raise KitUnavailable(f"MP-{k}: rotor hub at y = {y:.3f} m has no lift boom under it")
        top = boom_z[side] + tube_od_m / 2.0
        if top + t > z_hub:
            raise KitUnavailable(f"MP-{k}: the motor plate on the lowered boom would reach into the lift motor")
        outline = motor_plate_outline(motor_plate_diameter_m(_lift_motor_diameter_m(components, k - 1)))
        plates.append(KitPart(f"MP-{k}", "plate", outline, bd.Location((x, y, top))))
    return plates


# ------------------------------------------------------------------------------------------ clamps ---
def _rib_bands(spar: SparPlan, layout: AircraftLayout, settings: FabricationSettings,
               prefix: str) -> list[tuple[str, float, float]]:
    """(label, |y| from, |y| to) of each rib slab of one panel: centred on its station, the root doubler outboard."""
    t_rib, t_root = settings.sheet_for("rib").thickness_m, settings.sheet_for("doubler").thickness_m
    bands = []
    for i, y in enumerate(wing_rib_stations(spar.joint_y_m, layout.wing.semi_span_m)):
        lo, hi = (y, y + t_root) if i == 0 else (y - t_rib / 2.0, y + t_rib / 2.0)
        bands.append((f"{prefix}{i + 1:02d}", lo, hi))
    return bands


def _clear_of_ribs(label: str, y_boom: float, span: tuple[float, float], tube_od_m: float,
                   spar: SparPlan, layout: AircraftLayout, settings: FabricationSettings) -> tuple[float, float]:
    """Plate span trimmed one rib-sheet thickness clear of every rib slab it would reach."""
    gap, (lo, hi) = settings.sheet_for("rib").thickness_m, span
    prefix = "WR-" if label.startswith("BC-R") else "WL-"
    for rib, r0, r1 in _rib_bands(spar, layout, settings, prefix):
        if r0 < y_boom + tube_od_m / 2.0 and r1 > y_boom - tube_od_m / 2.0:
            raise KitUnavailable(f"the lift boom at {label} runs through rib {rib}")
        if r0 >= y_boom:
            hi = min(hi, r0 - gap)
        else:
            lo = max(lo, r1 + gap)
    return lo, hi


def _seat_z(layout: AircraftLayout, spar: SparPlan, settings: FabricationSettings,
            span: tuple[float, float], y_boom: float) -> tuple[float, float]:
    """Spar member underside at both ends of the span: the largest member over it, (OD + clearance) / 2 below the
    axis, stretched by the axis' slope (dihedral and camber growth) so it clears the member everywhere."""
    radius = max(spar_hole_m(spar, y, settings) for y in (*span, y_boom)) / 2.0
    z0, z1 = (socket_axis_m(layout, y)[1] for y in span)
    stretch = math.hypot(1.0, (z1 - z0) / (span[1] - span[0]))
    return z0 - radius * stretch, z1 - radius * stretch


def _clamp(boom: Boom, layout: AircraftLayout, spar: SparPlan, settings: FabricationSettings) -> _Clamp:
    side, y_boom = (RIGHT if boom.y_m > 0 else LEFT), abs(boom.y_m)
    label = f"BC-{'R' if side == RIGHT else 'L'}1"
    od = boom_tube(boom.diameter_m, settings).od_m
    hole_d = od + settings.fit_clearance_m
    reach = hole_d / 2.0 + CLAMP_CHEEK_M
    span = _clear_of_ribs(label, y_boom, (y_boom - reach, y_boom + reach), od, spar, layout, settings)
    if min(y_boom - span[0], span[1] - y_boom) - hole_d / 2.0 < MIN_WEB_M - WEB_EPS_M:
        raise KitUnavailable(f"{label}: a wing rib leaves less than {MIN_WEB_M * 1e3:.0f} mm beside the boom hole")
    seat = _seat_z(layout, spar, settings, span, y_boom)
    under_hub = boom.z_m + boom.diameter_m / 2.0 - settings.sheet_for("plate").thickness_m - od / 2.0
    boom_z = min(under_hub, min(seat) - MIN_WEB_M - hole_d / 2.0)
    return _Clamp(side, socket_axis_m(layout, y_boom)[0], span, seat,
                  boom_z - hole_d / 2.0 - CLAMP_CHEEK_M, Hole(y_boom, boom_z, hole_d))


def _clamp_outline(clamp: _Clamp, label: str) -> bd.Face:
    """Local X = aircraft y (signed), local Y = z: bottom edge, saddle top edge, boom hole."""
    s = clamp.side
    (y0, z0), (y1, z1) = sorted(zip((s * y for y in clamp.y_span_m), clamp.seat_z_m))
    corners = [(y0, clamp.bottom_z_m, 0.0), (y1, clamp.bottom_z_m, 0.0), (y1, z1, 0.0), (y0, z0, 0.0)]
    plain = bd.Face(bd.Wire.make_polygon(corners, close=True))
    hole = Hole(s * clamp.boom.x_m, clamp.boom.y_m, clamp.boom.diameter_m)
    if not keeps_web(plain, hole, []):
        raise KitUnavailable(f"{label}: no {MIN_WEB_M * 1e3:.0f} mm web round the boom hole")
    return drill(plain, (hole,), label)


def _clamp_plates(clamp: _Clamp, settings: FabricationSettings) -> list[KitPart]:
    t, name = settings.sheet_for("clamp").thickness_m, "R" if clamp.side == RIGHT else "L"
    outline = _clamp_outline(clamp, f"BC-{name}1")
    return [KitPart(f"BC-{name}1", "clamp", outline, station_location(clamp.x_m - t)),
            KitPart(f"BC-{name}2", "clamp", outline, station_location(clamp.x_m))]


# ------------------------------------------------------------------------------------------- tubes ---
def _tube(label: str, role: TubeRole, boom: Boom, z_m: float, settings: FabricationSettings) -> TubePart:
    return TubePart(label, role, boom_tube(boom.diameter_m, settings), boom.length_m,
                    (boom.x_start_m, boom.y_m, z_m), X_AXIS)


def _tail_tubes(layout: AircraftLayout, settings: FabricationSettings) -> list[TubePart]:
    boom: Optional[Boom] = layout.tail_boom
    return [] if boom is None else [_tube("TB-1", "tail_boom", boom, boom.z_m, settings)]


def boom_parts(
    layout: AircraftLayout, spar: SparPlan, settings: FabricationSettings,
    components: Sequence[PlacedComponent] = (),
) -> tuple[tuple[KitPart, ...], tuple[TubePart, ...]]:
    """Clamp plates `BC-…`, motor plates `MP-1…4` and tubes (`TB-1`, `LB-R`, `LB-L`).

    Fixed-wing layouts get only the tail-boom tube. `components` (optional) supplies the lift motors
    (`vtol_motor_k`), whose diameter sizes the motor plates.
    """
    tubes, parts = _tail_tubes(layout, settings), []
    boom_z: dict[int, float] = {}
    for boom in layout.vtol_booms:
        clamp = _clamp(boom, layout, spar, settings)
        boom_z[clamp.side] = clamp.boom.y_m
        parts += _clamp_plates(clamp, settings)
        tubes.append(_tube(f"LB-{'R' if clamp.side == RIGHT else 'L'}", "lift_boom", boom, clamp.boom.y_m, settings))
    if layout.vtol_booms:
        od = boom_tube(layout.vtol_booms[0].diameter_m, settings).od_m
        parts += _motor_plates(layout, boom_z, od, components, settings)
    return tuple(parts), tuple(tubes)
