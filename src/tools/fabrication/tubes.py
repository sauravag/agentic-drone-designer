"""Stock-tube spar and socket sizing for the laser-cut airframe kit.

Picks carbon tubes from stock so the spar survives the larger of the flight (elliptic lift)
and hover (boom reaction) bending moments, stepping down telescopically where the wing thins.
All values SI. Must not import from the schemas' consumers.
"""

import math
from dataclasses import dataclass
from typing import Callable, Final, Optional

from src.schemas.fabrication import FabricationSettings, TubeStock
from src.tools.aero_cad.layout import MIN_BOOM_DIAMETER_M
from src.tools.aero_cad.structure import RIB_PITCH_M
from src.units import G0

MIN_WEB_M: Final[float] = 0.002  # wood left above and below the spar tube
FIT_EPS_M: Final[float] = 1e-9  # float tolerance on diameter comparisons
HALF_WING_FRACTION: Final[float] = 0.5  # one wing carries half the weight
ELLIPTIC_MOMENT_FACTOR: Final[float] = 2.0 / (3.0 * math.pi)  # prefactor of the exact elliptic moment
ELLIPTIC_SQUARE_OFFSET: Final[float] = 2.0
ELLIPTIC_ACOS_FACTOR: Final[float] = 3.0
BISECTION_ITERATIONS: Final[int] = 50
HOVER_ARMS: Final[int] = 2  # two booms' thrust reacts through the spar per side
HOVER_SHARE: Final[float] = 0.25  # each motor carries a quarter of the weight
ROOT_SPAN_M: Final[float] = 0.0


class KitUnavailable(Exception):
    """The airframe cannot be built from stock; the message is user-facing."""


@dataclass(frozen=True)
class SparSegment:
    tube: TubeStock
    y_start_m: float
    y_end_m: float


@dataclass(frozen=True)
class SparPlan:
    segments: tuple[SparSegment, ...]
    socket: TubeStock
    joint_y_m: float
    insertion_m: float


def elliptic_moment_nm(mtow_kg: float, n: float, semi_span_m: float, y_m: float) -> float:
    """Bending moment at span station y (from the centreline) for an elliptic lift distribution.

    M = (2 P b / 3 pi) [(2 + xi^2) sqrt(1 - xi^2) - 3 xi acos(xi)], P = n W / 2, xi = y / b.
    """
    xi = min(max(y_m / semi_span_m, 0.0), 1.0)
    half_lift = n * mtow_kg * G0 * HALF_WING_FRACTION
    shape = (ELLIPTIC_SQUARE_OFFSET + xi**2) * math.sqrt(1.0 - xi**2) - ELLIPTIC_ACOS_FACTOR * xi * math.acos(xi)
    return ELLIPTIC_MOMENT_FACTOR * half_lift * semi_span_m * shape


def hover_moment_nm(mtow_kg: float, arm_m: float) -> float:
    """Moment from the motors' thrust on one side acting at the given arm."""
    return HOVER_ARMS * (mtow_kg * G0 * HOVER_SHARE) * max(arm_m, 0.0)


def bending_stress_pa(tube: TubeStock, moment_nm: float) -> float:
    """Outer-fibre bending stress of a tube under a moment."""
    return moment_nm * (tube.od_m / 2.0) / tube.second_moment_m4


def joint_y_m(fuselage_radius_m: float, wing_z_m: float) -> float:
    """Span station where the wing spar meets the fuselage skin."""
    return math.sqrt(max(fuselage_radius_m**2 - wing_z_m**2, 0.0))


def boom_tube(diameter_m: float, settings: FabricationSettings) -> TubeStock:
    """Stock tube for a boom of layout `diameter_m`: the largest tube of at least MIN_BOOM_DIAMETER_M that is no
    wider than the layout boom, or, when the layout boom is thinner than that floor, the smallest floor tube."""
    floor_m = MIN_BOOM_DIAMETER_M - FIT_EPS_M
    within = [t for t in settings.tubes if floor_m <= t.od_m <= diameter_m + FIT_EPS_M]
    if within:
        return within[-1]  # tubes are sorted by OD
    floor = next((t for t in settings.tubes if t.od_m >= floor_m), None)
    if floor is None:
        raise KitUnavailable(f"no stock tube for a {diameter_m * 1e3:.0f} mm boom")
    return floor


def lift_boom_tube(thrust_n: float, arm_m: float, max_diameter_m: float, settings: FabricationSettings) -> TubeStock:
    """R21: the smallest stock tube (lighter wall first at equal OD) whose bending stress under one lift motor's
    thrust at `arm_m` from the clamp is within sigma_allow / safety_factor, no wider than the layout boom (or the
    MIN_BOOM_DIAMETER_M floor when the layout boom is thinner than that)."""
    floor_m, moment = MIN_BOOM_DIAMETER_M - FIT_EPS_M, thrust_n * max(arm_m, 0.0)
    cap_m = max(max_diameter_m, MIN_BOOM_DIAMETER_M) + FIT_EPS_M
    fits = sorted((t for t in settings.tubes if floor_m <= t.od_m <= cap_m), key=lambda t: (t.od_m, t.area_m2))
    tube = next((t for t in fits if _passes(t, moment, settings)), None)
    if tube is None:
        raise KitUnavailable(
            f"no stock lift boom up to {cap_m * 1e3:.0f} mm carries {thrust_n:.0f} N at a {arm_m * 1e3:.0f} mm arm"
        )
    return tube


def usable_depth_m(local_thickness_m: float, settings: FabricationSettings) -> float:
    """Room for a spar tube once the top and bottom webs and the hole's fit clearance are reserved."""
    return local_thickness_m - 2.0 * MIN_WEB_M - settings.fit_clearance_m


def _allowable_pa(settings: FabricationSettings) -> float:
    return settings.sigma_allow_pa / settings.safety_factor


def _moment_at(
    mtow_kg: float, semi_span_m: float, y_m: float, boom_y_m: Optional[float], settings: FabricationSettings
) -> float:
    """Governing moment at station y: flight (elliptic) vs hover (boom reaction)."""
    flight = elliptic_moment_nm(mtow_kg, settings.n_limit, semi_span_m, y_m)
    hover = 0.0 if boom_y_m is None else hover_moment_nm(mtow_kg, boom_y_m - y_m)
    return max(flight, hover)


def _passes(tube: TubeStock, moment_nm: float, settings: FabricationSettings) -> bool:
    return bending_stress_pa(tube, moment_nm) <= _allowable_pa(settings)


def _unavailable_at(y_m: float) -> KitUnavailable:
    return KitUnavailable(f"no stock tube fits the wing at y = {y_m:.3f} m")


def _stations(joint_m: float, semi_span_m: float) -> list[float]:
    """Joint, every rib pitch outboard strictly inside the span, then the tip."""
    count = int(math.floor((semi_span_m - joint_m) / RIB_PITCH_M))
    inner = [joint_m + k * RIB_PITCH_M for k in range(1, count + 1)]
    return [joint_m] + [y for y in inner if y < semi_span_m - FIT_EPS_M] + [semi_span_m]


def _step_down(
    current: TubeStock, depth_m: float, moment_nm: float, settings: FabricationSettings
) -> Optional[TubeStock]:
    """Largest stock tube or rod that slides inside `current`, fits the depth and carries the moment."""
    rods = tuple(TubeStock(od_m=r, id_m=0.0) for r in settings.rods_m)
    for tube in sorted(settings.tubes + rods, key=lambda t: (t.od_m, t.id_m), reverse=True):
        nests = tube.od_m <= current.id_m + FIT_EPS_M and current.id_m - tube.od_m <= settings.fit_clearance_m + FIT_EPS_M
        if nests and tube.od_m <= depth_m + FIT_EPS_M and _passes(tube, moment_nm, settings):
            return tube
    return None


def _pick_socket(root: TubeStock, moment_nm: float, settings: FabricationSettings) -> TubeStock:
    for tube in settings.tubes:
        sleeves = tube.id_m >= root.od_m - FIT_EPS_M and tube.id_m - root.od_m <= settings.fit_clearance_m + FIT_EPS_M
        if sleeves and _passes(tube, moment_nm, settings):
            return tube
    raise KitUnavailable(
        f"no stock tube can sleeve the {root.od_m * 1e3:.0f} mm spar tube as a fuselage socket"
    )


def _check_insertion(insertion_m: float, root: TubeStock, settings: FabricationSettings) -> None:
    """The spar root must reach far enough into the socket (skin to centreline) to carry its moment."""
    needed = settings.min_insertion_diameters * root.od_m
    if insertion_m < needed - FIT_EPS_M:
        raise KitUnavailable(
            f"spar insertion {insertion_m * 1e3:.0f} mm (fuselage skin to centreline) is shorter than the "
            f"{needed * 1e3:.0f} mm needed for the {root.od_m * 1e3:.0f} mm spar tube"
        )


def depth_limit_m(depth_at: Callable[[float], float], od_m: float, inboard_m: float, outboard_m: float) -> float:
    """Outermost y in [inboard, outboard) where the depth still admits `od_m` (depth is non-increasing)."""
    lo, hi = inboard_m, outboard_m
    for _ in range(BISECTION_ITERATIONS):
        mid = 0.5 * (lo + hi)
        if depth_at(mid) >= od_m - FIT_EPS_M:
            lo = mid
        else:
            hi = mid
    return lo


def plan_spar(
    mtow_kg: float,
    semi_span_m: float,
    joint_y_m: float,
    depth_at: Callable[[float], float],
    boom_y_m: Optional[float],
    settings: FabricationSettings,
) -> SparPlan:
    """Size a telescoping tube spar from the fuselage joint to the tip, plus its socket."""
    if boom_y_m is not None and boom_y_m < 0.0:
        raise ValueError(f"boom_y_m must be a distance from the centreline, got {boom_y_m}")
    root_moment = _moment_at(mtow_kg, semi_span_m, joint_y_m, boom_y_m, settings)
    root_depth = depth_at(joint_y_m)
    root = next(
        (t for t in settings.tubes if t.od_m <= root_depth + FIT_EPS_M and _passes(t, root_moment, settings)),
        None,
    )
    if root is None:
        raise _unavailable_at(joint_y_m)

    segments: list[SparSegment] = []
    current, start = root, joint_y_m
    for y in _stations(joint_y_m, semi_span_m)[1:]:
        while depth_at(y) < current.od_m - FIT_EPS_M:
            y_step = depth_limit_m(depth_at, current.od_m, start, y)
            if y_step <= start + FIT_EPS_M:
                raise _unavailable_at(y)
            moment = _moment_at(mtow_kg, semi_span_m, y_step, boom_y_m, settings)
            smaller = _step_down(current, depth_at(y_step), moment, settings)
            if smaller is None:
                raise _unavailable_at(y)
            segments.append(SparSegment(current, start, y_step))
            current, start = smaller, y_step
    segments.append(SparSegment(current, start, semi_span_m))

    centre_moment = _moment_at(mtow_kg, semi_span_m, 0.0, boom_y_m, settings)
    socket = _pick_socket(root, centre_moment, settings)
    _check_insertion(joint_y_m, root, settings)
    # The socket spans the fuselage only; each spar root plugs in from the skin to the centreline.
    return SparPlan(tuple(segments), socket, joint_y_m, joint_y_m)
