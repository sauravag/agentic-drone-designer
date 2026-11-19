"""Stock-tube spar and socket sizing for the laser-cut airframe kit.

Picks carbon tubes from stock so the spar survives the larger of the flight (elliptic lift)
and hover (boom reaction) bending moments, stepping down telescopically where the wing thins.
All values SI. Must not import from the schemas' consumers.
"""

import math
from dataclasses import dataclass
from typing import Callable, Final, Optional

from src.schemas.fabrication import FabricationSettings, TubeStock
from src.tools.aero_cad.structure import RIB_PITCH_M
from src.units import G0

MIN_WEB_M: Final[float] = 0.006  # wood left above and below the spar tube
FIT_EPS_M: Final[float] = 1e-9  # float tolerance on diameter comparisons
HALF_WING_FRACTION: Final[float] = 0.5  # one wing carries half the weight
ELLIPTIC_CENTROID_FACTOR: Final[float] = 4.0 / (3.0 * math.pi)  # centroid of an elliptic load
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


def wing_root_moment_nm(mtow_kg: float, exposed_semi_span_m: float, n_limit: float) -> float:
    """Root bending moment of one wing under an elliptic load at the limit load factor."""
    half_lift = n_limit * mtow_kg * G0 * HALF_WING_FRACTION
    return half_lift * ELLIPTIC_CENTROID_FACTOR * exposed_semi_span_m


def hover_moment_nm(mtow_kg: float, arm_m: float) -> float:
    """Moment from the motors' thrust on one side acting at the given arm."""
    return HOVER_ARMS * (mtow_kg * G0 * HOVER_SHARE) * max(arm_m, 0.0)


def bending_stress_pa(tube: TubeStock, moment_nm: float) -> float:
    """Outer-fibre bending stress of a tube under a moment."""
    return moment_nm * (tube.od_m / 2.0) / tube.second_moment_m4


def joint_y_m(fuselage_radius_m: float, wing_z_m: float) -> float:
    """Span station where the wing spar meets the fuselage skin."""
    return math.sqrt(max(fuselage_radius_m**2 - wing_z_m**2, 0.0))


def usable_depth_m(local_thickness_m: float, settings: FabricationSettings) -> float:
    """Room for a spar tube once the top and bottom webs are reserved."""
    return local_thickness_m - 2.0 * MIN_WEB_M


def _allowable_pa(settings: FabricationSettings) -> float:
    return settings.sigma_allow_pa / settings.safety_factor


def _moment_at(
    mtow_kg: float, semi_span_m: float, y_m: float, boom_y_m: Optional[float], settings: FabricationSettings
) -> float:
    """Governing moment at station y: flight on the remaining span vs hover from the boom."""
    flight = wing_root_moment_nm(mtow_kg, semi_span_m - y_m, settings.n_limit)
    hover = 0.0 if boom_y_m is None else hover_moment_nm(mtow_kg, boom_y_m - y_m)
    return max(flight, hover)


def _passes(tube: TubeStock, moment_nm: float, settings: FabricationSettings) -> bool:
    return bending_stress_pa(tube, moment_nm) <= _allowable_pa(settings)


def _unavailable_at(y_m: float) -> KitUnavailable:
    return KitUnavailable(f"no stock tube fits the wing at y = {y_m:.3f} m")


def _stations(joint_m: float, semi_span_m: float) -> list[float]:
    """Joint, then every rib pitch outboard, strictly inside the span."""
    count = int(math.floor((semi_span_m - joint_m) / RIB_PITCH_M))
    return [joint_m + k * RIB_PITCH_M for k in range(1, count + 1) if joint_m + k * RIB_PITCH_M < semi_span_m - FIT_EPS_M]


def _step_down(
    current: TubeStock, depth_m: float, moment_nm: float, settings: FabricationSettings
) -> Optional[TubeStock]:
    """Largest stock tube that slides inside `current`, fits the depth and carries the moment."""
    for tube in reversed(settings.tubes):
        nests = tube.od_m <= current.id_m + FIT_EPS_M and current.id_m - tube.od_m <= settings.fit_clearance_m + FIT_EPS_M
        if nests and tube.od_m <= depth_m + FIT_EPS_M and _passes(tube, moment_nm, settings):
            return tube
    return None


def _pick_socket(
    root: TubeStock, moment_nm: float, settings: FabricationSettings
) -> TubeStock:
    for tube in settings.tubes:
        sleeves = tube.id_m >= root.od_m - FIT_EPS_M and tube.id_m - root.od_m <= settings.fit_clearance_m + FIT_EPS_M
        if sleeves and _passes(tube, moment_nm, settings):
            return tube
    raise KitUnavailable(
        f"no stock tube can sleeve the {root.od_m * 1e3:.0f} mm spar tube as a fuselage socket"
    )


def plan_spar(
    mtow_kg: float,
    semi_span_m: float,
    joint_y_m: float,
    depth_at: Callable[[float], float],
    boom_y_m: Optional[float],
    settings: FabricationSettings,
) -> SparPlan:
    """Size a telescoping tube spar from the fuselage joint to the tip, plus its socket."""
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
    for y in _stations(joint_y_m, semi_span_m):
        depth = depth_at(y)
        if depth >= current.od_m - FIT_EPS_M:
            continue
        moment = _moment_at(mtow_kg, semi_span_m, y, boom_y_m, settings)
        smaller = _step_down(current, depth, moment, settings)
        if smaller is None:
            raise _unavailable_at(y)
        segments.append(SparSegment(current, start, y))
        current, start = smaller, y
    segments.append(SparSegment(current, start, semi_span_m))

    centre_moment = _moment_at(mtow_kg, semi_span_m, ROOT_SPAN_M, boom_y_m, settings)
    socket = _pick_socket(root, centre_moment, settings)
    return SparPlan(tuple(segments), socket, joint_y_m, settings.insertion_fraction * semi_span_m)
