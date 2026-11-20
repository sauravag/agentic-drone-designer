"""Wing-box plate bosses (R27/R31/R32/R33): the plate-stock doublers glued on a wing-box plate's inner face where
the plate alone cannot carry its loads (SI, plate-local x, z).

- Socket bosses `WM-RD` / `WM-LD`: when the plate cannot bear its wing's shear on the socket (R27: sigma_b =
  F / (OD t) <= ply_bearing_allow_pa / FS, a boss doubling t) or its web under the socket shears out under the
  downward load (R32/R33: F / (2 w t) <= ply_shear_allow_pa / FS), an annulus SOCKET_BOSS_WIDTH_M wide plus a skirt
  below the hole, grown in SOCKET_BOSS_DEPTH_STEP_M steps until the web passes (at most SOCKET_BOSS_MAX_DEPTH_M).
  The skirt must stay above the crutch.
- Pin bosses `WM-RPD` / `WM-LPD` (R31): an annulus round the pin hole clipped to the plate's real material.

`fuselage_parts` places them; this module sizes and outlines them.
"""

import math
from typing import Final, Optional

import build123d as bd

from src.schemas.fabrication import FabricationSettings
from src.tools.fabrication.rib_outline import Hole, circle_face, common, rect_face, single_face
from src.tools.fabrication.tubes import (
    BOSS_THICKNESS_FACTOR,
    FIT_EPS_M,
    MIN_WEB_M,
    KitUnavailable,
    PlateLoads,
    SparPlan,
)
from src.tools.fabrication.wing_parts import PIN_BOSS_WIDTH_M

SOCKET_BOSS_WIDTH_M: Final[float] = 0.006  # R27 boss annulus radial width (stays inside the socket-station formers)
SOCKET_BOSS_DEPTH_STEP_M: Final[float] = 0.002  # R33: the boss's material below the socket grows in these steps
SOCKET_BOSS_MAX_DEPTH_M: Final[float] = 0.030  # R33: deepest boss skirt below the socket hole
BOSS_LABELS: Final[dict[int, str]] = {1: "WM-RD", -1: "WM-LD"}  # socket bosses
PIN_BOSS_LABELS: Final[tuple[str, str]] = ("WM-RPD", "WM-LPD")  # plate-side pin bosses (right, left)


def _stress_pa(load_n: float, area_m2: float) -> float:
    return load_n / area_m2


def _web_shear_pa(load_n: float, boss_depth_m: float, thickness_m: float) -> float:
    """R32 shear-out under the socket: two planes through the MIN_WEB_M plate web plus `boss_depth_m` of boss below
    the hole, each `thickness_m` thick (plate and boss are the same stock)."""
    return _stress_pa(load_n, 2.0 * (MIN_WEB_M + boss_depth_m) * thickness_m)


def socket_boss_depth_m(spar: SparPlan, settings: FabricationSettings, loads: PlateLoads) -> Optional[float]:
    """Boss material below the socket hole, or None when the plain plate suffices. Checks R27 bearing
    F_shear / (OD t) <= ply_bearing_allow_pa / FS (a boss doubles t) and R32/R33 web shear-out F_down / (2 w t) <=
    ply_shear_allow_pa / FS. A boss starts SOCKET_BOSS_WIDTH_M deep below the hole and grows by
    SOCKET_BOSS_DEPTH_STEP_M until the web passes; raises `KitUnavailable` beyond SOCKET_BOSS_MAX_DEPTH_M, or when
    even a boss cannot bear the socket."""
    t, od = settings.sheet_for("plate").thickness_m, spar.socket.od_m
    bearing_allow = settings.ply_bearing_allow_pa / settings.safety_factor
    shear_allow = settings.ply_shear_allow_pa / settings.safety_factor
    if _stress_pa(loads.shear_n, od * t) <= bearing_allow and _web_shear_pa(loads.down_n, 0.0, t) <= shear_allow:
        return None
    bossed_bearing = _stress_pa(loads.shear_n, od * BOSS_THICKNESS_FACTOR * t)
    if bossed_bearing > bearing_allow:
        raise KitUnavailable(f"socket bearing: {bossed_bearing / 1e6:.1f} MPa on the wing-box plate even with a boss "
                             f"doubler (allowable {bearing_allow / 1e6:.1f} MPa)")
    steps = math.ceil((SOCKET_BOSS_MAX_DEPTH_M - SOCKET_BOSS_WIDTH_M) / SOCKET_BOSS_DEPTH_STEP_M - FIT_EPS_M)
    for k in range(steps + 1):
        depth = min(SOCKET_BOSS_WIDTH_M + k * SOCKET_BOSS_DEPTH_STEP_M, SOCKET_BOSS_MAX_DEPTH_M)
        if _web_shear_pa(loads.down_n, depth, t) <= shear_allow:
            return depth
    worst = _web_shear_pa(loads.down_n, SOCKET_BOSS_MAX_DEPTH_M, t)
    raise KitUnavailable(
        f"socket web shear: {loads.down_n:.0f} N down on the socket shears out the web under it even with a "
        f"{SOCKET_BOSS_MAX_DEPTH_M * 1e3:.0f} mm boss ({worst / 1e6:.1f} MPa > {shear_allow / 1e6:.2f} MPa)"
    )


def socket_boss_outline(hole: Hole, depth_m: float) -> bd.Face:
    """The socket boss before drilling: an annulus SOCKET_BOSS_WIDTH_M wide round the hole, plus a skirt as wide as
    the annulus reaching `depth_m` below the hole (R33; it grows only downward, clear of the socket formers)."""
    radius = hole.diameter_m / 2.0 + SOCKET_BOSS_WIDTH_M
    ring = circle_face(Hole(hole.x_m, hole.y_m, 2.0 * radius))
    skirt = rect_face(hole.x_m - radius, hole.y_m - hole.diameter_m / 2.0 - depth_m, hole.x_m + radius, hole.y_m)
    return single_face(ring.fuse(skirt).clean(), BOSS_LABELS[1])


def check_skirt_clears_crutch(socket: Hole, depth_m: float, crutch_t_m: float) -> None:
    """The socket boss's skirt (its bottom `depth_m` below the hole) must stay above the crutch's top face."""
    if socket.y_m - socket.diameter_m / 2.0 - depth_m <= crutch_t_m / 2.0:
        raise KitUnavailable("socket web shear: the socket boss would reach down to the crutch")


def pin_boss_blank(plate: bd.Face, plate_pin: Hole, boss_pin: Hole, label: str) -> bd.Face:
    """R31 plate pin boss before drilling: an annulus PIN_BOSS_WIDTH_M wide round `boss_pin` (where the oblique pin
    line crosses the boss slab, the hole the boss is drilled at), clipped to the plate's real material (its outline
    less tab notches, openings and other holes), so it never hangs over a notch, opening or edge. The plate's own
    pin hole (`plate_pin`, at the plate mid-plane) and the boss's hole count as material: the boss is drilled
    afterwards, so neither leaves a sliver void."""
    ring = circle_face(Hole(boss_pin.x_m, boss_pin.y_m, boss_pin.diameter_m + 2.0 * PIN_BOSS_WIDTH_M))
    material = plate.fuse(circle_face(plate_pin), circle_face(boss_pin)).clean()
    return single_face(common(ring, material), label)
