"""Wing stiffness: tip deflection at 1 g (R44), aileron reversal (R43) and the balsa D-box torsion fallback (R45).

Euler-Bernoulli bending and St-Venant/Bredt torsion on a wing clamped at the spar joint; the section is stepped by
spar segment (the inboard member at a step). Both integrals use the midpoint rule over STIFFNESS_INTEGRATION_STEPS
strips with cumulative sums. Only the carbon tube carries bending; the D-box adds torsion only.

Reversal is strip theory with the spar on the aerodynamic centre (SPAR_CHORD_STATION == QUARTER_CHORD), so the twist
comes only from the aileron's C_m,delta and is linear in q: rolling effectiveness = q A - q^2 B and q_R = A / B.

Limitations (not modelled):
- V_R is an equivalent airspeed (sea-level density) compared with 1.15 x V_NE built from the cruise speed, which is a
  true airspeed at altitude; the comparison is therefore conservative at altitude.
- The D-cell's shear centre lies a little ahead of its web, so with the D-box the elastic axis sits slightly forward
  of the aerodynamic centre: a small divergence tendency that is not modelled.
- Flutter and gust loads are not modelled.
"""
import math
from dataclasses import dataclass, replace
from typing import Callable, Final

import numpy as np

from src.schemas.fabrication import FabricationSettings, TubeStock
from src.tools.aero_cad.airfoil_geometry import AirfoilShape
from src.tools.aero_cad.layout import AILERON_CHORD_FRACTION, AILERON_ETA
from src.tools.aero_cad.structure import SPAR_CHORD_STATION
from src.tools.fabrication.tubes import (
    DYNAMIC_PRESSURE_FACTOR, MIN_WEB_M, SEA_LEVEL_DENSITY_KG_M3, KitUnavailable, SparPlan, elliptic_moment_nm,
)
from src.tools.sizing.aero_model import VNE_TO_CRUISE

REVERSAL_MARGIN: Final[float] = 1.15  # R43: reversal speed must exceed this x V_NE
TIP_DEFLECTION_FRACTION: Final[float] = 0.05  # R44: tip deflection limit, fraction of the semi-span
TIP_DEFLECTION_LOAD_FACTOR: Final[float] = 1.0  # R44: evaluated at 1 g
DBOX_HOLE_ALLOWANCE_M: Final[float] = 0.0005  # oblique-hole growth, so the rib keeps MIN_WEB_M round the real hole
STIFFNESS_INTEGRATION_STEPS: Final[int] = 400  # midpoint rule, joint to tip
HALF: Final[float] = 0.5
BREDT_FACTOR: Final[float] = 4.0  # GJ = 4 A_m^2 G t / p
HELMBOLD_SQUARE_OFFSET: Final[float] = 4.0  # a = 2 pi AR / (2 + sqrt(AR^2 + 4))
HELMBOLD_DENOMINATOR_OFFSET: Final[float] = 2.0
HINGE_COS_OFFSET: Final[float] = 1.0  # theta_h = acos(2E - 1)
HINGE_COS_FACTOR: Final[float] = 2.0


@dataclass(frozen=True)
class DCell:
    """Unit-chord D-cell ahead of a web at x/c = web_fraction."""
    area_ratio: float  # enclosed area / c^2 (outer contour, closed by the web)
    arc_ratio: float  # contour length web(upper) -> nose -> web(lower), / c
    web_ratio: float  # web height / c


def d_cell(airfoil: AirfoilShape, web_fraction: float) -> DCell:
    """D-cell of the airfoil ahead of the web (Selig order: upper TE -> LE, then lower LE -> TE)."""
    if not 0.0 < web_fraction < 1.0:
        raise ValueError(f"web_fraction must lie strictly between 0 and 1, got {web_fraction}")
    pts = np.asarray(airfoil.coordinates, dtype=float)
    nose = int(np.argmin(pts[:, 0]))
    upper = pts[: nose + 1][::-1]  # LE -> TE
    lower = pts[nose:]
    z_up = float(np.interp(web_fraction, upper[:, 0], upper[:, 1]))
    z_lo = float(np.interp(web_fraction, lower[:, 0], lower[:, 1]))
    fore_up = upper[1:][upper[1:, 0] < web_fraction][::-1]  # toward the nose
    fore_lo = lower[1:][lower[1:, 0] < web_fraction]
    contour = np.vstack([[web_fraction, z_up], fore_up, upper[:1], fore_lo, [[web_fraction, z_lo]]])
    x, z = contour[:, 0], contour[:, 1]
    area = HALF * abs(float(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1))))
    arc = float(np.hypot(np.diff(x), np.diff(z)).sum())
    return DCell(area_ratio=area, arc_ratio=arc, web_ratio=z_up - z_lo)


def dbox_web_x_m(chord_m: float, od_m: float, settings: FabricationSettings) -> float:
    """Chordwise position of the D-box web: ahead of the spar hole, leaving MIN_WEB_M of rib round it."""
    hole_m = od_m + settings.fit_clearance_m
    return SPAR_CHORD_STATION * chord_m - (HALF * hole_m + MIN_WEB_M + DBOX_HOLE_ALLOWANCE_M)


def dbox_gj_nm2(cell: DCell, chord_m: float, settings: FabricationSettings) -> float:
    """Bredt torsional stiffness of the single-cell D-box, midline area."""
    t = settings.dbox_sheet_thickness_m
    perimeter = (cell.arc_ratio + cell.web_ratio) * chord_m
    area_mid = cell.area_ratio * chord_m**2 - HALF * t * perimeter
    return BREDT_FACTOR * area_mid**2 * settings.balsa_shear_modulus_pa * t / perimeter


def _strips(root_m: float, semi_span_m: float) -> tuple[np.ndarray, float]:
    """Midpoints of the integration strips from the joint to the tip, and their width."""
    dy = (semi_span_m - root_m) / STIFFNESS_INTEGRATION_STEPS
    return root_m + (np.arange(STIFFNESS_INTEGRATION_STEPS) + HALF) * dy, dy


def tip_deflection_m(ei_at: Callable[[float], float], moment_at: Callable[[float], float],
                     root_m: float, semi_span_m: float) -> float:
    """delta = integral_root^s M(y) (s - y) / EI(y) dy for a wing clamped at root_m."""
    y, dy = _strips(root_m, semi_span_m)
    integrand = [moment_at(v) * (semi_span_m - v) / ei_at(v) for v in y]
    return float(np.sum(integrand) * dy)


def lift_slope_per_rad(aspect_ratio: float) -> float:
    """Helmbold finite-wing lift slope."""
    root = math.sqrt(aspect_ratio**2 + HELMBOLD_SQUARE_OFFSET)
    return 2.0 * math.pi * aspect_ratio / (HELMBOLD_DENOMINATOR_OFFSET + root)


def flap_derivatives(chord_fraction: float, lift_slope: float) -> tuple[float, float]:
    """Thin-airfoil flap theory: (C_L,delta, C_m,delta about the quarter chord) per radian."""
    theta = math.acos(HINGE_COS_FACTOR * chord_fraction - HINGE_COS_OFFSET)
    cl_delta = lift_slope / math.pi * (math.pi - theta + math.sin(theta))
    cm_delta = -lift_slope / math.tau * HALF * math.sin(theta) * (1.0 - math.cos(theta))
    return cl_delta, cm_delta


def reversal_speed_ms(gj_at: Callable[[float], float], chord_at: Callable[[float], float], root_m: float,
                      semi_span_m: float, aileron_m: tuple[float, float], lift_slope: float,
                      cl_delta: float, cm_delta: float) -> float:
    """Equivalent airspeed at which aileron rolling effectiveness q A - q^2 B reaches zero (math.inf if B <= 0).

    Arms y are from the aircraft centreline; theta per unit q is the cumulative midpoint sum of T / GJ."""
    y, dy = _strips(root_m, semi_span_m)
    chord = np.array([chord_at(v) for v in y])
    gj = np.array([gj_at(v) for v in y])
    # fraction of each strip inside the aileron span (strips straddling an aileron edge count partially)
    overlap = np.clip(np.minimum(y + HALF * dy, aileron_m[1]) - np.maximum(y - HALF * dy, aileron_m[0]), 0.0, dy)
    weight = overlap / dy
    moment_strip = weight * chord**2 * cm_delta * dy
    torque = np.cumsum(moment_strip[::-1])[::-1] - HALF * moment_strip  # tip-inward, half the own strip
    twist_strip = torque / gj * dy
    twist = np.cumsum(twist_strip) - HALF * twist_strip  # midpoint: half the own strip
    a_term = float(np.sum(weight * chord * cl_delta * y * dy))
    b_term = -float(np.sum(chord * lift_slope * twist * y * dy))
    if b_term <= 0.0:
        return math.inf
    q_reversal = a_term / b_term
    return math.sqrt(q_reversal / (DYNAMIC_PRESSURE_FACTOR * SEA_LEVEL_DENSITY_KG_M3))


@dataclass(frozen=True)
class WingStiffness:
    """Result of the stiffness checks: tip deflection at 1 g (R44) and aileron reversal speed (R43), each with
    its limit, and whether the balsa D-box (R45) was counted in the torsional stiffness."""
    tip_deflection_m: float
    tip_deflection_limit_m: float
    reversal_speed_ms: float
    reversal_required_ms: float
    dbox: bool

    @property
    def deflection_ok(self) -> bool:
        return self.tip_deflection_m <= self.tip_deflection_limit_m

    @property
    def reversal_ok(self) -> bool:
        return self.reversal_speed_ms > self.reversal_required_ms

    @property
    def passes(self) -> bool:
        return self.deflection_ok and self.reversal_ok


def wing_stiffness(spar: SparPlan, airfoil: AirfoilShape, with_dbox: bool, mtow_kg: float, semi_span_m: float,
                   chord_at: Callable[[float], float], aspect_ratio: float, v_cruise_ms: float,
                   settings: FabricationSettings) -> WingStiffness:
    """Tip deflection at 1 g and aileron reversal of the wing built on `spar`, with or without the D-box."""
    segments = spar.segments

    def member_at(y_m: float) -> TubeStock:
        # inboard member at a step boundary; stations past the last segment use the last member
        return next((s.tube for s in segments if y_m <= s.y_end_m), segments[-1].tube)

    def ei_at(y_m: float) -> float:
        return settings.carbon_modulus_pa * member_at(y_m).second_moment_m4

    def gj_at(y_m: float) -> float:
        tube = member_at(y_m)
        gj = settings.carbon_shear_modulus_pa * tube.polar_moment_m4
        if not with_dbox:
            return gj
        chord = chord_at(y_m)
        cell = d_cell(airfoil, dbox_web_x_m(chord, tube.od_m, settings) / chord)
        return gj + dbox_gj_nm2(cell, chord, settings)

    def moment_at(y_m: float) -> float:
        return elliptic_moment_nm(mtow_kg, TIP_DEFLECTION_LOAD_FACTOR, semi_span_m, y_m)

    lift_slope = lift_slope_per_rad(aspect_ratio)
    cl_delta, cm_delta = flap_derivatives(AILERON_CHORD_FRACTION, lift_slope)
    aileron = (AILERON_ETA[0] * semi_span_m, AILERON_ETA[1] * semi_span_m)
    return WingStiffness(
        tip_deflection_m=tip_deflection_m(ei_at, moment_at, spar.joint_y_m, semi_span_m),
        tip_deflection_limit_m=TIP_DEFLECTION_FRACTION * semi_span_m,
        reversal_speed_ms=reversal_speed_ms(gj_at, chord_at, spar.joint_y_m, semi_span_m, aileron, lift_slope,
                                            cl_delta, cm_delta),
        reversal_required_ms=REVERSAL_MARGIN * VNE_TO_CRUISE * v_cruise_ms,
        dbox=with_dbox,
    )


def _failures(report: WingStiffness) -> str:
    """Each failing check with its value and limit (D18 style), joined with '; '."""
    found: list[str] = []
    if not report.deflection_ok:
        found.append(f"tip deflection {report.tip_deflection_m * 1e3:.0f} mm at {TIP_DEFLECTION_LOAD_FACTOR:.0f} g "
                     f"(limit {report.tip_deflection_limit_m * 1e3:.0f} mm, "
                     f"{TIP_DEFLECTION_FRACTION:.0%} of the semi-span)")
    if not report.reversal_ok:
        found.append(f"aileron reversal at {report.reversal_speed_ms:.1f} m/s "
                     f"(needs above {report.reversal_required_ms:.1f} m/s = {REVERSAL_MARGIN:.2f} x V_NE)")
    return "; ".join(found)


def stiffen_spar(spar: SparPlan, check: Callable[[SparPlan, bool], WingStiffness],
                 settings: FabricationSettings) -> SparPlan:
    """Spar with its stiffness report: the bare tube when it passes, else with the balsa D-box (R45) when that
    cures the aileron reversal. A tip deflection failure has no fallback (the D-box adds no bending stiffness)."""
    report = check(spar, False)
    if not report.deflection_ok:
        raise KitUnavailable(f"the wing spar is too flexible: {_failures(report)}")
    if report.passes:
        return replace(spar, stiffness=report, dbox=False)
    boxed = check(spar, True)
    if boxed.passes:
        return replace(spar, stiffness=boxed, dbox=True)
    raise KitUnavailable(f"the wing is too flexible even with the {settings.dbox_sheet_thickness_m * 1e3:.1f} mm "
                         f"balsa D-box: {_failures(boxed)}")
