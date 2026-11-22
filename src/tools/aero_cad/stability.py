"""Stability & control derivatives from VSPAERO vortex-lattice results (pure, deterministic math).

VSPAERO reports moments in OpenVSP geometry axes (x aft, y right, z up). Flight-dynamics body axes
(x forward, y right, z down) are a 180 deg rotation about y, so: Cl = -CMx, Cm = +CMy, Cn = -CMz.

VLM models only the lifting surfaces (thin-sheet bodies corrupt the solution), so the fuselage's
destabilising pitch and yaw contributions are added from handbook correlations:
  * Cm_alpha,fus = Kf * Wf^2 * Lf / (S * c)  [per deg]   (Nelson, Flight Stability, eq. 2.45 / Fig 2.14)
  * Cn_beta,fus  = -1.3 * Vol / (S * b) * (Df / Wf) [per rad] (Raymer, Aircraft Design, eq. 16.47)

Control derivatives: the magnitude is measured; the sign follows the project command convention,
because which way OpenVSP's "positive" group deflection moves each surface is arbitrary:
  positive aileron  -> right-wing-down roll   => Cl_da > 0
  positive elevator -> trailing edge down     => Cm_de < 0
  positive rudder   -> trailing edge left     => Cn_dr < 0

Drag polar: VLM is inviscid, so its k misses the profile drag that grows with lift. Kroo's correction
(Aircraft Design: Synthesis and Analysis, 1/e = 1/e_inviscid + 0.38 CD0 pi AR) adds it: k = k_VLM + 0.38 CD0.
`ld_deviations` evaluates that polar and the sizing polar at the sizing cruise and loiter CLs (critic gate 2).
"""

import math
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np

from src.schemas.state import AeroStabilityOutput, SizingOutput
from src.tools.aero_cad.layout import QUARTER_CHORD, AircraftLayout
from src.tools.sizing.aero_model import lift_to_drag
from src.units import G0

LINEAR_ALPHA_WINDOW_DEG: Final[tuple[float, float]] = (-4.0, 4.0)
MIN_LINEAR_POINTS: Final[int] = 2
MIN_CL_ALPHA_PER_RAD: Final[float] = 1.0  # a real wing is ~4-6 /rad; below 1 the geometry is broken
ELEVATOR_MAX_ROLL_FRACTION: Final[float] = 0.2  # |dCMx| / |dCMy| for a symmetric elevator deflection
ALPHA_MATCH_TOL_DEG: Final[float] = 1e-6
AILERON_SIGN: Final[float] = 1.0
ELEVATOR_SIGN: Final[float] = -1.0
RUDDER_SIGN: Final[float] = -1.0
VISCOUS_LIFT_DRAG_FACTOR: Final[float] = 0.38  # Kroo: viscous lift-dependent drag, k += 0.38 * CD0
LD_DEVIATION_TOLERANCE: Final[float] = 0.15  # VSPAERO vs sizing L/D, either side, at each CL (critic gate 2)

# Nelson Fig. 2.14 (approximate digitisation): Kf vs wing-root quarter-chord position / body length.
KF_STATION: Final[tuple[float, ...]] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
KF_VALUE: Final[tuple[float, ...]] = (0.005, 0.007, 0.010, 0.016, 0.023, 0.035, 0.050)
FUSELAGE_CN_BETA_COEFF: Final[float] = 1.3
FUSELAGE_VOLUME_FACTOR: Final[float] = 0.6  # streamlined pod volume / circumscribing cylinder


class AeroAnalysisError(RuntimeError):
    """VSPAERO results are non-physical or the solver set-up violated an assumed convention."""


@dataclass(frozen=True)
class AeroCoefficients:
    """One converged VSPAERO case, geometry axes."""

    alpha_deg: float
    beta_deg: float
    cl: float
    cdi: float
    cmx: float
    cmy: float
    cmz: float

    def body_moments(self) -> tuple[float, float, float]:
        """(roll Cl, pitch Cm, yaw Cn) in body axes."""
        return -self.cmx, self.cmy, -self.cmz


@dataclass(frozen=True)
class VspaeroCases:
    sweep: tuple[AeroCoefficients, ...]  # alpha sweep at beta = 0, no deflection
    beta: AeroCoefficients  # alpha = 0, beta = +5 deg
    aileron: AeroCoefficients  # alpha = 0, antisymmetric deflection
    elevator: AeroCoefficients  # alpha = 0, symmetric deflection
    rudder: AeroCoefficients  # alpha = 0
    deflection_deg: float


@dataclass(frozen=True)
class ReferenceGeometry:
    sref_m2: float
    cref_m: float
    bref_m: float
    x_ref_m: float  # moment reference point, metres aft of the nose


def cruise_lift_coefficient(mass_kg: float, rho: float, speed_ms: float, sref_m2: float) -> float:
    return 2.0 * mass_kg * G0 / (rho * speed_ms**2 * sref_m2)


def fuselage_cm_alpha_per_rad(layout: AircraftLayout, ref: ReferenceGeometry) -> float:
    station: float = (layout.wing.x_le_root_m + QUARTER_CHORD * layout.wing.root_chord_m) / layout.fuselage_length_m
    kf: float = float(np.interp(station, KF_STATION, KF_VALUE))
    per_deg: float = kf * layout.fuselage_diameter_m**2 * layout.fuselage_length_m / (ref.sref_m2 * ref.cref_m)
    return math.degrees(per_deg)  # x (180/pi): per-degree -> per-radian


def fuselage_cn_beta_per_rad(layout: AircraftLayout, ref: ReferenceGeometry) -> float:
    d: float = layout.fuselage_diameter_m
    volume: float = FUSELAGE_VOLUME_FACTOR * math.pi / 4 * d**2 * layout.fuselage_length_m
    return -FUSELAGE_CN_BETA_COEFF * volume / (ref.sref_m2 * ref.bref_m)  # round body: Df / Wf = 1


def _linear_fit(sweep: tuple[AeroCoefficients, ...], values: list[float]) -> tuple[float, float]:
    """(slope per rad, value at alpha = 0) over the linear alpha window."""
    lo, hi = LINEAR_ALPHA_WINDOW_DEG
    pts = [(c.alpha_deg, v) for c, v in zip(sweep, values) if lo <= c.alpha_deg <= hi]
    if len(pts) < MIN_LINEAR_POINTS:
        raise AeroAnalysisError(f"need >= {MIN_LINEAR_POINTS} sweep points in alpha {LINEAR_ALPHA_WINDOW_DEG} deg")
    alpha, val = np.array(pts).T
    slope_per_deg, intercept = np.polyfit(alpha, val, 1)
    return math.degrees(float(slope_per_deg)), float(intercept)


def _zero_alpha_case(sweep: tuple[AeroCoefficients, ...]) -> AeroCoefficients:
    for case in sweep:
        if math.isclose(case.alpha_deg, 0.0, abs_tol=ALPHA_MATCH_TOL_DEG) and math.isclose(
                case.beta_deg, 0.0, abs_tol=ALPHA_MATCH_TOL_DEG):
            return case
    raise AeroAnalysisError("alpha sweep must contain the alpha = 0, beta = 0 baseline case")


def _check_control_conventions(cases: VspaeroCases, base: AeroCoefficients) -> None:
    """Guard the mirrored-surface gain convention: a wrong gain sign silently swaps aileron and flap."""
    d_lift: float = abs(cases.aileron.cl - base.cl)
    d_roll: float = abs(cases.aileron.cmx - base.cmx)
    if d_lift >= d_roll:
        raise AeroAnalysisError(f"aileron deflection behaves like a flap (dCL {d_lift:.4g} >= dCMx {d_roll:.4g})")
    e_roll: float = abs(cases.elevator.cmx - base.cmx)
    e_pitch: float = abs(cases.elevator.cmy - base.cmy)
    if e_roll >= ELEVATOR_MAX_ROLL_FRACTION * e_pitch:
        raise AeroAnalysisError(f"elevator deflection is not symmetric (dCMx {e_roll:.4g}, dCMy {e_pitch:.4g})")


def _induced_drag_factor(sweep: tuple[AeroCoefficients, ...]) -> float:
    """Least-squares k in CDi = k * CL^2."""
    cl = np.array([c.cl for c in sweep])
    cdi = np.array([c.cdi for c in sweep])
    return float(np.dot(cdi, cl**2) / np.dot(cl**2, cl**2))


def _derivative(case: AeroCoefficients, base: AeroCoefficients, axis: int, step_deg: float) -> float:
    return (case.body_moments()[axis] - base.body_moments()[axis]) / math.radians(step_deg)


def _control_derivative(case: AeroCoefficients, base: AeroCoefficients, axis: int, step_deg: float,
                        sign: float) -> float:
    """Measured magnitude, conventional sign (see module docstring)."""
    return sign * abs(_derivative(case, base, axis, step_deg))


def derive_stability(cases: VspaeroCases, ref: ReferenceGeometry, layout: AircraftLayout,
                     cd0: float, cl_cruise: float) -> AeroStabilityOutput:
    """Finite-difference derivatives (per rad, body axes) + fuselage corrections + neutral point."""
    cl_alpha, cl0 = _linear_fit(cases.sweep, [c.cl for c in cases.sweep])
    if cl_alpha < MIN_CL_ALPHA_PER_RAD:
        raise AeroAnalysisError(f"lift-curve slope {cl_alpha:.3f}/rad is non-physical; check VSPAERO geometry sets")
    base: AeroCoefficients = _zero_alpha_case(cases.sweep)
    _check_control_conventions(cases, base)
    cm_alpha_vlm, _ = _linear_fit(cases.sweep, [c.body_moments()[1] for c in cases.sweep])
    cm_alpha: float = cm_alpha_vlm + fuselage_cm_alpha_per_rad(layout, ref)
    k: float = _induced_drag_factor(cases.sweep) + VISCOUS_LIFT_DRAG_FACTOR * cd0
    return AeroStabilityOutput(
        cl0=cl0, cl_alpha_per_rad=cl_alpha, cd0=cd0,
        ld_cruise=lift_to_drag(cl_cruise, cd0, k), induced_drag_factor=k,
        cm_alpha_per_rad=cm_alpha,
        cn_beta_per_rad=_derivative(cases.beta, base, 2, cases.beta.beta_deg) + fuselage_cn_beta_per_rad(layout, ref),
        cl_beta_per_rad=_derivative(cases.beta, base, 0, cases.beta.beta_deg),
        cl_delta_a_per_rad=_control_derivative(cases.aileron, base, 0, cases.deflection_deg, AILERON_SIGN),
        cm_delta_e_per_rad=_control_derivative(cases.elevator, base, 1, cases.deflection_deg, ELEVATOR_SIGN),
        cn_delta_r_per_rad=_control_derivative(cases.rudder, base, 2, cases.deflection_deg, RUDDER_SIGN),
        x_np_m=ref.x_ref_m - cm_alpha / cl_alpha * ref.cref_m,
    )


@dataclass(frozen=True)
class LDPoint:
    """Sizing and VSPAERO L/D at one lift coefficient; deviation = VSPAERO / sizing - 1."""
    name: Literal["cruise", "loiter"]
    cl: float
    sizing_ld: float
    vspaero_ld: float

    @property
    def deviation(self) -> float:
        return self.vspaero_ld / self.sizing_ld - 1.0


def ld_deviations(sizing: SizingOutput, aero: AeroStabilityOutput) -> tuple[LDPoint, LDPoint]:
    """Both drag polars at the sizing cruise and loiter CLs. ValueError for a record that predates the fields."""
    if sizing.cl_cruise is None or sizing.cl_loiter is None or aero.induced_drag_factor is None:
        raise ValueError("the design predates the same-CL L/D check; re-run it")
    k_sizing: float = 1.0 / (math.pi * sizing.oswald_e * sizing.aspect_ratio)
    k_vspaero: float = aero.induced_drag_factor

    def point(name: Literal["cruise", "loiter"], cl: float) -> LDPoint:
        return LDPoint(name, cl, lift_to_drag(cl, sizing.cd0, k_sizing), lift_to_drag(cl, aero.cd0, k_vspaero))

    return point("cruise", sizing.cl_cruise), point("loiter", sizing.cl_loiter)
