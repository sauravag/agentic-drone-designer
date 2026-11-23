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
  positive ruddervator (each V-tail panel alone) -> trailing edge down => Cm < 0 for both panels
A V-tail's aggregated gate values follow by linearity from the two panel records: Cm_de = Cm_L + Cm_R (both trailing
edges down, the elevator convention) and Cn_dr = Cn_L - Cn_R (left down, right up: both trailing edges move left, the
rudder convention). On an upright V a panel's trailing edge down pushes the tail away from that panel, so the left
panel yaws nose-left (Cn_L < 0) and the right nose-right: Cn_dr < 0 is checked, which also catches swapped panels.

Drag polar: VLM is inviscid, so its k misses the profile drag that grows with lift. Kroo's correction
(Aircraft Design: Synthesis and Analysis, 1/e = 1/e_inviscid + 0.38 CD0 pi AR) adds it: k = k_VLM + 0.38 CD0.
`ld_deviations` evaluates that polar and the sizing polar at the sizing cruise and loiter CLs (critic gate 2).
"""

import math
from dataclasses import dataclass
from typing import Final, Literal, Optional

import numpy as np

from src.schemas.state import AeroStabilityOutput, ControlSurfaceDerivatives, SizingOutput
from src.tools.aero_cad.layout import QUARTER_CHORD, AircraftLayout
from src.tools.sizing.aero_model import lift_to_drag
from src.units import G0

LINEAR_ALPHA_WINDOW_DEG: Final[tuple[float, float]] = (-4.0, 4.0)
MIN_LINEAR_POINTS: Final[int] = 2
MIN_CL_ALPHA_PER_RAD: Final[float] = 1.0  # a real wing is ~4-6 /rad; below 1 the geometry is broken
ELEVATOR_MAX_ROLL_FRACTION: Final[float] = 0.2  # |dCMx| / |dCMy| for a symmetric elevator deflection
PANEL_MAX_ROLL_FRACTION: Final[float] = 0.2  # |dCMx| / |dCMy| for one ruddervator alone (both deflected: ~1)
PANEL_MIN_YAW_FRACTION: Final[float] = 0.01  # |dCMz| / |dCMy| for one ruddervator alone (G1: 0.05; symmetric: 0)
SYMMETRIC_PITCH_DOMINANCE: Final[float] = 5.0  # raw |dCMy_R - dCMy_L| / |dCMy_R + dCMy_L| (G1: ~130)
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
    """The run matrix. A conventional tail has the elevator and rudder cases, a V-tail the two ruddervator cases."""

    sweep: tuple[AeroCoefficients, ...]  # alpha sweep at beta = 0, no deflection
    beta: AeroCoefficients  # alpha = 0, beta = +5 deg
    aileron: AeroCoefficients  # alpha = 0, antisymmetric deflection
    deflection_deg: float
    elevator: Optional[AeroCoefficients] = None  # alpha = 0, symmetric deflection
    rudder: Optional[AeroCoefficients] = None  # alpha = 0
    ruddervator_left: Optional[AeroCoefficients] = None  # alpha = 0, left V-tail panel alone (raw OpenVSP direction)
    ruddervator_right: Optional[AeroCoefficients] = None  # alpha = 0, right V-tail panel alone


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
    if cases.elevator is not None:
        e_roll: float = abs(cases.elevator.cmx - base.cmx)
        e_pitch: float = abs(cases.elevator.cmy - base.cmy)
        if e_roll >= ELEVATOR_MAX_ROLL_FRACTION * e_pitch:
            raise AeroAnalysisError(f"elevator deflection is not symmetric (dCMx {e_roll:.4g}, dCMy {e_pitch:.4g})")


def _check_vtail_conventions(panel_cases: dict[str, AeroCoefficients], base: AeroCoefficients,
                             left: ControlSurfaceDerivatives, right: ControlSurfaceDerivatives) -> None:
    """V-tail rule, on the raw VSPAERO panel cases (before the per-panel sign normalisation, which would make any sign
    check pass by construction) and then on the normalised records:
      * one panel alone: pitch dominates roll (equal gains, i.e. both panels antisymmetric, roll as much as they pitch);
      * yaw per panel: |dCMz| is a real fraction of |dCMy| (both panels deflected together symmetrically, gains
        (1, -1), give no yaw);
      * symmetric pitch: OpenVSP deflects a mirrored pair antisymmetrically under equal gains, so the two raw panel
        cases (gains (0, 1) and (1, 0)) must pitch with opposite signs, and their symmetric combination (right minus
        left) must dominate the antisymmetric one; a same-sign pair means the gain convention or the geometry is wrong
        and the "both trailing edges up" pair would not pitch nose up;
      * yaw sign: on the normalised records (trailing edge down) the left panel yaws nose-left and the right
        nose-right, so the antisymmetric pair (left down, right up) yaws nose-left; swapped panels or an inverted V
        fail.
    The roll coupling is recorded, not gated."""
    deltas: dict[str, tuple[float, float, float]] = {
        name: (case.cmx - base.cmx, case.cmy - base.cmy, case.cmz - base.cmz) for name, case in panel_cases.items()}
    for name, (d_roll, d_pitch, d_yaw) in deltas.items():
        if abs(d_roll) >= PANEL_MAX_ROLL_FRACTION * abs(d_pitch):
            raise AeroAnalysisError(f"{name} case does not deflect one panel alone "
                                    f"(dCMx {d_roll:.4g}, dCMy {d_pitch:.4g})")
        if abs(d_yaw) < PANEL_MIN_YAW_FRACTION * abs(d_pitch):
            raise AeroAnalysisError(f"{name} case does not yaw (dCMz {d_yaw:.4g}, dCMy {d_pitch:.4g}): both panels "
                                    "deflected together, not one alone")
    pitch_left, pitch_right = deltas["ruddervator_left"][1], deltas["ruddervator_right"][1]
    symmetric, antisymmetric = abs(pitch_right - pitch_left), abs(pitch_right + pitch_left)
    if pitch_left * pitch_right >= 0 or symmetric < SYMMETRIC_PITCH_DOMINANCE * antisymmetric:
        raise AeroAnalysisError("ruddervator panels do not pitch as a mirrored pair (raw dCMy left "
                                f"{pitch_left:.4g}, right {pitch_right:.4g}): the symmetric deflection would not pitch")
    if not left.cn_per_rad < 0 < right.cn_per_rad:
        raise AeroAnalysisError("ruddervators deflected trailing-edges-left do not yaw the nose left (panels swapped "
                                f"or the V inverted): Cn_L {left.cn_per_rad:.4g}, Cn_R {right.cn_per_rad:.4g}")


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


def _surface_record(name: str, case: AeroCoefficients, base: AeroCoefficients, step_deg: float,
                    primary_axis: int, sign: float) -> ControlSurfaceDerivatives:
    """All four derivatives of one deflection case. The primary axis carries the command-convention sign; the
    other coefficients flip with it so the record describes one consistent deflection direction."""
    step_rad: float = math.radians(step_deg)
    roll, pitch, yaw = (_derivative(case, base, axis, step_deg) for axis in range(3))
    primary: float = (roll, pitch, yaw)[primary_axis]
    flip: float = sign * math.copysign(1.0, primary) if primary else sign
    return ControlSurfaceDerivatives(name=name, cl_per_rad=flip * (case.cl - base.cl) / step_rad,
                                     cl_roll_per_rad=flip * roll, cm_per_rad=flip * pitch, cn_per_rad=flip * yaw)


def tractor_surface_records(cases: VspaeroCases, base: AeroCoefficients) -> tuple[ControlSurfaceDerivatives, ...]:
    """Aileron, elevator and rudder records from the three deflection cases."""
    if cases.elevator is None or cases.rudder is None:
        raise AeroAnalysisError("a conventional tail needs the elevator and rudder deflection cases")
    step: float = cases.deflection_deg
    return (_surface_record("aileron", cases.aileron, base, step, 0, AILERON_SIGN),
            _surface_record("elevator", cases.elevator, base, step, 1, ELEVATOR_SIGN),
            _surface_record("rudder", cases.rudder, base, step, 2, RUDDER_SIGN))


def vtail_surface_records(cases: VspaeroCases, base: AeroCoefficients) -> tuple[ControlSurfaceDerivatives, ...]:
    """Aileron and the two ruddervator records, each panel's sign set by its own pitch (trailing edge down)."""
    if cases.ruddervator_left is None or cases.ruddervator_right is None:
        raise AeroAnalysisError("a V-tail needs both ruddervator deflection cases")
    step: float = cases.deflection_deg
    return (_surface_record("aileron", cases.aileron, base, step, 0, AILERON_SIGN),
            _surface_record("ruddervator_left", cases.ruddervator_left, base, step, 1, ELEVATOR_SIGN),
            _surface_record("ruddervator_right", cases.ruddervator_right, base, step, 1, ELEVATOR_SIGN))


def _pitch_yaw_controls(cases: VspaeroCases, base: AeroCoefficients) -> tuple[
        float, float, tuple[ControlSurfaceDerivatives, ...]]:
    """(Cm_delta_e, Cn_delta_r, per-surface records) for the tail the cases describe."""
    step: float = cases.deflection_deg
    if cases.ruddervator_left is not None and cases.ruddervator_right is not None:
        records = vtail_surface_records(cases, base)
        _, left, right = records
        _check_vtail_conventions({"ruddervator_left": cases.ruddervator_left,
                                  "ruddervator_right": cases.ruddervator_right}, base, left, right)
        return left.cm_per_rad + right.cm_per_rad, left.cn_per_rad - right.cn_per_rad, records
    if cases.elevator is None or cases.rudder is None:
        raise AeroAnalysisError("the run matrix has neither the elevator and rudder cases nor the ruddervator cases")
    return (_control_derivative(cases.elevator, base, 1, step, ELEVATOR_SIGN),
            _control_derivative(cases.rudder, base, 2, step, RUDDER_SIGN), tractor_surface_records(cases, base))


def derive_stability(cases: VspaeroCases, ref: ReferenceGeometry, layout: AircraftLayout,
                     cd0: float, cl_cruise: float) -> AeroStabilityOutput:
    """Finite-difference derivatives (per rad, body axes) + fuselage corrections + neutral point."""
    cl_alpha, cl0 = _linear_fit(cases.sweep, [c.cl for c in cases.sweep])
    if cl_alpha < MIN_CL_ALPHA_PER_RAD:
        raise AeroAnalysisError(f"lift-curve slope {cl_alpha:.3f}/rad is non-physical; check VSPAERO geometry sets")
    base: AeroCoefficients = _zero_alpha_case(cases.sweep)
    _check_control_conventions(cases, base)
    cm_delta_e, cn_delta_r, records = _pitch_yaw_controls(cases, base)
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
        cm_delta_e_per_rad=cm_delta_e, cn_delta_r_per_rad=cn_delta_r,
        x_np_m=ref.x_ref_m - cm_alpha / cl_alpha * ref.cref_m,
        control_surfaces=records,
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
