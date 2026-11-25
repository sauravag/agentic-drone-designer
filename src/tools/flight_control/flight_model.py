"""One physical flight model shared by the ArduPilot parameter generator and the JSBSim exporter.

Everything is derived deterministically (SRS "Zero LLM Math Rule") from the M2 sizing, the COTS BOM,
the M3 VSPAERO derivatives and the as-built CAD mass properties, in SI units:

* Static/control derivatives come from VSPAERO. VSPAERO references moments to the wing aerodynamic
  centre, so pitch stiffness is re-based on the as-built CG: Cm_alpha = -CL_alpha * SM. The other
  moment derivatives move by ~0.1 MAC, negligible next to the tail arm, and are used as-is.
* Damping derivatives (Clp, Cmq, Cnr, ...) are not in the VSPAERO run matrix; they are estimated
  with standard strip-theory / tail-volume (DATCOM-style) formulas from the sized geometry, with the tail volumes
  from the tail surfaces' projected (effective) areas, so a V-tail damps like its horizontal and vertical shares.
* The cruise thrust acts at the propeller hub the CAD stage placed (nose tractor or boom-end pusher), pointing forward.
* Propulsion (spec 6.6): every propeller is a `Propulsor` (propeller tables + motor winding + ESC on the pack bus,
  `propulsion_plant`), so thrust follows throttle -> voltage -> rpm at the simulated airspeed, saturating at the
  tip-Mach-0.80 rpm. A lift rotor's max thrust is its propulsor at full throttle, static, in the model's air (not the
  catalogue rating), and ArduPilot's Q_M_THST_EXPO / Q_M_THST_HOVER are fitted to that same curve.

Because the autopilot gains and the SITL plant both read this model, the SITL test checks the closed
loop (rate/angle controllers, actuator lag, inertia coupling, transition), not the derivative estimates.
"""

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Final, Literal, Optional

from src.schemas.assumptions import DEFAULT_ASSUMPTIONS
from src.schemas.components import SelectedBOM
from src.schemas.requirements import OperationalRequirements
from src.schemas.state import (AeroStabilityOutput, CADArtifacts, ControlSurfaceDerivatives, SizingOutput,
                               effective_tail_areas)
from src.tools.aero_cad.layout import DEFAULT_WING_STATION, WING_TAPER_RATIO, layout_from_sizing
from src.tools.aero_cad.stability import cruise_lift_coefficient
from src.tools.flight_control.control_channels import (AILERON_MAX_RAD, ELEVATOR_MAX_RAD,  # noqa: F401 (re-exported)
                                                       RUDDER_MAX_RAD, ControlChannel, channels_with_steering,
                                                       control_channels)
from src.tools.flight_control.propulsion_plant import plant_propulsor
from src.tools.gear import GearGeometry, PropellerDisc
from src.tools.propulsion.propulsor import Propulsor
from src.tools.sizing.aero_model import VNE_TO_CRUISE, air_properties, induced_drag_factor, speed_of_sound_ms
from src.tools.sizing.layouts import layout_spec
from src.tools.sizing.mass_model import as_built_gear
from src.units import G0

SITL_ALTITUDE_M: Final[float] = 0.0  # SITL flies near sea level; gains are scheduled with airspeed
HTAIL_ASPECT_RATIO: Final[float] = 4.0  # matches the M3 layout
VTAIL_EFFECTIVE_ASPECT: Final[float] = 1.55 * 1.5  # Raymer end-plate factor x fin AR
TAIL_EFFICIENCY: Final[float] = 0.9  # dynamic-pressure ratio at the tail
FUSELAGE_DAMPING_FACTOR: Final[float] = 1.1  # fuselage/wing contribution to Cmq (Nelson)
ROLL_YAW_CL_FACTOR_CLR: Final[float] = 0.25  # Clr ~ CL/4 (strip theory)
ROLL_YAW_CL_FACTOR_CNP: Final[float] = -0.125  # Cnp ~ -CL/8 (strip theory)
ADVERSE_YAW_FACTOR: Final[float] = -0.2  # Cn_delta_a ~ -0.2 CL Cl_delta_a
SERVO_TIME_CONSTANT_S: Final[float] = 0.05
MOTOR_LAG_BASE_S: Final[float] = 0.03  # first-order spin-up time constant = base + slope * diameter
MOTOR_LAG_PER_DIAMETER_S_M: Final[float] = 0.06
ROTOR_TORQUE_PER_THRUST_DIAMETER: Final[float] = 0.05  # Q / (T D) ~ C_Q / C_T
THROTTLE_SOLVE_TOL: Final[float] = 1e-4  # throttle bracket at which the thrust -> throttle bisection stops
# Thrust within this relative margin of the target counts as reaching it: a flat (current-limited) top is flat
# only up to the rpm solve's tolerance (propulsor ROOT_RTOL 1e-9), so its last digits must not move the trim.
THRUST_MATCH_RTOL: Final[float] = 1e-9
LIFT_CURVE_POINTS: Final[int] = 21  # lift-rotor throttle samples the Q_M_THST_EXPO fit uses
TRANSITION_SPEED_FACTOR: Final[float] = 1.2  # SRS: accelerate 0 -> 1.2 Vs
TRANSITION_DT_S: Final[float] = 0.01
MAX_TRANSITION_S: Final[float] = 60.0


class FlightModelError(ValueError):
    """The design cannot be flown (e.g. no thrust margin to accelerate, or lift rotors below weight)."""


@dataclass(frozen=True)
class AeroCoefficients:
    """Body-axis non-dimensional derivatives about the as-built CG (per rad)."""

    cl0: float
    cl_alpha: float
    cl_max: float
    cl_q: float
    cd0: float
    k_induced: float
    cl_delta_e: float
    cm0: float
    cm_alpha: float
    cm_q: float
    cm_delta_e: float
    cy_beta: float
    cl_beta: float
    cl_p: float
    cl_r: float
    cl_delta_a: float
    cn_beta: float
    cn_p: float
    cn_r: float
    cn_delta_r: float
    cn_delta_a: float


@dataclass(frozen=True)
class LiftRotor:
    """One lift rotor in CAD axes (x aft of nose, y right, z up) with its ArduPilot motor number."""

    motor_number: int  # 1..4, ArduPilot QUAD/X numbering
    x_m: float
    y_m: float
    z_m: float
    spin: Literal[1, -1]  # +1: CCW seen from above -> reaction torque yaws the nose right
    max_thrust_n: float  # the lift propulsor at full throttle, static, in the model's air
    diameter_m: float


@dataclass(frozen=True)
class FlightModel:
    mass_kg: float
    ixx_kgm2: float
    iyy_kgm2: float
    izz_kgm2: float
    ixz_kgm2: float
    x_cg_m: float
    static_margin: float
    wing_area_m2: float
    wingspan_m: float
    mean_chord_m: float
    fuselage_length_m: float
    fuselage_diameter_m: float
    tail_arm_m: float
    aero: AeroCoefficients
    v_stall_ms: float
    v_cruise_ms: float
    v_ne_ms: float
    rho_kgm3: float
    mu_pa_s: float
    speed_of_sound_ms: float
    cruise_propulsor: Propulsor
    cruise_prop_diameter_m: float
    cruise_motor_lag_s: float
    lift_rotors: tuple[LiftRotor, ...]
    lift_propulsor: Optional[Propulsor]  # one per rotor, all alike; None without lift rotors
    lift_motor_lag_s: float
    gear: GearGeometry  # the one landing-gear rule; kind "none" for a hand launch
    launch: Literal["runway", "hand", "vertical"]
    wing_tip_z_m: float  # height of the wing tips above the fuselage axis (dihedral), where a gearless aircraft rests
    # Per-surface derivatives for layouts whose surfaces are not the tractor triple; empty means the tractor
    # aileron/elevator/rudder, derived from the aggregated `aero` so a perturbed plant keeps them consistent.
    surfaces: tuple[ControlSurfaceDerivatives, ...] = ()
    propeller_x_m: float = 0.0  # cruise propeller hub (thrust station), CAD axes; the nose for records without one
    propeller_z_m: float = 0.0

    @property
    def is_quadplane(self) -> bool:
        return bool(self.lift_rotors)

    @property
    def channels(self) -> tuple[ControlChannel, ...]:
        """One channel per servo-driven surface, plus the dedicated ground-steering channel when no rudder steers
        and there is a nose wheel to steer (a gearless aircraft needs none)."""
        channels = control_channels(self.aero, self.surfaces)
        if self.gear.kind == "none":
            return channels
        return channels_with_steering(channels, self.is_quadplane)

    @cached_property
    def transition_s(self) -> float:
        """`transition_time_s`, solved once per model: its Euler loop root-solves the propulsor at every step. A
        `replace`d model is a new instance and solves again."""
        return _solve_transition_s(self)

    @property
    def weight_n(self) -> float:
        return self.mass_kg * G0

    def dynamic_pressure(self, speed_ms: float) -> float:
        return 0.5 * self.rho_kgm3 * speed_ms**2

    def cruise_thrust_n(self, throttle: float, speed_ms: float) -> float:
        """Delivered cruise thrust at `throttle` (clamped to [0, 1], the ESC duty) and airspeed, in the model's air.
        Non-decreasing in throttle and flat once the rpm reaches tip Mach 0.80."""
        return self.cruise_propulsor.thrust_n(min(max(throttle, 0.0), 1.0), max(speed_ms, 0.0), self.rho_kgm3,
                                              self.mu_pa_s, self.speed_of_sound_ms)

    def cruise_throttle_for(self, thrust_n: float, speed_ms: float) -> Optional[float]:
        """The lowest throttle (within THROTTLE_SOLVE_TOL) giving `thrust_n` at `speed_ms`, or None when full throttle
        cannot. Bisection, so a flat (saturated) top resolves to where the saturation starts."""
        if self.cruise_thrust_n(1.0, speed_ms) < thrust_n:
            return None
        reached: float = thrust_n * (1.0 - THRUST_MATCH_RTOL)
        lo, hi = 0.0, 1.0
        while hi - lo > THROTTLE_SOLVE_TOL:
            mid: float = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if self.cruise_thrust_n(mid, speed_ms) >= reached else (mid, hi)
        return hi

    def lift_thrust_n(self, throttle: float) -> float:
        """Static thrust of one lift rotor at `throttle` (clamped to [0, 1]) in the model's air."""
        if self.lift_propulsor is None:
            raise FlightModelError("this flight model has no lift rotors")
        return self.lift_propulsor.thrust_n(min(max(throttle, 0.0), 1.0), 0.0, self.rho_kgm3, self.mu_pa_s,
                                            self.speed_of_sound_ms)

    def lift_thrust_fractions(self) -> tuple[list[float], list[float]]:
        """(throttles, thrust / full-throttle thrust) of one lift rotor on LIFT_CURVE_POINTS evenly spaced points."""
        throttles: list[float] = [i / (LIFT_CURVE_POINTS - 1) for i in range(LIFT_CURVE_POINTS)]
        full: float = self.lift_thrust_n(1.0)
        return throttles, [self.lift_thrust_n(u) / full for u in throttles]

    def cruise_lift_coefficient(self, speed_ms: float) -> float:
        return cruise_lift_coefficient(self.mass_kg, self.rho_kgm3, speed_ms, self.wing_area_m2)

    def drag_n(self, speed_ms: float, cl: float) -> float:
        return self.dynamic_pressure(speed_ms) * self.wing_area_m2 * (self.aero.cd0 + self.aero.k_induced * cl**2)

    def trim_alpha_rad(self, speed_ms: float) -> float:
        return (self.cruise_lift_coefficient(speed_ms) - self.aero.cl0) / self.aero.cl_alpha


def _helmbold_lift_slope(aspect: float) -> float:
    """Finite-wing lift-curve slope per rad (Helmbold, low sweep)."""
    return 2 * math.pi * aspect / (2 + math.sqrt(aspect**2 + 4))


def tail_areas_m2(sizing: SizingOutput) -> tuple[float, float]:
    """(horizontal, vertical) projected tail areas from the tail surfaces; a record that predates them carries the
    same effective areas in its legacy fields."""
    if sizing.tail_surfaces:
        return effective_tail_areas(sizing.tail_surfaces)
    return sizing.horizontal_tail_area_m2, sizing.vertical_tail_area_m2


def _damping(sizing: SizingOutput, aero: AeroStabilityOutput, cl_cruise: float) -> dict[str, float]:
    """Strip-theory and tail-volume damping derivatives (Nelson / DATCOM approximations)."""
    s, b, c, arm = sizing.wing_area_m2, sizing.wingspan_m, sizing.mean_chord_m, sizing.tail_arm_m
    a_h: float = _helmbold_lift_slope(HTAIL_ASPECT_RATIO)
    a_v: float = _helmbold_lift_slope(VTAIL_EFFECTIVE_ASPECT)
    s_h, s_v = tail_areas_m2(sizing)
    v_h: float = s_h * arm / (s * c)
    v_v: float = s_v * arm / (s * b)
    taper: float = WING_TAPER_RATIO
    return {
        "cl_q": 2 * TAIL_EFFICIENCY * a_h * v_h,
        "cl_delta_e": -aero.cm_delta_e_per_rad * c / arm,  # tail lift that produces Cm_delta_e
        "cm_q": -2 * TAIL_EFFICIENCY * a_h * v_h * (arm / c) * FUSELAGE_DAMPING_FACTOR,
        "cl_p": -(aero.cl_alpha_per_rad / 12) * (1 + 3 * taper) / (1 + taper),
        "cn_r": -2 * TAIL_EFFICIENCY * a_v * v_v * (arm / b) - aero.cd0 / 4,
        "cy_beta": -TAIL_EFFICIENCY * a_v * s_v / s,
        "cl_r": ROLL_YAW_CL_FACTOR_CLR * cl_cruise,
        "cn_p": ROLL_YAW_CL_FACTOR_CNP * cl_cruise,
        "cn_delta_a": ADVERSE_YAW_FACTOR * cl_cruise * aero.cl_delta_a_per_rad,
    }


def _coefficients(sizing: SizingOutput, aero: AeroStabilityOutput, cad: CADArtifacts,
                  cl_cruise: float) -> AeroCoefficients:
    cm_alpha: float = -aero.cl_alpha_per_rad * cad.static_margin
    alpha_trim: float = (cl_cruise - aero.cl0) / aero.cl_alpha_per_rad
    return AeroCoefficients(
        cl0=aero.cl0, cl_alpha=aero.cl_alpha_per_rad, cl_max=sizing.cl_max, cd0=aero.cd0,
        k_induced=induced_drag_factor(sizing.aspect_ratio),
        cm0=-cm_alpha * alpha_trim,  # tail incidence set so the cruise point trims with zero elevator
        cm_alpha=cm_alpha, cm_delta_e=aero.cm_delta_e_per_rad,
        cl_beta=aero.cl_beta_per_rad, cl_delta_a=aero.cl_delta_a_per_rad,
        cn_beta=aero.cn_beta_per_rad, cn_delta_r=aero.cn_delta_r_per_rad,
        **_damping(sizing, aero, cl_cruise),
    )


def motor_lag_s(diameter_m: float) -> float:
    return MOTOR_LAG_BASE_S + MOTOR_LAG_PER_DIAMETER_S_M * diameter_m


def _lift_rotors(sizing: SizingOutput, bom: SelectedBOM, aero: AeroStabilityOutput, cad: CADArtifacts,
                 max_thrust_n: float) -> tuple[LiftRotor, ...]:
    """Rotor hubs from the M3 layout, re-based on the as-built wing AC (VSPAERO moment reference)."""
    if bom.vtol_motor is None or bom.vtol_propeller is None:
        return ()
    layout = layout_from_sizing(sizing, DEFAULT_WING_STATION)
    x_ac_actual: float = aero.x_np_m + aero.cm_alpha_per_rad / aero.cl_alpha_per_rad * layout.wing.mac_m
    shift: float = x_ac_actual - layout.wing.x_ac_m
    rotors: list[LiftRotor] = []
    for x, y, z in layout.rotor_positions:
        front: bool = x + shift < cad.x_cg_m
        right: bool = y > 0
        number: int = {(True, True): 1, (False, False): 2, (True, False): 3, (False, True): 4}[(front, right)]
        rotors.append(LiftRotor(motor_number=number, x_m=x + shift, y_m=y, z_m=z, spin=1 if number <= 2 else -1,
                                max_thrust_n=max_thrust_n, diameter_m=bom.vtol_propeller.diameter_m))
    if sorted(r.motor_number for r in rotors) != [1, 2, 3, 4]:
        raise FlightModelError(f"lift rotors do not form a QUAD/X layout: {[(r.x_m, r.y_m) for r in rotors]}")
    return tuple(sorted(rotors, key=lambda r: r.motor_number))


TRACTOR_SURFACES: Final[tuple[str, ...]] = ("aileron", "elevator", "rudder")


def _model_surfaces(aero: AeroStabilityOutput) -> tuple[ControlSurfaceDerivatives, ...]:
    """Per-surface derivatives the flight model keeps. The tractor triple stays on the aggregated (measured plus
    estimated) derivatives so its parameters and JSBSim file are unchanged; other surface sets are used as recorded."""
    if tuple(r.name for r in aero.control_surfaces) == TRACTOR_SURFACES:
        return ()
    return aero.control_surfaces


def _wing_tip_z_m(sizing: SizingOutput) -> float:
    wing = layout_from_sizing(sizing, DEFAULT_WING_STATION).wing
    return wing.z_at(wing.semi_span_m)


def _propeller_disc(sizing: SizingOutput, bom: SelectedBOM, cad: CADArtifacts) -> PropellerDisc:
    """The cruise propeller where the CAD stage packed it. A record without the hub position predates it and is a
    nose tractor (hub at the origin); any other propeller needs the recorded station."""
    diameter: float = bom.cruise_propeller.diameter_m
    if cad.propeller_x_m is not None and cad.propeller_z_m is not None:
        return PropellerDisc(cad.propeller_x_m, cad.propeller_z_m, diameter)
    propeller = layout_spec(sizing.airframe_layout).propeller
    if propeller != "tractor_nose":
        raise ValueError(f"the CAD record has no propeller station, which a {propeller!r} propeller needs")
    return PropellerDisc(0.0, 0.0, diameter)


def _propulsors(sizing: SizingOutput, bom: SelectedBOM,
                pusher_installation_factor: float) -> tuple[Propulsor, Optional[Propulsor]]:
    """(cruise, lift) propulsors; the pusher installation loss applies to any layout whose propeller is not a nose
    tractor (the propulsion matching's rule)."""
    is_tractor: bool = layout_spec(sizing.airframe_layout).propeller == "tractor_nose"
    cruise: Propulsor = plant_propulsor(bom.cruise_motor, bom.cruise_propeller, bom.cruise_esc, bom.battery,
                                        1.0 if is_tractor else pusher_installation_factor)
    if bom.vtol_motor is None or bom.vtol_propeller is None:
        return cruise, None
    return cruise, plant_propulsor(bom.vtol_motor, bom.vtol_propeller, bom.vtol_esc, bom.battery)


def build_flight_model(req: OperationalRequirements, sizing: SizingOutput, bom: SelectedBOM,
                       aero: AeroStabilityOutput, cad: CADArtifacts,
                       pusher_installation_factor: float = DEFAULT_ASSUMPTIONS.pusher_efficiency_factor) -> FlightModel:
    """Assemble the flight model; raises FlightModelError if the design cannot hover/accelerate.
    `pusher_installation_factor` is the run's DesignAssumptions.pusher_efficiency_factor (default: the default)."""
    rho, mu = air_properties(SITL_ALTITUDE_M)
    sound: float = speed_of_sound_ms(SITL_ALTITUDE_M)
    cl_cruise: float = cruise_lift_coefficient(cad.as_built_mass_kg, rho, req.cruise_speed_ms, sizing.wing_area_m2)
    cruise, lift = _propulsors(sizing, bom, pusher_installation_factor)
    lift_max_n: float = 0.0 if lift is None else lift.thrust_n(1.0, 0.0, rho, mu, sound)
    propeller: PropellerDisc = _propeller_disc(sizing, bom, cad)
    gear: GearGeometry = as_built_gear(sizing, cad.x_cg_m, propeller)
    model = FlightModel(
        mass_kg=cad.as_built_mass_kg, ixx_kgm2=cad.ixx_kgm2, iyy_kgm2=cad.iyy_kgm2, izz_kgm2=cad.izz_kgm2,
        ixz_kgm2=cad.ixz_kgm2, x_cg_m=cad.x_cg_m, static_margin=cad.static_margin,
        wing_area_m2=sizing.wing_area_m2, wingspan_m=sizing.wingspan_m,
        mean_chord_m=sizing.mean_chord_m, fuselage_length_m=sizing.fuselage_length_m,
        fuselage_diameter_m=sizing.fuselage_diameter_m, tail_arm_m=sizing.tail_arm_m,
        aero=_coefficients(sizing, aero, cad, cl_cruise),
        v_stall_ms=sizing.v_stall_ms, v_cruise_ms=req.cruise_speed_ms, v_ne_ms=VNE_TO_CRUISE * req.cruise_speed_ms,
        rho_kgm3=rho, mu_pa_s=mu, speed_of_sound_ms=sound, cruise_propulsor=cruise,
        cruise_prop_diameter_m=bom.cruise_propeller.diameter_m,
        cruise_motor_lag_s=motor_lag_s(bom.cruise_propeller.diameter_m),
        lift_rotors=_lift_rotors(sizing, bom, aero, cad, lift_max_n), lift_propulsor=lift,
        lift_motor_lag_s=motor_lag_s(bom.vtol_propeller.diameter_m) if bom.vtol_propeller else 0.0,
        gear=gear, launch=sizing.launch_method, wing_tip_z_m=_wing_tip_z_m(sizing), surfaces=_model_surfaces(aero),
        propeller_x_m=propeller.x_m, propeller_z_m=propeller.z_m,
    )
    _check_flyable(model)
    return model


def _check_flyable(model: FlightModel) -> None:
    a = model.aero
    positive = {"mass": model.mass_kg, "Ixx": model.ixx_kgm2, "Iyy": model.iyy_kgm2, "Izz": model.izz_kgm2,
                "CL_alpha": a.cl_alpha, "Cl_delta_a": a.cl_delta_a, "-Cm_delta_e": -a.cm_delta_e,
                "-Cl_p": -a.cl_p, "-Cm_q": -a.cm_q}
    bad = {name: value for name, value in positive.items() if not (math.isfinite(value) and value > 0)}
    if bad:
        raise FlightModelError(f"flight model has non-physical values (expected finite and correctly signed): {bad}")
    cruise_drag: float = model.drag_n(model.v_cruise_ms, model.cruise_lift_coefficient(model.v_cruise_ms))
    full_n: float = model.cruise_thrust_n(1.0, model.v_cruise_ms)
    if full_n <= cruise_drag:
        raise FlightModelError(f"cruise propeller cannot overcome {cruise_drag:.1f} N drag at {model.v_cruise_ms} m/s "
                               f"(full-throttle thrust {full_n:.1f} N)")
    lift_n: float = sum(r.max_thrust_n for r in model.lift_rotors)
    if model.is_quadplane and lift_n <= model.weight_n:
        raise FlightModelError(f"lift rotors cannot lift the as-built aircraft ({lift_n:.1f} N full-throttle static "
                               f"thrust for {model.weight_n:.1f} N weight)")


def transition_time_s(model: FlightModel) -> float:
    """Time for the cruise motor alone to accelerate 0 -> 1.2 Vs against parasite drag (rotors carry
    the weight, so the wing is unloaded); cached on the model."""
    return model.transition_s


def _solve_transition_s(model: FlightModel) -> float:
    """The transition_time_s integration: explicit Euler, deterministic step."""
    target: float = TRANSITION_SPEED_FACTOR * model.v_stall_ms
    speed, elapsed = 0.0, 0.0
    while speed < target:
        net: float = model.cruise_thrust_n(1.0, speed) - model.drag_n(speed, 0.0)
        if net <= 0 or elapsed > MAX_TRANSITION_S:
            raise FlightModelError(f"cannot reach the transition speed {target:.1f} m/s (stuck at {speed:.1f} m/s)")
        speed += net / model.mass_kg * TRANSITION_DT_S
        elapsed += TRANSITION_DT_S
    return elapsed
