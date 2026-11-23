"""One physical flight model shared by the ArduPilot parameter generator and the JSBSim exporter.

Everything is derived deterministically (SRS "Zero LLM Math Rule") from the M2 sizing, the COTS BOM,
the M3 VSPAERO derivatives and the as-built CAD mass properties, in SI units:

* Static/control derivatives come from VSPAERO. VSPAERO references moments to the wing aerodynamic
  centre, so pitch stiffness is re-based on the as-built CG: Cm_alpha = -CL_alpha * SM. The other
  moment derivatives move by ~0.1 MAC, negligible next to the tail arm, and are used as-is.
* Damping derivatives (Clp, Cmq, Cnr, ...) are not in the VSPAERO run matrix; they are estimated
  with standard strip-theory / tail-volume (DATCOM-style) formulas from the sized geometry.
* Propulsion: momentum-theory static thrust with a power-limited roll-off for the cruise propeller;
  lift rotors use the catalog max thrust and ArduPilot's own thrust-curve shape (MOT_THST_EXPO), so the
  autopilot's linearisation matches the simulated plant.

Because the autopilot gains and the SITL plant both read this model, the SITL test checks the closed
loop (rate/angle controllers, actuator lag, inertia coupling, transition), not the derivative estimates.
"""

import math
from dataclasses import dataclass
from typing import Final, Literal

from src.schemas.components import SelectedBOM
from src.schemas.requirements import OperationalRequirements
from src.schemas.state import AeroStabilityOutput, CADArtifacts, ControlSurfaceDerivatives, SizingOutput
from src.tools.aero_cad.layout import DEFAULT_WING_STATION, WING_TAPER_RATIO, layout_from_sizing
from src.tools.aero_cad.stability import cruise_lift_coefficient
from src.tools.flight_control.control_channels import (AILERON_MAX_RAD, ELEVATOR_MAX_RAD,  # noqa: F401 (re-exported)
                                                       RUDDER_MAX_RAD, ControlChannel, channels_with_steering,
                                                       control_channels)
from src.tools.sizing.aero_model import VNE_TO_CRUISE, air_properties, induced_drag_factor
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
PROP_FIGURE_OF_MERIT: Final[float] = 0.6  # static figure of merit, cruise propeller
PROP_EFFICIENCY: Final[float] = 0.75  # in-flight propulsive efficiency
MOTOR_LAG_BASE_S: Final[float] = 0.03  # first-order spin-up time constant = base + slope * diameter
MOTOR_LAG_PER_DIAMETER_S_M: Final[float] = 0.06
ROTOR_TORQUE_PER_THRUST_DIAMETER: Final[float] = 0.05  # Q / (T D) ~ C_Q / C_T
THRUST_EXPO: Final[float] = 0.65  # ArduPilot Q_M_THST_EXPO default; the plant uses the same curve
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
    max_thrust_n: float
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
    cruise_static_thrust_n: float
    cruise_shaft_power_w: float
    cruise_prop_diameter_m: float
    cruise_motor_lag_s: float
    lift_rotors: tuple[LiftRotor, ...]
    lift_motor_lag_s: float
    # Per-surface derivatives for layouts whose surfaces are not the tractor triple; empty means the tractor
    # aileron/elevator/rudder, derived from the aggregated `aero` so a perturbed plant keeps them consistent.
    surfaces: tuple[ControlSurfaceDerivatives, ...] = ()

    @property
    def is_quadplane(self) -> bool:
        return bool(self.lift_rotors)

    @property
    def channels(self) -> tuple[ControlChannel, ...]:
        """One channel per servo-driven surface, plus the dedicated ground-steering channel when no rudder steers."""
        return channels_with_steering(control_channels(self.aero, self.surfaces, self.is_quadplane), self.is_quadplane)

    @property
    def weight_n(self) -> float:
        return self.mass_kg * G0

    def dynamic_pressure(self, speed_ms: float) -> float:
        return 0.5 * self.rho_kgm3 * speed_ms**2

    def cruise_thrust_n(self, throttle: float, speed_ms: float) -> float:
        """Available thrust: static (momentum theory) blending into the power limit eta*P/V."""
        power_limited: float = PROP_EFFICIENCY * self.cruise_shaft_power_w / max(speed_ms, 1e-6)
        full: float = 1.0 / math.sqrt(self.cruise_static_thrust_n**-2 + power_limited**-2)
        return min(max(throttle, 0.0), 1.0) * full

    def cruise_lift_coefficient(self, speed_ms: float) -> float:
        return cruise_lift_coefficient(self.mass_kg, self.rho_kgm3, speed_ms, self.wing_area_m2)

    def drag_n(self, speed_ms: float, cl: float) -> float:
        return self.dynamic_pressure(speed_ms) * self.wing_area_m2 * (self.aero.cd0 + self.aero.k_induced * cl**2)

    def trim_alpha_rad(self, speed_ms: float) -> float:
        return (self.cruise_lift_coefficient(speed_ms) - self.aero.cl0) / self.aero.cl_alpha


def _helmbold_lift_slope(aspect: float) -> float:
    """Finite-wing lift-curve slope per rad (Helmbold, low sweep)."""
    return 2 * math.pi * aspect / (2 + math.sqrt(aspect**2 + 4))


def _damping(sizing: SizingOutput, aero: AeroStabilityOutput, cl_cruise: float) -> dict[str, float]:
    """Strip-theory and tail-volume damping derivatives (Nelson / DATCOM approximations)."""
    s, b, c, arm = sizing.wing_area_m2, sizing.wingspan_m, sizing.mean_chord_m, sizing.tail_arm_m
    a_h: float = _helmbold_lift_slope(HTAIL_ASPECT_RATIO)
    a_v: float = _helmbold_lift_slope(VTAIL_EFFECTIVE_ASPECT)
    v_h: float = sizing.horizontal_tail_area_m2 * arm / (s * c)
    v_v: float = sizing.vertical_tail_area_m2 * arm / (s * b)
    taper: float = WING_TAPER_RATIO
    return {
        "cl_q": 2 * TAIL_EFFICIENCY * a_h * v_h,
        "cl_delta_e": -aero.cm_delta_e_per_rad * c / arm,  # tail lift that produces Cm_delta_e
        "cm_q": -2 * TAIL_EFFICIENCY * a_h * v_h * (arm / c) * FUSELAGE_DAMPING_FACTOR,
        "cl_p": -(aero.cl_alpha_per_rad / 12) * (1 + 3 * taper) / (1 + taper),
        "cn_r": -2 * TAIL_EFFICIENCY * a_v * v_v * (arm / b) - aero.cd0 / 4,
        "cy_beta": -TAIL_EFFICIENCY * a_v * sizing.vertical_tail_area_m2 / s,
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


def _static_thrust_n(shaft_power_w: float, diameter_m: float, rho: float) -> float:
    """Momentum theory: T = (FM * P)^(2/3) * (2 rho A)^(1/3)."""
    area: float = math.pi * diameter_m**2 / 4
    return (PROP_FIGURE_OF_MERIT * shaft_power_w) ** (2 / 3) * (2 * rho * area) ** (1 / 3)


def _lift_rotors(sizing: SizingOutput, bom: SelectedBOM, aero: AeroStabilityOutput,
                 cad: CADArtifacts) -> tuple[LiftRotor, ...]:
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
                                max_thrust_n=bom.vtol_motor.max_thrust_n, diameter_m=bom.vtol_propeller.diameter_m))
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


def build_flight_model(req: OperationalRequirements, sizing: SizingOutput, bom: SelectedBOM,
                       aero: AeroStabilityOutput, cad: CADArtifacts) -> FlightModel:
    """Assemble the flight model; raises FlightModelError if the design cannot hover/accelerate."""
    rho, _ = air_properties(SITL_ALTITUDE_M)
    cl_cruise: float = cruise_lift_coefficient(cad.as_built_mass_kg, rho, req.cruise_speed_ms, sizing.wing_area_m2)
    shaft_w: float = min(bom.cruise_motor.max_power_w, bom.cruise_propeller.max_power_w)
    model = FlightModel(
        mass_kg=cad.as_built_mass_kg, ixx_kgm2=cad.ixx_kgm2, iyy_kgm2=cad.iyy_kgm2, izz_kgm2=cad.izz_kgm2,
        ixz_kgm2=cad.ixz_kgm2, x_cg_m=cad.x_cg_m, static_margin=cad.static_margin,
        wing_area_m2=sizing.wing_area_m2, wingspan_m=sizing.wingspan_m,
        mean_chord_m=sizing.mean_chord_m, fuselage_length_m=sizing.fuselage_length_m,
        fuselage_diameter_m=sizing.fuselage_diameter_m, tail_arm_m=sizing.tail_arm_m,
        aero=_coefficients(sizing, aero, cad, cl_cruise),
        v_stall_ms=sizing.v_stall_ms, v_cruise_ms=req.cruise_speed_ms, v_ne_ms=VNE_TO_CRUISE * req.cruise_speed_ms,
        rho_kgm3=rho, cruise_static_thrust_n=_static_thrust_n(shaft_w, bom.cruise_propeller.diameter_m, rho),
        cruise_shaft_power_w=shaft_w, cruise_prop_diameter_m=bom.cruise_propeller.diameter_m,
        cruise_motor_lag_s=motor_lag_s(bom.cruise_propeller.diameter_m),
        lift_rotors=_lift_rotors(sizing, bom, aero, cad),
        lift_motor_lag_s=motor_lag_s(bom.vtol_propeller.diameter_m) if bom.vtol_propeller else 0.0,
        surfaces=_model_surfaces(aero),
    )
    _check_flyable(model)
    return model


def _check_flyable(model: FlightModel) -> None:
    a = model.aero
    positive = {"mass": model.mass_kg, "Ixx": model.ixx_kgm2, "Iyy": model.iyy_kgm2, "Izz": model.izz_kgm2,
                "CL_alpha": a.cl_alpha, "Cl_delta_a": a.cl_delta_a, "-Cm_delta_e": -a.cm_delta_e,
                "-Cl_p": -a.cl_p, "-Cm_q": -a.cm_q, "shaft power": model.cruise_shaft_power_w}
    bad = {name: value for name, value in positive.items() if not (math.isfinite(value) and value > 0)}
    if bad:
        raise FlightModelError(f"flight model has non-physical values (expected finite and correctly signed): {bad}")
    cruise_drag: float = model.drag_n(model.v_cruise_ms, model.cruise_lift_coefficient(model.v_cruise_ms))
    if model.cruise_thrust_n(1.0, model.v_cruise_ms) <= cruise_drag:
        raise FlightModelError(f"cruise propeller cannot overcome {cruise_drag:.1f} N drag at {model.v_cruise_ms} m/s")
    if model.is_quadplane and sum(r.max_thrust_n for r in model.lift_rotors) <= model.weight_n:
        raise FlightModelError("lift rotors cannot lift the as-built aircraft")


def transition_time_s(model: FlightModel) -> float:
    """Time for the cruise motor alone to accelerate 0 -> 1.2 Vs against parasite drag (rotors carry
    the weight, so the wing is unloaded). Explicit Euler, deterministic step."""
    target: float = TRANSITION_SPEED_FACTOR * model.v_stall_ms
    speed, elapsed = 0.0, 0.0
    while speed < target:
        net: float = model.cruise_thrust_n(1.0, speed) - model.drag_n(speed, 0.0)
        if net <= 0 or elapsed > MAX_TRANSITION_S:
            raise FlightModelError(f"cannot reach the transition speed {target:.1f} m/s (stuck at {speed:.1f} m/s)")
        speed += net / model.mass_kg * TRANSITION_DT_S
        elapsed += TRANSITION_DT_S
    return elapsed
