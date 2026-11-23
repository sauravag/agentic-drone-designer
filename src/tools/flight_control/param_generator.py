"""ArduPilot Plane / QuadPlane parameter file from the flight model (SRS 4.4 Module D.1).

Names follow ArduPilot Plane 4.7 (the SITL binary in the Docker image). The SRS uses older names;
they map as: ARSPD_FBW_MIN/MAX -> AIRSPEED_MIN/MAX, TRIM_ARSPD_CM (cm/s) -> AIRSPEED_CRUISE (m/s),
RLL2SRV_P/I/D/FF -> RLL_RATE_P/I/D/FF, PTCH2SRV_* -> PTCH_RATE_* (the *2SRV_TCONST names remain).

Gain design (all at the cruise speed, which becomes SCALING_SPEED so ArduPilot's speed scaler is 1):
* Roll feed-forward = steady-state aileron per unit roll rate = -L_p / L_delta_a, with
  L_delta_a = q S b Cl_delta_a / Ixx (SRS) and K_ff = -L_p the roll-damping rate. Pitch likewise from
  M_delta_e = q S c Cm_delta_e / Iyy, including the alpha change a steady pull-up needs.
  ArduPilot's rate-loop output is +-45 "degrees" for full surface throw, so FF[1/(deg/s)] =
  deflection-per-rate * 45 / max_deflection_deg.
* Rate P places the closed-loop rate pole at w = max(5 rad/s, 1.5 |L_p|): P = (w + L_p) / L_delta_a, so
  it scales inversely with the control-power-to-inertia ratio. In a steady hold the angle loop acts like
  an angle PI controller with proportional gain (FF + P) / tau and integral gain I / tau, so
  I = (FF + P) / (3 tau) puts the PI zero at a third of the angle bandwidth: it removes trim and
  speed-change (ramp) disturbances without adding overshoot. D = P * servo lag.
* QuadPlane rate P = w_q * Ixx / (2 L_arm T_max) (SRS: proportional to Ixx / (L_arm T_max)); the
  factor 2 is ArduPilot's normalised X-frame mix (4 motors x 0.5 factor).
"""

import math
from pathlib import Path
from typing import Final, Mapping, Optional

from src.schemas.components import SelectedBOM
from src.schemas.state import SizingOutput
from src.tools.flight_control.control_channels import (LIFT_MOTOR_SERVOS, THROTTLE_SERVO,
                                                       THROTTLE_SERVO_FUNCTION)
from src.tools.flight_control.flight_model import (ROTOR_TORQUE_PER_THRUST_DIAMETER, SERVO_TIME_CONSTANT_S,
                                                   THRUST_EXPO, FlightModel, transition_time_s)

ParamSet = dict[str, float]

AP_FULL_SCALE_DEG: Final[float] = 45.0  # rate-controller output for full surface deflection
STALL_TO_MIN_AIRSPEED: Final[float] = 1.25  # SRS
NE_TO_MAX_AIRSPEED: Final[float] = 0.85  # SRS
ASSIST_TO_STALL: Final[float] = 1.15  # SRS Q_ASSIST_SPEED
ROTATE_TO_STALL: Final[float] = 1.1  # take-off rotation speed
ANGLE_TIME_CONSTANT_S: Final[float] = 0.35  # RLL2SRV/PTCH2SRV_TCONST
RATE_BANDWIDTH_RAD_S: Final[float] = 5.0  # minimum closed-loop rate bandwidth, ~1/(4 servo lags)
RATE_BANDWIDTH_TO_DAMPING: Final[float] = 1.5  # rate pole at 1.5x the axis's natural damping rate
FF_BOUNDS: Final[tuple[float, float]] = (0.15, 1.5)  # SRS clamp on RLL/PTCH feed-forward
P_FLOOR_TO_FF: Final[float] = 0.25  # minimum rate P as a fraction of FF (ArduPilot defaults ~0.23)
INTEGRAL_CORNER_FRACTION: Final[float] = 1.0 / 3.0  # PI zero at 1/3 of the angle-loop bandwidth 1/tau
Q_BANDWIDTH_MAX_RAD_S: Final[float] = 18.0
Q_BANDWIDTH_LAG_FACTOR: Final[float] = 0.4  # rate bandwidth <= 0.4 / motor lag
Q_YAW_BANDWIDTH_FRACTION: Final[float] = 0.25
Q_D_TO_P: Final[float] = 0.015
Q_YAW_I_TO_P: Final[float] = 0.1
Q_RATE_P_BOUNDS: Final[tuple[float, float]] = (0.01, 1.0)
Q_ACCEL_AUTHORITY: Final[float] = 0.25  # fraction of the full differential-thrust angular acceleration
Q_ACCEL_BOUNDS_CDEG: Final[tuple[float, float]] = (3_000.0, 180_000.0)
Q_HOVER_BOUNDS: Final[tuple[float, float]] = (0.125, 0.6875)  # ArduPilot Q_M_THST_HOVER range
MOTOR_SPIN_MIN: Final[float] = 0.15  # Q_M_SPIN_MIN / MAX; the SITL plant uses the same mapping
MOTOR_SPIN_MAX: Final[float] = 0.95
TRIM_THROTTLE_BOUNDS: Final[tuple[float, float]] = (10.0, 90.0)
ROLL_LIMIT_DEG: Final[float] = 45.0
PITCH_LIMIT_MAX_DEG: Final[float] = 20.0
PITCH_LIMIT_MIN_DEG: Final[float] = -20.0
ELECTRIC_CRITICAL_TO_LOW: Final[float] = 0.5  # BATT_CRT_MAH = half the low-battery reserve
HYBRID_BUFFER_LOW_FRACTION: Final[float] = 0.30  # hybrid: keep enough buffer for a VTOL landing
HYBRID_BUFFER_CRITICAL_FRACTION: Final[float] = 0.15
MAH_PER_AH: Final[float] = 1000.0
SITL_TAKEOFF_ALT_M: Final[float] = 60.0
QUAD_SERVO_NUMBERS: Final[tuple[int, ...]] = LIFT_MOTOR_SERVOS  # outputs reserved for lift motors 1-4
QUAD_MOTOR_FIRST_FUNCTION: Final[int] = 33
QUAD_SERVO_FUNCTIONS: Final[dict[str, int]] = {
    f"SERVO{servo}_FUNCTION": QUAD_MOTOR_FIRST_FUNCTION + i for i, servo in enumerate(QUAD_SERVO_NUMBERS)}


class ParamGenerationError(ValueError):
    """The flight model gives an inconsistent envelope (e.g. cruise below the minimum airspeed)."""


def _to_ap_units(deflection_per_rate: float, max_deflection_rad: float) -> float:
    return deflection_per_rate * AP_FULL_SCALE_DEG / math.degrees(max_deflection_rad)


def _rate_gains(prefix: str, ff_phys: float, damping: float, control_power: float, max_defl: float,
                gain_scale: float) -> ParamSet:
    """FF/P/I/D for one fixed-wing axis. damping (1/s, <0) and control_power (rad/s^2 per rad).
    gain_scale multiplies P and I; D follows it down but never rises above its model value: on the high-control-power
    corner a raised D drives a ~13 Hz rate-loop oscillation through the 50 Hz loop and the servo lag."""
    lo, hi = FF_BOUNDS
    ff: float = min(max(_to_ap_units(ff_phys, max_defl), lo), hi)
    bandwidth: float = max(RATE_BANDWIDTH_RAD_S, RATE_BANDWIDTH_TO_DAMPING * abs(damping))
    p_phys: float = (bandwidth + damping) / control_power
    p: float = max(_to_ap_units(p_phys, max_defl), P_FLOOR_TO_FF * ff)
    i: float = (ff + p) * INTEGRAL_CORNER_FRACTION / ANGLE_TIME_CONSTANT_S
    return {f"{prefix}_FF": ff, f"{prefix}_P": gain_scale * p, f"{prefix}_I": gain_scale * i,
            f"{prefix}_D": min(gain_scale, 1.0) * p * SERVO_TIME_CONSTANT_S}


def fixed_wing_gains(model: FlightModel, gain_scale: float = 1.0) -> ParamSet:
    a, v = model.aero, model.v_cruise_ms
    qs: float = model.dynamic_pressure(v) * model.wing_area_m2
    b, c = model.wingspan_m, model.mean_chord_m
    l_da: float = qs * b * a.cl_delta_a / model.ixx_kgm2
    l_p: float = qs * b**2 * a.cl_p / (2 * v * model.ixx_kgm2)
    m_de: float = abs(qs * c * a.cm_delta_e / model.iyy_kgm2)
    m_q: float = qs * c**2 * a.cm_q / (2 * v * model.iyy_kgm2)
    alpha_per_q: float = model.mass_kg * v / (qs * a.cl_alpha)  # steady pull-up: dn = V q / g
    ff_pitch: float = abs((a.cm_alpha * alpha_per_q + a.cm_q * c / (2 * v)) / a.cm_delta_e)
    gains: ParamSet = {"RLL2SRV_TCONST": ANGLE_TIME_CONSTANT_S, "PTCH2SRV_TCONST": ANGLE_TIME_CONSTANT_S}
    max_defl: dict[str, float] = {c.surface: c.max_deflection_rad for c in model.channels}  # the JSBSim limits
    missing: list[str] = [name for name in ("aileron", "elevator") if name not in max_defl]
    if missing:
        raise ParamGenerationError(f"rate gains need an aileron and an elevator channel; missing {missing}")
    gains |= _rate_gains("RLL_RATE", -l_p / l_da, l_p, l_da, max_defl["aileron"], gain_scale)
    gains |= _rate_gains("PTCH_RATE", ff_pitch, m_q, m_de, max_defl["elevator"], gain_scale)
    return gains


def hover_throttle(model: FlightModel) -> float:
    """Q_M_THST_HOVER is in ArduPilot's linearised-thrust units (MOT_THST_EXPO is applied afterwards),
    so it is simply the hover thrust fraction W / sum(T_max)."""
    fraction: float = model.weight_n / sum(r.max_thrust_n for r in model.lift_rotors)
    lo, hi = Q_HOVER_BOUNDS
    return min(max(fraction, lo), hi)


def _q_axis(name: str, inertia: float, torque_per_unit: float, bandwidth: float, gain_scale: float) -> ParamSet:
    lo, hi = Q_RATE_P_BOUNDS
    p: float = min(max(gain_scale * bandwidth * inertia / torque_per_unit, lo), hi)  # clamp after scaling
    accel_cdeg: float = Q_ACCEL_AUTHORITY * torque_per_unit / inertia * math.degrees(1) * 100
    a_lo, a_hi = Q_ACCEL_BOUNDS_CDEG
    is_yaw: bool = name == "YAW"
    gains: ParamSet = {f"Q_A_RAT_{name}_P": p, f"Q_A_RAT_{name}_I": p * (Q_YAW_I_TO_P if is_yaw else 1.0),
                       f"Q_A_RAT_{name}_D": 0.0 if is_yaw else p * Q_D_TO_P}
    gains[f"Q_A_ACC_{name[0]}_MAX"] = min(max(accel_cdeg, a_lo), a_hi)
    return gains


def quadplane_params(model: FlightModel, gain_scale: float = 1.0) -> ParamSet:
    rotors = model.lift_rotors
    t_max: float = rotors[0].max_thrust_n
    arm_roll: float = sum(abs(r.y_m) for r in rotors) / len(rotors)
    arm_pitch: float = sum(abs(r.x_m - model.x_cg_m) for r in rotors) / len(rotors)
    yaw_arm: float = ROTOR_TORQUE_PER_THRUST_DIAMETER * rotors[0].diameter_m
    bandwidth: float = min(Q_BANDWIDTH_MAX_RAD_S, Q_BANDWIDTH_LAG_FACTOR / model.lift_motor_lag_s)
    params: ParamSet = {
        "Q_ENABLE": 1, "Q_FRAME_CLASS": 1, "Q_FRAME_TYPE": 1,
        "Q_ASSIST_SPEED": round(ASSIST_TO_STALL * model.v_stall_ms, 1),
        "Q_TRANSITION_MS": int(round(1000 * transition_time_s(model))),
        "Q_M_THST_EXPO": THRUST_EXPO, "Q_M_THST_HOVER": round(hover_throttle(model), 4),
        "Q_M_SPIN_MIN": MOTOR_SPIN_MIN, "Q_M_SPIN_MAX": MOTOR_SPIN_MAX, **QUAD_SERVO_FUNCTIONS,
    }
    params |= _q_axis("RLL", model.ixx_kgm2, 2 * arm_roll * t_max, bandwidth, gain_scale)
    params |= _q_axis("PIT", model.iyy_kgm2, 2 * arm_pitch * t_max, bandwidth, gain_scale)
    params |= _q_axis("YAW", model.izz_kgm2, 2 * yaw_arm * t_max, Q_YAW_BANDWIDTH_FRACTION * bandwidth, gain_scale)
    return params


def envelope_params(model: FlightModel) -> ParamSet:
    a_min: float = round(STALL_TO_MIN_AIRSPEED * model.v_stall_ms, 1)
    a_max: float = round(NE_TO_MAX_AIRSPEED * model.v_ne_ms, 1)
    if not a_min < model.v_cruise_ms < a_max:
        raise ParamGenerationError(f"cruise {model.v_cruise_ms} m/s outside [{a_min}, {a_max}] (1.25 Vs, 0.85 VNE)")
    drag: float = model.drag_n(model.v_cruise_ms, model.cruise_lift_coefficient(model.v_cruise_ms))
    trim_pct: float = 100 * drag / model.cruise_thrust_n(1.0, model.v_cruise_ms)
    lo, hi = TRIM_THROTTLE_BOUNDS
    params: ParamSet = {
        "AIRSPEED_MIN": a_min, "AIRSPEED_MAX": a_max, "AIRSPEED_CRUISE": round(model.v_cruise_ms, 1),
        "AIRSPEED_STALL": round(model.v_stall_ms, 1), "SCALING_SPEED": round(model.v_cruise_ms, 1),
        "ARSPD_USE": 1, "TRIM_THROTTLE": int(round(min(max(trim_pct, lo), hi))), "THR_MAX": 100,
        "ROLL_LIMIT_DEG": ROLL_LIMIT_DEG, "PTCH_LIM_MAX_DEG": PITCH_LIMIT_MAX_DEG,
        "PTCH_LIM_MIN_DEG": PITCH_LIMIT_MIN_DEG, **fw_servo_functions(model),
    }
    if model.launch != "hand":  # a hand-launched aircraft has no take-off roll to configure
        params |= {"TKOFF_ROTATE_SPD": round(ROTATE_TO_STALL * model.v_stall_ms, 1), "TKOFF_THR_MAX": 100}
    return params


def fw_servo_functions(model: FlightModel) -> ParamSet:
    """SERVOn_FUNCTION for every surface channel plus the cruise throttle, in servo order. Every output number
    must be unique across the surfaces, the throttle and, only when the model has lift motors, the QuadPlane motor
    outputs 5-8, or one function would silently overwrite another."""
    reserved: tuple[int, ...] = QUAD_SERVO_NUMBERS if model.is_quadplane else ()
    servos: list[int] = [*(c.servo for c in model.channels), THROTTLE_SERVO, *reserved]
    clashes: list[int] = sorted({servo for servo in servos if servos.count(servo) > 1})
    if clashes:
        raise ValueError(f"servo output(s) {clashes} are assigned to more than one function")
    functions: dict[int, int] = {c.servo: c.servo_function for c in model.channels}
    functions[THROTTLE_SERVO] = THROTTLE_SERVO_FUNCTION
    return {f"SERVO{servo}_FUNCTION": functions[servo] for servo in sorted(functions)}


def battery_params(sizing: SizingOutput, bom: SelectedBOM, is_quadplane: bool) -> ParamSet:
    capacity_mah: float = bom.battery.capacity_wh / bom.battery.voltage_v * MAH_PER_AH
    if sizing.propulsion_type == "series_hybrid":
        low, critical = HYBRID_BUFFER_LOW_FRACTION, HYBRID_BUFFER_CRITICAL_FRACTION
    else:
        low, critical = sizing.energy_reserve_fraction, ELECTRIC_CRITICAL_TO_LOW * sizing.energy_reserve_fraction
    return {"BATT_MONITOR": 4, "BATT_CAPACITY": int(round(capacity_mah)),
            "BATT_LOW_MAH": int(round(low * capacity_mah)), "BATT_CRT_MAH": int(round(critical * capacity_mah)),
            "BATT_FS_LOW_ACT": 1,  # RTL
            "BATT_FS_CRT_ACT": 4 if is_quadplane else 2}  # QLand / Land


def generate_params(model: FlightModel, sizing: SizingOutput, bom: SelectedBOM, gain_scale: float = 1.0) -> ParamSet:
    """Complete vehicle parameter set. gain_scale multiplies the rate P/I gains (SITL damping retry); the fixed-wing
    rate D is only ever scaled down."""
    if gain_scale <= 0:
        raise ParamGenerationError(f"gain_scale must be positive, got {gain_scale}")
    params: ParamSet = envelope_params(model) | fixed_wing_gains(model, gain_scale)
    if model.is_quadplane:
        params |= quadplane_params(model, gain_scale)
    return params | battery_params(sizing, bom, model.is_quadplane)


def sitl_overrides() -> ParamSet:
    """Simulation-only settings layered on top of the vehicle file (never part of ardupilot.param)."""
    rc: ParamSet = {}
    for ch in range(1, 5):  # RC sticks and surface servos calibrated to the bridge's 1000-2000 us map
        trim: int = 1000 if ch == 3 else 1500
        rc |= {f"RC{ch}_MIN": 1000, f"RC{ch}_MAX": 2000, f"RC{ch}_TRIM": trim,
               f"SERVO{ch}_MIN": 1000, f"SERVO{ch}_MAX": 2000, f"SERVO{ch}_TRIM": trim}
    return rc | {
        "BATT_MONITOR": 0,  # the JSON physics backend reports no battery
        "THR_FAILSAFE": 0, "RC_OVERRIDE_TIME": -1, "FS_GCS_ENABL": 0,
        "LOG_DISARMED": 0, "TKOFF_ALT": SITL_TAKEOFF_ALT_M,
        "GCS_PID_MASK": 3,  # stream roll+pitch PID_TUNING for the report
        "STAB_PITCH_DOWN": 0,  # FBWA stick maps exactly onto the pitch demand during the doublets
        "FLTMODE_CH": 16,  # park the RC mode switch on an unused channel; the GCS commands modes
    }


def _format(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.6g}"


def write_param_file(params: Mapping[str, float], path: Path, header: Optional[str] = None) -> Path:
    """MAVProxy/SITL format ("NAME VALUE"), sorted for reproducible diffs."""
    bad: list[str] = [name for name, value in params.items() if not math.isfinite(value)]
    if bad:
        raise ParamGenerationError(f"non-finite parameter values: {bad}")
    lines: list[str] = [f"# {line}" for line in (header or "").splitlines()]
    lines += [f"{name} {_format(value)}" for name, value in sorted(params.items())]
    path.write_text("\n".join(lines) + "\n")
    return path


def read_param_file(path: Path) -> ParamSet:
    params: ParamSet = {}
    for raw in path.read_text().splitlines():
        line: str = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name, value = line.replace(",", " ").split()[:2]
        params[name] = float(value)
    return params
