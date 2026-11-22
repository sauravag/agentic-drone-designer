"""JSBSim aircraft configuration from the flight model (SRS 4.4 Module D.2).

Writes <root>/aircraft/<name>/<name>.xml with:
* metrics (wing area/span/MAC) and the CG as the aero reference point (moments are about the CG);
* mass_balance from the Build123d inertia tensor;
* a 3-point landing gear (steerable nose wheel) sized so the cruise propeller clears the ground;
* actuators: first-order servo and motor lags (servo time constant, motor lags from the BOM);
* propulsion thrust tables: cruise thrust vs (throttle, airspeed) and lift-rotor thrust vs motor
  output using ArduPilot's spin-range/expo mapping, applied as external forces plus rotor torques;
* the aerodynamic build-up (lift with a stall break, drag polar, side force, roll/pitch/yaw moments).

Inputs the SITL bridge drives: fcs/{aileron,elevator,rudder}-cmd-norm (+-1, positive = right roll /
trailing edge down / trailing edge left, i.e. the conventional delta_a, delta_e, delta_r signs),
fcs/steer-cmd-norm, fcs/cruise-throttle-cmd and fcs/lift<N>-cmd for motor N (ArduPilot output 0..1).
XML is built with ElementTree, so no value is ever spliced into markup.
"""

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

from src.tools.flight_control.control_channels import ControlChannel
from src.tools.flight_control.flight_model import (ROTOR_TORQUE_PER_THRUST_DIAMETER, SERVO_TIME_CONSTANT_S,
                                                   THRUST_EXPO, FlightModel)
from src.tools.flight_control.param_generator import MOTOR_SPIN_MAX, MOTOR_SPIN_MIN
from src.tools.gear import GearContact, tricycle_gear

N_TO_LBF: Final[float] = 0.224809
M_TO_FT: Final[float] = 3.28084
STALL_DROP_DEG: Final[float] = 10.0  # alpha range over which CL falls after the stall
POST_STALL_CL_FRACTION: Final[float] = 0.7
ALPHA_TABLE_LIMIT_DEG: Final[float] = 30.0
THROTTLE_BREAKPOINTS: Final[tuple[float, ...]] = (0.0, 0.25, 0.5, 0.75, 1.0)
AIRSPEED_BREAKPOINTS: Final[int] = 8
AIRSPEED_TABLE_OVER_VNE: Final[float] = 1.2
LIFT_THROTTLE_BREAKPOINTS: Final[int] = 11
GEAR_STATIC_DEFLECTION_M: Final[float] = 0.02
GEAR_DAMPING_RATIO: Final[float] = 0.6
NOSE_STEER_MAX_DEG: Final[float] = 20.0
STATIC_FRICTION: Final[float] = 0.8
DYNAMIC_FRICTION: Final[float] = 0.5
ROLLING_FRICTION: Final[float] = 0.02


@dataclass(frozen=True)
class JSBSimModel:
    root_dir: Path  # JSBSim root; the aircraft lives in root_dir/aircraft/<name>
    name: str
    xml_path: Path
    rest_height_m: float  # CG height above the gear contact points
    lift_motor_numbers: tuple[int, ...]
    channels: tuple[ControlChannel, ...]  # the surfaces the bridge drives, one servo each


def _sub(parent: ET.Element, tag: str, text: object = None, **attrs: str) -> ET.Element:
    node = ET.SubElement(parent, tag, attrs)
    if text is not None:
        node.text = f" {text:.6g} " if isinstance(text, float) else f" {text} "
    return node


def _vector(parent: ET.Element, tag: str, values: Sequence[float], **attrs: str) -> None:
    node = _sub(parent, tag, **attrs)
    for axis, value in zip("xyz", values):
        _sub(node, axis, float(value))


def _table(parent: ET.Element, row_prop: str, rows: Sequence[float], values: Sequence[float]) -> None:
    table = _sub(parent, "table")
    _sub(table, "independentVar", row_prop, lookup="row")
    _sub(table, "tableData").text = "\n" + "\n".join(f"{r:.6g}\t{v:.6g}" for r, v in zip(rows, values)) + "\n"


def _table_2d(parent: ET.Element, row_prop: str, rows: Sequence[float], col_prop: str, cols: Sequence[float],
              grid: Sequence[Sequence[float]]) -> None:
    table = _sub(parent, "table")
    _sub(table, "independentVar", row_prop, lookup="row")
    _sub(table, "independentVar", col_prop, lookup="column")
    header: str = "\t" + "\t".join(f"{c:.6g}" for c in cols)
    body: str = "\n".join(f"{r:.6g}\t" + "\t".join(f"{v:.6g}" for v in line) for r, line in zip(rows, grid))
    _sub(table, "tableData").text = f"\n{header}\n{body}\n"


def _ground_reactions(root: ET.Element, model: FlightModel, contacts: Sequence[GearContact]) -> None:
    gear = _sub(root, "ground_reactions")
    spring: float = model.weight_n / len(contacts) / GEAR_STATIC_DEFLECTION_M
    damping: float = 2 * GEAR_DAMPING_RATIO * math.sqrt(spring * model.mass_kg / len(contacts))
    for c in contacts:
        node = _sub(gear, "contact", type="BOGEY", name=c.name)
        _vector(node, "location", (c.x_m, c.y_m, c.z_m), unit="M")
        _sub(node, "static_friction", STATIC_FRICTION)
        _sub(node, "dynamic_friction", DYNAMIC_FRICTION)
        _sub(node, "rolling_friction", ROLLING_FRICTION)
        _sub(node, "spring_coeff", spring, unit="N/M")
        _sub(node, "damping_coeff", damping, unit="N/M/SEC")
        _sub(node, "max_steer", NOSE_STEER_MAX_DEG if c.steerable else 0.0, unit="DEG")
        _sub(node, "brake_group", "NONE")
        _sub(node, "retractable", 0)


def _metrics_and_mass(root: ET.Element, model: FlightModel) -> None:
    metrics = _sub(root, "metrics")
    _sub(metrics, "wingarea", model.wing_area_m2, unit="M2")
    _sub(metrics, "wingspan", model.wingspan_m, unit="M")
    _sub(metrics, "chord", model.mean_chord_m, unit="M")
    for name, x in (("AERORP", model.x_cg_m), ("EYEPOINT", model.x_cg_m), ("VRP", 0.0)):
        _vector(metrics, "location", (x, 0.0, 0.0), unit="M", name=name)
    mass = _sub(root, "mass_balance", negated_crossproduct_inertia="false")
    for tag, value in (("ixx", model.ixx_kgm2), ("iyy", model.iyy_kgm2), ("izz", model.izz_kgm2),
                       ("ixy", 0.0), ("ixz", model.ixz_kgm2), ("iyz", 0.0)):
        _sub(mass, tag, value, unit="KG*M2")
    _sub(mass, "emptywt", model.mass_kg, unit="KG")
    _vector(mass, "location", (model.x_cg_m, 0.0, 0.0), unit="M", name="CG")


def _actuator(channel: ET.Element, name: str, cmd: str, time_constant: float, gain: float) -> None:
    lag = _sub(channel, "lag_filter", name=f"fcs/{name}-lag")
    _sub(lag, "input", cmd)
    _sub(lag, "c1", 1.0 / time_constant)
    pure = _sub(channel, "pure_gain", name=f"fcs/{name}")
    _sub(pure, "input", f"fcs/{name}-lag")
    _sub(pure, "gain", gain)


def lift_thrust_curve(max_thrust_n: float) -> tuple[list[float], list[float]]:
    """Thrust (N) vs ArduPilot motor output: spin range [MIN, MAX] -> 0..1, then (1-e) u + e u^2."""
    outputs = [i / (LIFT_THROTTLE_BREAKPOINTS - 1) for i in range(LIFT_THROTTLE_BREAKPOINTS)]
    thrusts: list[float] = []
    for out in outputs:
        u: float = min(max((out - MOTOR_SPIN_MIN) / (MOTOR_SPIN_MAX - MOTOR_SPIN_MIN), 0.0), 1.0)
        thrusts.append(max_thrust_n * ((1 - THRUST_EXPO) * u + THRUST_EXPO * u * u))
    return outputs, thrusts


def _cruise_propulsion(channel: ET.Element, model: FlightModel) -> None:
    _actuator(channel, "cruise-throttle", "fcs/cruise-throttle-cmd", model.cruise_motor_lag_s, 1.0)
    speeds = [AIRSPEED_TABLE_OVER_VNE * model.v_ne_ms * i / (AIRSPEED_BREAKPOINTS - 1)
              for i in range(AIRSPEED_BREAKPOINTS)]
    grid = [[model.cruise_thrust_n(t, v) * N_TO_LBF for v in speeds] for t in THROTTLE_BREAKPOINTS]
    fn = _sub(channel, "fcs_function", name="propulsion/cruise-thrust-lbs")
    _table_2d(_sub(fn, "function"), "fcs/cruise-throttle", THROTTLE_BREAKPOINTS, "velocities/vt-fps",
              [v * M_TO_FT for v in speeds], grid)
    _sub(fn, "output", "external_reactions/cruise/magnitude")


def _lift_propulsion(channel: ET.Element, model: FlightModel) -> None:
    for rotor in model.lift_rotors:
        n: int = rotor.motor_number
        _actuator(channel, f"lift{n}", f"fcs/lift{n}-cmd", model.lift_motor_lag_s, 1.0)
        outputs, thrusts = lift_thrust_curve(rotor.max_thrust_n)
        fn = _sub(channel, "fcs_function", name=f"propulsion/lift{n}-thrust-lbs")
        _table(_sub(fn, "function"), f"fcs/lift{n}", outputs, [t * N_TO_LBF for t in thrusts])
        _sub(fn, "output", f"external_reactions/lift{n}/magnitude")
        torque = _sub(channel, "pure_gain", name=f"propulsion/lift{n}-torque-lbsft")
        _sub(torque, "input", f"propulsion/lift{n}-thrust-lbs")
        _sub(torque, "gain", rotor.spin * ROTOR_TORQUE_PER_THRUST_DIAMETER * rotor.diameter_m * M_TO_FT)
        _sub(torque, "output", f"external_reactions/lift{n}-torque/magnitude-lbsft")


def _flight_control(root: ET.Element, model: FlightModel) -> None:
    fcs = _sub(root, "flight_control", name="isr_forge")
    for prop in ["fcs/cruise-throttle-cmd", *(f"fcs/lift{r.motor_number}-cmd" for r in model.lift_rotors)]:
        _sub(fcs, "property", prop, value="0")
    surfaces = _sub(fcs, "channel", name="surfaces")
    for c in model.channels:
        _actuator(surfaces, f"{c.surface}-pos-rad", f"fcs/{c.surface}-cmd-norm", SERVO_TIME_CONSTANT_S,
                  c.max_deflection_rad)
    propulsion = _sub(fcs, "channel", name="propulsion")
    _cruise_propulsion(propulsion, model)
    _lift_propulsion(propulsion, model)


def _external_reactions(root: ET.Element, model: FlightModel) -> None:
    ext = _sub(root, "external_reactions")
    cruise = _sub(ext, "force", name="cruise", frame="BODY", unit="LBS")
    _vector(cruise, "location", (0.0, 0.0, 0.0), unit="M")
    _vector(cruise, "direction", (1.0, 0.0, 0.0))
    for r in model.lift_rotors:
        force = _sub(ext, "force", name=f"lift{r.motor_number}", frame="BODY", unit="LBS")
        _vector(force, "location", (r.x_m, r.y_m, r.z_m), unit="M")
        _vector(force, "direction", (0.0, 0.0, -1.0))
        moment = _sub(ext, "moment", name=f"lift{r.motor_number}-torque", frame="BODY", unit="LBSFT")
        _vector(moment, "direction", (0.0, 0.0, 1.0))


def _coefficient(axis: ET.Element, name: str, value: float, *props: str) -> None:
    product = _sub(_sub(axis, "function", name=f"aero/coefficient/{name}"), "product")
    for prop in ("aero/qbar-psf", "metrics/Sw-sqft", *props):
        _sub(product, "property", prop)
    _sub(product, "value", value)


def lift_table(model: FlightModel) -> tuple[list[float], list[float]]:
    """CL vs alpha (rad): linear to +-CLmax, then a post-stall drop."""
    a = model.aero
    alpha_stall: float = (a.cl_max - a.cl0) / a.cl_alpha
    alpha_neg_stall: float = (-a.cl_max - a.cl0) / a.cl_alpha
    drop: float = math.radians(STALL_DROP_DEG)
    limit: float = math.radians(ALPHA_TABLE_LIMIT_DEG)
    post: float = POST_STALL_CL_FRACTION * a.cl_max
    alphas = [-limit, alpha_neg_stall - drop, alpha_neg_stall, alpha_stall, alpha_stall + drop, limit]
    return alphas, [-post, -post, -a.cl_max, a.cl_max, post, post]


# (surface, derivative) -> JSBSim coefficient name and the axis it belongs to; the order is the XML order
SURFACE_COEFFICIENTS: Final[tuple[tuple[str, str, str, str], ...]] = (
    ("elevator", "cl", "CLde", "LIFT"), ("aileron", "cl_roll", "Clda", "ROLL"),
    ("elevator", "cm", "Cmde", "PITCH"), ("rudder", "cn", "Cndr", "YAW"), ("aileron", "cn", "Cnda", "YAW"))


def _require_mapped_derivatives(model: FlightModel) -> None:
    """A non-zero derivative with no coefficient name would be silently dropped from the plant."""
    mapped: set[tuple[str, str]] = {(surface, attr) for surface, attr, _, _ in SURFACE_COEFFICIENTS}
    for channel in model.channels:
        for attr, value in vars(channel.derivatives).items():
            if value != 0.0 and (channel.surface, attr) not in mapped:
                raise ValueError(f"{channel.surface} has a non-zero {attr} derivative but no JSBSim coefficient "
                                 "is mapped for it (add it to SURFACE_COEFFICIENTS)")


def _surface_terms(model: FlightModel, axis_name: str) -> tuple[tuple[str, float, tuple[str, ...]], ...]:
    """Control-surface coefficients of one axis: one per non-zero derivative of each channel."""
    _require_mapped_derivatives(model)
    by_surface = {c.surface: c for c in model.channels}
    return tuple((name, value, (f"fcs/{surface}-pos-rad",))
                 for surface, attr, name, axis in SURFACE_COEFFICIENTS if axis == axis_name and surface in by_surface
                 if (value := getattr(by_surface[surface].derivatives, attr)) != 0.0)


def _moment_axis(aero: ET.Element, axis_name: str, length_prop: str,
                 terms: Sequence[tuple[str, float, tuple[str, ...]]]) -> None:
    axis = _sub(aero, "axis", name=axis_name)
    for name, value, props in terms:
        _coefficient(axis, name, value, length_prop, *props)


def _aerodynamics(root: ET.Element, model: FlightModel) -> None:
    a = model.aero
    aero = _sub(root, "aerodynamics")
    lift = _sub(aero, "axis", name="LIFT")
    product = _sub(_sub(lift, "function", name="aero/coefficient/CLalpha"), "product")
    _sub(product, "property", "aero/qbar-psf")
    _sub(product, "property", "metrics/Sw-sqft")
    _table(product, "aero/alpha-rad", *lift_table(model))
    _coefficient(lift, "CLq", a.cl_q, "aero/ci2vel", "velocities/q-aero-rad_sec")
    for name, value, props in _surface_terms(model, "LIFT"):
        _coefficient(lift, name, value, *props)
    drag = _sub(aero, "axis", name="DRAG")
    _coefficient(drag, "CD0", a.cd0)
    _coefficient(drag, "CDi", a.k_induced, "aero/cl-squared")
    _coefficient(_sub(aero, "axis", name="SIDE"), "CYb", a.cy_beta, "aero/beta-rad")
    p, q, r = "velocities/p-aero-rad_sec", "velocities/q-aero-rad_sec", "velocities/r-aero-rad_sec"
    _moment_axis(aero, "ROLL", "metrics/bw-ft", (
        ("Clb", a.cl_beta, ("aero/beta-rad",)), ("Clp", a.cl_p, ("aero/bi2vel", p)),
        ("Clr", a.cl_r, ("aero/bi2vel", r)), *_surface_terms(model, "ROLL")))
    _moment_axis(aero, "PITCH", "metrics/cbarw-ft", (
        ("Cm0", a.cm0, ()), ("Cmalpha", a.cm_alpha, ("aero/alpha-rad",)),
        ("Cmq", a.cm_q, ("aero/ci2vel", q)), *_surface_terms(model, "PITCH")))
    _moment_axis(aero, "YAW", "metrics/bw-ft", (
        ("Cnb", a.cn_beta, ("aero/beta-rad",)), ("Cnp", a.cn_p, ("aero/bi2vel", p)),
        ("Cnr", a.cn_r, ("aero/bi2vel", r)), *_surface_terms(model, "YAW")))


def export_jsbsim(model: FlightModel, root_dir: Path, name: str = "isr_forge") -> JSBSimModel:
    """Write the aircraft XML under root_dir/aircraft/<name>/ and return how to load it."""
    if not name.replace("_", "").isalnum():
        raise ValueError(f"aircraft name must be alphanumeric/underscore, got {name!r}")
    root = ET.Element("fdm_config", name=name, version="2.0", release="ALPHA")
    _sub(_sub(root, "fileheader"), "description", "Generated by ISR-Forge (VSPAERO derivatives, CAD inertia)")
    contacts = tricycle_gear(model.fuselage_length_m, model.fuselage_diameter_m, model.cruise_prop_diameter_m,
                             model.mean_chord_m, model.wingspan_m, model.x_cg_m).contacts
    _metrics_and_mass(root, model)
    _ground_reactions(root, model, contacts)
    _sub(root, "propulsion")
    _flight_control(root, model)
    _external_reactions(root, model)
    _aerodynamics(root, model)
    aircraft_dir: Path = root_dir / "aircraft" / name
    aircraft_dir.mkdir(parents=True, exist_ok=True)
    xml_path: Path = aircraft_dir / f"{name}.xml"
    ET.indent(root)
    ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)
    return JSBSimModel(root_dir=root_dir, name=name, xml_path=xml_path,
                       rest_height_m=-min(c.z_m for c in contacts),
                       lift_motor_numbers=tuple(r.motor_number for r in model.lift_rotors),
                       channels=model.channels)
