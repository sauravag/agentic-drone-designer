"""Flight-control unit tests split from test_flight_control.py (file size): per-servo surfaces and steering,
hand launch and gear-aware export, V-tail ruddervators and pusher thrust, and the rate-loop D cap. Fixtures and
loaders come from test_flight_control.
"""

import json
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from typing import Any, Final

import pytest

from src.schemas.state import ControlSurfaceDerivatives
from src.tools.flight_control import param_generator as pg
from src.tools.flight_control import sitl_bridge
from src.tools.flight_control.control_channels import (ControlChannel, SurfaceDerivatives, channels_with_steering,
                                                       control_channels, jsbsim_stem, steering_channel)
from src.tools.flight_control.flight_model import (ELEVATOR_MAX_RAD, SERVO_TIME_CONSTANT_S, FlightModel,
                                                   build_flight_model)
from src.tools.flight_control.jsbsim_exporter import export_jsbsim
from src.tools.flight_control.perturbation import SLUGGISH, perturbed_model
from src.tools.gear import GearInputs, PropellerDisc, landing_gear
from src.tools.sizing.layouts import TAILED_BAND
from src.tools.sizing.topology import HAND_LAUNCH_HEIGHT_M, HAND_LAUNCH_PITCH_DEG, HAND_LAUNCH_SPEED_MS
from tests.test_flight_control import (KNOWN_PARAMS, DesignInputs, g2, load_design,  # noqa: F401 (g2 is a fixture)
                                       load_vtail_design)

XML_REL: Final[float] = 1e-5  # the JSBSim export writes numbers to 6 significant digits ("%.6g")


# ------------------------------------------- per-servo surfaces, steering, servo allocation (phase 2, Task 4) ---
def _surface(name: str, **overrides: float) -> ControlSurfaceDerivatives:
    values = {"cl_per_rad": 0.1, "cl_roll_per_rad": 0.2, "cm_per_rad": -0.3, "cn_per_rad": 0.4, **overrides}
    return ControlSurfaceDerivatives(name=name, **values)


def _tractor_surfaces(a: Any) -> tuple[ControlSurfaceDerivatives, ...]:
    """Tractor records built from the aggregated derivatives, to prove the per-surface mapping path matches."""
    return (
        _surface("aileron", cl_per_rad=0.0, cl_roll_per_rad=a.cl_delta_a, cm_per_rad=0.0, cn_per_rad=a.cn_delta_a),
        _surface("elevator", cl_per_rad=a.cl_delta_e, cl_roll_per_rad=0.0, cm_per_rad=a.cm_delta_e, cn_per_rad=0.0),
        _surface("rudder", cl_per_rad=0.0, cl_roll_per_rad=0.0, cm_per_rad=0.0, cn_per_rad=a.cn_delta_r),
    )


def _vtail_surfaces() -> tuple[ControlSurfaceDerivatives, ...]:
    return (_surface("aileron"), _surface("ruddervator_left"), _surface("ruddervator_right"))


def _zero_derivative_vtail() -> tuple[ControlSurfaceDerivatives, ...]:
    """V-tail records with zero derivatives (only the channel names and property stems matter)."""
    zero = {"cl_per_rad": 0.0, "cl_roll_per_rad": 0.0, "cm_per_rad": 0.0, "cn_per_rad": 0.0}
    return tuple(_surface(n, **zero) for n in ("aileron", "ruddervator_left", "ruddervator_right"))


def test_legacy_aero_record_gives_tractor_channels() -> None:
    _, sizing, bom, aero, cad = load_design("legacy_g1")  # saved before phase 2: no control_surfaces
    assert aero.control_surfaces == ()
    model = build_flight_model(*load_design("legacy_g1"))
    assert [(c.surface, c.servo, c.servo_function) for c in model.channels] == [
        ("aileron", 1, 4), ("elevator", 2, 19), ("rudder", 4, 21)]


def test_channels_from_surface_records_keep_tractor_values() -> None:
    model = build_flight_model(*load_design("g1"))
    assert control_channels(model.aero, _tractor_surfaces(model.aero)) == control_channels(model.aero)


def test_surface_channels_follow_the_servo_map() -> None:
    model = build_flight_model(*load_design("g1"))
    channels = control_channels(model.aero, _vtail_surfaces())
    assert [(c.surface, c.servo, c.servo_function) for c in channels] == [
        ("aileron", 1, 4), ("ruddervator_left", 2, 79), ("ruddervator_right", 4, 80)]
    assert channels[1].derivatives == SurfaceDerivatives(cl=0.1, cl_roll=0.2, cm=-0.3, cn=0.4)


def test_unknown_surface_name_is_refused() -> None:
    model = build_flight_model(*load_design("g1"))
    with pytest.raises(ValueError, match="canard"):
        control_channels(model.aero, (_surface("canard"),))


def test_rudder_steers_the_nose_wheel() -> None:
    model = build_flight_model(*load_design("g1"))
    steer = steering_channel(control_channels(model.aero), quad_motors=False)
    assert steer is not None and (steer.surface, steer.servo, steer.servo_function) == ("rudder", 4, 21)
    assert steer.steers_nose_wheel
    assert [c.surface for c in model.channels if c.steers_nose_wheel] == ["rudder"]


def test_no_rudder_gets_ground_steering_servo() -> None:
    model = build_flight_model(*load_design("g1"))
    vt = _vtail_surfaces()
    steer = steering_channel(control_channels(model.aero, vt), quad_motors=False)
    assert steer is not None
    assert (steer.surface, steer.servo, steer.servo_function) == ("steering", 5, 26)
    assert steer.steers_nose_wheel
    quad = steering_channel(control_channels(model.aero, vt), quad_motors=True)
    assert quad is not None and quad.servo == 9


def test_vtail_model_channels_include_the_steering_channel() -> None:
    model = replace(build_flight_model(*load_design("g1")), surfaces=_vtail_surfaces())
    assert [(c.surface, c.servo, c.servo_function) for c in model.channels] == [
        ("aileron", 1, 4), ("ruddervator_left", 2, 79), ("ruddervator_right", 4, 80), ("steering", 5, 26)]
    assert [c.surface for c in model.channels if c.steers_nose_wheel] == ["steering"]


def test_fixed_wing_does_not_reserve_quad_servos() -> None:
    fixed_wing = replace(build_flight_model(*load_design("g1")), surfaces=_vtail_surfaces())
    functions = pg.fw_servo_functions(fixed_wing)
    assert functions == {"SERVO1_FUNCTION": 4, "SERVO2_FUNCTION": 79, "SERVO3_FUNCTION": 70, "SERVO4_FUNCTION": 80,
                         "SERVO5_FUNCTION": 26}  # steering on output 5, not lift motor 1 (33)


def test_quadplane_keeps_outputs_5_to_8_for_lift_motors(g2: Any) -> None:
    (_, sizing, bom, _, _), model = g2
    params = pg.generate_params(model, sizing, bom)
    assert [params[f"SERVO{n}_FUNCTION"] for n in (5, 6, 7, 8)] == [33, 34, 35, 36]


def test_perturbation_scales_every_surface() -> None:
    model = replace(build_flight_model(*load_design("g1")), surfaces=_vtail_surfaces())
    perturbed = perturbed_model(model, SLUGGISH, TAILED_BAND)
    for before, after in zip(model.surfaces, perturbed.surfaces, strict=True):
        assert after.name == before.name
        assert after.cl_roll_per_rad == pytest.approx(SLUGGISH.control_power * before.cl_roll_per_rad)
        assert after.cm_per_rad == pytest.approx(SLUGGISH.control_power * before.cm_per_rad)
        assert after.cn_per_rad == pytest.approx(SLUGGISH.control_power * before.cn_per_rad)
        assert after.cl_per_rad == before.cl_per_rad
    assert model.surfaces == _vtail_surfaces()  # the input is not mutated


def test_bridge_drives_steering_from_the_channel_that_steers() -> None:
    model = replace(build_flight_model(*load_design("g1")), surfaces=_vtail_surfaces())
    fdm: dict[str, float] = {}
    pwm = [1500, 1500, 1000, 1500, 1750]  # servo 5 (steering) at +0.5
    sitl_bridge._apply_channels(fdm, model.channels, pwm)
    assert fdm["fcs/steer-cmd-norm"] == pytest.approx(0.5)


def test_no_free_output_for_steering_is_an_error() -> None:
    model = build_flight_model(*load_design("g1"))
    full = tuple(ControlChannel(f"s{n}", n, 100 + n, 1.0, 0.3, SurfaceDerivatives()) for n in range(1, 17))
    with pytest.raises(ValueError, match="steering"):
        channels_with_steering(full, quad_motors=False)
    assert model.channels  # the real tractor still builds


def test_missing_pitch_surface_is_a_clear_parameter_error() -> None:
    lone_panel = (_surface("aileron"), _surface("ruddervator_left"))  # neither an elevator nor a ruddervator pair
    model = replace(build_flight_model(*load_design("g1")), surfaces=lone_panel)
    with pytest.raises(pg.ParamGenerationError, match="elevator.*ruddervator_right"):
        pg.fixed_wing_gains(model)


def test_jsbsim_stem_hyphenates_surface_names() -> None:
    assert jsbsim_stem("ruddervator_left") == "ruddervator-left"
    assert [jsbsim_stem(n) for n in ("aileron", "elevator", "rudder")] == ["aileron", "elevator", "rudder"]


def test_bridge_and_exporter_agree_on_property_names(tmp_path: Path) -> None:
    model = replace(build_flight_model(*load_design("g1")), surfaces=_zero_derivative_vtail())
    fdm: dict[str, float] = {}
    sitl_bridge._apply_channels(fdm, model.channels, [1500] * 16)
    assert "fcs/ruddervator-left-cmd-norm" in fdm and "fcs/ruddervator_left-cmd-norm" not in fdm
    root = ET.parse(export_jsbsim(model, tmp_path).xml_path).getroot()
    exported_inputs = {node.text.strip() for node in root.findall("flight_control/channel/lag_filter/input")
                       if node.text and node.text.strip().endswith("-cmd-norm")}
    assert {f"fcs/{jsbsim_stem(c.surface)}-cmd-norm" for c in model.channels} == exported_inputs
    assert exported_inputs == {name for name in fdm if name.endswith("-cmd-norm") and name != "fcs/steer-cmd-norm"}


# ------------------------------------------------ hand launch and gear-aware export (phase 2, Task 5) ---
def _hand_design() -> DesignInputs:
    req, sizing, bom, aero, cad = load_design("g1")
    return req, sizing.model_copy(update={"launch_method": "hand", "gear_kind": "none"}), bom, aero, cad


def test_flight_model_gear_comes_from_the_launch_method_and_the_real_propeller() -> None:
    _, sizing, bom, _, _ = load_design("g1")
    runway = build_flight_model(*load_design("g1"))
    expected = landing_gear(GearInputs(
        "runway", sizing.fuselage_length_m, sizing.fuselage_diameter_m, sizing.mean_chord_m, sizing.wingspan_m,
        runway.x_cg_m, PropellerDisc(0.0, 0.0, bom.cruise_propeller.diameter_m)))
    assert runway.launch == "runway" and runway.gear == expected and runway.gear.kind == "tricycle"
    hand = build_flight_model(*_hand_design())
    assert hand.launch == "hand" and hand.gear.kind == "none" and hand.gear.contacts == ()


def test_export_without_gear_writes_belly_contacts(tmp_path: Path) -> None:
    model = build_flight_model(*_hand_design())
    xml = export_jsbsim(model, tmp_path).xml_path.read_text()
    assert xml.count('type="STRUCTURE"') == 4 and 'type="BOGEY"' not in xml
    contacts = {c.get("name"): c for c in ET.fromstring(xml).findall("ground_reactions/contact")}

    def at(name: str) -> tuple[float, float, float]:
        loc = contacts[name].find("location")
        return tuple(float(loc.find(axis).text) for axis in "xyz")  # type: ignore[union-attr,return-value]

    belly_z = -model.fuselage_diameter_m / 2
    assert at("BELLY_CG") == pytest.approx((model.x_cg_m, 0.0, belly_z), rel=XML_REL)
    assert at("BELLY_AFT") == pytest.approx((0.8 * model.fuselage_length_m, 0.0, belly_z), rel=XML_REL)
    for name, side in (("LEFT_WINGTIP", -1), ("RIGHT_WINGTIP", 1)):  # the tips keep it from rolling inverted
        assert at(name) == pytest.approx((model.x_cg_m, side * model.wingspan_m / 2, model.wing_tip_z_m),
                                         rel=XML_REL)
    assert all(float(c.find("max_steer").text) == 0.0 for c in contacts.values())  # type: ignore[union-attr]


def test_runway_export_is_unchanged_bogeys(tmp_path: Path) -> None:
    xml = export_jsbsim(build_flight_model(*load_design("g1")), tmp_path).xml_path.read_text()
    assert xml.count('type="BOGEY"') == 3 and 'type="STRUCTURE"' not in xml


def test_hand_launch_params_omit_runway_takeoff() -> None:
    inputs = _hand_design()
    params = pg.generate_params(build_flight_model(*inputs), inputs[1], inputs[2])
    runway = load_design("g1")
    runway_params = pg.generate_params(build_flight_model(*runway), runway[1], runway[2])
    assert "TKOFF_ROTATE_SPD" in runway_params
    assert not {name for name in params if name.startswith("TKOFF_")}
    assert set(runway_params) - set(params) == {name for name in runway_params if name.startswith("TKOFF_")}


def test_gearless_aircraft_needs_no_steering_channel() -> None:
    model = replace(build_flight_model(*_hand_design()), surfaces=_vtail_surfaces())
    assert [c.surface for c in model.channels] == ["aileron", "ruddervator_left", "ruddervator_right"]
    assert "SERVO5_FUNCTION" not in pg.fw_servo_functions(model)


def test_gearless_aircraft_is_caught_by_the_ground_model_in_jsbsim(tmp_path: Path) -> None:
    jsbsim = pytest.importorskip("jsbsim")
    model = build_flight_model(*_hand_design())
    jsb = export_jsbsim(model, tmp_path)
    assert jsb.rest_height_m == pytest.approx(model.fuselage_diameter_m / 2) and jsb.launch == "hand"
    fdm = jsbsim.FGFDMExec(str(tmp_path), None)
    fdm.set_debug_level(0)
    assert fdm.load_model(jsb.name)
    fdm["ic/h-agl-ft"] = (jsb.rest_height_m + 0.01) / 0.3048
    fdm.run_ic()
    for _ in range(1200):
        fdm.run()
    assert abs(fdm["attitude/phi-deg"]) < 5 and model.wing_tip_z_m > 0  # the wingtip contacts keep it upright
    assert fdm["velocities/vt-fps"] * 0.3048 < 0.5


def test_hand_launch_initial_state_is_airborne_at_throw_speed(tmp_path: Path) -> None:
    jsbsim = pytest.importorskip("jsbsim")
    jsb = export_jsbsim(build_flight_model(*_hand_design()), tmp_path)
    bridge = sitl_bridge.JsonPhysicsBridge(jsb, sitl_bridge.Home(), port=0)
    fdm = jsbsim.FGFDMExec(str(tmp_path), None)
    fdm.set_debug_level(0)
    assert fdm.load_model(jsb.name)
    bridge._initialise(fdm)
    assert fdm["position/h-agl-ft"] * 0.3048 == pytest.approx(HAND_LAUNCH_HEIGHT_M, abs=0.05)
    assert fdm["velocities/u-fps"] * 0.3048 == pytest.approx(HAND_LAUNCH_SPEED_MS, rel=1e-3)
    assert fdm["attitude/theta-deg"] == pytest.approx(HAND_LAUNCH_PITCH_DEG, abs=0.05)


def test_hand_launch_is_held_until_the_throttle_opens(tmp_path: Path) -> None:
    jsbsim = pytest.importorskip("jsbsim")
    jsb = export_jsbsim(build_flight_model(*_hand_design()), tmp_path)
    bridge = sitl_bridge.JsonPhysicsBridge(jsb, sitl_bridge.Home(), port=0)
    bridge._fdm = bridge._load_fdm()

    def frame(count: int, throttle_pwm: int) -> dict[str, Any]:
        pwm = [1500, 1500, throttle_pwm, 1500, *([1000] * 12)]
        packet = sitl_bridge.PWM_16.pack(sitl_bridge.MAGIC_16, 400, count, *pwm)
        return json.loads(bridge._handle(packet))

    held = [frame(n, 1000) for n in range(1, 50)]
    assert all(s["position"] == [0.0, 0.0, 0.0] and s["airspeed"] == 0.0 for s in held)
    assert held[-1]["timestamp"] > held[0]["timestamp"]  # ArduPilot's clock keeps running
    truth_t = [sample["t_s"] for sample in bridge.stats.truth]
    assert len(truth_t) >= 2 and truth_t == sorted(truth_t) and len(set(truth_t)) == len(truth_t)  # logged in the hold
    thrown = [frame(n, 2000) for n in range(50, 100)]
    assert thrown[-1]["airspeed"] > HAND_LAUNCH_SPEED_MS / 2 and thrown[-1]["position"][0] > 0.0
    assert thrown[0]["timestamp"] > held[-1]["timestamp"]


def test_a_hand_launched_quadplane_is_refused_not_held_forever() -> None:
    flight_test = pytest.importorskip("src.tools.flight_control.flight_test", exc_type=ImportError)
    with pytest.raises(flight_test.SitlFlightError, match="quadplane.*hand-launched"):
        flight_test.fly_acceptance_test(None, {}, True, 0.0, "hand")  # type: ignore[arg-type]


# ------------------------------------------- V-tail ruddervators, mixing and pusher thrust (phase 2, Task 7) ---
VTAIL_LAYOUT: str = "single_boom_vtail_pusher"
AP_PITCH_UP: float = 1.0  # ArduPilot scaled elevator output above zero demands nose up
AP_YAW_RIGHT: float = 1.0  # ArduPilot scaled rudder output above zero demands nose right


@pytest.fixture(scope="module")
def vtail_g1_design() -> DesignInputs:
    return load_vtail_design()


@pytest.fixture(scope="module")
def vtail_model(vtail_g1_design: DesignInputs) -> FlightModel:
    return build_flight_model(*vtail_g1_design)


def _panels(model: FlightModel) -> tuple[ControlChannel, ControlChannel]:
    by_name = {c.surface: c for c in model.channels}
    return by_name["ruddervator_left"], by_name["ruddervator_right"]


def test_vtail_fixture_is_the_vtail_pusher(vtail_g1_design: DesignInputs) -> None:
    _, sizing, _, aero, cad = vtail_g1_design
    assert sizing.airframe_layout == VTAIL_LAYOUT and [t.control for t in sizing.tail_surfaces] == ["ruddervator"]
    assert [s.name for s in aero.control_surfaces] == ["aileron", "ruddervator_left", "ruddervator_right"]
    assert cad.propeller_x_m is not None and cad.propeller_x_m > cad.x_cg_m  # boom-end pusher behind the CG


def test_vtail_channels_and_servos(vtail_g1_design: DesignInputs, vtail_model: FlightModel) -> None:
    model = vtail_model
    assert [(c.surface, c.servo, c.servo_function) for c in model.channels][:3] == [
        ("aileron", 1, 4), ("ruddervator_left", 2, 79), ("ruddervator_right", 4, 80)]
    params = pg.generate_params(model, *vtail_g1_design[1:3])
    assert params["MIXING_GAIN"] == 0.5 and params["SERVO5_FUNCTION"] == 26
    assert (params["SERVO2_FUNCTION"], params["SERVO4_FUNCTION"]) == (79, 80)
    assert set(params) <= KNOWN_PARAMS


def test_vtail_panels_carry_their_own_measured_derivatives(vtail_g1_design: DesignInputs,
                                                           vtail_model: FlightModel) -> None:
    aero = vtail_g1_design[3]
    left, right = _panels(vtail_model)
    records = {s.name: s for s in aero.control_surfaces}
    for channel in (left, right):
        record = records[channel.surface]
        assert channel.derivatives == SurfaceDerivatives(cl=record.cl_per_rad, cl_roll=record.cl_roll_per_rad,
                                                         cm=record.cm_per_rad, cn=record.cn_per_rad)
    assert left.derivatives.cm < 0 and right.derivatives.cm < 0  # positive panel deflection = trailing edge down
    assert left.derivatives.cl_roll * right.derivatives.cl_roll < 0  # roll coupling of opposite sign
    assert aero.cm_delta_e_per_rad == pytest.approx(left.derivatives.cm + right.derivatives.cm)  # by linearity
    assert aero.cn_delta_r_per_rad == pytest.approx(left.derivatives.cn - right.derivatives.cn)


def _ardupilot_vtail_mix(elevator: float, rudder: float, mixing_gain: float) -> tuple[float, float]:
    """ArduPlane's V-tail mixer (Plane::channel_function_mixer(k_rudder, k_elevator, k_vtail_right, k_vtail_left)):
    (left, right) scaled outputs; verified against arduplane 4.7 in SITL (Task 7 report)."""
    return (elevator + rudder) * mixing_gain, (elevator - rudder) * mixing_gain


def _moments(model: FlightModel, params: pg.ParamSet, elevator: float, rudder: float) -> tuple[float, float]:
    """(Cm, Cn) per unit of panel throw from ArduPilot demands, through reversal, PWM and the bridge sign."""
    outputs = dict(zip(("ruddervator_left", "ruddervator_right"),
                       _ardupilot_vtail_mix(elevator, rudder, params["MIXING_GAIN"])))
    cm = cn = 0.0
    for channel in _panels(model):
        reversal = -1.0 if params[f"SERVO{channel.servo}_REVERSED"] else 1.0
        deflection = channel.bridge_sign * reversal * outputs[channel.surface]
        cm += channel.derivatives.cm * deflection
        cn += channel.derivatives.cn * deflection
    return cm, cn


def test_vtail_pitch_up_and_yaw_right_demands_give_nose_up_and_nose_right(vtail_g1_design: DesignInputs,
                                                                          vtail_model: FlightModel) -> None:
    params = pg.generate_params(vtail_model, *vtail_g1_design[1:3])
    pitch_cm, pitch_cn = _moments(vtail_model, params, AP_PITCH_UP, 0.0)
    yaw_cm, yaw_cn = _moments(vtail_model, params, 0.0, AP_YAW_RIGHT)
    assert pitch_cm > 0 and abs(pitch_cn) < 0.1 * pitch_cm
    assert yaw_cn > 0 and abs(yaw_cm) < 0.1 * abs(pitch_cm)


def test_vtail_reversal_follows_each_panels_pitch_sign(vtail_g1_design: DesignInputs,
                                                      vtail_model: FlightModel) -> None:
    """VSPAERO records are normalised to trailing edge down (Cm < 0), so the generated design always gets
    SERVOn_REVERSED 0; the rule is a guard for records built any other way, exercised here with a hand-flipped panel."""
    params = pg.generate_params(vtail_model, *vtail_g1_design[1:3])
    assert (params["SERVO2_REVERSED"], params["SERVO4_REVERSED"]) == (0, 0)
    flipped = tuple(s.model_copy(update={"cm_per_rad": -s.cm_per_rad, "cn_per_rad": -s.cn_per_rad,
                                         "cl_per_rad": -s.cl_per_rad, "cl_roll_per_rad": -s.cl_roll_per_rad})
                    if s.name == "ruddervator_left" else s for s in vtail_model.surfaces)
    flipped_params = pg.generate_params(replace(vtail_model, surfaces=flipped), *vtail_g1_design[1:3])
    assert flipped_params["SERVO2_REVERSED"] == 1 - params["SERVO2_REVERSED"]
    assert flipped_params["SERVO4_REVERSED"] == params["SERVO4_REVERSED"]


def test_vtail_pitch_gains_use_the_mixed_control_power(vtail_g1_design: DesignInputs,
                                                       vtail_model: FlightModel) -> None:
    left, right = _panels(vtail_model)
    mixed = 0.5 * (left.derivatives.cm + right.derivatives.cm)
    # a tractor whose elevator has the mixed pitch power and the ruddervator throw must get the same pitch gains
    equivalent = replace(vtail_model, surfaces=(), aero=replace(vtail_model.aero, cm_delta_e=mixed))
    assert left.max_deflection_rad == ELEVATOR_MAX_RAD
    vtail = pg.fixed_wing_gains(vtail_model)
    tractor = pg.fixed_wing_gains(equivalent)
    assert {k: v for k, v in vtail.items() if k.startswith("PTCH")} == pytest.approx(
        {k: v for k, v in tractor.items() if k.startswith("PTCH")})
    assert vtail["RLL_RATE_P"] == pytest.approx(tractor["RLL_RATE_P"])


def test_tractor_params_have_no_mixing_or_reversal() -> None:
    inputs = load_design("g1")
    params = pg.generate_params(build_flight_model(*inputs), inputs[1], inputs[2])
    assert "MIXING_GAIN" not in params and not any(name.endswith("_REVERSED") for name in params)


def test_vtail_damping_reads_the_effective_tail_areas(vtail_g1_design: DesignInputs,
                                                      vtail_model: FlightModel) -> None:
    req, sizing, bom, aero, cad = vtail_g1_design
    bigger = tuple(t.model_copy(update={"area_m2": 2 * t.area_m2}) for t in sizing.tail_surfaces)
    doubled = build_flight_model(req, sizing.model_copy(update={"tail_surfaces": bigger}), bom, aero, cad)
    assert doubled.aero.cm_q / vtail_model.aero.cm_q == pytest.approx(2.0, rel=0.05)
    assert doubled.aero.cy_beta == pytest.approx(2 * vtail_model.aero.cy_beta)


def _cruise_force(root: ET.Element) -> ET.Element:
    force = next(f for f in root.findall("external_reactions/force") if f.get("name") == "cruise")
    return force


def _xyz(node: ET.Element) -> tuple[float, ...]:
    return tuple(float(node.find(axis).text) for axis in "xyz")  # type: ignore[union-attr,arg-type]


def test_pusher_thrust_at_propeller_station(vtail_g1_design: DesignInputs, vtail_model: FlightModel,
                                            tmp_path: Path) -> None:
    cad = vtail_g1_design[4]
    force = _cruise_force(ET.parse(export_jsbsim(vtail_model, tmp_path).xml_path).getroot())
    assert _xyz(force.find("location")) == pytest.approx(  # type: ignore[arg-type]
        (cad.propeller_x_m, 0.0, cad.propeller_z_m), rel=1e-5, abs=1e-9)  # the XML keeps 6 significant digits
    assert _xyz(force.find("direction")) == (1.0, 0.0, 0.0)  # type: ignore[arg-type] # pushes forward


def test_tractor_thrust_stays_at_the_nose(tmp_path: Path) -> None:
    force = _cruise_force(ET.parse(export_jsbsim(build_flight_model(*load_design("g1")), tmp_path).xml_path)
                          .getroot())
    assert _xyz(force.find("location")) == (0.0, 0.0, 0.0)  # type: ignore[arg-type]


def test_vtail_jsbsim_has_one_coefficient_set_per_panel(vtail_model: FlightModel, tmp_path: Path) -> None:
    root = ET.parse(export_jsbsim(vtail_model, tmp_path).xml_path).getroot()
    for channel in _panels(vtail_model):
        stem = jsbsim_stem(channel.surface)
        for axis, value in (("LIFT", channel.derivatives.cl), ("ROLL", channel.derivatives.cl_roll),
                            ("PITCH", channel.derivatives.cm), ("YAW", channel.derivatives.cn)):
            terms = [f for f in root.findall(f"aerodynamics/axis[@name='{axis}']/function")
                     if any(p.text and p.text.strip() == f"fcs/{stem}-pos-rad" for p in f.iter("property"))]
            assert len(terms) == 1, (channel.surface, axis)
            assert float(terms[0].find("product/value").text) == pytest.approx(value, rel=1e-5)  # type: ignore


def test_vtail_bridge_drives_each_panel_from_its_own_servo(vtail_model: FlightModel) -> None:
    left, right = _panels(vtail_model)
    fdm: dict[str, float] = {}
    pwm = [1500] * 16
    pwm[left.servo - 1], pwm[right.servo - 1] = 2000, 1250
    sitl_bridge._apply_channels(fdm, vtail_model.channels, pwm)
    assert fdm["fcs/ruddervator-left-cmd-norm"] == pytest.approx(left.bridge_sign * 1.0)
    assert fdm["fcs/ruddervator-right-cmd-norm"] == pytest.approx(right.bridge_sign * -0.5)


def test_truth_log_records_each_channels_surface(vtail_model: FlightModel, tmp_path: Path) -> None:
    jsbsim = pytest.importorskip("jsbsim")
    jsb = export_jsbsim(vtail_model, tmp_path)
    bridge = sitl_bridge.JsonPhysicsBridge(jsb, sitl_bridge.Home(), port=0)
    fdm = jsbsim.FGFDMExec(str(tmp_path), None)
    fdm.set_debug_level(0)
    assert fdm.load_model(jsb.name)
    bridge._initialise(fdm)
    bridge._record_truth(fdm)  # a V-tail has no elevator property: reading one would stop the bridge
    assert {"aileron_rad", "ruddervator_left_rad", "ruddervator_right_rad"} <= set(bridge.stats.truth[-1])


def test_vtail_model_rests_on_its_gear_in_jsbsim(vtail_model: FlightModel, tmp_path: Path) -> None:
    jsbsim = pytest.importorskip("jsbsim")
    jsb = export_jsbsim(vtail_model, tmp_path)
    fdm = jsbsim.FGFDMExec(str(tmp_path), None)
    fdm.set_debug_level(0)
    assert fdm.load_model(jsb.name)  # every per-panel command property is declared
    fdm["ic/h-agl-ft"] = (jsb.rest_height_m + 0.01) / 0.3048
    fdm.run_ic()
    for _ in range(1200):
        fdm.run()
    assert abs(fdm["attitude/theta-deg"]) < 10 and abs(fdm["attitude/phi-deg"]) < 1
    assert fdm["velocities/vt-fps"] * 0.3048 < 0.5


# ------------------------------------------------------------------- rate-loop D path (Task 8 fix) ---
def _roll_loop(model: FlightModel) -> tuple[float, float]:
    """(L_delta_a in rad/s^2 per rad, aileron throw in rad) as the gain design computes them."""
    qs = model.dynamic_pressure(model.v_cruise_ms) * model.wing_area_m2
    throw = next(c.max_deflection_rad for c in model.channels if c.surface == "aileron")
    return qs * model.wingspan_m * model.aero.cl_delta_a / model.ixx_kgm2, throw


def _d_loop_gain(params: pg.ParamSet, prefix: str, control_power: float, throw_rad: float) -> float:
    """The D path's dimensionless high-frequency loop gain D_phys x control power (D_phys from ArduPilot units)."""
    return params[f"{prefix}_D"] / pg._to_ap_units(1.0, throw_rad) * control_power


def test_light_roll_inertia_caps_the_rate_d_path(vtail_g1_design: DesignInputs) -> None:
    """The V-tail's as-built roll inertia (kit, 0.095 kg m^2) makes its bare roll mode fast (|L_p| ~ 80 1/s); D = P x
    servo lag then gives the D path a loop gain ~2 (the tractor's is 0.73), which rings the rate loop at ~13 Hz on the
    hot corner once the retry raises P. D is capped at D_LOOP_GAIN_MAX; P, I and FF are unchanged."""
    model = build_flight_model(*vtail_g1_design)
    l_da, throw = _roll_loop(model)
    params = pg.fixed_wing_gains(model)
    assert _d_loop_gain(params, "RLL_RATE", l_da, throw) == pytest.approx(pg.D_LOOP_GAIN_MAX, rel=1e-9)
    uncapped_p = params["RLL_RATE_P"]
    assert params["RLL_RATE_D"] < uncapped_p * SERVO_TIME_CONSTANT_S  # the cap binds here


@pytest.mark.parametrize("case", ["g1", "g2"])
def test_reference_tractors_keep_d_equal_to_p_times_servo_lag(case: str) -> None:
    """The cap sits above both reference tractors' D-path loop gain (0.73 and 0.63): their gains are unchanged."""
    model = build_flight_model(*load_design(case))
    l_da, throw = _roll_loop(model)
    params = pg.fixed_wing_gains(model)
    assert params["RLL_RATE_D"] == pytest.approx(params["RLL_RATE_P"] * SERVO_TIME_CONSTANT_S, rel=1e-12)
    assert _d_loop_gain(params, "RLL_RATE", l_da, throw) < pg.D_LOOP_GAIN_MAX
