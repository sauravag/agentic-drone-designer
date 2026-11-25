"""Milestone 4: ArduPilot parameter generator, JSBSim exporter and the headless SITL doublet test.

Unit tests run anywhere (fixtures are real M3 G1/G2 designs saved as JSON). The acceptance test
(`test_sitl_acceptance`) needs the arduplane binary and runs inside the Docker image:
    docker compose run --rm isr-forge pytest -q tests/test_flight_control.py
"""

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from src.schemas.components import SelectedBOM
from src.schemas.requirements import OperationalRequirements
from src.schemas.state import AeroStabilityOutput, CADArtifacts, SizingOutput
from src.tools.flight_control import param_generator as pg
from src.tools.flight_control.flight_model import (ELEVATOR_MAX_RAD, FlightModel, FlightModelError, build_flight_model,
                                                   transition_time_s)
from src.tools.flight_control.jsbsim_exporter import export_jsbsim, lift_table, lift_thrust_curve
from src.tools.flight_control.step_metrics import StepAnalysisError, altitude_loss_m, analyse_step
from src.tools.gear import TIP_BACK_ANGLE_DEG, GearInputs, PropellerDisc, landing_gear

DATA = Path(__file__).parent / "data"
KNOWN_PARAMS: frozenset[str] = frozenset((DATA / "arduplane_4.7.2_param_names.txt").read_text().split())
DesignInputs = tuple[OperationalRequirements, SizingOutput, SelectedBOM, AeroStabilityOutput, CADArtifacts]


def load_design(case: str) -> DesignInputs:
    folder = DATA / "flight_control" / case

    def read(name: str, cls: Any) -> Any:
        return cls.model_validate(json.loads((folder / f"{name}.json").read_text()))

    return (read("requirements", OperationalRequirements), read("sizing", SizingOutput), read("bom", SelectedBOM),
            read("aero", AeroStabilityOutput), read("cad", CADArtifacts))


@pytest.fixture(scope="module", params=["g1", "g2"])
def design(request: pytest.FixtureRequest) -> tuple[str, DesignInputs, FlightModel]:
    inputs = load_design(request.param)
    return request.param, inputs, build_flight_model(*inputs)


@pytest.fixture(scope="module")
def g2() -> tuple[DesignInputs, FlightModel]:
    inputs = load_design("g2")
    return inputs, build_flight_model(*inputs)


def load_vtail_design() -> DesignInputs:
    """The saved V-tail G1 design (scripts/generate_vtail_g1_fixture.py)."""
    return load_design("g1_vtail")


# ---------------------------------------------------------------- flight model ---
def test_flight_model_uses_cad_mass_and_rebased_pitch_stiffness(design: Any) -> None:
    _, (req, _, _, aero, cad), model = design
    assert model.mass_kg == cad.as_built_mass_kg and model.ixx_kgm2 == cad.ixx_kgm2
    assert model.aero.cm_alpha == pytest.approx(-aero.cl_alpha_per_rad * cad.static_margin)
    assert model.aero.cl_p < 0 and model.aero.cm_q < 0 and model.aero.cn_r < 0  # damping derivatives
    assert model.v_cruise_ms == req.cruise_speed_ms and model.v_ne_ms > model.v_cruise_ms


def test_cruise_trims_with_zero_elevator(design: Any) -> None:
    _, _, model = design
    a, alpha = model.aero, model.trim_alpha_rad(model.v_cruise_ms)
    assert a.cm0 + a.cm_alpha * alpha == pytest.approx(0.0, abs=1e-12)


def test_cruise_thrust_falls_with_speed_and_rises_with_throttle(design: Any) -> None:
    _, _, m = design
    assert m.cruise_thrust_n(1.0, m.v_cruise_ms) < m.cruise_thrust_n(1.0, 5.0) < m.cruise_thrust_n(1.0, 0.0)
    thrusts = [m.cruise_thrust_n(i / 10, 10.0) for i in range(11)]
    assert all(b >= a for a, b in zip(thrusts, thrusts[1:])) and thrusts[0] == 0.0 < thrusts[-1]
    assert m.cruise_thrust_n(2.0, 10.0) == m.cruise_thrust_n(1.0, 10.0)


def test_quadplane_rotors_follow_ardupilot_quad_x_numbering(g2: Any) -> None:
    _, model = g2
    rotors = {r.motor_number: r for r in model.lift_rotors}
    assert sorted(rotors) == [1, 2, 3, 4]
    front_right, rear_left, front_left, rear_right = rotors[1], rotors[2], rotors[3], rotors[4]
    assert front_right.x_m < model.x_cg_m < rear_left.x_m and front_right.y_m > 0 > rear_left.y_m
    assert front_left.y_m < 0 < rear_right.y_m
    assert [r.spin for r in model.lift_rotors] == [1, 1, -1, -1]  # motors 1, 2 CCW; 3, 4 CW


def test_underpowered_design_is_rejected(g2: Any) -> None:
    (req, sizing, bom, aero, cad), _ = g2
    weak = bom.model_copy(update={"vtol_motor": bom.vtol_motor.model_copy(update={"max_power_w": 100.0})})
    with pytest.raises(FlightModelError, match="lift rotors"):
        build_flight_model(req, sizing, weak, aero, cad)
    slow = bom.model_copy(update={"cruise_motor": bom.cruise_motor.model_copy(
        update={"max_power_w": 10.0, "continuous_power_w": 10.0})})
    with pytest.raises(FlightModelError, match="cruise propeller"):
        build_flight_model(req, sizing, slow, aero, cad)


def test_transition_time_grows_with_mass(g2: Any) -> None:
    _, model = g2
    assert 0 < transition_time_s(model) < transition_time_s(replace(model, mass_kg=1.5 * model.mass_kg))


# ------------------------------------------------------------ parameter generator ---
def test_every_generated_parameter_exists_in_arduplane_4_7(design: Any) -> None:
    _, (_, sizing, bom, _, _), model = design
    names = set(pg.generate_params(model, sizing, bom)) | set(pg.sitl_overrides())
    assert names - KNOWN_PARAMS == set()


def test_servo_functions_refuse_a_shared_servo_number(design: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, model = design
    aileron, *others = model.channels
    clashing = (replace(aileron, servo=others[0].servo), *others)
    with pytest.raises(ValueError, match="servo"):
        pg.fw_servo_functions(_with_channels(monkeypatch, model, clashing))


def test_servo_functions_refuse_a_surface_on_the_throttle_servo(design: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, model = design
    aileron, *others = model.channels
    with pytest.raises(ValueError, match="servo"):
        pg.fw_servo_functions(_with_channels(monkeypatch, model, (replace(aileron, servo=3), *others)))


def test_servo_functions_refuse_a_surface_on_a_quadplane_motor_output(g2: Any,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    _, model = g2
    aileron, *others = model.channels
    with pytest.raises(ValueError, match="servo"):
        pg.fw_servo_functions(_with_channels(monkeypatch, model, (replace(aileron, servo=pg.QUAD_SERVO_NUMBERS[0]), *others)))


def test_rate_gains_take_their_deflection_limits_from_the_channels(design: Any,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, model = design
    base = pg.fixed_wing_gains(model)
    aileron, *others = model.channels
    wide = replace(aileron, max_deflection_rad=2 * aileron.max_deflection_rad)
    scaled = pg.fixed_wing_gains(_with_channels(monkeypatch, model, (wide, *others)))
    assert scaled["RLL_RATE_P"] != base["RLL_RATE_P"]  # a wider aileron throw changes the roll gains
    assert scaled["PTCH_RATE_P"] == base["PTCH_RATE_P"]


def test_envelope_follows_the_srs_formulas(design: Any) -> None:
    _, (req, sizing, _, _, _), model = design
    params = pg.envelope_params(model)
    assert params["AIRSPEED_MIN"] == round(1.25 * sizing.v_stall_ms, 1)
    assert params["AIRSPEED_MAX"] == round(0.85 * model.v_ne_ms, 1)
    assert params["AIRSPEED_CRUISE"] == round(req.cruise_speed_ms, 1) == params["SCALING_SPEED"]
    assert 10 <= params["TRIM_THROTTLE"] <= 90


def test_feed_forward_is_steady_state_deflection_per_rate_within_srs_clamp(design: Any) -> None:
    _, _, model = design
    gains = pg.fixed_wing_gains(model)
    a, v = model.aero, model.v_cruise_ms
    deflection_per_rate = -a.cl_p * model.wingspan_m / (2 * v * a.cl_delta_a)  # = -L_p / L_delta_a
    expected = deflection_per_rate * pg.AP_FULL_SCALE_DEG / 20.0  # 20 deg aileron throw
    assert gains["RLL_RATE_FF"] == pytest.approx(min(max(expected, 0.15), 1.5))
    for axis in ("RLL_RATE", "PTCH_RATE"):
        assert 0.15 <= gains[f"{axis}_FF"] <= 1.5
        assert gains[f"{axis}_P"] >= pg.P_FLOOR_TO_FF * gains[f"{axis}_FF"]
        expected_i = (gains[f"{axis}_FF"] + gains[f"{axis}_P"]) / (3 * pg.ANGLE_TIME_CONSTANT_S)
        assert gains[f"{axis}_I"] == pytest.approx(expected_i)
    assert math.degrees(ELEVATOR_MAX_RAD) == pytest.approx(20.0)


def test_gain_scale_damps_p_i_d_but_not_feed_forward(design: Any) -> None:
    _, (_, sizing, bom, _, _), model = design
    full, damped = pg.generate_params(model, sizing, bom), pg.generate_params(model, sizing, bom, gain_scale=0.75)
    for name in ("RLL_RATE_P", "RLL_RATE_I", "PTCH_RATE_D"):
        assert damped[name] == pytest.approx(0.75 * full[name])
    assert damped["RLL_RATE_FF"] == full["RLL_RATE_FF"]
    with pytest.raises(pg.ParamGenerationError):
        pg.generate_params(model, sizing, bom, gain_scale=0.0)


def test_raising_the_gain_scale_holds_the_rate_d_gain_at_its_model_value(design: Any) -> None:
    # Docker SITL, g1 hot plant: D x 1/0.75 drove a ~13 Hz roll-rate oscillation (settling 1.5 s -> 4.7 s);
    # P and I x 1/0.75 with D held settled in 1.5 s and still fixed the sluggish plant (3.04 s -> 2.76 s).
    _, (_, sizing, bom, _, _), model = design
    full, raised = pg.generate_params(model, sizing, bom), pg.generate_params(model, sizing, bom, gain_scale=1 / 0.75)
    for axis in ("RLL_RATE", "PTCH_RATE"):
        assert raised[f"{axis}_P"] == pytest.approx(full[f"{axis}_P"] / 0.75)
        assert raised[f"{axis}_I"] == pytest.approx(full[f"{axis}_I"] / 0.75)
        assert raised[f"{axis}_D"] == pytest.approx(full[f"{axis}_D"])


def test_quadplane_parameters(g2: Any) -> None:
    (_, sizing, bom, _, _), model = g2
    params = pg.generate_params(model, sizing, bom)
    assert (params["Q_ENABLE"], params["Q_FRAME_CLASS"], params["Q_FRAME_TYPE"]) == (1, 1, 1)
    assert params["Q_ASSIST_SPEED"] == round(1.15 * sizing.v_stall_ms, 1)
    assert params["Q_TRANSITION_MS"] == round(1000 * transition_time_s(model))
    assert params["Q_M_THST_HOVER"] == pytest.approx(model.weight_n / (4 * model.lift_rotors[0].max_thrust_n),
                                                     abs=1e-4)
    assert [params[f"SERVO{5 + i}_FUNCTION"] for i in range(4)] == [33, 34, 35, 36]


def test_quadplane_rate_p_is_proportional_to_inertia_over_arm_thrust(g2: Any) -> None:
    _, model = g2
    base = pg.quadplane_params(model)["Q_A_RAT_RLL_P"]
    heavier = pg.quadplane_params(replace(model, ixx_kgm2=1.5 * model.ixx_kgm2))["Q_A_RAT_RLL_P"]
    assert heavier == pytest.approx(1.5 * base)


def test_quadplane_rate_p_stays_within_its_clamp_at_the_maximum_gain_scale(g2: Any) -> None:
    _, model = g2
    heavy = replace(model, ixx_kgm2=100 * model.ixx_kgm2, iyy_kgm2=100 * model.iyy_kgm2)  # raw P past the clamp
    params = pg.quadplane_params(heavy, gain_scale=1 / 0.75**2)
    for axis in ("RLL", "PIT"):
        assert params[f"Q_A_RAT_{axis}_P"] <= pg.Q_RATE_P_BOUNDS[1]


def test_fixed_wing_has_no_quadplane_parameters() -> None:
    req, sizing, bom, aero, cad = load_design("g1")
    params = pg.generate_params(build_flight_model(req, sizing, bom, aero, cad), sizing, bom)
    assert not any(name.startswith("Q_") for name in params)


def test_battery_parameters_come_from_the_bom(design: Any) -> None:
    _, (_, sizing, bom, _, _), model = design
    params = pg.battery_params(sizing, bom, model.is_quadplane)
    assert params["BATT_CAPACITY"] == round(bom.battery.capacity_wh / bom.battery.voltage_v * 1000)
    assert params["BATT_CAPACITY"] > params["BATT_LOW_MAH"] > params["BATT_CRT_MAH"] > 0


def test_cruise_outside_envelope_is_rejected(design: Any) -> None:
    _, _, model = design
    with pytest.raises(pg.ParamGenerationError, match="outside"):
        pg.envelope_params(replace(model, v_cruise_ms=model.v_stall_ms))


def test_param_file_round_trip(tmp_path: Path) -> None:
    params = {"Q_ENABLE": 1.0, "AIRSPEED_MIN": 16.25, "RLL_RATE_FF": 0.370884123}
    path = pg.write_param_file(params, tmp_path / "a.param", header="line one\nline two")
    text = path.read_text()
    assert text.startswith("# line one\n# line two\n") and "Q_ENABLE 1\n" in text
    assert pg.read_param_file(path) == pytest.approx({**params, "RLL_RATE_FF": 0.370884})


# --------------------------------------------------------------- JSBSim exporter ---
def test_jsbsim_xml_structure(design: Any, tmp_path: Path) -> None:
    case, _, model = design
    jsb = export_jsbsim(model, tmp_path)
    root = ET.parse(jsb.xml_path).getroot()
    assert root.tag == "fdm_config"
    for section in ("metrics", "mass_balance", "ground_reactions", "flight_control", "external_reactions",
                    "aerodynamics"):
        assert root.find(section) is not None, section
    assert {axis.get("name") for axis in root.iter("axis")} == {"LIFT", "DRAG", "SIDE", "ROLL", "PITCH", "YAW"}
    assert jsb.lift_motor_numbers == ((1, 2, 3, 4) if case == "g2" else ())
    assert len(root.findall("external_reactions/moment")) == len(jsb.lift_motor_numbers)


def _with_channels(monkeypatch: pytest.MonkeyPatch, model: FlightModel, channels: tuple[Any, ...]) -> FlightModel:
    """The model with its control channels swapped (they are derived from the aero data, not stored)."""
    monkeypatch.setattr(FlightModel, "channels", property(lambda self: channels))
    return model


def test_exporter_refuses_a_derivative_it_has_no_coefficient_for(design: Any, tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, model = design
    aileron, *others = model.channels
    elevator = next(c for c in others if c.surface == "elevator")
    unmapped = replace(elevator, derivatives=replace(elevator.derivatives, cn=0.05))  # no elevator yawing moment
    rest = tuple(c for c in others if c.surface != "elevator")
    with pytest.raises(ValueError, match="elevator.*cn"):
        export_jsbsim(_with_channels(monkeypatch, model, (aileron, unmapped, *rest)), tmp_path)


def test_gear_meets_the_tip_back_rule_and_clears_the_propeller(design: Any) -> None:
    _, _, model = design
    nose, left, right = landing_gear(GearInputs(
        "runway", model.fuselage_length_m, model.fuselage_diameter_m, model.mean_chord_m, model.wingspan_m,
        model.x_cg_m, PropellerDisc(0.0, 0.0, model.cruise_prop_diameter_m))).contacts
    assert nose.x_m < model.x_cg_m < left.x_m == right.x_m
    assert math.degrees(math.atan2(left.x_m - model.x_cg_m, -left.z_m)) >= TIP_BACK_ANGLE_DEG - 1e-9
    assert -left.z_m > model.cruise_prop_diameter_m / 2


def test_lift_table_peaks_at_cl_max(design: Any) -> None:
    _, _, model = design
    alphas, cls = lift_table(model)
    assert alphas == sorted(alphas) and max(cls) == model.aero.cl_max


def test_lift_thrust_curve_spans_the_spin_range(g2: Any) -> None:
    _, model = g2
    outputs, thrusts = lift_thrust_curve(model)
    assert thrusts[0] == 0.0 and thrusts[-1] == pytest.approx(model.lift_rotors[0].max_thrust_n)
    assert all(b >= a for a, b in zip(thrusts, thrusts[1:]))
    assert thrusts[outputs.index(0.1)] == 0.0  # below Q_M_SPIN_MIN the rotor gives no thrust


def test_exporter_rejects_unsafe_names(design: Any, tmp_path: Path) -> None:
    _, _, model = design
    with pytest.raises(ValueError, match="alphanumeric"):
        export_jsbsim(model, tmp_path, name="../evil")


def test_exported_model_rests_on_its_gear_in_jsbsim(design: Any, tmp_path: Path) -> None:
    jsbsim = pytest.importorskip("jsbsim")
    _, _, model = design
    jsb = export_jsbsim(model, tmp_path)
    fdm = jsbsim.FGFDMExec(str(tmp_path), None)
    fdm.set_debug_level(0)
    assert fdm.load_model(jsb.name)
    fdm["ic/h-agl-ft"] = (jsb.rest_height_m + 0.01) / 0.3048
    fdm.run_ic()
    for _ in range(1200):
        fdm.run()
    assert abs(fdm["attitude/theta-deg"]) < 10 and abs(fdm["attitude/phi-deg"]) < 1
    assert fdm["velocities/vt-fps"] * 0.3048 < 0.5  # no creeping with zero throttle
    assert fdm["inertia/weight-lbs"] * 0.45359237 == pytest.approx(model.mass_kg, rel=1e-3)


# ------------------------------------------------------------------ JSON bridge ---
def test_servo_packet_decoding() -> None:
    bridge = pytest.importorskip("src.tools.flight_control.sitl_bridge", exc_type=ImportError)
    packet = bridge.PWM_16.pack(bridge.MAGIC_16, 1200, 7, *([1500] * 16))
    assert bridge.decode_servo_packet(packet) == (1200, 7, tuple([1500] * 16))
    for bad in (b"\x00" * 4, bridge.PWM_16.pack(1234, 1200, 7, *([1500] * 16)),
                bridge.PWM_16.pack(bridge.MAGIC_16, 0, 7, *([1500] * 16))):
        with pytest.raises(ValueError):
            bridge.decode_servo_packet(bad)


def test_north_east_is_zero_at_the_origin_and_grows_with_latitude() -> None:
    bridge = pytest.importorskip("src.tools.flight_control.sitl_bridge", exc_type=ImportError)
    assert bridge.north_east_m(0.5, 0.1, 0.5, 0.1) == (0.0, 0.0)
    north, east = bridge.north_east_m(0.5 + 1e-5, 0.1, 0.5, 0.1)
    assert north == pytest.approx(1e-5 * bridge.EARTH_RADIUS_M) and east == 0.0


def test_truth_samples_carry_heading_and_position() -> None:
    bridge = pytest.importorskip("src.tools.flight_control.sitl_bridge", exc_type=ImportError)
    keys = {key for key, *_ in bridge.TRUTH_PROPS} | set(bridge.TRUTH_POSITION_KEYS)
    assert {"heading_deg", "north_m", "east_m"} <= keys


# ----------------------------------------------------------------- step metrics ---
def _first_order(tau: float, target: float = 15.0, n: int = 400, dt: float = 0.02) -> tuple[list, list, list]:
    times = [i * dt for i in range(n)]
    response = [0.0 if t < 1.0 else target * (1 - math.exp(-(t - 1.0) / tau)) for t in times]
    reference = [0.0 if t < 1.0 else target for t in times]
    return times, response, reference


def test_first_order_step_has_no_overshoot_and_settles_in_three_tau() -> None:
    result = analyse_step(*_first_order(0.4), step_time_s=1.0, window_s=6.0)
    assert result.overshoot == 0.0 and result.step_deg == pytest.approx(15.0)
    assert result.settling_s == pytest.approx(math.log(20) * 0.4, abs=0.03)  # 5 % band
    assert result.passes(0.2, 3.0)


def test_underdamped_step_reports_overshoot() -> None:
    times = [i * 0.01 for i in range(800)]
    zeta, wn = 0.3, 4.0
    wd = wn * math.sqrt(1 - zeta**2)
    resp = [0.0 if t < 1 else 10 * (1 - math.exp(-zeta * wn * (t - 1)) * (math.cos(wd * (t - 1))
            + zeta / math.sqrt(1 - zeta**2) * math.sin(wd * (t - 1)))) for t in times]
    result = analyse_step(times, resp, [0.0 if t < 1 else 10.0 for t in times], 1.0, 6.0)
    assert result.overshoot == pytest.approx(math.exp(-zeta * math.pi / math.sqrt(1 - zeta**2)), abs=0.01)
    assert not result.passes(0.2, 3.0)


def test_response_that_never_settles_reports_the_window() -> None:
    times, resp, ref = _first_order(5.0)
    assert analyse_step(times, resp, ref, 1.0, 6.0).settling_s == 6.0


def test_step_analysis_rejects_missing_or_tiny_steps() -> None:
    times, resp, ref = _first_order(0.4, target=0.5)
    with pytest.raises(StepAnalysisError, match="too small"):
        analyse_step(times, resp, ref, 1.0, 6.0)
    with pytest.raises(StepAnalysisError, match="not enough"):
        analyse_step(times, resp, ref, 100.0, 6.0)


def test_altitude_loss() -> None:
    assert altitude_loss_m([40.0, 38.5, 39.9, 41.0], 40.0) == pytest.approx(1.5)
    assert altitude_loss_m([40.0, 41.0], 40.0) == 0.0
    with pytest.raises(StepAnalysisError):
        altitude_loss_m([], 40.0)
