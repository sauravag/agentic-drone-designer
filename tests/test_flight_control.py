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
from src.schemas.state import AeroStabilityOutput, CADArtifacts, FCSArtifacts, SizingOutput
from src.tools.flight_control import param_generator as pg
from src.tools.flight_control.flight_model import (ELEVATOR_MAX_RAD, FlightModel, FlightModelError,
                                                   build_flight_model, transition_time_s)
from src.tools.flight_control.jsbsim_exporter import (TIP_BACK_ANGLE_DEG, export_jsbsim, gear_contacts,
                                                      lift_table, lift_thrust_curve)
from src.tools.flight_control.perturbation import HOT, SLUGGISH, PlantCase, perturbed_model
from src.tools.flight_control.step_metrics import StepAnalysisError, altitude_loss_m, analyse_step

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


def test_cruise_thrust_falls_with_speed_and_scales_with_throttle(design: Any) -> None:
    _, _, m = design
    assert m.cruise_thrust_n(1.0, 0.0) == pytest.approx(m.cruise_static_thrust_n, rel=1e-6)
    assert m.cruise_thrust_n(1.0, m.v_cruise_ms) < m.cruise_thrust_n(1.0, 5.0)
    assert m.cruise_thrust_n(0.5, 10.0) == pytest.approx(0.5 * m.cruise_thrust_n(1.0, 10.0))
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
    weak = bom.model_copy(update={"vtol_motor": bom.vtol_motor.model_copy(update={"max_thrust_n": 10.0})})
    with pytest.raises(FlightModelError, match="lift rotors"):
        build_flight_model(req, sizing, weak, aero, cad)
    slow = bom.model_copy(update={"cruise_motor": bom.cruise_motor.model_copy(update={"max_power_w": 1.0})})
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


def test_quadplane_parameters(g2: Any) -> None:
    (_, sizing, bom, _, _), model = g2
    params = pg.generate_params(model, sizing, bom)
    assert (params["Q_ENABLE"], params["Q_FRAME_CLASS"], params["Q_FRAME_TYPE"]) == (1, 1, 1)
    assert params["Q_ASSIST_SPEED"] == round(1.15 * sizing.v_stall_ms, 1)
    assert params["Q_TRANSITION_MS"] == round(1000 * transition_time_s(model))
    assert params["Q_M_THST_HOVER"] == pytest.approx(model.weight_n / (4 * bom.vtol_motor.max_thrust_n), abs=1e-4)
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


def test_gear_meets_the_tip_back_rule_and_clears_the_propeller(design: Any) -> None:
    _, _, model = design
    nose, left, right = gear_contacts(model)
    assert nose.x_m < model.x_cg_m < left.x_m == right.x_m
    assert math.degrees(math.atan2(left.x_m - model.x_cg_m, -left.z_m)) >= TIP_BACK_ANGLE_DEG - 1e-9
    assert -left.z_m > model.cruise_prop_diameter_m / 2


def test_lift_table_peaks_at_cl_max_and_thrust_curve_matches_ardupilot(design: Any) -> None:
    _, _, model = design
    alphas, cls = lift_table(model)
    assert alphas == sorted(alphas) and max(cls) == model.aero.cl_max
    outputs, thrusts = lift_thrust_curve(100.0)
    assert thrusts[0] == 0.0 and thrusts[-1] == pytest.approx(100.0)
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


# ------------------------------------------------- pilot helpers and the validator ---
def test_stick_mapping() -> None:
    ft = pytest.importorskip("src.tools.flight_control.flight_test", exc_type=ImportError)
    assert (ft.stick_pwm(0.0), ft.stick_pwm(1.0), ft.stick_pwm(-2.0)) == (1500, 2000, 1000)
    assert (ft.throttle_pwm(0.0), ft.throttle_pwm(0.5), ft.throttle_pwm(1.5)) == (1000, 1500, 2000)
    params = {"AIRSPEED_MIN": 16.0, "AIRSPEED_MAX": 24.0, "AIRSPEED_CRUISE": 20.0,
              "PTCH_LIM_MAX_DEG": 20.0, "PTCH_LIM_MIN_DEG": -10.0}
    assert ft.cruise_stick_pwm(params) == 1500
    assert (ft.pitch_stick_pwm(10.0, params), ft.pitch_stick_pwm(-5.0, params)) == (1750, 1250)


def _synthetic_log(ft: Any, roll_tau: float, transition_dip_m: float = 0.0, ringing: bool = False,
                   ring_decay_s: float = 2.0) -> Any:
    """roll_tau 2.0 settles too slowly; ringing=True adds a ~60 % roll overshoot instead (ring_decay_s 0.5: ~15 %)."""
    log = ft.FlightLog()
    for i in range(1500):
        t = i * 0.04
        d = t - 20
        roll = 0.0 if t < 20 else 15 * (1 - math.exp(-d / ring_decay_s) * math.cos(3 * d)) if ringing else (
            0.0 if t < 20 else 15 * (1 - math.exp(-d / roll_tau)))
        pitch = 3.0 if t < 40 else 13 - 10 * math.exp(-(t - 40) / 0.3)
        alt = 40 - (transition_dip_m if 5 < t < 8 else 0.0)
        log.samples.append(ft.Sample(t, roll, pitch, 0.0 if t < 20 else 15.0, 3.0 if t < 40 else 13.0, alt, 22, 20))
    log.roll_step_t_s, log.pitch_step_t_s = 20.0, 40.0
    log.transition_start_t_s, log.transition_end_t_s, log.transition_start_alt_m = 4.0, 10.0, 40.0
    return log


def test_score_applies_all_three_srs_criteria() -> None:
    ft = pytest.importorskip("src.tools.flight_control.flight_test", exc_type=ImportError)
    sv = pytest.importorskip("src.tools.flight_control.sitl_validator", exc_type=ImportError)
    good = sv.score(_synthetic_log(ft, 0.4, transition_dip_m=2.0), 1.0)
    assert good.passed and good.transition_alt_loss_m == pytest.approx(2.0)
    assert not sv.score(_synthetic_log(ft, 2.0), 1.0).passed  # roll settles in ~6 s
    assert not sv.score(_synthetic_log(ft, 0.4, transition_dip_m=20.0), 1.0).passed


def test_next_gain_scale_goes_down_for_overshoot_and_crashes_up_for_settling() -> None:
    ft = pytest.importorskip("src.tools.flight_control.flight_test", exc_type=ImportError)
    sv = pytest.importorskip("src.tools.flight_control.sitl_validator", exc_type=ImportError)
    overshoot = sv.score(_synthetic_log(ft, 0.4, ringing=True), 1.0)
    slow = sv.score(_synthetic_log(ft, 2.0), 1.0)
    crash = sv.AttemptResult(gain_scale=1.0, passed=False, error="SitlFlightError: stall")
    assert overshoot.roll.overshoot >= 0.2 and slow.roll.overshoot < 0.2
    assert sv.next_gain_scale(overshoot, 1.0, sv.DIRECTION_NONE) == (pytest.approx(0.75), sv.DIRECTION_DOWN)
    assert sv.next_gain_scale(slow, 1.0, sv.DIRECTION_NONE) == (pytest.approx(1 / 0.75), sv.DIRECTION_UP)
    assert sv.next_gain_scale(crash, 1.0, sv.DIRECTION_NONE) == (pytest.approx(0.75), sv.DIRECTION_DOWN)
    assert sv.next_gain_scale(slow, 0.75, sv.DIRECTION_UP) is not None  # same direction continues


def test_next_gain_scale_stops_when_the_step_would_reverse() -> None:
    ft = pytest.importorskip("src.tools.flight_control.flight_test", exc_type=ImportError)
    sv = pytest.importorskip("src.tools.flight_control.sitl_validator", exc_type=ImportError)
    overshoot = sv.score(_synthetic_log(ft, 0.4, ringing=True), 1.0)
    slow = sv.score(_synthetic_log(ft, 2.0), 1.0)
    assert sv.next_gain_scale(overshoot, 1.0, sv.DIRECTION_UP) is None
    assert sv.next_gain_scale(slow, 0.75, sv.DIRECTION_DOWN) is None


def test_next_gain_scale_stops_instead_of_exceeding_the_maximum() -> None:
    ft = pytest.importorskip("src.tools.flight_control.flight_test", exc_type=ImportError)
    sv = pytest.importorskip("src.tools.flight_control.sitl_validator", exc_type=ImportError)
    slow = sv.score(_synthetic_log(ft, 2.0), 1.0)
    assert sv.MAX_GAIN_SCALE == pytest.approx(1 / 0.75**2)
    assert sv.next_gain_scale(slow, 1 / 0.75, sv.DIRECTION_UP) == (pytest.approx(1 / 0.75**2), sv.DIRECTION_UP)
    assert sv.next_gain_scale(slow, 1 / 0.75 / 0.75, sv.DIRECTION_UP) is None


def test_retry_loop_stops_at_the_maximum_gain_scale(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ft = pytest.importorskip("src.tools.flight_control.flight_test", exc_type=ImportError)
    sv = pytest.importorskip("src.tools.flight_control.sitl_validator", exc_type=ImportError)
    scales: list[float] = []

    def fake_attempt(model: Any, params: Any, work: Path, scale: float, config: Any) -> Any:
        scales.append(scale)
        log = _synthetic_log(ft, 2.0)  # always settling-only
        return sv.score(log, scale), log

    monkeypatch.setattr(sv, "_attempt", fake_attempt)
    monkeypatch.setattr(sv, "sitl_available", lambda binary: True)
    fcs = sv.generate_and_verify_fcs(*load_design("g1"), tmp_path)
    assert scales == pytest.approx([1.0] * 3 + [1 / 0.75] * 3 + [1 / 0.75**2] * 3)
    assert not fcs.passed and fcs.attempts == 3


def test_retry_loop_raises_gains_after_a_settling_only_failure(monkeypatch: pytest.MonkeyPatch,
                                                              tmp_path: Path) -> None:
    ft = pytest.importorskip("src.tools.flight_control.flight_test", exc_type=ImportError)
    sv = pytest.importorskip("src.tools.flight_control.sitl_validator", exc_type=ImportError)
    scales: list[float] = []

    def fake_attempt(model: Any, params: Any, work: Path, scale: float, config: Any) -> Any:
        scales.append(scale)
        log = _synthetic_log(ft, 2.0 if scale <= 1.0 else 0.4)
        return sv.score(log, scale), log

    monkeypatch.setattr(sv, "_attempt", fake_attempt)
    monkeypatch.setattr(sv, "sitl_available", lambda binary: True)
    fcs = sv.generate_and_verify_fcs(*load_design("g1"), tmp_path)
    assert scales == pytest.approx([1.0] * 3 + [1 / 0.75] * 3)
    assert fcs.passed and fcs.attempts == 2 and fcs.gain_scale == pytest.approx(1 / 0.75)


def test_retry_loop_stops_when_no_single_scale_satisfies_both_corners(monkeypatch: pytest.MonkeyPatch,
                                                                     tmp_path: Path) -> None:
    ft = pytest.importorskip("src.tools.flight_control.flight_test", exc_type=ImportError)
    sv = pytest.importorskip("src.tools.flight_control.sitl_validator", exc_type=ImportError)
    scales: list[float] = []

    def fake_attempt(model: Any, params: Any, work: Path, scale: float, config: Any) -> Any:
        scales.append(scale)
        log = _synthetic_log(ft, 0.4, ringing=True) if scale > 1.0 else _synthetic_log(ft, 2.0)
        return sv.score(log, scale), log

    monkeypatch.setattr(sv, "_attempt", fake_attempt)
    monkeypatch.setattr(sv, "sitl_available", lambda binary: True)
    fcs = sv.generate_and_verify_fcs(*load_design("g1"), tmp_path)
    assert scales == pytest.approx([1.0] * 3 + [1 / 0.75] * 3)  # settling up, then overshoot would reverse: stop
    assert not fcs.passed and fcs.attempts == 2


def test_a_failed_run_ships_the_least_severe_attempt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ft = pytest.importorskip("src.tools.flight_control.flight_test", exc_type=ImportError)
    sv = pytest.importorskip("src.tools.flight_control.sitl_validator", exc_type=ImportError)

    def fake_attempt(model: Any, params: Any, work: Path, scale: float, config: Any) -> Any:
        # scale 1.0 settles slightly over 3 s; scale 1.333 overshoots ~60 %, then the down-step would reverse
        log = _synthetic_log(ft, 1.1) if scale <= 1.0 else _synthetic_log(ft, 0.4, ringing=True)
        return sv.score(log, scale), log

    monkeypatch.setattr(sv, "_attempt", fake_attempt)
    monkeypatch.setattr(sv, "sitl_available", lambda binary: True)
    inputs = load_design("g1")
    fcs = sv.generate_and_verify_fcs(*inputs, tmp_path)
    assert not fcs.passed and fcs.attempts == 2
    assert fcs.gain_scale == pytest.approx(1.0) and 3.0 <= fcs.roll_settling_s < 3.6 and fcs.roll_overshoot < 0.2
    assert json.loads((tmp_path / "sitl_report.json").read_text())["replay_attempt"] == 1
    header = (tmp_path / "ardupilot.param").read_text().splitlines()[:3]
    assert any("rate-gain scale 1.000" in line for line in header)
    shipped = pg.read_param_file(tmp_path / "ardupilot.param")
    model = build_flight_model(*inputs)
    assert shipped["RLL_RATE_P"] == pytest.approx(pg.fixed_wing_gains(model, gain_scale=1.0)["RLL_RATE_P"], rel=1e-4)


def test_equally_severe_failed_attempts_ship_the_lower_gain_scale() -> None:
    ft = pytest.importorskip("src.tools.flight_control.flight_test", exc_type=ImportError)
    sv = pytest.importorskip("src.tools.flight_control.sitl_validator", exc_type=ImportError)
    log = _synthetic_log(ft, 2.0)
    flown = [(sv.score(log, 1.0), log, {}, 1), (sv.score(log, 1 / 0.75), log, {}, 2)]
    assert sv._shipped_attempt(flown)[3] == 1


def test_retry_loop_scales_gains_by_0_75_until_it_passes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ft = pytest.importorskip("src.tools.flight_control.flight_test", exc_type=ImportError)
    sv = pytest.importorskip("src.tools.flight_control.sitl_validator", exc_type=ImportError)
    scales: list[float] = []

    def fake_attempt(model: Any, params: Any, work: Path, scale: float, config: Any) -> Any:
        scales.append(scale)
        failing = len(scales) <= 6  # three flights per attempt
        log = _synthetic_log(ft, 0.4, ringing=True) if failing else _synthetic_log(ft, 0.4)
        return sv.score(log, scale), log

    monkeypatch.setattr(sv, "_attempt", fake_attempt)
    monkeypatch.setattr(sv, "sitl_available", lambda binary: True)
    fcs = sv.generate_and_verify_fcs(*load_design("g1"), tmp_path)
    assert scales == pytest.approx([1.0] * 3 + [0.75] * 3 + [0.5625] * 3)
    assert fcs.passed and fcs.attempts == 3 and fcs.gain_scale == pytest.approx(0.5625)
    report = json.loads((tmp_path / "sitl_report.json").read_text())
    assert report["passed"] and len(report["attempts"]) == 3
    assert report["replay_attempt"] == 3
    assert pg.read_param_file(tmp_path / "ardupilot.param")["RLL_RATE_FF"] > 0


def test_validator_raises_when_no_flight_completes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sv = pytest.importorskip("src.tools.flight_control.sitl_validator", exc_type=ImportError)

    def crash(*args: Any) -> Any:
        return sv.AttemptResult(gain_scale=args[3], passed=False, error="crash"), None

    monkeypatch.setattr(sv, "_attempt", crash)
    monkeypatch.setattr(sv, "sitl_available", lambda binary: True)
    with pytest.raises(sv.SitlValidationError, match="crash"):
        sv.generate_and_verify_fcs(*load_design("g1"), tmp_path, max_attempts=2)
    report = json.loads((tmp_path / "sitl_report.json").read_text())
    assert report["passed"] is False and report["replay_attempt"] is None


def test_validator_says_no_nominal_flight_completed_when_only_perturbed_plants_flew(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ft = pytest.importorskip("src.tools.flight_control.flight_test", exc_type=ImportError)
    sv = pytest.importorskip("src.tools.flight_control.sitl_validator", exc_type=ImportError)

    def fake_attempt(model: Any, params: Any, work: Path, scale: float, config: Any) -> Any:
        if work.name.startswith("attempt_"):  # the nominal flight always crashes
            return sv.AttemptResult(gain_scale=scale, passed=False, error="SitlFlightError: stall"), None
        log = _synthetic_log(ft, 0.4)
        return sv.score(log, scale), log

    monkeypatch.setattr(sv, "_attempt", fake_attempt)
    monkeypatch.setattr(sv, "sitl_available", lambda binary: True)
    with pytest.raises(sv.SitlValidationError, match="no nominal SITL flight completed"):
        sv.generate_and_verify_fcs(*load_design("g1"), tmp_path, max_attempts=2)


def test_worst_case_takes_each_metric_from_its_worst_case() -> None:
    ft = pytest.importorskip("src.tools.flight_control.flight_test", exc_type=ImportError)
    sv = pytest.importorskip("src.tools.flight_control.sitl_validator", exc_type=ImportError)
    nominal = sv.score(_synthetic_log(ft, 0.4, transition_dip_m=2.0), 1.0)
    hot = sv.score(_synthetic_log(ft, 0.4, ringing=True, ring_decay_s=0.5), 1.0)  # rings: overshoot under 20 %
    slow = sv.score(_synthetic_log(ft, 0.9), 1.0)  # slower roll: longer settling, still < 3 s
    assert 0.1 < hot.roll.overshoot < 0.2 and hot.roll.settling_s < slow.roll.settling_s < 3.0
    combined = sv.combine(1.0, [sv.CaseScore("nominal", nominal), sv.CaseScore("hot", hot),
                                sv.CaseScore("sluggish", slow)])
    assert combined.roll.overshoot == pytest.approx(hot.roll.overshoot)  # from hot
    assert combined.roll.settling_s == pytest.approx(slow.roll.settling_s)  # from sluggish
    larger = max(("hot", hot), ("sluggish", slow), key=lambda c: sv._case_severity(c[1]))[0]
    assert combined.governing_case == larger
    assert combined.transition_alt_loss_m == pytest.approx(2.0) and combined.passed


def test_a_crashed_perturbed_flight_fails_the_attempt() -> None:
    ft = pytest.importorskip("src.tools.flight_control.flight_test", exc_type=ImportError)
    sv = pytest.importorskip("src.tools.flight_control.sitl_validator", exc_type=ImportError)
    good = sv.score(_synthetic_log(ft, 0.4), 1.0)
    crash = sv.AttemptResult(gain_scale=1.0, passed=False, error="SitlFlightError: stall")
    combined = sv.combine(1.0, [sv.CaseScore("nominal", good), sv.CaseScore("hot", crash),
                                sv.CaseScore("sluggish", good)])
    assert not combined.passed and combined.governing_case == "hot" and "hot: SitlFlightError" in combined.error


def test_a_crashed_nominal_flight_fails_with_no_transition_loss() -> None:
    ft = pytest.importorskip("src.tools.flight_control.flight_test", exc_type=ImportError)
    sv = pytest.importorskip("src.tools.flight_control.sitl_validator", exc_type=ImportError)
    good = sv.score(_synthetic_log(ft, 0.4, transition_dip_m=2.0), 1.0)
    crash = sv.AttemptResult(gain_scale=1.0, passed=False, error="SitlFlightError: stall")
    combined = sv.combine(1.0, [sv.CaseScore("nominal", crash), sv.CaseScore("hot", good)])
    assert not combined.passed and combined.transition_alt_loss_m == 0.0
    assert combined.governing_case == "nominal" and "nominal: SitlFlightError" in combined.error


def test_every_attempt_flies_nominal_and_both_perturbed_plants(monkeypatch: pytest.MonkeyPatch,
                                                              tmp_path: Path) -> None:
    ft = pytest.importorskip("src.tools.flight_control.flight_test", exc_type=ImportError)
    sv = pytest.importorskip("src.tools.flight_control.sitl_validator", exc_type=ImportError)
    flights: list[tuple[str, float, int, float]] = []
    params_seen: list[Any] = []

    def fake_attempt(model: Any, params: Any, work: Path, scale: float, config: Any) -> Any:
        flights.append((work.name, model.static_margin, id(params), scale))
        params_seen.append(params)
        slow_first = scale == 1.0 and work.name == "sluggish"
        log = _synthetic_log(ft, 0.4, ringing=slow_first)  # overshoot on the sluggish plant: gains go down
        return sv.score(log, scale), log

    monkeypatch.setattr(sv, "_attempt", fake_attempt)
    monkeypatch.setattr(sv, "sitl_available", lambda binary: True)
    fcs = sv.generate_and_verify_fcs(*load_design("g1"), tmp_path)
    assert len(flights) == 6
    assert [f[0] for f in flights] == ["attempt_1", "hot", "sluggish", "attempt_2", "hot", "sluggish"]
    assert len({f[2] for f in flights[:3]}) == 1 and len({f[2] for f in flights[3:]}) == 1
    assert [f[3] for f in flights] == pytest.approx([1.0] * 3 + [0.75] * 3)
    assert flights[1][1] == pytest.approx(0.08) and flights[2][1] == pytest.approx(0.14)
    assert fcs.attempts == 2 and fcs.governing_case == "nominal"
    assert len(fcs.perturbation_cases) == 3
    report = json.loads((tmp_path / "sitl_report.json").read_text())
    assert len(report["attempts"][0]["cases"]) == 3 and report["replay_attempt"] == 2
    header = (tmp_path / "ardupilot.param").read_text().splitlines()[:3]
    assert any(f"worst case: {fcs.governing_case}" in line for line in header)


def test_combine_rejects_an_empty_case_list() -> None:
    sv = pytest.importorskip("src.tools.flight_control.sitl_validator", exc_type=ImportError)
    with pytest.raises(ValueError, match="combine needs at least one case"):
        sv.combine(1.0, [])


def test_param_header_says_unverified_only_when_nominal_did_not_fly() -> None:
    ft = pytest.importorskip("src.tools.flight_control.flight_test", exc_type=ImportError)
    sv = pytest.importorskip("src.tools.flight_control.sitl_validator", exc_type=ImportError)
    req = load_design("g1")[0]
    good = sv.score(_synthetic_log(ft, 0.4), 1.0)
    crash = sv.AttemptResult(gain_scale=1.0, passed=False, error="SitlFlightError: stall")
    hot_down = sv.combine(1.0, [sv.CaseScore("nominal", good), sv.CaseScore("hot", crash)])
    nominal_down = sv.combine(1.0, [sv.CaseScore("nominal", crash), sv.CaseScore("hot", good)])
    hot_header, nominal_header = sv._param_header(req, hot_down), sv._param_header(req, nominal_down)
    assert "FAILED" in hot_header and "unverified" not in hot_header and "worst case: hot" in hot_header
    assert "FAILED (unverified)" in nominal_header and "worst case: nominal" in nominal_header


def test_a_plant_that_always_crashes_fails_end_to_end(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ft = pytest.importorskip("src.tools.flight_control.flight_test", exc_type=ImportError)
    sv = pytest.importorskip("src.tools.flight_control.sitl_validator", exc_type=ImportError)

    def fake_attempt(model: Any, params: Any, work: Path, scale: float, config: Any) -> Any:
        if work.name == "hot":
            return sv.AttemptResult(gain_scale=scale, passed=False, error="SitlFlightError: stall"), None
        log = _synthetic_log(ft, 0.4)
        return sv.score(log, scale), log

    monkeypatch.setattr(sv, "_attempt", fake_attempt)
    monkeypatch.setattr(sv, "sitl_available", lambda binary: True)
    fcs = sv.generate_and_verify_fcs(*load_design("g1"), tmp_path, max_attempts=2)
    assert fcs.passed is False and fcs.governing_case == "hot"
    hot = next(c for c in fcs.perturbation_cases if c.case == "hot")
    assert hot.error
    assert json.loads((tmp_path / "sitl_report.json").read_text())["replay_attempt"] == 2
    header = (tmp_path / "ardupilot.param").read_text().splitlines()[:3]
    assert any("FAILED" in line for line in header) and not any("unverified" in line for line in header)


def test_old_fcs_records_validate_with_nominal_governing() -> None:
    payload = {"param_path": "p", "sitl_report_path": "r", "passed": True, "roll_overshoot": 0.1,
               "pitch_overshoot": 0.1, "roll_settling_s": 1.0, "pitch_settling_s": 1.0,
               "max_transition_altitude_loss_m": 0.0}
    fcs = FCSArtifacts.model_validate(payload)
    assert fcs.governing_case == "nominal" and fcs.perturbation_cases == ()


def test_validator_requires_arduplane(tmp_path: Path) -> None:
    sv = pytest.importorskip("src.tools.flight_control.sitl_validator", exc_type=ImportError)
    with pytest.raises(sv.SitlUnavailableError):
        sv.generate_and_verify_fcs(*load_design("g1"), tmp_path, config=sv.SitlConfig(binary="no-such-arduplane"))


# ------------------------------------------------------------- SITL acceptance ---
@pytest.mark.docker
@pytest.mark.parametrize("case", ["g1", "g2"])
def test_sitl_acceptance(case: str, tmp_path: Path) -> None:
    """SRS M4 gate: ardupilot.param flown in arduplane + JSBSim through take-off (VTOL climb and transition
    for the quadplane) and +15 deg roll / +10 deg pitch steps: overshoot < 20 %, settling < 3 s, and
    < 15 m altitude loss in the transition."""
    sv = pytest.importorskip("src.tools.flight_control.sitl_validator", exc_type=ImportError)
    if not sv.sitl_available():
        pytest.skip("arduplane not installed (run in the Docker image)")
    fcs = sv.generate_and_verify_fcs(*load_design(case), tmp_path)
    assert fcs.passed, json.loads(Path(fcs.sitl_report_path).read_text())["attempts"]
    assert fcs.roll_overshoot < 0.20 and fcs.pitch_overshoot < 0.20
    assert fcs.roll_settling_s < 3.0 and fcs.pitch_settling_s < 3.0
    assert fcs.max_transition_altitude_loss_m < 15.0
    report = json.loads(Path(fcs.sitl_report_path).read_text())
    truth = json.loads((tmp_path / "sitl" / f"attempt_{report['replay_attempt']}" / "jsbsim_truth.json").read_text())
    assert {"heading_deg", "north_m", "east_m"} <= set(truth[-1])
    params = pg.read_param_file(Path(fcs.param_path))
    assert set(params) <= KNOWN_PARAMS and ("Q_ENABLE" in params) == (case == "g2")
    last = report["attempts"][-1]
    assert len(last["cases"]) == 3
    assert (tmp_path / "sitl" / f"attempt_{len(report['attempts'])}" / "hot" / "jsbsim_truth.json").exists()


@pytest.mark.docker
def test_far_out_of_range_plant_fails_the_gate(tmp_path: Path) -> None:
    """The perturbation check can fail: nominal gains on a plant with 0.3x control power are too slow to settle."""
    sv = pytest.importorskip("src.tools.flight_control.sitl_validator", exc_type=ImportError)
    if not sv.sitl_available():
        pytest.skip("arduplane not installed (run in the Docker image)")
    req, sizing, bom, aero, cad = load_design("g1")
    model = build_flight_model(req, sizing, bom, aero, cad)
    params = pg.generate_params(model, sizing, bom)
    extreme = perturbed_model(model, PlantCase("extreme", 0.3, 1.0, 1.0, model.static_margin))
    result, _ = sv._attempt(extreme, params, tmp_path, 1.0, sv.SitlConfig())
    assert not result.passed


# ------------------------------------------------------------ robustness (review) ---
def test_non_finite_parameters_are_refused(tmp_path: Path) -> None:
    with pytest.raises(pg.ParamGenerationError, match="non-finite"):
        pg.write_param_file({"RLL_RATE_P": float("nan")}, tmp_path / "x.param")


def test_wrong_signed_control_derivative_is_rejected(design: Any) -> None:
    _, (req, sizing, bom, aero, cad), _ = design
    with pytest.raises(FlightModelError, match="non-physical"):
        build_flight_model(req, sizing, bom, aero.model_copy(update={"cl_delta_a_per_rad": -0.2}), cad)


class _SilentConnection:
    """Stands in for a pymavlink connection that never delivers telemetry."""
    target_system = target_component = 1

    class mav:  # noqa: N801 - mirrors pymavlink's attribute name
        @staticmethod
        def rc_channels_override_send(*args: Any) -> None:
            return None

    def recv_match(self, **kwargs: Any) -> None:
        return None


def test_pilot_reports_a_stall_when_simulation_time_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    ft = pytest.importorskip("src.tools.flight_control.flight_test", exc_type=ImportError)
    monkeypatch.setattr(ft, "STALL_TIMEOUT_S", 0.2)
    with pytest.raises(ft.SitlFlightError, match="stalled"):
        ft.SitlPilot(_SilentConnection()).wait(10.0)


def test_pilot_surfaces_bridge_or_process_failure() -> None:
    ft = pytest.importorskip("src.tools.flight_control.flight_test", exc_type=ImportError)
    pilot = ft.SitlPilot(_SilentConnection(), health=lambda: "arduplane exited with code -11")
    with pytest.raises(ft.SitlFlightError, match="code -11"):
        pilot.pump()


def test_runtime_errors_become_failed_attempts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sv = pytest.importorskip("src.tools.flight_control.sitl_validator", exc_type=ImportError)

    def port_in_use(*args: Any) -> Any:
        raise OSError("[Errno 98] Address already in use")

    monkeypatch.setattr(sv, "fly", port_in_use)
    result, log = sv._attempt(None, {}, tmp_path / "a", 1.0, sv.SitlConfig())
    assert log is None and not result.passed and "Address already in use" in result.error


# ---- perturbation ---

@pytest.mark.parametrize("case", [HOT, SLUGGISH], ids=lambda c: c.name)
def test_perturbed_model_scales_only_the_listed_terms(design: Any, case: PlantCase) -> None:
    _, _, model = design
    hot = perturbed_model(model, case)
    a, h = model.aero, hot.aero
    cp, dm, ine = case.control_power, case.damping, case.inertia
    assert h.cl_delta_a == pytest.approx(cp * a.cl_delta_a) and h.cm_delta_e == pytest.approx(cp * a.cm_delta_e)
    assert h.cl_p == pytest.approx(dm * a.cl_p) and h.cm_q == pytest.approx(dm * a.cm_q)
    for name in ("ixx_kgm2", "iyy_kgm2", "izz_kgm2", "ixz_kgm2"):
        assert getattr(hot, name) == pytest.approx(ine * getattr(model, name))
    untouched = {"cl_alpha", "cl0", "cd0", "cn_beta", "cn_r", "cl_beta", "cn_delta_r", "cy_beta"}
    assert all(getattr(h, k) == getattr(a, k) for k in untouched)
    assert hot.lift_rotors == model.lift_rotors and hot.mass_kg == model.mass_kg
    assert model.aero.cl_p == a.cl_p  # input not mutated


def test_perturbed_model_moves_the_cg_to_the_case_margin(design: Any) -> None:
    _, _, model = design
    for case in (HOT, SLUGGISH):
        p = perturbed_model(model, case)
        assert p.static_margin == case.static_margin
        assert p.aero.cm_alpha == pytest.approx(-model.aero.cl_alpha * case.static_margin)
        assert p.aero.cm0 == pytest.approx(model.aero.cm0 + model.aero.cl0 * (model.static_margin - case.static_margin))


def test_a_case_at_the_current_margin_leaves_pitch_stiffness_alone(design: Any) -> None:
    _, _, model = design
    same = PlantCase("same", 1.0, 1.0, 1.0, model.static_margin)
    assert perturbed_model(model, same) == model


def test_aft_cg_is_less_stable_and_forward_cg_more_stable(design: Any) -> None:
    _, _, model = design
    assert perturbed_model(model, HOT).aero.cm_alpha >= model.aero.cm_alpha
    assert perturbed_model(model, SLUGGISH).aero.cm_alpha <= model.aero.cm_alpha


@pytest.mark.parametrize("kwargs", [
    {"control_power": 0.0}, {"damping": -1.0}, {"inertia": float("nan")},
    {"static_margin": float("inf")}, {"static_margin": float("nan")},
    {"control_power": float("inf")}, {"damping": float("inf")}, {"inertia": float("inf")},
])
def test_plant_case_rejects_invalid_values(kwargs: dict[str, float]) -> None:
    valid = {"control_power": 1.0, "damping": 1.0, "inertia": 1.0, "static_margin": 0.1}
    with pytest.raises(ValueError):
        PlantCase("bad", **{**valid, **kwargs})
