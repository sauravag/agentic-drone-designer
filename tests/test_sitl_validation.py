"""Milestone 4: SITL validator (retry loop, perturbed plants, pilot) and the perturbation cases.

Split from test_flight_control.py, whose fixtures and helpers it reuses. The acceptance tests marked `docker`
need the arduplane binary:
    docker compose run --rm isr-forge pytest -q tests/test_sitl_validation.py
"""

import json
import math
from pathlib import Path
from typing import Any

import pytest

from src.schemas.requirements import OperationalRequirements
from src.schemas.state import FCSArtifacts
from src.tools.flight_control import param_generator as pg
from src.tools.flight_control.flight_model import FlightModelError, build_flight_model
from src.tools.flight_control.perturbation import HOT, SLUGGISH, PlantCase, perturbed_model
from src.tools.sizing.layouts import TAILED_BAND, StaticMarginBand
from src.ui.examples import G1_ELECTRIC_FW
from tests.test_flight_control import KNOWN_PARAMS, design, load_design  # noqa: F401 (design is a fixture)

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


def test_retry_loop_stops_when_the_gain_step_would_reverse(monkeypatch: pytest.MonkeyPatch,
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
    assert "different plants" in report["combined_step_note"]  # the combined roll/pitch is a worst-case mix
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


def test_sitl_config_freezes_a_plain_dict_of_extra_params() -> None:
    from src.tools.flight_control import sitl_validator as sv
    extra = {"SERVO2_REVERSED": 1}
    config = sv.SitlConfig(extra_params=extra)
    extra["SERVO2_REVERSED"] = 0
    assert config.extra_params["SERVO2_REVERSED"] == 1
    with pytest.raises(TypeError):
        config.extra_params["X"] = 1  # type: ignore[index]


@pytest.mark.docker
def test_reversed_elevator_fails_sitl(tmp_path: Path) -> None:
    """A wrong servo reversal must fail SITL: ArduPilot does the mixing, the bridge never un-mixes it."""
    sv = pytest.importorskip("src.tools.flight_control.sitl_validator", exc_type=ImportError)
    if not sv.sitl_available():
        pytest.skip("arduplane not installed (run in the Docker image)")
    req, sizing, bom, aero, cad = load_design("g1")
    model = build_flight_model(req, sizing, bom, aero, cad)
    params = pg.generate_params(model, sizing, bom)
    baseline, _ = sv._attempt(model, params, tmp_path / "baseline", 1.0, sv.SitlConfig())
    assert baseline.passed and baseline.error is None, baseline  # same harness and plant flies fine unreversed
    reversed_, _ = sv._attempt(model, params, tmp_path / "reversed", 1.0,
                               sv.SitlConfig(extra_params={"SERVO2_REVERSED": 1}))
    # a diverged or crashed flight is a failed attempt, and the error names a flight failure, not a setup problem
    assert not reversed_.passed and reversed_.error is not None
    assert "SitlFlightError" in reversed_.error
    assert "Floating point exception" in reversed_.error or "timed out" in reversed_.error  # divergence / no climb


@pytest.mark.docker
def test_far_out_of_range_plant_fails_the_gate(tmp_path: Path) -> None:
    """The perturbation check can fail: nominal gains on a plant with 0.3x control power are too slow to settle."""
    sv = pytest.importorskip("src.tools.flight_control.sitl_validator", exc_type=ImportError)
    if not sv.sitl_available():
        pytest.skip("arduplane not installed (run in the Docker image)")
    req, sizing, bom, aero, cad = load_design("g1")
    model = build_flight_model(req, sizing, bom, aero, cad)
    params = pg.generate_params(model, sizing, bom)
    extreme = perturbed_model(model, PlantCase("extreme", 0.3, 1.0, 1.0, "minimum"),
                              StaticMarginBand(model.static_margin, model.static_margin, model.static_margin))
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
    hot = perturbed_model(model, case, TAILED_BAND)
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
        margin = getattr(TAILED_BAND, case.margin_end)
        p = perturbed_model(model, case, TAILED_BAND)
        assert p.static_margin == margin
        assert p.aero.cm_alpha == pytest.approx(-model.aero.cl_alpha * margin)
        assert p.aero.cm0 == pytest.approx(model.aero.cm0 + model.aero.cl0 * (model.static_margin - margin))


def test_a_case_at_the_current_margin_leaves_pitch_stiffness_alone(design: Any) -> None:
    _, _, model = design
    same = PlantCase("same", 1.0, 1.0, 1.0, "minimum")
    flat_band = StaticMarginBand(model.static_margin, model.static_margin, model.static_margin)
    assert perturbed_model(model, same, flat_band) == model


def test_aft_cg_is_less_stable_and_forward_cg_more_stable(design: Any) -> None:
    _, _, model = design
    assert perturbed_model(model, HOT, TAILED_BAND).aero.cm_alpha >= model.aero.cm_alpha
    assert perturbed_model(model, SLUGGISH, TAILED_BAND).aero.cm_alpha <= model.aero.cm_alpha


@pytest.mark.parametrize("kwargs", [
    {"control_power": 0.0}, {"damping": -1.0}, {"inertia": float("nan")},
    {"margin_end": "middle"},
    {"control_power": float("inf")}, {"damping": float("inf")}, {"inertia": float("inf")},
])
def test_plant_case_rejects_invalid_values(kwargs: dict[str, float]) -> None:
    valid = {"control_power": 1.0, "damping": 1.0, "inertia": 1.0, "margin_end": "minimum"}
    with pytest.raises(ValueError):
        PlantCase("bad", **{**valid, **kwargs})


@pytest.mark.docker
def test_hand_launched_g1_passes_sitl(tmp_path: Path) -> None:
    """A hand-launched G1, sized for the throw (stall <= throw speed / 1.2) and built through OpenVSP and CAD, has no
    gear: it starts in the air at throw speed, climbs to the take-off altitude without touching the ground, then
    passes the same doublet criteria as every other design."""
    sv = pytest.importorskip("src.tools.flight_control.sitl_validator", exc_type=ImportError)
    if not sv.sitl_available():
        pytest.skip("arduplane not installed (run in the Docker image)")
    pytest.importorskip("openvsp")
    from src.tools.aero_cad.build import build_aero_cad
    from src.tools.database.matcher import select_bom
    from src.tools.database.seed_data import seed_database
    from src.tools.sizing.mission_solver import component_needs, size_aircraft
    req = OperationalRequirements.model_validate({**G1_ELECTRIC_FW.model_dump(), "runway_available": False,
                                                  "launch_method": "hand"})
    sizing = size_aircraft(req)
    bom = select_bom(seed_database(tmp_path / "cots_catalog.db"), component_needs(sizing, req))
    aero, cad = build_aero_cad(sizing, bom, req.service_ceiling_m, tmp_path / "aero_cad")
    fcs = sv.generate_and_verify_fcs(req, sizing, bom, aero, cad, tmp_path)
    assert fcs.passed, json.loads(Path(fcs.sitl_report_path).read_text())["attempts"]
    params = pg.read_param_file(Path(fcs.param_path))
    assert not {name for name in params if name.startswith("TKOFF_")}
    report = json.loads(Path(fcs.sitl_report_path).read_text())
    truth = json.loads((tmp_path / "sitl" / f"attempt_{report['replay_attempt']}" / "jsbsim_truth.json").read_text())
    assert min(sample["agl_m"] for sample in truth) > 0.0  # never touched the ground
