"""Headless ArduPilot SITL + JSBSim acceptance test (SRS 4.4 Module D.2) and the damping retry loop.

One attempt: write ardupilot.param (+ simulation-only overrides), then fly it three times: against the
nominal JSBSim plant and against the "hot" and "sluggish" perturbed plants (perturbation.py). Each flight
exports the aircraft, starts the JSON physics bridge and `arduplane --model JSON`, and flies the doublet
sequence over MAVLink (udpin:127.0.0.1:14550). The attempt gates roll/pitch overshoot and settling on the
worst of the three, and the transition altitude loss on the nominal flight.
The perturbed plants change only the roll/pitch control power and damping, the inertia and the static margin;
yaw and the VTOL rotor terms are flown nominal.
`generate_and_verify_fcs` retries with the rate gains scaled two ways (SRS 5.6, `next_gain_scale`): x0.75 after an
overshoot failure or a crashed plant, /0.75 after a settling-only failure, up to MAX_GAIN_SCALE. A step that would
reverse the previous one stops the loop: no scale on the 0.75 grid satisfies both the hot and the sluggish corner.
When no attempt passes, the least severe flown attempt is shipped (R42).
"""

import json
import logging
import math
import subprocess
import threading
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Optional, Sequence

from pymavlink import mavutil

from src.schemas.components import SelectedBOM
from src.schemas.requirements import OperationalRequirements
from src.schemas.state import (AeroStabilityOutput, CADArtifacts, FCSArtifacts, PlantCaseResult,
                               SizingOutput)
from src.tools.flight_control import param_generator as pg
from src.tools.flight_control.flight_model import FlightModel, build_flight_model
from src.tools.flight_control.flight_test import (PITCH_STEP_DEG, STEP_HOLD_S, FlightLog, SitlPilot,
                                                  fly_acceptance_test)
from src.tools.flight_control.jsbsim_exporter import export_jsbsim
from src.tools.flight_control.runtime import (ARDUPLANE_BINARY, MAX_OVERSHOOT, MAX_SETTLING_S,
                                              MAX_TRANSITION_ALT_LOSS_M, SitlUnavailableError,
                                              SitlValidationError, sitl_available)
from src.tools.flight_control.perturbation import NOMINAL_CASE, PERTURBED_CASES, perturbed_model
from src.tools.flight_control.sitl_bridge import Home, JsonPhysicsBridge
from src.tools.flight_control.step_metrics import StepAnalysisError, StepResponse, altitude_loss_m, analyse_step
from src.tools.sizing import layouts
from src.tools.sizing.layouts import StaticMarginBand

logger = logging.getLogger(__name__)

DAMPING_FACTOR: Final[float] = 0.75  # SRS 5.6
MAX_ATTEMPTS: Final[int] = 4  # first run + 3 retries
MAX_GAIN_SCALE: Final[float] = 1.0 / DAMPING_FACTOR**2  # R41: at most two up-steps (~1.78x)
GAIN_SCALE_TOLERANCE: Final[float] = 1e-9  # relative: 1/0.75/0.75 and 1/0.75**2 differ in the last bit
DIRECTION_DOWN: Final[int] = -1  # gains lowered: curbs overshoot and crashes
DIRECTION_UP: Final[int] = 1  # gains raised: speeds up a slow settling
DIRECTION_NONE: Final[int] = 0  # no step taken yet
PARAM_FILE: Final[str] = "ardupilot.param"
SITL_PARAM_FILE: Final[str] = "sitl_overrides.param"
REPORT_FILE: Final[str] = "sitl_report.json"
TRUTH_FILE: Final[str] = "jsbsim_truth.json"  # simulator ground truth of each attempt
TELEMETRY_FILE: Final[str] = "telemetry.json"  # full-rate log of each attempt, kept for diagnosis
TELEMETRY_DECIMATION: Final[int] = 3  # every 3rd 25 Hz sample (~8 Hz) in the report
PROCESS_KILL_TIMEOUT_S: Final[float] = 10.0
LOG_TAIL_LINES: Final[int] = 20
COMBINED_STEP_NOTE: Final[str] = (
    "Each attempt's roll/pitch is the worst over the plants that flew: overshoot and settling_s can come from "
    "different plants, while step_deg, target_deg and peak_deg are the governing case's, so peak_deg need not "
    "match overshoot. The per-plant values are in 'cases'.")
_SITL_LOCK = threading.Lock()  # one SITL at a time: the physics/MAVLink UDP ports are fixed


@dataclass(frozen=True)
class SitlConfig:
    speedup: float = 10.0
    rate_hz: int = 1200  # ArduPilot default; a quadplane's 300 Hz loop needs >= 1.8x that from the IMU
    physics_port: int = 9002  # ArduPilot's JSON backend default
    mavlink_port: int = 14550  # SRS
    home: Home = field(default_factory=Home)
    binary: str = ARDUPLANE_BINARY
    extra_params: Mapping[str, float] = field(default_factory=dict)  # SITL-only parameters layered on top (tests)


@dataclass(frozen=True)
class AttemptResult:
    gain_scale: float
    passed: bool
    roll: Optional[StepResponse] = None
    pitch: Optional[StepResponse] = None
    transition_alt_loss_m: float = 0.0
    trim_throttle_pct: Optional[float] = None
    error: Optional[str] = None
    cases: tuple["CaseScore", ...] = ()  # per-plant scores behind a combined attempt
    governing_case: str = NOMINAL_CASE


@dataclass(frozen=True)
class CaseScore:
    case: str
    result: AttemptResult


def _launch(config: SitlConfig, work_dir: Path) -> "subprocess.Popen[bytes]":
    command: list[str] = [
        config.binary, "--wipe", "--model", "JSON:127.0.0.1", "--speedup", str(config.speedup),
        "--rate", str(config.rate_hz), "--home", config.home.arg(), "--defaults", f"{PARAM_FILE},{SITL_PARAM_FILE}",
        "--serial0", f"udpclient:127.0.0.1:{config.mavlink_port}",
    ]
    with (work_dir / "arduplane.log").open("wb") as log:  # the child keeps its own descriptor
        return subprocess.Popen(command, cwd=work_dir, stdout=log, stderr=subprocess.STDOUT)


def climb_throttle_pct(model: FlightModel) -> float:
    """Extra throttle to climb at +PITCH_STEP_DEG at cruise speed: dT = W sin(step) / T_available(V)."""
    extra_n: float = model.weight_n * math.sin(math.radians(PITCH_STEP_DEG))
    return 100.0 * extra_n / model.cruise_thrust_n(1.0, model.v_cruise_ms)


def _stop(proc: "subprocess.Popen[bytes]") -> None:
    proc.kill()
    try:
        proc.wait(timeout=PROCESS_KILL_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        logger.error("arduplane (pid %d) did not exit after SIGKILL", proc.pid)


def _log_tail(work_dir: Path) -> str:
    path: Path = work_dir / "arduplane.log"
    lines: list[str] = path.read_text(errors="replace").splitlines() if path.is_file() else []
    return " | ".join(line for line in lines[-LOG_TAIL_LINES:] if line.strip())


def _health(bridge: JsonPhysicsBridge, proc: "subprocess.Popen[bytes]", work_dir: Path) -> Callable[[], Optional[str]]:
    """Pilot callback: why the simulation can no longer progress (None while healthy)."""
    def check() -> Optional[str]:
        if bridge.stats.fatal is not None:
            return f"JSBSim bridge failed: {bridge.stats.fatal}"
        code: Optional[int] = proc.poll()
        return None if code is None else f"arduplane exited with code {code}: {_log_tail(work_dir)}"
    return check


def _finish(bridge: JsonPhysicsBridge, work_dir: Path) -> None:
    logger.info("SITL bridge: %d frames at %d Hz, errors=%s", bridge.stats.frames, bridge.stats.frame_rate,
                bridge.stats.errors[:3])
    (work_dir / TRUTH_FILE).write_text(json.dumps(bridge.stats.truth))


def fly(model: FlightModel, params: pg.ParamSet, work_dir: Path, config: SitlConfig) -> FlightLog:
    """One SITL flight. Raises SitlFlightError if a phase cannot complete. Every resource is released by
    the ExitStack even if a later cleanup step fails (arduplane is killed before the logs are written)."""
    pg.write_param_file(params, work_dir / PARAM_FILE)
    overrides: pg.ParamSet = pg.sitl_overrides() | dict(config.extra_params)
    pg.write_param_file(overrides, work_dir / SITL_PARAM_FILE, header="SITL-only overrides")
    jsb = export_jsbsim(model, work_dir / "jsbsim")
    merged: pg.ParamSet = params | overrides
    with _SITL_LOCK, JsonPhysicsBridge(jsb, config.home, config.physics_port) as bridge, ExitStack() as stack:
        stack.callback(_finish, bridge, work_dir)
        proc = _launch(config, work_dir)
        stack.callback(_stop, proc)  # guard: arduplane dies even if the MAVLink socket cannot open
        conn = mavutil.mavlink_connection(f"udpin:127.0.0.1:{config.mavlink_port}", source_system=255)
        stack.callback(conn.close)
        pilot = SitlPilot(conn, health=_health(bridge, proc, work_dir))
        stack.callback(_dump_log, pilot.log, work_dir / TELEMETRY_FILE)
        stack.callback(_stop, proc)  # LIFO: runs first, so arduplane stops before the logs are written
        return fly_acceptance_test(pilot, merged, model.is_quadplane, climb_throttle_pct(model))


def _dump_log(log: FlightLog, path: Path) -> None:
    path.write_text(json.dumps({"texts": log.texts, "samples": [asdict(s) for s in log.samples], "pid": log.pid}))


def _step(log: FlightLog, attr: str, step_t: Optional[float]) -> StepResponse:
    if step_t is None:
        raise StepAnalysisError(f"no {attr} step was flown")
    s = log.samples
    return analyse_step([x.t_s for x in s], [getattr(x, f"{attr}_deg") for x in s],
                        [getattr(x, f"nav_{attr}_deg") for x in s], step_t, STEP_HOLD_S)


def score(log: FlightLog, gain_scale: float) -> AttemptResult:
    roll = _step(log, "roll", log.roll_step_t_s)
    pitch = _step(log, "pitch", log.pitch_step_t_s)
    loss: float = 0.0
    if log.transition_start_t_s is not None and log.transition_start_alt_m is not None:
        end: float = log.transition_end_t_s if log.transition_end_t_s is not None else log.samples[-1].t_s
        segment: list[float] = [x.alt_m for x in log.samples if log.transition_start_t_s <= x.t_s <= end]
        loss = altitude_loss_m(segment, log.transition_start_alt_m)
    passed: bool = (roll.passes(MAX_OVERSHOOT, MAX_SETTLING_S) and pitch.passes(MAX_OVERSHOOT, MAX_SETTLING_S)
                    and loss < MAX_TRANSITION_ALT_LOSS_M)
    return AttemptResult(gain_scale=gain_scale, passed=passed, roll=roll, pitch=pitch, transition_alt_loss_m=loss,
                         trim_throttle_pct=log.trim_throttle_pct)


def _ratio(step: StepResponse) -> float:
    return max(step.overshoot / MAX_OVERSHOOT, step.settling_s / MAX_SETTLING_S)


def _case_severity(result: AttemptResult) -> float:
    if result.roll is None or result.pitch is None:
        return math.inf  # a crashed flight is the worst case
    return max(_ratio(result.roll), _ratio(result.pitch))


def _worst_step(results: Sequence[AttemptResult], attr: str, governing: AttemptResult) -> Optional[StepResponse]:
    """Max overshoot and max settling over the completed cases; step geometry from the governing case.

    The result is a synthetic mix (COMBINED_STEP_NOTE): peak_deg is the governing case's and need not match the
    overshoot, which may come from another plant."""
    steps: list[StepResponse] = [s for s in (getattr(r, attr) for r in results) if s is not None]
    if not steps:
        return None
    base: StepResponse = getattr(governing, attr) or steps[0]
    return StepResponse(step_deg=base.step_deg, target_deg=base.target_deg, peak_deg=base.peak_deg,
                        overshoot=max(s.overshoot for s in steps), settling_s=max(s.settling_s for s in steps))


def combine(gain_scale: float, scores: Sequence[CaseScore]) -> AttemptResult:
    """Worst-case roll/pitch over all plants; transition loss from the nominal flight only.

    A crashed plant has no step metrics: the worst values cover the plants that completed a flight, while the
    crashed plant governs (its error is carried as "<case>: <error>") and the attempt fails."""
    if not scores:
        raise ValueError("combine needs at least one case")
    results: list[AttemptResult] = [c.result for c in scores]
    governing: CaseScore = max(scores, key=lambda c: _case_severity(c.result))
    roll = _worst_step(results, "roll", governing.result)
    pitch = _worst_step(results, "pitch", governing.result)
    nominal: Optional[AttemptResult] = next((c.result for c in scores if c.case == NOMINAL_CASE), None)
    loss: float = nominal.transition_alt_loss_m if nominal is not None else 0.0
    errors: list[str] = [f"{c.case}: {c.result.error}" for c in scores if c.result.error]
    all_done: bool = all(c.result.roll is not None and c.result.pitch is not None for c in scores)
    passed: bool = (all_done and roll is not None and pitch is not None
                    and roll.passes(MAX_OVERSHOOT, MAX_SETTLING_S) and pitch.passes(MAX_OVERSHOOT, MAX_SETTLING_S)
                    and loss < MAX_TRANSITION_ALT_LOSS_M)
    return AttemptResult(gain_scale=gain_scale, passed=passed, roll=roll, pitch=pitch, transition_alt_loss_m=loss,
                         trim_throttle_pct=nominal.trim_throttle_pct if nominal else None,
                         error="; ".join(errors) or None, cases=tuple(scores), governing_case=governing.case)


def _report(attempts: list[AttemptResult], final_log: Optional[FlightLog], params: pg.ParamSet,
            replay_attempt: Optional[int]) -> dict[str, Any]:
    return {
        "passed": any(a.passed for a in attempts),
        "replay_attempt": replay_attempt,  # attempt number whose sitl/attempt_N trace is the final flight
        "combined_step_note": COMBINED_STEP_NOTE,
        "thresholds": {"max_overshoot": MAX_OVERSHOOT, "max_settling_s": MAX_SETTLING_S,
                       "max_transition_alt_loss_m": MAX_TRANSITION_ALT_LOSS_M},
        "attempts": [asdict(a) for a in attempts],
        "final_params": params,
        "status_texts": [{"t_s": t, "text": text} for t, text in (final_log.texts if final_log else [])],
        "telemetry": [asdict(s) for s in final_log.samples[::TELEMETRY_DECIMATION]] if final_log else [],
    }


def _attempt(model: FlightModel, params: pg.ParamSet, work: Path, scale: float,
             config: SitlConfig) -> tuple[AttemptResult, Optional[FlightLog]]:
    """Fly and score one attempt. Runtime failures (flight phase timeouts, a crashed arduplane or bridge,
    UDP/JSBSim errors, unscoreable telemetry) become a failed attempt so the retry loop can continue."""
    work.mkdir(parents=True, exist_ok=True)
    try:
        log: FlightLog = fly(model, params, work, config)
        return score(log, scale), log
    except (OSError, RuntimeError, StepAnalysisError) as exc:  # SitlFlightError is a RuntimeError
        logger.warning("SITL attempt (gain scale %.3f) failed: %s", scale, exc)
        return AttemptResult(gain_scale=scale, passed=False, error=f"{type(exc).__name__}: {exc}"), None


def _fly_cases(model: FlightModel, params: pg.ParamSet, work: Path, scale: float,
               config: SitlConfig, band: StaticMarginBand) -> tuple[AttemptResult, Optional[FlightLog]]:
    """Fly one parameter set against the nominal plant (in `work`) and each perturbed plant (`work/<case>`).
    Returns the combined attempt and the nominal log, which is the one replayed in the viewer."""
    nominal, nominal_log = _attempt(model, params, work, scale, config)
    scores: list[CaseScore] = [CaseScore(NOMINAL_CASE, nominal)]
    for case in PERTURBED_CASES:
        result, _ = _attempt(perturbed_model(model, case, band), params, work / case.name, scale, config)
        scores.append(CaseScore(case.name, result))
    return combine(scale, scores), nominal_log


def _case_results(result: AttemptResult) -> tuple[PlantCaseResult, ...]:
    """Per-plant metrics for the saved design; a crashed flight records zeros plus its error."""
    def metrics(score_: AttemptResult) -> dict[str, float]:
        roll, pitch = score_.roll, score_.pitch
        return {"roll_overshoot": roll.overshoot if roll else 0.0, "roll_settling_s": roll.settling_s if roll else 0.0,
                "pitch_overshoot": pitch.overshoot if pitch else 0.0,
                "pitch_settling_s": pitch.settling_s if pitch else 0.0}
    return tuple(PlantCaseResult(case=c.case, error=c.result.error, **metrics(c.result)) for c in result.cases)


def _param_header(req: OperationalRequirements, result: AttemptResult) -> str:
    nominal_crashed: bool = any(c.case == NOMINAL_CASE and c.result.error for c in result.cases) or (
        not result.cases and result.error is not None)
    status: str = "PASSED" if result.passed else "FAILED (unverified)" if nominal_crashed else "FAILED"
    return (f"ISR-Forge ArduPilot Plane 4.7 parameters for '{req.mission_name}'\n"
            f"SITL doublet test {status} at rate-gain scale {result.gain_scale:.3f}, "
            f"worst case: {result.governing_case}")


def next_gain_scale(result: AttemptResult, scale: float, last_direction: int) -> Optional[tuple[float, int]]:
    """Gain scale and direction for the retry after a failed attempt, or None to stop.
    Overshoot at or over the limit, or a crashed plant, lowers the gains; a settling-only failure raises them.
    Reversing the previous step means no scale on the 0.75 grid satisfies both corners, so the loop stops; so does an
    up-step past MAX_GAIN_SCALE (R41)."""
    steps: list[StepResponse] = [x for x in (result.roll, result.pitch) if x is not None]
    needs_less: bool = result.error is not None or any(x.overshoot >= MAX_OVERSHOOT for x in steps)
    needs_more: bool = not needs_less and any(x.settling_s >= MAX_SETTLING_S for x in steps)
    direction: int = DIRECTION_UP if needs_more else DIRECTION_DOWN
    if last_direction == -direction:
        return None
    new_scale: float = scale / DAMPING_FACTOR if direction == DIRECTION_UP else scale * DAMPING_FACTOR
    if new_scale > MAX_GAIN_SCALE * (1.0 + GAIN_SCALE_TOLERANCE):
        return None
    return new_scale, direction


FlownAttempt = tuple[AttemptResult, FlightLog, pg.ParamSet, int]  # result, nominal log, params, attempt number


def _attempt_severity(result: AttemptResult) -> float:
    """R42: the governing case's largest limit ratio; a crashed plant makes the attempt infinitely severe."""
    return math.inf if result.error is not None else _case_severity(result)


def _shipped_attempt(flown: Sequence[FlownAttempt]) -> FlownAttempt:
    """The passing attempt if there is one, else the least severe flown attempt, ties to the lower gain scale."""
    passing: list[FlownAttempt] = [f for f in flown if f[0].passed]
    if passing:
        return passing[-1]
    return min(flown, key=lambda f: (_attempt_severity(f[0]), f[0].gain_scale))


def generate_and_verify_fcs(req: OperationalRequirements, sizing: SizingOutput, bom: SelectedBOM,
                            aero: AeroStabilityOutput, cad: CADArtifacts, out_dir: Path,
                            config: SitlConfig = SitlConfig(), max_attempts: int = MAX_ATTEMPTS) -> FCSArtifacts:
    """Generate ardupilot.param, fly it in SITL and retry with two-way gain steps (`next_gain_scale`).

    The loop ends when an attempt passes, a step would reverse or pass MAX_GAIN_SCALE, or attempts run out.
    Writes out_dir/ardupilot.param and out_dir/sitl_report.json for the shipped attempt: the passing one, else the
    least severe attempt whose nominal flight completed (R42, `_shipped_attempt`)."""
    if not sitl_available(config.binary):
        raise SitlUnavailableError(f"{config.binary} not found: run inside the ISR-Forge Docker image")
    model: FlightModel = build_flight_model(req, sizing, bom, aero, cad)
    attempts: list[AttemptResult] = []
    band: StaticMarginBand = layouts.static_margin_band(sizing.airframe_layout)
    flown: list[FlownAttempt] = []
    params: pg.ParamSet = {}
    scale: float = 1.0
    direction: int = DIRECTION_NONE
    for index in range(max_attempts):
        params = pg.generate_params(model, sizing, bom, gain_scale=scale)
        result, log = _fly_cases(model, params, out_dir / "sitl" / f"attempt_{index + 1}", scale, config, band)
        attempts.append(result)
        if log is not None:
            flown.append((result, log, params, index + 1))
        if result.passed:
            break
        step = next_gain_scale(result, scale, direction)
        if step is None:
            break
        scale, direction = step
    report_path: Path = out_dir / REPORT_FILE
    if not flown:  # still leave the generated (unverified) parameters and the attempt log behind
        report_path.write_text(json.dumps(_report(attempts, None, params, None), indent=2))
        pg.write_param_file(params, out_dir / PARAM_FILE, header=_param_header(req, attempts[-1]))
        raise SitlValidationError(f"no nominal SITL flight completed: {attempts[-1].error}")
    best, best_log, best_params, best_number = _shipped_attempt(flown)
    report_path.write_text(json.dumps(_report(attempts, best_log, best_params, best_number), indent=2))
    param_path: Path = pg.write_param_file(best_params, out_dir / PARAM_FILE, header=_param_header(req, best))
    if best.roll is None or best.pitch is None:  # score() always sets both; guard against future changes
        raise SitlValidationError("a scored SITL attempt is missing its step responses")
    return FCSArtifacts(param_path=str(param_path), sitl_report_path=str(report_path), passed=best.passed,
                        roll_overshoot=best.roll.overshoot, pitch_overshoot=best.pitch.overshoot,
                        roll_settling_s=best.roll.settling_s, pitch_settling_s=best.pitch.settling_s,
                        max_transition_altitude_loss_m=best.transition_alt_loss_m, gain_scale=best.gain_scale,
                        attempts=len(attempts), governing_case=best.governing_case,
                        perturbation_cases=_case_results(best))
