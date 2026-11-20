# SITL Perturbation Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fly every SITL gain attempt against the nominal plant plus a "hot" and a "sluggish" perturbed plant,
with the same parameter file, and gate roll and pitch on the worst of the three.

**Architecture:**
- A pure `perturbation.py` module returns perturbed copies of the frozen `FlightModel`.
- `sitl_validator.generate_and_verify_fcs` flies each attempt's single parameter set against the nominal plant and
  both perturbed plants, then combines the scores.
- `FCSArtifacts` carries the worst case plus the result for each case, and the presenters, warnings and docs show
  them.

**Tech Stack:** Python 3.11, dataclasses, Pydantic v2, pytest, ArduPilot SITL + JSBSim (Docker only).

**Spec:** `docs/superpowers/specs/2026-11-20-sitl-perturbation-check-design.md`

## Global Constraints

- Hot case: control power (`cl_delta_a`, `cm_delta_e`) x1.3, damping (`cl_p`, `cm_q`) x0.7, inertia
  (`ixx`, `iyy`, `izz`, `ixz`) x0.8, static margin `SM_MIN` = 0.08.
- Sluggish case: control power x0.7, damping x1.3, inertia x1.2, static margin `SM_MAX` = 0.14.
- CG shift from SM0 to SM': `cm_alpha' = -cl_alpha * SM'` and `cm0' = cm0 + cl0 * (SM0 - SM')`.
  Tail incidence is unchanged.
- The parameter file is always generated from the **nominal** model. Perturbation touches only the flown plant.
- Gates are unchanged: overshoot < 0.20 and settling < 3.0 s on the **worst** case. Transition loss < 15 m is
  judged on the **nominal** flight only.
- A crashed or timed-out perturbed flight fails the attempt. The 0.75x retry reflies all three plants.
- Nominal flight folder stays `sitl/attempt_N/`, so viewer replay paths are unchanged. Perturbed flights go in
  `sitl/attempt_N/hot/` and `sitl/attempt_N/sluggish/`.
- Old saved designs must still validate: new `FCSArtifacts` fields have defaults.
- No mutation: use `dataclasses.replace` and `model_copy`. Named constants only. Functions < 50 lines,
  files < 800 lines, TDD.
- Commits end with:
  `Co-Authored-By: WOZCODE <contact@withwoz.com>`
  `Claude-Session: https://claude.ai/code/session_01RJfaxdHYf7KowUaPPCoa8G`

## Review Focus

1. **A perturbed flight crashes but the nominal passes.** The attempt must fail, and the error must name the case,
   not be silently dropped. Pinned in Task 2 (`test_a_crashed_perturbed_flight_fails_the_attempt`).
2. **Worst overshoot and worst settling come from different cases.** Both worst values must be reported, and
   `governing_case` must be the case with the highest limit ratio. Pinned in Task 2
   (`test_worst_case_takes_each_metric_from_its_worst_case`).
3. **A saved design from before this change is loaded.** `FCSArtifacts` without the new keys must validate, with
   `governing_case == "nominal"`. Pinned in Task 2.
4. **A design whose as-built SM is already at or past a limit** (for example SM0 = 0.08 for the hot case). The
   perturbation must give a zero or negative shift, not an error. Pinned in Task 1.
5. **A fixed-wing design and a quadplane.** Perturbation must leave lift rotors and propulsion untouched. Pinned in
   Task 1 (g1 and g2 fixtures).

---

### Task 1: Perturbed plants

**Files:**
- Create: `src/tools/flight_control/perturbation.py`
- Modify:
  - `src/tools/flight_control/flight_model.py`: add field `static_margin: float` to `FlightModel` after `x_cg_m`,
    set from `cad.static_margin` in `build_flight_model`.
  - `src/tools/aero_cad/stability.py`: move `SM_MIN`/`SM_MAX` here.
  - `src/tools/aero_cad/cad_packer.py`: re-import `SM_MIN`/`SM_MAX` from `stability` and remove its own definitions.
- Test: `tests/test_flight_control.py` (new section `# ---- perturbation ---`)

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class PlantCase: name: str; control_power: float; damping: float; inertia: float;
    static_margin: float`
  - `HOT: Final[PlantCase] = PlantCase("hot", 1.3, 0.7, 0.8, SM_MIN)`
  - `SLUGGISH: Final[PlantCase] = PlantCase("sluggish", 0.7, 1.3, 1.2, SM_MAX)`
  - `PERTURBED_CASES: Final[tuple[PlantCase, ...]] = (HOT, SLUGGISH)`
  - `NOMINAL_CASE: Final[str] = "nominal"`
  - `def perturbed_model(model: FlightModel, case: PlantCase) -> FlightModel`

- [ ] **Step 1: Write the failing tests**

```python
from src.tools.flight_control.perturbation import HOT, SLUGGISH, PlantCase, perturbed_model

def test_perturbed_model_scales_only_the_listed_terms(design: Any) -> None:
    _, _, model = design
    hot = perturbed_model(model, HOT)
    a, h = model.aero, hot.aero
    assert h.cl_delta_a == pytest.approx(1.3 * a.cl_delta_a) and h.cm_delta_e == pytest.approx(1.3 * a.cm_delta_e)
    assert h.cl_p == pytest.approx(0.7 * a.cl_p) and h.cm_q == pytest.approx(0.7 * a.cm_q)
    for name in ("ixx_kgm2", "iyy_kgm2", "izz_kgm2", "ixz_kgm2"):
        assert getattr(hot, name) == pytest.approx(0.8 * getattr(model, name))
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
```

- [ ] **Step 2: Run them and verify they fail**

Run: `.venv/bin/python -m pytest -q tests/test_flight_control.py -k perturbed`
Expected: FAIL with `ModuleNotFoundError: src.tools.flight_control.perturbation`.

- [ ] **Step 3: Implement**

1. Move `SM_MIN`/`SM_MAX`.
2. Add `FlightModel.static_margin`.
3. Write `perturbed_model` with `dataclasses.replace` on both `FlightModel` and `AeroCoefficients`, using the CG
   formulas in Global Constraints.
4. Give the module a docstring that states the two cases, their purpose (worst overshoot / worst settling), and the
   known gap: mixed corners are not flown.

- [ ] **Step 4: Run the tests and verify they pass**

Run: `.venv/bin/python -m pytest -q tests/test_flight_control.py tests/test_aero_cad.py -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Commit** `feat: perturbed SITL plants (hot and sluggish) from the flight model`

### Task 2: Fly and gate the worst case

**Files:**
- Modify:
  - `src/tools/flight_control/sitl_validator.py`
  - `src/schemas/state.py`: `FCSArtifacts` and a new `PlantCaseResult`.
- Test: `tests/test_flight_control.py`

**Interfaces:**
- Consumes: `perturbed_model`, `PERTURBED_CASES`, `NOMINAL_CASE` (Task 1).
- Produces:
  - `class PlantCaseResult(_Frozen)` in `state.py`, with fields:
    - `case: str`
    - `roll_overshoot: NonNegative`
    - `roll_settling_s: NonNegative`
    - `pitch_overshoot: NonNegative`
    - `pitch_settling_s: NonNegative`
    - `error: Optional[str] = None`
  - `FCSArtifacts` gains:
    - `governing_case: str = "nominal"`
    - `perturbation_cases: tuple[PlantCaseResult, ...] = ()`
  - `FCSArtifacts` overshoot and settling fields now hold the worst case.
  - In `sitl_validator.py`, `AttemptResult` gains `cases: tuple[CaseScore, ...] = ()` and
    `governing_case: str = NOMINAL_CASE`.
    - `@dataclass(frozen=True) class CaseScore: case: str; result: AttemptResult`
  - `def combine(gain_scale: float, scores: Sequence[CaseScore]) -> AttemptResult`:
    - roll, pitch and transition values come from these rules:
      - each overshoot is the max over cases;
      - each settling time is the max over cases;
      - transition loss comes from the nominal case only;
      - `passed` = every case completed, and the worst values pass the gates.
    - `governing_case` = the case with the highest `max(overshoot / MAX_OVERSHOOT, settling / MAX_SETTLING_S)` over
      roll and pitch. A crashed case governs and its `error` is carried as `"<case>: <error>"`.
    - The combined `roll`/`pitch` `StepResponse` are built field by field from those maxima. Use the governing
      case's `step_deg`/`target_deg`/`peak_deg`.
  - The private loop helper `_fly_cases(model, params, work, scale, config) -> tuple[AttemptResult, Optional[FlightLog]]`
    flies nominal into `work`, then each case into `work / case.name`, through the existing `_attempt`. It returns
    the combined result and the **nominal** log, which is used for replay.

- [ ] **Step 1: Write the failing tests**

```python
def test_worst_case_takes_each_metric_from_its_worst_case() -> None:
    ft = pytest.importorskip("src.tools.flight_control.flight_test", exc_type=ImportError)
    sv = pytest.importorskip("src.tools.flight_control.sitl_validator", exc_type=ImportError)
    nominal = sv.score(_synthetic_log(ft, 0.4, transition_dip_m=2.0), 1.0)
    slow = sv.score(_synthetic_log(ft, 0.9), 1.0)  # slower roll: longer settling, still < 3 s
    combined = sv.combine(1.0, [sv.CaseScore("nominal", nominal), sv.CaseScore("hot", nominal),
                                sv.CaseScore("sluggish", slow)])
    assert combined.roll.settling_s == pytest.approx(slow.roll.settling_s)
    assert combined.transition_alt_loss_m == pytest.approx(2.0) and combined.governing_case == "sluggish"
    assert combined.passed

def test_a_crashed_perturbed_flight_fails_the_attempt() -> None:
    ft = pytest.importorskip("src.tools.flight_control.flight_test", exc_type=ImportError)
    sv = pytest.importorskip("src.tools.flight_control.sitl_validator", exc_type=ImportError)
    good = sv.score(_synthetic_log(ft, 0.4), 1.0)
    crash = sv.AttemptResult(gain_scale=1.0, passed=False, error="SitlFlightError: stall")
    combined = sv.combine(1.0, [sv.CaseScore("nominal", good), sv.CaseScore("hot", crash),
                                sv.CaseScore("sluggish", good)])
    assert not combined.passed and combined.governing_case == "hot" and "hot: SitlFlightError" in combined.error

def test_every_attempt_flies_nominal_and_both_perturbed_plants(monkeypatch, tmp_path) -> None:
    # fake _attempt records (work dir name, model.static_margin, params id, scale); first attempt's sluggish
    # flight settles slowly (tau 2.0), the second passes. Assert:
    #   flights == 6, folders == [attempt_1, hot, sluggish, attempt_2, hot, sluggish] (by .name),
    #   the three flights of one attempt share one params dict, scales == [1, 1, 1, 0.75, 0.75, 0.75],
    #   hot/sluggish models carry SM 0.08 / 0.14, fcs.attempts == 2, fcs.governing_case is set,
    #   len(fcs.perturbation_cases) == 3, report["attempts"][0]["cases"] has 3 entries,
    #   report["replay_attempt"] == 2, and the param header names the governing case.

def test_old_fcs_records_validate_with_nominal_governing() -> None:
    payload = {"param_path": "p", "sitl_report_path": "r", "passed": True, "roll_overshoot": 0.1,
               "pitch_overshoot": 0.1, "roll_settling_s": 1.0, "pitch_settling_s": 1.0,
               "max_transition_altitude_loss_m": 0.0}
    fcs = FCSArtifacts.model_validate(payload)
    assert fcs.governing_case == "nominal" and fcs.perturbation_cases == ()
```

Update `test_retry_loop_scales_gains_by_0_75_until_it_passes` and `test_validator_raises_when_no_flight_completes`
for three flights per attempt: scales are repeated three times per attempt. Keep their original assertions on
attempts, gain scale and `replay_attempt`.

- [ ] **Step 2: Run them and verify they fail**

Run: `.venv/bin/python -m pytest -q tests/test_flight_control.py -k "worst_case or crashed_perturbed or every_attempt or old_fcs or retry_loop or no_flight"`
Expected: FAIL (`combine`/`CaseScore` missing; a single flight per attempt).

- [ ] **Step 3: Implement** `combine`, `CaseScore`, `_fly_cases`, the new `AttemptResult` fields, and the
  `_report` cases. Use `asdict` as today. Then make these changes:
  1. `_param_header` appends `, worst case: <governing_case>`.
  2. `generate_and_verify_fcs` uses `_fly_cases` and fills the two new `FCSArtifacts` fields from the best
     attempt's cases.
  3. Update the module docstring to describe the three-plant attempt.

- [ ] **Step 4: Run the tests and verify they pass**

Run: `.venv/bin/python -m pytest -q tests/test_flight_control.py tests/test_design_archive.py -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Commit** `feat: gate SITL roll and pitch on the worst of nominal, hot and sluggish plants`

### Task 3: Show it and say it

**Files:**
- Modify:
  - `src/ui/presenters.py` (`fcs_rows`)
  - `src/pipeline.py:110-115` (SITL failure warning)
  - `README.md:5`
  - `REQUIREMENTS.md` (the lines reworded to "consistent in simulation")
  - `src/agent/report.py` (heading)
  - `tests/test_agent_e2e.py` (heading assertion)
  - `tests/test_flight_control.py` (Docker tests)
- Test: `tests/test_ui.py` or the existing presenter test file that covers `fcs_rows` (grep `fcs_rows` in `tests/`).

**Interfaces:**
- Consumes: `FCSArtifacts.governing_case`, `FCSArtifacts.perturbation_cases` (Task 2).

- [ ] **Step 1: Write the failing tests**
  - `test_fcs_rows_name_the_governing_case_and_list_each_plant`:
    - an `FCSArtifacts` with three `PlantCaseResult`s gives a row `"Worst case (of nominal, hot, sluggish)"` with
      Result equal to the governing case;
    - it gives one row per case with that case's roll/pitch overshoot and settling in a single `Result` string;
    - an old `FCSArtifacts` (no cases) gives exactly today's rows.
  - `test_sitl_failure_warning_names_the_governing_case`: `design_warnings` with a failed `FCSArtifacts`
    (`governing_case="hot"`) contains `"worst case: hot"`.
  - Docker (`@pytest.mark.docker`):
    - `test_sitl_acceptance` additionally asserts `len(report["attempts"][-1]["cases"]) == 3` and that
      `sitl/attempt_N/hot/jsbsim_truth.json` exists.
    - New `test_far_out_of_range_plant_fails_the_gate[g1]`: generate params from the nominal model, fly
      `perturbed_model(model, PlantCase("extreme", 3.0, 1.0, 1.0, model.static_margin))` once through `sv._attempt`,
      and assert `not result.passed`.
- [ ] **Step 2: Run them and verify they fail** (the host test files only)

Run: `.venv/bin/python -m pytest -q tests/test_ui.py -k "fcs_rows or governing" -p no:cacheprovider`
Expected: FAIL.

- [ ] **Step 3: Implement**
  1. Rows and warning text.
  2. Report heading `## Flight control and SITL robustness check`, with the matching e2e assertion.
  3. README line 5 and the REQUIREMENTS lines: replace the interim "consistent in simulation" wording with
     "checked in SITL against two extreme plant corners: ±30 % control power and damping, ±20 % inertia, and both
     static-margin limits (0.08 and 0.14)".
  4. In the same wording, keep the caveat that mixed corners are not flown.
- [ ] **Step 4: Run the host suite and verify it passes**

Run: `.venv/bin/python -m pytest -q -p no:cacheprovider`
Expected: all pass (Docker tests skipped).

- [ ] **Step 5: Run the Docker tests and verify they pass**

Run: `docker run --rm -v "$PWD":/app -w /app isr-forge:latest python -m pytest -q -p no:cacheprovider tests/test_flight_control.py -m docker`
Expected:
- `test_sitl_acceptance[g1,g2]` pass;
- `test_far_out_of_range_plant_fails_the_gate[g1]` passes.

If g1 or g2 fails on a perturbed case, **do not loosen the gate or the factors**. Report the per-case numbers from
`sitl_report.json`. That is a real finding about the gains.

- [ ] **Step 6: Commit** `feat: show per-plant SITL results and reword the robustness claim`
