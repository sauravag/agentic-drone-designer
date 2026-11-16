# Designer in the Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Configurable design assumptions, an option explorer for non-converged designs, a design chat that
tests what-ifs with the real solver, and linked re-runs that build a design history.

**Architecture:** A frozen `DesignAssumptions` model replaces the hidden sizing/catalog constants and is
threaded through sizing, the catalog matcher, the agent state and the UI. A deterministic evaluator (sizing +
catalog + sizing-only gates) powers both the explorer (closure and improvement modes) and the chat's what-ifs.
The chat is a check-and-answer loop over `StructuredLLM.parse` (no tool calling). Accepting a change starts a
new full run that records its parent.

**Tech Stack:** Python 3.11, pydantic v2, LangGraph, Streamlit (`chat_input`/`chat_message`, AppTest), pytest.

**Spec:** `docs/superpowers/specs/2026-11-16-designer-in-the-loop-design.md`

## Global Constraints

- Type hints everywhere; files < 800 lines; functions < 50 lines; no magic numbers (named `Final` constants);
  frozen pydantic models / dataclasses; validate at boundaries; never swallow errors silently.
- Zero LLM Math: every number shown in options or chat comes from the deterministic tools.
- Works on both providers through `StructuredLLM.parse` only; LLM provider/model remain `.env`-only.
- All LLM-written text shown in Streamlit goes through `src.agent.report.defang` / `inline`.
- Defaults must reproduce today's results exactly: every existing test passes unchanged except where a task
  says otherwise.
- Tests: host `.venv/bin/python -m pytest -q -p no:cacheprovider`; full suite in Docker
  `docker run --rm -v "$PWD":/app -w /app isr-forge:latest python -m pytest -q -p no:cacheprovider`.
  Coverage ≥ 80 %.
- Commits: conventional commits ending with the session attribution lines; commit only at the end of each
  task, never push.
- Reference failing run for fixtures: `REQ_FAILED` = electric quadplane VTOL, payload 5 kg / 25 W, endurance
  6 h, radius 100 km, cruise 30 m/s, ceiling 500 m, stall 13 m/s, no runway (copy of
  `outputs/20261116T030906Z-g1-electric-fixed-wing/requirements.json`, add to `tests/reference_designs.py`).

## Review Focus

1. Assumptions set in the panel must apply to a **description** run as well as the structured form — test in
   Task 11.
2. A chat proposal or edited review panel that changes nothing (all values equal current) must be refused with
   "no change", not start an identical run — test in Task 9.
3. A closure option that is re-run and then fails an airframe gate must come back in airframe mode (chat, no
   cards), not re-offer the same closure options — test in Task 8.
4. A closing option must respect the designer's `group_constraint`: an option over the requested group limit
   is not "closing" — test in Task 6.
5. Reopening an earlier run from the history must show that run's own assumptions, not the panel's current
   values — test in Task 12.

---

### Task 1: `DesignAssumptions` model as the single source of technology constants

**Files:**
- Create: `src/schemas/assumptions.py`
- Modify: `src/schemas/components.py:14-15` (widen `MIN/MAX_SPECIFIC_ENERGY_WH_KG` to 80/500)
- Modify: `src/tools/database/seed_data.py:31-33` (derive pack constants from the model defaults)
- Test: `tests/test_assumptions.py`

**Interfaces:**
- Produces: `DesignAssumptions(BaseModel)` (frozen, `extra="forbid"`) with the 12 fields, defaults and bounds of
  spec §5.1 (`liion_specific_energy_wh_kg=200 [100,500]`, `liion_energy_density_wh_l=250 [100,1000]`,
  `liion_max_c_rate=3 [1,30]`, `lipo_specific_energy_wh_kg=155 [80,300]`, `lipo_energy_density_wh_l=280
  [100,800]`, `lipo_max_c_rate=25 [5,100]`, `battery_reserve_fraction=0.20 [0.10,0.40]`,
  `fuel_reserve_fraction=0.10 [0.05,0.40]`, `bsfc_kg_per_kwh=0.50 [0.25,1.20]`, `drivetrain_efficiency=0.68
  [0.40,0.95]`, `hover_figure_of_merit=0.72 [0.40,0.90]`, `lift_motor_efficiency=0.85 [0.50,0.98]`), each with a
  `Field(description=...)` used as UI help.
- Produces: `DEFAULT_ASSUMPTIONS: Final[DesignAssumptions]`;
  `DesignAssumptions.reserve_for(propulsion: str) -> float` (fuel reserve for `series_hybrid`, battery reserve
  otherwise); `.specific_energy_wh_kg(chem: Literal["li_ion","lipo"]) -> float`;
  `.energy_density_wh_l(chem) -> float`; `.max_c_rate(chem) -> float`;
  `.non_default() -> dict[str, tuple[float, float]]` (field → (default, value));
  `.with_reserve_floor(floor: float) -> DesignAssumptions` (both reserves raised to at least `floor`).

- [ ] **Step 1: Write the failing tests**

```python
def test_defaults_match_todays_constants():
    a = DEFAULT_ASSUMPTIONS
    assert (a.liion_specific_energy_wh_kg, a.liion_max_c_rate, a.lipo_specific_energy_wh_kg) == (200, 3, 155)
    assert (a.battery_reserve_fraction, a.fuel_reserve_fraction, a.bsfc_kg_per_kwh) == (0.20, 0.10, 0.50)
    assert (a.drivetrain_efficiency, a.hover_figure_of_merit, a.lift_motor_efficiency) == (0.68, 0.72, 0.85)
    assert a.non_default() == {}

@pytest.mark.parametrize(("field", "value"), [("liion_specific_energy_wh_kg", 99), ("liion_specific_energy_wh_kg", 501),
    ("liion_max_c_rate", 0.5), ("drivetrain_efficiency", 0.99), ("battery_reserve_fraction", 0.05)])
def test_out_of_bounds_values_are_rejected(field, value):
    with pytest.raises(ValidationError):
        DesignAssumptions(**{field: value})

def test_reserve_floor_and_helpers():
    a = DesignAssumptions(liion_specific_energy_wh_kg=300).with_reserve_floor(0.15)
    assert a.reserve_for("series_hybrid") == 0.15 and a.reserve_for("electric") == 0.20
    assert a.non_default()["liion_specific_energy_wh_kg"] == (200, 300)
    assert a.specific_energy_wh_kg("li_ion") == 300 and a.max_c_rate("lipo") == 25

def test_catalog_pack_constants_come_from_the_defaults():
    from src.tools.database import seed_data
    assert seed_data.PACK_SPECIFIC_ENERGY_WH_KG == {"li_ion": 200.0, "lipo": 155.0}
    assert seed_data.PACK_ENERGY_DENSITY_WH_L == {"li_ion": 250.0, "lipo": 280.0}
```

- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_assumptions.py -q` — expected FAIL (module missing).
- [ ] **Step 3: Implement** `src/schemas/assumptions.py`; set `seed_data.PACK_*` from
  `DEFAULT_ASSUMPTIONS.specific_energy_wh_kg/energy_density_wh_l`; widen the `Battery` bounds.
- [ ] **Step 4: Run** the new tests and the full host suite — expected PASS, no other test changes.
- [ ] **Step 5: Commit** `feat: DesignAssumptions model for technology constants`.

---

### Task 2: Thread assumptions through sizing

**Files:**
- Modify: `src/tools/sizing/mission_profile.py` (remove `LIION_*`, `LIPO_*`, `MIN_SOC_RESERVE`, `FUEL_RESERVE`,
  `BSFC_KG_PER_J`, `DRIVETRAIN_EFFICIENCY`, `FIGURE_OF_MERIT`, `VTOL_MOTOR_EFFICIENCY` as sources; keep
  `HYBRID_BUFFER_SOC_FLOOR: Final[float] = 0.20` for the hybrid buffer pack, spec §5.1)
- Modify: `src/tools/sizing/mission_solver.py` (`_Design` gains `assumptions`; `size_aircraft`, `reserve_fraction`)
- Modify: every importer of the removed constants (grep `LIION_WH_PER_KG|MIN_SOC_RESERVE|FUEL_RESERVE|DRIVETRAIN_EFFICIENCY|FIGURE_OF_MERIT|VTOL_MOTOR_EFFICIENCY|BSFC_KG_PER_J`, e.g. `src/agent/prompts.py`, flight-control modules, tests) to read `DEFAULT_ASSUMPTIONS`
- Test: `tests/test_sizing.py` (append)

**Interfaces:**
- Consumes: `DesignAssumptions`, `DEFAULT_ASSUMPTIONS` (Task 1).
- Produces: `size_aircraft(req, overrides=None, assumptions: DesignAssumptions = DEFAULT_ASSUMPTIONS) -> SizingOutput`;
  `reserve_fraction(params: dict[str, float], propulsion: Propulsion, assumptions: DesignAssumptions = DEFAULT_ASSUMPTIONS) -> float`
  (override if present, else `assumptions.reserve_for(propulsion)`);
  `forward_power_w(..., eta: float)`, `hover_power_w(..., figure_of_merit: float, motor_efficiency: float)`,
  `simulate(..., bsfc_kg_per_j: float)`, `electric_storage(trace, reserve, assumptions)`,
  `hybrid_storage(trace, gen_w, fuel_reserve, assumptions)` — explicit parameters, no module-level defaults read
  inside physics.

- [ ] **Step 1: Write the failing tests**

```python
def test_default_assumptions_reproduce_the_reference_designs():  # parametrize over the existing references
    assert size_aircraft(G1_ELECTRIC_FW).mtow_kg == size_aircraft(G1_ELECTRIC_FW, None, DEFAULT_ASSUMPTIONS).mtow_kg

def test_c_rate_drives_electric_vtol_battery_mass():  # spec § findings: 46.7 kg at 3C, ~15.9 kg at 5C
    req = REQ_FAILED.model_copy(update={"endurance_hours": 1.0, "combat_radius_km": 20.0, "cruise_speed_ms": 22.0})
    base = size_aircraft(req)
    fast = size_aircraft(req, None, DesignAssumptions(liion_max_c_rate=5))
    assert base.mtow_kg == pytest.approx(46.7, rel=0.02) and fast.mtow_kg == pytest.approx(15.9, rel=0.05)

def test_specific_energy_and_efficiency_change_the_design():
    req = REQ_FAILED.model_copy(update={"endurance_hours": 3.0, "combat_radius_km": 50.0, "cruise_speed_ms": 22.0})
    assert size_aircraft(req, None, DesignAssumptions(liion_specific_energy_wh_kg=300)).mtow_kg == pytest.approx(23.1, rel=0.03)
    with pytest.raises(SizingDivergenceError):
        size_aircraft(req)  # 200 Wh/kg spirals

def test_reserve_comes_from_assumptions_unless_overridden():
    a = DesignAssumptions(battery_reserve_fraction=0.30)
    assert reserve_fraction({}, "electric", a) == 0.30 and reserve_fraction({"reserve_fraction": 0.25}, "electric", a) == 0.25
    assert reserve_fraction({}, "series_hybrid", a) == 0.10
```

- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_sizing.py -q` — expected FAIL (signature).
- [ ] **Step 3: Implement** the threading. `_Design` carries `assumptions`; `_evaluate` passes efficiencies,
  BSFC (`bsfc_kg_per_kwh / KWH_TO_J`) and storage parameters explicitly.
- [ ] **Step 4: Run** the full host suite — expected PASS (no existing expectations change).
- [ ] **Step 5: Commit** `refactor: sizing reads technology constants from DesignAssumptions`.

---

### Task 3: Solver oscillation fix

**Files:**
- Modify: `src/tools/sizing/mission_solver.py` (`_converge`)
- Test: `tests/test_sizing.py` (append)

**Interfaces:**
- Produces: same `_converge(design) -> _Evaluation` contract; new constants `MIN_DAMPING: Final[float] = 0.05`,
  `OSCILLATION_WINDOW: Final[int] = 6`.

- [ ] **Step 1: Write the failing test**

```python
def test_oscillating_fixed_point_now_converges():
    req = REQ_FAILED.model_copy(update={"endurance_hours": 1.0, "combat_radius_km": 20.0})  # 30 m/s: oscillated
    sizing = size_aircraft(req)
    assert 10.0 < sizing.mtow_kg < 150.0

def test_genuine_spirals_still_raise():
    with pytest.raises(SizingDivergenceError, match="mass spiral"):
        size_aircraft(REQ_FAILED)
```

- [ ] **Step 2: Run** — expected the first test FAILS with "did not converge in 1000 iterations".
- [ ] **Step 3: Implement**: keep the last `OSCILLATION_WINDOW` residual signs of `mtow_out - mtow`; when they
  alternate for the whole window, halve the damping (floor `MIN_DAMPING`) and clear the window. Spiral
  detection (`DIVERGENCE_MTOW_KG`) is unchanged.
- [ ] **Step 4: Run** the full host suite — expected PASS.
- [ ] **Step 5: Commit** `fix: damp oscillating MTOW iterations in the sizing solver`.

---

### Task 4: Catalog batteries follow the assumptions

**Files:**
- Modify: `src/tools/database/matcher.py` (`match_battery`, `select_bom`)
- Modify: `src/tools/database/seed_data.py` (export `pack_envelope(capacity_wh, specific_energy_wh_kg, energy_density_wh_l) -> tuple[float, float, float, float]` = mass, L, W, H; `_battery_rows` uses it)
- Modify: `src/tools/sizing/mission_solver.py` (`component_needs` unchanged signature)
- Test: `tests/test_database.py` (append)

**Interfaces:**
- Consumes: `DesignAssumptions` (Task 1).
- Produces: `rescale_pack(battery: Battery, assumptions: DesignAssumptions) -> Battery` (new copy: mass and box
  from the assumption's specific energy / energy density for its chemistry, `max_c_rate` = assumption's max C);
  `match_battery(db_path, energy_j, peak_w, chemistry=None, assumptions=DEFAULT_ASSUMPTIONS) -> Battery`
  (filters and picks the lightest **after** rescaling); `select_bom(db_path, needs, assumptions=DEFAULT_ASSUMPTIONS)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_default_assumptions_leave_catalog_packs_unchanged(db):
    for b in load_catalog(db, "batteries"):
        assert rescale_pack(b, DEFAULT_ASSUMPTIONS) == b

def test_higher_specific_energy_makes_the_same_pack_lighter(db):
    b = next(b for b in load_catalog(db, "batteries") if b.chemistry == "li_ion")
    light = rescale_pack(b, DesignAssumptions(liion_specific_energy_wh_kg=300))
    assert light.mass_kg == pytest.approx(b.capacity_wh / 300, rel=1e-3) and light.capacity_wh == b.capacity_wh

def test_c_rate_assumption_changes_which_packs_fit(db):
    with pytest.raises(ComponentNotFoundError):
        match_battery(db, 500 * WH_TO_J, 20_000, "li_ion")
    assert match_battery(db, 500 * WH_TO_J, 20_000, "li_ion", DesignAssumptions(liion_max_c_rate=30)).chemistry == "li_ion"
```
(Adjust the 500 Wh / 20 kW figures to a case the seeded catalog actually rejects at 3C and accepts at 30C; pick
them from `seed_data.BATTERIES` when writing the test.)

- [ ] **Step 2: Run** — expected FAIL.
- [ ] **Step 3: Implement** as specified; `select_bom` passes `assumptions` to `match_battery`.
- [ ] **Step 4: Run** the full host suite — expected PASS.
- [ ] **Step 5: Commit** `feat: catalog battery packs follow the design assumptions`.

---

### Task 5: Assumptions in the agent run, critic reserve floor, persistence, CLI spec

**Files:**
- Modify: `src/schemas/state.py` (`DesignState.design_assumptions: DesignAssumptions`)
- Modify: `src/agent/graph.py` (`initial_state`, `run_agent`)
- Modify: `src/agent/nodes.py` (`parse_requirements` writes `assumptions.json`; `run_sizing`, `match_cots`,
  `design_critic`/`_adjust`, `compile_report` summary)
- Modify: `src/agent/critic.py` (`apply_adjustments`, `_current`, `_step`)
- Modify: `src/agent/report.py` (`ReportInputs.design_assumptions`; "## Design assumptions" section)
- Modify: `src/main.py` (`_load_spec`)
- Test: `tests/test_agent.py` (append)

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: `run_agent(ctx, user_prompt="", requirements=None, sizing_overrides=None, design_assumptions: Optional[DesignAssumptions] = None) -> DesignState`
  (`None` means `DEFAULT_ASSUMPTIONS`; Task 9 adds a `lineage` parameter);
  `apply_adjustments(overrides, propulsion, llm_choices, gates, cad, stage=None, assumptions=DEFAULT_ASSUMPTIONS)`
  — a reserve step never goes below `assumptions.reserve_for(propulsion)`;
  `run_summary.json` gains `"non_default_assumptions": {field: [default, value]}`;
  `main._load_spec(path) -> tuple[OperationalRequirements, DesignAssumptions]` (optional top-level
  `"assumptions"` object, removed before validating the requirements).

- [ ] **Step 1: Write the failing tests**

```python
def test_run_uses_and_saves_the_designer_assumptions(tmp_path, db):
    a = DesignAssumptions(liion_specific_energy_wh_kg=260)
    state = run_agent(_ctx(tmp_path, db, None), requirements=G1_ELECTRIC_FW, design_assumptions=a)
    out = Path(state["out_dir"])
    assert DesignAssumptions.model_validate_json((out / "assumptions.json").read_text()) == a
    assert state["sizing_results"].mtow_kg < size_aircraft(G1_ELECTRIC_FW).mtow_kg
    assert "## Design assumptions" in (out / "design_report.md").read_text()
    assert json.loads((out / "run_summary.json").read_text())["non_default_assumptions"] == {
        "liion_specific_energy_wh_kg": [200.0, 260.0]}

def test_critic_never_lowers_the_reserve_below_the_designer_floor():
    a = DesignAssumptions(battery_reserve_fraction=0.20)
    llm = [Adjustment(knob="reserve_fraction", direction="decrease", reason="x")]
    overrides, changes = critic.apply_adjustments({}, "electric", llm, [], None, None, a)
    assert overrides.get("reserve_fraction", 0.20) >= 0.20 and not changes

def test_cli_spec_accepts_an_assumptions_block(tmp_path, db):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({**G1_ELECTRIC_FW.model_dump(mode="json"), "assumptions": {"liion_max_c_rate": 6}}))
    assert main([f"--spec={spec}", f"--outputs={tmp_path / 'o'}", f"--db={db}", "--no-aero", "--quiet"]) == EXIT_CONVERGED
    (saved,) = (tmp_path / "o").glob("*/assumptions.json")
    assert json.loads(saved.read_text())["liion_max_c_rate"] == 6
```

- [ ] **Step 2: Run** — expected FAIL.
- [ ] **Step 3: Implement.** The state key is `design_assumptions` (the existing `assumptions` key holds parser
  notes — do not rename it). Nodes pass `state["design_assumptions"]` to `size_aircraft` and `select_bom`.
- [ ] **Step 4: Run** the full host suite — expected PASS (update the one existing critic test that asserts a
  reserve *decrease* if any, noting the new floor rule).
- [ ] **Step 5: Commit** `feat: design assumptions in agent runs, reports and the CLI spec`.

---

### Task 6: Change sets and the closure evaluator

**Files:**
- Create: `src/schemas/changes.py`
- Create: `src/agent/evaluator.py`
- Test: `tests/test_evaluator.py`

**Interfaces:**
- Consumes: Tasks 1–5; `critic.evaluate_sizing_gates`, `critic.RESERVE_MIN`, `component_needs`, `select_bom`.
- Produces (`src/schemas/changes.py`, all frozen, `extra="forbid"`; every field `Optional[...] = None` so the
  LLM strict schema is an explicit allow-list):
  - `RequirementChanges`: `propulsion_type`, `airframe_topology`, `runway_available`, `group_constraint`,
    `payload_mass_kg`, `payload_power_w`, `endurance_hours`, `combat_radius_km`, `cruise_speed_ms`,
    `service_ceiling_m`, `max_stall_speed_ms` (types as in `OperationalRequirements`; not `mission_name`).
  - `AssumptionChanges`: one optional field per `DesignAssumptions` field.
  - `KnobChanges`: `aspect_ratio`, `tail_volume_h`, `tail_volume_v`, `fuselage_length_factor`.
  - `ChangeSet`: `label: str (1..120 chars)`, `requirements: RequirementChanges`, `assumptions: AssumptionChanges`,
    `knobs: KnobChanges`; methods `is_empty() -> bool`, `describe(req, assumptions) -> list[tuple[str, str, str]]`
    (field label, old, new; only fields that differ).
  - `apply_changes(req, assumptions, change: ChangeSet) -> tuple[OperationalRequirements, DesignAssumptions, dict[str, float]]`
    — validates through the models (raises `pydantic.ValidationError` out of bounds) and `OVERRIDE_BOUNDS` for knobs
    (raises `ValueError`).
- Produces (`src/agent/evaluator.py`):
  - `@dataclass(frozen=True) Evaluation`: `closes: bool`, `mtow_kg: Optional[float]`, `dod_group: Optional[int]`,
    `propulsion: Optional[str]`, `battery_mass_kg: Optional[float]`, `fuel_mass_kg: Optional[float]`,
    `reason: str` (failed gate details or the sizing/catalog error); `as_facts() -> dict[str, object]`.
  - `evaluate(req, assumptions, knobs: Mapping[str, float], db_path: Path) -> Evaluation` — spec §6.2: sizing
    with `assumptions.with_reserve_floor(RESERVE_MIN)`, catalog match with the same assumptions,
    `evaluate_sizing_gates`; `closes` only if all pass. Never raises for design errors (`pipeline.DESIGN_ERRORS`).

- [ ] **Step 1: Write the failing tests**

```python
def test_the_failed_run_does_not_close_but_hybrid_does(db):
    assert not evaluate(REQ_FAILED, DEFAULT_ASSUMPTIONS, {}, db).closes
    hybrid = evaluate(REQ_FAILED.model_copy(update={"propulsion_type": "series_hybrid"}), DEFAULT_ASSUMPTIONS, {}, db)
    assert hybrid.closes and hybrid.dod_group == 2 and 12 < hybrid.mtow_kg < 25  # ≈ 16.6 kg at 10 %, recheck at 15 %

def test_an_option_over_the_requested_group_is_not_closing(db):  # Review Focus 4
    req = REQ_FAILED.model_copy(update={"propulsion_type": "series_hybrid", "group_constraint": 1})
    result = evaluate(req, DEFAULT_ASSUMPTIONS, {}, db)
    assert not result.closes and "Group 1" in result.reason

def test_change_sets_apply_validate_and_describe():
    cs = ChangeSet(label="runway", requirements=RequirementChanges(runway_available=True, airframe_topology="fixed_wing"),
                   assumptions=AssumptionChanges(liion_max_c_rate=5), knobs=KnobChanges())
    req, a, knobs = apply_changes(REQ_FAILED, DEFAULT_ASSUMPTIONS, cs)
    assert req.airframe_topology == "fixed_wing" and a.liion_max_c_rate == 5 and knobs == {}
    assert ("Max discharge (Li-ion)", "3", "5") in [(l, o, n) for l, o, n in cs.describe(REQ_FAILED, DEFAULT_ASSUMPTIONS)]
    with pytest.raises(ValidationError):
        apply_changes(REQ_FAILED, DEFAULT_ASSUMPTIONS, cs.model_copy(update={"assumptions": AssumptionChanges(liion_max_c_rate=99)}))
    assert ChangeSet(label="x", requirements=RequirementChanges(), assumptions=AssumptionChanges(), knobs=KnobChanges()).is_empty()
```
(Use whatever display labels you define for `describe`; keep them in one `Final` dict next to the models.)

- [ ] **Step 2: Run** — expected FAIL.
- [ ] **Step 3: Implement** both modules.
- [ ] **Step 4: Run** — expected PASS.
- [ ] **Step 5: Commit** `feat: change sets and the sizing-plus-catalog evaluator`.

---

### Task 7: Option explorer (closure and improvement modes)

**Files:**
- Create: `src/agent/explorer.py`
- Test: `tests/test_explorer.py`

**Interfaces:**
- Consumes: `ChangeSet`, `apply_changes`, `evaluate`, `Evaluation` (Task 6); `DesignState`.
- Produces:
  - `ExplorerMode = Literal["closure", "airframe", "improvement", "none"]`;
    `explorer_mode(state: DesignState) -> ExplorerMode`: `closure` when not converged and the final failure is a
    sizing/cots stage error or a failed `reserve` / group gate without CAD collisions; `airframe` when not
    converged on `static_margin`, `lift_to_drag`, CAD collisions or an `aero_cad` stage error; `improvement` when
    converged; `none` for clarification/fatal states.
  - `@dataclass(frozen=True) ExploreOption`: `lever: str`, `change: ChangeSet`, `result: Evaluation`,
    `relative_change: float`, `closes: bool`, `mtow_saved_kg: Optional[float]` (improvement mode),
    `group_drop: bool`; `to_json() -> dict`.
  - `explore_closure(req, assumptions, db_path, time_budget_s: float = CLOSURE_BUDGET_S) -> list[ExploreOption]`
  - `explore_improvement(req, assumptions, baseline: SizingOutput, db_path, time_budget_s: float = IMPROVEMENT_BUDGET_S) -> list[ExploreOption]`
  - Constants: `CLOSURE_BUDGET_S: Final[float] = 20.0`, `IMPROVEMENT_BUDGET_S: Final[float] = 10.0`, and a
    `LEVERS` table with per-lever resolution (0.1 h, 1 km, 0.1 kg, 1 m/s, 10 Wh/kg, 0.5 C) and improvement step
    (−1 h, −10 km, −0.5 kg, −2 m/s, +50 Wh/kg, +2 C).
- Algorithm (spec §6.2–6.3): discrete levers are one evaluation (propulsion flips to the other powertrain; runway
  lever sets `runway_available=True, airframe_topology="fixed_wing"`; group lever clears `group_constraint`);
  monotonic continuous levers bisect between the requested value and the lever bound; cruise speed scans
  `[1.2 × max_stall_speed_ms, 1.2 × requested]` in 8 steps then bisects at the closing transition nearest the
  request. Levers whose current value already equals the target (e.g. propulsion already hybrid) are skipped.
  Ranking: closing first by `relative_change` ascending, then non-closing. When the time budget runs out,
  remaining levers are returned as non-closing with `reason="not evaluated (time budget)"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_closure_options_for_the_failed_run(db):
    options = explore_closure(REQ_FAILED, DEFAULT_ASSUMPTIONS, db)
    by_lever = {o.lever: o for o in options}
    assert by_lever["propulsion"].closes and by_lever["endurance"].closes
    assert by_lever["endurance"].change.requirements.endurance_hours < 6.0
    battery = by_lever["liion_specific_energy"]
    assert (not battery.closes) or battery.change.assumptions.liion_specific_energy_wh_kg > 350  # top of range only
    closing = [o for o in options if o.closes]
    assert closing == sorted(closing, key=lambda o: o.relative_change)
    for o in closing:  # every option re-validates and re-evaluates as closing
        req, a, knobs = apply_changes(REQ_FAILED, DEFAULT_ASSUMPTIONS, o.change)
        assert evaluate(req, a, knobs, db).closes

def test_time_budget_marks_unevaluated_levers(db):
    options = explore_closure(REQ_FAILED, DEFAULT_ASSUMPTIONS, db, time_budget_s=0.0)
    assert options and all(not o.closes and "time budget" in o.result.reason for o in options)

def test_improvement_mode_reports_savings(db):
    baseline = size_aircraft(G1_ELECTRIC_FW)
    options = explore_improvement(G1_ELECTRIC_FW, DEFAULT_ASSUMPTIONS, baseline, db)
    assert any(o.mtow_saved_kg and o.mtow_saved_kg > 0 for o in options)

@pytest.mark.parametrize(("update", "mode"), [
    ({"converged": True}, "improvement"),
    ({"stage_error": {"stage": "sizing", "message": "spiral"}}, "closure"),
    ({"gate_results": [{"name": "static_margin", "passed": False, "detail": ""}]}, "airframe"),
    ({"stage_error": {"stage": "aero_cad", "message": "pack"}}, "airframe"),
    ({"fatal_error": "x"}, "none")])
def test_explorer_mode(update, mode):
    assert explorer_mode(initial_state("", G1_ELECTRIC_FW) | update) == mode
```

- [ ] **Step 2: Run** — expected FAIL.
- [ ] **Step 3: Implement** `src/agent/explorer.py` (keep helpers < 50 lines; one function per search kind).
- [ ] **Step 4: Run** — expected PASS; check the closure run on `REQ_FAILED` takes < 20 s on the host.
- [ ] **Step 5: Commit** `feat: option explorer with closure and improvement modes`.

---

### Task 8: Options in the agent run, report and CLI

**Files:**
- Modify: `src/schemas/state.py` (`DesignState.explorer_mode: str`, `explore_options: list[dict]`,
  `options_explanation: Optional[str]`)
- Modify: `src/agent/nodes.py` (`compile_report`: compute mode; in `closure` mode run `explore_closure`, write
  `options.json`; with an LLM, one call for the explanation)
- Modify: `src/agent/agent_schemas.py` (`OptionsExplanation(BaseModel)`: `why_it_failed: str`, `trade_offs: str`)
- Modify: `src/agent/prompts.py` (`OPTIONS_SYSTEM`)
- Modify: `src/agent/report.py` ("## Options that close the design" table + explanation; number audit covers it)
- Modify: `src/main.py` (print the option list after a non-converged run)
- Test: `tests/test_agent.py` (append)

**Interfaces:**
- Consumes: Task 7.
- Produces: state keys above; `options.json` = `[ExploreOption.to_json(), ...]`; the explanation is defanged
  and audited with `report.unsupported_numbers`-style checking against the options' facts (generalise to
  `report.stray_numbers(text: str, facts_text: str) -> list[str]` and use it in both places).

- [ ] **Step 1: Write the failing tests**

```python
def test_a_non_converged_sizing_failure_offers_options(tmp_path, db):
    state = run_agent(_ctx(tmp_path, db, ScriptedLLM()), requirements=REQ_FAILED)
    assert state["explorer_mode"] == "closure" and any(o["closes"] for o in state["explore_options"])
    out = Path(state["out_dir"])
    assert (out / "options.json").is_file() and "## Options that close the design" in (out / "design_report.md").read_text()

def test_a_closure_option_that_fails_an_airframe_gate_comes_back_in_airframe_mode(tmp_path, db, monkeypatch):  # Review Focus 3
    monkeypatch.setattr("src.agent.nodes._gates", lambda state: [critic.GateResult("static_margin", False, "SM 3 %")])
    state = run_agent(_ctx(tmp_path, db, None, max_iterations=0),
                      requirements=REQ_FAILED.model_copy(update={"propulsion_type": "series_hybrid"}))
    assert state["explorer_mode"] == "airframe" and state["explore_options"] == []

def test_cli_prints_options_after_a_failed_run(tmp_path, db, capsys):
    spec = tmp_path / "spec.json"; spec.write_text(REQ_FAILED.model_dump_json())
    assert main([f"--spec={spec}", f"--outputs={tmp_path / 'o'}", f"--db={db}", "--no-aero", "--quiet"]) == EXIT_NOT_CONVERGED
    assert "Options that close the design" in capsys.readouterr().out
```
(`ScriptedLLM` gains an `OptionsExplanation` answer in `tests/agent_fakes.py`.)

- [ ] **Step 2: Run** — expected FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full host suite — expected PASS.
- [ ] **Step 5: Commit** `feat: closure options in the agent run, report and CLI`.

---

### Task 9: Linked re-runs and the agent service API

**Files:**
- Modify: `src/agent/graph.py` (`run_agent(..., lineage: Optional[RunLineage] = None)`)
- Modify: `src/schemas/state.py` (`RunLineage` frozen model: `parent_run_id: str`, `change_set: ChangeSet`,
  `change_source: Literal["option", "chat", "manual"]`; `DesignState.lineage`)
- Modify: `src/agent/nodes.py` (`run_summary.json` and report header include the lineage)
- Modify: `src/ui/agent_service.py`
- Test: `tests/test_agent_service.py` (append)

**Interfaces:**
- Consumes: Tasks 5–8.
- Produces (`agent_service`):
  - `run(..., design_assumptions: Optional[DesignAssumptions] = None)` (applies to prompt and form runs).
  - `AgentRunView` gains `design_assumptions`, `explorer_mode`, `options: tuple[dict, ...]`,
    `options_explanation: Optional[str]`, `lineage: Optional[RunLineage]`.
  - `NoChangeError(ValueError)`.
  - `rerun(view: AgentRunView, change: ChangeSet, source: Literal["option","chat","manual"], outputs_root, db_path, with_aero_cad, with_sitl, progress) -> AgentRunView`
    — raises `NoChangeError` when `change.is_empty()` or `describe()` returns no differing field; otherwise
    `apply_changes` on the view's requirements/assumptions and runs the structured path with knobs as
    `sizing_overrides` and the lineage.
  - `explore(view, db_path) -> tuple[ExploreOption, ...]` — on-demand improvement mode for converged views.

- [ ] **Step 1: Write the failing tests**

```python
def test_rerun_links_to_the_parent_and_applies_the_change(tmp_path, no_key):
    first = _run(tmp_path, requirements=REQ_FAILED)
    option = next(o for o in first.options if o["closes"])
    change = ChangeSet.model_validate(option["change"])
    second = agent_service.rerun(first, change, "option", tmp_path / "o", tmp_path / "c.db", False, False, lambda _: None)
    assert second.lineage.parent_run_id == first.run_id and second.status == agent_service.STATUS_CONVERGED
    assert json.loads((second.out_dir / "run_summary.json").read_text())["lineage"]["change_source"] == "option"

def test_a_change_that_changes_nothing_is_refused(tmp_path, no_key):  # Review Focus 2
    first = _run(tmp_path, requirements=G1_ELECTRIC_FW)
    same = ChangeSet(label="same", requirements=RequirementChanges(endurance_hours=G1_ELECTRIC_FW.endurance_hours),
                     assumptions=AssumptionChanges(), knobs=KnobChanges())
    with pytest.raises(agent_service.NoChangeError):
        agent_service.rerun(first, same, "chat", tmp_path / "o", tmp_path / "c.db", False, False, lambda _: None)

def test_description_runs_use_the_panel_assumptions(tmp_path, monkeypatch):  # Review Focus 1 (service level)
    monkeypatch.setattr(agent_service, "make_llm", lambda: ScriptedLLM())
    a = DesignAssumptions(liion_max_c_rate=6)
    view = _run(tmp_path, prompt=SRS_PROMPT, design_assumptions=a)
    assert view.design_assumptions == a
```

- [ ] **Step 2: Run** — expected FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full host suite — expected PASS.
- [ ] **Step 5: Commit** `feat: linked re-runs with lineage and the explorer service API`.

---

### Task 10: Design chat engine (check-and-answer loop)

**Files:**
- Create: `src/agent/design_chat.py`
- Modify: `src/agent/agent_schemas.py` (`DesignerTurn`)
- Modify: `src/agent/prompts.py` (`DESIGNER_SYSTEM`)
- Modify: `src/ui/agent_service.py` (`chat(...)`)
- Modify: `tests/agent_fakes.py` (`ScriptedLLM` answers `DesignerTurn` from a scripted list)
- Test: `tests/test_design_chat.py`; `tests/test_providers.py` (append schema round-trip)

**Interfaces:**
- Consumes: `ChangeSet`, `apply_changes`, `evaluate` (Task 6); `ExploreOption` (Task 7); `AgentRunView` (Task 9).
- Produces:
  - `DesignerTurn(BaseModel)`: `message: str`, `what_ifs: list[ChangeSet]` (max `MAX_WHAT_IFS`),
    `proposal: Optional[ChangeSet]`, `done: bool`.
  - `@dataclass(frozen=True) WhatIfResult`: `change: ChangeSet`, `result: Optional[Evaluation]`, `error: Optional[str]`.
  - `@dataclass(frozen=True) ChatMessage`: `role: Literal["designer","agent"]`, `text: str`,
    `what_ifs: tuple[WhatIfResult, ...] = ()`, `proposal: Optional[ChangeSet] = None`,
    `stray_numbers: tuple[str, ...] = ()`, `rounds: int = 0`.
  - `@dataclass(frozen=True) ChatContext`: `requirements`, `assumptions`, `failed_gates: tuple[str, ...]`,
    `options: tuple[dict, ...]`, `allow_knobs: bool` (true only in airframe mode), `db_path: Path`.
  - `answer(llm: StructuredLLM, ctx: ChatContext, history: tuple[ChatMessage, ...], message: str) -> ChatMessage`
    — up to `MAX_ROUNDS: Final[int] = 3` rounds; each round sends `DESIGNER_SYSTEM` + a JSON briefing (spec §7.2:
    requirements, assumptions, failed gates, options, last `HISTORY_BUDGET: Final[int] = 12` messages, what-if
    results so far, allowed fields with bounds); evaluates `what_ifs` (knob changes when `allow_knobs` is false and
    out-of-bounds values become `WhatIfResult(error=...)`); stops when `done` or no what-ifs; the returned message's
    `stray_numbers` lists numbers not present in the briefing + results (`report.stray_numbers`); a proposal that
    fails validation is dropped and its error appended to the text. `LLMError` propagates.
  - `agent_service.chat(view: AgentRunView, history, message, db_path) -> ChatMessage` — builds the context from
    the view; raises `ValueError` without an LLM.

- [ ] **Step 1: Write the failing tests** (module helpers: `_ctx(db, allow_knobs=False) -> ChatContext` on
  `REQ_FAILED` with default assumptions and no options; `_endurance(h)`, `_c_rate(c)`, `_knob(**kw)` build a
  `ChangeSet` changing only that field)

```python
def test_the_agent_tests_a_what_if_before_answering(db):
    turns = [DesignerTurn(message="checking", what_ifs=[_endurance(4.0)], proposal=None, done=False),
             DesignerTurn(message="4 h closes", what_ifs=[], proposal=_endurance(4.0), done=True)]
    llm = ScriptedLLM(designer_turns=turns)
    reply = answer(llm, _ctx(db), (), "Can we keep electric with less endurance?")
    assert reply.rounds == 2 and reply.proposal == _endurance(4.0)
    assert len(reply.what_ifs) == 1 and reply.what_ifs[0].result is not None
    assert "what_if_results" in llm.calls[-1][1]  # the second round saw the solver's numbers

def test_round_limit_and_invalid_changes(db):
    loop = [DesignerTurn(message="more", what_ifs=[_c_rate(99)], proposal=None, done=False)] * 5
    reply = answer(ScriptedLLM(designer_turns=loop), _ctx(db), (), "try 99C")
    assert reply.rounds == MAX_ROUNDS and reply.what_ifs[0].error

def test_knob_changes_are_refused_outside_airframe_mode(db):
    turn = DesignerTurn(message="x", what_ifs=[_knob(aspect_ratio=14)], proposal=None, done=True)
    reply = answer(ScriptedLLM(designer_turns=[turn]), _ctx(db, allow_knobs=False), (), "raise AR")
    assert "knob" in (reply.what_ifs[0].error or "")

def test_numbers_the_solver_never_produced_are_flagged(db):
    turn = DesignerTurn(message="It will weigh 12.34 kg", what_ifs=[], proposal=None, done=True)
    assert "12.34" in answer(ScriptedLLM(designer_turns=[turn]), _ctx(db), (), "weight?").stray_numbers

def test_designer_turn_round_trips_through_the_openai_sdk():  # tests/test_providers.py
    turn = DesignerTurn(message="m", what_ifs=[], proposal=None, done=True)
    assert _llm(_reply(_completion(turn.model_dump_json()))).parse("s", "p", DesignerTurn) == turn
```

- [ ] **Step 2: Run** — expected FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full host suite — expected PASS.
- [ ] **Step 5: Commit** `feat: design chat engine with solver-checked what-ifs`.

---

### Task 11: UI — design assumptions panel

**Files:**
- Create: `src/ui/assumptions_panel.py`
- Modify: `src/ui/app.py` (render the panel above the forms; pass `design_assumptions` to `agent_service.run`)
- Test: `tests/test_ui.py` (append)

**Interfaces:**
- Consumes: `DesignAssumptions` (Task 1), `agent_service.run(..., design_assumptions=...)` (Task 9).
- Produces: `render_assumptions_panel() -> None` (expander "Design assumptions", four groups per spec §5.3, keys
  `assume_<field>`, help from field descriptions, "Reset to defaults" button key `reset_assumptions`, caption when
  non-default); `current_assumptions() -> tuple[Optional[DesignAssumptions], list[str]]` (model or validation
  messages); session defaults from `DEFAULT_ASSUMPTIONS`.

- [ ] **Step 1: Write the failing tests**

```python
def test_assumptions_panel_shows_defaults_and_applies_to_description_runs(tmp_path, monkeypatch):  # Review Focus 1
    at = _llm_app(tmp_path, monkeypatch)  # existing helper with a ScriptedLLM
    assert at.number_input(key="assume_liion_specific_energy_wh_kg").value == 200
    at.number_input(key="assume_liion_max_c_rate").set_value(6.0)
    at.text_area(key="prompt").set_value(SRS_PROMPT); at.button(key="submit_prompt").click(); at.run()
    (saved,) = (tmp_path / "outputs").glob("*/assumptions.json")
    assert json.loads(saved.read_text())["liion_max_c_rate"] == 6.0

def test_reset_restores_defaults(tmp_path, monkeypatch):
    at = _llm_app(tmp_path, monkeypatch)
    at.number_input(key="assume_drivetrain_efficiency").set_value(0.9); at.run()
    at.button(key="reset_assumptions").click(); at.run()
    assert at.number_input(key="assume_drivetrain_efficiency").value == pytest.approx(0.68)
```

- [ ] **Step 2: Run** — expected FAIL.
- [ ] **Step 3: Implement**; number inputs use the model bounds as `min_value`/`max_value`.
- [ ] **Step 4: Run** the UI tests and the full host suite — expected PASS.
- [ ] **Step 5: Commit** `feat: design assumptions panel in the web form`.

---

### Task 12: UI — option cards, review panel, linked re-run and design history

**Files:**
- Create: `src/ui/design_loop.py`
- Modify: `src/ui/app.py` (keep < 400 lines; move rendering helpers out if needed)
- Test: `tests/test_ui.py` (append)

**Interfaces:**
- Consumes: `AgentRunView` fields and `rerun`, `explore`, `NoChangeError` (Task 9); `ChangeSet.describe` (Task 6).
- Produces: `render_options(view) -> None` (cards with change text, computed outcome, "checked with sizing +
  catalog only", Select button key `select_option_<i>`; explanation above, defanged; improvement-mode button
  `explore_alternatives` on converged views); `render_review_panel(view) -> None` (old → new rows, editable new
  values, Re-run button `rerun`, shows `NoChangeError` as an error); `render_history() -> None` (session list of
  views, keys `open_run_<run_id>`). Session keys: `runs` (list of views, newest last), `last_run`,
  `pending_change` (`tuple[ChangeSet, source]`).
- Produces (tests): helper `_form_app_with(tmp_path, monkeypatch, req: OperationalRequirements, llm: Optional[StructuredLLM] = None) -> AppTest`
  in `tests/test_ui.py` — fills the structured form from `req` via `presenters.form_defaults`, unchecks aero/CAD,
  patches `agent_service.llm_problem`/`make_llm` to the given LLM (or no LLM), submits, returns the app. Task 13
  reuses it.

- [ ] **Step 1: Write the failing tests**

```python
def test_failed_design_offers_cards_and_a_linked_rerun(tmp_path, monkeypatch):
    at = _form_app_with(tmp_path, monkeypatch, REQ_FAILED)  # helper: fill the structured form, aero off, submit
    assert any(b.key.startswith("select_option_") for b in at.button)
    at.button(key="select_option_0").click(); at.run()
    at.button(key="rerun").click(); at.run()
    summaries = sorted((tmp_path / "outputs").glob("*/run_summary.json"))
    assert json.loads(summaries[-1].read_text())["lineage"]["change_source"] == "option"
    assert len([b for b in at.button if b.key.startswith("open_run_")]) == 2

def test_reopening_a_run_shows_its_own_assumptions(tmp_path, monkeypatch):  # Review Focus 5
    at = _form_app_with(tmp_path, monkeypatch, G1_ELECTRIC_FW)
    first = at.session_state["last_run"].run_id
    at.number_input(key="assume_liion_specific_energy_wh_kg").set_value(300.0)
    at.button(key="submit").click(); at.run()
    at.button(key=f"open_run_{first}").click(); at.run()
    assert at.session_state["last_run"].design_assumptions.liion_specific_energy_wh_kg == 200

def test_converged_design_can_explore_improvements(tmp_path, monkeypatch):
    at = _form_app_with(tmp_path, monkeypatch, G1_ELECTRIC_FW)
    at.button(key="explore_alternatives").click(); at.run()
    assert any("kg" in m.value for m in at.markdown if "−" in m.value or "-" in m.value)
```

- [ ] **Step 2: Run** — expected FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — expected PASS.
- [ ] **Step 5: Commit** `feat: option cards, review panel, linked re-runs and design history`.

---

### Task 13: UI — design chat panel

**Files:**
- Create: `src/ui/chat_panel.py`
- Modify: `src/ui/app.py`
- Test: `tests/test_ui.py` (append)

**Interfaces:**
- Consumes: `agent_service.chat`, `ChatMessage` (Task 10); `pending_change` session key (Task 12).
- Produces: `render_chat(view) -> None` — `st.chat_message` history from session key `chat_<run_id>`,
  `st.chat_input` key `design_chat`; opens expanded when `explorer_mode` is `closure` or `airframe`, collapsed
  otherwise; agent text defanged; what-if results as a small table under the reply; stray numbers as a caption
  warning; a proposal shows a "Review this change" button (`use_proposal_<n>`) that fills `pending_change` with
  source `chat`; without an LLM a caption explains how to enable it; `LLMError` shows a short error and keeps the
  history. Saves `design_chat.json` in the run folder after each reply.

- [ ] **Step 1: Write the failing tests**

```python
def test_chat_tests_a_what_if_and_hands_a_proposal_to_the_review_panel(tmp_path, monkeypatch):
    turns = [DesignerTurn(message="3 h closes", what_ifs=[_endurance(3.0)], proposal=_endurance(3.0), done=True)]
    at = _form_app_with(tmp_path, monkeypatch, REQ_FAILED, llm=ScriptedLLM(designer_turns=turns))
    at.chat_input(key="design_chat").set_value("Keep electric?").run()
    assert any("3 h closes" in m.markdown[0].value for m in at.chat_message if m.name == "assistant")
    at.button(key="use_proposal_0").click(); at.run()
    assert at.session_state["pending_change"][1] == "chat"
    assert any((tmp_path / "outputs").glob("*/design_chat.json"))

def test_chat_is_disabled_without_a_key(tmp_path, monkeypatch):
    at = _form_app_with(tmp_path, monkeypatch, REQ_FAILED, llm=None)
    assert not [c for c in at.chat_input if c.key == "design_chat"] or at.chat_input(key="design_chat").disabled
```

- [ ] **Step 2: Run** — expected FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — expected PASS.
- [ ] **Step 5: Commit** `feat: design chat panel in the web form`.

---

### Task 14: Documentation and full verification

**Files:**
- Modify: `README.md` (design assumptions, option explorer, chat, linked re-runs, `--spec` assumptions block,
  new run files `assumptions.json`, `options.json`, `design_chat.json`)
- Modify: `REQUIREMENTS.md` untouched (SRS is the user's document)

- [ ] **Step 1:** Update the README sections listed above.
- [ ] **Step 2: Run** the full host suite with coverage — expected PASS, coverage ≥ 80 %.
- [ ] **Step 3: Run** the full Docker suite — expected PASS (rebuild the image only if dependencies changed).
- [ ] **Step 4:** Mandatory code review by a python-reviewer agent over the whole branch diff; fix CRITICAL/HIGH.
- [ ] **Step 5: Commit** `docs: designer-in-the-loop usage`.
