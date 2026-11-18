# Avionics Stack Choice + Optional Jetson Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the designer pick the avionics stack (Cube Orange+ with an optional Jetson companion, or the ARK Jetson PAB stack with the Jetson built in), size the aircraft with it, and show every avionics assumption.

**Architecture:** Two new requirement fields (`flight_controller`, `companion_computer`) and eight new `DesignAssumptions` fields; `DesignAssumptions` owns the avionics mass/power formulas; sizing records what it used in `SizingOutput`, and the parts list, CAD tray, UI and report read from there.

**Tech Stack:** Python 3.11, pydantic v2 (frozen models), Streamlit 1.67 (`AppTest`), pytest.

**Spec:** `docs/superpowers/specs/2026-11-18-companion-computer-design.md`

## Global Constraints

- Defaults reproduce today's results exactly for Cube Orange+ without Jetson: `avionics_base_mass_kg` 0.227 + `cube_orange_plus_mass_kg` 0.073 = 0.3; `avionics_base_power_w` 10.0; `avionics_power_per_kg_w` 0.4; `avionics_mass_per_mtow` 0.01.
- Other defaults: `ark_pab_stack_mass_kg` 0.166, `companion_computer_mass_kg` 0.2, `jetson_power_w` 25.0.
- Bounds: base mass 0.0–5.0; per-MTOW 0.0–0.05; base power 1.0–200.0; per-kg power 0.0–5.0; Cube 0.03–0.5; ARK 0.08–1.0; companion 0.02–3.0; Jetson power 1.0–100.0.
- `flight_controller: Literal["cube_orange_plus", "ark_jetson_pab"] = "cube_orange_plus"`; `companion_computer: bool = False`; with `ark_jetson_pab`, `companion_computer` is normalised to False before validation.
- Form and parser defaults for a new design: Cube Orange+ with `companion_computer=True`.
- Example presets never set `flight_controller` / `companion_computer`.
- Closure / improvement explorers never change `flight_controller` / `companion_computer`.
- `src/schemas/assumptions.py` must not import from `src.tools`.
- Functions < 50 lines, files < 800 lines, no magic numbers, SI units, type hints, frozen models (no mutation).
- Commit messages end with:
  ```
  Co-Authored-By: WOZCODE <contact@withwoz.com>
  Claude-Session: https://claude.ai/code/session_01RJfaxdHYf7KowUaPPCoa8G
  ```

## Review Focus

1. ARK chosen with the Jetson box ticked → the Jetson must be counted once (0.166 kg stack, +25 W), never plus 0.2 kg. (Task 1 pins at schema; Task 3 pins in BOM.)
2. Loading a preset after unticking the Jetson → the untick must survive. (Task 6.)
3. Re-run through the review panel that switches the stack → `apply_changes` revalidates, so ARK normalisation must still apply. (Task 1.)
4. Old `requirements.json` without the new keys → Cube, no Jetson. (Task 1.)
5. CAD tray and sizing disagreeing on avionics mass after an assumption edit → tray must follow `sizing.avionics_mass_kg`. (Task 4.)

**UI decision (user, 2026-11-18):** live greying-out. The two avionics widgets sit in an "Avionics" section **outside and directly above** `st.form("requirements")` (below the description form), so changing the select reruns the script and the checkbox is `disabled` (and shown ticked, caption "built into the ARK stack") when ARK is selected. Form runs read both keys from `st.session_state` as today. Description runs keep using the parser's avionics fields.

---

### Task 0: Baseline guard for the reference designs

**Files:**
- Create: `tests/data/avionics_baseline.json`, `tests/test_avionics.py`

**Interfaces:** none (pure regression guard, written while the code is unchanged).

- [ ] **Step 1:** Generate the baseline from current code: run `python -I -c` (with `PYTHONPATH=.` from repo root) calling `size_aircraft(req)` for `G1_ELECTRIC_FW`, `G2_ELECTRIC_VTOL`, `G3_HYBRID_VTOL` (`src/ui/examples.py`) and dumping `{name: {mtow_kg, empty_mass_kg, battery_mass_kg, fuel_mass_kg, cruise_power_w, hover_power_w}}` to `tests/data/avionics_baseline.json` (full float precision).
- [ ] **Step 2:** Write `test_reference_designs_unchanged_without_jetson`, parametrised over the three presets, asserting each of the six fields equals the baseline with `pytest.approx(rel=1e-9)`.
- [ ] **Step 3:** Run `pytest tests/test_avionics.py -q` → 3 passed.
- [ ] **Step 4:** Commit `test: pin reference-design sizing before the avionics change`.

### Task 1: Schemas — requirements, assumptions, change sets

**Files:**
- Modify: `src/schemas/requirements.py`, `src/schemas/assumptions.py`, `src/schemas/changes.py`
- Test: `tests/test_avionics.py`

**Interfaces — Produces:**
- `FlightControllerId = Literal["cube_orange_plus", "ark_jetson_pab"]` and `ARK_JETSON_PAB: Final = "ark_jetson_pab"` in `src/schemas/requirements.py`.
- `OperationalRequirements.flight_controller`, `.companion_computer`, property `.jetson_fitted -> bool`.
- `DesignAssumptions.avionics_mass_kg(self, mtow_kg: float, flight_controller: str, companion_computer: bool) -> float`
- `DesignAssumptions.avionics_power_w(self, mtow_kg: float, jetson_fitted: bool) -> float`
- `RequirementChanges.flight_controller: Optional[FlightControllerId]`, `.companion_computer: Optional[bool]`; `AssumptionChanges` gains the eight fields (`Optional[float]`); `FIELD_LABELS` gains `"flight_controller": "Flight controller"`, `"companion_computer": "Jetson companion computer"` and eight assumption labels with units (e.g. `"jetson_power_w": "Jetson power (W)"`).

- [ ] **Step 1: Failing tests:**
  - `test_old_requirements_json_loads_as_cube_without_jetson`: `OperationalRequirements.model_validate(json.loads(Path("tests/data/flight_control/g1/requirements.json").read_text()))` → `flight_controller == "cube_orange_plus"`, `companion_computer is False`, `jetson_fitted is False`.
  - `test_ark_normalises_companion_and_fits_jetson`: `OperationalRequirements(**{**G1.model_dump(), "flight_controller": "ark_jetson_pab", "companion_computer": True})` → `companion_computer is False`, `jetson_fitted is True`.
  - `test_apply_changes_switching_to_ark_normalises`: `apply_changes(req_with_companion, DEFAULT_ASSUMPTIONS, ChangeSet(label="ark", requirements=RequirementChanges(flight_controller="ark_jetson_pab"), assumptions=AssumptionChanges(), knobs=KnobChanges()))[0].companion_computer is False`.
  - `test_avionics_mass_formula`: at `mtow=10.0`: Cube/no Jetson → `0.3 + 0.1 = 0.4`; Cube/Jetson → `0.6`; ARK → `0.227 + 0.1 + 0.166 = 0.493` (`pytest.approx`).
  - `test_avionics_power_formula`: at `mtow=10.0`: no Jetson → `14.0`; Jetson → `39.0`.
  - `test_avionics_assumption_bounds`: parametrised `(field, below, above)` over the eight fields; each out-of-bounds value raises `ValidationError`.
- [ ] **Step 2:** `pytest tests/test_avionics.py -q` → new tests FAIL (AttributeError / ValidationError).
- [ ] **Step 3:** Implement: fields on `OperationalRequirements`; a `@model_validator(mode="before")` `_ark_carries_the_jetson` that returns a new dict with `companion_computer=False` when `flight_controller == ARK_JETSON_PAB` (no mutation of the input); eight `Field`s with descriptions (used as help text) and the two methods on `DesignAssumptions`; change-set fields and labels.
- [ ] **Step 4:** `pytest tests/test_avionics.py tests/test_schemas.py tests/test_assumptions.py -q` → PASS.
- [ ] **Step 5:** Commit `feat: avionics stack and Jetson fields in requirements and assumptions`.

### Task 2: Sizing uses the assumptions and records avionics mass/power

**Files:**
- Modify: `src/tools/sizing/mission_profile.py:20-21,80-81`, `src/tools/sizing/mass_model.py:30-31,121-143`, `src/tools/sizing/mission_solver.py:154,168,281-315`, `src/schemas/state.py` (`SizingOutput`)
- Test: `tests/test_avionics.py`, existing `tests/test_aero_cad.py:322-324` call sites

**Interfaces:**
- Consumes: Task 1 methods.
- Produces:
  - `hotel_power_w(req: OperationalRequirements, mtow_kg: float, assumptions: DesignAssumptions) -> float`
  - `empty_mass_breakdown(mtow_kg, geom, airframe, loads, avionics_mass_kg: float) -> dict[str, float]`
  - `airframe_mass_kg(mtow_kg, empty_mass_kg, loads, avionics_mass_kg: float) -> float`
  - `SizingOutput.avionics_mass_kg: Positive`, `SizingOutput.avionics_power_w: Positive`

- [ ] **Step 1: Failing tests:**
  - `test_hotel_power_adds_jetson_watts`: at fixed `mtow=10.0`, `hotel_power_w(cube_jetson_req, 10.0, DEFAULT_ASSUMPTIONS) - hotel_power_w(cube_req, 10.0, DEFAULT_ASSUMPTIONS) == approx(25.0)`.
  - `test_breakdown_avionics_entry_is_the_argument`: `empty_mass_breakdown(..., avionics_mass_kg=0.5)["avionics"] == 0.5`.
  - `test_sizing_records_avionics_and_grows_with_jetson`: on G1, Cube+Jetson `avionics_power_w - payload` larger by 25; `mtow(Cube+Jetson) > mtow(ARK) > mtow(Cube, no Jetson)`; `sizing.avionics_mass_kg == approx(DEFAULT_ASSUMPTIONS.avionics_mass_kg(sizing.mtow_kg, ...))`.
  - Task 0 test must still pass unchanged.
- [ ] **Step 2:** `pytest tests/test_avionics.py -q` → FAIL (TypeError on new signatures).
- [ ] **Step 3:** Implement: delete `AVIONICS_POWER_BASE_W`, `AVIONICS_POWER_PER_KG`, `AVIONICS_BASE_KG`, `AVIONICS_PER_MTOW`; in `_evaluate` compute `avionics = assumptions.avionics_mass_kg(mtow, req.flight_controller, req.companion_computer)` and pass it on; carry `avionics_mass_kg` / `avionics_power_w` on `_Evaluation` and set them in `_to_output`. Update `tests/test_aero_cad.py:322-324` calls to pass `avionics_mass_kg=DEFAULT_ASSUMPTIONS.avionics_mass_kg(mtow, "cube_orange_plus", False)`.
- [ ] **Step 4:** `pytest tests/test_avionics.py tests/test_sizing.py tests/test_aero_cad.py -q` → PASS (Task 0 baseline still matches).
- [ ] **Step 5:** Commit `feat: sizing takes avionics mass and power from the design assumptions`.

### Task 3: Parts list lines for the flight controller and Jetson

**Files:**
- Modify: `src/schemas/components.py`, `src/tools/sizing/mission_solver.py:344-357` (`component_needs`), `src/tools/database/matcher.py:155-179`, `src/ui/presenters.py:99-115`
- Test: `tests/test_avionics.py`

**Interfaces — Produces:**
- `FlightController(_Component)` and `CompanionComputer(_Component)`, each with `power_w: Optional[float] = Field(None, ge=0)` and `reference_price_usd: str = Field(..., min_length=1, max_length=80)`.
- `SelectedBOM.flight_controller: Optional[FlightController] = None`, `SelectedBOM.companion_computer: Optional[CompanionComputer] = None`, both added to `total_hardware_mass_kg`.
- `ComponentNeeds.flight_controller: FlightControllerId = "cube_orange_plus"`, `ComponentNeeds.companion_computer: bool = False`.
- In `matcher.py`: `AVIONICS_RESEARCH_DATE: Final = "2026-11-18"` and `avionics_parts(needs: ComponentNeeds, assumptions: DesignAssumptions) -> tuple[FlightController, Optional[CompanionComputer]]`, names/prices from one `Final` dict:
  - `cube_orange_plus`: "CubePilot Cube Orange+ (ADS-B carrier)", "≈ $450–546"
  - `ark_jetson_pab`: "ARK Jetson PAB Carrier + ARKV6X + Jetson Orin NX 16 GB", "≈ $1,750–2,600 (carrier $800, ARKV6X $400, Orin NX 16 GB $599 1KU–$1,400 retail)"
  - companion: "NVIDIA Jetson Orin NX + carrier + heatsink", "≈ $1,000–1,400 (Orin NX 16 GB retail) + carrier"
- `bom_rows` adds columns "Power (W)" and "Ref. price" (empty for catalog parts) and rows "Flight controller" / "Companion computer"; the Cube's power cell reads "in avionics allowance".

- [ ] **Step 1: Failing tests:**
  - `test_cube_bom_has_flight_controller_only`: `avionics_parts(...)` → Cube line `mass_kg == 0.073`, companion `None`.
  - `test_cube_with_jetson_adds_companion`: companion `mass_kg == 0.2`, `power_w == 25.0`.
  - `test_ark_bom_has_stack_and_no_companion`: line `mass_kg == 0.166`, `power_w == 25.0`, companion `None`.
  - `test_total_hardware_mass_counts_avionics_lines`: `SelectedBOM` with both lines → total = previous total + 0.073 + 0.2.
  - `test_bom_rows_show_avionics_rows`: `bom_rows` contains "Flight controller" with "in avionics allowance" for the Cube.
  - `test_component_needs_copies_avionics_choice`.
- [ ] **Step 2:** `pytest tests/test_avionics.py -q` → FAIL (ImportError).
- [ ] **Step 3:** Implement; `select_bom` calls `avionics_parts(needs, assumptions)` and passes both to `SelectedBOM`.
- [ ] **Step 4:** `pytest tests/test_avionics.py tests/test_database.py tests/test_schemas.py tests/test_ui.py tests/test_pipeline.py -q` → PASS (existing total-mass tests constructing `SelectedBOM` by hand are unchanged because the new lines default to None).
- [ ] **Step 5:** Commit `feat: flight controller and Jetson lines in the parts list`.

### Task 4: CAD avionics tray reads the sized avionics mass

**Files:**
- Modify: `src/tools/aero_cad/components.py:20-25,167-173`, `src/tools/aero_cad/cad_packer.py:174-182`
- Test: `tests/test_aero_cad.py`

**Interfaces — Consumes:** `SizingOutput.avionics_mass_kg`, `airframe_mass_kg(..., avionics_mass_kg)` (Task 2).

- [ ] **Step 1: Failing test** `test_tray_mass_follows_sized_avionics`: take a `SizingOutput` from `size_aircraft(G1_with_ark)`; assert `sliders(sizing, bom)[1].mass_kg == approx(sizing.avionics_mass_kg + SYSTEMS_FRACTION * sizing.mtow_kg)`, and the same with `model_copy(update={"avionics_mass_kg": 1.0})` (the tray moves with it). Host-only; no OpenVSP.
- [ ] **Step 2:** `pytest tests/test_aero_cad.py -q -k tray` → FAIL.
- [ ] **Step 3:** Implement: drop the `AVIONICS_*` imports; pass `sizing.avionics_mass_kg` in `sized_airframe_mass_kg`.
- [ ] **Step 4:** `pytest tests/test_aero_cad.py -q` → PASS (OpenVSP tests skip on host).
- [ ] **Step 5:** Commit `feat: CAD avionics tray carries the chosen avionics stack`.

### Task 5: Parser, prompt and explorer guard

**Files:**
- Modify: `src/agent/agent_schemas.py:55-77`, `src/agent/prompts.py:7-16`, `src/agent/parsing.py:96-125`
- Test: `tests/test_agent.py`, `tests/test_explorer.py`

**Interfaces — Produces:** `RequirementsDraft.flight_controller: Optional[FlightControllerId] = Field(None, description=...)`, `RequirementsDraft.companion_computer: Optional[bool] = Field(None, description=...)`; `parsing._avionics_fields(draft) -> dict[str, object]` (null → `"cube_orange_plus"`, null → `True`).

- [ ] **Step 1: Failing tests:**
  - `test_parser_defaults_to_cube_with_jetson`: `to_requirements(SRS_DRAFT, db)` → Cube, `companion_computer is True`.
  - `test_parser_reads_ark`: draft with `flight_controller="ark_jetson_pab"` → ARK, `jetson_fitted`.
  - `test_parser_respects_explicit_no_companion`: `companion_computer=False` → False.
  - `test_explorers_never_touch_avionics` (in `test_explorer.py`): over the `closure_options` fixture plus `explore_improvement(G1_ELECTRIC_FW, DEFAULT_ASSUMPTIONS, db)`, every option's `change.requirements.flight_controller is None` and `.companion_computer is None`.
- [ ] **Step 2:** `pytest tests/test_agent.py tests/test_explorer.py -q` → parser tests FAIL.
- [ ] **Step 3:** Implement; add one sentence to `PARSER_SYSTEM`: ARK / PAB → `ark_jetson_pab`, Cube → `cube_orange_plus`, otherwise null; `companion_computer` true for autonomous / onboard AI / a named Jetson or companion computer, false only when explicitly excluded, otherwise null. (The parser tool schema comes from `RequirementsDraft`.)
- [ ] **Step 4:** `pytest tests/test_agent.py tests/test_explorer.py tests/test_design_chat.py -q` → PASS.
- [ ] **Step 5:** Commit `feat: descriptions can choose the avionics stack and Jetson`.

### Task 6: Web form, assumptions panel, results header, report

**Files:**
- Modify: `src/ui/presenters.py:15-65`, `src/ui/app.py:40-70,110-121,205-215`, `src/ui/assumptions_panel.py:22-46`
- Test: `tests/test_ui.py`

**Interfaces — Produces:**
- `presenters.FIELD_LABELS` gains `flight_controller`, `companion_computer` (this feeds the form submit dict at `app.py:313` and `report.requirement_rows` automatically).
- `presenters.FLIGHT_CONTROLLER_NAMES: Final[dict[str, str]]` = `{"cube_orange_plus": "Cube Orange+", "ark_jetson_pab": "ARK Jetson PAB (Orin NX 16 GB)"}`.
- `presenters.HARDWARE_KEYS: Final = ("flight_controller", "companion_computer")`, `presenters.NEW_DESIGN_HARDWARE: Final = {"flight_controller": "cube_orange_plus", "companion_computer": True}`.
- `presenters.avionics_summary(req, assumptions) -> str` returning exactly: "Cube Orange+ · Jetson Orin NX companion (25 W, 0.2 kg)", "Cube Orange+ · no Jetson", "ARK Jetson PAB + Orin NX 16 GB (25 W, 0.166 kg)" (numbers from the assumptions, `%g` formatting).
- `assumptions_panel.GROUPS["Avionics"]` with the eight fields; `GROUP_HELP["Avionics"]` explaining: shared allowance = base + per-MTOW (includes the flight controller's own power); Cube / ARK masses apply to the chosen stack; companion mass only for Cube with the Jetson ticked; Jetson power whenever a Jetson is fitted.

- [ ] **Step 1: Failing tests** (`AppTest.from_file`, as existing UI tests):
  - `test_avionics_inputs_default_to_cube_with_jetson`: `at.selectbox(key="flight_controller").value == "cube_orange_plus"`, `at.checkbox(key="companion_computer").value is True`.
  - `test_preset_change_keeps_avionics_choice`: untick, change `preset`, run → checkbox still False.
  - `test_jetson_checkbox_greys_out_live_under_ark`: select `ark_jetson_pab`, run (no submit) → the Jetson checkbox shown is disabled and ticked; switch back to Cube → enabled again with the earlier Cube choice intact.
  - `test_avionics_group_in_assumptions_panel`: number input for `jetson_power_w` exists with value 25.0.
  - `test_avionics_summary_texts`: the three exact strings above.
  - `test_report_requirements_list_avionics`: `report.requirement_rows(req)` contains a ("Flight controller", ...) row.
- [ ] **Step 2:** `pytest tests/test_ui.py -q` → FAIL.
- [ ] **Step 3:** Implement: `_load_preset` skips `HARDWARE_KEYS`; `_init_state` `setdefault`s each `NEW_DESIGN_HARDWARE` item; a new `_avionics_inputs()` rendered just before `_requirements_form()` (outside any form) with a `selectbox` (`format_func=FLIGHT_CONTROLLER_NAMES.get`) and the `checkbox` (`disabled=` ARK selected; when disabled it displays ticked without overwriting the stored Cube choice — render it with `value=True` under a separate display key, keeping `companion_computer` untouched; caption "built into the ARK stack"); `st.caption("Avionics: " + avionics_summary(...))` under the headline metrics (`app.py:205-213`) from `view.requirements` / `view.design_assumptions` when `view.requirements` is set.
- [ ] **Step 4:** `pytest -q` (full host suite) → all pass; coverage ≥ 80 % (`pytest --cov=src -q`).
- [ ] **Step 5:** Commit `feat: avionics stack choice and assumptions in the web form`.

### Task 7: Docs and live check

**Files:**
- Modify: `README.md` (outputs / assumptions section)

- [ ] **Step 1:** Add an "Avionics" paragraph to the README: the two stacks, the optional Jetson, the eight assumptions with defaults, and the research date and sources from the spec.
- [ ] **Step 2:** Rebuild and restart the container; check that the form shows the select and checkbox, and run G1 with ARK and with Cube+Jetson: parts list shows the lines, header caption shows the stack, MTOW ordering Cube+Jetson > ARK > Cube alone.
- [ ] **Step 3:** Commit `docs: avionics stack choice`.
