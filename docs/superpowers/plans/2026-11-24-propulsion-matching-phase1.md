# Propulsion Matching (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fixed propulsive efficiencies with a blade-element propeller model and a DC motor model, match
motor winding, propeller, cell-built pack and ESC for cruise and lift, and feed the matched efficiencies back into
the sizing loop and the simulator.

**Architecture:** A new `src/tools/propulsion/` package owns the physics (blade-element propeller, cached CT/CP
tables, motor model, a `Propulsor` that couples them, the matching search). Sizing gets an outer fixed-point loop:
size with frozen efficiencies → match hardware → recompute efficiencies → resize, until the matched set repeats.
The matched set travels in `SizingOutput.propulsion_match`; `select_bom` picks those exact rows by name, and the
flight model and JSBSim export build their thrust from the same `Propulsor`.

**Tech Stack:** Python 3.11, numpy, aerosandbox/NeuralFoil (already a dependency), pydantic v2, pytest; ArduPilot
SITL + JSBSim for the gate.

**Spec:** `docs/superpowers/specs/2026-11-24-propulsion-electrical-avionics-design.md` (phase 1 = sections 6.1–6.8;
section 7 is a later plan).

## Global Constraints

- Branch `feat/propulsion-matching`. Host tests: `.venv/bin/python -m pytest -q -p no:cacheprovider`. Docker:
  `docker run --rm -v "$PWD":/app -w /app isr-forge:latest python -m pytest -q -p no:cacheprovider`.
- `PROP_TIP_MACH_MAX = 0.70`; `MAX_MATCH_THROTTLE = 0.85` (nominal voltage); throttle ≤ 1.0 at the sagged voltage.
- `LI_ION_LOADED_MIN_V_CELL = 3.3`, `LIPO_LOADED_MIN_V_CELL = 3.5`; `ESC_CURRENT_MARGIN = 1.2`;
  `MAX_PROPULSION_PASSES = 6`.
- Validation gate: model peak efficiency and J at peak within ±10 %, CT0 within ±15 %, for ≥ 3 UIUC datasets.
- No silent fallback to a fixed efficiency: every model or match failure raises a readable error naming the
  limiting constraint (`PropulsionMatchError`, a `SizingDivergenceError` subclass).
- Every estimate is flagged in the report: generic blade shape, motor Rm/I0/max current, ESC efficiency, cell data.
- Spec 6.8 (user): controller's-call constants may be tuned during the build with a ledger ruling and reason,
  reported at the end; reference missions and user decisions never change.
- Pusher layouts keep `pusher_efficiency_factor` as an installation loss on propeller efficiency.
- Saved records must still load: assumptions carrying `drivetrain_efficiency`, `hover_figure_of_merit`,
  `lift_motor_efficiency`; `bom.json` batteries without cell fields; `SizingOutput` without `propulsion_match`.
- Frozen dataclasses / frozen pydantic models; type hints everywhere; files ≤ 800 lines; functions < 50 lines.
- No secrets in source; the only new environment variable is `PROP_TABLE_CACHE_DIR` (optional).
- Commits end with:
  `Co-Authored-By: WOZCODE <contact@withwoz.com>` and
  `Claude-Session: https://claude.ai/code/session_01RJfaxdHYf7KowUaPPCoa8G`.

## Review Focus

1. Thrust beyond the largest catalogue motor/propeller/ESC → `PropulsionMatchError` naming the constraint
   (power, current, tip Mach, throttle, diameter), never an `IndexError`/`ValueError` from an empty `min()`.
   Pinned in Task 5.
2. Airspeed above the propeller's pitch speed (windmilling, CT ≤ 0) or J outside the table → thrust solve returns
   "not achievable", never NaN or negative thrust. Pinned in Tasks 2 and 4.
3. Legacy records (old assumptions keys, old battery rows, no `propulsion_match`) load and re-run. Pinned in
   Tasks 3, 4 and 6.
4. Series hybrid where no generator voltage variant sits in the bus window → readable refusal naming the voltage.
   Pinned in Tasks 4 and 5.
5. A truncated or stale-version propeller cache file → rebuilt with a logged warning, never a crash or a stale
   table. Pinned in Task 2.

---

### Task 1: Blade-element propeller model and UIUC validation

**Files:**
- Create: `src/tools/propulsion/__init__.py`, `src/tools/propulsion/bemt.py`
- Create: `tests/data/uiuc/` (three APC thin-electric datasets with static and advance-ratio files, plus
  `SOURCES.md` with the citation: Brandt, Deters, Ananda & Selig, UIUC Propeller Data Site)
- Test: `tests/test_propeller_bemt.py`

**Interfaces:**
- Produces:
  - `BladeGeometry` (frozen dataclass): `diameter_m: float`, `pitch_m: float`, `n_blades: int`,
    `stations: tuple[float, ...]` (r/R), `chord_over_radius: tuple[float, ...]`.
  - `generic_blade(diameter_m: float, pitch_m: float, n_blades: int = 2) -> BladeGeometry`.
  - `PropPoint` (frozen dataclass): `thrust_n`, `torque_nm`, `shaft_power_w`, `efficiency` (0.0 at zero airspeed).
  - `bemt_point(blade: BladeGeometry, rpm: float, airspeed_ms: float, rho: float, mu: float,
    speed_of_sound_ms: float) -> PropPoint`.
  - `PROP_AIRFOIL: Final[str]` (generic thin propeller section, e.g. `"clarky"`), `PROP_MODEL_VERSION: Final[int] = 1`.

- [ ] **Step 1: Download three UIUC datasets** (APC Thin Electric, e.g. 9x6E, 10x7E, 11x7E — whichever exist in
  Volume 1 with both static and advance-ratio files) into `tests/data/uiuc/`, and write `SOURCES.md` (URL, file
  names, retrieval date).
- [ ] **Step 2: Write the failing tests**

```python
def test_generic_blade_has_constant_geometric_pitch(): # twist = atan(P / (2*pi*r)) at every station
def test_static_thrust_positive_and_efficiency_zero():
    p = bemt_point(generic_blade(0.254, 0.178), 6000, 0.0, 1.225, 1.81e-5, 340.3)
    assert p.thrust_n > 0 and p.efficiency == 0.0
def test_thrust_falls_to_zero_near_pitch_speed():  # V = 0.95 * P * rpm/60 -> thrust < 10 % of static
@pytest.mark.parametrize("dataset", UIUC_DATASETS)
def test_matches_uiuc_peak_efficiency_and_j(dataset):  # |eta_peak_model/eta_peak_data - 1| <= 0.10, same for J
@pytest.mark.parametrize("dataset", UIUC_DATASETS)
def test_matches_uiuc_static_ct(dataset):              # |CT0_model/CT0_data - 1| <= 0.15
```

- [ ] **Step 3: Run** `.venv/bin/python -m pytest tests/test_propeller_bemt.py -q` — Expected: FAIL (module missing).
- [ ] **Step 4: Implement `bemt.py`.** Blade-element momentum with Prandtl tip and hub loss; section CL/CD from one
  NeuralFoil alpha sweep per (Reynolds, Mach) node of a small grid, interpolated (`lru_cache` the sweeps — never call
  NeuralFoil per section per iteration). Chord distribution and hub fraction are the builder's choice inside the
  validation gate (spec 8).
- [ ] **Step 5: Run the tests** — Expected: PASS. If a dataset misses the gate, tune the chord/airfoil first; tuning a
  tolerance needs a ledger ruling (spec 6.8).
- [ ] **Step 6: Commit** `feat: blade-element propeller model validated against UIUC data`.

### Task 2: Propeller tables with disk cache

**Files:**
- Create: `src/tools/propulsion/prop_tables.py`
- Modify: `.gitignore` (add `.cache/`)
- Test: `tests/test_prop_tables.py`

**Interfaces:**
- Consumes: Task 1 `generic_blade`, `bemt_point`, `PROP_MODEL_VERSION`, `PROP_AIRFOIL`.
- Produces:
  - `PropellerTable` (frozen dataclass): `diameter_m`, `pitch_m`, `j: tuple[float, ...]`,
    `reynolds75: tuple[float, ...]`, `ct: tuple[tuple[float, ...], ...]`, `cp: tuple[tuple[float, ...], ...]`
    with methods `coefficients(j: float, reynolds75: float) -> tuple[float, float]` (raises `ValueError` outside
    the J range), `thrust_n(rpm, airspeed_ms, rho, mu) -> float`, `torque_nm(rpm, airspeed_ms, rho, mu) -> float`,
    `pitch_speed_ms(rpm) -> float`.
  - `propeller_table(diameter_m: float, pitch_m: float) -> PropellerTable` (memory `lru_cache` + JSON on disk under
    `os.environ.get("PROP_TABLE_CACHE_DIR", "<repo>/.cache/prop_tables")`, file key includes
    `PROP_MODEL_VERSION`, `PROP_AIRFOIL`, D and P).

- [ ] **Step 1: Write the failing tests**

```python
def test_table_matches_bemt_within_two_percent():    # random (rpm, V) inside the grid, thrust and torque
def test_coefficients_outside_j_range_raise():       # j > table max -> ValueError
def test_thrust_never_negative():                    # windmilling region clamps CT at 0 for thrust_n
def test_cache_round_trip(tmp_path, monkeypatch):    # second call reads disk, no bemt_point calls (monkeypatch spy)
def test_corrupt_cache_file_is_rebuilt(tmp_path, monkeypatch):  # truncated JSON -> rebuilt, warning logged
def test_stale_version_not_used(tmp_path, monkeypatch):         # file with another version key is ignored
```

- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_prop_tables.py -q` — Expected: FAIL.
- [ ] **Step 3: Implement.** CT = T/(ρn²D⁴), CP = P/(ρn³D⁵), n in rev/s, Re75 from the 0.75R chord and blade speed;
  write the cache atomically (temp file + rename).
- [ ] **Step 4: Run the tests** — Expected: PASS.
- [ ] **Step 5: Commit** `feat: cached propeller CT/CP tables`.

### Task 3: Motor model and catalogue (motors, ESCs, propellers)

**Files:**
- Create: `src/tools/propulsion/motor.py`
- Modify: `src/schemas/components.py` (motor fields, `Esc`, propeller fields), `src/tools/database/seed_data.py`
  (rows, new `escs` table, `catalog_rows`), `src/tools/database/matcher.py` (`part_by_name`)
- Test: `tests/test_motor_model.py`, `tests/test_database.py`

**Interfaces:**
- Produces:
  - `MotorPoint` (frozen dataclass): `rpm`, `current_a`, `voltage_v`, `shaft_power_w`, `efficiency`.
  - `motor_point(kv_rpm_per_v: float, rm_ohm: float, i0_a: float, rpm: float, torque_nm: float) -> MotorPoint`
    (Kt = 60/(2π·KV); I = torque/Kt + I0; V = I·Rm + rpm/KV).
  - `CruiseMotor` / `VtolMotor` gain `rm_ohm`, `i0_a`, `max_current_a` as `Optional[float] = None` (legacy rows
    load; matching skips rows with `None`); `voltage_v` keeps its name and now means the winding's max rated
    voltage (document it on the field).
  - `Esc(_Component)`: `continuous_current_a`, `burst_current_a`, `max_voltage_v`, `efficiency` (0.5–1.0),
    `length_m`, `width_m`, `height_m`.
  - `Propeller` gains `is_folding: bool = False`.
  - `catalog_rows(table: str) -> list[BaseModel]` in `seed_data.py`: the validated rows `seed_database` inserts
    (single source for the DB and for sizing-time matching).
  - `part_by_name(db_path: Path, table: str, name: str) -> BaseModel` in `matcher.py`
    (`ComponentNotFoundError` when absent).
- Catalogue content (estimates, flagged): two to three KV windings per existing motor size (same mass and size;
  Rm ∝ 1/KV², I0 ∝ KV, max current from max power at the winding's voltage); propellers at several P/D
  (cruise 0.5–1.0, lift 0.3–0.6) per diameter, plus folding cruise variants (hub mass added); ESC classes covering
  the motors' currents and voltages up to the axial-flux sizes. Generate winding and pitch variants in code from
  the base rows, not by hand-copying.

- [ ] **Step 1: Write the failing tests**

```python
def test_motor_point_no_load():   # torque 0 -> current == i0, voltage == rpm/kv + i0*rm
def test_motor_efficiency_peaks_between_no_load_and_stall():
def test_every_motor_size_has_at_least_two_windings():
def test_every_cruise_diameter_has_several_pitches_and_a_folding_variant():
def test_escs_cover_every_motor_current_and_voltage():  # for each motor: some ESC with cont >= 1.2*max_current, V ok
def test_catalog_rows_match_seeded_database(tmp_path):
def test_legacy_motor_row_without_constants_loads():
def test_part_by_name_missing_raises():
```

- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_motor_model.py tests/test_database.py -q` — FAIL.
- [ ] **Step 3: Implement** the model, schema fields and rows.
- [ ] **Step 4: Run the tests** — Expected: PASS.
- [ ] **Step 5: Commit** `feat: motor model, KV windings, ESC and propeller pitch catalogue`.

### Task 4: Cell-built packs, generator voltages, Propulsor

**Files:**
- Create: `src/tools/database/packs.py`, `src/tools/propulsion/propulsor.py`, `src/schemas/propulsion.py`
- Modify: `src/schemas/components.py` (`Cell`, `Battery` cell fields, `IceGenerator.output_voltage_v`),
  `src/tools/database/seed_data.py` (`cells` replaces `batteries`; genset voltage variants),
  `src/tools/database/matcher.py` (`match_battery` builds packs; `match_ice_generator` takes a voltage window)
- Test: `tests/test_packs.py`, `tests/test_propulsor.py`, `tests/test_database.py`

**Interfaces:**
- Consumes: Task 2 `PropellerTable`, Task 3 `motor_point`, `Esc`.
- Produces:
  - `Cell(_Component)`: `chemistry`, `nominal_voltage_v`, `max_voltage_v`, `capacity_wh`, `max_current_a`, dims.
  - `Battery` gains `cell_name: Optional[str] = None`, `series: Optional[int] = None`, `parallel: Optional[int] = None`.
  - `build_pack(cell: Cell, series: int, energy_j: float, peak_power_w: float,
    assumptions: DesignAssumptions) -> Battery` — P = smallest meeting energy and peak current at the nominal pack
    voltage; capacity = S·P·cell Wh; mass and box from `pack_envelope` with the run's specific energy (the editable
    assumption still governs pack mass); `max_c_rate` from the cell current.
  - `loaded_min_voltage_v(chemistry: str, series: int) -> float` and `charged_voltage_v(cell: Cell, series: int) -> float`.
  - `IceGenerator.output_voltage_v: Optional[float] = None` (legacy rows None); `match_ice_generator(db_path,
    required_power_w, bus_window_v: tuple[float, float])`.
  - `OperatingPoint` (pydantic, frozen, `src/schemas/propulsion.py`): `flight_point: str`,
    `propulsor: Literal["cruise", "lift"]`, `rpm`, `current_a`, `throttle`, `prop_efficiency` (figure of merit for
    hover points), `motor_efficiency`, `tip_mach`, `electrical_power_w`, `sagged: bool`.
  - `Propulsor` (frozen dataclass): `table: PropellerTable`, `kv`, `rm_ohm`, `i0_a`, `max_current_a`,
    `esc_efficiency`, `bus_voltage_v`, `installation_factor: float = 1.0`, with
    `point_for_thrust(name: str, thrust_n, airspeed_ms, rho, mu, speed_of_sound_ms, bus_voltage_v)
    -> Optional[OperatingPoint]` (None when not achievable) and
    `thrust_n(throttle, airspeed_ms, rho, mu) -> float` (solves rpm where motor torque at throttle·bus voltage
    equals propeller torque; current clamped at `max_current_a`; ≥ 0).
  - `propulsor_from_parts(motor, propeller: Propeller, esc: Esc, battery: Battery,
    installation_factor: float = 1.0) -> Propulsor`.

- [ ] **Step 1: Write the failing tests**

```python
def test_pack_parallel_count_meets_energy_and_current():
def test_pack_voltage_is_series_times_nominal():
def test_hybrid_buffer_pack_is_lipo():
def test_legacy_battery_without_cells_loads():
def test_generator_outside_bus_window_not_matched():       # ComponentNotFoundError naming the window
def test_point_for_thrust_round_trips_thrust_n():          # thrust_n(point.throttle, ...) ~= requested within 1 %
def test_point_for_thrust_above_pitch_speed_is_none():
def test_thrust_n_is_monotonic_in_throttle():
def test_current_limit_caps_thrust():
```

- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_packs.py tests/test_propulsor.py tests/test_database.py -q`
  — FAIL.
- [ ] **Step 3: Implement**, removing the fixed `BATTERIES` pack rows (tests that named them move to cell-built packs).
- [ ] **Step 4: Run the tests** — Expected: PASS.
- [ ] **Step 5: Commit** `feat: cell-built packs, generator voltages and the propulsor model`.

### Task 5: Matching search

**Files:**
- Create: `src/tools/propulsion/matching.py`
- Modify: `src/schemas/propulsion.py` (`PropulsionMatch`)
- Test: `tests/test_propulsion_matching.py`

**Interfaces:**
- Consumes: Tasks 3–4 (`catalog_rows`, `build_pack`, `Propulsor`, `OperatingPoint`, `loaded_min_voltage_v`).
- Produces:
  - `FlightPoint` (frozen dataclass): `name: str`, `is_hover: bool`, `airspeed_ms`, `rho`, `mu`,
    `speed_of_sound_ms`, `thrust_n` (per propulsor), `duration_s`, `check_sagged: bool`.
  - `MatchContext` (frozen dataclass): `points: tuple[FlightPoint, ...]`, `chemistry: Optional[str]`,
    `is_hybrid: bool`, `lift_rotor_count: int`, `min_lift_rotor_diameter_m: float`,
    `max_cruise_prop_diameter_m: float`, `require_folding_prop: bool`, `pusher_factor: float`,
    `hotel_w: float`, `assumptions: DesignAssumptions`.
  - `PropulsionMatch` (pydantic, frozen): `cell_name`, `series`, `cruise_motor`, `cruise_propeller`,
    `cruise_esc`, `lift_motor: Optional[str]`, `lift_rotor: Optional[str]`, `lift_esc: Optional[str]`,
    `operating_points: tuple[OperatingPoint, ...]`, `efficiencies: dict[str, float]` (keys: forward phase names
    `"Climb"`, `"Ingress"`, `"Loiter"`, `"Egress"`, and for quadplanes `"Hover"`, `"Hover_Peak"` = ideal induced
    power / electrical power), `propulsion_mass_kg: float`, `notes: tuple[str, ...] = ()`, and a `set_key`
    property (the tuple spec 6.4 defines as "the set").
  - `match_propulsion(context: MatchContext) -> PropulsionMatch`.
  - `PropulsionMatchError(SizingDivergenceError)`.
  - Constants `PROP_TIP_MACH_MAX`, `MAX_MATCH_THROTTLE`, `ESC_CURRENT_MARGIN` (Global Constraints values).
- Rules (spec 6.3–6.4): every point at nominal voltage with throttle ≤ 0.85; points with `check_sagged` (climb and
  hover) also at the loaded-minimum voltage with throttle ≤ 1.0; current ≤ motor `max_current_a`; power within
  motor max; tip Mach ≤ 0.70; cruise D ≤ `max_cruise_prop_diameter_m`; lift rotor = smallest catalogue lift
  diameter ≥ the minimum; ESC = lightest with continuous current ≥ 1.2 × highest matched current and voltage ≥
  charged pack voltage; hybrid: S only where a generator voltage variant fits the bus window. Objective:
  propulsion set mass + estimated battery (electric) or fuel (hybrid) mass from the mission energy at that set's
  efficiencies. Prune by motor power before solving rpm.

- [ ] **Step 1: Write the failing tests**

```python
def test_g1_cruise_point_picks_consistent_voltage():   # motor.voltage_v >= charged pack voltage of chosen S
def test_chosen_set_meets_every_rule():                # throttle, current, tip Mach, sagged climb
def test_more_thrust_never_picks_a_lighter_set():
def test_thrust_beyond_catalogue_names_the_constraint():
    with pytest.raises(PropulsionMatchError, match="current|power|tip Mach|throttle|diameter"): ...
def test_quadplane_lift_rotor_not_smaller_than_disk_loading_diameter():
def test_hand_launch_requires_folding_prop():
def test_hybrid_without_generator_in_bus_window_refused():
def test_efficiencies_cover_every_forward_phase_and_hover():
```

- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_propulsion_matching.py -q` — FAIL.
- [ ] **Step 3: Implement** (search pruning and tie-breaking are the builder's, spec 8).
- [ ] **Step 4: Run the tests** — Expected: PASS; one match for the G1 reference flight points runs in < 5 s.
- [ ] **Step 5: Commit** `feat: propulsion matching search`.

### Task 6: Sizing integration (outer loop, efficiencies, mass, records)

**Files:**
- Create: `src/tools/sizing/propulsion_loop.py`
- Modify:
  - `src/tools/sizing/mission_profile.py`: `simulate`, `_step_power_w`, `forward_power_w`, `hover_power_w` take a
    `PropulsionEfficiencies`.
  - `src/tools/sizing/mission_solver.py`: `_Design` gains the efficiencies and the frozen set; `_evaluate`;
    `_size_powertrain` wrapped by the loop; `_to_output`; `component_needs`.
  - `src/tools/sizing/mass_model.py`: propulsion mass from the set when present; the matched cruise D replaces
    `PROP_TO_FUSELAGE_DIAMETER_RATIO` after pass 0 (gear, pusher gap and clearance use it).
  - `src/schemas/state.py`: `SizingOutput.propulsion_match: Optional[PropulsionMatch] = None`.
  - `src/schemas/components.py`: `ComponentNeeds` gains the matched part names; `SelectedBOM` gains
    `cruise_esc: Optional[Esc]`, `vtol_esc: Optional[Esc]`; `total_hardware_mass_kg` counts them (lift ESC × rotor
    count).
  - `src/tools/database/matcher.py`: `select_bom` uses the names and builds the pack with `build_pack`.
  - `src/schemas/assumptions.py`, `src/schemas/changes.py`, `src/ui/assumptions_panel.py`: remove the three
    efficiencies; legacy keys accepted and dropped by a `mode="before"` validator (logged).
  - `src/tools/aero_cad/components.py`: each motor item includes its ESC mass; the lift-motor install allowance
    no longer counts the ESC.
- Test: `tests/test_propulsion_loop.py`, `tests/test_sizing.py`, `tests/test_assumptions.py`, `tests/test_schemas.py`

**Interfaces:**
- Consumes: Task 5 `match_propulsion`, `MatchContext`, `FlightPoint`, `PropulsionMatch`, `PropulsionMatchError`.
- Produces:
  - `PropulsionEfficiencies` (frozen dataclass, `mission_profile.py`): `forward: Mapping[str, float]`,
    `hover: float`, `hover_peak: float`; `seed_efficiencies(spec: LayoutSpec) -> PropulsionEfficiencies` (pass 0
    only: 0.68 × pusher factor forward, 0.72 × 0.85 hover, as named `SEED_*` constants).
  - `flight_points(ev: _Evaluation) -> tuple[FlightPoint, ...]` (per forward phase at its mean weight: thrust =
    propulsive power × efficiency used / speed; hover at T/W 1 and `VTOL_THRUST_TO_WEIGHT` per rotor at take-off
    density; climb and hover flagged `check_sagged`; speed of sound from the ISA temperature at that altitude plus
    the requirement's ISA offset — add `speed_of_sound_ms(altitude_m, temperature_offset_c)` beside
    `air_properties` in `aero_model.py`).
  - `size_with_matching(req, airframe, propulsion, params, assumptions) -> _Evaluation` — the outer loop
    (spec 6.5): stop when `set_key` repeats consecutively; on a cycle take the heavier set, re-size once with it
    and add a note; more than `MAX_PROPULSION_PASSES` → `PropulsionMatchError`.

- [ ] **Step 1: Write the failing tests**

```python
def test_simulate_uses_per_phase_efficiency():         # halving "Loiter" efficiency doubles loiter propulsive power
def test_outer_loop_converges_for_g1_reference():      # propulsion_match set, set_key stable on a re-run
def test_cycle_takes_heavier_set(monkeypatch):         # fake match alternating A/B -> heavier chosen, note present
def test_no_convergence_raises(monkeypatch):           # always-new sets -> PropulsionMatchError after 6 passes
def test_propulsion_mass_comes_from_matched_set():
def test_select_bom_returns_the_matched_parts():       # names in bom == names in sizing.propulsion_match
def test_legacy_assumptions_with_efficiency_keys_load():
def test_legacy_sizing_record_without_match_loads():
```

- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_propulsion_loop.py -q` — FAIL.
- [ ] **Step 3: Implement.** Keep `mission_solver.py` under 800 lines by putting the loop in `propulsion_loop.py`.
- [ ] **Step 4: Run** `tests/test_propulsion_loop.py tests/test_sizing.py tests/test_assumptions.py tests/test_schemas.py`
  — Expected: PASS except numeric baselines Task 9 re-baselines (list them in the report; don't edit them here).
- [ ] **Step 5: Commit** `feat: matched propulsion efficiencies drive the sizing loop`.

### Task 7: Simulator, JSBSim export and ArduPilot parameters

**Files:**
- Modify: `src/tools/flight_control/flight_model.py` (drop `PROP_EFFICIENCY`; `FlightModel` carries the cruise and
  lift `Propulsor`; `cruise_thrust_n(throttle, speed)` and the lift max thrust come from them),
  `src/tools/flight_control/jsbsim_exporter.py` (`lift_thrust_curve` from the lift propulsor; cruise table from the
  cruise propulsor), `src/tools/flight_control/param_generator.py` (`Q_M_THST_EXPO` fitted to the lift curve;
  `Q_M_THST_HOVER` from the new max thrust; battery failsafe voltages from the series count and
  `LI_ION_LOADED_MIN_V_CELL`/`LIPO_LOADED_MIN_V_CELL`)
- Test: `tests/test_flight_control.py`, `tests/test_sitl_validation.py`

**Interfaces:**
- Consumes: Task 4 `propulsor_from_parts`, Task 6 `SelectedBOM.cruise_esc`/`vtol_esc`.
- Produces: `fit_thrust_expo(throttles: Sequence[float], thrust_fraction: Sequence[float]) -> float` (least squares
  on ArduPilot's T/Tmax = (1−e)u + e·u², clamped to [0, 1]).

- [ ] **Step 1: Write the failing tests**

```python
def test_cruise_thrust_matches_propulsor():
def test_lift_max_thrust_from_propulsor_not_catalogue():
def test_fit_thrust_expo_recovers_known_curve():       # synthetic curve with e=0.65 -> 0.65 +/- 0.01
def test_param_expo_and_hover_follow_the_rotor_model():
def test_battery_failsafe_voltages_follow_series_count():
def test_no_prop_efficiency_constant_left():           # 'PROP_EFFICIENCY' absent from flight_model
```

- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_flight_control.py -q` — FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `tests/test_flight_control.py` — PASS; SITL tests may need Task 9's baselines (report them).
- [ ] **Step 5: Commit** `feat: simulator and ArduPilot parameters use the matched propulsion`.

### Task 8: Report, critic and UI

**Files:**
- Modify: `src/agent/presenters.py` (a "Propulsion match" table: flight point, propulsor, rpm, current, throttle,
  propeller efficiency or FM, motor efficiency, tip Mach; BOM lines for ESCs and the pack as `<cell> <S>S<P>P`),
  the report assumptions list (flagged: generic blade shape, motor constants, ESC efficiency, cell data), the critic
  (a `PropulsionMatchError` is reported with its constraint and no knob that cannot fix it — follow the
  "unfixable" pattern used for the L/D gate), the UI assumptions panel (three efficiency rows gone)
- Test: `tests/test_agent.py`, `tests/test_ui.py`, `tests/test_report_pdf.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_report_has_propulsion_match_table():
def test_report_lists_propulsion_estimates_as_assumptions():
def test_critic_names_propulsion_constraint():
def test_assumptions_panel_has_no_efficiency_rows():
def test_bom_lines_show_escs_and_pack_configuration():
```

- [ ] **Step 2: Run** the three test files — FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat: report, critic and UI show the propulsion match`.

### Task 9: Re-baseline and full gate

**Files:**
- Modify: golden and fixture data (`tests/test_layout_golden.py` snapshot data, `tests/reference_designs.py`
  expectations, `tests/data/*` as the suite requires)
- Create: `docs/superpowers/plans/2026-11-24-propulsion-rebaseline-notes.md` (what moved and why, per design)

- [ ] **Step 1:** Run all six reference designs through the full `--spec` pipeline (host); record MTOW, battery,
  matched set and SITL result before (master) and after.
- [ ] **Step 2:** Any design that stops closing: diagnose; tune only controller's-call constants, each with a ledger
  ruling (spec 6.8); never change a reference mission.
- [ ] **Step 3:** Regenerate the golden snapshots and fixtures once; write the re-baseline notes.
- [ ] **Step 4:** Host suite — Expected: all pass. Docker suite — Expected: all pass.
- [ ] **Step 5: Commit** `test: re-baseline reference designs on matched propulsion`.
