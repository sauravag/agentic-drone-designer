# Airframe Layouts Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the single-boom V-tail pusher (fixed-wing and quadplane) and hand launch for Group 1 fixed-wing, re-baseline every layout to explicit boom mass and drag, and close the phase-1 interface gaps they need.

**Architecture:** Builds on phase 1's `LayoutSpec` (layout id → tail / booms / propeller / wing) and its general geometry (tail surfaces, booms, propeller station, per-surface control channels, shared gear). Phase 2 adds a launch method beside the topology, one landing-gear rule fed by the real propeller position, per-servo control-surface derivatives from VSPAERO, a V-tail tail kind, a boom-end propeller station and the V-tail kit. Subsystems branch on `LayoutSpec` properties and the resolved launch method, never on the layout id.

**Tech Stack:** Python 3.11, pydantic v2, AeroSandbox/NeuralFoil, OpenVSP/VSPAERO (Docker), build123d, JSBSim + ArduPilot SITL (Docker), pytest.

**Spec:** `docs/superpowers/specs/2026-11-22-airframe-layouts-design.md` (sections 1–5, Phases item 2, and the 2026-11-23 decisions log entries: probe result, taller gear, hand launch).

## Global Constraints

- Host tests: `.venv/bin/python -m pytest -q -p no:cacheprovider`. Docker (OpenVSP, JSBSim, ArduPilot): `docker run --rm -v "$PWD":/app -w /app isr-forge:latest python -m pytest -q -p no:cacheprovider` (full run ≈ 22 min; run focused files per task, the full suite in Task 9).
- Subsystems branch on `LayoutSpec` properties and `resolved_launch_method(...)`, never on the layout id string.
- Phase 2 deliberately re-baselines every layout, the tractor included, **once** (Task 2): explicit tail-boom and lift-boom tubes, boom wetted-area drag plus a stopped-rotor term replacing `QUADPLANE_DRAG_PENALTY`, and the change named in the README/REQUIREMENTS changelog. Every other task leaves the tractor reference outputs (golden snapshots) unchanged.
- V-tail sizing: total area `S_h + S_v`, dihedral `atan(sqrt(S_v / S_h))`, 2 panels (one mirrored surface), control kind `ruddervator`; effective areas stay Σ S·cos²Γ (horizontal) and Σ S·sin²Γ (vertical).
- Servo functions: aileron 4, elevator 19, rudder 21, V-tail left 79, V-tail right 80, ground steering 26, throttle 70, lift motors 33–36.
- `DesignAssumptions.pusher_efficiency_factor`, default 0.95 (bounds 0.80–1.00), applied to the cruise drivetrain efficiency for pusher layouts only.
- Hand launch: fixed-wing only, no landing gear, MTOW ≤ 9.07 kg (Group 1, 20 lb); a hand-launched pusher always gets a folding-propeller build-sheet note; a fixed-wing without a runway must state `launch_method="hand"` (validation error otherwise).
- Gear legs: runway aircraft size the main-gear legs so the propeller tip clears the ground at the 12° take-off rotation; VTOL legs give level clearance; hand-launched aircraft have no gear.
- No secrets in code; the LLM model is changed only from `.env` (unchanged here).
- Fix minor review issues in the task that raised them; do not defer them.
- Commit messages end with the two attribution lines from the session (`Co-Authored-By: WOZCODE <contact@withwoz.com>` and the `Claude-Session:` line).

## Review Focus

1. **Records saved before phase 2** (no `control_surfaces` in the aero output, no `launch_method`, no gear kind) still load, rebuild a flight model and export JSBSim exactly as today. Pinned in Task 4 (`test_legacy_aero_record_gives_tractor_channels`) and Task 1 (`test_old_requirements_record_defaults_launch`).
2. **A hand-launched aircraft reaching code that assumes gear contacts** (exporter, SITL bridge, kit residual gear mass): it must never crash on an empty contact list. Pinned in Task 3 (`test_hand_launch_gear_has_no_contacts_and_zero_mass`) and Task 5 (`test_export_without_gear_writes_belly_contacts`).
3. **A reversed ruddervator servo** must make the SITL check fail, the V-tail analogue of phase 1's reversed-elevator test. Pinned in Task 7 (`test_reversed_ruddervator_fails_sitl`, Docker).
4. **Design chat switching an existing design to the V-tail pusher or to hand launch**: the new fields flow end to end, and an unbuilt layout still gives the "not available yet" message. Pinned in Task 9 (`test_chat_switches_to_vtail_pusher`) and Task 1 (`test_parse_hand_launch_phrase`).
5. **A small aircraft whose pusher disc reaches the V-tail or the ground**: reported as a CAD collision-class failure the critic sees, not a silent pass. Pinned in Task 6 (`test_prop_disc_into_tail_is_a_collision`, `test_prop_ground_strike_is_reported`).

## Phase-1 carry-overs: where each one lands

| Carry-over (memory, 2026-11-23) | Task |
|---|---|
| Per-surface VSPAERO derivatives; perturbation scaling moved onto the channels | 4, 7 |
| Gear kind "none" (empty contacts crash the exporter); gear mass from the gear | 3, 5 |
| Steering not tied to the rudder | 4 |
| Critic `sm_knob` becomes a required keyword | 1 |
| Servos 5–8 reserved only when quad motors exist | 4 |
| `PropellerStation` diameter from the BOM | 6 |
| Reversed-ruddervator SITL test | 7 |
| Echo the layout in the parse summary | 1 |
| Boom-end pusher feasibility probe | **done 2026-11-23** (spec decisions log) |
| Tail-surface lateral (y) position | **deferred to phase 3**: V-tail panels are mirrored about y = 0; only twin booms need an offset |
| `wing_sweep_deg` STEPS / OVERRIDE_BOUNDS | **deferred to phase 5**: no phase-2 layout has the sweep knob |

---

### Task 1: Launch method and agent carry-overs

**Files:**
- Modify: `src/schemas/requirements.py` (field + validator), `src/tools/sizing/topology.py`, `src/agent/agent_schemas.py`, `src/agent/parsing.py`, `src/agent/critic.py`, `src/agent/nodes.py`, `src/ui/presenters.py`, `src/ui/app.py`, `src/schemas/changes.py` (design-chat field if requirement changes are listed there)
- Test: `tests/test_layouts.py`, `tests/test_agent.py`, `tests/test_sizing.py`

**Interfaces:**
- Produces: `OperationalRequirements.launch_method: Literal["auto", "runway", "hand"] = "auto"`; `resolved_launch_method(req: OperationalRequirements) -> Literal["runway", "hand", "vertical"]` in `src/tools/sizing/topology.py`; `SizingOutput.launch_method: Literal["runway", "hand", "vertical"] = "runway"` (old records default to runway); `HAND_LAUNCH_MAX_MTOW_KG: Final[float] = 9.07` in `topology.py`.
- Produces: `critic.static_margin_fixes(..., *, band, sm_knob)` and `critic.gate_fix_hints(..., *, band, sm_knob)`, with `sm_knob` required (no default). The two functions are today's `critic.py:170` and `:193` signatures; keep their names.

Resolution rules (validator in `requirements.py`, resolver in `topology.py`):
- `quadplane_vtol` → `vertical`; `launch_method` must be `auto` (else ValueError "a quadplane launches vertically").
- `hand` → resolves the topology to `fixed_wing` when it is `auto`; with `quadplane_vtol` it is rejected (above).
- `runway`, or `auto` on a fixed-wing, needs `runway_available=True`. Otherwise ValueError "a fixed-wing without a runway needs launch_method='hand' (or choose a VTOL quadplane)".
- `auto` topology with `runway_available=False` and launch `auto` stays a quadplane (today's behaviour).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_layouts.py
def test_hand_launch_resolves_fixed_wing():
    req = G1_ELECTRIC_FW.model_copy(update={"runway_available": False, "launch_method": "hand"})
    req = OperationalRequirements.model_validate(req.model_dump())
    assert resolved_launch_method(req) == "hand"
    assert resolve_topology(req) == "fixed_wing"

def test_fixed_wing_without_runway_must_say_how_it_launches():
    with pytest.raises(ValidationError, match="needs launch_method='hand'"):
        OperationalRequirements.model_validate({**G1_ELECTRIC_FW.model_dump(), "runway_available": False,
                                                "airframe_topology": "fixed_wing"})

def test_quadplane_rejects_hand_launch():
    with pytest.raises(ValidationError, match="launches vertically"):
        OperationalRequirements.model_validate({**G2_ELECTRIC_VTOL.model_dump(), "launch_method": "hand",
                                                "airframe_topology": "quadplane_vtol"})

def test_old_requirements_record_defaults_launch():
    data = G1_ELECTRIC_FW.model_dump(); data.pop("launch_method")
    assert OperationalRequirements.model_validate(data).launch_method == "auto"

# tests/test_sizing.py
def test_hand_launch_over_group1_fails_sizing():
    req = OperationalRequirements.model_validate({**G2_ELECTRIC_VTOL.model_dump(), "airframe_topology": "fixed_wing",
                                                  "runway_available": False, "launch_method": "hand"})
    with pytest.raises(SizingError, match=r"hand launch.*9\.07 kg"):   # G2 sizes to ~22 kg
        size_aircraft(req)

def test_hand_launched_g1_sizes_with_launch_recorded():
    req = OperationalRequirements.model_validate({**G1_ELECTRIC_FW.model_dump(), "runway_available": False,
                                                  "launch_method": "hand"})
    assert size_aircraft(req).launch_method == "hand"

# tests/test_agent.py
def test_parse_hand_launch_phrase(): ...   # "hand-launched Group 1 drone, 1 kg camera, 2 h" -> launch_method "hand",
                                           # topology fixed_wing, the echo note mentions "hand launch"
def test_parse_echo_names_layout(): ...    # a draft with airframe_layout set -> the echo note contains its LAYOUT_LABELS text
def test_critic_sm_knob_is_required():
    with pytest.raises(TypeError):
        critic.static_margin_fixes(gates, cad, stage, assumptions, band=TAILED_BAND)  # sm_knob omitted
```

Use the existing exception the sizing stage raises for a non-closing design in place of `SizingError` (find it in `mission_solver.py`).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_layouts.py tests/test_sizing.py tests/test_agent.py -k "launch or echo or sm_knob"`
Expected: FAIL (`launch_method` unknown, `resolved_launch_method` not defined, `sm_knob` still has a default)

- [ ] **Step 3: Implement**

- The field, validator and resolver as specified. `resolve_topology` (topology.py:39) returns `fixed_wing` when `launch_method == "hand"`.
- `size_aircraft` records `launch_method` and raises the Group 1 error after convergence: "hand launch needs MTOW ≤ 9.07 kg (Group 1); this design sizes to X kg".
- Parser: `RequirementsDraft.launch_method: Optional[Literal["runway", "hand"]]`. The prompt maps "hand-launched", "hand launch" and "thrown" to `hand`. A drafted `hand` with an unstated runway sets `runway_available=False`. Pass the field through as phase 1 does for `airframe_layout` (parsing.py:137).
- Add `"airframe_layout"` and `"launch_method"` to the parse echo, using `LAYOUT_LABELS` for the layout.
- Critic: drop the `= TAILED_SM_KNOB` defaults. `nodes.py:216` always passes the knob from `static_margin_basis`.
- UI: a "Launch" select (auto / runway / hand) next to the Airframe select, and a "Launch" row in `sizing_summary`.

- [ ] **Step 4: Run the tests to verify they pass, then the host suite**

Run: the Step 2 command, then `.venv/bin/python -m pytest -q -p no:cacheprovider`
Expected: PASS. The golden snapshots are unchanged: `launch_method` is a new sizing key, so add it to `NEW_SIZING_KEYS` in `tests/golden.py`.

- [ ] **Step 5: Commit** — `feat: launch method (runway/hand) for fixed-wing, required critic SM knob, layout in parse echo`

---

### Task 2: Re-baseline: explicit booms, stopped-rotor drag (golden updated once)

**Files:**
- Modify: `src/tools/sizing/layouts.py`, `src/tools/sizing/mass_model.py`, `src/tools/sizing/aero_model.py`, `src/tools/sizing/mission_solver.py`, `src/tools/aero_cad/layout.py`, `README.md`, `REQUIREMENTS.md`
- Move: `tests/data/golden/layouts_phase1/` → `tests/data/golden/layouts_phase2/` (regenerated), `tests/golden.py` (`GOLDEN_DIR`), `scripts/regenerate_layout_golden.py` (docstring)
- Test: `tests/test_sizing.py`, `tests/test_layouts.py`

**Interfaces:**
- Produces (in `src/tools/sizing/layouts.py`, moved from `aero_cad/layout.py`, which re-imports them): `TAIL_BOOM_START_FRACTION`, `TAIL_BOOM_DIAMETER_FRACTION`, `MIN_BOOM_DIAMETER_M`, `ROTOR_CLEARANCE_FRACTION`, `VTOL_BOOM_DIAMETER_FRACTION`, `VTOL_BOOM_OVERHANG_FRACTION`, `DEFAULT_WING_STATION`.
- Produces: `tail_boom_diameter_m(fuselage_diameter_m: float) -> float`; `lift_boom_geometry(rotor_diameter_m, wing_root_chord_m, wing_x_le_m, wing_x_ac_m) -> tuple[float, float]` (length, diameter of each of the 2 lift booms; the arithmetic of today's `_vtol_arrangement`, layout.py:248-254, which now calls it).
- Produces: `BoomSet(count: int, length_m: float, diameter_m: float)` and `sizing_booms(geom: Geometry, spec: LayoutSpec, rotor_diameter_m: float) -> tuple[BoomSet, ...]` in `mass_model.py`. The tail boom runs from `TAIL_BOOM_START_FRACTION × L` to the tail trailing edge, estimated at `DEFAULT_WING_STATION`. Lift booms come only when `rotor_diameter_m > 0`.
- Produces: `parasite_drag_coefficient(..., booms: Sequence[BoomSet], rotor_count: int, rotor_diameter_m: float)` replacing `is_quadplane`.

Model (my values, from the spec):
- Boom tube mass = `count × length × CF_TUBE_WALL_KG_M2 × π × diameter`, with `CF_TUBE_WALL_KG_M2 = 1.6` (1 mm carbon wall at 1600 kg/m³, the same density as `COMPOSITE_DENSITY_KG_M3`).
- Boom drag = Σ `count × Cf(Re_boom) × FF_boom × π d L / S_wing`. `FF_boom` is the fuselage form-factor formula at the boom's fineness; `Re_boom` uses the boom length.
- `BOOM_FRACTION` (4 % MTOW lift-boom lump) is removed. `airframe_mass_kg` adds the explicit boom tubes instead.
- `QUADPLANE_DRAG_PENALTY` is removed: lift-boom drag plus `rotor_count × STOPPED_ROTOR_DRAG_COEFF × rotor_diameter_m² / S_wing`. Calibrate `STOPPED_ROTOR_DRAG_COEFF` once so that G2's quadplane increment (lift-boom drag + rotor term) equals 0.008 at G2's re-baselined sizing. Write the derivation in a comment next to the constant.
- The tail boom is an addition (the pod terms never included it). Its mass goes in the structure breakdown as `"tail_boom"`, the lift booms as `"lift_booms"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_tail_boom_is_an_explicit_tube():
    s = size_aircraft(G1_ELECTRIC_FW)
    boom = s.empty_mass_breakdown["tail_boom"]
    assert boom == pytest.approx(expected_tube_mass(s), rel=1e-6)   # helper recomputes count*L*1.6*pi*d from s
    assert "lift_booms" not in s.empty_mass_breakdown

def test_lift_booms_replace_boom_fraction():
    s = size_aircraft(G2_ELECTRIC_VTOL)
    assert s.empty_mass_breakdown["lift_booms"] > 0
    assert not hasattr(mass_model, "BOOM_FRACTION")

def test_stopped_rotor_term_calibrated_on_g2():
    s = size_aircraft(G2_ELECTRIC_VTOL)
    assert quadplane_drag_increment(s) == pytest.approx(0.008, abs=1e-4)

def test_layout_lift_booms_use_shared_geometry(): ...  # layout_from_sizing(g2).vtol_booms[0].length_m == lift_boom_geometry(...)[0]
```

(Read the breakdown from wherever `SizingOutput` exposes `empty_mass_breakdown` today; add the two keys there. `quadplane_drag_increment` is a small test helper that calls `parasite_drag_coefficient` with and without the lift booms and rotors.)

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_sizing.py tests/test_layouts.py -k "boom or rotor"`
Expected: FAIL

- [ ] **Step 3: Implement** the model above. `size_geometry` keeps its signature. `mission_solver` builds the boom sets once per iteration and passes them to both the mass and the drag build-up.

- [ ] **Step 4: Regenerate the golden snapshots once, deliberately**

Run: `.venv/bin/python scripts/regenerate_layout_golden.py`, then the same in Docker (adds `vsp_geometry`). The output folder is now `layouts_phase2/`; delete `layouts_phase1/`.
Record the before/after MTOW, empty mass and CD0 of G1, G2 and G3 in the commit body and in a README/REQUIREMENTS changelog line: "2026-11-23: every layout re-baselined to explicit tail/lift-boom tubes and boom drag (+ stopped-rotor term); reference values updated once."
Expected: `tests/test_layout_golden.py` passes against the new folder, and the host suite passes. Update any test that pinned old reference numbers in the same commit, giving the reason.

- [ ] **Step 5: Commit** — `feat: re-baseline sizing to explicit boom tubes and boom drag; golden snapshots moved to layouts_phase2`

---

### Task 3: One landing-gear rule (runway rotation, VTOL level, hand launch: none)

**Files:**
- Modify: `src/tools/gear.py`, `src/tools/sizing/mass_model.py`, `src/tools/sizing/mission_solver.py`, `src/tools/fabrication/kit_mass.py`, `src/schemas/state.py`
- Test: `tests/test_gear.py`, `tests/test_sizing.py`

**Interfaces:**
- Consumes: `resolved_launch_method` (Task 1).
- Produces in `gear.py`:
  - `PropellerDisc(x_m: float, z_m: float, diameter_m: float)`: the cruise propeller in structural axes.
  - `GearInputs(launch: Literal["runway", "hand", "vertical"], fuselage_length_m, fuselage_diameter_m, mean_chord_m, wingspan_m, x_cg_m, propeller: PropellerDisc)`.
  - `landing_gear(inputs: GearInputs) -> GearGeometry`.
  - `GearGeometry` gains `leg_height_m: float` (0 for none).
  - `gear_mass_kg(gear: GearGeometry, airframe_topology: str, mtow_kg: float) -> float`.
  - `ROTATION_ANGLE_DEG: Final[float] = 12.0` and `PROP_TIP_CLEARANCE_FRACTION: Final[float] = 0.25` (of propeller radius).
- `tricycle_gear` is replaced by `landing_gear`. The JSBSim exporter switches in Task 5.
- `SizingOutput.gear_kind: Literal["none", "tricycle"] = "tricycle"`.

Rule:
- `hand` → `GearGeometry("none", (), 0.0)`, mass 0.
- `runway` and `vertical` → tricycle with the contacts placed as today. The leg height `h = -z` is the largest of:
  - today's rule, `radius × (1 + GEAR_CLEARANCE_FRACTION)`;
  - for `vertical`, the level tip clearance `z_prop - r + h ≥ PROP_TIP_CLEARANCE_FRACTION × r`;
  - for `runway`, the tip clearance at `ROTATION_ANGLE_DEG` about the main-gear contact: `(z_tip - z_main)·cosθ - (x_prop - x_main)·sinθ ≥ PROP_TIP_CLEARANCE_FRACTION × r`, solved for `h`.
- `gear_mass_kg = GEAR_FRACTION[topology] × MTOW × h / h_today`, where `h_today` is today's rule. So the tractor's mass is unchanged and taller legs scale linearly.
- Sizing builds `GearInputs` from its own estimates:
  - the wing at `DEFAULT_WING_STATION`;
  - `x_cg = x_ac_wing − band.target × chord`;
  - for `tractor_nose`, the propeller at x = 0, z = 0. The `pusher_boom_end` estimate (at the tail-boom end from Task 2's boom sets) is added in Task 6, where that layout first exists; until then any other propeller kind raises "not built yet";
  - diameter: sizing does not know the propeller before the BOM, so it uses today's estimate (keep the formula `tricycle_gear` used for the radius). The CAD stage recomputes the gear from real values in Task 6.

- [ ] **Step 1: Write the failing tests**

```python
def test_hand_launch_gear_has_no_contacts_and_zero_mass():
    gear = landing_gear(_inputs(launch="hand"))
    assert gear.kind == "none" and gear.contacts == () and gear.leg_height_m == 0.0
    assert gear_mass_kg(gear, "fixed_wing", 5.0) == 0.0

def test_tractor_runway_gear_matches_today():
    gear = landing_gear(_inputs(launch="runway", prop=PropellerDisc(0.0, 0.0, 0.279)))
    assert gear.leg_height_m == pytest.approx(0.175, abs=1e-3)    # probe: G1 leg 0.175 m
    assert gear_mass_kg(gear, "fixed_wing", 5.0) == pytest.approx(0.04 * 5.0)

def test_boom_end_pusher_runway_gear_clears_at_rotation():
    prop = PropellerDisc(x_m=1.097, z_m=0.0, diameter_m=0.279)    # probe G1 pusher
    gear = landing_gear(_inputs(launch="runway", prop=prop, x_cg=0.355))
    main = gear.contacts[1]; th = math.radians(12.0); r = 0.1395
    clearance = (prop.z_m - r - main.z_m) * math.cos(th) - (prop.x_m - main.x_m) * math.sin(th)
    assert clearance == pytest.approx(0.25 * r, abs=1e-6)
    assert gear.leg_height_m > 0.175

def test_vertical_gear_level_clearance(): ...       # tip above ground by >= 0.25 r at level attitude
def test_sizing_records_gear_kind(): ...            # hand-launched G1 -> "none", landing_gear mass 0; G1 runway -> "tricycle"
```

(`_inputs` is a test helper building `GearInputs` with G1-like numbers: L 0.878, D 0.166, chord 0.25, span 2.4.)

- [ ] **Step 2: Run to verify they fail** — `.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_gear.py tests/test_sizing.py -k gear` → FAIL

- [ ] **Step 3: Implement.** In `kit_mass.residual_gear_item` (kit_mass.py:132-136), take the gear mass from the packed design's gear (pass `GearGeometry` in) and return no item for kind `none`.

- [ ] **Step 4: Run the tests and the host suite.** The golden snapshots stay unchanged (the tractor's leg and mass are unchanged by construction; `gear_kind` goes in `NEW_SIZING_KEYS`). Expected: PASS.

- [ ] **Step 5: Commit** — `feat: one landing-gear rule (prop clearance at rotation, level for VTOL, none for hand launch) with mass from leg height`

---

### Task 4: Per-servo control surfaces, steering channel, servo allocation

**Files:**
- Modify: `src/schemas/state.py`, `src/tools/aero_cad/stability.py`, `src/tools/aero_cad/openvsp_builder.py`, `src/tools/flight_control/control_channels.py`, `src/tools/flight_control/flight_model.py`, `src/tools/flight_control/perturbation.py`, `src/tools/flight_control/param_generator.py`, `src/tools/flight_control/sitl_bridge.py`
- Test: `tests/test_flight_control.py`, `tests/test_aero_cad.py`

**Interfaces:**
- Produces `ControlSurfaceDerivatives(_Frozen)` in `state.py`: `name: str` (`"aileron"`, `"elevator"`, `"rudder"`, `"ruddervator_left"`, `"ruddervator_right"`), and `cl_per_rad`, `cl_roll_per_rad`, `cm_per_rad`, `cn_per_rad`, each a float.
- Produces `AeroStabilityOutput.control_surfaces: tuple[ControlSurfaceDerivatives, ...] = ()`. It is empty in old records.
- Produces `control_channels(aero: AeroCoefficients, surfaces: Sequence[ControlSurfaceDerivatives] = (), quad_motors: bool = False) -> tuple[ControlChannel, ...]`:
  - from `surfaces` when given, in order;
  - when empty, today's tractor triple from the aggregated derivatives (the legacy path).
- Produces `ControlChannel.steers_nose_wheel: bool = False`.
- Produces `steering_channel(channels, quad_motors) -> ControlChannel | None`:
  - the rudder channel when it exists (with `steers_nose_wheel=True`);
  - otherwise a dedicated channel: surface `"steering"`, servo function 26, servo = the first output not used by surfaces, the throttle (3) or the lift motors (5–8 only when `quad_motors`).
- Produces `FlightModel.surfaces: tuple[ControlSurfaceDerivatives, ...]`; `FlightModel.channels` derives from it.
- Produces `perturbed_model(model, case, band)`, which now scales every `FlightModel.surfaces` entry (and the aggregated fields, as today) by the case's control-power factor.

Rules:
- Servo map for named surfaces: `aileron` 1 / 4 / +1; `elevator` 2 / 19 / −1; `rudder` 4 / 21 / −1; `ruddervator_left` 2 / 79; `ruddervator_right` 4 / 80. Bridge signs for the ruddervators come in Task 7.
- `fw_servo_functions` reserves outputs 5–8 only when the model has lift motors.
- The bridge drives `fcs/steer-cmd-norm` from whichever channel has `steers_nose_wheel` (replacing `STEERING_SURFACE`, sitl_bridge.py:41).
- `derive_stability` fills `control_surfaces` for the tractor from the three existing deflection cases (aileron, elevator, rudder, with all four coefficients each). The tractor's channels, parameters and JSBSim file are unchanged.

- [ ] **Step 1: Write the failing tests**

```python
def test_legacy_aero_record_gives_tractor_channels():
    _, sizing, bom, aero, cad = load_design("g1")              # saved before phase 2: no control_surfaces
    assert aero.control_surfaces == ()
    model = build_flight_model(*load_design("g1"))
    assert [(c.surface, c.servo, c.servo_function) for c in model.channels] == [
        ("aileron", 1, 4), ("elevator", 2, 19), ("rudder", 4, 21)]

def test_channels_from_surface_records_keep_tractor_values():
    surfaces = tractor_surfaces_from(aero)   # helper: the three records derive_stability writes
    assert control_channels(coeffs, surfaces) == control_channels(coeffs)

def test_no_rudder_gets_ground_steering_servo():
    vt = (ControlSurfaceDerivatives(name="aileron", ...), ControlSurfaceDerivatives(name="ruddervator_left", ...),
          ControlSurfaceDerivatives(name="ruddervator_right", ...))
    steer = steering_channel(control_channels(coeffs, vt), quad_motors=False)
    assert (steer.surface, steer.servo, steer.servo_function) == ("steering", 5, 26)
    assert steering_channel(control_channels(coeffs, vt, quad_motors=True), quad_motors=True).servo == 9

def test_fixed_wing_does_not_reserve_quad_servos():
    params = generate_params(fixed_wing_model, sizing, bom)
    assert "SERVO5_FUNCTION" not in params or params["SERVO5_FUNCTION"] != 33

def test_perturbation_scales_every_surface(): ...   # SLUGGISH case: each surface's cm/cn/cl_roll scaled by the factor
```

The golden params and JSBSim snapshots must stay unchanged; that is the tractor regression gate.

- [ ] **Step 2: Run to verify they fail** — `.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_flight_control.py -k "channel or steering or quad_servos or perturbation"` → FAIL

- [ ] **Step 3: Implement** as specified. `_require_mapped_derivatives` (jsbsim_exporter.py:213) maps by surface name, so new names need a JSBSim channel stem each: `ruddervator_left` → `ruddervator-left`.

- [ ] **Step 4: Run the tests, the host suite, and Docker** `tests/test_aero_cad.py tests/test_layout_golden.py` (VSPAERO path) → PASS, golden unchanged

- [ ] **Step 5: Commit** — `feat: per-servo control-surface derivatives, dedicated ground-steering channel, quad servos reserved only with lift motors`

---

### Task 5: Hand-launch SITL and gear-aware JSBSim export

**Files:**
- Modify: `src/tools/flight_control/jsbsim_exporter.py`, `src/tools/flight_control/flight_model.py`, `src/tools/flight_control/flight_test.py`, `src/tools/flight_control/sitl_bridge.py`, `src/tools/flight_control/sitl_validator.py`, `src/tools/flight_control/param_generator.py`
- Test: `tests/test_flight_control.py`, `tests/test_sitl_validation.py` (Docker)

**Interfaces:**
- Consumes: `landing_gear`, `GearInputs`, `PropellerDisc` (Task 3); `SizingOutput.launch_method` (Task 1).
- Produces: `FlightModel.gear: GearGeometry` (built once in `build_flight_model` from the model's geometry and launch method), replacing the exporter's own `tricycle_gear` call (jsbsim_exporter.py:272).
- Produces: `FlightModel.launch: Literal["runway", "hand", "vertical"]`.
- Produces: `HAND_LAUNCH_SPEED_MS: Final[float] = 10.0`, `HAND_LAUNCH_HEIGHT_M: Final[float] = 2.0`, `HAND_LAUNCH_PITCH_DEG: Final[float] = 10.0` in `flight_test.py`.

Rules:
- A gear of kind `none` exports two belly `STRUCTURE` contacts: under the CG and at the pod's 80 % station, at z = −D/2, non-steerable. JSBSim then has a ground model and nothing crashes on empty contacts.
- Hand launch in SITL:
  - the bridge's initial state is airborne: `HAND_LAUNCH_HEIGHT_M` above the ground, `HAND_LAUNCH_SPEED_MS` along the body x axis, `HAND_LAUNCH_PITCH_DEG` nose up;
  - the pilot arms in FBWA at full throttle and holds the take-off pitch stick;
  - the test passes when it climbs to `TAKEOFF_ALT_FRACTION × TKOFF_ALT` without touching the ground;
  - `TKOFF_THR_MINACC` and the other runway-only params are not written for hand launch.
- `_take_off_fixed_wing` branches on `model.launch` (runway roll as today, or the hand launch above).

- [ ] **Step 1: Write the failing tests**

```python
def test_export_without_gear_writes_belly_contacts(tmp_path):
    model = replace(g1_model, gear=GearGeometry("none", (), 0.0), launch="hand")
    xml = export_jsbsim(model, tmp_path).xml_path.read_text()
    assert xml.count('type="STRUCTURE"') == 2 and 'type="BOGEY"' not in xml

def test_hand_launch_params_omit_runway_takeoff(): ...   # generate_params(hand model) has no TKOFF_ROTATE_SPD

# tests/test_sitl_validation.py (Docker; skip when ArduPilot is absent, like the existing SITL tests)
def test_hand_launched_g1_passes_sitl(): ...             # G1 with launch "hand" (Task 1's requirements) -> SITL passes
```

- [ ] **Step 2: Run to verify they fail** — host: `-k "belly or hand_launch_params"` → FAIL

- [ ] **Step 3: Implement** as specified.

- [ ] **Step 4: Run** the host suite (golden unchanged: the runway tractor's gear contacts are identical), then in Docker: `python -m pytest -q -p no:cacheprovider tests/test_sitl_validation.py` → PASS

- [ ] **Step 5: Commit** — `feat: hand-launch SITL take-off and gear-aware JSBSim export (belly contacts without gear)`

---

### Task 6: V-tail pusher sizing, layout, propeller station and clearance checks

**Files:**
- Modify: `src/tools/sizing/layouts.py`, `src/tools/sizing/mass_model.py`, `src/tools/sizing/mission_solver.py`, `src/schemas/assumptions.py`, `src/schemas/changes.py`, `src/tools/aero_cad/layout.py`, `src/tools/aero_cad/components.py`, `src/tools/aero_cad/cad_packer.py`, `src/tools/aero_cad/build.py`
- Create: `src/tools/aero_cad/clearance.py` (propeller–tail, propeller–ground and belly-line checks)
- Test: `tests/test_layouts.py`, `tests/test_aero_cad.py`

**Interfaces:**
- Produces:
  - `_SPECS["single_boom_vtail_pusher"] = LayoutSpec("v", "centre", "pusher_boom_end", "straight", "tail_volume_h", TAILED_BAND)`. Not yet in `ENABLED_LAYOUTS` (Task 9).
  - `VTAIL_PAIR_ASPECT_RATIO: Final[float] = 4.0` (tip-to-tip span² / total area, inside the spec's 3–6 band).
  - `PUSHER_TAIL_GAP_FRACTION: Final[float] = 0.10` (axial gap, tail trailing edge to propeller disc, as a fraction of propeller diameter).
- Produces: `size_geometry(..., spec)` returning one `TailSurface(name="vtail_pair", control="ruddervator", area_m2=S_h+S_v, span_m=sqrt(4.0·S), taper_ratio=HTAIL_TAPER_RATIO, dihedral_deg=degrees(atan(sqrt(S_v/S_h))), arm_m=arm, is_symmetric=True)` for `tail == "v"`. `tail_arm_m` keeps today's centre-boom rule.
- Produces: `DesignAssumptions.pusher_efficiency_factor: float = Field(0.95, ge=0.80, le=1.00)`. `mission_solver` multiplies the drivetrain efficiency by it when `spec.propeller != "tractor_nose"`.
- Produces: `PropellerStation.diameter_m: float = 0.0` and `layout_from_sizing(..., propeller_diameter_m: float = 0.0)`. `build.py` passes `bom.cruise_propeller.diameter_m`.
- For `pusher_boom_end`:
  - the tail boom ends at `tail_te + gap`, where `gap = PUSHER_TAIL_GAP_FRACTION × diameter`;
  - the hub sits at boom end + motor length;
  - the station has `is_pusher=True` and `z = boom.z_m`.
- Produces: `components.cruise_motor(bom: SelectedBOM, layout: AircraftLayout) -> PlacedComponent`. The motor sits ahead of its station (the nose bulkhead for a tractor, the boom end for a pusher).
- Produces in `clearance.py`:
  - `PropClearance(tail_gap_m: float, ground_m: Optional[float], belly_m: Optional[float], needs_folding_prop: bool)`;
  - `prop_clearance(layout: AircraftLayout, gear: GearGeometry, x_cg_m: float, launch: str) -> PropClearance`. It uses `ROTATION_ANGLE_DEG` for runway, level for vertical, and `BELLY_LANDING_PITCH_DEG: Final[float] = 5.0` for the belly line with the hand-launch gear `none`.
  - `needs_folding_prop` is True for any hand-launched pusher, or when the belly clearance is below `PROP_TIP_CLEARANCE_FRACTION × r`.
- Packing: `pack_layout` builds the real gear (`landing_gear` from the packed CG and the layout's propeller) and evaluates `prop_clearance`. A negative `tail_gap_m` or `ground_m` adds 1 to `collision_count`, so the critic sees a collision-class failure. `PackingResult` gains `gear: GearGeometry` and `clearance: PropClearance`.

- [ ] **Step 1: Write the failing tests**

```python
def test_vtail_area_and_dihedral():
    g = size_geometry(5.0, 0.6, 9.0, 0.45, 0.035, 1.0, layout_spec("single_boom_vtail_pusher"))
    (t,) = g.tail_surfaces
    s_h = 0.45 * 0.6 * g.mean_chord_m / g.tail_arm_m; s_v = 0.035 * 0.6 * g.wingspan_m / g.tail_arm_m
    assert t.control == "ruddervator" and t.area_m2 == pytest.approx(s_h + s_v)
    assert t.dihedral_deg == pytest.approx(math.degrees(math.atan(math.sqrt(s_v / s_h))))
    assert effective_tail_areas(g.tail_surfaces) == pytest.approx((s_h, s_v))

def test_pusher_efficiency_applies_only_to_pushers(): ...  # vtail pusher eta = 0.95 x tractor eta at equal assumptions

def test_boom_end_station_behind_vtail():
    layout = layout_from_sizing(vtail_g1_sizing, propeller_diameter_m=0.279)
    te = max(t.planform.x_le_root_m + t.planform.root_chord_m for t in layout.tail_surfaces)
    assert layout.propeller.is_pusher and layout.tail_boom.x_end_m == pytest.approx(te + 0.0279)

def test_prop_disc_into_tail_is_a_collision(): ...         # force PUSHER_TAIL_GAP_FRACTION negative via monkeypatch -> collision_count >= 1
def test_prop_ground_strike_is_reported(): ...             # runway gear legs pinned short (monkeypatch landing_gear) -> clearance.ground_m < 0, collision
def test_hand_launched_pusher_needs_folding_prop(): ...    # launch "hand", pusher -> needs_folding_prop True
def test_vtail_pusher_g1_balances_on_host(): ...           # balance_design with _synthetic_aero: is_balanced, 0 interferences
```

- [ ] **Step 2: Run to verify they fail** — `.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_layouts.py tests/test_aero_cad.py -k "vtail or pusher or prop_"` → FAIL

- [ ] **Step 3: Implement** as specified. `tail_arm_m` and `_tail_booms` take the spec. `_propeller_station` handles `tractor_nose` and `pusher_boom_end` and raises for `pusher_pod` ("not built yet").

- [ ] **Step 4: Run the tests and the host suite** → PASS, golden unchanged (the tractor station gains only `diameter_m`, which `layout_view` does not read)

- [ ] **Step 5: Commit** — `feat: V-tail pusher sizing, boom-end propeller station, pusher efficiency, propeller clearance checks`

---

### Task 7: V-tail in VSPAERO, flight control and SITL

**Files:**
- Modify: `src/tools/aero_cad/openvsp_builder.py`, `src/tools/aero_cad/stability.py`, `src/tools/flight_control/flight_model.py`, `src/tools/flight_control/control_channels.py`, `src/tools/flight_control/param_generator.py`, `src/tools/flight_control/jsbsim_exporter.py`
- Test: `tests/test_aero_cad.py` (Docker), `tests/test_flight_control.py`, `tests/test_sitl_validation.py` (Docker)

**Interfaces:**
- Consumes: `ControlSurfaceDerivatives`, `control_channels`, `steering_channel` (Task 4); the V-tail `TailSurface` and `PropellerStation` (Task 6).
- Produces:
  - VSPAERO geom `"VTailPair"`: one mirrored wing with the panel dihedral and the ruddervator control from `TAIL_CONTROL_ETA` / `TAIL_CONTROL_CHORD_FRACTION`.
  - `run_vspaero` deflects each panel alone (mirror gains `(1, 0)` and `(0, 1)`) and returns `ruddervator_left` / `ruddervator_right` cases for `tail == "v"`.
  - `derive_stability` writes both records. It also writes the aggregated gate values by linearity: `cm_delta_e_per_rad` is the symmetric combination (the two panels' cm summed with the signs that make both deflect trailing-edge-up together); `cn_delta_r_per_rad` is the antisymmetric combination.
- `_check_control_conventions` (stability.py:126) gains the V-tail rule:
  - symmetric trailing-edge-up gives nose-up Cm;
  - the antisymmetric combination gives yaw of the expected sign;
  - the roll coupling is recorded, not gated.
- `flight_model` damping uses `effective_tail_areas(sizing.tail_surfaces)` for `v_h` and `v_v` (flight_model.py:166-175 read the legacy fields, which are already effective areas; keep the values and read them through the surfaces).
- ArduPilot:
  - `SERVO2_FUNCTION 79`, `SERVO4_FUNCTION 80`, and `MIXING_GAIN` written explicitly as 0.5 (the default);
  - the pitch and yaw rate gains use the effective mixed control power `MIXING_GAIN × (cm_L + cm_R)` and `MIXING_GAIN × |cn_L − cn_R|`;
  - `SERVOn_REVERSED` for each panel is chosen from the sign of that panel's `cm_per_rad`, so that an ArduPilot nose-up elevator demand gives a nose-up moment through the bridge sign.
- JSBSim: cruise thrust acts at `layout.propeller` (x, z) and points forward for pushers. Each ruddervator is its own channel with its own coefficients. The steering servo drives the nose wheel.

- [ ] **Step 1: Write the failing tests**

```python
# host
def test_vtail_channels_and_servos():
    model = build_flight_model(*vtail_g1_design)   # saved fixture written by this task (Docker run, committed JSON)
    assert [(c.surface, c.servo, c.servo_function) for c in model.channels][:3] == [
        ("aileron", 1, 4), ("ruddervator_left", 2, 79), ("ruddervator_right", 4, 80)]
    params = generate_params(model, *vtail_g1_design[1:3])
    assert params["MIXING_GAIN"] == 0.5 and params["SERVO5_FUNCTION"] == 26

def test_pusher_thrust_at_propeller_station(tmp_path): ...  # exported XML thrust location x == layout propeller x

# Docker
def test_vtail_panel_derivatives_from_vspaero(): ...         # both panels: symmetric sum gives cm < 0 per rad TE-up convention;
                                                             # antisymmetric gives |cn| > 0; left/right roll coupling opposite signs
def test_reversed_ruddervator_fails_sitl(): ...              # nominal V-tail G1 passes _attempt; flipping SERVO2_REVERSED fails it
                                                             # (same pairing pattern as test_reversed_elevator_fails_sitl)
```

Save the V-tail G1 design fixture under `tests/data/flight_control/g1_vtail/` the way `load_design` reads `g1` (requirements, sizing, bom, aero, cad JSON from one Docker pipeline run).

- [ ] **Step 2: Run to verify they fail** — host `-k "vtail or pusher_thrust"` → FAIL; Docker `tests/test_aero_cad.py -k vtail` → FAIL

- [ ] **Step 3: Implement** as specified.

- [ ] **Step 4: Run** the host suite (golden unchanged), then Docker `tests/test_aero_cad.py tests/test_flight_control.py tests/test_sitl_validation.py` → PASS

- [ ] **Step 5: Commit** — `feat: V-tail ruddervators in VSPAERO (per panel), ArduPilot V-tail mixing, pusher thrust station, reversed-ruddervator SITL test`

---

### Task 8: V-tail and boom-end motor kit

**Files:**
- Modify: `src/tools/fabrication/build_kit.py`, `src/tools/fabrication/wing_parts.py`, `src/tools/fabrication/tubes.py`, `src/tools/fabrication/boom_parts.py`, `src/tools/fabrication/fittings.py`, `src/tools/fabrication/kit_labels.py` (build-sheet notes)
- Test: `tests/test_fabrication_kit.py`, `tests/test_fabrication_parts.py`, `tests/test_fabrication_structure.py`

**Interfaces:**
- Consumes: `layout.tail_surfaces`, `layout.propeller`, `PackingResult.clearance` (Task 6).
- Produces: the kit iterates `layout.tail_surfaces` instead of `layout.htail` / `layout.vtail`. Each surface gets a panel frame: root point, span direction `(0, cos Γ, sin Γ)` for a mirrored surface and `(0, 0, 1)` for a fin. Tail spars, ribs, edge strips and film are placed in that frame.
  - Functions renamed: `_tail_spars` (build_kit.py:234), `_edge_strips` (:272), `_film` calls (:331), `tail_spar_loads` / `tail_spar_tubes` (wing_parts.py:402, :426). They return per-surface results keyed by surface name.
  - The tractor's parts, positions and labels stay identical (golden kit snapshot unchanged).
- New parts for `tail == "v"`:
  - `VR-1` V-tail root block, ply, clamped on the boom, its two faces at ±Γ carrying the panel spar sockets;
  - one rib set and one spar tube per panel (`VL-`/`VR-` rib prefixes, following the H-tail's `H{n}` scheme).
- New parts for `pusher_boom_end`: `MP-B` motor plate and `MC-B` boom clamp at the boom end. The nose `MP-C` plate is not built.
  - `cruise_motor_plate` reads `layout.propeller.is_pusher` and returns the boom-end parts.
  - It still raises `KitUnavailable` when the motor reaches back into its plate.
- Load checks:
  - `tail_boom_loads` (tubes.py:318) uses each surface's projected loads: the vertical component Σ L·sin Γ sets the boom side load, the horizontal component Σ L·cos Γ the pitch load;
  - add the boom-end motor mass × the load factor at the boom tip to the boom bending.
- Build-sheet notes: "Motor wiring runs along the tail boom to the boom-end motor" (boom-end pusher) and "Folding propeller required (belly landing)" when `clearance.needs_folding_prop`.

- [ ] **Step 1: Write the failing tests**

```python
def test_tractor_kit_unchanged_after_generalisation(): ...  # golden kit snapshot (tests/test_layout_golden.py) still passes: run it here
def test_vtail_kit_has_two_panels_at_dihedral():
    kit = _vtail_kit("g1")
    spars = [t for t in kit.tubes if t.role == "tail_spar"]
    assert len(spars) == 2
    gamma = vtail_surface.dihedral_deg
    for t in spars:
        assert math.degrees(math.atan2(abs(t.axis[2]), abs(t.axis[1]))) == pytest.approx(gamma, abs=0.5)
    assert any(p.label == "VR-1" for p in kit.parts)

def test_boom_end_motor_plate_replaces_nose_plate():
    labels = {p.label for p in _vtail_kit("g1").parts}
    assert "MP-B" in labels and "MC-B" in labels and "MP-C" not in labels

def test_vtail_boom_check_includes_motor_mass(): ...        # tail_boom_loads moment grows by m_motor*g*n*arm vs. the same boom without the motor
def test_folding_prop_note_for_hand_launched_pusher(): ...  # build sheet text contains "Folding propeller required"
```

- [ ] **Step 2: Run to verify they fail** — `.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_fabrication_kit.py tests/test_fabrication_parts.py tests/test_fabrication_structure.py -k "vtail or boom_end or folding"` → FAIL

- [ ] **Step 3: Implement** as specified. If a V-tail check cannot be met (bed fit, stock tube, bending), the kit is unavailable with the reason, as today.

- [ ] **Step 4: Run** the fabrication tests and the host suite; golden kit unchanged → PASS

- [ ] **Step 5: Commit** — `feat: V-tail kit (root block, per-panel spars and ribs) and boom-end motor plate/clamp; folding-prop and wiring notes`

---

### Task 9: Enable the V-tail pusher, reference designs, docs, phase gate

**Files:**
- Modify: `src/schemas/requirements.py` (`ENABLED_LAYOUTS`), `src/agent/parsing.py` (layout phrases), `src/ui/app.py` (Layout select, if phase 1 did not add it), `src/ui/examples.py`, `tests/reference_designs.py`, `README.md`, `REQUIREMENTS.md`
- Test: `tests/test_layouts.py`, `tests/test_agent.py`, `tests/test_aero_cad.py`, `tests/test_sitl_validation.py` (Docker)

**Interfaces:**
- Produces: `ENABLED_LAYOUTS` adds `("single_boom_vtail_pusher", "fixed_wing")` and `("single_boom_vtail_pusher", "quadplane_vtol")`.
- Produces: `G1_VTAIL_PUSHER_FW` and `G2_VTAIL_PUSHER_VTOL` in `src/ui/examples.py` (G1/G2 with `airframe_layout="single_boom_vtail_pusher"`), registered in `EXAMPLE_MISSIONS` and `tests.reference_designs.CASES` as `"g1_vtail"` and `"g2_vtail"`. Add `G1_HAND_LAUNCH` (G1, `runway_available=False`, `launch_method="hand"`) as `"g1_hand"`.
- Parser: "V-tail" or "V tail" with "pusher" → `single_boom_vtail_pusher`. Any other layout phrase still drafts its id and gets the "not available yet" message from validation.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("case", ["g1_vtail", "g2_vtail"])
def test_vtail_pusher_closes_sizing_and_cad(case):   # host: size_aircraft + select_bom + balance_design(_synthetic_aero) + kit
    design = _close(case)
    assert design.packing.is_balanced and design.packing.interference_count == 0
    assert design.packing.kit is not None or "No laser kit" in (design.packing.kit_warning or "")  # G2 is Group 2: kit must build
    assert design.packing.clearance.ground_m is None or design.packing.clearance.ground_m >= 0

def test_hand_launched_g1_closes(): ...               # g1_hand: gear "none", kit built, folding-prop note only if belly clearance short
def test_vtail_pusher_validation_enabled(): ...       # both combos validate; twin-boom layouts still raise "not available yet"
def test_chat_switches_to_vtail_pusher(): ...         # design chat "make it a V-tail pusher" on a G1 design -> requirements layout set,
                                                      # re-run sizing has one ruddervator TailSurface
```

- [ ] **Step 2: Run to verify they fail** → FAIL (validation still rejects the layout)

- [ ] **Step 3: Implement.** README/REQUIREMENTS: describe the layout, the launch method, the gear rule, the re-baseline changelog line (Task 2) and the deferred carry-overs (y offset → phase 3, sweep levers → phase 5).

- [ ] **Step 4: Phase gate (each one must pass)**

Run:
1. `.venv/bin/python -m pytest -q -p no:cacheprovider` → all pass
2. Docker full suite: `docker run --rm -v "$PWD":/app -w /app isr-forge:latest python -m pytest -q -p no:cacheprovider` → all pass
3. Docker full pipeline with SITL, one run per new combination plus hand launch: `python -m src.main` on the G1 V-tail pusher, G2 V-tail pusher and G1 hand-launch example prompts (or the pipeline entry the phase-1 end-to-end run used) → all five gates pass, SITL passes, kit built. Record the MTOW, static margin, wing station and propeller clearances in the final report.

- [ ] **Step 5: Commit** — `feat: enable single-boom V-tail pusher (fixed-wing and quadplane) with reference designs and hand-launch example`
