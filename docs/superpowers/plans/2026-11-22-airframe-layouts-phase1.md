# Airframe Layouts — Phase 1 (Refactor) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the airframe layout an explicit, validated input read by every stage, with general tail surfaces,
booms, a propeller station, a shared gear description and per-surface control channels, while every output of
today's reference designs stays identical.

**Architecture:** A `layouts.py` module owns `LayoutSpec` (tail kind, booms, propeller, wing, static-margin band,
tail-arm rule) for each built layout; only `pod_boom_tractor` exists in phase 1. Sizing emits a tuple of
`TailSurface`; `AircraftLayout` carries `tail_surfaces`, `tail_booms` and a `PropellerStation`; flight control
drives JSBSim and the SITL bridge from one `ControlChannel` table. A golden snapshot taken *before* any change pins
"unchanged".

**Tech Stack:** Python 3.11, pydantic v2, AeroSandbox, OpenVSP (Docker only), JSBSim, ArduPilot SITL, pytest.

**Spec:** `docs/superpowers/specs/2026-11-22-airframe-layouts-design.md` (phase 1 = "Refactor" under *Phases*).

## Global Constraints

- Phase 1 is a pure refactor: sizing output, CAD/VSPAERO inputs, kit parts, ArduPilot parameters and the JSBSim
  model of the reference designs are unchanged. The golden test (Task 1) is the gate; snapshots are never
  regenerated in phase 1.
- Layout ids: `pod_boom_tractor` (default), `single_boom_vtail_pusher`, `twin_boom_htail_pusher`,
  `twin_boom_inverted_v_pusher`, `flying_wing`. Orthogonal to `airframe_topology`.
- Phase 1 enables only (`pod_boom_tractor`, `fixed_wing`) and (`pod_boom_tractor`, `quadplane_vtol`); validation
  rejects every other combination with a clear message.
- Subsystems branch on `LayoutSpec` properties, never on the layout id.
- Static-margin band for tailed layouts: 8–14 %, target 11 % (today's `SM_MIN`, `SM_MAX`, `SM_TARGET`).
- Legacy `horizontal_tail_area_m2` / `vertical_tail_area_m2` are effective areas, Σ S·cos²Γ and Σ S·sin²Γ over the
  tail surfaces; `tail_arm_m` is the tail arm. Records without `tail_surfaces` / `airframe_layout` still load.
- Repo rules: TDD, functions < 50 lines, files < 800 lines, no magic numbers, frozen/immutable data, type hints, SI.
- Commit messages end with `Co-Authored-By: WOZCODE <contact@withwoz.com>` and
  `Claude-Session: https://claude.ai/code/session_01RJfaxdHYf7KowUaPPCoa8G`.
- Test commands. Host: `.venv/bin/python -m pytest -q -p no:cacheprovider <paths>`. Docker (OpenVSP, JSBSim,
  ArduPilot): `docker run --rm -v "$PWD":/app -w /app isr-forge:latest python -m pytest -q -p no:cacheprovider <paths>`.

## Review Focus

1. A design record stored before phase 1 (no `airframe_layout`, no `tail_surfaces`) must still load in the UI, the
   archive and design chat — pinned by `test_pre_layout_sizing_record_loads` (Task 3).
2. `airframe_topology="auto"` with a layout built for only one topology must be rejected up front, not after
   sizing picks a topology — pinned by `test_auto_topology_needs_layout_enabled_for_both` (Task 2).
3. A user asking for "a flying wing" gets a clear "not available yet" message, not a silent tractor — pinned by
   `test_parser_maps_flying_wing_and_validation_explains` (Task 2).
4. Editing a legacy tail-area field without the surfaces must fail validation rather than size one tail and build
   another — pinned by `test_effective_areas_must_match_surfaces` (Task 3).
5. A servo reversal error must make the SITL check fail (the circularity the spec removes) — pinned by
   `test_reversed_elevator_fails_sitl` (Task 7, Docker).

---

### Task 1: Golden snapshot of today's outputs

**Files:**
- Create: `tests/golden.py`, `tests/test_layout_golden.py`, `scripts/regenerate_layout_golden.py`
- Create: `tests/data/golden/layouts_phase1/<case>/<key>.json`

**Interfaces:**
- Produces in `tests/golden.py`:
  - `canonical(obj: object) -> str`: JSON, sorted keys, floats rounded to 12 significant digits, dataclasses via
    `dataclasses.asdict`, pydantic via `model_dump(mode="json")`, tuples as lists.
  - `NEW_SIZING_KEYS: frozenset[str] = frozenset({"airframe_layout", "tail_surfaces"})`, excluded from the sizing
    snapshot (Task 3 adds them; their values are pinned by unit tests instead).
  - `layout_view(layout) -> dict`: attribute access, not `asdict`, so the snapshot survives Task 4's field
    reshuffle: `fuselage_length_m, fuselage_diameter_m, wing, htail, vtail, tail_boom, vtol_booms,
    rotor_positions, rotor_diameter_m, airfoil, wing_station`.
  - `snapshot(case: str) -> dict[str, str]` with keys:
    - `sizing`: `size_aircraft(CASES[case])` dumped with `exclude=NEW_SIZING_KEYS` (g1, g2, g3);
    - `layout`: `layout_view(layout_from_sizing(sizing))` (g1, g2, g3);
    - `params`: `generate_params(build_flight_model(*load_design(case)), sizing, bom)` (g1, g2);
    - `jsbsim`: the aircraft XML text written by `export_jsbsim` into a tmp dir (g1, g2);
    - `kit`: for every part of the kit built as `tests/test_fabrication_kit._build` builds it, its name, role and
      outline vertices (g1, g2);
    - `vsp_geometry` (Docker only, `pytest.importorskip("openvsp")`): for each component of
      `build_model(layout, tmp)`, its name, type and span/chord/sweep/dihedral/x/z parm values (g1, g2).

- [ ] **Step 1:** Write `scripts/regenerate_layout_golden.py`, which writes each `snapshot(case)[key]` to its file.
  Run it on `d125b73` (before any phase 1 code change), on the host and in Docker (for `vsp_geometry`).
  Expected: files for every case/key listed above.
- [ ] **Step 2:** Write `test_reference_outputs_unchanged[case-key]`, parametrised over the stored files, asserting
  `snapshot(case)[key] == stored text`. Run host and Docker. Expected: PASS.
- [ ] **Step 3:** Commit `test: golden snapshot of reference outputs before the layout refactor`.

From here on a changed snapshot means the code is wrong: fix the code, never the snapshot.

### Task 2: Layout ids, `LayoutSpec` and validation

**Files:**
- Modify: `src/schemas/requirements.py`, `src/agent/agent_schemas.py`, `src/agent/parsing.py` (mapping and
  `ECHO_FIELDS`), `src/schemas/changes.py` (`RequirementChanges` field and label), `src/tools/sizing/mass_model.py`
  (`TAIL_ARM_FRACTION` moves out)
- Create: `src/tools/sizing/layouts.py`
- Test: `tests/test_layouts.py`
- Docs: `REQUIREMENTS.md` (schema field)

**Interfaces:**
- Produces in `src/schemas/requirements.py` (the ids live in schemas because `topology.py` imports requirements):
  - `AirframeLayout = Literal["pod_boom_tractor", "single_boom_vtail_pusher", "twin_boom_htail_pusher",
    "twin_boom_inverted_v_pusher", "flying_wing"]`
  - `LAYOUT_LABELS: Final[dict[str, str]]`, with `"pod_boom_tractor": "Pod-and-boom tractor"`
  - `ENABLED_LAYOUTS: Final[frozenset[tuple[str, str]]] = frozenset({("pod_boom_tractor", "fixed_wing"),
    ("pod_boom_tractor", "quadplane_vtol")})`
  - `OperationalRequirements.airframe_layout: AirframeLayout = "pod_boom_tractor"`
- Produces in `src/tools/sizing/layouts.py`:
  ```python
  @dataclass(frozen=True)
  class StaticMarginBand:
      minimum: float
      maximum: float
      target: float

  @dataclass(frozen=True)
  class LayoutSpec:
      tail: Literal["conventional", "v", "inverted_v", "twin_fin_h", "none"]
      booms: Literal["centre", "twin", "none"]
      propeller: Literal["tractor_nose", "pusher_pod", "pusher_boom_end"]
      wing: Literal["straight", "swept_reflex"]
      static_margin_knob: Literal["tail_volume_h", "wing_sweep_deg"]
      static_margin: StaticMarginBand

  TAILED_BAND: Final[StaticMarginBand] = StaticMarginBand(0.08, 0.14, 0.11)
  TAIL_ARM_FRACTION: Final[float] = 0.6   # of fuselage length, centre boom (moved from mass_model.py)
  def layout_spec(layout: str) -> LayoutSpec          # ValueError for a layout not built yet
  def tail_arm_m(spec: LayoutSpec, fuselage_length_m: float) -> float
  ```
  The only spec in phase 1 is `pod_boom_tractor = LayoutSpec("conventional", "centre", "tractor_nose", "straight",
  "tail_volume_h", TAILED_BAND)`. `tail_arm_m` returns `TAIL_ARM_FRACTION * fuselage_length_m` for `booms ==
  "centre"` and raises `ValueError` otherwise (later phases add their rules).

- [ ] **Step 1: Write the failing tests** in `tests/test_layouts.py`:
  - `test_default_layout_is_the_tractor`: `G1_ELECTRIC_FW.airframe_layout == "pod_boom_tractor"`.
  - `test_unbuilt_layout_is_rejected_with_reason` (`model_copy` skips validation, so rebuild from a dump):
    `OperationalRequirements(**{**G1_ELECTRIC_FW.model_dump(), "airframe_layout": "flying_wing"})` raises
    `ValidationError` whose text contains `"flying_wing"` and `"not available yet"`.
  - `test_auto_topology_needs_layout_enabled_for_both`: monkeypatch `ENABLED_LAYOUTS` to
    `{("pod_boom_tractor", "fixed_wing")}`; `airframe_topology="auto"` is rejected and `"fixed_wing"` accepted.
  - `test_tractor_spec_and_tail_arm`: `layout_spec("pod_boom_tractor").static_margin == StaticMarginBand(0.08, 0.14,
    0.11)`; `tail_arm_m(spec, 1.5) == pytest.approx(0.9)`; `layout_spec("flying_wing")` raises `ValueError`.
  - `test_parser_maps_flying_wing_and_validation_explains`: a draft with `airframe_layout="flying_wing"` is mapped
    into the requirement fields, and building the requirements raises the "not available yet" message.
  - `test_design_chat_change_carries_layout`: applying `RequirementChanges(airframe_layout="pod_boom_tractor")`
    yields requirements with that layout.
- [ ] **Step 2:** Run `tests/test_layouts.py`. Expected: FAIL (field and module missing).
- [ ] **Step 3:** Implement. Validator message: `f"airframe layout {layout!r} is not available yet for {topology};
  available: {', '.join(sorted(...))}"`. With `topology == "auto"`, both topologies must be enabled.
- [ ] **Step 4:** Run `tests/test_layouts.py tests/test_schemas.py tests/test_agent.py tests/test_design_chat.py
  tests/test_sizing.py tests/test_layout_golden.py`. Expected: PASS.
- [ ] **Step 5:** Commit `feat: airframe_layout requirement and LayoutSpec (tractor only)`.

### Task 3: Sizing emits tail surfaces

**Files:**
- Modify: `src/schemas/state.py`, `src/tools/sizing/mass_model.py` (`Geometry`, `size_geometry`),
  `src/tools/sizing/mission_solver.py`, `src/tools/sizing/layouts.py` (tail AR/taper constants arrive),
  `src/tools/aero_cad/layout.py` (imports them), `src/ui/presenters.py` and `src/agent/report.py` (Layout row)
- Test: `tests/test_layouts.py`

**Interfaces:**
- Consumes: `layout_spec`, `tail_arm_m`, `LAYOUT_LABELS` (Task 2).
- Produces in `src/schemas/state.py`:
  ```python
  class TailSurface(_Frozen):
      name: str                                         # "htail", "vtail"
      control: Literal["elevator", "rudder", "ruddervator"]
      area_m2: Positive                                 # whole surface (both panels if symmetric)
      span_m: Positive                                  # tip to tip if symmetric, else fin height
      taper_ratio: Positive
      dihedral_deg: float                               # 0 horizontal, 90 vertical fin
      arm_m: Positive                                   # wing AC to surface AC
      is_symmetric: bool

  def effective_tail_areas(surfaces: Sequence[TailSurface]) -> tuple[float, float]   # (Σ S cos²Γ, Σ S sin²Γ)
  ```
  `SizingOutput` gains `airframe_layout: AirframeLayout = "pod_boom_tractor"` and
  `tail_surfaces: tuple[TailSurface, ...] = ()`; `horizontal_tail_area_m2`, `vertical_tail_area_m2` and
  `tail_arm_m` become `NonNegative`. A model validator checks, when `tail_surfaces` is non-empty, that the legacy
  areas equal `effective_tail_areas` (rel 1e-9).
- Produces: `Geometry.tail_surfaces`; `size_geometry(..., spec: LayoutSpec)` computes the tail arm with
  `tail_arm_m(spec, fus_length)`.
- `HTAIL_ASPECT_RATIO = 4.0`, `HTAIL_TAPER_RATIO = 0.6`, `VTAIL_ASPECT_RATIO = 1.5`, `VTAIL_TAPER_RATIO = 0.6` move
  to `layouts.py`. Span = √(AR·S) for the H-tail and height = √(AR·S) for the fin, the same numbers `layout.py`
  computes today.

- [ ] **Step 1: Write the failing tests:**
  - `test_tractor_sizing_has_htail_and_fin`: for g1, names `["htail", "vtail"]`, dihedrals `(0.0, 90.0)`, controls
    `("elevator", "rudder")`, `sizing.airframe_layout == "pod_boom_tractor"`, and
    `effective_tail_areas(sizing.tail_surfaces) == pytest.approx((sizing.horizontal_tail_area_m2,
    sizing.vertical_tail_area_m2), rel=1e-12)`.
  - `test_effective_areas_must_match_surfaces`: `SizingOutput.model_validate({**sizing.model_dump(),
    "horizontal_tail_area_m2": 9.9})` raises `ValidationError`.
  - `test_pre_layout_sizing_record_loads`: `tests/data/flight_control/g1/sizing.json` validates with
    `airframe_layout == "pod_boom_tractor"` and `tail_surfaces == ()`.
  - `test_sizing_summary_shows_layout`: the presenter's sizing rows include `("Layout", "Pod-and-boom tractor")`,
    and the Markdown report's sizing table has a `Layout` row.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `tests/test_layouts.py tests/test_sizing.py tests/test_ui.py tests/test_report_pdf.py
  tests/test_agent.py tests/test_layout_golden.py`. Expected: PASS (sizing snapshot unchanged with
  `NEW_SIZING_KEYS` excluded).
- [ ] **Step 5:** Commit `refactor: sizing emits tail surfaces; tail arm from the layout`.

### Task 4: General geometry in `AircraftLayout` and its readers

**Files:**
- Modify: `src/tools/aero_cad/layout.py`, `src/tools/aero_cad/openvsp_builder.py`, `src/tools/aero_cad/components.py`
- Test: `tests/test_layouts.py`, existing aero_cad and fabrication tests

**Interfaces:**
- Consumes: `SizingOutput.tail_surfaces` (Task 3), `layout_spec` (Task 2).
- Produces in `layout.py`:
  ```python
  @dataclass(frozen=True)
  class PlacedTailSurface:
      name: str
      control: str
      planform: SurfacePlanform

  @dataclass(frozen=True)
  class PropellerStation:
      x_m: float
      y_m: float
      z_m: float
      diameter_m: float
      is_pusher: bool
  ```
  `AircraftLayout` replaces the `htail`, `vtail` and `tail_boom` fields with `tail_surfaces:
  tuple[PlacedTailSurface, ...]`, `tail_booms: tuple[Boom, ...]` and `propeller: PropellerStation`, and keeps
  read-only properties `htail` / `vtail` (the planform of the surface with that name; `ValueError` if there is
  none) and `tail_boom` (the single centre boom, or None). `SurfacePlanform` gains `sweep_deg: float = 0.0`.
- Switched to the general fields in this task: `openvsp_builder.build_model` (one VSP wing per tail surface,
  vertical when its dihedral is 90°; one pod per tail boom) and `components.py` (obstacles from every tail boom).
  The kit modules keep reading `htail` / `vtail` / `tail_boom`. *Ruling: the kit is generalised in the phase that
  adds the first non-conventional tail (phase 2); the properties keep the phase-1 kit identical and fail loudly
  for a layout without those surfaces.*

- [ ] **Step 1: Write the failing tests:**
  - `test_layout_places_every_tail_surface`: for g1, names `["htail", "vtail"]`, `layout.htail is
    layout.tail_surfaces[0].planform`, `len(layout.tail_booms) == 1`, `layout.propeller.is_pusher is False`,
    `layout.propeller.x_m == 0.0`.
  - `test_missing_surface_property_fails_loudly`: `dataclasses.replace(layout, tail_surfaces=())` raises
    `ValueError` on `.htail`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. The `tractor_nose` propeller sits at x = 0 on the fuselage axis, with the cruise
  propeller diameter the CAD propeller disc already uses.
- [ ] **Step 4:** Run host `tests/test_layouts.py tests/test_aero_cad.py tests/test_fabrication_*.py
  tests/test_wing_stiffness.py tests/test_layout_golden.py`, then Docker `tests/test_aero_cad.py
  tests/test_layout_golden.py`. Expected: PASS, including the `vsp_geometry` snapshot.
- [ ] **Step 5:** Commit `refactor: AircraftLayout holds general tail surfaces, booms and propeller station`.

### Task 5: Static-margin band from the layout

**Files:**
- Modify: `src/tools/aero_cad/stability.py`, `src/tools/aero_cad/cad_packer.py`, `src/tools/aero_cad/build.py`,
  `src/agent/critic.py`, `src/pipeline.py`, `src/tools/flight_control/perturbation.py`
- Test: `tests/test_layouts.py`

**Interfaces:**
- Consumes: `layout_spec(sizing.airframe_layout).static_margin` (Task 2).
- Produces: `SM_MIN`, `SM_MAX` and `SM_TARGET` are removed. Every reader takes a `StaticMarginBand` from the
  sizing's layout: the critic gate (text stays `f"SM {sm:.3f} (band {band.minimum}-{band.maximum})"`), the
  critic's increase/decrease advice, the pipeline note, the packer's battery-slide target and `build.py`'s
  candidate ranking. `PlantCase.static_margin` becomes `margin_end: Literal["minimum", "maximum"]` (HOT →
  minimum, SLUGGISH → maximum), and `perturbed_model(model, case, band)` reads the value from the band.

- [ ] **Step 1: Write the failing test** `test_every_reader_uses_the_layout_band`: monkeypatch `layout_spec` to
  return a spec with band `(0.05, 0.10, 0.07)`; assert the critic's static-margin gate passes SM 0.06 and fails
  0.12, the pipeline note appears for 0.12, the packer's slide target is 0.07, and
  `perturbed_model(model, HOT, band).static_margin == 0.05`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. Afterwards `grep -rn "SM_MIN\|SM_MAX\|SM_TARGET" src tests` finds nothing.
- [ ] **Step 4:** Run `tests/test_layouts.py tests/test_evaluator.py tests/test_pipeline.py tests/test_aero_cad.py
  tests/test_flight_control.py tests/test_layout_golden.py`. Expected: PASS.
- [ ] **Step 5:** Commit `refactor: static-margin band comes from the layout`.

### Task 6: One shared gear description

**Files:**
- Create: `src/tools/gear.py`
- Modify: `src/tools/flight_control/jsbsim_exporter.py`, `src/tools/fabrication/kit_mass.py`,
  `src/tools/sizing/mass_model.py` (`GEAR_FRACTION` moves out)
- Test: `tests/test_layouts.py`

**Interfaces:**
- Produces in `src/tools/gear.py`:
  ```python
  @dataclass(frozen=True)
  class GearContact:
      name: str
      x_m: float
      y_m: float
      z_m: float
      steerable: bool

  @dataclass(frozen=True)
  class GearGeometry:
      kind: Literal["none", "tricycle"]
      contacts: tuple[GearContact, ...]

  def tricycle_gear(fuselage_length_m: float, fuselage_diameter_m: float, prop_diameter_m: float,
                    mean_chord_m: float, wingspan_m: float, x_cg_m: float) -> GearGeometry
  def gear_mass_kg(airframe_topology: str, mtow_kg: float) -> float     # GEAR_FRACTION × MTOW, as today
  ```
  Moving here: `GearContact`, `GEAR_CLEARANCE_FRACTION`, `NOSE_GEAR_STATION`, `MAIN_GEAR_AFT_OF_CG_CHORDS`,
  `TIP_BACK_ANGLE_DEG`, `MAIN_GEAR_TRACK_FRACTION` and `GEAR_FRACTION`. Spring and damping constants stay in the
  exporter (simulation, not geometry). Phase 1 builds only `tricycle`; `none` exists for phase 2's hand-launched G1.
  The exporter calls `tricycle_gear` from the flight model's fields; `residual_gear_item` and `mass_model` call
  `gear_mass_kg`.

- [ ] **Step 1: Write the failing test** `test_gear_is_one_description`: for g1, the exporter's contacts equal
  `tricycle_gear(...)` built from the flight model, and `residual_gear_item(sizing, layout).mass_kg ==
  gear_mass_kg(sizing.airframe_topology, sizing.mtow_kg)`.
- [ ] **Step 2:** Run. Expected: FAIL (module missing).
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `tests/test_layouts.py tests/test_flight_control.py tests/test_fabrication_kit.py
  tests/test_sizing.py tests/test_layout_golden.py`. Expected: PASS (jsbsim, kit and sizing snapshots unchanged).
- [ ] **Step 5:** Commit `refactor: one gear description shared by sizing, the kit and the JSBSim model`.

### Task 7: Control channels per physical surface

**Files:**
- Create: `src/tools/flight_control/control_channels.py`
- Modify: `src/tools/flight_control/flight_model.py`, `param_generator.py`, `jsbsim_exporter.py`, `sitl_bridge.py`
- Test: `tests/test_layouts.py`, `tests/test_flight_control.py` (SITL test, Docker)

**Interfaces:**
- Produces in `control_channels.py`:
  ```python
  @dataclass(frozen=True)
  class SurfaceDerivatives:          # per rad of this surface's own deflection
      cl: float = 0.0                # lift
      cl_roll: float = 0.0           # rolling moment
      cm: float = 0.0
      cn: float = 0.0

  @dataclass(frozen=True)
  class ControlChannel:
      surface: str                   # JSBSim name stem: "aileron", "elevator", "rudder"
      servo: int                     # ArduPilot output number (1-based)
      servo_function: int            # SERVOn_FUNCTION
      bridge_sign: float             # +1 / -1: PWM above trim -> positive deflection
      max_deflection_rad: float
      derivatives: SurfaceDerivatives

  def control_channels(aero: AeroCoefficients) -> tuple[ControlChannel, ...]
  ```
  Tractor channels: aileron (servo 1, function 4, sign +1, `AILERON_MAX_RAD`, `cl_roll=cl_delta_a`,
  `cn=cn_delta_a`); elevator (2, 19, −1, `ELEVATOR_MAX_RAD`, `cl=cl_delta_e`, `cm=cm_delta_e`); rudder (4, 21, −1,
  `RUDDER_MAX_RAD`, `cn=cn_delta_r`). Throttle (servo 3, function 70) and the lift motors are unchanged.
  `FlightModel` and `JSBSimModel` gain `channels: tuple[ControlChannel, ...]`.
- The exporter writes one actuator per channel and one coefficient per non-zero derivative, with today's names
  (`fcs/<surface>-cmd-norm`, `fcs/<surface>-pos-rad`; `CLde`, `Clda`, `Cmde`, `Cndr`, `Cnda` from a fixed
  `(surface, axis) -> name` map), so the XML is unchanged.
- The bridge's `_apply` loops over the channels: `fdm[f"fcs/{c.surface}-cmd-norm"] = c.bridge_sign *
  _centred(pwm[c.servo - 1])`; nose-wheel steering stays tied to the rudder servo. `FW_SERVO_FUNCTIONS` is built
  from the channels plus throttle. **No un-mixing anywhere:** a future mixed surface is its own channel.

- [ ] **Step 1: Write the failing tests:**
  - `test_tractor_channels`: `[(c.surface, c.servo, c.servo_function) for c in control_channels(model.aero)] ==
    [("aileron", 1, 4), ("elevator", 2, 19), ("rudder", 4, 21)]`.
  - `test_servo_functions_come_from_channels`: `generate_params(...)` has `SERVO1_FUNCTION=4`,
    `SERVO2_FUNCTION=19`, `SERVO3_FUNCTION=70`, `SERVO4_FUNCTION=21`.
  - `test_bridge_applies_each_channel` (a dict stands in for the FDM; no JSBSim): PWM 1600 on servo 2 sets
    `fcs/elevator-cmd-norm` to `-_centred(1600)`.
  - `test_reversed_elevator_fails_sitl` (Docker; skipped the way the existing SITL tests in
    `tests/test_flight_control.py` skip without ArduPilot): run the g1 SITL check as the existing SITL test does,
    with `SERVO2_REVERSED=1` added to the parameters; assert the check reports failure.
- [ ] **Step 2:** Run the host tests. Expected: FAIL (module missing).
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run host `tests/test_layouts.py tests/test_flight_control.py tests/test_layout_golden.py`, then
  Docker `tests/test_flight_control.py tests/test_layout_golden.py`. Expected: PASS; the params and jsbsim
  snapshots are unchanged; `test_reversed_elevator_fails_sitl` passes (the check fails, as asserted).
- [ ] **Step 5:** Commit `refactor: JSBSim and the SITL bridge driven by per-surface control channels`.

### Task 8: Documentation and full verification

**Files:**
- Modify: `README.md` (layout field; phase 1 offers only the tractor), `REQUIREMENTS.md` (tail surfaces, static-margin
  band per layout, gear description, control channels)

- [ ] **Step 1:** Update the docs.
- [ ] **Step 2:** Run the full host suite. Expected: all pass (1,262 + the new tests; skips as today).
- [ ] **Step 3:** Run the full Docker suite. Expected: all pass (1,276 + the new tests; 1 skip).
- [ ] **Step 4:** Run the full g1 pipeline end to end in Docker and compare `run_summary.json` (MTOW, static margin,
  gate results, SITL pass) with a run on `d125b73`. Expected: identical.
- [ ] **Step 5:** Commit `docs: airframe layout field and the phase 1 refactor`.
