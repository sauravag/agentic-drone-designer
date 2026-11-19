# Laser-Cut Airframe Kit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every Group 1–2 design gets a laser-cut kit (DXF sheets, build sheet, 3D parts) whose flat part outlines are also the source of the as-built mass, CG and inertia.

**Architecture:** A new `src/tools/fabrication/` package turns `AircraftLayout` + `FabricationSettings` into a `Kit`: 2D Build123d faces per part (with 3D placement), tubes sized by deterministic bending checks, linear stock and covering. `cad_packer.pack_layout` swaps the ribs/spar/OML-shell lump for the kit's mass items (falling back to today's model with a warning when no kit is possible). Nesting, kerf offset, DXF, build sheet and 3D export run once for the final design.

**Tech Stack:** Python 3.12, Build123d 0.13 (2D faces, offset, extrude, `ExportDXF`), ezdxf 1.4.5 (tests read DXFs back), pydantic v2 frozen models, Streamlit 1.67, pytest.

**Spec:** `docs/superpowers/specs/2026-11-19-laser-cut-airframe-design.md`

## Global Constraints

- Deterministic; SI internally (m, kg, N, Pa); DXF in millimetres. Zero LLM Math.
- Frozen pydantic models with `extra="forbid"` for settings; frozen dataclasses for geometry results.
- Functions < 50 lines, files < 800 lines, no magic numbers (named `Final` constants), type hints everywhere.
- TDD: each test is seen failing before its code; 80 %+ coverage on `src/tools/fabrication/`.
- `layout.py` is the single source of positions; fabrication reads it, never writes back.
- Kit failure never fails the run: fall back to the shell model and add a warning naming the reason.
- Defaults verbatim from the spec: bed 600 × 400 mm, margin 5 mm, kerf 0.15 mm; birch ply 3 mm and 1.5 mm at
  680 kg/m³; stringers 3 × 3 mm balsa 160 kg/m³; LE 6 × 6 mm balsa; TE 5 × 15 mm triangular balsa; film
  0.09 kg/m²; fuselage covering 0.24 kg/m²; glue/hardware 10 % of structure; carbon 1600 kg/m³; σ_allow 500 MPa;
  FS 1.5; n_limit 4.0; minimum spar insertion (skin to centreline) 3 × spar OD; fit clearance ≤ 0.2 mm; tube list thin-wall
  6×4 … 25×23 (2 mm OD steps, 1 mm wall) and thick-wall 20×16, 24×20, 28×24, 32×28, 36×32, 40×36 (each ID equals
  the next smaller OD, so every tube has a nested partner); rods 3, 4, 5 mm. (Ruling while planning: the spec's
  thick-wall list 25×21, 30×26, 35×31 had no nested partners, so no socket could ever be found for those spars;
  the spec's defaults are updated to this list.)
- Rib pitch stays 0.20 m (`structure.RIB_PITCH_M`). Aileron span 0.55–0.95, hinge at 75 % chord; tail hinge at
  65 % chord (reuse `openvsp_builder.AILERON_ETA`, `AILERON_CHORD_FRACTION`, `TAIL_CONTROL_CHORD_FRACTION` by
  moving them to a module without the OpenVSP import if needed — `openvsp_builder` must not be imported outside
  Docker).
- Commits end with the two attribution lines from the session (`Co-Authored-By: WOZCODE …`, `Claude-Session: …`).

## Review Focus

1. A rib whose airfoil is thinner at the spar than the chosen tube (thin tip, small Group 1 wing) — the kit must
   fall back with "no stock tube fits the wing at y = … m", never emit a rib with the hole breaking out. (Task 2.)
2. A part longer than the bed in both orientations (crutch of a 1.5 m pod, Group 2 root rib > 600 mm) — split
   with a dovetail, and no nested part ever exceeds the bed. (Tasks 5, 8.)
3. Kerf compensation direction: outer outlines grow and holes shrink — a slot cut at nominal thickness then
   kerf-offset must be narrower than nominal by exactly kerf. (Task 8.)
4. Mass bookkeeping: kit + residual gear + components must equal the reported as-built mass, with no OML shell
   mass when a kit exists and no kit mass when it falls back. (Task 7.)
5. Old runs and `design.json` without `fabrication`: views, reruns and the zip must still work with defaults.
   (Task 10.)

---

## File structure

| File | Responsibility |
|---|---|
| `src/schemas/fabrication.py` (new) | `FabricationSettings`, `SheetStock`, `TubeStock`, `DEFAULT_FABRICATION` |
| `src/tools/fabrication/__init__.py` (new) | empty |
| `src/tools/fabrication/kit.py` (new) | `KitPart`, `TubePart`, `LinearStock`, `Kit`, `KitUnavailable` |
| `src/tools/fabrication/tubes.py` (new) | moments, stress, stock selection, `SparPlan` |
| `src/tools/fabrication/wing_parts.py` (new) | wing + tail ribs, root doublers, servo plates |
| `src/tools/fabrication/fuselage_parts.py` (new) | crutch, half-formers, firewall, wing-mount side plates, openings |
| `src/tools/fabrication/fittings.py` (new) | component plates, boom clamps, motor plates, splitting with dovetails |
| `src/tools/fabrication/build_kit.py` (new) | `build_kit`, kit mass items, residual gear, covering |
| `src/tools/fabrication/nesting.py` (new) | shelf packing onto sheets |
| `src/tools/fabrication/kit_export.py` (new) | kerf offset, DXF per sheet, `build_sheet.md`, `parts.csv`, 3D solids |
| `src/tools/aero_cad/cad_packer.py` | use kit mass; `PackingResult.kit`, `.kit_warning`; export kit |
| `src/tools/aero_cad/build.py`, `src/pipeline.py` | pass `FabricationSettings` through |
| `src/tools/aero_cad/mesh_export.py` | kit parts in the viewer model |
| `src/schemas/state.py` | `CADArtifacts.kit_dir`, `.build_sheet_path`, `.kit_mass_kg`, `.kit_warning`; `DesignState.fabrication` |
| `src/agent/graph.py`, `src/agent/nodes.py`, `src/ui/agent_service.py` | carry settings; write `fabrication.json` |
| `src/ui/design_archive.py` | `DesignRecord.fabrication`; kit paths relative |
| `src/ui/fabrication_panel.py` (new), `src/ui/app.py`, `src/ui/presenters.py` | settings expander; kit summary; kit files listed |
| `REQUIREMENTS.md` §4.3, `README.md` | spar rule replaced; kit outputs documented |

Tests: `tests/test_fabrication_settings.py`, `tests/test_fabrication_tubes.py`, `tests/test_fabrication_parts.py`,
`tests/test_fabrication_kit.py`, `tests/test_fabrication_export.py` (new); `tests/test_aero_cad.py`,
`tests/test_design_archive.py`, `tests/test_ui.py` (modified). Fabrication tests reuse `test_aero_cad`'s
`designs` / `_fitted_layout` helpers: move `catalog`, `designs`, `_fitted_layout`, `_synthetic_aero`,
`_synthetic_oml` into `tests/conftest.py` (fixtures) and `tests/cad_helpers.py` (functions) in Task 3.

---

### Task 1: Fabrication settings schema

**Files:** Create `src/schemas/fabrication.py`; Test `tests/test_fabrication_settings.py`

**Interfaces — Produces:**
- `PartFamily = Literal["rib", "former", "crutch", "plate", "doubler", "clamp", "bulkhead"]`
- `SheetStock(name: str, thickness_m: float>0, density_kg_m3: float>0, families: tuple[PartFamily, ...])`
- `TubeStock(od_m: float>0, id_m: float>=0)` (validator: `id_m < od_m`; `id_m == 0` is a solid rod); property
  `area_m2`, `second_moment_m4`.
- `FabricationSettings(bed_width_m=0.600, bed_height_m=0.400, margin_m=0.005, kerf_m=0.00015, sheets=…,
  tubes=…, rods_m=(0.003, 0.004, 0.005), stringer_side_m=0.003, stringer_density_kg_m3=160.0,
  le_side_m=0.006, te_height_m=0.005, te_chord_m=0.015, film_kg_m2=0.09, fuselage_cover_kg_m2=0.24,
  fuselage_cover_thickness_m=0.0015, glue_fraction=0.10, carbon_density_kg_m3=1600.0,
  sigma_allow_pa=500e6, safety_factor=1.5, n_limit=4.0, min_insertion_diameters=3.0, fit_clearance_m=0.0002)`.
  Validators: every `PartFamily` covered by exactly one sheet; tubes sorted by OD after validation.
  Method `sheet_for(family: PartFamily) -> SheetStock`.
- `DEFAULT_FABRICATION: Final[FabricationSettings]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_defaults_match_the_spec():
    s = DEFAULT_FABRICATION
    assert (s.bed_width_m, s.bed_height_m, s.margin_m, s.kerf_m) == (0.600, 0.400, 0.005, 0.00015)
    assert s.sheet_for("rib").thickness_m == 0.0015 and s.sheet_for("former").thickness_m == 0.003
    assert {(round(t.od_m * 1e3), round(t.id_m * 1e3)) for t in s.tubes} >= {(25, 23), (36, 32), (40, 36), (6, 4)}

def test_every_family_needs_exactly_one_sheet():
    three_mm = DEFAULT_FABRICATION.sheet_for("former")
    with pytest.raises(ValidationError, match="rib"):
        FabricationSettings(sheets=(three_mm,))  # the 3 mm sheet does not list "rib"

def test_tube_id_must_be_below_od():
    with pytest.raises(ValidationError):
        TubeStock(od_m=0.010, id_m=0.010)

def test_round_trips_json():
    assert FabricationSettings.model_validate_json(DEFAULT_FABRICATION.model_dump_json()) == DEFAULT_FABRICATION

def test_second_moment_of_a_tube():
    t = TubeStock(od_m=0.025, id_m=0.023)
    assert t.second_moment_m4 == pytest.approx(math.pi / 64 * (0.025**4 - 0.023**4))
```

- [ ] **Step 2:** Run `.venv/bin/python -m pytest tests/test_fabrication_settings.py -q` — Expected: FAIL (`ModuleNotFoundError`).
- [ ] **Step 3:** Implement `src/schemas/fabrication.py` per the Interfaces block.
- [ ] **Step 4:** Re-run — Expected: 5 passed.
- [ ] **Step 5:** Commit `feat: fabrication settings schema`.

---

### Task 2: Tube sizing and structural checks

**Files:** Create `src/tools/fabrication/__init__.py`, `src/tools/fabrication/tubes.py`; Test `tests/test_fabrication_tubes.py`

**Interfaces — Consumes:** `TubeStock`, `FabricationSettings`. **Produces:**
- `class KitUnavailable(Exception)` — defined here (re-exported by `kit.py` in Task 3), message is user-facing.
- `wing_root_moment_nm(mtow_kg: float, exposed_semi_span_m: float, n_limit: float) -> float` =
  `n_limit · mtow·g/2 · 4·s/(3π)`.
- `hover_moment_nm(mtow_kg: float, arm_m: float) -> float` = `2 · (mtow·g/4) · arm`.
- `bending_stress_pa(tube: TubeStock, moment_nm: float) -> float` = `M·(od/2)/I`.
- `joint_y_m(fuselage_radius_m: float, wing_z_m: float) -> float` = `√(R² − z²)`.
- `@dataclass(frozen=True) SparSegment(tube: TubeStock, y_start_m: float, y_end_m: float)`;
  `@dataclass(frozen=True) SparPlan(segments: tuple[SparSegment, ...], socket: TubeStock, joint_y_m: float,
  insertion_m: float)`.
- `usable_depth_m(local_thickness_m: float, settings) -> float` = thickness − 2·`MIN_WEB_M`
  (`MIN_WEB_M = 0.006`, defined here, reused by Task 3; film covering adds no thickness).
- `plan_spar(mtow_kg, semi_span_m, joint_y_m, depth_at: Callable[[float], float], boom_y_m: Optional[float],
  settings) -> SparPlan` where `depth_at(y)` is `usable_depth_m` of the airfoil thickness at 25 % chord at span station y. Algorithm: root segment = smallest stock tube whose stress
  under max(flight, hover) moment at `joint_y` ≤ σ_allow/FS and whose OD ≤ `depth_at` over its extent; walk
  outboard rib by rib, and where `depth_at(y)` < current OD, step to the largest tube that (a) fits inside the
  current one within `fit_clearance_m`, (b) fits the depth, (c) passes the moment at that y (moment at y uses the
  same elliptic-centroid formula on the remaining span s' = semi_span − y). Socket = smallest tube with
  `id ≥ root.od` and `id − root.od ≤ fit_clearance_m` that passes the moment at the centreline (arm from y = 0).
  Raise `KitUnavailable` naming the station or the missing nested tube.

- [ ] **Step 1: Write failing tests**

```python
def test_reviewer_numbers_25kg():
    m = wing_root_moment_nm(25.0, 2.0, 4.0)
    assert m == pytest.approx(4.0 * 25 * 9.80665 / 2 * 4 * 2.0 / (3 * math.pi), rel=1e-9)  # ~416 N·m
    assert bending_stress_pa(TubeStock(od_m=0.025, id_m=0.023), m) > 900e6

def test_25kg_design_takes_a_32mm_spar_and_36mm_socket():
    plan = plan_spar(25.0, 2.1, 0.1, lambda y: 0.050, None, DEFAULT_FABRICATION)
    assert round(plan.segments[0].tube.od_m * 1e3) == 32   # 28×24 → 419 MPa fails; 32×28 → 312 MPa ≤ 333
    assert (round(plan.socket.od_m * 1e3), round(plan.socket.id_m * 1e3)) == (36, 32)

def test_spar_steps_down_inside_the_previous_tube():
    depth = lambda y: 0.030 if y < 0.8 else 0.019
    plan = plan_spar(5.0, 1.4, 0.06, depth, None, DEFAULT_FABRICATION)
    outer, inner = plan.segments[0].tube, plan.segments[1].tube
    assert inner.od_m <= outer.id_m and outer.id_m - inner.od_m <= 0.0002
    assert plan.segments[1].y_start_m == pytest.approx(plan.segments[0].y_end_m)

def test_socket_is_larger_than_the_spar():
    plan = plan_spar(5.0, 1.4, 0.06, lambda y: 0.030, None, DEFAULT_FABRICATION)
    assert plan.socket.id_m >= plan.segments[0].tube.od_m

def test_hover_moment_governs_when_it_is_larger():
    gentle = DEFAULT_FABRICATION.model_copy(update={"n_limit": 1.0})  # flight 23.6 N·m < hover 43.9 N·m
    flight = plan_spar(8.0, 1.5, 0.08, lambda y: 0.06, None, gentle)
    hover = plan_spar(8.0, 1.5, 0.08, lambda y: 0.06, 1.2, gentle)
    assert hover.segments[0].tube.od_m > flight.segments[0].tube.od_m

def test_too_thin_wing_is_unavailable():
    with pytest.raises(KitUnavailable, match="no stock tube fits the wing at y ="):
        plan_spar(25.0, 2.1, 0.1, lambda y: 0.020, None, DEFAULT_FABRICATION)

def test_joint_is_on_the_skin():
    assert joint_y_m(0.10, 0.06) == pytest.approx(0.08)
```

- [ ] **Step 2:** Run `.venv/bin/python -m pytest tests/test_fabrication_tubes.py -q` — Expected: FAIL (import error).
- [ ] **Step 3:** Implement `tubes.py`. g = `9.80665` as a named constant (reuse one from `src/units.py` if present).
- [ ] **Step 4:** Re-run — Expected: 7 passed.
- [ ] **Step 5:** Commit `feat: stock-tube spar and socket sizing`.

---

### Task 3: Kit model and wing/tail ribs

**Files:** Create `src/tools/fabrication/kit.py`, `src/tools/fabrication/wing_parts.py`, `tests/cad_helpers.py`; Modify `tests/conftest.py`, `tests/test_aero_cad.py` (import moved helpers); Test `tests/test_fabrication_parts.py`

**Interfaces — Consumes:** `SparPlan`, `KitUnavailable` (Task 2), `AircraftLayout`, `AirfoilShape`, `FabricationSettings`. **Produces:**
- `@dataclass(frozen=True) KitPart(label: str, family: PartFamily, outline: bd.Face, location: bd.Location,
  quantity: int = 1)`; outline is nominal (no kerf), in the part's own XY plane in metres; `location` maps it
  into the aircraft frame (x aft, y right, z up), extrusion along local +Z by the sheet thickness.
  Property `area_m2`.
- `@dataclass(frozen=True) TubePart(label: str, role: Literal["spar","socket","tail_boom","tail_spar",
  "lift_boom","pin"], tube: TubeStock, length_m: float, start_m: Vec3, axis: Vec3)`.
- `@dataclass(frozen=True) LinearStock(label: str, kind: Literal["stringer","le","te"], section_area_m2: float,
  density_kg_m3: float, length_m: float, start_m: Vec3, end_m: Vec3)`.
- `@dataclass(frozen=True) Kit(parts: tuple[KitPart,...], tubes: tuple[TubePart,...], linear:
  tuple[LinearStock,...], film_area_m2: float, fuselage_cover_area_m2: float, spar: SparPlan,
  joins: tuple[str, ...])` (`joins`: human-readable "C-1a + C-1b at x = 0.512 m").
- `wing_ribs(layout, airfoil, spar: SparPlan, settings) -> tuple[KitPart, ...]`: right panel labels `WR-01…`,
  left `WL-01…` (mirrored location), stations every `RIB_PITCH_M` from `spar.joint_y_m` (R2);
  ribs inside the aileron span are two parts `WR-07` and `WR-07A`; every rib's hole is the local spar member
  OD + `fit_clearance_m` (the socket stays inside the fuselage, R11); root rib `WR-01` is family `doubler`
  (3 mm) with a 4 mm pin hole at 70 % chord; one servo plate `WR-SV` / `WL-SV` per aileron bay.
- `tail_ribs(layout, airfoil, settings) -> tuple[KitPart, ...]`: labels `HR-`, `HL-`, `V-`, plus `TM-1`
  mount plate; tail spar = smallest stock tube fitting the tail root depth (no stress check; record as
  `TubePart` in Task 7).

Geometry rules (from spec): airfoil inset by covering on top/bottom (film: 0 m), LE notch `le_side_m` square,
TE notch to `te_chord_m`, spar hole at 25 % chord on the camber line, wiring hole Ø 8 mm at 50 % chord,
lightening holes keep a minimum web `MIN_WEB_M = 0.006` from every edge and hole (constant; shape is the
builder's call).

- [ ] **Step 1: Write failing tests.** Add fixture `g1_kit_inputs` in `tests/conftest.py` returning
  `(layout, airfoil, spar_plan, DEFAULT_FABRICATION)` for the `g1` design (`_fitted_layout`, `load_airfoil`,
  `plan_spar` with `depth_at` from the airfoil), in that order — `wing_ribs(*g1_kit_inputs)` must work.

```python
def test_rib_hole_matches_the_tube_through_it(g1_kit_inputs):
    ribs = wing_ribs(*g1_kit_inputs)
    outer = ribs_by_label(ribs)["WR-05"]
    spar = g1_kit_inputs[2]
    tube = next(s.tube for s in spar.segments if s.y_start_m <= outer.location.position.Y < s.y_end_m)
    assert hole_diameters(outer.outline)[0] == pytest.approx(tube.od_m + 0.0002, abs=1e-6)  # outside insertion

def test_left_ribs_mirror_right(g1_kit_inputs):
    ribs = ribs_by_label(wing_ribs(*g1_kit_inputs))
    assert ribs["WL-03"].location.position.Y == pytest.approx(-ribs["WR-03"].location.position.Y)

def test_aileron_ribs_split_at_75_percent(g1_kit_inputs): ...   # WR-k and WR-kA exist only for 0.55–0.95 semi-span
def test_every_rib_keeps_a_minimum_web(g1_kit_inputs): ...      # each inner wire ≥ MIN_WEB_M from outer wire and other holes
def test_root_rib_is_a_3mm_doubler_with_pin_hole(g1_kit_inputs): ...  # WR-01.family == "doubler"; a 4 mm hole at 70 % chord
def test_ribs_over_the_socket_have_socket_holes(g1_kit_inputs): ...   # stations < joint_y + insertion → socket.od + clearance
def test_rib_count_follows_the_20cm_pitch(g1_kit_inputs): ...         # len(WR-*) == len(structure.rib_stations) from joint_y
def test_tail_ribs_split_at_65_percent(g1_kit_inputs): ...            # HR-k / HR-kA split x == 0.65 chord
```
Helpers (`hole_diameters`, `ribs_by_label`) live in `tests/cad_helpers.py`.

- [ ] **Step 2:** Run `.venv/bin/python -m pytest tests/test_fabrication_parts.py -q` — Expected: FAIL.
- [ ] **Step 3:** Move fixtures/helpers out of `tests/test_aero_cad.py`; implement `kit.py` and `wing_parts.py`.
- [ ] **Step 4:** Run `.venv/bin/python -m pytest tests/test_fabrication_parts.py tests/test_aero_cad.py -q` — Expected: all pass (aero_cad unchanged count).
- [ ] **Step 5:** Commit `feat: kit model and wing/tail rib outlines`.

---

### Task 4: Fuselage parts

**Files:** Create `src/tools/fabrication/fuselage_parts.py`; Test `tests/test_fabrication_parts.py`

**Interfaces — Consumes:** `KitPart`, `SparPlan`, `layout.fuselage_radius_at`, `components.PlacedComponent`. **Produces:**
- `former_stations(layout, spar: SparPlan, settings) -> tuple[float, ...]`: every `FORMER_PITCH_M = 0.12` from
  `FIRST_FORMER_FRACTION = 0.10` to `TAIL_BOOM_START_FRACTION` of length, plus stations ±(socket OD/2 + 10 mm)
  around the spar x, deduplicated (merge within 20 mm).
- `fuselage_parts(layout, spar, components: Sequence[PlacedComponent], settings) -> tuple[KitPart, ...]`:
  crutch `C-1` (horizontal plate z = 0, width = 2·(r(x) − cover) along x, openings under each internal component
  footprint + `COMPONENT_GAP_M`); half-formers `F-nnU` / `F-nnL` (semi-disc of r(x) − cover, tab into a crutch
  slot, `STRINGER_COUNT = 8` notches at equal angles, opening = component's yz bounding box + gap where a
  component crosses x); firewall `FW-1a`, `FW-1b` (two solid 3 mm discs at the motor station); wing-mount side
  plates `WM-L`, `WM-R` (xz plane, y = ±(r − cover), slotted into the crutch, holes for socket OD + clearance and
  the 4 mm pin at 70 % root chord); tail-boom sleeve holes in the last two formers. Empty `components` gives
  formers with only lightening holes (used during balance estimates — see Task 7).

- [ ] **Step 1: Write failing tests**

```python
def test_formers_follow_the_oml_minus_covering(g1_layout, …):
    f = by_label(parts)["F-03U"]
    assert outline_radius(f) == pytest.approx(fuselage_radius_at(g1_layout, f.location.position.X) - 0.0015, abs=1e-5)

def test_component_crossing_a_former_gets_an_opening(...)   # battery yz box ⊂ opening of every former it spans
def test_crutch_has_an_opening_under_each_component(...)
def test_wing_mount_plates_carry_socket_and_pin_holes(...)  # hole Ø = socket.od + clearance, 4 mm pin
def test_firewall_is_two_solid_layers(...)
def test_slot_width_equals_mating_thickness(...)            # crutch slots == 0.003 nominal
```

- [ ] **Step 2:** Run `.venv/bin/python -m pytest tests/test_fabrication_parts.py -q -k "former or crutch or wing_mount or firewall or slot"` — Expected: FAIL.
- [ ] **Step 3:** Implement `fuselage_parts.py`.
- [ ] **Step 4:** Re-run — Expected: 6 passed.
- [ ] **Step 5:** Commit `feat: fuselage crutch, formers and wing mounts`.

---

### Task 5: Fittings and splitting oversize parts

**Files:** Create `src/tools/fabrication/fittings.py`; Test `tests/test_fabrication_parts.py`

**Interfaces — Produces:**
- `component_plates(components, layout, settings) -> tuple[KitPart, ...]`: one `plate` per internal component
  (`P-<component name>`), footprint = component x/y size + 2·gap, z just below the component.
- `boom_parts(layout, spar, settings) -> tuple[tuple[KitPart, ...], tuple[TubePart, ...]]`: quadplane only —
  clamp plates `BC-L1/L2/R1/R2` (saddle holes: spar OD at its y, boom OD), motor plates `MP-1..4` (centre mark
  hole Ø 3 mm + slotted cross at 16/19/25 mm), lift-boom tubes = smallest stock with OD ≥ layout boom diameter;
  tail boom tube = smallest stock with OD ≥ `layout.tail_boom.diameter_m`. Fixed-wing: no clamps/motor plates.
- `split_to_bed(part: KitPart, settings) -> tuple[KitPart, ...]`: if the part's bounding box fits the usable bed
  (bed − 2·margin) in either orientation, return it unchanged; else cut perpendicular to its long axis into the
  fewest equal pieces that fit, each cut moved up to ±`SPLIT_SEARCH_M = 0.03` to avoid holes/slots, joined by a
  dovetail (neck `DOVETAIL_NECK_M = 0.010`, flare `DOVETAIL_FLARE_M = 0.016`, depth `0.012`) plus a doubler part
  `<label>-D<k>` of the same family; labels `<label>a`, `<label>b`, …; returns pieces + doublers. Raise
  `KitUnavailable` if the short side alone exceeds the bed's short side.
- `join_notes(pieces) -> tuple[str, ...]`.

- [ ] **Step 1: Write failing tests**

```python
def test_part_that_fits_is_untouched(): ...  # 0.2 × 0.1 m plate → (part,) identical
def test_long_crutch_splits_into_pieces_that_fit(): # 1.4 m × 0.12 m plate → 3 pieces, each within 590 × 390 mm
def test_split_pieces_area_plus_dovetails_conserves_area(): # sum(pieces) == original area within 1e-9 m²
def test_split_avoids_holes(): # no cut line within 3 mm of an inner wire
def test_part_wider_than_the_bed_is_unavailable(): # 0.5 m × 0.5 m → KitUnavailable
def test_fixed_wing_has_no_boom_clamps(designs): ...  # g1 → no BC-/MP- labels, no lift_boom tubes
def test_boom_tube_is_stock_at_least_layout_diameter(designs): ...  # g2 lift booms: tube.od ≥ boom.diameter_m, tube ∈ settings.tubes
```

- [ ] **Step 2–4:** Run `.venv/bin/python -m pytest tests/test_fabrication_parts.py -q` (FAIL → implement → PASS).
- [ ] **Step 5:** Commit `feat: fittings, boom clamps and bed-size splitting`.

---

### Task 6: `build_kit` and kit mass items

**Files:** Create `src/tools/fabrication/build_kit.py`; Test `tests/test_fabrication_kit.py`

**Interfaces — Consumes:** Tasks 2–5. **Produces:**
- `build_kit(sizing: SizingOutput, layout: AircraftLayout, airfoil: AirfoilShape,
  components: Sequence[PlacedComponent], settings: FabricationSettings) -> Kit` — raises `KitUnavailable` for
  `sizing.dod_group == 3` ("laser-cut construction covers Group 1–2 designs only") and propagates tube/split
  failures. Calls `split_to_bed` on every part. Adds linear stock: `STRINGER_COUNT` stringers along the fuselage between the
  first and last former, LE and TE strips per panel and tail surface. `film_area_m2` = wetted area of wing +
  tails (2 × planform area); `fuselage_cover_area_m2` = lateral area of the pod from `fuselage_radius_at`.
- `kit_mass_items(kit: Kit, settings) -> list[mp.MassItem]`: each `KitPart` → `mp.box_item`-equivalent from its
  face (mass = area × thickness × density, CG = face centroid mapped by `location`, inertia of a thin plate —
  use `mp.solid_item` only in tests as the oracle); tubes → `mp.cylinder_item` hollow equivalent (mass from
  annulus area); linear stock → point/rod items; covering → film and fuselage-cover items (uniform over the
  wing/tail planform centroids and pod centroid); glue/hardware = `glue_fraction` × structure total at the
  structure CG.
- `kit_mass_kg(kit, settings) -> float`.
- `residual_gear_item(sizing: SizingOutput, layout) -> mp.MassItem`: `GEAR_FRACTION[topology] · mtow_kg` spread
  over the pod (same centroid the OML shell gives the fuselage; a `cylinder_item` over the pod length is fine).

- [ ] **Step 1: Write failing tests**

```python
def test_group_3_is_unavailable(designs): ...  # match "Group 1–2"
def test_plate_mass_matches_extruded_solid(designs):  # kit_mass_items part vs mp.solid_item(extrude) within 1e-6 rel
def test_kit_mass_is_sum_of_parts_tubes_stock_covering_and_glue(designs): ...  # kit_mass_kg == Σ items; glue == 10 % of structure
def test_every_reference_design_in_group_1_2_gets_a_kit(designs): ...  # "g1" (4.1 kg FW) and "g2" (19.4 kg quadplane)
def test_kit_is_deterministic(designs): ...  # two builds → identical labels and masses
```

- [ ] **Step 2–4:** `.venv/bin/python -m pytest tests/test_fabrication_kit.py -q` FAIL → implement → PASS.
- [ ] **Step 5:** Commit `feat: build the kit and its mass items`.

---

### Task 7: Packing uses the kit (with fallback)

**Files:** Modify `src/tools/aero_cad/cad_packer.py`, `src/tools/aero_cad/build.py`, `src/pipeline.py`; Test `tests/test_aero_cad.py`

**Interfaces — Produces:**
- `pack_layout(sizing, bom, aero, layout, oml_step, settings: FabricationSettings = DEFAULT_FABRICATION) -> PackingResult`.
- `PackingResult` gains `kit: Optional[Kit]`, `kit_warning: Optional[str]`, `mass_items: tuple[mp.MassItem, ...]`
  (the items combined into `mass`); keeps `structure` (fallback only;
  `None` when a kit exists — make it `Optional[WingStructure]`) and `shell_areal_density_kg_m2` (0.0 with a kit).
- `balance_design(..., settings=DEFAULT_FABRICATION)`, `build_aero_cad(..., settings=DEFAULT_FABRICATION)`,
  `pipeline.run_aero_cad(..., settings=DEFAULT_FABRICATION)`.

Order inside `pack_layout` (decided here; the spec leaves the sequencing open):
1. Try `build_kit(..., components=fixed, settings)` → estimate items (formers with openings only for fixed parts).
   On `KitUnavailable` → today's path, `kit_warning = f"No laser kit: {exc}"`.
2. Static items = kit items + residual gear + fixed components; socket `TubePart` added as an on-axis obstacle
   (a `PlacedComponent` with `is_internal=False`, zero mass) so sliders avoid it.
3. Solve sliders exactly as today.
4. Rebuild the kit with all placed components (exact openings); final `mass = combine(kit items + gear +
   components)`; static margin from that exact mass (the slider solve used the estimate; `is_balanced` judges the
   exact value).

- [ ] **Step 1: Write failing tests** (in `test_aero_cad.py`, synthetic aero/OML as existing tests do)

```python
def test_kit_replaces_the_shell_mass(designs, tmp_path):
    r = pack(designs["g1"], tmp_path)
    assert r.kit is not None and r.shell_areal_density_kg_m2 == 0.0 and r.structure is None
    names = {i.name for i in r.mass_items}   # expose `mass_items: tuple[MassItem,...]` on PackingResult
    assert "oml_shell" not in names and "landing_gear" in names

def test_as_built_mass_is_kit_plus_gear_plus_components(designs, tmp_path): ...  # r.mass.mass_kg == kit_mass + gear + Σ components
def test_group_3_falls_back_with_a_warning(designs, tmp_path):
    r = pack(designs["g3"], tmp_path)
    assert r.kit is None and "Group 1–2" in r.kit_warning and r.structure is not None

def test_battery_never_overlaps_the_socket(designs, tmp_path): ...  # g1, g2: battery x-range disjoint from socket x ± OD/2 + gap
def test_formers_have_openings_for_the_placed_battery(designs, tmp_path): ...  # every former crossing the battery has an inner wire ⊇ its yz box
```
Update `test_airframe_lighter_than_ribs_and_spars_raises` and the `structure` tests to run on the fallback path
(Group 3 or a forced `KitUnavailable` via monkeypatch). Keep `structure.py` for the fallback.

- [ ] **Step 2:** `.venv/bin/python -m pytest tests/test_aero_cad.py -q` — Expected: new tests FAIL.
- [ ] **Step 3:** Implement; thread `settings` through `balance_design`, `build_aero_cad`, `run_aero_cad`.
- [ ] **Step 4:** `.venv/bin/python -m pytest tests/test_aero_cad.py tests/test_fabrication_*.py -q` — Expected: all pass.
- [ ] **Step 5:** Commit `feat: as-built mass from the laser kit`.

---

### Task 8: Nesting, kerf and DXF

**Files:** Create `src/tools/fabrication/nesting.py`; Create `src/tools/fabrication/kit_export.py` (DXF part); Test `tests/test_fabrication_export.py`

**Interfaces — Produces:**
- `@dataclass(frozen=True) Placement(part: KitPart, sheet: int, x_m: float, y_m: float, rotated: bool)`;
  `nest(parts: Sequence[KitPart], settings) -> dict[str, tuple[Placement, ...]]` keyed by sheet stock name;
  shelf packing by descending bounding-box height, rotation 0°/90° (pick the orientation that fits with lower
  height), `margin_m` around every part and the bed edge; parts with `quantity > 1` are placed that many times.
- `kerf_outline(face: bd.Face, kerf_m: float) -> bd.Face`: outer wire offset +kerf/2, inner wires −kerf/2.
- `write_sheets(placements, settings, out_dir: Path) -> tuple[Path, ...]`: `kit/sheets/<stock-name-slug>-<n>.dxf`,
  units mm, layers `CUT_INNER`, `CUT_OUTER`, `ENGRAVE` (label text, height `LABEL_HEIGHT_M = 0.004`).

- [ ] **Step 1: Write failing tests**

```python
def test_no_part_leaves_the_bed_or_overlaps(kit): ...       # every placed bbox within margin; pairwise disjoint + margin
def test_rotation_used_when_only_90_degrees_fits(): ...  # 0.35 × 0.55 m part → rotated=True
def test_kerf_grows_outlines_and_shrinks_holes():
    f = plate_with_hole(0.1, 0.05, hole_d=0.01)
    k = kerf_outline(f, 0.00015)
    assert outer_width(k) == pytest.approx(0.10015) and hole_diameters(k)[0] == pytest.approx(0.00985)
def test_dxf_round_trip_layers_and_units(kit, tmp_path):
    doc = ezdxf.readfile(write_sheets(nest(kit.parts, S), S, tmp_path)[0])
    assert {"CUT_INNER", "CUT_OUTER", "ENGRAVE"} <= {l.dxf.name for l in doc.layers}
    assert doc.header["$INSUNITS"] == 4  # millimetres
def test_every_label_engraved_once(kit, tmp_path): ...  # ENGRAVE TEXT entities across sheets == Σ quantities
```

- [ ] **Step 2–4:** `.venv/bin/python -m pytest tests/test_fabrication_export.py -q` FAIL → implement → PASS.
- [ ] **Step 5:** Commit `feat: nest kit parts and export kerf-compensated DXF sheets`.

---

### Task 9: Build sheet, parts list and 3D export

**Files:** Modify `src/tools/fabrication/kit_export.py`, `src/tools/aero_cad/cad_packer.py` (`export_artifacts`, `_export_assembly`), `src/tools/aero_cad/mesh_export.py`, `src/schemas/state.py`; Test `tests/test_fabrication_export.py`, `tests/test_aero_cad.py`, `tests/test_mesh_export.py`

**Interfaces — Produces:**
- `export_kit(kit: Kit, settings, out_dir: Path) -> KitFiles` (`@dataclass(frozen=True) KitFiles(kit_dir: Path,
  build_sheet: Path, parts_csv: Path, sheets: tuple[Path, ...])`).
- `kit_solids(kit, settings) -> list[bd.Solid]` (extrude each outline by its sheet thickness, apply `location`;
  tubes as hollow cylinders).
- `build_sheet.md` sections, in order: Summary (kit mass, sheets per stock), Parts (label, family, stock,
  quantity, station), Tubes (role, OD × ID mm, length mm), Linear stock (kind, section, total length), Joins
  (`kit.joins`), Covering (film area and mass, fuselage cover area and mass), Hardware (2 nylon wing bolts M5,
  2 captive-nut plates, 2 × 4 mm anti-rotation pins, 2 wing connectors).
- `parts.csv` columns: `label,family,stock,thickness_mm,quantity,area_mm2,mass_g,sheet`.
- `CADArtifacts` new optional fields: `kit_dir: Optional[str]`, `build_sheet_path: Optional[str]`,
  `kit_mass_kg: Optional[NonNegative]`, `kit_warning: Optional[str]`.
- `export_artifacts`: when `result.kit` exists, call `export_kit` and include `kit_solids` in `assembly.step`
  (replacing ribs/spars); `mesh_export.model_parts` makes the "structure" part from the kit solids.

- [ ] **Step 1: Write failing tests**

```python
def test_build_sheet_lists_every_label(kit, tmp_path): ...  # every KitPart.label and TubePart.label in build_sheet.md
def test_parts_csv_mass_sums_to_part_mass(kit, tmp_path): ...  # Σ mass_g / 1000 == Σ part masses within 1e-6
def test_kit_solid_volume_matches_part_mass(kit): ...  # Σ solid.volume × density == Σ part masses within 1e-6 rel
def test_artifacts_carry_kit_paths(designs, tmp_path): ...    # pack_aircraft → cad.build_sheet_path exists
def test_fallback_artifacts_carry_the_warning(designs, tmp_path): ...  # g3 → kit_dir None, kit_warning set
def test_viewer_structure_part_comes_from_the_kit(...): ...  # structure part built from kit_solids, non-empty
```

- [ ] **Step 2–4:** Run the three test files: FAIL → implement → PASS.
- [ ] **Step 5:** Commit `feat: build sheet, parts list and kit in the 3D assembly`.

---

### Task 10: Carry settings through the run and the export

**Files:** Modify `src/schemas/state.py` (`DesignState.fabrication`), `src/agent/graph.py` (`initial_state`, `run_agent` take `fabrication: Optional[FabricationSettings] = None`), `src/agent/nodes.py` (write `fabrication.json`; pass to `pipeline.run_aero_cad`), `src/ui/agent_service.py` (`run(..., fabrication=None)`, `_run_agent`, `AgentRunView.fabrication: FabricationSettings = DEFAULT_FABRICATION`, `to_view`, the rerun-with-change path keeps the parent view's settings), `src/ui/design_archive.py` (`DesignRecord.fabrication: FabricationSettings = DEFAULT_FABRICATION`; `_CAD_PATHS` adds `kit_dir`, `build_sheet_path`); Test `tests/test_agent_service.py`, `tests/test_design_archive.py`, `tests/test_agent.py`

- [ ] **Step 1: Write failing tests**

```python
def test_run_carries_fabrication_settings(...): # run(..., fabrication=custom) → view.fabrication == custom, fabrication.json written
def test_rerun_keeps_the_parent_settings(...):
def test_record_includes_fabrication_and_relative_kit_paths(tmp_path):
    rec = design_record(_view(tmp_path, cad=_cad(tmp_path).model_copy(update={
        "kit_dir": str(tmp_path / "kit"), "build_sheet_path": str(tmp_path / "kit" / "build_sheet.md")})))
    assert rec.cad.build_sheet_path == "kit/build_sheet.md" and rec.fabrication == DEFAULT_FABRICATION
def test_old_record_without_fabrication_validates(): # JSON lacking the key → defaults
def test_zip_includes_kit_folder(tmp_path): # kit/sheets/birch-ply-1-5mm-1.dxf present under r1/
```

- [ ] **Step 2–4:** `.venv/bin/python -m pytest tests/test_agent_service.py tests/test_design_archive.py tests/test_agent.py -q` FAIL → implement → PASS.
- [ ] **Step 5:** Commit `feat: fabrication settings travel with the design`.

---

### Task 11: UI — settings panel, kit summary, kit files

**Files:** Create `src/ui/fabrication_panel.py`; Modify `src/ui/app.py`, `src/ui/presenters.py`; Test `tests/test_ui.py`

**Interfaces — Produces:**
- `render_fabrication_panel() -> None` (expander **"Laser kit settings"** directly under "Design assumptions";
  number inputs for bed width/height (mm), margin (mm), kerf (mm); a `st.data_editor` for sheets and for tubes
  (OD/ID in mm); "Reset to defaults" button; validation errors shown with `st.error`).
- `current_fabrication() -> tuple[Optional[FabricationSettings], list[str]]`; `_execute` blocks the run on errors
  exactly as `_assumptions_or_stop` does, and passes `fabrication=` to `agent_service.run`.
- `presenters.kit_summary(cad: CADArtifacts) -> Optional[str]`: e.g. "Laser kit: 1.84 kg, 6 sheets (4 × 1.5 mm
  birch ply, 2 × 3 mm birch ply)"; `None` without a kit. Results show it next to the as-built mass; a
  `kit_warning` shows as `st.warning`.
- `presenters.downloadable_files` also lists `kit/build_sheet.md` and `kit/parts.csv` (add `.md`/`.csv` to
  `DOWNLOAD_SUFFIXES` if missing; DXFs reach the user through the zip).

- [ ] **Step 1: Write failing AppTest tests**

```python
def test_kit_settings_expander_present(...)
def test_invalid_kerf_blocks_the_run(...)          # kerf 0 → error, agent_service.run not called
def test_custom_bed_reaches_the_run(...)           # monkeypatched run receives fabrication.bed_width_m == 0.8
def test_kit_summary_and_warning_shown(...)
def test_build_sheet_offered_for_download(...)
```

- [ ] **Step 2–4:** `.venv/bin/python -m pytest tests/test_ui.py -q` FAIL → implement → PASS.
- [ ] **Step 5:** Commit `feat: laser kit settings and results in the app`.

---

### Task 12: Docs and full suite

**Files:** Modify `REQUIREMENTS.md` §4.3 item 2 (spar rule → stock-tube selection with flight/hover bending check and fit; kit outputs), `README.md` (run-files table: `kit/`; one paragraph on building the kit and the settings).

- [ ] **Step 1:** Edit the docs.
- [ ] **Step 2:** Run `.venv/bin/python -m pytest -q` — Expected: all pass (≥ 672 + new), 13 skipped or fewer.
- [ ] **Step 3:** Run `.venv/bin/python -m pytest --cov=src/tools/fabrication --cov-report=term-missing tests/test_fabrication_*.py -q` — Expected: ≥ 80 %.
- [ ] **Step 4:** Commit `docs: laser-cut kit in the SRS and README`.
