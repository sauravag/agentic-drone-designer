# Laser-Kit Wing Stiffness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The laser-kit wing spar is checked for tip deflection and aileron reversal, not only strength. When the
tube alone reverses too early, the kit adds a balsa leading-edge D-box. The D-box is reported as area and mass, like
the covering. When even that is not enough, the kit is refused with a message naming the shortfall.

**Architecture:**
- **New module.** A pure module, `src/tools/fabrication/wing_stiffness.py`, holds:
  - the D-cell torsion geometry;
  - the two checks;
  - `stiffen_spar`, which runs the checks on the strength spar, falls back to the D-box, or raises `KitUnavailable`.
- **Strength sizing stays separate.** `kit_spar_plan` (strength) is unchanged, because `kit_pod`'s diameter search
  calls it. A new `build_kit.stiffened_spar_plan` wraps it, and `build_kit` and the test fixtures use the new one.
- **Consumers.** `SparPlan` gains `stiffness` and `dbox` fields, which the kit consumers read:
  - the wing ribs are inset ahead of the D-box web;
  - the D-box becomes a covering entry with mass;
  - the build sheet gains a "Wing stiffness" section.

**Tech Stack:** Python 3.12, pydantic v2, numpy, build123d, pytest.

**Spec:** the Design section below. It is binding and records the user's decisions of 2026-11-20 and 2026-11-21.
The user asked for no separate design doc.

## Design (the spec)

**Why.** The kit's spar checks are strength-only (`tubes.py`). In an open-rib, film-covered wing, all the twist and
bending stiffness is the one carbon tube's. A strong-enough wing can still twist enough to reverse the ailerons, or
bend enough to wreck the film and the hinges. Today only the lift booms have a stiffness check (R22,
`tubes.lift_boom_tube`).

**User decisions (binding):**
- **R43 aileron reversal:** the reversal speed must be greater than `1.15 × V_NE`, with `V_NE = 1.4 × cruise speed`
  (`VNE_TO_CRUISE`). The margin started at 1.2 and was lowered to 1.15 on 2026-11-21: with the D-box placed as built,
  g2 reaches 36.1 m/s against the 37.0 m/s that 1.2 would need.
- **R44 tip deflection:** at most `5 %` of the semi-span at `1 g` (elliptic lift, W/2 per half wing). This was
  changed on 2026-11-21 from "10 % at limit load": the correct physics put g2 at 245 mm against 181 mm even with the
  D-box.
- **Settings:** a new carbon tube shear modulus setting, default `5 GPa`.
- **R45 fallback:** a `1.5 mm` balsa hand-wrapped leading-edge D-box plus a spar shear web, reported as area and mass
  like the covering.
- **The D-box counts for torsion only (2026-11-21).** Its sheet is never credited with bending stiffness: it
  buckles near balsa's crushing stress, and it is not anchored at the plug-in root. So a deflection failure has no
  fallback: the kit is refused.

**My calls (ledger them):**
- **Cruise speed reaches the kit as `SizingOutput.v_cruise_ms: Optional[Positive] = None`.**
  - `mission_solver` always sets it.
  - It is optional because `tests/data/flight_control/{g1,g2}/sizing.json` and archived `DesignRecord`s (format 1)
    reload `SizingOutput`, and must keep validating.
  - When it is `None`, `stiffened_spar_plan` raises `KitUnavailable("the design has no cruise speed (it predates
    the wing stiffness checks); re-run it to build a kit")`.
  - `VNE_TO_CRUISE` moves to `src/tools/sizing/aero_model.py`. `flight_model` imports it from there.
- **Structure:** Euler-Bernoulli bending and St-Venant/Bredt torsion.
  - The wing is clamped at `spar.joint_y_m`.
  - The section is stepped by spar segment, using the inboard member at a step.
  - Both checks integrate with the midpoint rule, `STIFFNESS_INTEGRATION_STEPS = 400` strips from the joint to the
    tip, using cumulative sums (O(N)).
- **Tip deflection (R44):** `δ = ∫_joint^s M(y)·(s − y) / (E_c I(y)) dy`.
  - `M = elliptic_moment_nm(mtow, TIP_DEFLECTION_LOAD_FACTOR = 1.0, s, y)`.
  - The limit is `TIP_DEFLECTION_FRACTION = 0.05 × layout.wing.semi_span_m`.
  - Only the tube counts.
- **Reversal (R43):** strip theory with the spar on the aerodynamic centre (`SPAR_CHORD_STATION == QUARTER_CHORD`).
  - A test pins that equality, so moving the spar breaks a named test.
  - With the spar on the AC, the twist comes only from the aileron's `C_m,δ` and is linear in `q`. The rolling
    effectiveness is therefore `q·A − q²·B`, so `q_R = A / B` with no iteration.
    - `A = Σ_aileron c·C_L,δ·y·dy`.
    - `B = −Σ_all c·a·θ₁·y·dy`.
    - `θ₁` is the twist per unit q, the cumulative midpoint sum of `T/GJ`.
    - `T(y)` is the tip-inward sum of `c²·C_m,δ` over aileron strips.
    - The roll arm `y` is measured **from the aircraft centreline**.
  - **Lift slope:** Helmbold, `a = 2π AR / (2 + sqrt(AR² + 4))`, with `AR = sizing.aspect_ratio`.
  - **Aileron (thin-airfoil flap theory):** at `E = AILERON_CHORD_FRACTION` (0.25),
    - `θ_h = acos(2E − 1)`;
    - `C_L,δ = (a/π)(π − θ_h + sin θ_h)`;
    - `C_m,δ = −(a/2π)·½ sin θ_h (1 − cos θ_h)`.
  - **Aileron span:** `AILERON_ETA` (0.55–0.95) × semi-span.
  - **Speed:** `V_R = sqrt(2 q_R / ρ_SL)`, an equivalent airspeed, compared with `1.15 × 1.4 × v_cruise_ms`. This is
    conservative, because cruise speed is a true airspeed at altitude; say so in the module docstring.
  - **Limitations, named in the module docstring:**
    - the D-cell's shear centre lies a little ahead of its web, so with the D-box the elastic axis sits slightly
      forward of the AC (a small divergence tendency, not modelled);
    - flutter and gusts are not modelled.
- **D-box torsion (R45):**
  - The D-cell is the airfoil contour from the web (upper) round the nose to the web (lower), closed by the vertical
    shear web. Its nose is the LE stock, which the sheets butt onto.
  - The web stands at `x_web = SPAR_CHORD_STATION·c − (hole/2 + MIN_WEB_M + DBOX_HOLE_ALLOWANCE_M)`:
    - `hole = member OD + fit_clearance_m`;
    - `DBOX_HOLE_ALLOWANCE_M = 0.0005` covers the oblique-hole growth, so the rib keeps `MIN_WEB_M` round the real
      hole.
  - The same function, `dbox_web_x_m(chord_m, od_m, settings)`, places the rib inset's end and the model's web, so
    the two can't drift apart.
  - **Bredt:** `GJ_box = 4 A_m² G_b t / p`, added to the tube's `G_c J`, where:
    - `A_m = area_ratio·c² − ½ t p` (the midline);
    - `p = (arc_ratio + web_ratio)·c`;
    - `t = dbox_sheet_thickness_m`.
- **Selection (`stiffen_spar`):**
  - If the tube fails deflection → `KitUnavailable` (no fallback).
  - If the tube passes both checks → no D-box.
  - If the tube fails reversal → retry with the D-box. If that passes, the D-box is fitted. If not →
    `KitUnavailable`.
  - Every message names each failing check with its value and limit, in the D18 style:
    - `"tip deflection {d:.0f} mm at 1 g (limit {lim:.0f} mm, 5% of the semi-span)"`;
    - `"aileron reversal at {v:.1f} m/s (needs above {req:.1f} m/s = 1.15 x V_NE)"`.

    These are prefixed with `"the wing spar is too flexible"` (tube only) or with
    `"the wing is too flexible even with the {t_mm:.1f} mm balsa D-box"`.
  - No stiffer-root-tube search: the spars are depth-limited, so none fits, and a bigger root would change the
    socket and the pod (review finding 6).
- **Settings (four new `FabricationSettings` fields):**
  - `carbon_shear_modulus_pa = 5e9`;
  - `balsa_shear_modulus_pa = 0.13e9` (Wood Handbook, G_LT/E_L ≈ 0.037 at E_L 3.4 GPa for balsa of about
    160 kg/m³);
  - `dbox_sheet_thickness_m = 0.0015`.

  The D-box areal mass is derived as `dbox_sheet_thickness_m × stringer_density_kg_m3` (the kit's balsa density),
  not a separate setting.
- **D-box outputs:**
  - **Covering:** a `CoverSurface` with kind `"dbox"`, label `"dbox_wing"`.
    - Its surface is the planform strip from the LE to the root web fraction, on both panels, from the joint to the
      tip.
    - Its `area_m2` is the material area, `2 × Σ (arc_ratio + web_ratio)·c·dy`.
  - **Mass:** that area × the areal mass.
    - It is glued wood, so it also joins the glue-fraction base. This differs from film, which is ironed on.
  - **Ribs:** the wing ribs (the root doubler included) are inset by `t` for `x < x_web`. The tail ribs are not.
  - **Build sheet:** a new "Wing stiffness" section after "Covering", giving:
    - tip deflection against its limit, and reversal speed against its requirement;
    - "D-box: yes/no", with its area and mass when fitted;
    - the build note (Task 4);
    - "Wing stiffness: not checked" when the kit's spar has no report.

**Non-goals:** flutter, divergence, gust loads, tail-surface stiffness, spar caps or glass cloth, D-box bending
credit, changing the strength torsion speed (R30 still uses V_A; using V_NE there is a follow-up to offer).

**Expected outcome** (probe `scratchpad/stiff/probe4.py`, 2026-11-21, the model above with `_fitted_layout`):

| | g1 | g2 |
|---|---|---|
| Tip deflection at 1 g | 44.0 mm (limit 49.5 mm) | 79.3 mm (limit 90.5 mm) |
| Reversal, tube only | 29.1 m/s (needs 29.0 m/s) | 33.1 m/s (needs 35.4 m/s) |
| Reversal, D-box | 34.3 m/s | 36.0 m/s |

So g1 passes on the tube alone, by under 0.5 %. g2 needs the D-box and passes by 1.5 %. Because g1's margin is
within numerical noise, its test pins "passes both" and not whether the D-box is fitted.

## Global Constraints

- SI units; type hints everywhere; no magic numbers (named `Final` constants); frozen dataclasses and pydantic
  models, never mutated (`dataclasses.replace`).
- Functions < 50 lines, files < 800 lines (`tubes.py` is 541 now, so new logic goes in `wing_stiffness.py`).
- TDD: write each test and watch it fail before writing its code; 80 %+ coverage on new code.
- `src/tools/fabrication/*` must not import `flight_control`, `agent`, `ui` or `pipeline`.
- New settings are `Field(DEFAULT_<FIELD>, gt=0.0, description=…)` with `DEFAULT_<FIELD>` constants
  (`test_every_default_is_a_named_constant` in `tests/test_fabrication_settings.py` enforces the naming).
- Exact values:
  - `carbon_shear_modulus_pa` 5e9, `balsa_shear_modulus_pa` 0.13e9, `dbox_sheet_thickness_m` 0.0015;
  - `REVERSAL_MARGIN` 1.15, `VNE_TO_CRUISE` 1.4;
  - `TIP_DEFLECTION_FRACTION` 0.05, `TIP_DEFLECTION_LOAD_FACTOR` 1.0;
  - `DBOX_HOLE_ALLOWANCE_M` 0.0005, `STIFFNESS_INTEGRATION_STEPS` 400.
- Commits end with:
  `Co-Authored-By: WOZCODE <contact@withwoz.com>`
  `Claude-Session: https://claude.ai/code/session_01RJfaxdHYf7KowUaPPCoa8G`
- Host test command: `.venv/bin/python -m pytest -q -x`.

## Review Focus

1. **Thin margins.** g1's tube-only reversal passes by 0.45 % and g2's D-box reversal by 1.5 %.
   - Don't pin the D-box state for g1.
   - For g2, pin "D-box fitted and passes".
   - Changes to the integration or geometry that move g2 past its margin are findings, not test edits.
2. **The rib inset against the spar-hole web.** Every inset wing rib must still pass `_checked` / `keeps_web`. The
   inset ends at `dbox_web_x_m`, not at the spar station. Tasks 2 and 4 test real g2 kit ribs.
3. **Mass.** The D-box reaches `kit_mass` exactly once, at `area × t × stringer_density_kg_m3`, and is in the glue
   base. The test uses non-default settings, so a missing kind mapping can't pass by coincidence.
4. **`kit_pod` stays strength-only.** `kit_spar_plan` must not run the stiffness checks: a stiffness failure there
   would be misread as "no wing box fits".
5. **Old `SizingOutput` records.** They must still validate (`v_cruise_ms` absent), and the kit refuses them with
   the re-run message.

---

### Task 1: Inputs: cruise speed, V_NE constant, settings

**Files:**
- Modify:
  - `src/schemas/state.py` (SizingOutput);
  - `src/tools/sizing/mission_solver.py:294` (`_to_output`);
  - `src/tools/sizing/aero_model.py`;
  - `src/tools/flight_control/flight_model.py:39`;
  - `src/schemas/fabrication.py` (defaults ~line 104-120, fields after `carbon_modulus_pa`);
  - `src/tools/fabrication/tubes.py:14-16` (rewrite the R30 limitation so it is true: `SizingOutput` now has
    `v_cruise_ms`, but the strength torsion still uses `max(V_A, loiter)`).
- Test: `tests/test_sizing.py`, `tests/test_schemas.py`, `tests/test_fabrication_settings.py`

**Interfaces — Produces:**
- `SizingOutput.v_cruise_ms: Optional[Positive] = None`;
- `aero_model.VNE_TO_CRUISE`;
- settings `carbon_shear_modulus_pa`, `balsa_shear_modulus_pa`, `dbox_sheet_thickness_m`.

- [ ] **Step 1: Failing tests**

```python
def test_sizing_output_carries_the_requested_cruise_speed():          # test_sizing.py
    assert size_aircraft(CASES["g1"]).v_cruise_ms == CASES["g1"].cruise_speed_ms

def test_vne_ratio_is_shared_from_the_aero_model():                    # test_sizing.py
    from src.tools.flight_control import flight_model
    from src.tools.sizing.aero_model import VNE_TO_CRUISE
    assert VNE_TO_CRUISE == 1.4 and flight_model.VNE_TO_CRUISE is VNE_TO_CRUISE

def test_old_sizing_records_without_cruise_speed_still_validate():     # test_schemas.py
    # a valid SizingOutput dict built as the file's other tests do, without v_cruise_ms -> v_cruise_ms is None

def test_stiffness_settings_defaults():                                # test_fabrication_settings.py
    s = FabricationSettings()
    assert (s.carbon_shear_modulus_pa, s.balsa_shear_modulus_pa, s.dbox_sheet_thickness_m) == (5e9, 0.13e9, 0.0015)

@pytest.mark.parametrize("field", ["carbon_shear_modulus_pa", "balsa_shear_modulus_pa", "dbox_sheet_thickness_m"])
def test_stiffness_settings_reject_non_positive(field):
    with pytest.raises(ValidationError):
        FabricationSettings(**{field: 0.0})
```

- [ ] **Step 2:** Run `.venv/bin/python -m pytest -q tests/test_sizing.py tests/test_schemas.py tests/test_fabrication_settings.py`.
  Expected: the new tests FAIL.
- [ ] **Step 3: Implement.**
  - The setting descriptions:
    - "Carbon tube / rod shear modulus (Pa), for spar torsional stiffness (R43)";
    - "Balsa sheet in-plane shear modulus (Pa), for the leading-edge D-box torsion (R45)";
    - "Leading-edge D-box balsa sheet thickness (m); its mass uses the stringer (balsa) density (R45)".
  - Move `VNE_TO_CRUISE` and its comment to `aero_model`. `flight_model` keeps the name importable.
- [ ] **Step 4:** Run `.venv/bin/python -m pytest -q -x`. Expected: all pass.
- [ ] **Step 5: Commit:** `feat: carry cruise speed to the kit and add the wing stiffness settings (R43-R45)`

---

### Task 2: Stiffness model: D-cell, section, deflection, reversal

**Files:**
- Create: `src/tools/fabrication/wing_stiffness.py`
- Test: `tests/test_wing_stiffness.py` (new)

**Interfaces:**
- Consumes:
  - Task 1;
  - `TubeStock` (`second_moment_m4`, `polar_moment_m4`);
  - `AirfoilShape.coordinates` (Selig order: upper surface TE→LE, then lower LE→TE, in x/c and z/c);
  - `SPAR_CHORD_STATION`, `QUARTER_CHORD`, `AILERON_ETA`, `AILERON_CHORD_FRACTION`;
  - `MIN_WEB_M`, `SEA_LEVEL_DENSITY_KG_M3`, `elliptic_moment_nm`, `SparPlan` (all from `tubes.py`);
  - `VNE_TO_CRUISE`.
- Produces:

```python
@dataclass(frozen=True)
class DCell:                      # unit-chord D-cell ahead of a web at x/c = web_fraction
    area_ratio: float             # enclosed area / c^2 (outer contour, closed by the web)
    arc_ratio: float              # contour length web(upper) -> nose -> web(lower), / c
    web_ratio: float              # web height / c

def d_cell(airfoil: AirfoilShape, web_fraction: float) -> DCell: ...
def dbox_web_x_m(chord_m: float, od_m: float, settings: FabricationSettings) -> float: ...
def dbox_gj_nm2(cell: DCell, chord_m: float, settings: FabricationSettings) -> float: ...       # Bredt, midline
def tip_deflection_m(ei_at: Callable[[float], float], moment_at: Callable[[float], float],
                     root_m: float, semi_span_m: float) -> float: ...
def lift_slope_per_rad(aspect_ratio: float) -> float: ...
def flap_derivatives(chord_fraction: float, lift_slope: float) -> tuple[float, float]: ...    # (C_L,d, C_m,d)
def reversal_speed_ms(gj_at: Callable[[float], float], chord_at: Callable[[float], float], root_m: float,
                      semi_span_m: float, aileron_m: tuple[float, float], lift_slope: float,
                      cl_delta: float, cm_delta: float) -> float: ...   # math.inf if B <= 0

@dataclass(frozen=True)
class WingStiffness:
    tip_deflection_m: float
    tip_deflection_limit_m: float
    reversal_speed_ms: float
    reversal_required_ms: float
    dbox: bool
    @property
    def deflection_ok(self) -> bool: ...   # <=
    @property
    def reversal_ok(self) -> bool: ...     # strictly >
    @property
    def passes(self) -> bool: ...

def wing_stiffness(spar: SparPlan, airfoil: AirfoilShape, with_dbox: bool, mtow_kg: float, semi_span_m: float,
                   chord_at: Callable[[float], float], aspect_ratio: float, v_cruise_ms: float,
                   settings: FabricationSettings) -> WingStiffness: ...
```

`wing_stiffness` takes GJ(y) as `G_c J(member)`, plus, with the D-box, `dbox_gj_nm2(d_cell(airfoil, x_web/c), c, s)`
with `x_web = dbox_web_x_m(c, member.od_m, settings)`. EI(y) is `E_c I(member)` only.

- [ ] **Step 1: Failing tests**

```python
def _half_t(x, t=0.12):  # NACA 00xx half thickness
    return 5 * t * (0.2969 * x**0.5 - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

def test_d_cell_of_naca0012_matches_the_analytic_thickness():
    cell = d_cell(load_airfoil("naca0012"), 0.20)
    assert cell.area_ratio == pytest.approx(quad(lambda x: 2 * _half_t(x), 0, 0.20)[0], rel=0.02)
    assert cell.web_ratio == pytest.approx(2 * _half_t(0.20), rel=0.02)
    assert cell.arc_ratio > 2 * 0.20

def test_dbox_web_keeps_the_spar_hole_web():
    s = FabricationSettings()
    x = dbox_web_x_m(0.30, 0.014, s)
    assert x == pytest.approx(0.25 * 0.30 - ((0.014 + s.fit_clearance_m) / 2 + 0.002 + 0.0005))

def test_dbox_torsion_is_bredt_with_the_midline_area():
    s, c = FabricationSettings(), 0.25; t = s.dbox_sheet_thickness_m
    p = (0.55 + 0.12) * c; a_m = 0.03 * c**2 - 0.5 * t * p
    assert dbox_gj_nm2(DCell(0.03, 0.55, 0.12), c, s) == pytest.approx(4 * a_m**2 * s.balsa_shear_modulus_pa * t / p)

def test_tip_deflection_matches_uniform_load_cantilever():
    ei, w, L = 120.0, 30.0, 0.9
    assert tip_deflection_m(lambda y: ei, lambda y: w * (L - y) ** 2 / 2, 0.0, L) == pytest.approx(
        w * L**4 / (8 * ei), rel=1e-3)

def test_tip_deflection_from_an_offset_root_matches_tip_load_cantilever():
    ei, p, r, s = 80.0, 10.0, 0.1, 1.0
    assert tip_deflection_m(lambda y: ei, lambda y: p * (s - y), r, s) == pytest.approx(
        p * (s - r) ** 3 / (3 * ei), rel=1e-3)

def test_flap_theory_at_quarter_chord_aileron():
    cld, cmd = flap_derivatives(0.25, 2 * math.pi)
    assert cld == pytest.approx(3.8264, abs=1e-3) and cmd == pytest.approx(-0.6495, abs=1e-3)

def test_helmbold_lift_slope():
    assert lift_slope_per_rad(10.0) == pytest.approx(2 * math.pi * 10 / (2 + math.sqrt(104)))

def test_reversal_matches_closed_form_for_a_uniform_full_span_aileron():
    gj, c, s, a = 40.0, 0.2, 1.0, 5.0
    cld, cmd = flap_derivatives(0.25, a)
    q_r = 12 * cld * gj / (5 * a * c**2 * abs(cmd) * s**2)
    v = reversal_speed_ms(lambda y: gj, lambda y: c, 0.0, s, (0.0, s), a, cld, cmd)
    assert v == pytest.approx(math.sqrt(2 * q_r / SEA_LEVEL_DENSITY_KG_M3), rel=2e-3)

def test_reversal_closed_form_with_an_offset_root_uses_the_centreline_arm():
    # theta = q c^2 Cmd f(y)/GJ, f = s(y-r) - (y^2-r^2)/2; zero of ∫_r^s c(a theta + Cld) y dy:
    # q_R = Cld (s^2 - r^2)/2 * GJ / (a c^2 |Cmd| ∫_r^s f(y) y dy)   (integral by scipy quad in the test)
    ...

def test_reversal_speed_scales_with_the_root_of_torsional_stiffness(): ...   # 4x GJ -> 2x V_R

def test_spar_sits_on_the_aerodynamic_centre():   # the reversal model's assumption
    assert SPAR_CHORD_STATION == QUARTER_CHORD

def test_wing_stiffness_uses_the_user_limits(g1_kit_inputs, designs):
    layout, airfoil, spar, settings = g1_kit_inputs; sizing = designs["g1"][0]
    w = wing_stiffness(spar, airfoil, False, sizing.mtow_kg, layout.wing.semi_span_m, layout.wing.chord_at,
                       sizing.aspect_ratio, sizing.v_cruise_ms, settings)
    assert w.tip_deflection_limit_m == pytest.approx(0.05 * layout.wing.semi_span_m)
    assert w.reversal_required_ms == pytest.approx(1.15 * 1.4 * sizing.v_cruise_ms)
    assert not w.dbox

def test_dbox_raises_reversal_speed_and_leaves_deflection(g2_kit_inputs, designs): ...
```

- [ ] **Step 2:** Run `.venv/bin/python -m pytest -q tests/test_wing_stiffness.py`. Expected: FAIL (no module).
- [ ] **Step 3: Implement.**
  - `d_cell` splits the contour at its minimum-x point, then interpolates the upper and lower surfaces at
    `web_fraction`. It keeps the points ahead of that station, then computes the shoelace area (closed by the web),
    the arc length without the web, and the web height.
  - The rest follows the Design section.
- [ ] **Step 4:** Run the same command. Expected: PASS.
- [ ] **Step 5: Commit:** `feat: wing stiffness model: D-cell torsion, tip deflection and aileron reversal (R43-R45)`

---

### Task 3: Selection and wiring

**Files:**
- Modify:
  - `src/tools/fabrication/tubes.py` (`SparPlan` gains trailing fields
    `stiffness: Optional["WingStiffness"] = None` and `dbox: bool = False`, with a `TYPE_CHECKING` import; rename
    `_size` to the public `tube_size_label`);
  - `wing_stiffness.py` (`stiffen_spar`);
  - `build_kit.py` (`stiffened_spar_plan`; `build_kit` uses it);
  - `tests/conftest.py` (`_kit_inputs` uses `stiffened_spar_plan`, "as build_kit plans it").
- Test: `tests/test_wing_stiffness.py`, `tests/test_fabrication_structure.py` (import `_fitted_layout` from
  `tests.cad_helpers`)

**Interfaces — Produces:**
- `stiffen_spar(spar: SparPlan, check: Callable[[SparPlan, bool], WingStiffness], settings: FabricationSettings) -> SparPlan`:
  returns `dataclasses.replace(spar, stiffness=…, dbox=…)` or raises `KitUnavailable`.
- `stiffened_spar_plan(sizing, layout, airfoil, settings) -> SparPlan`.

- [ ] **Step 1: Failing tests**

```python
def test_stiff_tube_gets_no_dbox(): ...                     # fake check passes for the tube -> dbox False, report kept
def test_reversal_failure_falls_back_to_the_dbox(): ...     # tube fails reversal only, dbox passes -> dbox True
def test_deflection_failure_is_refused_without_trying_the_dbox():
    # message starts "the wing spar is too flexible", contains "tip deflection" "mm at 1 g" "limit"; check called once
def test_dbox_that_still_reverses_is_refused():
    # message contains "even with the 1.5 mm balsa D-box" and "aileron reversal at" "1.15 x V_NE"
def test_kit_spar_plan_stays_strength_only(designs): ...    # .stiffness is None and .dbox is False (kit_pod relies on it)
def test_sizing_without_cruise_speed_is_refused(designs):   # model_copy(update={"v_cruise_ms": None}) -> KitUnavailable "re-run"

@pytest.mark.parametrize("case", ["g1", "g2"])
def test_reference_kits_pass_both_stiffness_checks(case, designs):        # test_fabrication_structure.py
    spar = stiffened_spar_plan(sizing, layout, airfoil, SETTINGS)          # _fitted_layout as conftest does
    assert spar.stiffness.passes

def test_g2_needs_the_dbox(designs):
    assert stiffened_spar_plan(...g2...).dbox                             # Expected outcome table
```

- [ ] **Step 2:** Run `.venv/bin/python -m pytest -q tests/test_wing_stiffness.py tests/test_fabrication_structure.py`.
  Expected: FAIL.
- [ ] **Step 3: Implement** per the Design section's Selection. `stiffened_spar_plan` refuses `v_cruise_ms is None`
  first, then calls `stiffen_spar(kit_spar_plan(...), check, settings)`, where `check` closes over `wing_stiffness`.
- [ ] **Step 4:** Run `.venv/bin/python -m pytest -q -x`.
  - Expected: all pass, except rib, kit-mass or rib-area pins that change because the g2 fixture now carries a D-box
    spar. Those are fixed in Task 4: run Task 4's failing set and note it in the report.
  - If `test_g2_needs_the_dbox` or the g2 pass fails, stop and report the numbers. Don't touch a limit.
- [ ] **Step 5: Commit:** `feat: stiffness-checked spar with a balsa D-box fallback (R43-R45)`

---

### Task 4: D-box in the kit: ribs, covering, mass, build sheet, docs

**Files:**
- Modify:
  - `src/tools/fabrication/rib_outline.py:59` (`notched_profile(..., inset_m: float = 0.0, inset_end_x_m: float = 0.0)`);
  - `wing_parts.py:_wing_rib` (when `spar.dbox`, pass `inset_m = dbox_sheet_thickness_m` and
    `inset_end_x_m = dbox_web_x_m(chord, member.od_m, settings)`);
  - `kit.py` (`CoverKind` gains `"dbox"`; `Kit.dbox_area_m2`);
  - `kit_mass.py` (areal density by kind; the D-box joins the glue base);
  - `build_kit.py:_covering`;
  - `kit_export.py` (`BUILD_SHEET_SECTIONS`, `_wing_stiffness`, `_covering` line);
  - `REQUIREMENTS.md` (fabrication bullets ~line 179);
  - `docs/superpowers/specs/2026-11-19-laser-cut-airframe-design.md` (settings list ~line 256-274);
  - `README.md` if it lists the kit checks.
- Test: `tests/test_fabrication_parts.py` (ribs), `tests/test_fabrication_kit.py`, `tests/test_fabrication_export.py`

- [ ] **Step 1: Failing tests**

```python
def test_inset_rib_shrinks_only_ahead_of_the_web():
    # naca2412, c 0.25: thickness at x = 0.10 c drops by ~2 t; outline aft of inset_end_x_m identical to the plain one
def test_every_g2_wing_rib_keeps_its_spar_web(g2_kit_inputs):
    # g2 spar has dbox: wing_ribs(*g2_kit_inputs) builds (no KitUnavailable); every WR-/WL- spar hole keeps MIN_WEB_M
def test_tail_ribs_are_not_inset(...): ...
def test_dbox_covering_area_and_mass():
    # g2 kit with FabricationSettings(stringer_density_kg_m3=200.0): area == 2 * Σ (arc+web) c dy;
    # "dbox_wing" mass == area * t * 200; glue item == glue_fraction * (structure + dbox)
def test_kit_without_dbox_has_no_dbox_covering(...): ...
def test_build_sheet_has_a_wing_stiffness_section(...):
    # headings (..., "Covering", "Wing stiffness", "Hardware"); deflection + limit, reversal + requirement,
    # "D-box: yes", area and mass, and the build note
def test_build_sheet_says_not_checked_without_a_report(...): ...
```

Update the `BUILD_SHEET_SECTIONS` copy in `tests/test_fabrication_export.py:267`.

- [ ] **Step 2:** Run the three test files. Expected: the new tests FAIL.
- [ ] **Step 3: Implement.**
  - **`notched_profile` with `inset_m > 0`:** the result is `(airfoil ∩ {x ≥ e}) ∪ (nose ∩ {x < e})`, with
    `e = inset_end_x_m`. The `nose` piece is the airfoil cut at `e + 2·inset_m` and offset by `−inset_m`
    (`bd.offset(face, amount=-inset_m)`). After that comes the LE notch and TE cut, as now. Never offset the whole
    airfoil: an inward offset of the sharp trailing edge self-intersects.
  - **Covering:** the `_covering` D-box surface follows `_planform_face`, generalised to the fraction pair
    `(LE_FRACTION, dbox_web_x_m(c_root, root.od_m)/c_root)`, from `spar.joint_y_m` to the tip, both panels.
  - **Build note** (the thickness comes from settings):
    "Leading-edge D-box (R45): glue {t} mm balsa sheet, grain spanwise, over the inset rib noses on the top and
    bottom of each panel, from the leading-edge stock back to the step in the ribs ahead of the spar. Butt the
    sheets onto the leading-edge stock, which closes the box. Glue a {t} mm balsa shear web, grain vertical, across
    each rib bay at that step. Sand the leading-edge stock and the sheet edges to blend before covering."
  - **Docs:** REQUIREMENTS gets one bullet: the two checks with their limits, the D-box fallback (torsion only), the
    refusal and the "Wing stiffness" section. The spec settings list gains the three settings with R43-R45.
- [ ] **Step 4:** Run `.venv/bin/python -m pytest -q -x`. Expected: all pass. Ledger each updated pin (kit mass,
  rib areas) with its reason. Never loosen a structural check.
- [ ] **Step 5:** Run the OpenVSP Docker end-to-end suite (the repo's Docker test command; see
  `docker-compose.yml`). Expected: all pass except the known skip.
- [ ] **Step 6: Commit:** `feat: D-box ribs, covering and mass, and the wing stiffness build-sheet section (R45)`
