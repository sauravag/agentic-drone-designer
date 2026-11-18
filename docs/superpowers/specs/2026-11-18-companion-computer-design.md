# Avionics Stack Choice (Cube Orange+ / ARK Jetson PAB), Optional Jetson, Visible Avionics Assumptions — Design

**Date:** 2026-11-18 · **Status:** draft for approval (revision 2: flight-controller choice added)

## Intent

The aircraft are designed for autonomous operation, so the avionics must be able to carry an NVIDIA Jetson
Orin NX. The designer picks the **avionics stack** per design:

1. **CubePilot Cube Orange+** flight controller. It has no onboard compute; a **Jetson Orin NX on its own carrier
   board** can be added as a separate companion computer, and that is **optional** (the designer's setup:
   **0.2 kg, 25 W**).
2. **ARK Jetson PAB Carrier + ARKV6X + Jetson Orin NX 16 GB** — one board stack with the flight controller and
   the Jetson both mounted on it. The Jetson is **always** part of this stack.

The user also asked what the avionics assumptions are and wants them "listed out somewhere". Today they are
four hard-coded constants that appear nowhere in the app or the report:

| | Today | Where |
|---|---|---|
| Avionics mass | 0.3 kg + 0.01 × MTOW | `src/tools/sizing/mass_model.py:30-31` |
| Avionics power (whole flight) | 10 W + 0.4 W/kg × MTOW | `src/tools/sizing/mission_profile.py:20-21` |

Why it matters: 25 W is 2-3× the whole current avionics allowance and is drawn for the entire flight, and the
two stacks differ by ~0.1 kg with a Jetson fitted. The design must account for both.

## Researched SWaP-C (2026-11-18)

| | Cube Orange+ Standard Set (ADS-B carrier) | ARK Jetson PAB Carrier + ARKV6X + Orin NX 16 GB + heatsink |
|---|---|---|
| Mass | **73 g** (retailers; one says 75 g) | **166 g** (ARK: carrier 80 g, +Jetson 28 g, +heatsink 52 g, +ARKV6X 6 g) |
| Size | Cube 38.25 × 38.25 × 22.3 mm; carrier 94.5 × 44.3 × 17.3 mm | Carrier 116 × 72 × 23 mm; Orin NX module 69.6 × 45 mm |
| Power | Rated 14 W incl. peripheral pass-through; own draw not published (small, inside the FC allowance) | Orin NX 16 GB modes 10 / 15 / 25 W (40 W "Super"); carrier needs 5 V ≥ 4 A |
| Reference price (USD) | ≈ $450–546 | Carrier $800 + ARKV6X $400 (bundle $750) + Orin NX 16 GB ($599 1KU launch; ≈ $1,000–1,400 retail) |
| Notes | Triple-redundant IMUs | USA-built, NDAA compliant, 1 W heater |

Sources: arkelectron.com (ARK Jetson PAB Carrier, ARKV6X, ARKV6X Bundle); docs.px4.io (Cube Orange+, ARK
Jetson PAB Carrier); GetFPV, Rotorama, RobotShop, Lumenier (Cube Orange+ Standard Set); NVIDIA developer
forum and blog (Orin NX 16 GB launch price, power modes). Prices move; they are reference values only.

## Goals

- A **"Flight controller"** choice in the requirements form's Configuration section: *Cube Orange+* or
  *ARK Jetson PAB (Orin NX 16 GB)*.
- A **"Jetson companion computer"** checkbox under it, active only for the Cube Orange+; for the ARK stack it
  shows ticked and disabled with the note "built into the ARK stack".
- Both are changeable on any re-run through the review panel and settable from a description.
- An **Avionics** group in the Design assumptions panel showing every avionics value with its default,
  editable.
- The flight controller and (when fitted) the Jetson are named lines in the parts list with mass, power and a
  reference price; their mass goes into the CAD avionics tray (so the CG includes it); the report states the
  stack, whether a Jetson is fitted, and lists every avionics assumption.

## Non-goals

- A catalog of flight controllers or companion computers beyond these two stacks and one Jetson.
- Cost as a design variable: prices are shown for reference only (the catalog has no cost model).
- Jetson power modes as a selector: the designer edits the wattage (default 25 W).
- Autonomy sensors, thermal / cooling analysis, separate CAD solids for the boards.

## Anti-goals

- Old runs silently reinterpreted as having a Jetson or a different flight controller.
- The agent changing the avionics on its own: closure / improvement options never change the flight
  controller or the Jetson.
- Avionics mass computed in two places that can disagree (sizing vs CAD).
- Any change to results of designs with the Cube Orange+ and no Jetson at default assumptions (reference
  designs stay identical).
- An ARK stack without a Jetson, or a Cube design counting two Jetsons.

## Constraints

- Python with type hints, functions < 50 lines, files < 800 lines, no magic numbers, TDD, 80 %+ coverage.
- "Zero LLM Math": every number from deterministic code; the parser only reads a choice and a yes/no.
- SI units internally; assumption fields carry their unit in the name (`_kg`, `_w`).
- `src/schemas/assumptions.py` must not import from `src.tools`.

## Approaches considered

- **A. Choices in the requirements, numbers in the design assumptions (chosen by the user).** The stack and
  the Jetson are choices about the aircraft (like "Runway available"); masses and watts are properties of the
  hardware (like battery Wh/kg). Reuses the form, parser, change set / review panel / chat and assumptions
  panel.
- **B. Everything in the assumptions panel.** The panel and change-set formatting are numeric-only, and a
  description could no longer set it. Rejected.
- **C. A separate "Avionics" section with its own inputs.** A third input kind threaded through re-run, chat,
  change sets and the report. Most work for the same result. Rejected.

## Design

### 1. Schemas

- `OperationalRequirements`:
  - `flight_controller: Literal["cube_orange_plus", "ark_jetson_pab"] = "cube_orange_plus"`
  - `companion_computer: bool = False` — a separate Jetson on its own carrier; meaningful only with the Cube.
  - Property `jetson_fitted: bool` = `flight_controller == "ark_jetson_pab" or companion_computer`.
  - A model validator normalises `companion_computer` to False when the ARK stack is chosen (the Jetson is
    part of the stack), so the stored requirements never claim a second Jetson.
  - **Schema defaults (Cube, no Jetson)** make a requirements.json written before this change load as it
    was, and keep the reference designs in `src/ui/examples.py` exact. New designs start with the Jetson
    ticked from the form and the parser (below).
- `DesignAssumptions` gains eight fields:

  | Field | Default | Bounds | Meaning |
  |---|---|---|---|
  | `avionics_base_mass_kg` | 0.227 | 0.0 – 5.0 | Shared allowance: GPS, radios, power module, wiring |
  | `avionics_mass_per_mtow` | 0.01 | 0.0 – 0.05 | Allowance growth with MTOW |
  | `avionics_base_power_w` | 10.0 | 1.0 – 200.0 | Shared power incl. the flight controller's own draw |
  | `avionics_power_per_kg_w` | 0.4 | 0.0 – 5.0 | Power growth with MTOW |
  | `cube_orange_plus_mass_kg` | 0.073 | 0.03 – 0.5 | Cube + ADS-B carrier |
  | `ark_pab_stack_mass_kg` | 0.166 | 0.08 – 1.0 | ARK carrier + ARKV6X + Orin NX + heatsink |
  | `companion_computer_mass_kg` | 0.2 | 0.02 – 3.0 | Separate Jetson + its carrier + heatsink (Cube only) |
  | `jetson_power_w` | 25.0 | 1.0 – 100.0 | Orin NX power, whichever stack carries it |

  0.227 + 0.073 = 0.3, so Cube without a Jetson reproduces today's mass exactly; power is unchanged too.
  Methods own the formulas so sizing, CAD and the report share one definition:
  - `avionics_mass_kg(mtow_kg, flight_controller, companion_computer)` = allowance + stack mass
    (+ companion mass for a Cube with the Jetson).
  - `avionics_power_w(mtow_kg, jetson_fitted)` = allowance power (+ `jetson_power_w` when fitted).
- `RequirementChanges` gains `flight_controller` and `companion_computer` (Optional); `AssumptionChanges`
  gains the eight fields; `FIELD_LABELS` gains labels for all ten.
- `SizingOutput` gains `avionics_mass_kg` and `avionics_power_w` (what sizing actually used).
- `components.py`: `FlightController(_Component)` and `CompanionComputer(_Component)`, both with
  `power_w: Optional[float]` and `reference_price_usd: str` (display text, e.g. "≈ $450–546");
  `SelectedBOM.flight_controller: Optional[FlightController] = None` and
  `SelectedBOM.companion_computer: Optional[CompanionComputer] = None`, both counted in
  `total_hardware_mass_kg`; `ComponentNeeds` gains `flight_controller` and `companion_computer`.

### 2. Sizing

- `hotel_power_w(req, mtow_kg, assumptions)` = payload power + `assumptions.avionics_power_w(mtow,
  req.jetson_fitted)`; module constants `AVIONICS_POWER_*` are removed.
- `mass_model`: `AVIONICS_BASE_KG` / `AVIONICS_PER_MTOW` are removed; `empty_mass_breakdown` and
  `airframe_mass_kg` take the avionics mass as an argument. The breakdown keeps the key `"avionics"`.
- `mission_solver` threads the assumptions and requirements through and records `avionics_mass_kg` /
  `avionics_power_w` in `SizingOutput`.

### 3. Parts list (matcher)

`select_bom` always adds the flight-controller line, from the assumptions, not the DB:

- Cube: `FlightController(name="CubePilot Cube Orange+ (ADS-B carrier)", mass_kg=cube_orange_plus_mass_kg,
  power_w=None)` — its power is inside the shared allowance; the parts list shows "in avionics allowance".
- ARK: `FlightController(name="ARK Jetson PAB Carrier + ARKV6X + Jetson Orin NX 16 GB",
  mass_kg=ark_pab_stack_mass_kg, power_w=jetson_power_w)`.
- Cube with the Jetson: also `CompanionComputer(name="NVIDIA Jetson Orin NX + carrier + heatsink",
  mass_kg=companion_computer_mass_kg, power_w=jetson_power_w)`.

Names and reference prices live in one constants table in the matcher module (with the research date), not in
the DB. `presenters.bom_rows` gains "Flight controller" and "Companion computer" rows.

### 4. CAD

`components.sliders` and `cad_packer.sized_airframe_mass_kg` read `sizing.avionics_mass_kg` instead of the
constants. Tray mass = `sizing.avionics_mass_kg + SYSTEMS_FRACTION × MTOW` (as today, now holding the chosen
stack and Jetson), so the packer balances the CG with it and the X-ray legend shows the tray's real mass.

### 5. Agent

- The parse draft gains `flight_controller: Optional[Literal[...]]` and `companion_computer: Optional[bool]`.
  The prompt: ARK / PAB / "Jetson carrier with autopilot" → `ark_jetson_pab`; Cube / Cube Orange →
  `cube_orange_plus`; otherwise null. Companion: true for autonomous / onboard AI / a named Jetson or companion
  computer; false only if explicitly excluded; otherwise null.
- `parsing.py`: flight controller null → `cube_orange_plus`; companion null → True (the ARK normalisation in
  the schema then applies).
- The closure / improvement explorers never set either field (enforced by a test). The chat may propose them
  (the designer reviews every change before a re-run).

### 6. Web form and report

- Configuration section: a "Flight controller" select (key `flight_controller`, default Cube Orange+) and the
  "Jetson companion computer" checkbox (key `companion_computer`, ticked on first render; disabled and shown
  ticked with a caption when ARK is selected). Loading an example preset leaves both alone (presets describe
  missions, not hardware).
- Assumptions panel: a new **"Avionics"** group with the eight fields and help text explaining the formula,
  which values apply to which stack, and that the shared allowance includes the flight controller's power.
- Results header: an avionics line, e.g. "Avionics: Cube Orange+ · Jetson Orin NX companion (25 W, 0.2 kg)",
  "Avionics: Cube Orange+ · no Jetson", or "Avionics: ARK Jetson PAB + Orin NX 16 GB (25 W, 0.166 kg)".
  The report's requirements table shows both fields; the design-assumptions table lists the eight new rows
  automatically.

## Testing

- Default assumptions + Cube + no Jetson reproduce today's sizing for every reference design exactly.
- Cube + Jetson raises hotel power by exactly 25 W and avionics mass by 0.2 kg at a fixed MTOW; ARK raises
  them by 25 W and 0.093 kg (0.166 − 0.073); converged MTOW of G1 rises in both cases, less with ARK.
- `avionics_mass_kg` / `avionics_power_w` formulas for all three combinations; bounds of the eight fields.
- ARK + `companion_computer=True` normalises to False and `jetson_fitted` is True (no double count).
- Old `requirements.json` without the fields → Cube, no Jetson.
- BOM: flight-controller line always present with the right name and mass; companion line only for Cube with
  the Jetson; both counted in total hardware mass.
- CAD tray mass uses `sizing.avionics_mass_kg` (host test, no OpenVSP).
- Parser: nulls → Cube + Jetson; "ARK" → ARK; explicit "no companion" → False (fake LLM).
- Explorer: no option ever sets `flight_controller` or `companion_computer`.
- UI (`AppTest`): select and checkbox present; checkbox ticked on first render and disabled under ARK; both
  survive a preset change; Avionics group shown.

## Calls made (builder/designer may revisit)

- Schema defaults Cube + no Jetson; form/parser default Cube + Jetson (old runs stay honest, reference results
  unchanged, new designs start autonomous).
- The shared allowance base drops to 0.227 kg so the Cube becomes an explicit 73 g line with today's totals
  unchanged; the flight controller's own power stays inside the shared 10 W allowance for both stacks.
- One `jetson_power_w` for both stacks (same Orin NX module, same power mode).
- Board masses ride inside the avionics tray rather than as their own CAD solids.
- Reference prices are display-only text with the research date; not editable assumptions.
- `SizingOutput.avionics_mass_kg` / `avionics_power_w` are required; the coming save/open feature owns legacy
  runs.

## Left to the builder

- Exact help text, labels, caption wording and where the avionics line sits in the results header.
