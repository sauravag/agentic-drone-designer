# Designer in the loop: configurable assumptions, option explorer and design chat

Date: 2026-11-16 · Status: draft for approval · Project: ISR-Forge (`isr-forge/`)

## 1. Intent

Today a run that cannot close simply ends "NOT converged". Run `20261116T030906Z-g1-electric-fixed-wing`
(electric quadplane VTOL, 5 kg payload, 6 h endurance, 100 km radius at 30 m/s, no runway) is the trigger: the
sizing loop spirals past 1,797 kg, the critic can only move four airframe knobs, and it gives up after one
iteration. The real trade-offs (propulsion, endurance, payload, runway, technology assumptions) belong to the
human designer, and the app gives them no way in.

The user wants the app to keep the designer in charge of trade-offs:

1. **Design assumptions are visible and configurable**, with defaults, instead of hidden constants.
2. **When a design does not converge, the app opens a conversation**: first computed options that actually
   close the design, then a chat with an "agentic designer" to discuss and analyse those options and explore
   anything they do not cover, ending in a design the designer accepts.

The app should feel flexible: a failed run is the start of a design discussion, not the end.

## 2. Goals, non-goals, anti-goals

**Goals**
- Every engineering assumption that decides whether a design closes is shown on screen with its default and
  can be changed per run; the values used are saved with the run.
- On non-convergence the designer sees, within seconds, a short list of single changes that make the design
  close, each with its computed result.
- A chat lets the designer ask about options, compare them, and try their own what-ifs; the agent tests
  what-ifs with the real solver before answering.
- Accepting an option or an agreed proposal starts a new full run linked to the previous one, building a
  visible design history.

**Non-goals (this project)**
- Exposing structural calibration constants (wing/fuselage areal densities, systems fractions).
- A separate VTOL battery pack architecture (noted as future work; the C-rate assumption covers most of it).
- Multi-user sessions, persistence across browser sessions, or chat in the CLI (the CLI gets assumptions and
  the option list, not the chat).
- Native LLM tool calling (rejected, see Approaches).

**Anti-goals (failure even if it "works")**
- The LLM inventing or computing an engineering number. Every number shown in options or chat comes from the
  deterministic tools (SRS "Zero LLM Math" rule).
- The agent changing the design without the designer confirming the exact change set.
- Options that are presented as closing but fail when re-run in full (the option must state that it was
  checked with sizing + catalog only; aero/CAD/SITL may still find issues).
- The chat letting a form user change the LLM provider/model (those stay `.env`-only, as decided earlier).

## 3. Constraints

- Python 3.11, type hints, no magic numbers, immutability
  (frozen pydantic models / dataclasses), validate at boundaries, TDD with ≥ 80 % coverage.
- Works identically on both LLM providers (OpenRouter, Anthropic) through the existing `StructuredLLM.parse`
  protocol with strict JSON-schema outputs. No provider-specific features.
- The structured form must keep working with no API key: option cards work without an LLM; only the chat and
  the plain-English explanation need one.
- All LLM-written text shown in Streamlit passes through `report.defang` / `inline`.
- SI units internally; UI may show Wh/kg, %, kg/kWh.

## 4. Approaches considered

**How the chat gets real numbers**
- **A. Check-and-answer loop over structured outputs (chosen).** Each LLM turn returns a fixed schema
  (message, what-ifs to test, optional proposal). The app runs the what-ifs through sizing and feeds results
  back; the LLM may do a few rounds before replying. Reuses `StructuredLLM.parse`, works on any OpenRouter
  model with structured outputs, keeps Zero LLM Math.
- B. Native tool calling. More fluid, but a second implementation per adapter, and OpenRouter tool +
  strict-schema support varies by model. Rejected.

**What accepting a change does**
- **Linked re-run (chosen).** A fresh full run seeded with the changes, recording `parent_run_id`; each run
  stays self-contained and reproducible; the UI shows the lineage.
- Pause and resume the original run with LangGraph checkpoints. Rejected: needs checkpoint storage and keeps a
  run open for the length of a conversation.

**Catalog batteries when the designer changes energy density**
- **Catalog follows the assumption (chosen by the user).** Catalog pack mass and volume are derived from the
  run's specific energy and energy density, so the BOM matches the sizing. Packs become notional when the
  value differs from the default; the report says so.
- Catalog fixed at default, with a mismatch warning. Rejected.

**Interaction style on failure**
- **Computed options first, then chat (chosen by the user, "option C").** Free-text-only chat and
  knob-only control were rejected.

## 5. Part 1 — Design assumptions

### 5.1 Model

New frozen pydantic model `DesignAssumptions` (new module, e.g. `src/schemas/assumptions.py`), every field
with a default equal to today's constant and explicit bounds:

| Field | Default | Bounds | Replaces |
|---|---|---|---|
| `liion_specific_energy_wh_kg` | 200 | 100–500 | `mission_profile.LIION_WH_PER_KG`, `seed_data.PACK_SPECIFIC_ENERGY_WH_KG["li_ion"]` |
| `liion_energy_density_wh_l` | 250 | 100–1000 | `seed_data.PACK_ENERGY_DENSITY_WH_L["li_ion"]` |
| `liion_max_c_rate` | 3 | 1–30 | `LIION_MAX_C` |
| `lipo_specific_energy_wh_kg` | 155 | 80–300 | `LIPO_WH_PER_KG`, catalog lipo |
| `lipo_energy_density_wh_l` | 280 | 100–800 | catalog lipo |
| `lipo_max_c_rate` | 25 | 5–100 | `LIPO_MAX_C` |
| `battery_reserve_fraction` | 0.20 | 0.10–0.40 | `MIN_SOC_RESERVE` for the electric end reserve only; the hybrid buffer pack's SoC floor stays an internal constant (it sizes a hover buffer, not mission endurance) |
| `fuel_reserve_fraction` | 0.10 | 0.05–0.40 | `FUEL_RESERVE` |
| `bsfc_kg_per_kwh` | 0.50 | 0.25–1.20 | `BSFC_KG_PER_J` |
| `drivetrain_efficiency` | 0.68 | 0.40–0.95 | `DRIVETRAIN_EFFICIENCY` |
| `hover_figure_of_merit` | 0.72 | 0.40–0.90 | `FIGURE_OF_MERIT` |
| `lift_motor_efficiency` | 0.85 | 0.50–0.98 | `VTOL_MOTOR_EFFICIENCY` |

The `Battery` component validator range (`components.MIN/MAX_SPECIFIC_ENERGY_WH_KG`, today 80–350) widens to
cover the assumption bounds.

### 5.2 Threading

- `size_aircraft(req, overrides, assumptions=DesignAssumptions())` passes the object down to the mission
  profile and storage functions; module-level constants become the model's defaults only (one source of
  truth). No global mutation.
- Catalog: battery mass and envelope are derived from the run's assumptions at match time (the catalog keeps
  capacity, C-rate, voltage; mass/volume are computed), so sizing and BOM agree. The C-rate filter in the
  matcher uses the assumption's max C for catalog packs of that chemistry.
- Critic and reserve: the designer's reserve is the **floor**. Today `reserve_fraction` is a critic knob that
  may go down (the failed run went 0.20 → 0.15); after this change the critic may only raise it, never below
  the designer's value. The energy-reserve gate (≥ 15 %) is unchanged.
- Persistence: each run writes `assumptions.json`; the report gets a "Design assumptions" table marking
  values that differ from defaults; `run_summary.json` records whether any are non-default.
- CLI: `--spec` JSON may contain an optional `assumptions` object (same fields; missing fields take
  defaults). Natural-language runs use defaults unless the form sets them.

### 5.3 UI

An expander "Design assumptions" on the main page (above both input forms, applying to both), grouped as
Battery (electric), Hybrid buffer pack, Hybrid engine, Efficiencies, with units, per-field help text, a
"Reset to defaults" button, and a caption when any value is non-default. Values persist in the Streamlit
session and are validated through `DesignAssumptions` before a run.

### 5.4 Solver robustness (prerequisite)

Found during design: the electric quadplane case `1 h / 20 km / 30 m/s / 5 kg` ends with "MTOW did not converge
in 1000 iterations" (oscillation), while 22 m/s converges at 46.7 kg. The explorer assumes relaxing a
requirement never makes closure worse, and designers will read an oscillation as "impossible". The fixed-point
iteration must detect oscillation (e.g. sign changes in the MTOW update) and fall back to stronger damping or a
bracketed root find on `mtow_out(mtow_in) - mtow_in`. A genuine spiral still raises `SizingDivergenceError`.

## 6. Part 2 — Option explorer

### 6.1 When, and which goal

The explorer has two modes, chosen from why the run ended:

- **Closure mode** — automatic when a run finishes `not_converged` because of sizing divergence, a catalog
  failure, or a sizing-only gate (energy reserve, group mass). Finds single changes that make the design close
  (§6.2–6.4).
- **Airframe failures** — when sizing closes but the run failed on static margin, L/D or CAD collisions, there
  are no option cards (every requirement relaxation would "close" with zero change). The chat opens
  automatically with the failed gates explained, and its `ChangeSet` allow-list additionally includes the
  airframe knobs (`aspect_ratio`, `tail_volume_h`, `tail_volume_v`, `fuselage_length_factor`, within
  `mission_solver.OVERRIDE_BOUNDS`), applied as starting overrides for the linked re-run.
- **Improvement mode** — on demand ("Explore alternatives") after a converged run. For each lever in §6.3 it
  reports the sensitivity instead of closure: MTOW saved and whether the design drops a DoD group for a
  stated step (builder's choice of step, e.g. endurance − 1 h, payload − 0.5 kg, Li-ion + 50 Wh/kg,
  C-rate + 2 C), ranked by MTOW saved per relative change. Cards read e.g. "Endurance 6 → 5 h: − 3.1 kg".
  The chat is available alongside.

### 6.2 What "closes" means

A candidate **closes** when sizing converges, the catalog match succeeds, **and** the sizing-only critic gates
pass (`critic.evaluate_sizing_gates`: energy reserve ≥ 15 %, MTOW within the group limit). To avoid showing an
option that the critic would immediately bounce, each evaluation starts from the designer's assumptions with
default airframe knobs and applies the reserve floor `max(designer's reserve, gate minimum)` for the powertrain
being evaluated (so a hybrid candidate is sized with a 15 % fuel reserve, not the 10 % default). The linked
re-run carries that same reserve in its change set and starts the critic from default knobs. Options still say
"checked with sizing + catalog only" because aero/CAD/SITL gates are not evaluated.

### 6.3 Candidates

Each candidate is a **single change** to the requirements or the assumptions, evaluated with sizing + catalog
match only (≈ 0.02–0.15 s per sizing call measured on the host), never aero/CAD/SITL:

| Lever | Kind | Search |
|---|---|---|
| Propulsion → series hybrid (or electric if hybrid fails) | discrete | one evaluation |
| Allow runway / fixed-wing instead of VTOL | discrete | one evaluation; the change set sets both `runway_available=true` and `airframe_topology=fixed_wing` (one lever, two fields) |
| Endurance | continuous, ↓ | largest value ≤ requested that closes |
| Combat radius | continuous, ↓ | largest value ≤ requested that closes |
| Payload mass | continuous, ↓ | largest value ≤ requested that closes |
| Cruise speed | continuous, both ways | closure is not monotonic in speed (stall floor below, drag above): coarse scan from `1.2 × stall` to the requested speed and a little above, then bisect at the first transition nearest the request |
| Group constraint | discrete | drop / raise the constraint |
| Li-ion specific energy | continuous, ↑ | smallest value ≤ bound that closes |
| Li-ion max C-rate | continuous, ↑ | smallest value ≤ bound that closes |

The other continuous levers are treated as monotonic (relaxing them never makes closure worse, given the §5.4
solver fix) and use bisection between the bound and the requested value to a stated resolution (builder's
choice, e.g. 0.1 h, 1 km, 0.1 kg, 1 m/s, 10 Wh/kg, 0.5 C). A candidate that does not close anywhere in its
range is shown as "does not close within range" so the designer learns what will not help. A candidate that
closes also reports MTOW, DoD group, battery/fuel mass, and the delta from the request. The whole explorer has a
time cap (builder's choice, target well under 30 s on the host).

Options are ranked: closing options first, by the smallest relative change to the request; then non-closing.
Each option is a typed `ChangeSet` (requirement field updates + assumption field updates), validated through
`OperationalRequirements` / `DesignAssumptions`.

### 6.4 Output

Option cards above the results tabs, each with: the change in plain words ("Endurance 6.0 h → 3.2 h"), the
computed outcome, a "checked with sizing + catalog only" note, and a **Select** button that loads the change set
into a review panel (see 8). Saved with the run as `options.json` and listed in the report. The CLI prints the
option list after a non-converged run; it does not start a chat.

With an LLM configured, one LLM call writes a short explanation of why the design failed and what each option
trades away (prose only, defanged; numbers referenced must come from the computed options — the existing
narrative number-check applies).

## 7. Part 3 — Design chat

### 7.1 Placement and availability

A chat panel below the option cards (Streamlit `chat_input` / `chat_message`). Opens automatically on
non-convergence (closure mode and airframe failures, §6.1), available on demand after any run. Needs an LLM; without one, a caption explains how to
enable it and the option cards still work. History lives in the Streamlit session and is saved to the run
folder as `design_chat.json` (messages, what-ifs, results).

### 7.2 Check-and-answer loop

Each designer message triggers up to N rounds (builder's choice, e.g. 3). In each round the LLM receives a
briefing and returns `DesignerTurn` (strict schema):

- `message: str` — reply to the designer (prose, no new numbers beyond those in results it was given);
- `what_ifs: list[ChangeSet]` (bounded, e.g. ≤ 4) — changes it wants checked before answering;
- `proposal: Optional[ChangeSet]` — a concrete change set it recommends accepting;
- `done: bool` — true when it is ready to answer.

If `what_ifs` is non-empty and `done` is false, the app evaluates them (same evaluator as the explorer) and
starts the next round with the results appended; the final round's `message` is shown with the what-if
results rendered as a small table next to it, so the designer sees the numbers came from the solver.

**Briefing** (built deterministically each round): the current requirements and assumptions, the failed gates
and critic log summary, the option list with results, the chat so far (trimmed to a budget), what-if results so
far, the allowed `ChangeSet` fields with their bounds, and the rules (no arithmetic, cite only given numbers,
never claim a re-run happened).

### 7.3 Safety and validation

- `ChangeSet` fields are an explicit allow-list (requirement fields + assumption fields, plus the airframe
  knobs after an airframe failure, §6.1; not mission name, not LLM settings, not file paths). Out-of-bounds values are rejected and reported back to the LLM as a
  failed what-if, not silently clamped.
- A proposal never runs by itself: it populates the review panel; the designer must press Re-run.
- LLM failures (`LLMError`) show a short error in the chat and leave the options usable.
- Cost: each designer message costs 1–N LLM calls; the UI shows the round count per answer.

## 8. Part 4 — Review panel and linked re-run

- Selecting an option or a chat proposal fills a **review panel** listing each changed field old → new
  (requirements and assumptions); the designer may edit the values before pressing **Re-run**.
- Re-run starts a normal full run (respecting the current aero/CAD and SITL checkboxes) with the merged
  requirements and assumptions, using the structured path (no re-parse of the original prompt). The new run
  records `parent_run_id`, `change_set` and `change_source` (`option`, `chat`, `manual`) in
  `run_summary.json` and in the report.
- The page shows a **design history** strip for the session: run id, status, the change that led to it; any
  earlier run can be reopened in the results view.

## 9. Decided vs left to the builder

**Decided with the user**: assumptions on screen with defaults and configurable (§5); catalog follows the
assumptions (§4); computed options first, then chat (§6–7); check-and-answer loop over structured outputs (§7.2);
linked re-runs (§8); LLM provider/model stay `.env`-only.

**My calls (correctable)**: the assumption list and bounds in §5.1; structural constants excluded; reserve is a
floor for the critic; the lever list in §6.2; options checked with sizing + catalog only; chat also available
after converged runs but auto-opened only on failure; CLI gets assumptions and option list but no chat; solver
oscillation fix included as a prerequisite.

**Decided with the user after the builder check**: airframe-gate failures get the chat (with airframe knobs)
instead of option cards; converged runs get an improvement-mode explorer (§6.1).

**Settled after the builder check (my calls)**: definition of "closes" and the reserve floor in explorer
evaluations (§6.2); the hybrid buffer SoC stays internal (§5.1); runway lever sets two fields; cruise speed is
scanned, other levers bisected (§6.3); corrected the battery-energy test expectation (§10).

**Left to the builder**: module layout; bisection resolutions and the explorer time cap; N rounds and
what-if limits; chat history trimming; exact card and panel layout; how sessions keep multiple runs in memory.

## 10. Testing

- Assumptions: defaults reproduce today's results exactly (all existing sizing/BOM tests unchanged);
  non-default values change sizing and catalog consistently; bounds rejected at the boundary; reserve floor
  respected by the critic.
- Solver: the oscillating case converges; genuine spirals still raise.
- Explorer: on the failed run's requirements, the hybrid option closes (≈ 16.6 kg, Group 2 measured with a
  10 % fuel reserve; recheck at the 15 % floor), reduced-endurance closes, and the battery-energy option closes
  only near the top of its range (measured: spirals at 350 Wh/kg, closes at 500 Wh/kg at ≈ 37 kg) — i.e. it is
  ranked low as a large change; ranking and time cap; every option re-validates.
- Chat: scripted LLM drives what-if rounds, proposals, round limit, invalid change sets, LLM errors; the
  OpenRouter adapter test covers the `DesignerTurn` schema through the real SDK.
- Modes: an airframe-gate failure (e.g. static margin) yields no option cards and a chat whose allow-list includes
  the knobs; a converged run's improvement mode reports MTOW savings and group drops.
- UI: AppTest for the assumptions expander, option cards on a failing design, review panel → linked re-run
  with `parent_run_id`, chat disabled without a key.
