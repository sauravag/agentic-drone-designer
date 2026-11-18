# Selectable Payload Catalogue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user pick the payload from researched NextVision/SIYI gimbals, the generic catalogue, or Custom,
with editable mass and power. Avionics moves into a live requirements box.

**Architecture:**
- **One payload source:** a pure catalogue module, `src/schemas/payload_catalog.py`, holds all payload rows. It
  seeds SQLite and serves the UI, the parser and requirement changes.
- **Requirements:** `OperationalRequirements.payload_model` names the unit. `payload_mass_kg` and `payload_power_w`
  remain the values used, so sizing is untouched.
- **Parts list:** `select_bom` emits the named unit with the entered values. Custom keeps today's generic-row match.
- **UI:** the requirements box is an `st.fragment` inside a bordered container, so model prefill and the ARK
  greying are live.

**Tech Stack:** Python 3.12, pydantic v2, SQLite, Streamlit 1.67 (`st.fragment`, `st.rerun(scope="app")`), pytest
with `streamlit.testing.v1.AppTest`.

**Spec:** `docs/superpowers/specs/2026-11-18-payload-catalog-design.md`

## Global Constraints

- **Pinned results stay unchanged and green:**
  - the reference sizing baseline (`tests/test_avionics.py::test_reference_designs_unchanged_without_jetson`,
    `tests/data/avionics_baseline.json`);
  - the G1/G2 flight-control fixtures (`tests/data/flight_control/`).
- **Custom payloads:** their parts lists stay byte-identical to today's.
- **Catalogue data:**
  - payload research date `"2026-11-18"`;
  - the gimbal rows are exactly the spec's table (19 gimbals);
  - `CUSTOM_PAYLOAD = "custom"`, shown in the UI as "Custom payload";
  - catalogue names are append-only (a rename keeps an alias).
- **Dropdown caption (exact copy):** "Masses and powers are datasheet values (typical power). NextVision masses are
  probably the bare camera, so add your mount if it's separate."
- **Coding rules:**
  - TDD;
  - functions under 50 lines;
  - files under 800 lines. `tests/test_ui.py` is already at 807, so new UI tests go in
    `tests/test_ui_requirements.py`;
  - no magic numbers, frozen pydantic models, SI units, type hints.
- **Theme work stays uncommitted.** Leave the uncommitted theme work out of every commit, staging by hunk:
  - `.streamlit/`;
  - the Dockerfile line;
  - the `app.py` primary-button change;
  - the `tests/test_ui.py` theme tests.
- **Commit messages** end with `Co-Authored-By: WOZCODE <contact@withwoz.com>` and
  `Claude-Session: https://claude.ai/code/session_01RJfaxdHYf7KowUaPPCoa8G`.
- **Tests:** run them with `.venv/bin/python -m pytest`.

## Review Focus

1. **A named model edited back to exactly the catalogue values** must read as not edited: compare with a tolerance
   and don't append "(edited)".
2. **A stale local `catalog.db` built before this change** (no `vendor` column, no gimbal rows) must be reseeded on
   the next run, not fail with a pydantic or SQL error. Task 0 tests this.
3. **Selecting a model and then switching to Custom** keeps the gimbal's numbers rather than resetting to the
   preset. Task 5 tests this.
4. **Switching the example preset** resets `payload_model` to the preset's Custom, along with its numbers. Task 5
   tests this.
5. **A description naming a model with different casing or no vendor** ("zt30", "Siyi ZT-30") resolves to that
   model. Task 3 tests this.

---

### Task 0: Payload catalogue module, seeding, catalogue version

**Files:**
- Create: `src/schemas/payload_catalog.py`
- Modify:
  - `src/schemas/components.py` (`Payload`)
  - `src/tools/database/seed_data.py` (payload rows, nullable columns, `user_version`)
  - `src/tools/database/matcher.py` (`match_payload`: generic rows only)
  - `src/pipeline.py` (`ensure_catalog`)
- Test: `tests/test_payload_catalog.py` (new)

**Interfaces:**
- Produces, in `components.py`:
  - `Payload.vendor: Optional[str] = None`;
  - `Payload.reference_price_usd: Optional[str] = None`.
- Produces, in `payload_catalog.py`:
  - `PAYLOAD_RESEARCH_DATE: Final[str]`;
  - `CUSTOM_PAYLOAD: Final[str] = "custom"`;
  - `PAYLOADS: tuple[Payload, ...]`: the gimbals plus the 16 existing generic rows, unchanged;
  - `payload_spec(name: str) -> Optional[Payload]`: an exact name or alias; `None` for `custom` or an unknown name;
  - `resolve_payload_name(text: str) -> Optional[str]`: case- and punctuation-insensitive, with the vendor prefix
    optional;
  - `payload_choices() -> tuple[str, ...]`;
  - `payload_label(name: str) -> str`: the dropdown text.
- Produces, in `seed_data.py`: `CATALOG_VERSION: Final[int]` and `catalog_version(db_path) -> int`.

- [ ] **Step 1: Write the failing tests** in `tests/test_payload_catalog.py`:
  - **`test_gimbal_rows_match_research`:** parametrised over the spec table's 19 rows (name, sensor_type, mass_kg,
    power_w, dimensions in metres).
    - `payload_spec(name)` equals those values.
    - `vendor` is in `{"NextVision", "SIYI"}`.
    - The SIYI price strings match the spec.
  - **`test_generic_rows_unchanged`:** the 16 generic rows equal the old `seed_data.PAYLOADS` tuples (copied into the
    test as the pin), and each has `vendor is None`.
  - **`test_choices_order`:**
    - 12 NextVision, by ascending mass;
    - then 7 SIYI, by ascending mass;
    - then the 16 generic rows, by ascending mass;
    - then `"custom"` last.
  - **`test_resolve_payload_name`:**
    - `"zt30"`, `"Siyi ZT-30"` and `"SIYI ZT30"` resolve to `"SIYI ZT30"`;
    - `"NextVision Raptor"` and `"raptor"` resolve to `"NextVision Raptor"`;
    - `"Foo 9000"` and `"custom"` give `None`.
  - **`test_payload_label`:**
    - `payload_label("SIYI ZT30") == "SIYI ZT30 — EO/IR, 0.854 kg, 9 W"`;
    - `payload_label("Mini SAR") == "Generic · Mini SAR — 1.80 kg, 60 W"`;
    - `payload_label("custom") == "Custom payload"`.
  - **`test_seeded_db_has_gimbals_with_vendor(tmp_path)`:** after seeding, `load_catalog(db, "payloads")` holds 35
    rows, and `SIYI ZT30` has `vendor == "SIYI"`.
  - **`test_custom_match_never_picks_a_gimbal(tmp_path)`:**
    - `match_payload(db, 0.1)` gives `"Micro EO 2-axis gimbal"` (generic, 0.10 kg), not the 0.095 kg A8 mini or
      any other gimbal;
    - `match_payload(db, 0.5, "eo_ir")` gives `"Compact EO/IR 3-axis zoom"`.
  - **`test_stale_catalog_is_reseeded(tmp_path)`:** create a db and set `PRAGMA user_version = 0`. After
    `pipeline.ensure_catalog(db)`, `catalog_version(db) == CATALOG_VERSION` and the gimbal rows load.
- [ ] **Step 2: Run them; they must fail.**
  - Run: `.venv/bin/python -m pytest tests/test_payload_catalog.py -q`
  - Expected: errors, because `src.schemas.payload_catalog` doesn't exist yet.
- [ ] **Step 3: Implement.**
  - Build the rows in `payload_catalog.py` from a tuple table, constructed the same way `_tuple_rows` does it.
    `seed_data.build_catalog()["payloads"]` becomes `list(PAYLOADS)`. Delete the old `PAYLOADS` and
    `PAYLOAD_FIELDS` from `seed_data`.
  - `_create_table_sql` emits `NOT NULL` only for non-Optional fields. Use the same Optional unwrap check that
    `_sql_type` already does.
  - `seed_database` sets `PRAGMA user_version = CATALOG_VERSION`; start it at 2.
  - `ensure_catalog` reseeds when the file is missing or `catalog_version(db_path) != CATALOG_VERSION`.
  - `match_payload` filters on `p.vendor is None`.
  - `resolve_payload_name` lowercases, strips everything that isn't `[a-z0-9]`, and compares against each gimbal's
    full name and its name without the vendor prefix.
  - Add a module-level `ALIASES: dict[str, str]`, empty for now, which `payload_spec` consults.
- [ ] **Step 4: Run the tests.**
  - Run: `.venv/bin/python -m pytest tests/test_payload_catalog.py tests/test_database.py tests/test_avionics.py -q`
  - Expected: all pass.
- [ ] **Step 5: Commit** with the message `feat: researched NextVision and SIYI gimbals in the payload catalogue`.

### Task 1: Requirement field, parts-list line, CAD

**Files:**
- Modify:
  - `src/schemas/requirements.py`
  - `src/schemas/components.py` (`ComponentNeeds.payload_model`)
  - `src/tools/sizing/mission_solver.py` (`component_needs`)
  - `src/tools/database/matcher.py` (`select_bom`, new `payload_part`)
- Test: `tests/test_payload_catalog.py`

**Interfaces:**
- Consumes: `payload_spec` and `CUSTOM_PAYLOAD` (Task 0).
- Produces:
  - `OperationalRequirements.payload_model: str = CUSTOM_PAYLOAD`. A `field_validator` rejects anything that is
    neither `custom` nor known to `payload_spec`.
  - The property `OperationalRequirements.payload_edited -> bool`: true when the model is named and either
    `abs(mass - spec.mass) > PAYLOAD_EDIT_TOLERANCE` or the same for power. The tolerance is a named constant
    (`1e-6`) in `requirements.py`.
  - `ComponentNeeds.payload_model: str = CUSTOM_PAYLOAD`.
  - `matcher.payload_part(db_path, needs) -> Payload`.
  - `EDITED_SUFFIX: Final[str] = " (edited)"` in `matcher.py`.

- [ ] **Step 1: Write the failing tests.**
  - **`test_requirements_default_custom_and_old_json_loads`:** G1 example JSON without the key loads with
    `payload_model == "custom"`.
  - **`test_unknown_model_rejected`:** raises a `ValidationError` whose message contains `"Foo 9000"`.
  - **`test_payload_edited`:**
    - ZT30 at 0.854 kg / 9 W gives False;
    - ZT30 at 0.9 kg gives True;
    - Custom gives False.
  - **`test_named_model_bom_line(tmp_path)`:**
    - Needs with `payload_model="SIYI ZT30"` and `payload_mass_kg=0.854` give `bom.payload.name == "SIYI ZT30"`,
      mass 0.854, power 9, the catalogued dimensions and `vendor == "SIYI"`.
    - With `payload_mass_kg=0.9` and `payload_power_w=12`, the name is `"SIYI ZT30 (edited)"`, mass 0.9, power 12.
  - **`test_custom_bom_line_matches_today(tmp_path)`:** Custom needs of 1.0 kg give `"EO/IR 30x zoom + LRF gimbal"`
    at 1.40 kg, which is today's match.
  - **`test_component_needs_copies_payload_model`.**
  - **`test_named_model_reaches_cad_payload_box`:** run a G1 sizing with the ZT30 at 0.854 kg via `run_sizing`, then
    `select_bom`. `aero_cad.components.payload(bom, layout).mass_kg` must equal 0.854. Reuse the layout helper
    already in `tests/test_aero_cad.py`.
- [ ] **Step 2: Run them; they must fail.**
  - Run: `.venv/bin/python -m pytest tests/test_payload_catalog.py -q`
  - Expected: the new tests fail on the missing field or attribute.
- [ ] **Step 3: Implement.**
  - **`payload_part`:**
    - **Custom:** `match_payload(db_path, needs.payload_mass_kg, needs.payload_sensor_type, needs.payload_power_w)`,
      exactly as today.
    - **Named model:** `spec.model_copy(update=...)` with the needs' mass and power (the spec power when the needs'
      power is `None`), plus the edited suffix when edited. "Edited" is judged against the spec with the same
      tolerance constant.
  - `select_bom` calls `payload_part`.
  - `component_needs` passes `payload_model=req.payload_model`, and passes `payload_power_w=req.payload_power_w`
    only for a named model. Custom keeps `None`, as today.
- [ ] **Step 4: Run the tests.**
  - Run: `.venv/bin/python -m pytest tests/test_payload_catalog.py tests/test_avionics.py tests/test_database.py tests/test_flight_control.py tests/test_aero_cad.py -q`
  - Expected: all pass, with no fixture edits.
- [ ] **Step 5: Commit** with the message `feat: requirements name the payload model; the parts list carries that unit`.

### Task 2: Requirement changes, explorer, review panel

**Files:**
- Modify:
  - `src/schemas/changes.py`
  - `src/agent/explorer.py`
  - `src/ui/design_loop.py` (`_edit_widget`)
- Test: `tests/test_payload_catalog.py`

**Interfaces:**
- Consumes: `payload_spec`, `payload_choices` and `payload_label`.
- Produces:
  - `RequirementChanges.payload_model: Optional[str] = None`;
  - `FIELD_LABELS["payload_model"] = "Payload"`.

- [ ] **Step 1: Write the failing tests.**
  - **`test_model_change_fills_mass_and_power`:** from G1 (Custom, 1.0 kg / 15 W), a change setting
    `payload_model="SIYI ZT30"` gives 0.854 kg / 9 W through `apply_changes`. `describe` lists the Payload, Payload
    mass and Payload power rows.
  - **`test_model_change_keeps_stated_mass`:** model ZT30 plus `payload_mass_kg=1.0` gives mass 1.0 and power 9.
  - **`test_switch_to_custom_keeps_numbers`:** from ZT30 at 0.854 kg / 9 W, switching to Custom keeps 0.854 kg / 9 W.
  - **`test_explorer_payload_lever_switches_named_model_to_custom`:**
    - `_lever_change` for `payload_mass` on a ZT30 requirement yields `{payload_mass_kg: v, payload_model: "custom"}`
      in the requirements;
    - on a Custom requirement it yields only the mass.
  - **`test_review_panel_payload_model_is_a_selectbox`**, a unit test with no AppTest: `design_loop._choices` stays
    as it is, and a new `design_loop._field_choices("requirements", "payload_model")` equals `payload_choices()`.
- [ ] **Step 2: Run them; they must fail.**
  - Run: `.venv/bin/python -m pytest tests/test_payload_catalog.py -q`
  - Expected: the new tests fail.
- [ ] **Step 3: Implement.**
  - When `changes.payload_model` is set to a named model, `_requirement_values` fills mass and power from the spec
    for whichever of the two the change doesn't set.
  - On apply, `payload_model` is validated through `OperationalRequirements`, as now.
  - The explorer adds `payload_model: custom` when the `payload_mass` lever changes a named model. Either
    `_lever_change` gains the current requirement, or `_lever_probe` adds the key.
  - `_edit_widget` uses `_field_choices`. That returns `payload_choices()` for `payload_model` and falls back to
    `_choices` otherwise, and renders with `format_func=payload_label`.
- [ ] **Step 4: Run the tests.**
  - Run: `.venv/bin/python -m pytest tests/test_payload_catalog.py tests/test_avionics.py tests/test_explorer.py tests/test_design_loop.py -q`.
    If `test_design_loop.py` doesn't exist, use the file that `grep -l design_loop tests/` names.
  - Expected: all pass.
- [ ] **Step 5: Commit** with the message `feat: payload model in requirement changes, explorer and review panel`.

### Task 3: Descriptions name a model

**Files:**
- Modify:
  - `src/agent/agent_schemas.py` (`RequirementsDraft.payload_model`)
  - `src/agent/prompts.py`
  - `src/agent/parsing.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `resolve_payload_name` and `payload_spec`.
- Produces: `RequirementsDraft.payload_model: Optional[str] = Field(default=None, description="The payload's model name if the user names a specific product (e.g. 'SIYI ZT30', 'NextVision Raptor')")`.

- [ ] **Step 1: Write the failing tests.**
  - **`test_named_model_without_mass_uses_catalogue`:** a draft with `payload_model="zt30"`, no `payload_mass` and an
    endurance gives requirements with `payload_model == "SIYI ZT30"`, 0.854 kg and 9 W, without a clarification.
  - **`test_named_model_with_stated_mass_is_edited`:** a ZT30 draft with 1.2 kg gives mass 1.2, power 9 and
    `payload_edited`.
  - **`test_unknown_model_note_and_mass_required`:**
    - a draft with `payload_model="Foo 9000"` and no mass gets a clarification asking for the payload mass;
    - with a mass, the result is Custom and the notes contain "Payload 'Foo 9000' is not in the catalogue; treated
      as a custom payload".
  - **`test_parser_prompt_lists_gimbals`:** every gimbal name appears in `PARSER_SYSTEM`.
- [ ] **Step 2: Run them; they must fail.**
  - Run: `.venv/bin/python -m pytest tests/test_agent.py -q -k "model or gimbals"`
  - Expected: the new tests fail.
- [ ] **Step 3: Implement.**
  - Add `_payload_fields(draft, db_path, notes) -> Optional[dict]`. It returns `None` when there is no mass and no
    resolved model, which the required-fields check relies on. Payload mass stays required only when no model was
    resolved.
  - A resolved model with no stated power uses the spec power. An unresolved model keeps using
    `_payload_power_w`.
  - Keep `to_requirements` under 50 lines.
  - `PARSER_SYSTEM` gains one sentence built from the gimbal entries of `payload_choices()`.
- [ ] **Step 4: Run the tests.**
  - Run: `.venv/bin/python -m pytest tests/test_agent.py -q`
  - Expected: all pass.
- [ ] **Step 5: Commit** with the message `feat: descriptions can name a catalogue payload`.

### Task 4: Display (parts list, report rows, headline)

**Files:**
- Modify:
  - `src/ui/presenters.py`
  - `src/agent/report.py` (`requirement_rows`)
  - `src/ui/app.py` (the `_headline` caption only)
- Test: `tests/test_payload_catalog.py`

**Interfaces:**
- Produces:
  - `presenters.FIELD_LABELS["payload_model"] = "Payload"`;
  - `presenters.payload_display(req) -> str`: the model name or "Custom payload", plus `EDITED_SUFFIX` when edited;
  - `presenters.payload_summary(req) -> str`, for example `"SIYI ZT30 (0.854 kg, 9 W)"`.

- [ ] **Step 1: Write the failing tests.**
  - **`test_bom_rows_show_payload_vendor_price`:** the "Ref. price" cell is `"≈ $7,000"` for the ZT30 line and `"—"`
    for a generic line.
  - **`test_report_rows_show_payload_model`:** the requirement rows contain `("Payload", "SIYI ZT30 (edited)")` for
    an edited ZT30 and `("Payload", "Custom payload")` for G1.
  - **`test_payload_summary`:** returns `"SIYI ZT30 (0.854 kg, 9 W)"`.
- [ ] **Step 2: Run them; they must fail.**
- [ ] **Step 3: Implement.**
  - `requirement_rows` merges in `{"payload_model": payload_display(req)}`.
  - The `bom_rows` payload row uses `reference_price_usd or "—"` for its price.
  - `_headline` adds `"Payload: " + payload_summary(req)` next to the avionics caption.
- [ ] **Step 4: Run the tests.**
  - Run: `.venv/bin/python -m pytest tests/test_payload_catalog.py tests/test_avionics.py tests/test_agent.py -q`
  - Expected: all pass.
- [ ] **Step 5: Commit** with the message `feat: payload model in the parts list, report and results headline`.

### Task 5: Requirements box layout (fragment)

**Files:**
- Modify:
  - `src/ui/app.py`
  - `src/ui/presenters.py` (`NUMBER_FIELDS` drops "Payload")
  - `tests/test_ui.py` (helpers only)
- Create: `tests/test_ui_requirements.py`

**Interfaces:**
- Consumes: `payload_choices`, `payload_label` and `payload_spec`.
- Produces:
  - **widget keys:** `payload_model`, `payload_mass_kg`, `payload_power_w`, `flight_controller`, and
    `companion_computer_cube`/`_ark` (unchanged);
  - **button key:** `submit` (unchanged);
  - **session flag:** `PENDING_DESIGN_KEY = "pending_design"`.

- [ ] **Step 1: Write the failing tests** in `tests/test_ui_requirements.py`. Use AppTest with `ISR_FORGE_OUTPUTS`
  and `ISR_FORGE_DB` set to paths in `tmp_path`, as `tests/test_ui.py` does.
  - **`test_requirements_box_is_not_a_form`:** there is no `st.form` named "requirements". In the `subheader`
    values, "Start from an example" comes before "…or enter the requirements".
  - **`test_model_change_fills_mass_and_power`:** select `SIYI ZT30`.
    `number_input(key="payload_mass_kg").value == 0.854`, and the power is 9.
  - **`test_edited_mass_persists_across_reruns`:** set the mass to 1.0, change the cruise speed and run. The mass is
    still 1.0.
  - **`test_switch_to_custom_keeps_numbers`.**
  - **`test_preset_resets_payload_model`:** select ZT30, then the G2 preset. `payload_model == "custom"` and the mass
    is 3.0.
  - **`test_ark_greys_jetson_in_box`:** select `ark_jetson_pab`. `checkbox(key="companion_computer_ark").disabled`
    is true.
  - **`test_design_aircraft_runs_named_model`:**
    - uncheck aero, select ZT30, click `submit` and run;
    - there is no exception;
    - the parts-list dataframe contains "SIYI ZT30";
    - the headline caption contains "Payload: SIYI ZT30".
- [ ] **Step 2: Run them; they must fail.**
  - Run: `.venv/bin/python -m pytest tests/test_ui_requirements.py -q`
  - Expected: the tests fail.
- [ ] **Step 3: Implement.**
  - **Page order, top to bottom:**
    1. `st.subheader("Start from an example")`;
    2. the preset selectbox with `label_visibility="collapsed"`;
    3. `st.subheader("…or enter the requirements")`;
    4. `_requirements_box()`, decorated with `@st.fragment`, drawn inside `st.container(border=True)`.
  - **Inside the box:**
    - the mission name;
    - columns Payload | Mission | Envelope | Avionics;
    - Configuration;
    - `st.button("Design aircraft", type="primary", key="submit")`. On click, it sets
      `st.session_state[PENDING_DESIGN_KEY] = True` and calls `st.rerun(scope="app")`.
  - **Payload column:**
    - the selectbox `payload_model`, with `format_func=payload_label` and `on_change=_fill_payload`. `_fill_payload`
      writes the spec mass and power, and does nothing for Custom;
    - the two number inputs;
    - the exact caption.
  - **Avionics column:**
    - today's `_avionics_inputs` body, with the caption shortened to "A description sets its own avionics.";
    - delete `_avionics_inputs`.
  - **`main()`:** after the box, run `if st.session_state.pop(PENDING_DESIGN_KEY, False): _run_form(...)`.
  - **Number inputs:** `_number_inputs` iterates `NUMBER_FIELDS`, which now lacks Payload. The payload inputs keep
    their old bounds as `PAYLOAD_FIELDS` in `presenters`.
  - **`tests/test_ui.py` helpers:**
    - `_fill_form_widget` / `_form_app_with` set widget values directly and click `submit`;
    - drop the hardware-first workaround;
    - the file must not grow past 807 lines.
- [ ] **Step 4: Run the tests.**
  - Run: `.venv/bin/python -m pytest tests/test_ui_requirements.py tests/test_ui.py -q`
  - Expected: all pass, the theme tests included.
- [ ] **Step 5: Commit**, staging by hunk to leave the theme work out, with the message
  `feat: payload picker and avionics inside a live requirements box`.

### Task 6: README, full suite, live check

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write the README section.** Add "### Payload catalogue" with:
  - the 19-gimbal table (name, type, mass, power);
  - the research date and sources;
  - the caveat on what each mass includes;
  - a note on Custom.
- [ ] **Step 2: Run the full suite.**
  - Run: `.venv/bin/python -m pytest -q`
  - Expected: all pass, apart from the 13 known skips.
- [ ] **Step 3: Live check.**
  - Rebuild with `docker compose up -d --build ui`.
  - On G1, pick SIYI ZT30, set the mass to 1.0 and design.
  - Expected: the parts list shows "SIYI ZT30 (edited)", the headline caption is present, and the box greys the
    Jetson on ARK.
- [ ] **Step 4: Commit** with the message `docs: payload catalogue in the README`.
