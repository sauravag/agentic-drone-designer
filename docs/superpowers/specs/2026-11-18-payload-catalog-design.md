# Selectable payloads (EO / EO-IR gimbals) and the requirements box layout — design

**Status:** approved in conversation on 2026-11-18. The user chose "A" and then approved the layout sketch with
"I like the design … Just go with it".

## Intent

Today the user types a payload mass and power. Instead they want to **pick a real payload**:
- low-cost, low-weight EO and EO/IR gimbals from NextVision and SIYI. The user wrote "SaiEye", but no vendor of
  that name exists, and every model they had in mind is SIYI;
- the existing generic catalogue rows;
- a **Custom payload**.

Picking a model fills in its researched mass and power, and both stay editable. The requirements box is also
re-laid-out so Avionics is a column inside it, next to Payload, Mission and Envelope.

**Why:** sizing should come from real products. The parts list should show the unit the user meant, not "the
lightest generic row at or above the typed mass".

## Goals

1. The payload dropdown lists the researched gimbals, then the generic rows, then **Custom payload**.
2. Choosing a model fills **Mass (kg)** and **Power (W)** from the catalogue. Both stay editable, and the edits
   stick until the model changes again.
3. Sizing, the parts list and the CAD payload box all use the chosen unit with the entered values.
4. Descriptions can name a model ("carry a SIYI ZT30"). A description that gives only a mass becomes Custom.
5. The page follows the approved layout sketch:
   ```
   Start from an example                      ← larger, bold
   [ G1 electric fixed-wing            ▾ ]

   …or enter the requirements                 ← heading, outside the box
   ┌──────────────────────────────────────────────────────────────────────┐
   │ Mission name [______________________]                                │
   │ Payload            Mission          Envelope         Avionics        │
   │ [SIYI ZT30     ▾]  Endurance (h)    Ceiling (m)      [Cube Orange+ ▾] │
   │ Mass (kg) [0.854]  Radius (km)      Stall (m/s)      ☐ Jetson         │
   │ Power (W) [  9  ]  Cruise (m/s)     …                                 │
   │ Configuration: Airframe ▾  Propulsion ▾  Group ▾  Runway ☐            │
   │                                       [ Design aircraft ]            │
   └──────────────────────────────────────────────────────────────────────┘
   ```
   The Jetson box still greys out as soon as ARK is chosen. Payload mass and power refill as soon as the model
   changes.

## Non-goals

- **NextVision TRIP2/TRIP5/TRIP6 video processors:** these are not gimbals. Add their mass and power through Custom
  or by editing the fields.
- **Peak-power modelling:** sizing uses continuous power. Published peaks (12–26 W) are out of scope.
- **Automatic mount allowances:** catalogue mass is the datasheet figure. The user edits it if their mount adds
  weight (see the caption below).
- **Live prices:** reference prices are display-only, with a research date, the same as avionics.

## Anti-goals

- Changing any sizing result for an existing design or preset. The three reference designs stay pinned.
- A second source of truth for payload mass. `payload_mass_kg` and `payload_power_w` are always the values used.

## Approaches considered

- **A. Model plus editable mass and power (chosen).**
  - Adds `payload_model`. `payload_mass_kg` and `payload_power_w` keep their meaning as the values used.
  - This is the smallest change, and the sizing contract is unchanged.
- **B. Model plus separate optional override fields.**
  - A slightly cleaner record of what was edited.
  - But it threads two more fields through sizing, review, parsing and reports, and gives two ways to state the
    payload's weight.
- **C. Gimbal defaults in Design assumptions, like avionics.**
  - About 30 new fields, and the payload is split across two panels.
  - Rejected.

**Layout:** the requirements box becomes a bordered container drawn in an `st.fragment`, not an `st.form`.
- A form can't react to a widget until it is submitted, so there would be no live greying-out and no live prefill.
- A plain container without a fragment would re-render the whole page on every edit, results included.
- With a fragment, each edit reruns only the box, and **Design aircraft** triggers a full-app rerun.

## Researched catalogue (2026-11-18)

- **Sources:** nextvision-sys.com/compare/ and each camera page, the siyi.biz spec pages, and Reebot's MT11 spec
  page.
- **Power** is the typical/average (continuous) figure.
- **Mass** is the datasheet figure. NextVision doesn't say what its mass covers (probably the bare camera). SIYI
  figures include the mount or plate where noted.
- **Dimensions** are L×W×H in mm. A round body of Ø64 is entered as 64×64.

| Name in catalogue | Type | Mass (kg) | Power (W) | Dims (mm) | Ref. price |
|---|---|---|---|---|---|
| NextVision DragonEye2 | eo_ir | 0.125 | 3.5 | 40×40×65 | quote only |
| NextVision X80 | eo | 0.145 | 4.5 | 53×58×75 | quote only |
| NextVision Starling | eo_ir | 0.180 | 6 | 84×58×82 | quote only |
| NextVision Starling-LRF | eo_ir | 0.230 | 6 | 78×66×91 | quote only |
| NextVision Stinger | eo_ir | 0.230 | 7 | 61×60×91 | quote only |
| NextVision NightHawk2 | eo_ir | 0.250 | 8 | 64×64×94 | quote only |
| NextVision Swift | eo_ir | 0.290 | 6 | 64×67×97 | quote only |
| NextVision NightHawk2-V | eo_ir | 0.320 | 5 | 64×64×95 | quote only |
| NextVision NightHawk2-UZ | eo_ir | 0.320 | 6 | 64×64×94 | quote only |
| NextVision Raptor | eo_ir | 0.640 | 7 | 85×85×129 | quote only |
| NextVision Condor | eo_ir | 1.500 | 12 | 180×183×211 | quote only |
| NextVision Raven 2.5 | eo_ir | 2.150 | 20 | 154×174×238 | quote only |
| SIYI A8 mini | eo | 0.095 | 5 | 55×55×70 | ≈ $251 |
| SIYI ZT6 | eo_ir | 0.197 (with plate) | 6.5 | 73.5×75×131.5 | ≈ $2,940–3,700 |
| SIYI ZR10 | eo | 0.381 | 4 | 121×101×78 | ≈ $532 |
| SIYI UniPod MT11 | eo_ir | 0.5335 (with board) | 11.4 | 141.5×141.5×169 | not public |
| SIYI UniPod NV10 | eo | 0.628 | 15 (top of 8–15 continuous) | 141.5×141.5×183 | not public |
| SIYI ZR30 | eo | 0.668 (with mount) | 5 | 132×100×180 | ≈ $1,181 |
| SIYI ZT30 | eo_ir | 0.854 (with mount) | 9 | 134.5×140×201 | ≈ $7,000 |

**Where sources disagree:** this happens for the X80 and Raptor dimensions, the NightHawk2-UZ (320 vs 350 g) and
the Raven 2.5 (2.0 vs 2.15 kg). The table uses the compare-page figure or the heavier one.

**Excluded:**
- SIYI A2 mini: a single-axis FPV/agricultural camera, not an ISR gimbal.
- Legacy NextVision DragonEye gen 1 and Colibri.
- TRIP processors.

**Dropdown caption (exact copy):** "Masses and powers are datasheet values (typical power). NextVision masses are
probably the bare camera, so add your mount if it's separate."

## Design

### Catalogue data and lookup
- `Payload` (`src/schemas/components.py`) gains:
  - `vendor: Optional[str] = None` (None means a generic row);
  - `reference_price_usd: Optional[str] = None`.
- The gimbal rows join the generic rows in one researched list, with `PAYLOAD_RESEARCH_DATE = "2026-11-18"`.
- **One pure lookup, with no database, in `src/schemas/payload_catalog.py`.** The UI prefill, the parser and
  requirement changes all use it. It holds:
  - the rows;
  - `CUSTOM_PAYLOAD = "custom"`;
  - `payload_spec(name) -> Optional[Payload]`;
  - `payload_choices() -> tuple[str, ...]`: gimbals sorted by vendor then mass, then generic rows by mass, then
    `custom`.
- `seed_data` seeds the `payloads` table from these same rows.
- The lookup lives under `schemas` because `changes.py`, which is a schema module, needs it and must not import
  `tools`.
- **Stale catalogue:** `seed_database` writes `PRAGMA user_version = CATALOG_VERSION`, and `ensure_catalog`
  reseeds when the file's version differs. An existing local `catalog.db` then picks up the gimbals and the new
  columns.

### Requirements
- `OperationalRequirements.payload_model: str = CUSTOM_PAYLOAD`. It must be `custom` or a name `payload_spec` knows.
- **Default:** Custom, so every existing requirements file, preset and saved run is unchanged and loads as Custom
  with its own numbers.
- `payload_mass_kg` and `payload_power_w` keep their type, bounds and meaning: the values used.
- `payload_edited` is a property. It is true when a model is chosen and the entered mass or power differs from the
  catalogue's (with a small tolerance).

### Sizing
No change. Sizing reads only `payload_mass_kg` and `payload_power_w`, and the pinned reference-design baseline
proves it.

### Parts list and CAD
`select_bom` builds the payload line:
- **Named model:**
  - the catalogue part, with `mass_kg` and `power_w` replaced by the requirement's values;
  - " (edited)" is appended to the name when `payload_edited` is true.
- **Custom: exactly today's behaviour.**
  - `match_payload` keeps its mass, sensor-type and power filters, but restricted to **generic** rows (`vendor is
    None`). Adding gimbals therefore never changes a Custom design's part.
  - The matched row keeps its own catalogue mass and power.
  - `payload_warnings` (`src/pipeline.py`) still flags the gap between that row and what was requested.
  - The pinned G1/G2 parts-list, CAD and flight-control fixtures stay byte-for-byte valid.
  - Showing the *entered* mass for Custom was considered and dropped: it would shift the pinned designs and could
    move critic gates.
- **Named model:** the BOM sensor type comes from the catalogue row. `ComponentNeeds.payload_sensor_type` applies
  only to Custom.
- `payload_warnings` can never fire for a named model, because the line carries the requested values.
- `bom_rows` puts the vendor's reference price in the existing "Ref. price" column. Generic rows show "—".

### Descriptions (parser)
- `RequirementsDraft.payload_model: Optional[str]`, described as: "the payload's model name if the user names a
  specific product (e.g. 'SIYI ZT30', 'NextVision Raptor')".
- `PARSER_SYSTEM` gains one sentence listing the gimbal model names from `payload_choices()`.
- **Matching:** case- and punctuation-insensitive against catalogue names, with or without the vendor prefix.
- **Model found:**
  - mass and power come from the catalogue unless the user stated them;
  - stated values win and mark the model "edited";
  - payload mass is no longer a required fact.
- **Name not in the catalogue:**
  - a note is added: "Payload 'X' is not in the catalogue; treated as a custom payload";
  - today's rules then apply: mass is required, and power comes from a catalogue match by mass or the fallback.
- **No model named:** Custom, behaving exactly as today.

### Requirement changes (review panel, design chat)
- `RequirementChanges.payload_model: Optional[str]`, validated like the requirement field.
- **Changing model** without also setting mass or power: `_requirement_values` fills them from the catalogue.
- **Switching to Custom** keeps the current numbers.
- `describe` shows the implied mass and power rows, following the same pattern as the Jetson.
- The review panel renders `payload_model` as a selectbox over `payload_choices()`.
- **Design explorer:** the `payload_mass` lever (`src/agent/explorer.py`) proposes a lighter payload.
  - For a named model, that suggestion also sets `payload_model` to Custom: a lighter payload is a different unit,
    not an "edited" ZT30.
  - For Custom, nothing changes.

### Loading a requirements file with an unknown model
Validation fails with the unknown name in the message. Catalogue names are append-only: a rename keeps the old
name as an alias in `payload_spec`.

### UI (`src/ui/app.py`)
- **Page order:**
  1. the description form (unchanged);
  2. **"Start from an example"** as a bold subheader;
  3. the preset selectbox, label collapsed;
  4. the subheader "…or enter the requirements";
  5. the requirements box.
- **The requirements box:** an `@st.fragment` drawing `st.container(border=True)`. It contains:
  - the mission name;
  - four columns:
    - **Payload:** model selectbox, Mass, Power and the caption;
    - **Mission**;
    - **Envelope**;
    - **Avionics:** flight-controller selectbox, plus the Jetson box with the existing ARK greying;
  - Configuration;
  - the **Design aircraft** button.
- **Design aircraft** sets a pending-run flag and calls `st.rerun(scope="app")`.
  - `main()` pops the flag, then calls `_run_form` with the widget values from `session_state`, as today.
  - Validation errors and run status render where they do today: below the box, above the results.
- The model selectbox's `on_change` writes the catalogue mass and power into `payload_mass_kg` and
  `payload_power_w`. It writes nothing for Custom.
- The preset loader also sets `payload_model`.
- The avionics caption about descriptions moves into the Avionics column, shortened to: "A description sets its
  own avionics."
- Presets stay Custom with their current numbers (the pinned baselines).
- Delete `_avionics_inputs` (above the form) and its workaround in the test helper.

### Reports and display
- `requirement_rows` and `FIELD_LABELS` gain a "Payload" entry: the model's display name, with " (edited)" when
  edited, or "Custom".
- The results headline caption gains "Payload: SIYI ZT30 (0.854 kg, 9 W)".
- `to_context_json` carries `payload_model` automatically.

## Testing
- **Catalogue:**
  - every researched row matches the table above;
  - `payload_choices()` order;
  - `payload_spec` for a known name, a vendor-less alias and an unknown name.
- **Requirements:**
  - default `custom`;
  - an unknown model is rejected;
  - `payload_edited`;
  - old JSON without the field still loads.
- **Reference designs:** the pinned sizing baseline still matches at rel 1e-9.
- **Parts list:**
  - a named model with its own values;
  - edited values give the " (edited)" name;
  - the named model's mass reaches the CAD payload box;
  - Custom never picks a gimbal row;
  - the pinned G1/G2 flight-control fixtures still pass unchanged.
- **Explorer:** a payload-mass suggestion on a named model switches it to Custom.
- **Catalogue version:** a stale `user_version` triggers a reseed.
- **Parser:**
  - a named model with no mass gets its mass from the catalogue;
  - a named model with a stated mass is edited;
  - an unknown model gets the note and still requires a mass;
  - no model behaves as before.
- **Changes:**
  - a model change fills mass and power;
  - Custom keeps the current numbers;
  - `describe` shows the implied rows.
- **UI (AppTest), in a new `tests/test_ui_requirements.py`,** since `tests/test_ui.py` is already 807 lines:
  - the box has no form;
  - a model change refills mass and power;
  - edited values persist;
  - ARK greys the Jetson box;
  - **Design aircraft** runs a design;
  - the headings appear in order.
- **Live Docker check:** pick a SIYI ZT30 on G1, edit the mass, design, then check the parts list, the CAD and the
  caption.

## Calls made (builder's latitude noted)
- **Power default:** typical/average. For the NV10, the top of its 8–15 W continuous range.
- **Mass:** the datasheet figure, including the mount where SIYI includes it. No automatic mount allowance.
- **Which models:** Condor and Raven 2.5 are kept (they serve Group 2), the A2 mini is dropped, and the TRIP
  processors are excluded.
- **Lookup location:** `src/schemas/payload_catalog.py`, to keep the layer direction.
- **Custom parts list:** behaves as today, matching against generic rows only. The entered-mass correction was
  dropped (builder check): it would shift the pinned designs.
- **Explorer:** a lighter-payload suggestion switches a named model to Custom.
- **Unknown model names in saved files:** fail validation. Catalogue names are append-only.
- **Presets** stay Custom.
- **Left to the builder:** exact widget keys, the details of name matching, and how to test the fragment.
