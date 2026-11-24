# agentic-drone-designer

An agent that designs DoD Group 1–3 ISR UAVs from mission requirements: conceptual sizing (battery-electric
or series-hybrid, fixed-wing or quadplane VTOL), a COTS bill of materials, OpenVSP/VSPAERO aerodynamics,
Build123d internal CAD solved for static margin, ArduPilot parameters checked in a headless SITL flight test against two extreme corners of the roll/pitch aerodynamics (±30 % aileron/elevator power and roll/pitch damping, ±20 % inertia, and both static-margin limits, 0.08 and 0.14; yaw and the VTOL rotors are not perturbed, the transition is gated on the nominal flight, and mixed corners are not flown), and a design report. The full specification is in [REQUIREMENTS.md](REQUIREMENTS.md).

**Zero LLM math.** The LLM only reads the request, suggests which design knob to turn, and writes report
prose. Every engineering number comes from deterministic tools; the critic's knob steps are bounded and
rule-table fixes always win on failed gates; any figure in the LLM narrative that is not in the computed
results is flagged in the report.

## Quick start

```bash
cp .env.example .env            # optional: add OPENROUTER_API_KEY (or ANTHROPIC_API_KEY) for natural-language input
docker compose build            # OpenVSP, VSPAERO, ArduPilot arduplane, JSBSim (first build is slow)
docker compose up ui            # web form at http://localhost:8501 (bound to 127.0.0.1)
```

The web form accepts either a description (needs an LLM key) or structured requirements (works without a
key: rule-table critic, template report narrative). Both run the same agent and critic loop. Docker Compose
reads `.env`; when running on the host, `export OPENROUTER_API_KEY=...` in your shell instead.

### Designer in the loop

The designer stays in charge of the numbers the agent would otherwise assume. Four pieces, all in the web form:

**Design assumptions.** A panel above the forms holds the 20 technology constants the sizing chain uses; the
defaults reproduce the historical values exactly.

| Group | Values (default) |
|---|---|
| Li-ion | specific energy 200 Wh/kg, energy density 250 Wh/L, max discharge 3 C |
| LiPo | specific energy 155 Wh/kg, energy density 280 Wh/L, max discharge 25 C |
| Reserves | battery reserve 0.20, fuel reserve (hybrid) 0.10 |
| Propulsion and VTOL | generator BSFC 0.50 kg/kWh, drivetrain efficiency 0.68, hover figure of merit 0.72, lift motor efficiency 0.85 |
| Avionics | allowance 0.227 kg + 0.01 kg per kg MTOW and 10 W + 0.4 W per kg MTOW (GPS, radios, wiring, the flight controller's own power); Cube Orange+ 0.073 kg; ARK PAB stack 0.166 kg; Jetson companion 0.2 kg; Jetson power 25 W |

They apply to both description and form runs, and each has validated bounds. Changing a battery value makes the
catalog battery packs *notional*: capacity and voltage stay catalog values, but pack mass and size are derived
from your specific energy and density, and the report says so. The critic treats your reserve as a floor: it may raise the reserve to pass gate 4 but never
lowers it below the value you set.

**Option explorer.** When a run does not converge on sizing, catalog, reserve or group-limit failures, closure
mode runs automatically and lists single changes that would close the design (requirement relaxations such as
endurance, radius, payload or cruise speed, a different propulsion or airframe, a higher group limit, better
battery assumptions). Options are checked with sizing and catalog only, with no aero, CAD or SITL, so they are fast
but unverified against the airframe gates. They are ranked by how small the change is and stop at a time cap
(unchecked options are marked "not evaluated"). Airframe-gate failures (static margin, L/D) do not get cards:
they open the chat instead. On a converged run, **Explore alternatives** runs improvement mode and shows how
sensitive the design is to each lever.

**Design chat.** Ask what-ifs about a run ("what if the reserve were 25 %?"). It needs an LLM key. Every what-if is
checked with the real solver, any number in an answer that did not come from the solver is flagged, and
proposals only fill the review panel; nothing runs until you press Re-run. A message uses at most 3 LLM calls.
The conversation is saved with the run.

**Linked re-runs and design history.** Selecting an option or accepting a chat proposal fills a review panel with
the change (requirements, assumptions, or knobs); **Re-run** starts a new run linked to its parent. The design
history lists every run of the session with its parent and where the change came from, and Open reloads any of them.

### Avionics stack

The aircraft are designed for autonomous operation, so the **Avionics** column of the requirements box picks the
stack:

- **Cube Orange+**: the flight controller alone (73 g); a **Jetson Orin NX companion** on its own carrier board
  (0.2 kg, 25 W) is optional and ticked by default.
- **ARK Jetson PAB**: the ARK carrier with an ARKV6X flight controller and a Jetson Orin NX 16 GB on one board stack
  (166 g with heatsink, 25 W). The Jetson is always part of it, so the companion box greys out.

A description can choose too ("ARK", "Cube", "autonomous", "no companion computer"); unstated means Cube with a
Jetson. The Jetson's power is drawn for the whole flight and its mass goes into the CAD avionics tray, so it shows
up in endurance, MTOW and the CG. The parts list has a line for each board with a reference price (display only;
prices move). The masses, watts and the shared allowance are editable under Avionics in Design assumptions.
Figures researched 2026-11-18 from arkelectron.com, docs.px4.io, retailer listings (GetFPV, RobotShop, Lumenier)
and NVIDIA's Jetson announcements; the design spec lists them in full. Runs made before this feature read as a
Cube Orange+ without a Jetson, which is exactly what they sized.

### Payload catalogue

The **Payload** column of the requirements box picks the payload: a researched NextVision or SIYI gimbal, one of the
generic sensor classes (SAR, LiDAR, relay, heavy turrets…), or **Custom payload**. Choosing a model fills in its
datasheet mass and typical power; both stay editable, and an edited unit is listed as "(edited)" in the parts list
and the report. Sizing always uses the two numbers in the box. A description can name a model ("carry a SIYI ZT30");
its datasheet values are used unless the request states others.

| Gimbal | Type | Mass (kg) | Power (W, typical) | Reference price |
|---|---|---|---|---|
| NextVision DragonEye2 | EO/IR | 0.125 | 3.5 | quote only |
| NextVision X80 | EO | 0.145 | 4.5 | quote only |
| NextVision Starling | EO/IR | 0.18 | 6 | quote only |
| NextVision Starling-LRF | EO/IR | 0.23 | 6 | quote only |
| NextVision Stinger | EO/IR | 0.23 | 7 | quote only |
| NextVision NightHawk2 | EO/IR | 0.25 | 8 | quote only |
| NextVision Swift | EO/IR | 0.29 | 6 | quote only |
| NextVision NightHawk2-V | EO/IR | 0.32 | 5 | quote only |
| NextVision NightHawk2-UZ | EO/IR | 0.32 | 6 | quote only |
| NextVision Raptor | EO/IR | 0.64 | 7 | quote only |
| NextVision Condor | EO/IR | 1.5 | 12 | quote only |
| NextVision Raven 2.5 | EO/IR | 2.15 | 20 | quote only |
| SIYI A8 mini | EO | 0.095 | 5 | ≈ $251 |
| SIYI ZT6 | EO/IR | 0.197 | 6.5 | ≈ $2,940–3,700 |
| SIYI ZR10 | EO | 0.381 | 4 | ≈ $532 |
| SIYI UniPod MT11 | EO/IR | 0.5335 | 11.4 | not public |
| SIYI UniPod NV10 | EO | 0.628 | 15 | not public |
| SIYI ZR30 | EO | 0.668 | 5 | ≈ $1,181 |
| SIYI ZT30 | EO/IR | 0.854 | 9 | ≈ $7,000 |

Researched 2026-11-18 from nextvision-sys.com (compare page and camera pages), siyi.biz spec pages and Reebot's
MT11 page; prices are display only. NextVision does not say what its masses include (probably the bare camera, so
add your mount); SIYI's ZT6, ZR30, ZT30 and MT11 figures include their plate, mount or board. Peak draw (12-26 W
where published) is not modelled. Custom payloads, and every run made before this feature, are matched to the
lightest generic sensor class at or above their mass in the parts list, exactly as before.

### 3D viewer

Every run that reaches the CAD stage also writes `viewer.html`, shown in the web form's **3D** tab. Three views of
the same aircraft: **Shape** is the opaque outer skin; **X-ray** makes the skin transparent to show the internal
structure and every packed component, with a legend (name, colour, mass), a hover tooltip and the centre of
gravity marked; **Flight replay** plays the ArduPilot SITL flight back on the model (play/pause, 0.5x to 5x speed,
a time slider, a readout of altitude, airspeed, roll, pitch and heading, a flight-path trail, orbit or chase camera).

The replay is the real SITL flight, not an animation: it follows the logged position and attitude of the last
flown attempt. `viewer.html` is one self-contained file (Three.js inlined, no network), so it opens offline in
any browser and can be shared as is; the 3D tab has a download button. Runs made before this feature did not log
heading or position, so their replay stays in place and only rotates. Mesh export and the viewer build need the
aero/CAD stage (Docker image, `docker compose up ui`); a failure there is a run warning, not a failed run.

### Choosing the LLM

The provider and model are configuration. Every adapter returns schema-validated objects, so any model that
supports structured outputs works.

| Setting | Meaning |
|---|---|
| `OPENROUTER_API_KEY` | use [OpenRouter](https://openrouter.ai/models): any model it lists with structured outputs |
| `ANTHROPIC_API_KEY` | use Claude directly through the Anthropic SDK |
| `ISR_FORGE_LLM_PROVIDER` | `openrouter` or `anthropic`; unset picks whichever key is set (OpenRouter first) |
| `ISR_FORGE_MODEL` | model id in the provider's naming, e.g. `openai/gpt-5.6-sol`, `google/gemini-3.8-flash` |

Defaults: `anthropic/claude-opus-5.5` on OpenRouter, `claude-opus-5-5` on Anthropic. These are set **only** in
`.env` (or the shell environment on the host); the CLI and the web form show the active provider and model but
cannot change them, so a form user cannot pick a model billed to your key. Restart `docker compose up ui` after
editing `.env`. OpenRouter requests set
`provider.require_parameters`, so they only reach hosts that enforce the JSON schema; a model with no such host
fails with a clear error rather than returning free text.

**Airframe layout.** `OperationalRequirements.airframe_layout` picks the airframe arrangement; the web form
(Airframe layout select), the LLM draft and the design chat carry it. Two layouts are built, each for both
`fixed_wing` and `quadplane_vtol`:

- `pod_boom_tractor` — pod, centre tail boom, conventional tail (H-stab + fin), propeller on the nose.
- `single_boom_vtail_pusher` — the same pod and centre boom with a V-tail (one mirrored ruddervator pair sized from
  the H and V tail volumes, aspect ratio 4, dihedral atan √(S_V/S_H)) and the propeller at the boom end, a 10 %
  tail-arm gap behind the tail. Cruise efficiency takes `pusher_efficiency_factor` (0.95). VSPAERO models each
  ruddervator panel; ArduPilot mixes the pair as a V-tail (`SERVOn_FUNCTION` 79/80, mixing gain 0.5); the JSBSim
  model applies the pusher thrust at the propeller. The kit adds the V-tail root block `VR-1`, per-panel spars and
  ribs, and the boom-end motor plate and clamp (`MP-B`/`MC-B`), with a wiring note for the motor run along the boom.
  The parser maps "V-tail" or "V tail" with "pusher" to it.

The other ids (`twin_boom_htail_pusher`, `twin_boom_inverted_v_pusher`, `flying_wing`) are recognised but rejected
with "not available yet". The layout supplies the static-margin band (tailed layouts: 8-14 %, target 11 %) used by
the critic, the CAD packer, the pipeline and the SITL perturbation corners; the summary and report show a Layout row.
The examples include the G1 and G2 missions on the V-tail pusher and a hand-launched G1.

**Launch method.** `launch_method` is `auto` (a runway fixed-wing if there is a runway, else a VTOL quadplane),
`runway`, or `hand` (a thrown fixed-wing: no runway, no VTOL, Group 1 only, MTOW ≤ 9.07 kg). A hand launch is
flown in SITL as a 10 m/s throw from 2 m at 10° nose-up, the aircraft held at rest until the throttle opens. So the
throw stays 1.2 × above stall, a hand-launched design is sized with its stall speed capped at 10 / 1.2 = 8.33 m/s
(the hand-launched G1 therefore gets a larger wing: about 5.9 kg, 1.25 m²). A hand-launched design is also drag-sized for a 30 % laminar boundary layer on its wing and tails (`HAND_LAUNCH_LAMINAR_FRACTION`): sizing takes NeuralFoil's profile drag with transition forced at 30 % chord, and VSPAERO's ParasiteDrag runs 30 % laminar on the same surfaces, so the L/D gate compares like with like. Runway and VTOL designs keep NeuralFoil's free transition and OpenVSP's fully turbulent build-up. Records saved before launch methods
existed that hold a fixed-wing with no runway load as hand launches.

**Landing gear.** One rule (`src/tools/gear.py`) serves sizing, the kit and the JSBSim model. Runway and VTOL
aircraft get a tricycle whose leg is the tallest of the fuselage/propeller ground clearance and the propeller-tip
clearance (25 % of its radius) — taken at the 12° take-off rotation for a runway launch and level for a VTOL
quadplane. A hand-launched aircraft has no gear; if its propeller tip hangs below the belly line (taken at the 5°
belly-landing pitch) it needs a folding propeller, which the build sheet notes — tractors included.

**Not yet (carried over).** Tail surfaces sit on the centreline (the V-tail pair is mirrored about y = 0); the
lateral offset that twin booms need lands in phase 3. The `wing_sweep_deg` critic knob and its steps and bounds land
with the flying wing in phase 5.

**Known kit limit.** On the default fabrication stock the G2 V-tail pusher gets no laser kit: its wing-root torque
loads the anti-rotation pin beyond the largest stock rod (the run explains this and packs the shell model). A
heavier rod in the fabrication settings is the user's choice.

2026-11-23: every layout re-baselined to explicit tail/lift-boom tubes and boom drag (+ stopped-rotor term); reference
values updated once. The tail boom (1.6 kg/m² carbon wall × π d × length, from 0.9 of the pod to the tail trailing
edge) and the two lift booms (their real length) are booked in the empty-mass breakdown as `tail_boom` and
`lift_booms`, replacing the 4 % MTOW lift-boom lump. Their wetted-area drag plus a stopped-rotor term replaces the
0.008 quadplane drag lump, calibrated on G2 so lift booms + rotors take the lump's place inside the misc factor and
still sum to 0.008 there. `SizingOutput.empty_mass_breakdown` now carries the itemised empty mass. Reference
designs (MTOW / empty mass / CD0, before → after):
G1 4.073 → 4.158 kg / 1.750 → 1.814 kg / 0.01889 → 0.01894; G2 19.36 → 19.88 kg / 7.985 → 8.262 kg /
0.02175 → 0.02214; G3 181.9 → 136.6 kg / 81.60 → 59.20 kg / 0.02366 → 0.02300 (G3's lighter lift booms and
smaller wing drop its cruise Reynolds number just below 10⁶, so it now takes MH 32 instead of NACA 4415).

### Command line

```bash
docker compose run --rm isr-forge python -m src.main \
  "Design a runway-independent Group 2 VTOL ISR drone with a 3.5 kg EO/IR gimbal, 6-hour endurance, and 80 km datalink radius"

docker compose run --rm isr-forge python -m src.main --spec requirements.json --no-sitl
```

`--spec` accepts an optional top-level `"assumptions"` object next to the requirements; give only the values you
want to change (unknown names and out-of-range values are rejected):

```json
{
  "mission_name": "Group 2 VTOL ISR",
  "...": "other OperationalRequirements fields",
  "assumptions": {"liion_specific_energy_wh_kg": 250, "fuel_reserve_fraction": 0.15}
}
```

After a non-converged run the CLI also prints the options that close the design (the same explorer as the web form).

| Option | Meaning |
|---|---|
| `--spec FILE` | `OperationalRequirements` JSON instead of a description (no LLM needed) |
| `--no-aero` / `--no-sitl` | skip OpenVSP/CAD (and therefore SITL) / skip the SITL flight test |
| `--max-iterations N` | critic loop cap, 0–50 (default 8, per the SRS) |

Exit codes: `0` converged, `1` not converged or SITL failed, `2` clarification needed, `3` error.

## What a run produces

Each run writes `outputs/<UTC timestamp>-<mission>/`:

| File | Contents |
|---|---|
| `design_report.md` | requirements and assumptions, critic gates and iterations, sizing, energy budget, BOM, VSPAERO derivatives and polar, CAD/inertia, SITL metrics, warnings |
| `design_report.pdf` | the same report typeset, plus figures: a three-view of the packed aircraft, the mission power profile, the VSPAERO and sizing drag polars, and the SITL roll/pitch responses (a figure whose input is missing, or that fails, is left out; if the PDF cannot be written the run logs why and keeps the Markdown report) |
| `bom.json` | selected COTS parts and total hardware mass |
| `oml.step`, `oml.vsp3`, `assembly.step` | outer mould line, and internal structure with packed components |
| `viewer_model.json`, `viewer.html` | `viewer_model.json` is the display mesh (tessellated parts) and CG; `viewer.html` is the self-contained page with that mesh and the replayed SITL flight trace inlined |
| `kit/sheets/*.dxf` | laser-cut kit sheets (Group 1–2 designs): one DXF per sheet of stock, in mm, kerf-compensated |
| `kit/build_sheet.md`, `kit/parts.csv` | build sheet (parts, stock tubes, joins, centreline overwrap lay-up, covering) and the parts list |
| `ardupilot.param` | ArduPlane / QuadPlane parameters flown in SITL |
| `sitl_report.json` | take-off, transition and roll/pitch doublet telemetry and metrics |
| `critic_log.json`, `run_summary.json` | every critic pass and the final status |
| `requirements.json`, `sizing.json`, `aero.json`, `cad.json`, `fcs.json` | stage outputs |
| `assumptions.json` | the design assumptions the run used (defaults when unchanged) |
| `options.json` | explorer options with their sizing + catalog results (written when the explorer ran) |
| `design_chat.json` | the design chat for this run (written from the web form once you chat) |

**Download all files (.zip)** in the Files tab exports the whole run as `<run id>.zip`: every file above
(sub-folders included) under one `<run id>/` folder, plus `design.json`. `design.json` is a versioned record
(`"format": "isr-forge-design"`, `"format_version": 1`) of everything the results view shows: requirements,
assumptions, sizing, parts, aero, CAD, flight test, critic gates and history, explorer options and lineage, with
paths relative to the run folder. It is built when you download and exists only inside the zip; it is what a
future "open a shared design" feature will read.

## Building the laser-cut kit

Group 1–2 designs (MTOW up to 24.95 kg / 55 lb, `GROUP2_UPPER_KG` in `src/tools/sizing/topology.py`) get a laser-cut internal structure: plywood ribs, formers,
crutch and wing-mount plates cut from sheet, with stock carbon tubes for the spars and booms. Everything is in
`kit/` inside the run folder:

- `kit/sheets/<stock>-<n>.dxf`: one file per sheet of stock, in millimetres, with parts nested on the bed. Layers:
  `CUT_OUTER` (outer outlines), `CUT_INNER` (holes and slots) and `ENGRAVE` (part labels). Outlines are
  kerf-compensated, so slots press-fit.
- `kit/build_sheet.md`: the parts, the tubes to cut, glued joins, covering, wing stiffness checks and the kit mass. Its centreline joint
  section gives the carbon overwrap lay-up: cloth, layer count and thickness, lap per side, and the order to wrap
  the glued mitre between the two fuselage socket tubes (the spar roots slide into the sockets unglued, so the wings detach).
- `kit/parts.csv`: the parts list.

The kit mass and CG feed into the as-built design. To change the bed, kerf, sheet and tube stock or the structural
allowables, use **Laser kit settings** in the web form, or pass `--fabrication PATH` (a `FabricationSettings`
JSON) on the command line. The settings used are saved with the run. Group 3 designs, or a design where no stock
tube passes the structural checks or a part cannot fit the bed, get no kit: the run falls back to the shell mass
model and the report carries a warning saying why.

## Critic gates (SRS §5)

1. Static margin 8–14 %
2. VSPAERO L/D within ±15 % of the sizing L/D at the sizing cruise and loiter CL (VSPAERO's inviscid induced drag gets Kroo's viscous term, k + 0.38·CD0)
3. Zero CAD collisions and MTOW within the DoD group limit
4. Energy reserve ≥ 15 % (installed battery margin for electric, fuel reserve for hybrid)

A failed gate sends the design back to sizing with adjusted overrides (aspect ratio, fuselage length, tail
volumes, reserve), up to 8 iterations. The reserve knob may only be raised: the critic never lowers it below the
reserve you set in the design assumptions. The SRS example converges after one iteration: its 10 % hybrid fuel
reserve fails gate 4 and the critic raises it to 15 %.

## Known limitations

- The COTS catalog is representative, not vendor-verified. For a Custom payload the matcher snaps to the next sensor up, which
  can draw more power or weigh more than requested; the run warns when that makes the energy budget
  optimistic (set the payload power to the sensor's draw and re-run).
- As-built mass outside ±3 % of the sized MTOW is reported as a warning, not a gate (the SRS does not gate it).
- VSPAERO's vortex-lattice drag omits lift-rotor and nacelle drag, so quadplane L/D reads optimistic.
- The design wind is a steady headwind on ingress and egress only: the loiter and climb ignore it, and gusts are not
  modelled. Hot-and-high (take-off altitude, ISA offset) sets hover, VTOL and stall density; the SITL check still
  flies a standard day.

## Development

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e '.[physics,agent,ui,dev,report]'   # everything but OpenVSP/ArduPilot
.venv/bin/python -m pytest -q                                                   # OpenVSP/SITL tests skip on the host
docker run --rm -v "$PWD":/app -w /app isr-forge:latest python -m pytest -q -p no:cacheprovider   # full suite
```

The agent tests use a scripted LLM (no network, no cost); the OpenRouter adapter is tested through the real
`openai` SDK against a mock HTTP transport. `ISR_FORGE_LIVE_LLM=1` plus an API key runs the end-to-end test
against the configured provider and model (three or more calls per run).
