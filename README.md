# agentic-drone-designer

An agent that designs DoD Group 1–3 ISR UAVs from mission requirements: conceptual sizing (battery-electric
or series-hybrid, fixed-wing or quadplane VTOL), a COTS bill of materials, OpenVSP/VSPAERO aerodynamics,
Build123d internal CAD solved for static margin, ArduPilot parameters validated in a headless SITL flight
test, and a design report. The full specification is in [REQUIREMENTS.md](REQUIREMENTS.md).

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

**Design assumptions.** A panel above the forms holds the 12 technology constants the sizing chain uses; the
defaults reproduce the historical values exactly.

| Group | Values (default) |
|---|---|
| Li-ion | specific energy 200 Wh/kg, energy density 250 Wh/L, max discharge 3 C |
| LiPo | specific energy 155 Wh/kg, energy density 280 Wh/L, max discharge 25 C |
| Reserves | battery reserve 0.20, fuel reserve (hybrid) 0.10 |
| Propulsion and VTOL | generator BSFC 0.50 kg/kWh, drivetrain efficiency 0.68, hover figure of merit 0.72, lift motor efficiency 0.85 |

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
| `bom.json` | selected COTS parts and total hardware mass |
| `oml.step`, `oml.vsp3`, `assembly.step` | outer mould line, and internal structure with packed components |
| `ardupilot.param` | ArduPlane / QuadPlane parameters flown in SITL |
| `sitl_report.json` | take-off, transition and roll/pitch doublet telemetry and metrics |
| `critic_log.json`, `run_summary.json` | every critic pass and the final status |
| `requirements.json`, `sizing.json`, `aero.json`, `cad.json`, `fcs.json` | stage outputs |
| `assumptions.json` | the design assumptions the run used (defaults when unchanged) |
| `options.json` | explorer options with their sizing + catalog results (written when the explorer ran) |
| `design_chat.json` | the design chat for this run (written from the web form once you chat) |

## Critic gates (SRS §5)

1. Static margin 8–14 %
2. VSPAERO cruise L/D ≥ 0.9 × the sizing assumption
3. Zero CAD collisions and MTOW within the DoD group limit
4. Energy reserve ≥ 15 % (installed battery margin for electric, fuel reserve for hybrid)

A failed gate sends the design back to sizing with adjusted overrides (aspect ratio, fuselage length, tail
volumes, reserve), up to 8 iterations. The reserve knob may only be raised: the critic never lowers it below the
reserve you set in the design assumptions. The SRS example converges after one iteration: its 10 % hybrid fuel
reserve fails gate 4 and the critic raises it to 15 %.

## Known limitations

- The COTS catalog is representative, not vendor-verified. The matcher snaps to the next sensor up, which
  can draw more power or weigh more than requested; the run warns when that makes the energy budget
  optimistic (set the payload power to the sensor's draw and re-run).
- As-built mass outside ±3 % of the sized MTOW is reported as a warning, not a gate (the SRS does not gate it).
- VSPAERO's vortex-lattice drag omits lift-rotor and nacelle drag, so quadplane L/D reads optimistic.

## Development

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e '.[physics,agent,ui,dev]'   # everything but OpenVSP/ArduPilot
.venv/bin/python -m pytest -q                                                   # OpenVSP/SITL tests skip on the host
docker run --rm -v "$PWD":/app -w /app isr-forge:latest python -m pytest -q -p no:cacheprovider   # full suite
```

The agent tests use a scripted LLM (no network, no cost); the OpenRouter adapter is tested through the real
`openai` SDK against a mock HTTP transport. `ISR_FORGE_LIVE_LLM=1` plus an API key runs the end-to-end test
against the configured provider and model (three or more calls per run).
