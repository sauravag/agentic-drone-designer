# ISR-Forge

An agent that designs DoD Group 1–3 ISR UAVs from mission requirements: conceptual sizing (battery-electric
or series-hybrid, fixed-wing or quadplane VTOL), a COTS bill of materials, OpenVSP/VSPAERO aerodynamics,
Build123d internal CAD solved for static margin, ArduPilot parameters validated in a headless SITL flight
test, and a design report. The full specification is in [REQUIREMENTS.md](REQUIREMENTS.md).

**Zero LLM math.** Claude only reads the request, suggests which design knob to turn, and writes report
prose. Every engineering number comes from deterministic tools; the critic's knob steps are bounded and
rule-table fixes always win on failed gates; any figure in the LLM narrative that is not in the computed
results is flagged in the report.

## Quick start

```bash
cp .env.example .env            # optional: add ANTHROPIC_API_KEY for natural-language input
docker compose build            # OpenVSP, VSPAERO, ArduPilot arduplane, JSBSim (first build is slow)
docker compose up ui            # web form at http://localhost:8501 (bound to 127.0.0.1)
```

The web form accepts either a description (needs `ANTHROPIC_API_KEY`) or structured requirements (works
without a key: rule-table critic, template report narrative). Both run the same agent and critic loop.
Docker Compose reads `.env`; when running on the host, `export ANTHROPIC_API_KEY=...` in your shell instead.

### Command line

```bash
docker compose run --rm isr-forge python -m src.main \
  "Design a runway-independent Group 2 VTOL ISR drone with a 3.5 kg EO/IR gimbal, 6-hour endurance, and 80 km datalink radius"

docker compose run --rm isr-forge python -m src.main --spec requirements.json --no-sitl
```

| Option | Meaning |
|---|---|
| `--spec FILE` | `OperationalRequirements` JSON instead of a description (no LLM needed) |
| `--no-aero` / `--no-sitl` | skip OpenVSP/CAD (and therefore SITL) / skip the SITL flight test |
| `--max-iterations N` | critic loop cap, 0–50 (default 8, per the SRS) |
| `--model ID` | Claude model (default `ISR_FORGE_MODEL` or `claude-opus-5-5`) |

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

## Critic gates (SRS §5)

1. Static margin 8–14 %
2. VSPAERO cruise L/D ≥ 0.9 × the sizing assumption
3. Zero CAD collisions and MTOW within the DoD group limit
4. Energy reserve ≥ 15 % (installed battery margin for electric, fuel reserve for hybrid)

A failed gate sends the design back to sizing with adjusted overrides (aspect ratio, fuselage length, tail
volumes, reserve), up to 8 iterations. The SRS example converges after one iteration: its 10 % hybrid fuel
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

The agent tests use a scripted LLM (no network, no cost). `ISR_FORGE_LIVE_LLM=1` plus `ANTHROPIC_API_KEY`
runs the end-to-end test against Claude (three calls per run).
