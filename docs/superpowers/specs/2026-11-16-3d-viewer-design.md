# 3D Viewer and Flight Replay — Design

**Date:** 2026-11-16 · **Status:** draft for approval

## Intent

The designer wants to *see* the aircraft the agent produced, not only read numbers about it. Three views, all
requested explicitly by the user:

1. **Shape** — the outer mould line, orbit / zoom / pan.
2. **X-ray** — the skin goes translucent; packed components (battery, payload, motors, tank, generator,
   avionics), wing ribs and spars are shown in distinct colours, named on hover with their mass, and the
   as-built centre of gravity is marked.
3. **Flight replay** — the SITL flight that validated the design (take-off, transition, trim, roll and pitch
   steps) played back on the 3D model, with play / pause / speed / scrub, a trail and a chase camera.

Why: today the CAD exists only as STEP files the user must open in an external tool, and the SITL flight is a
table of step-response metrics. Seeing the packing and the flight builds trust in the design and makes
problems (a component in the wrong place, an ugly transition) obvious at a glance.

## Goals

- A **3D** tab in the web form, shown for any run that has CAD output.
- A standalone `viewer.html` in every run folder that opens offline in a browser, with no app or server.
- Replay of the real simulated flight (JSBSim ground truth), not a re-simulation.

## Non-goals

- Interactive flying (a pilot or ground station controlling the SITL in real time). This is a separate
  feature: it needs a live MAVLink port and a non-headless run mode.
- Animated control surfaces or spinning propellers: the OML has no separate movable surfaces.
- Editing geometry in the viewer.
- Back-filling viewers for runs made before this feature (the user can re-run a design).

## Anti-goals

- Loading anything from the internet at view time (the app binds to 127.0.0.1 and must work offline).
- A viewer that drifts from the engineering model: geometry and CG come from the same `PackingResult`
  the critic used, and the flight path from the same JSBSim trace the SITL scoring used. No values are
  recomputed or re-estimated in JavaScript ("Zero LLM Math" spirit: the viewer only displays).
- New Python dependencies.

## Constraints

- Python with type hints, functions < 50 lines, files < 800 lines, no magic numbers, TDD, 80 %+ coverage.
- CAD and SITL run only in the Docker image; the host test suite skips them, so the mesh export must be
  testable with build123d on the host (build123d is installed there) using simple solids.
- Three.js is vendored into the repo (pinned version, MIT licence file alongside) and inlined into the HTML.
- `viewer.html` size: target ≤ 8 MB including the inlined library and meshes.
- Everything numeric written to the page is serialised with `json.dumps` and injected into a single
  `<script type="application/json">` block, with `</` escaped, so no run data can break out of the script.

## Approaches considered

- **A. Three.js viewer (chosen by the user).** Smooth replay (only a transform changes per frame), x-ray via
  material opacity, standalone shareable HTML. Cost: ~300 lines of JavaScript not covered by pytest.
- **B. Plotly 3D in Streamlit.** Pure Python and easy to test, but needs a new dependency and animates by
  re-sending the whole mesh each frame, so the replay is choppy. Rejected.
- **C. PyVista / stpyvista.** Heavy VTK dependency, poor animation inside Streamlit. Rejected.

## Design

### 1. Mesh export (CAD stage, Python)

New module `src/tools/aero_cad/mesh_export.py`, called from `export_artifacts` right after `assembly.step`:

- Tessellate each solid of `PackingResult` with build123d (`Shape.tessellate(tolerance, angular_tolerance)`)
  into vertices + triangle indices, grouped into **parts**:
  - `skin` — the OML;
  - `structure` — all ribs and spars merged into **one** part (there are dozens of solids; one legend entry);
  - one part per `PlacedComponent`, carrying `name`, `kind`, `mass_kg`. Names are not guaranteed unique,
    so a repeated name gets a ` #2`, ` #3` suffix in export order. *(Builder-check answer, my call.)*
- Coordinates stay in the CAD frame (SI, **x aft from the nose, y right, z up**, as stated in
  `layout.py:1`; the OpenVSP OML is built from that same layout); the viewer converts.
- Tolerances are named constants chosen so the whole model stays within the size target; vertices are
  rounded to 0.1 mm and packed as base64 little-endian float32 / uint32 arrays.
- Output: `viewer_model.json` in the run folder:
  `{ "frame": "x_aft_y_right_z_up", "cg_m": [x,y,z], "mass_kg": …, "parts": [ {"name","group","mass_kg"|null,
  "positions_b64","indices_b64"} … ] }`.
- A failure here is logged as a run warning and leaves the run otherwise intact (the viewer is a display aid,
  not a gate); it is never swallowed silently.

### 2. Flight trace (SITL bridge)

Add three values to every truth sample in `sitl_bridge.py`:

- `heading_deg` (`attitude/psi-deg`);
- `north_m`, `east_m` — the same local-tangent-plane formulas `_state()` already uses for ArduPilot
  (factor the two lines into one helper used by both).

Existing keys are unchanged, so scoring and the report are untouched. Runs made before this change lack the
new keys; the viewer then replays altitude, roll and pitch in place with the heading held fixed, and shows a
note.

**Which flight is replayed.** `generate_and_verify_fcs` may fly up to `MAX_ATTEMPTS` attempts
(`sitl/attempt_N`) and reports `flown[-1]`, the last attempt that completed a flight. `sitl_report.json`
gains `"replay_attempt": N` for exactly that attempt, and the viewer reads the trace from that directory, never
by globbing. SITL runs once per run (only after the critic converges, `graph.py:49`) and every run has its own
folder, so no trace from an older geometry can be picked up. A flight that failed the SITL limits is still
replayed, with a "SITL: FAILED" badge. *(My call.)*

**Reference point.** The JSBSim model uses the CG as aero reference point (`jsbsim_exporter.py:131,138`); the
viewer pivots the model about the CAD CG and places the CG on the trace position. If JSBSim reports position
at its VRP (the nose, `VRP` at x = 0) instead, the error is the nose-to-CG distance (about 0.5 m on a
Group 1–2 design), invisible against a flight path hundreds of metres long. *(My call; noted, not corrected.)*

### 3. Viewer page (`src/viewer/`)

- `src/viewer/vendor/three.module.min.js`, `OrbitControls.js` + `LICENSE` (pinned Three.js release).
- `src/viewer/viewer.js` — the page logic; `src/viewer/template.html` — layout and styles.
- `src/viewer/build.py` — `build_viewer_html(model: dict, trace: list[dict] | None) -> str` inlines the
  library, the page script and the data block; `write_viewer(run_dir: Path) -> Path | None` reads
  `viewer_model.json` and the replayed attempt's `jsbsim_truth.json` (if any) and writes `viewer.html`.
- **Where it runs:** the web form and CLI run the agent graph, whose last node for every outcome is
  `compile_report` (`graph.py:67-68`, `nodes.py:283`). `write_viewer` is called there, before the report is
  written, so the report can link `viewer.html`. If `pipeline._execute` (the non-agent flow) is still called
  anywhere, it calls the same function at its end. A failure becomes a run warning.

Page behaviour:

- **Mode toggle:** Shape | X-ray. Shape: opaque skin, structure and components hidden. X-ray: skin at
  ~20 % opacity, structure grey, each component a distinct colour from a fixed palette, a legend listing
  component, colour and mass, hover highlight with a name + mass tooltip, and a CG marker (sphere plus
  three short axes) labelled with the CG position.
- **Replay panel** (only when a trace exists): play / pause, speed 0.5× / 1× / 2× / 5×, a time slider with
  phase marks if they can be read from the trace (take-off, transition, steps), a readout of time, altitude,
  airspeed, roll, pitch and heading, a flight-path trail and a camera choice (orbit or chase).
- **Frames:** CAD (x aft, y right, z up) → aircraft body (x forward, y right, z down) about the CG; the
  trace is NED with attitude as roll/pitch/heading (Z-Y-X Euler); the scene is drawn in Three.js's y-up
  convention. Each frame interpolates the trace linearly in time (angles via quaternion slerp).
- Ground grid and a 1 m scale bar; works in light and dark browser themes.

### 4. App integration

- `DESIGN_TABS` gains **"3D"**. The tab embeds `viewer.html` (`st.components.v1.html`, fixed height) and
  offers a download button for it. No run folder (`view.out_dir is None`) or no viewer file → an info message
  explaining the CAD stage did not run (or that this run predates the feature).
- The README gains a short "3D viewer" section and the `viewer_model.json` / `viewer.html` rows in the
  outputs table.

## Testing

- `mesh_export`: tessellating a box and a cylinder gives the right part count, names, groups, finite
  coordinates within the solid's bounding box, and indices in range; CG and mass are copied unchanged;
  the base64 round-trips.
- `sitl_bridge`: the truth sample contains `heading_deg`, `north_m`, `east_m`, equal to `_state()`'s
  values for the same JSBSim state (fake FDM, as in the existing bridge tests).
- `viewer.build`: output contains the data block exactly once, escapes `</script>` in a part name, includes
  the vendored library, makes no `http(s)://` script or style references, and handles a missing trace.
- `app`: the 3D tab shows the viewer when `viewer.html` exists and the info message otherwise
  (Streamlit `AppTest`, as in the existing UI tests).
- Manual browser check of the three views on one fixed-wing and one quadplane run (Docker), recorded in
  the PR.

## Left to the builder

- Exact tessellation tolerances, palette colours, camera defaults and panel styling.
- Whether phase marks can be derived reliably from the trace; if not, omit them.
- Tab height, and whether the tab reads the file lazily.
