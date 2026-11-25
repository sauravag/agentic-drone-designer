# Propulsion matching, electrical system and avionics — design

Date: 2026-11-24. Status: draft, awaiting user approval. Builder check done; answers folded in.
Origin: item 1 of the user's review of `outputs/debug-roll/20261121T075427Z-g1-electric-fixed-wing/` (7 gaps;
order agreed 2026-11-24: this, then control hardware, then the builder's pack, then EPP, then layouts phases 3–5).
Decisions marked "user" were made in the 2026-11-24 conversation; "controller's call" marks proposals the user has
not separately ruled on.

## 1. Intent

The reviewed G1 run paired a 14.8 V (4S) motor with a 21.6 V (6S) pack, listed no ESC, power module, GPS, airspeed
sensor, receiver or telemetry radio, and set `ARSPD_USE 1` without naming a sensor. The cause is structural: every
part is matched on one number (motor on continuous power, battery on energy and peak power), KV and voltage are
never compared, and propulsive efficiency is a fixed assumption (0.68 cruise, 0.72 hover figure of merit, 0.85 lift
motor, 0.75 in the flight model). The user wants a propulsion system and electrical system a builder can buy and
wire: motor, propeller, cell count and ESC matched from a real propeller and motor model whose efficiency feeds
the sizing, plus named avionics and a connection table.

## 2. Goals (user's decisions)

1. **Full propeller model (user: C).** Propeller performance comes from a blade-element calculation, not a fixed
   efficiency, and motor KV is chosen from the propeller rpm the thrust needs.
2. **Feeds back into sizing (user: 2).** The model's efficiency at each flight point replaces the fixed
   `drivetrain_efficiency`, `hover_figure_of_merit`, `lift_motor_efficiency` and the flight model's
   `PROP_EFFICIENCY` inside the sizing loop and the simulator. Every reference design re-baselines (once per phase).
3. **Lift rotors included (user).** Quadplane lift motors, rotors and ESCs are matched the same way, at hover.
4. **Propeller data from our own blade-element calculation (user: C)** using NeuralFoil (already used for the wing),
   checked in tests against a few measured UIUC propeller datasets.
5. **Named avionics with reference prices** (controller's call, follows the Cube Orange+ and named-gimbal pattern):
   GPS + compass, airspeed sensor, power module, RC receiver, telemetry/datalink, ESCs. They replace the lumped
   avionics mass and power allowance. Wiring mass is calculated, and a connection table is output.
6. **Radio links chosen by range (user: A):** lightest link whose rated range covers the mission radius with a
   margin, plus a line-of-sight (radio horizon) check; no link reaches → readable refusal.
7. **Two phases, propulsion first (user: A)**, each on its own branch with its own review and re-baseline.

## 3. Non-goals

- Servos, hinges, horns, throws (item 2, control hardware). Servo mass stays inside the residual systems fraction.
- Build-sheet W&B, rigging, acceptance card (item 3).
- Battery thermal modelling, cell-level internal resistance curves, ESC switching losses beyond a fixed efficiency.
- Real (vendor-named) motors and propellers: those catalogues stay generic classes with realistic data.
- Ground-station hardware beyond the ground antenna height used by the horizon check.

## 4. Anti-goals

- Any silent fallback to a fixed efficiency when the propeller or motor model fails: a failure is a readable
  refusal naming the limiting constraint.
- Passing a design whose motor, ESC, pack and generator voltages do not agree, or whose currents exceed any rating.
- A radio link that cannot reach the mission radius.
- Unflagged estimates: every estimated motor constant, the generic blade shape and every link range are listed as
  assumptions in the report.

## 5. Approaches

Depth of matching: A (common voltage only), B (A + pitch-speed check), **C (full propeller model; user)**.
Coupling to sizing: 1 (hardware chosen after sizing, efficiency only reported), **2 (efficiency feeds the sizing;
user)**. Propeller data: UIUC measured (≤ 19 in only), APC tables (≤ 27 in, predictions), **own blade-element
calculation (all sizes and hover; user)**. Radio links: **by range (user)** or fixed per group (rejected: a long
G1 mission would silently get a short-range radio). Sequencing: **one spec, two phases, propulsion first (user)**;
avionics first (would need placeholder currents and be redone) and one branch (one re-baseline but propulsion and
avionics mass changes tangled) were rejected.

## 6. Phase 1 — propulsion matching

### 6.1 Propeller model
- Blade-element momentum calculation with Prandtl tip and hub loss, on a **generic blade** defined by diameter D and
  pitch P: a fixed chord distribution (fraction of D vs r/R) and twist from constant geometric pitch, one generic
  thin propeller airfoil. Section lift and drag from NeuralFoil at the local Reynolds and Mach numbers.
- Outputs per (D, P, rpm, airspeed, density): thrust, shaft power, torque, efficiency; at zero airspeed, hover thrust
  and power (figure of merit follows).
- Results tabulated per catalogue propeller as CT(J) and CP(J) tables plus static CT0 and CP0 per Reynolds band, and
  cached on disk keyed by blade, airfoil and model version, so the sizing loop never runs NeuralFoil per iteration.
- Tip Mach limit `PROP_TIP_MACH_MAX = 0.70` (controller's call; noise and compressibility).
- Validation (test gate): for at least three UIUC datasets (small APC thin-electric and sport props), the model's
  peak efficiency and the J at peak are within ±10 % and CT0 within ±15 % (controller's call). The datasets go in
  as test fixtures with their citation.
- Flagged assumption: "generic blade shape; real propellers of the same D x P differ by several % in efficiency".

### 6.2 Motor model
- Three-constant DC motor: KV, winding resistance Rm, no-load current I0. Gives rpm, current, shaft torque and motor
  efficiency for a voltage and load torque.
- New catalogue columns `rm_ohm`, `i0_a`, `max_current_a` on cruise and lift motors (estimates, flagged,
  representative of current outrunner product lines). **Two or three KV windings per motor size** so matching can
  choose the winding.
- Existing `max_power_w` / `continuous_power_w` stay as thermal limits.

### 6.3 Electrical bus, ESC, generator
- **Packs are built from cells (controller's call, builder check).** A small `cells` catalogue (21700 Li-ion,
  LiPo pouch classes: nominal and max voltage, capacity, max continuous current, mass, size) replaces the fixed
  `batteries` pack rows. Series count S comes from the matching; parallel count P is the smallest that meets the
  energy and the peak current. Sizing keeps a continuous battery mass; the BOM pack rounds P up, and the existing
  as-built mass gate covers the rounding. The hybrid buffer pack is built the same way. `BATT_CAPACITY` follows.
- Pack series count sets the bus voltage. Matching evaluates at the nominal pack voltage, and the climb and hover
  points again at the end-of-reserve loaded voltage (`LI_ION_LOADED_MIN_V_CELL = 3.3`,
  `LIPO_LOADED_MIN_V_CELL = 3.5`, controller's call) so a sagged pack still meets the thrust.
- New `escs` catalogue (generic classes): continuous current, burst current, max cell count / voltage, mass, size,
  efficiency (fixed, flagged). Chosen: lightest with continuous current ≥ `ESC_CURRENT_MARGIN` (1.2) x the motor's
  highest matched current and voltage rating ≥ fully charged pack voltage. One ESC per motor.
- Series hybrid: **the buffer pack sets the bus voltage**; the generator must sit inside the bus window (charged
  to end-of-reserve pack voltage). Genset classes get voltage variants (e.g. 6S, 12S, 14S regulated DC output),
  as real gensets offer selectable outputs, so the bus voltage does not rule out every genset (controller's call).

### 6.4 Matching search
- Flight points: cruise, loiter, climb (from the mission plan); for quadplanes, hover at `VTOL_THRUST_TO_WEIGHT`.
- For each candidate set (pack series count, motor + winding, propeller): solve rpm where motor torque equals
  propeller torque for the required thrust at each point; require throttle ≤ `MAX_MATCH_THROTTLE` (0.85), current ≤
  motor `max_current_a`, power within the motor limits, tip Mach ≤ limit, and propeller diameter within the layout's
  clearance (gear rule, pusher tail gap, rotor spacing; the existing clearance checks use the chosen D).
- **Objective (controller's call, builder check):** minimise propulsion set mass (motors, propellers, ESCs) plus
  an estimated battery mass = mission energy with that set's efficiencies / specific energy, at the current pass's
  MTOW. No full resize per candidate; one resize per outer pass. "Set unchanged" means the same (S, motor and
  winding, propeller, lift motor and winding, lift rotor) tuple.
- **Propeller catalogue (controller's call):** several pitches per diameter (about P/D 0.5 to 1.0 for cruise,
  0.3 to 0.6 for lift rotors); folding variants for hand launch.
- **Matched diameters feed the airframe on the next pass (controller's call):** the cruise propeller D sets gear
  height and the clearance checks; the lift rotor is the smallest catalogue diameter at or above the disk-loading
  diameter, and its actual D sets disk area, boom length and rotor drag on the next pass.
- Pusher layouts keep `pusher_efficiency_factor` as an installation loss on the propeller efficiency.

### 6.5 Coupling to the sizing loop (stabilised)
- Outer fixed-point loop around today's MTOW root find: pass 0 uses today's assumption values as a seed only;
  after each converged sizing, match hardware, recompute the efficiency at every flight point from the matched set,
  and resize with those efficiencies held fixed. Stop when the matched set is unchanged between passes.
- Inside one sizing pass the hardware set is frozen, so the root find stays smooth (the balance-search pattern from
  phase 2 of the layouts work).
- If sets cycle (A → B → A), take the heavier set, which is feasible at both masses, and note it; no convergence
  within `MAX_PROPULSION_PASSES` (12) is a readable sizing failure. Amended 2026-11-25: raised from 6 under 6.8
  (a constant tuned for closure): G3 needs 6-8 passes to settle; the other reference designs settle in 2-3.
- The cruise propeller diameter replaces sizing's calibrated diameter ratio from layouts phase 2; rotor diameter
  still comes from disk loading, and the lift rotor is matched to it.
- The three efficiency assumptions leave the assumptions panel. Saved records that carry them still load
  (accepted and ignored, with a migration note).

### 6.6 Simulator and parameters
- The flight model's thrust uses the matched propeller's tables at the simulated airspeed and rpm (no 0.75),
  with a throttle → voltage → rpm motor loop. **Lift rotors switch too (controller's call, builder check):** their
  thrust comes from the rotor tables and motor model, and `MOT_THST_EXPO` and `Q_M_THST_HOVER` are derived from
  that curve instead of the catalogue `max_thrust_n`. SITL retuning risk is accepted; the SITL gate still applies.
  The JSBSim export uses the same tables (JSBSim's propeller element takes CT/CP vs J natively; the builder chooses
  whether to use it or the current force model with the tables).
- `BATT_CAPACITY` already follows the pack; the matched cell count also sets the battery failsafe voltages (builder
  picks the exact parameter set, checked against the ArduPilot docs).

### 6.7 Outputs and gate
- `bom.json`: ESC rows (count per motor), motor winding, propeller D x P, pack series count.
- Report: a "Propulsion match" table per flight point (rpm, current, throttle, propeller and motor efficiency,
  tip Mach) and the flagged assumptions; the critic names the limiting constraint on a no-match.
- Gate: host and Docker suites; all six reference designs through the full `--spec` pipeline converge, kits build
  where they built before, SITL passes; golden snapshots re-baselined once with a summary of what moved and why.

## 7. Phase 2 — electrical system and avionics

### 7.1 Catalogue
- New tables with named products and display-only reference prices (web research allowed; each price dated with
  its source; no source found → "price not found", never an invented figure): `gps_units` (GPS +
  compass), `airspeed_sensors` (with the ArduPilot `ARSPD_TYPE` value), `power_modules` (voltage, current, the
  ArduPilot monitor type and scaling), `rc_receivers` (protocol, range), `telemetry_links` (band, rated range,
  mass, power, air-side and ground-side units).
- Selection rules: power module rated for the charged pack voltage and the peak bus current; others the lightest
  that meets the requirement (range, protocol supported by the chosen flight controller).

### 7.2 Radio links (user: by range)
- Telemetry/datalink: rated range ≥ `LINK_RANGE_MARGIN` (1.25, controller's call) x combat radius.
- Radio horizon: `d = 4.12 x (sqrt(h_ground) + sqrt(h_aircraft))` km (4/3 earth, heights in m), with an editable
  ground antenna height (default 2 m) and the aircraft height above the ground station = service ceiling minus
  take-off altitude (controller's call; there is no separate cruise altitude); radius beyond the horizon → refusal.
- RC receiver (user, 2026-11-24: A): must cover a local zone (`RC_LOCAL_RANGE_KM`, default 2 km)
  for take-off and landing; beyond it the aircraft flies on the datalink, and the parameter file sets the RC
  failsafe to continue the mission in AUTO (loss of the datalink as well still triggers the existing return
  failsafe).

### 7.3 Placement, wiring, connection table
- New placed parts: GPS on a mast on the pod top (clear of the propeller and payload), airspeed pitot at the nose
  for pushers and on a wing for tractors (out of the propeller wash), receiver and telemetry in the avionics bay.
  The packer includes their mass and position.
- Wiring: each run from the packer positions (routed along the fuselage, then the wing or boom), gauge from the
  run's current with an ampacity table, mass per metre per gauge; connectors by current class (e.g. XT30/XT60/XT90/
  AS150). Wiring mass replaces the wiring share of the systems fraction.
- Connection table (`electrical/connections.csv` + a build-sheet section): from, to, voltage, current, gauge,
  length, connector.

### 7.4 Mass and power budget
- Named parts replace `avionics_base_mass_kg` (0.227 kg), its per-kg MTOW growth and the 10 W base avionics power.
  `SYSTEMS_FRACTION` drops to what remains (servos and miscellaneous) until control hardware lands.
- Parameters: `ARSPD_TYPE` from the sensor, GPS type, serial port protocols for the GPS and telemetry, RC protocol,
  battery monitor type and scaling from the power module.
- Gate as phase 1, with a second deliberate re-baseline.

### 6.8 When a reference design stops closing (user, 2026-11-24: A)
The controller's-call constants (tip Mach, `MAX_MATCH_THROTTLE`, margins, loaded cell voltages, validation
tolerances, link margin) may be adjusted during the build when a reference design stops closing or the validation
gate is missed. Each change gets a recorded ruling with its reason and is reported to the user at the end. The
reference missions and every user decision in this document never change.

## 8. Left to the builder
- The generic chord distribution, airfoil and Reynolds bands of the propeller model (within the validation gate).
- The exact motor constants per winding (realistic, flagged), ESC and link catalogue contents.
- Search pruning and tie-breaking in the matching search; cache format.
- Wire routing detail, ampacity table source, connector classes, file layout under `electrical/`.
