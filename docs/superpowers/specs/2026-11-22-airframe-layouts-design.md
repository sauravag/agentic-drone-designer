# Airframe layouts — design

**Date:** 2026-11-22 · **Status:** draft for approval (revised after the designer's review, 2026-11-22)

## Intent

Today every design is a pod-and-boom tractor with a conventional tail (H-tail plus one fin on a centre boom); a
quadplane adds two lift booms that carry only rotors. The designer wants to pick one of four further layouts that
real Group 1–3 ISR aircraft use, and get the whole pipeline for it: sizing, aero and stability, CAD, the 3D view,
ArduPilot parameters with the right surface mixing, the SITL check, and the laser-cut kit.

Layouts (the designer chooses; there is **no auto-pick**, and today's layout stays the default):

| id | Layout | Tail | Booms | Propeller |
|---|---|---|---|---|
| `pod_boom_tractor` (default) | Today's pod-and-boom tractor | H-tail + 1 fin | 1 centre boom | tractor, nose |
| `single_boom_vtail_pusher` | Single-boom V-tail pusher | V-tail (2 panels) | 1 centre boom | pusher **at the boom end, behind the V-tail** |
| `twin_boom_htail_pusher` | Twin-boom H-tail pusher | H-stab across the boom ends + 2 fins | 2 outboard booms; on a quadplane **the lift booms run aft and carry the tail** | pusher, pod tail, between the booms |
| `twin_boom_inverted_v_pusher` | Twin-boom inverted-V pusher | inverted V: panels rise from the boom ends to an **apex above** the booms (Aerosonde / Penguin B) | 2 outboard booms; on a quadplane **the lift booms run aft and carry the tail** | pusher, pod tail |
| `flying_wing` | Flying wing | none: winglet fins, elevons | none (VTOL: lift booms only) | pusher, pod tail |

Every layout works with both topologies (`fixed_wing`, `quadplane_vtol`): ten combinations. A combination is
accepted by requirement validation only once its phase has landed and it has a sizing-plus-CAD closure test; until
then validation rejects it with a clear message (no untested combination is reachable).

## Goals

- One layout field on the requirements, honoured by every stage, with the same five critic gates.
- Physically honest sizing per layout: tail sizing, tail and boom mass, boom drag, pusher installation loss,
  flying-wing pitch stability from sweep, washout and a reflexed airfoil.
- A laser-cut kit for every layout (the kit stays Group 1–2 only, as today).
- ArduPilot parameters with the right servo functions (V-tail, elevons) and a SITL check that flies them.

## Non-goals

- Automatic layout selection, or a layout lever in the closure explorer (the layout is the designer's choice).
- Canards, tail-sitters, tilt-rotors, blended bodies.
- New materials (EPP/EPO foam, composites): parked until the designer decides; the kit stays plywood + carbon.
- Wind-tunnel-grade flying-wing aerodynamics: VSPAERO plus the existing gates are the bar.

## Anti-goals (a failure even if it "works")

- A layout that sizes or flies differently from what its kit and CAD actually build (e.g. a V-tail sized as an H-tail).
- Silently changing today's designs: phase 1 is a pure refactor (sizing output, CAD/VSPAERO inputs, kit parts and
  ArduPilot parameters of the reference designs unchanged); phase 2 deliberately re-baselines every layout, the
  tractor included, to the explicit tail-boom mass and boom drag, updating the reference values once and saying so
  in the changelog (designer, 2026-11-22: comparisons between layouts must be fair).
- Four copies of the pipeline: per-layout code only where the physics or the parts genuinely differ.

## Approaches considered

1. **Shared layout model, phased — chosen by the designer (2026-11-22).** A small, fixed `LayoutSpec` per layout
   (tail kind, boom arrangement, propeller position) read by every subsystem; geometry becomes general (a list of
   tail surfaces with dihedral, a list of booms, a propeller station). Shipped in five phases, each merged and tested
   on its own.
2. Shared model, all at once: same design, one long branch. Rejected: nothing usable until the end, huge review.
3. Separate code per layout: rejected: about 4× the code across seven subsystems, drifting apart.

## Design

### 1. Requirements and the layout description

- `OperationalRequirements.airframe_layout: Literal[<the five ids>] = "pod_boom_tractor"`. Orthogonal to
  `airframe_topology`. Parser: maps "twin-boom", "V-tail", "inverted V", "H-tail", "flying wing", "pusher" phrases
  (the LLM draft gains an optional `airframe_layout`); no stated layout means the default. Design chat can change it.
- `LayoutSpec` (frozen dataclass, one module, e.g. `src/tools/sizing/layouts.py`) — the single place that knows what
  each layout is:
  - `tail: Literal["conventional", "v", "inverted_v", "twin_fin_h", "none"]`
  - `booms: Literal["centre", "twin", "none"]`; on a quadplane, `twin` booms **are** the lift booms extended aft
    (two booms, never four: collinear booms are one boom)
  - `propeller: Literal["tractor_nose", "pusher_pod", "pusher_boom_end"]`
  - `wing: Literal["straight", "swept_reflex"]`
  - the static-margin knob the critic uses (`tail_volume_h`, or `wing_sweep_deg` for the flying wing)
  - the static-margin band and the balance target: 8–14 %, target 11 % (today's) for tailed layouts; **5–10 %,
    target 7 %** for the flying wing (designer, 2026-11-22). The critic gate, the battery-slide target in the
    packer and the pipeline all read the band from the layout.
  Subsystems branch on these properties, never on the layout id.

### 2. Sizing (`src/tools/sizing/`)

- **Tail surfaces.** `size_geometry` keeps the tail-volume knobs and produces a tuple of `TailSurface` (name, planform
  area, span, dihedral, arm, control kind) instead of fixed H/V fields:
  - conventional: as today.
  - V / inverted V: total area `S_h + S_v`, dihedral `±atan(sqrt(S_v/S_h))` (projected-area method), 2 panels;
    control kind ruddervator.
  - twin-fin H: H-stab `S_h` spanning the boom separation; 2 fins of `S_v/2` each.
  - none: winglet fins sized from the fin volume `V_v` with the arm from the CG to the swept tip; no horizontal tail.
- **Tail arm separated from pod length (phase 1).** Today the tail arm is `TAIL_ARM_FRACTION` (0.6) × fuselage length
  (`mass_model.py`). Phase 1 makes the tail arm its own sizing quantity, computed for the tractor by exactly today's
  formula (outputs unchanged); other layouts set it from their own geometry, so a short twin-boom or flying-wing pod
  no longer drags the arm with it.
- **Tail arm and booms.** Centre boom: as today. Twin booms: boom separation = the greater of
  propeller diameter + clearance (pusher disc between the booms) and, on a quadplane, the existing lift-boom spacing
  (rotor clearance); the tail arm comes from the tail-volume sizing as today, setting the boom length.
  On a quadplane the rotors stay where the hover balance puts them today and the lift booms simply extend aft to
  the tail; the tail span is the boom spacing and the tail area still comes from the tail volume (*my call*).
- **Tail aspect-ratio bound.** With area from tail volume and span from boom spacing, the tail aspect ratio is an
  output, so it is bounded (the builder sets the band, roughly 3–6). Too low (big area, narrow spacing): the surface
  overhangs past the booms until the band is met. Too high (small area, wide spacing): the tail arm is shortened,
  which raises the required area. Each adjustment is reported in the sizing output.
- **Mass and drag re-baseline (phase 2), naming each lumped term it replaces** so nothing is counted twice:
  - the tail-boom share of the fuselage wetted area and fuselage areal mass (the boom is folded into the pod today)
    → an explicit tube per boom (count × length × tube mass per metre; wetted area per boom); the pod keeps only its
    own length;
  - `BOOM_FRACTION` (lift booms, 4 % of MTOW) → explicit lift-boom tubes from their real length; on twin-boom
    quadplanes the extended lift boom is counted once, not as a lift boom plus a tail boom;
  - `QUADPLANE_DRAG_PENALTY` (0.008, booms + stopped rotors) → boom wetted-area drag per boom plus a stopped-rotor
    term that is what remains of the lump;
  - `GEAR_FRACTION` → mass from the shared gear geometry where the layout has gear (section 3), unchanged where it
    has none.
  Tail mass comes from the actual surface list.
- **Drag.** Lifting-surface drag from the actual tail areas; boom drag as above; a pusher installation factor
  on the drivetrain efficiency — a new `DesignAssumptions.pusher_efficiency_factor`, default 0.95 (*my call: a
  propeller in the fuselage/boom wake loses a few percent; shown and editable like the other assumptions*).
- **Flying wing.** New knobs `wing_sweep_deg` (default 25°, bounds 15–35°) and the washout derived from it so the
  wing trims at the cruise CL with the design static margin (Panknin-style twist for a swept tailless wing). A
  reflexed airfoil family is added to airfoil selection for `swept_reflex` wings only (candidates from the
  AeroSandbox/UIUC set, e.g. MH 45 / MH 60 / MH 62; the builder confirms which are in the bundled database and picks
  by Reynolds number as today). The pod becomes a short centre body carrying payload, battery and motor.
- **Flying-wing trimmed CLmax.** Today wing area comes from the airfoil's untrimmed `cl_max_3d`
  (`mission_solver._wing_area_m2`). For `swept_reflex` wings the stall constraint uses a **trimmed** CLmax, reduced
  for washout (tips at lower incidence) and for the up-elevon needed to trim at that CL, so the wing comes out large
  enough to honestly meet the stall requirement.
- **Schema.** `SizingOutput` gains `airframe_layout` and `tail_surfaces`; `horizontal_tail_area_m2`,
  `vertical_tail_area_m2` keep their meaning as **effective areas**: Σ S·cos²Γ (horizontal) and Σ S·sin²Γ (vertical)
  over the tail surfaces, which is what the V-tail sizing formula already assumes; `tail_arm_m` is the tail arm.
  0 is allowed (`NonNegative`, the flying wing) so existing readers and old records keep working; `wing_sweep_deg` and `washout_deg` are added (0 for straight wings).

### 3. Aero, stability and CAD (`src/tools/aero_cad/`)

- `AircraftLayout` holds the general geometry: `tail_surfaces` (planform + dihedral + position), `booms` (list),
  `propeller` station, wing sweep and washout. `SurfacePlanform` gains dihedral (negative for the inverted V) and
  sweep.
- VSPAERO model: each tail surface is a wing component with its dihedral; booms are pods; the pusher propeller is a
  disk at its station (display, as today). Control derivatives per layout:
  - conventional and twin-fin H: as today (elevator, rudder; twin fins deflect together).
  - V / inverted V: ruddervators deflected symmetrically give `Cm_δe`, antisymmetrically `Cn_δr` (and the roll
    coupling is recorded); the inverted V's yaw-roll coupling has the opposite sign to the V-tail, which the gate
    checks accept. The first inverted-V test asserts that sign from VSPAERO, which also proves the apex is above.
  - flying wing: elevons symmetric give `Cm_δe`, antisymmetric `Cl_δa`; fixed winglets, no rudder:
    `cn_delta_r_per_rad` becomes Optional and is `None`.
- **Flying-wing trim check.** Static margin alone does not prove a tailless wing trims. The aero stage requires the
  pitching moment to reach zero at the cruise CL and at the near-stall CL used for sizing, each with a symmetric
  elevon deflection inside a trim limit (a fraction of elevon travel, leaving authority to manoeuvre). A miss is a
  stage failure through the existing gate path, and the critic acts on it with the layout's knobs (sweep, washout);
  no new gate.
- Stability corrections (pod Cm_α and Cn_β) use the actual pod; `_check_control_conventions` is generalised to the
  control kinds.
- CAD and packing: the cruise motor is placed at its station (nose, pod tail, boom end); the tractor firewall moves
  to the pod's aft bulkhead for `pusher_pod`; collision obstacles include every boom and the propeller disc. New
  geometric checks: the pusher disc clears the booms (twin) or the V-tail trailing edge (boom-end), and the
  propeller tip clears the ground, else the CAD stage reports a collision-class failure the critic can act on.
- **One gear geometry.** Today gear exists only in the JSBSim exporter; CAD and the kit carry it as `GEAR_FRACTION`
  mass spread over the pod. One gear description (none, skid, or wheeled with leg heights) is produced once and read
  by sizing (mass), CAD (propeller clearance at the rotation angle) and the JSBSim exporter (contact points).
  Hand-launched, belly-landing Group 1 aircraft have **no gear**: their check is the propeller disc against the
  belly line on landing (a folding propeller is allowed and noted on the build sheet).
- The balance loop is unchanged in principle (the tail stays rigid with the wing group: booms clamp to the spar).
- Viewer: no change (it renders whatever meshes the packer produces).

### 4. Flight control and SITL (`src/tools/flight_control/`)

- Servo functions per control kind (ArduPilot `SERVOn_FUNCTION`): ailerons 4; elevator 19; rudder 21 (twin fins on a
  Y-lead); V-tail 79/80 (`VTail left/right`); elevons 77/78 (`Elevon left/right`); throttle 70; lift motors 33–36
  as today. The inverted V reverses the yaw sense in the mixing (servo reversal), verified in SITL.
- **Per-surface aero tables, no un-mixing.** Today JSBSim has one elevator and one rudder channel. If the bridge
  un-mixed servo outputs with the same mix the parameters assume, a wrong reversal would cancel out and SITL would
  pass. Instead the JSBSim model gets one control channel and one set of coefficients per **physical surface** (left
  and right ruddervator, left and right elevon, ...), each taken from VSPAERO by deflecting that surface alone. The
  bridge maps each servo output straight to its surface, so ArduPilot does all the mixing and the signs come from
  geometry. A test flips one servo reversal and asserts the SITL check fails. Cruise thrust acts at the propeller
  station.
- `flight_model` damping derivatives are built from the actual tail surfaces (projected areas and arms); the flying
  wing uses wing-only estimates (sweep strip theory for `C_mq`, winglet fins for `C_nr`, `C_Yβ`).
- The critic and the gates are unchanged except: the static-margin fix uses the layout's knob (`wing_sweep_deg` for
  the flying wing); the "rudder power" row reads "n/a (no rudder)" for the flying wing.

### 5. Laser-cut kit (`src/tools/fabrication/`)

Shared parts stay shared (wing ribs, spar, sockets, crutch, wing-mount plates, lift booms and clamps). New or
changed parts per layout (part prefixes are the builder's call, following the existing scheme):
- `pusher_pod`: aft firewall and motor plate at the pod tail (the nose firewall becomes a plain former).
- Twin booms: two tail-boom tubes clamped to the spar with the existing saddle design; a tail-boom load check per
  boom (R26 split across two booms); H-stab spar and ribs spanning the boom ends with a fin mount at each boom end;
  inverted-V panels rising from each boom end to a ply apex joiner **above** the centreline.
- Twin-boom load path (new checks). With a centre boom the tail's pitching load goes into the pod; with booms clamped
  to the spar it twists the spar between the booms. Added: spar torsion from tail load × tail arm per boom, clamp
  slip torque of the round tube in its saddle against that moment, and that spar twist added to the reversal check.
- Twin-boom quadplanes (H-tail and inverted V): the lift-boom tube is sized for the combined hover bending and tail
  loads (a new combined load case: hover thrust at the hubs plus tail forces at the aft end, worst of the two flight
  states). It first lands in phase 3.
- Single-boom V-tail pusher: V-tail root block at the dihedral angle on the boom, two panel rib sets; boom-end motor
  plate and clamp; the motor wiring runs along the boom (build-sheet note).
- Flying wing: swept panel ribs (ribs parallel to the flight direction, swept spar), washout built into the rib
  incidences, elevon hinge and servo cut-outs reusing the aileron design, winglet fin ribs and a tip mount; no tail
  boom; a short centre-body pod; an **angled centre joiner** joining the two swept spar tubes, checked for the full
  root bending moment and its bearing on the tubes.
- **Swept-wing stiffness correction.** `wing_stiffness.py` assumes a straight wing with the spar on the aerodynamic
  centre. On a swept spar, bending twists the tips nose-down and the elevons also carry pitch. The flying-wing kit
  adds the swept beam's bending-torsion coupling to the reversal twist and checks elevon effectiveness in roll and
  pitch. If the correction or the joiner check is not in place, the flying wing ships with **no kit** (kit
  unavailable, with the reason) rather than passing a check it was never derived for (*my call: keep the kit in
  scope, with this as the fallback*).
- Every other layout's kit passes the same rules as today (bed fit, stock tubes, bending/torsion, socket bearing)
  plus the new checks above; if a check cannot be met the kit is unavailable and the run says why, as today.

### 6. UI, report, agent

- Form: a "Layout" select next to "Airframe", with a one-line description of each layout.
- Sizing summary and report: a Layout row and the tail-surface table; the PDF three-view shows the layout
  automatically (it draws the display mesh).
- Explorer: no layout lever. Parser and design chat: the new field.

## Phases (each its own branch, merged when green: host + Docker suites, plus a full SITL pipeline run per new layout)

1. **Refactor**: the layout field (default only), `LayoutSpec`, general tail surfaces/booms/propeller station
   through sizing, VSPAERO, CAD, flight control and kit; the tail arm separated from pod length; the shared gear
   description; per-surface JSBSim channels. Reference designs unchanged in every output.
2. **Single-boom V-tail pusher**, opening with a **feasibility probe**: the G1 and G2 reference cases through sizing
   and packing with the motor at the boom end, reporting whether the battery slide reaches the 11 % static-margin
   target and whether the propeller clears the ground (G2 gear) or the belly line (G1, no gear). The result goes to
   the designer before the rest of phase 2 is built. Then V-tail sizing and ruddervators, boom-end pusher, V-tail
   kit, plus the re-baseline of every layout (the lumped terms named in section 2).
3. **Twin-boom H-tail pusher** (twin booms, pod pusher, twin-fin H kit, lift booms carrying the tail and the
   combined load case on quadplanes, tail aspect-ratio bound, spar torsion and clamp slip checks).
4. **Twin-boom inverted-V pusher** (inverted V with the apex above and its apex joiner; reuses phase 3's twin booms).
5. **Flying wing** (sweep, washout, reflex airfoils, trimmed CLmax, trim check, winglets, elevons, swept kit with the
   stiffness correction and centre joiner, or no kit).

Each phase ships reference designs for **both** of its combinations (fixed-wing and quadplane), each with at least a
sizing-plus-CAD closure test, enables those combinations in validation, and updates README/REQUIREMENTS.

## Left to the builder

- Exact constants (clearances, boom diameters, winglet aspect ratio, apex joiner size, the tail aspect-ratio band,
  the elevon trim limit), part prefixes, the reflex
  airfoil list after checking the bundled database, and the internal shape of `LayoutSpec` — within the rules above.
- Whether a layout that cannot meet a geometric check (e.g. propeller ground strike) fails sizing or the CAD stage —
  either is fine if the run reports it clearly and the critic can act.

## Decisions log

- 2026-11-22 (designer): layouts as listed; designer chooses, no auto-pick; tractor stays default; scope includes
  the laser-cut kit for every layout; shared layout model, phased; single-boom pusher has the propeller at the boom
  end behind the V-tail; flying wing = swept + washout + reflexed airfoil + winglet fins.
- 2026-11-22 (designer, after the builder check): re-baseline every layout including the tractor to explicit
  boom mass and drag (phase 2); flying wing static margin 5–10 %, target 7 %.
- My calls (correct me): every layout × both topologies; twin-boom quadplanes have two booms (the lift booms carry
  the tail); rotors stay at today's hover-balance stations, booms extend aft, tail span = boom spacing; pusher
  efficiency factor 0.95 as an editable assumption; flying-wing winglets are fixed (no rudders); old tail fields
  kept as effective areas for compatibility.
- 2026-11-22 (designer's review), all adopted: inverted V apex above the booms, with a VSPAERO sign test; flying-wing
  trim check and trimmed-CLmax stall sizing; swept-wing stiffness correction and angled centre joiner (else no
  flying-wing kit); twin-boom spar torsion and clamp slip checks feeding the reversal twist; per-surface JSBSim aero
  tables so SITL catches mixing sign errors; tail arm separated from pod length in phase 1 and each replaced lump
  named for phase 2; a boom-end pusher feasibility probe opening phase 2 and one shared gear geometry; tail
  aspect-ratio bounds; every enabled combination closure-tested, untested ones rejected by validation; legacy tail
  fields defined as effective areas.
