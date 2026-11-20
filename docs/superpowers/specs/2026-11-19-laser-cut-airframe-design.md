# Laser-cut airframe kit — design

Date: 2026-11-19. Status: draft for approval.

## Intent

Every Group 1–2 design the app produces should come with a **kit you can physically build by laser cutting**:
cut files, a build sheet, and a 3D assembly of exactly those parts. The kit is the single description of the
airframe structure: what you cut, what the 3D viewer shows, and what the mass/CG/inertia used by the critic and
the flight-control (SITL) stage are all derived from the same flat part outlines. Today the structure is an
idealised model (ribs every 0.20 m, a tapered carbon frustum spar, and a composite OML shell carrying the "sized
airframe mass" as a lump); none of it can be bought or cut.

## Goals

1. Flat part outlines (2D, Build123d) generated deterministically from `layout.py` for:
   - wing ribs (left and right panels), with spar hole, LE/TE stock notches, lightening holes, wiring hole,
     aileron split at the hinge line, servo mounting plate in one rib bay per aileron;
   - tail ribs (horizontal and vertical surfaces) on their own spar tube, plus a ply mount plate to the tail boom;
   - fuselage: a flat **crutch** (horizontal plate through the fuselage axis) and **half-formers** (upper and lower)
     cut to the real OML cross-section at each station, inset by the covering thickness, tab-and-slot onto the
     crutch, notched for stringers, with pass-through openings for parts and lightening holes;
   - solid parts: motor firewall (two laminated 3 mm layers) and two **wing-mount side plates** — longitudinal
     (xz-plane) plates slotted into the crutch at each fuselage side, carrying the socket tube and the
     anti-rotation pin holes (a transverse bulkhead cannot hold a spanwise tube);
   - mounting plates: payload/gimbal, battery tray, avionics tray, and for hybrids fuel-tank and generator plates;
   - quadplane: boom clamp plates (saddle the spar and the boom) and lift-motor mount plates.
2. **Detachable left and right wings**: the fuselage carries a carbon **socket tube** fixed across it in the two
   wing-mount side plates, spanning the fuselage only; each wing's spar root plugs into it from the skin to the
   centreline (no rib carries a socket hole). The socket is the larger,
   stronger tube, so the weakest member at the joint is the spar, which is sized for the root load. Plus a carbon
   anti-rotation pin near the trailing edge; a nylon wing bolt into a captive-nut plate at each root; 3 mm ply root-rib doublers; a wiring connector at each root
   (listed on the build sheet, not modelled). On quadplanes each lift boom clamps to its wing and comes off with it.
3. **Real tube stock**: spar, joiner, tail boom, tail spars and lift booms are chosen from an editable list of
   commercial carbon tubes. Where the root tube does not fit the outboard ribs, a smaller tube is sleeved inside
   the larger at a step; each rib's hole matches the tube passing through it.
4. **Deterministic structural checks** (Zero LLM Math) for the spar at the root and at each step, and for the
   socket at the fuselage centreline, under the worse of (a) flight: limit load factor on the wing, (b) hover (quadplane): both rotors on a
   boom at hover thrust. The smallest stock tube that passes and fits is chosen; if none does, the kit is not
   produced and the run warns (see Fallback).
5. **Mass feedback**: kit part masses (area × thickness × density per part), tubes, stringers, LE/TE stock,
   covering (area × areal density) and a glue/hardware allowance replace the OML-shell lump mass in the as-built
   mass, CG and inertia on every design iteration.
6. **Cut files**: parts nested onto sheets of the bed size, one DXF per sheet, per stock; outer cuts, inner
   cuts and engraved part labels on separate layers; outlines compensated for kerf.
7. **Build sheet**: numbered parts list matching the engraved labels, sheet count per stock, tube cut list
   (OD/ID/length per piece, including socket and step sleeves), stringer and LE/TE stock lengths, rib and former
   stations, which split pieces join, covering area and estimated mass, hardware list (wing bolts, pins).
8. **Adjustable per design**: machine and stock settings are saved with the design, appear in the results and
   in the exported `design.json`, and re-running with changed settings regenerates the kit and masses.

## Non-goals

- Skin/covering geometry (covering is hand work; only area and mass are reported).
- True-shape nesting (bounding-box packing only; true-shape can come later).
- Group 3 designs, foam-core or composite-moulded construction.
- Detachable tail or tail boom (glued, permanent).
- Changing machine/stock settings through the design chat or the LLM (UI form only).
- Feeding kit mass back into the sizing loop (sizing keeps its airframe-mass model; the kit mass is checked
  against it via the existing as-built check).
- Motor/engine bolt patterns from datasheets (plates carry a centre mark; see "Left to the builder").
- G-code or laser-controller output (DXF only).

## Anti-goals (failure even if it technically works)

- A kit whose parts differ from what the 3D view, mass/CG or SITL used.
- A part larger than the bed, a tab that does not fit its slot at the stated kerf, or two parts overlapping on a
  sheet.
- A spar or joiner the checks pass that is not a tube you can buy from the stock list.
- Silent fallback: if the kit cannot be made, the run must say why.

## Constraints

- Deterministic, SI units internally (DXF written in millimetres), Zero LLM Math, frozen pydantic models,
  functions < 50 lines, files < 800 lines, no magic numbers, TDD, 80 %+ coverage.
- Build123d 0.13 (installed) for 2D/3D geometry; ezdxf 1.4.5 (installed, a Build123d dependency) for reading
  DXF back in tests. No new dependency unless the plan justifies it.
- `layout.py` remains the single source of positions; the kit reads from it, never the reverse.
- REQUIREMENTS.md §4.3's spar rule (`D_spar = 0.08 · t_airfoil`) is replaced by stock-tube selection with
  strength and fit checks; §4.3 is updated to say so. Rib pitch 0.20 m stays.
- The run must stay robust: kit failure never fails the design run.

## Defaults (all adjustable per design)

Machine "Hobby 600 × 400 laser": usable bed 600 × 400 mm, 5 mm edge and part spacing, kerf 0.15 mm.

| Stock | Thickness | Density | Default use |
|---|---|---|---|
| Birch ply | 3 mm | 680 kg/m³ | crutch, formers, bulkheads, plates, clamps, root doublers |
| Birch ply | 1.5 mm | 680 kg/m³ | wing and tail ribs |

Other defaults: stringers 3 × 3 mm balsa (160 kg/m³); LE 6 × 6 mm balsa, TE 5 × 15 mm triangular balsa; wing and
tail covering iron-on film 0.09 kg/m²; fuselage covering 1.5 mm balsa sheet (0.24 kg/m²); glue/hardware allowance
10 % of structure mass; carbon tube density 1600 kg/m³; carbon allowable bending stress 500 MPa with safety
factor 1.5; limit load factor 4.0; minimum spar insertion (fuselage skin to centreline) 3 × spar OD; tube fit
clearance ≤ 0.2 mm.

Default tube stock list (OD × ID, mm): thin-wall 6×4, 8×6, 10×8, 12×10, 14×12, 16×14, 18×16, 20×18, 22×20,
25×23; thick-wall 20×16, 24×20, 28×24, 32×28, 36×32, 40×36 (each ID equals the next smaller OD); solid rods 3, 4, 5 mm
for pins. Nested pairs (one tube's ID fits another's OD within the fit clearance) make the socket and the step
sleeves. With these, a 25 kg design needs a 32×28 mm spar in a 36×32 mm socket (usable root depth ≥ 32 mm), so
most Group 2 designs get a kit; a wing too thin for the tube it needs falls back with that reason.

## Decided with the user (2026-11-19)

Formers follow the modelled (rounded) fuselage shape; left and right wings detach; flat parts are the master copy
(approach A); hobby 600 × 400 laser with 3 mm and 1.5 mm birch ply as adjustable defaults; Group 1–2 coverage with
an extended thick-wall tube list and the joint turned round (fuselage socket, wing spars plug in) so most Group 2
designs get a kit.

## Approaches considered

- **A. Flat parts are the master copy — chosen.** A new `fabrication` module draws every part as a 2D outline
  from `layout.py`; the 3D assembly extrudes and places those outlines; masses come from them; nesting and DXF
  export run once for the final design. Most new code; the existing structure tests are reworked. Chosen because
  it is the only approach where the cut parts, 3D view and mass/CG/SITL are guaranteed identical.
- **B. Slice the existing 3D model.** Keep `structure.py`/`cad_packer.py` and section the solids at the end.
  Less disruption, but tabs/slots/notches/doublers/kerf are bolted on after the fact and there are two
  descriptions of the structure that can drift. Rejected.
- **C. Kit on demand, outside the loop.** A button generates the kit for a finished design. Zero run cost, but
  the critic and SITL never see the real masses. Rejected: breaks the mass-feedback goal.

Sub-choice in A: bounding-box shelf packing with 0°/90° rotation (simple, predictable, testable; ~20–30 % more
sheet than true-shape).

## The design

### Data model

- `FabricationSettings` (frozen pydantic, `extra="forbid"`): machine (bed width/height, margin, kerf), stock
  sheets (name, thickness, density, which part families it cuts), tube stock list, stringer/LE/TE stock,
  covering areal densities, allowances and structural constants above. Carried alongside `DesignAssumptions`
  through the run (same route: UI → `agent_service.run` → graph → CAD stage), stored on `AgentRunView`, and
  added to `DesignRecord` as an optional field with the defaults (`format_version` stays 1: no reader of the
  record exists yet).
- `KitPart` (frozen): id/label (e.g. `WR-03`, `WL-03`, `F-07U`, `C-1a`), family, stock name, 2D outline
  (Build123d `Face`, kerf-free nominal geometry), 3D placement (location in the aircraft frame), quantity.
- `TubePart`: role (spar, joiner, sleeve, tail boom, tail spar, lift boom, pin), OD, ID, length, placement.
- `Kit`: parts, tubes, linear stock (stringers, LE, TE with lengths), covering areas, warnings, and derived
  `MassItem`s for `mass_properties`.
- `CADArtifacts` gains optional `kit_dir`, `build_sheet_path` (relative-path handling added to
  `design_archive._CAD_PATHS`) and `kit_mass_kg`.

### Pipeline (inside the existing CAD stage)

1. `build.py` builds the layout as today.
2. If the design is Group 1–2 (`SizingOutput.dod_group`), tube selection and the structural checks run, and the
   kit's **flat outlines** are built (2D faces only, no 3D solids). Masses during the balance loop come from
   outline area × thickness × density, so the up-to-12 wing-station evaluations in `balance_design` stay cheap.
3. `cad_packer` packs components exactly as today (same z placement, `fits_radially`, containment). The socket
   tube is a collision obstacle (as the spar is today). Formers are **not** packing obstacles: they are placed
   after packing at a fixed pitch (builder's constant) plus the firewall and wing-mount stations, with an opening
   around every component they cross (component bounding box plus clearance). The crutch likewise has an opening
   under each component footprint (battery, trays, fuel tank), so components keep their current positions and
   the fuselage diameter and aero are unchanged. The kit's mass items replace `structure.py`'s ribs/spar and the
   OML-shell lump; the CG solve is unchanged.
4. **Residual mass**: landing gear (`GEAR_FRACTION` of the sized airframe mass) is not in the kit and stays a lump
   spread over the fuselage skin, as today. Lift booms become real kit tubes, so `BOOM_FRACTION` is no longer a
   lump when a kit exists.
5. **Booms and stock**: the layout's tail-boom and lift-boom diameters are minimums; the kit snaps each boom up to
   the nearest stock tube that passes. Booms are excluded from the VLM aero, so no flight numbers change; the 3D
   assembly and viewer show the kit tubes. The OML shell stays in the viewer as the outer surface; it no longer
   carries mass when a kit exists.
6. On the final design only: nest parts per stock onto sheets, export DXFs, write the build sheet, and include the
   kit solids in `assembly.step` and the viewer model.

### Fallback

Group 3 designs, a spar or joiner with no passing/fitting stock tube, or any part that cannot fit the bed even
when split: no kit is produced, the current shell mass model is used (as today), and the run reports a warning
naming the reason. The design run's status is unaffected.

### Geometry rules

- Kerf: outer outlines offset outward by kerf/2, inner cut-outs inward by kerf/2, at DXF export only (masses use
  nominal geometry).
- Tabs and slots: slot width = mating stock thickness (nominal; kerf compensation gives the press fit); tab
  length and spacing are named constants.
- Parts longer than the usable bed are split at a station chosen away from slots and openings, joined by a
  dovetail plus a glued doubler of the same stock; the build sheet lists the join.
- Formers: circular OML sections (`fuselage_radius_at`) inset by fuselage covering thickness, cut into upper and
  lower halves either side of the crutch; stringer notches at equal angular spacing.
- Ribs: airfoil at the station, inset by covering on top/bottom, LE/TE notches, spar hole at 25 % chord sized to
  the tube at that station plus fit clearance, wiring hole, lightening holes keeping a minimum web around edges and
  holes; ribs between 0.55–0.95 semi-span split at 75 % chord for the aileron (matching `openvsp_builder`'s
  `AILERON_ETA`/`AILERON_CHORD_FRACTION`); tail ribs split at 65 % chord for elevator/rudder
  (`TAIL_CONTROL_CHORD_FRACTION`).
- DXF: millimetres, layers `CUT_INNER`, `CUT_OUTER`, `ENGRAVE`; labels as vector text on `ENGRAVE`: open
  single-stroke polylines from a built-in stroke font (no TEXT entities, which laser software drops), one DXF
  group per label, shrunk to fit the part.
- Wing retention (R36): one M5 nylon bolt per side, spanwise on the wing's chord line midway between the spar and
  the pin (47.5 % chord), through an M5 clearance hole in the wing-box plate and in the root doubler
  (`WR-01`/`WL-01`), into an M5 nut held captive in a nut plate `NP-R`/`NP-L` (doubler stock: a 15 mm disc with
  the nut's hexagon) glued to the root doubler's outboard face. The bolt is driven from inside the fuselage. Every
  hole keeps the minimum web; the nut plate stays inside the root profile and clear of the spar, the pin and the
  pin boss; the washer seats clear of the socket, the pin and the plate bosses; the head is clear of every
  component. The build sheet gives the bolt length, its location and the order of operations.

### Structural checks

- Flight: root bending moment of one wing panel = n_limit · (MTOW·g/2) · ȳ, with ȳ = 4·s/(3π) of the exposed
  semi-span s (elliptic lift, no inertia relief: conservative). Stress σ = M·(OD/2)/I ≤ σ_allow/FS.
- Hover (quadplane): per wing, two rotors at hover thrust T = MTOW·g/4 each, acting at the boom's spanwise
  position; moment about the joint = 2T·(y_boom − y_joint).
- The joint is where the wing meets the fuselage skin at the wing's height: y_joint = √(R² − z_w²), with R the
  fuselage radius at the spar station and z_w the wing root height (`WING_Z_FRACTION`·D). The exposed semi-span
  s is measured from there.
- Checked for the spar at the joint and at each spar step, and for the socket at the fuselage centreline (moment
  arm measured from the centreline). Fit: tube OD ≤ local airfoil thickness at the
  spar station minus covering on both faces and a minimum rib web.

#### Added 2026-11-20 (external review, user-approved)

Each check below has every input in `FabricationSettings` or `SizingOutput`. A failure that no stock part or
doubler can fix raises `KitUnavailable` naming the check, and the run falls back to the shell model.

- **Centreline joint (carbon overwrap, the user's choice).** `SK-R` and `SK-L` meet in a glued mitre, and
  the glue line cannot carry the wing-root moment. The mitre is overwrapped with wet-laid carbon cloth. Treat
  the wrap as a thin tube on the socket OD: n layers of `overwrap_layer_thickness_m`. Choose the smallest n
  (≤ `overwrap_max_layers`) with M_centre·c/I_wrap ≤ `overwrap_allow_pa`/FS, where M_centre is the socket
  moment at the centreline (the worse of flight and hover, as above) and c = OD/2 + n·t.
  - Lap per side: L = max(2·OD, σ_wrap·n·t·FS/`epoxy_lap_shear_pa`).
  - The whole wrap (2L) must fit between the two side plates' inner faces.
  - Wrap mass: π·(OD + n·t)·n·t·2L·`overwrap_density_kg_m3`. It counts toward kit mass and the CG, it is a
    packing obstacle at its wrapped OD, and the build sheet lists it.
  - The spar roots inside the sockets are not glued to each other: the wings stay detachable.
- **Spar torsion.** Two torque sources, combined with the bending stress σ by
  (σ/σ_allow)² + (τ/`carbon_shear_allow_pa`)² ≤ 1/FS², at the joint, at every step and at the boom clamp:
  - Hover pitch control (quadplane): one rotor at `vtol_thrust_per_motor_n` and its fore/aft partner at
    zero, giving T = thrust × the longer boom arm from the spar. This torque is applied at the boom station
    and carried inboard. Yaw uses rotor reaction torque, which is small and ignored.
  - Flight: q_max·c²·|Cm0| per unit span plus the lift's offset from the spar, n_limit·L'(y)·(x_spar − x_ac)·c,
    integrated from the tip inboard. q_max is set by the design's maximum speed. Cm0 comes from the airfoil
    when it is known; otherwise use a named conservative default.
  - Shear stress: τ = T·(OD/2)/J, J = π(OD⁴ − ID⁴)/32.
- **Tail-surface load (R35).** A tail surface lifts at most q_max·S·`TAIL_CL_MAX` (q_max at V_A, R30), with no
  extra n_limit factor: CL_max at V_A is already the ceiling of tail lift, and FS applies through the allowable.
  Limitation: high-speed gust cases on the tails are not modelled and could exceed this load.
- **Tail boom strength.** Today the boom is chosen by diameter alone (R13); now it is also checked:
  - Bending: M = q_max·S_tail·`TAIL_CL_MAX` (R35) × the tail arm from the boom mount, using the horizontal
    and vertical tails' resultant.
  - Torsion: the fin's load × the height of its centre of pressure above the boom.
  - Same interaction check as the spar. If the largest tube ≤ layout diameter fails, the kit is unavailable.
- **Tail spar strength (R35).** Each surface's spar (horizontal tail, fin) carries its R35 load spread elliptically
  over its semi-span (each horizontal-tail panel half of S_h, the fin all of S_v over its height). The bending
  moment at the spar's root (the tail root, R20, or the fin start) must pass the tube check. The spar is the
  smallest stock tube or rod that passes and fits the root's usable depth; it runs out to where the depth stops
  admitting it (the moment only falls outboard). If none passes and fits, `KitUnavailable("tail spar …")` names
  the surface and the station.
- **Socket bearing in the side plates.** Each plate carries its wing's shear load: F = n_limit·MTOW·g/2 in
  flight (and the hover equivalent). The check is σ_b = F/(OD·t_plate) ≤ `ply_bearing_allow_pa`/FS.
  - If it fails, add a 3 mm socket-boss doubler (`WM-RD`, `WM-LD`) glued over the hole, an annulus of
    radial width `SOCKET_BOSS_WIDTH_M`, and recheck with the doubled thickness.
- **Crutch rail width.** Every side rail of the crutch beside a component opening must be at least
  `crutch_rail_min_m` wide. It carries fuselage bending, whereas MIN_WEB_M is only a cut-quality minimum.
  The kit-aware pod sizing (R23) includes this rule.

New settings (defaults, all adjustable):
- `carbon_shear_allow_pa` = 60 MPa (in-plane shear of roll-wrapped tube, conservative)
- `ply_bearing_allow_pa` = 15 MPa (birch ply edge bearing, conservative)
- `ply_shear_allow_pa` = 7 MPa (R33: birch aviation ply panel shear, conservative; used for shear-out checks such as
  the R32 web under the socket, whose boss grows below the hole in steps until the web passes)
- `overwrap_layer_thickness_m` = 0.25 mm (200 g/m² cloth, wet lay-up)
- `overwrap_allow_pa` = 250 MPa (hand-laid 0/90 cloth, conservative)
- `overwrap_density_kg_m3` = 1500
- `overwrap_max_layers` = 12 (R29: raised from 8, which g2 needed 9 of)
- `epoxy_lap_shear_pa` = 15 MPa
- `crutch_rail_min_m` = 8 mm

### UI

- An expander "Laser kit settings" next to the assumptions panel, editing `FabricationSettings` (bed, kerf,
  sheets, tube list as an editable table).
- The Files tab lists the kit DXFs and the build sheet; the zip already includes every run file.
- Results show kit mass, sheet count per stock and any kit warning.

### Outputs (run folder)

`kit/sheets/<stock>-<n>.dxf`, `kit/build_sheet.md`, `kit/parts.csv`. `assembly.step` and `viewer_model.json`
contain the kit parts. In the viewer the kit is two parts, `structure (ply)` and `structure (carbon)`, so X-ray view tells tubes from plywood.

## Calls I made (please check)

- `format_version` of `design.json` stays 1 when the fabrication settings are added.
- Tail and tail boom are glued, not detachable.
- Covering split: film on flying surfaces, 1.5 mm balsa sheet on the fuselage.
- Glue/hardware allowance 10 %; σ_allow 500 MPa, FS 1.5, n_limit 4.0.
- Kit failure falls back to the shell model with a warning instead of failing the run.
- Kit settings are edited in the UI only, not via chat.
- Crutch and formers get openings around components rather than moving components (no change to packing, fuselage
  size or aero).
- Formers are placed after packing; only the socket tube is a new packing obstacle.
- Landing gear stays a lump over the fuselage skin; lift booms become real tubes.
- Boom diameters snap up to stock; booms are outside the aero model, so no aero change.
- Wing mounts are longitudinal side plates; the joint is at the fuselage skin.
- Balance-loop masses come from 2D outlines; 3D solids, nesting and DXFs only for the final design.

## Left to the builder

- Module layout inside `src/tools/fabrication/` and file splits.
- Exact tab length/spacing, lightening-hole shapes, minimum web width, label font size.
- The splitting station selection algorithm and dovetail proportions.
- Motor and engine plates: centre mark plus a generic slotted cross pattern (16/19/25 mm); exact bolt patterns
  are drilled by hand.
- Whether nesting uses a small in-repo shelf packer or a justified dependency.
- How existing structure tests in `tests/test_aero_cad.py` are migrated.
