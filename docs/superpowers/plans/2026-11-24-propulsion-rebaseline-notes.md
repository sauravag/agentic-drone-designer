# Propulsion matching phase 1: re-baseline notes (2026-11-25)

These notes cover what moved when the fixed propulsive efficiencies were replaced by the matched blade-element
propeller + motor model (Tasks 1-8), and why. "Before" is master `df94b76` (the branch base). "After" is
`feat/propulsion-matching`. Every number below was regenerated from the code, not edited by hand:

- sizing and BOM: `size_aircraft` + `select_bom`, run on both checkouts on the host;
- pipeline, kit and SITL: the six reference designs through the full `python -m src.main --spec <case>.json`
  pipeline (aero/CAD, kit, SITL) in Docker, on both checkouts. The host `--spec` runs (no aero/CAD toolchain) gave
  the same sizing on the branch, and all six converged there;
- golden snapshots: `scripts/regenerate_layout_golden.py`, run on the host and then in Docker for `vsp_geometry`.
  The host and Docker texts are identical. They now live in `tests/data/golden/propulsion_matching/`;
- avionics baseline: `scripts/regenerate_avionics_baseline.py`;
- flight-control fixtures: `scripts/generate_flight_control_fixtures.py` (Docker; g1, g2, g1_vtail). The
  pre-matching records are kept as `legacy_g1` and `legacy_g2` for the legacy-path tests.

## Why the designs moved (all six)

1. **Matched efficiencies replace 0.68 and 0.72 x 0.85.** Sizing used a fixed forward efficiency of 0.68 (x 0.95
   for pushers) and a hover figure of merit of 0.72 x lift-motor efficiency 0.85, which is 0.612 electrical. Now
   the end-to-end efficiency at each flight point (propeller x motor x ESC) comes from the matched set. It is worked
   out at that point's thrust, air density, viscosity and speed of sound.
   - Forward flight on the small electric designs comes out below the old 0.68, worst in the climb: G1 climb 0.51
     and cruise 0.63; G1 V-tail climb 0.48 and cruise 0.59; G1 hand launch climb 0.36 on a folding propeller. These
     designs need more energy.
   - Hover comes out above the old 0.612 (G2 0.68), which helps the quadplanes. Their forward legs still drop,
     though (G2 climb 0.56, cruise 0.67).
   - The G3 forward legs come out at 0.66-0.73.
2. **Matched lift hardware mass.** Lift motors, rotors and ESCs are now real catalogue parts, sized for the matched
   rotor:
   - G2 rotor 0.603 -> 0.711 m (22x7 -> 28x9 in); VTOL system 1.51 -> 2.31 kg.
   - G3 rotor 1.127 -> 1.575 m (36x12 -> 62x24.8 in); VTOL system 10.4 -> 37.8 kg.
   - The heavier lift system grows the airframe with it.
3. **ESCs count in the mass.** Each motor now has its own ESC in the BOM, with its mass:
   - G1: 60 A 2S;
   - G2: 60 A 12S cruise + 4 x 30 A 12S lift;
   - G3: 200 A + 4 x 120 A, 250 V.
4. **Cell-built packs.** A pack is now built from a catalogue cell (S x P) instead of picked as a named pack with a
   voltage. The series count follows the matched motor (for example G1 6S -> 2S).
5. **No 15 % BOM pack margin.** The old battery matcher picked a pack 1.15 x the sized energy and peak power. Now P
   rounds up and nothing else is added (spec 6.3). This lowers the as-built mass relative to the sizing; the sizing
   itself does not move.
6. **G3 only.** The hybrid now flies matched HV lift motors and a 15 kW genset (generator 9.5 -> 20.2 kg). Together
   with the larger rotors, this more than doubles the empty mass (59 -> 137 kg). The parked-rotor drag and the
   Re-1e6 airfoil switch stay unchanged (user, 2026-11-25), so G3 re-baselines at 273.6 kg after 6 matching passes.

## Per design: before -> after

MTOW, empty mass and battery mass come from the sizing; the pack and the matched sets come from the BOM; the kit
outcome and the pipeline/SITL result come from the Docker `--spec` run. In the table, "no kit" with no reason means
Group 3 ("laser-cut construction covers Group 1-2 designs only"); for the other no-kit reasons, see the next section.

| Design | MTOW kg | Empty kg | Battery kg | Pack | Cruise set | Lift set | Kit | Pipeline / SITL |
|---|---|---|---|---|---|---|---|---|
| g1 | 4.158 -> 4.446 | 1.814 -> 1.930 | 1.344 -> 1.516 | 21700 6S4P -> 18650 energy 2S13P | 2814-900KV + 11x7 -> 2212-1713KV + 10x8 + ESC 60A 2S | - | built 0.905 kg -> **no kit** | converged, SITL pass -> converged, SITL pass |
| g2 | 19.879 -> 24.826 | 8.262 -> 10.937 | 8.617 -> 10.889 | 21700 12S10P -> 18650 high-drain 12S21P | 4325-400KV + 16x10 -> 5025-203KV + 18x12.6 + ESC 60A 12S | 4 x 6215-180KV + 22x7 -> 4 x 6215-111KV + 28x9 + ESC 30A 12S | built 4.271 kg -> **no kit** | converged, SITL pass -> **not converged** (see Gate status); saved-fixture SITL passes |
| g3 | 136.565 -> 273.649 (fuel 34.8 -> 71.2) | 59.202 -> 137.046 | 22.572 -> 45.442 | LiPo HV 6 kWh -> LiPo pouch 50S30P | AX-15 + 32x16 -> AX-40-25KV + 50x35 + ESC 200A 250V; genset 8 kW -> 15 kW | 4 x 12030-60KV + 36x12 -> 4 x 18050-16KV + 62x24.8 + ESC 120A 250V | no kit -> no kit | not converged -> not converged, both on the L/D gate (see Gate status); no SITL either side |
| g1_vtail | 4.681 -> 5.074 | 2.173 -> 2.343 | 1.509 -> 1.731 | 21700 6S4P -> 18650 high-drain 4S10P | 2814-900KV + 11x7 -> 2212-966KV + 10x7 + ESC 40A 4S | - | built 1.058 -> built 1.129 kg | converged, SITL pass -> converged, SITL pass |
| g2_vtail | 21.545 -> 26.498 (Group 2 -> 3) | 8.915 -> 11.426 | 9.631 -> 12.072 | 21700 14S12P -> 18650 high-drain 12S23P | 4325-400KV + 16x10 -> 5025-203KV + 18x12.6 + ESC 60A 12S | 4 x 6215-180KV + 22x7 -> 4 x 6215-111KV + 28x11.2 + ESC 40A 12S | no kit (anti-rotation pin, 300 N) -> no kit | not converged (shell mass 27.41 kg > Group 2) -> converged (sized as Group 3), SITL pass |
| g1_hand | 5.861 -> 7.104 | 3.338 -> 4.019 | 1.523 -> 2.085 | 21700 6S4P -> 18650 energy 3S12P | 2814-900KV + 11x7 -> 2814-1342KV + 11x7 folding + ESC 80A 3S | - | built 2.482 kg -> **no kit** | converged, SITL pass -> converged, SITL pass |

What else moved each design, on top of the common causes above:
- **g1**: climb efficiency 0.51 and cruise 0.63 against the old 0.68 add 13 % battery. It now flies a 2S pack on a
  1713 KV motor with a 10x8 propeller.
- **g2**: the forward legs (0.56-0.67 against 0.68) outweigh the hover gain (0.68 against 0.61). The lift hardware
  adds 0.8 kg. The heavier wing's root torque overloads the kit's anti-rotation pin, so the as-built mass comes
  from the shell model.
- **g3**: the lift hardware is 3.6 x heavier, on matched HV windings and 62 in rotors.
- **g1_vtail**: the pusher factor applies on top of the lower matched efficiencies (climb 0.48, cruise 0.59).
- **g2_vtail**: the same as g2, plus the pusher factor. It crosses 24.95 kg into Group 3, so it gets no kit by rule.
  Its as-built Group 3 now matches its sizing group, so the run converges.
- **g1_hand**: the folding propeller's low climb efficiency (0.36) in the hand-launch climb. The wing grows to
  4.25 m span, and its torsional stiffness no longer clears aileron reversal for the kit.

## Kit outcome (user decision, 2026-11-25: "accept, kit fixes later")

Kit code and kit limits are unchanged. The designs that no longer build a kit report "no kit" plus the reason:

| Design | No-kit reason |
|---|---|
| g1 | C-1: the slot for F-07 leaves less than 2 mm of crutch |
| g2 | anti-rotation pin: 356 N from the spar's root torque overloads even a 5 mm rod with boss doublers |
| g2_vtail | Group 3: laser-cut construction covers Group 1-2 designs only |
| g1_hand | the wing is too flexible even with the 1.5 mm balsa D-box: aileron reversal at 26.9 m/s (needs above 29.0 m/s = 1.15 x V_NE) |
| g1_vtail at the fixed wing station 0.45 | C-1 cannot be split to fit the 600 x 400 mm bed. The balanced station the pipeline picks still builds the kit. |

`test_matched_reference_designs_build_no_kit_and_say_why` pins these reasons. To keep the kit code covered:
- The kit, CAD-packing and fabrication tests build `kit_designs` (conftest). These are the reference designs, with
  g1, g2, g2_vtail and g1_hand swapped for their master records in `tests/data/kit_designs/` (provenance in its
  README). Those records still build kits.
- No G2-class quadplane sized by the matched propulsion builds a kit. In the probe:
  - around 20-25 kg, the anti-rotation pin is overloaded;
  - around 14-17 kg, no stock spar tube fits;
  - at 15.6 kg, aileron reversal fails;
  - at 19.6 kg (lighter payload), a former sliver is too thin.

  So the quadplane kit tests cannot move to a current design.
- `g1_kit` (G1 at 1.75 h, 4.12 kg, sized by the current code) builds the tractor kit. Its kit is stored as a golden
  snapshot, so current-code kit geometry stays under the gate.

## SITL

The g1, g2 and g1_vtail fixtures were regenerated on matched sets. On them, every SITL test in
`tests/test_sitl_validation.py` passes in Docker except the reversed-elevator test, which needed the change described
below. That includes G1, G2 and G1 V-tail acceptance, the perturbation check, and the reversed-ruddervator test.

No SITL harness change was needed. The Task 7 G1 failure came from the legacy 4S motor on a 6S pack.

The reversed-elevator test used to require the reversed flight to crash or time out. The matched G1, flown reversed,
completes the sequence while tumbling (pitch -87 to +79 deg, pitch step -14.5 deg against the +20 deg target), so it
fails on the score, not with a flight error. The test now accepts either outcome. It still requires that the flight
does not pass, and on the scored path that the validator's own pitch criterion (overshoot / settling limits) fails.
The pass criteria are unchanged.

In the Docker `--spec` runs, SITL passed for g1, g1_vtail, g2_vtail and g1_hand. It was not reached for g2 and g3,
which stop at the critic.

## Tuned constants

None in this task. The only constant tuned for closure in phase 1 is Task 6's `MAX_PROPULSION_PASSES`, 6 -> 12
(ruled; G3 needs 6 passes).

## Gate status

Reference designs through the full Docker `--spec` pipeline:

| | Converged | Not converged |
|---|---|---|
| Before | g1, g2, g1_vtail, g1_hand | g3 (L/D gate), g2_vtail (no kit, shell mass over Group 2) |
| After | g1, g1_vtail, g2_vtail, g1_hand | g3 (L/D gate, as before), g2 |

- **g2** stops converging. Without a kit, the shell mass model puts its as-built mass at 25.28 kg, 0.33 kg over the
  Group 2 limit of 24.95 kg, so the collisions_and_group gate fails with no rule fix. This is the same mechanism
  that kept g2_vtail from converging on master.
- **g3** fails the VSPAERO L/D gate both before and after: the L/D is high and the critic cannot fix it. On master,
  G3 passed the L/D gate at default settings and went out of band (+18 %) only after the critic raised the reserve to 0.15. The
  matched G3 fails it from the first iteration at default settings (VSPAERO L/D +24 % at defaults, +21-22 % after
  the reserve raise). The likely cause is the parked-rotor drag scaled into the sizing CD0 (kept, user decision 2),
  which VSPAERO does not see.

Both follow from the accepted kit regression or pre-date this branch. They are left for the kit-fix branch.

## Addendum (final review fixes, 2026-11-25): thin-air table axis

The propeller tables' density axis was extended down to 10 km on an ISA+40 day (the schema's corner) so high
service ceilings are no longer refused (11 density nodes instead of 8). The new node positions change the
interpolated sea-level and low-altitude coefficients in the fourth to fifth digit, so the goldens and the avionics
baseline were regenerated with their scripts. Every reference design keeps its matched set; MTOW moves by at most
0.03 %: g1 4.4462 -> 4.4460, g2 24.826 -> 24.824, g3 273.649 -> 273.724 (still 6 matching passes), g1_vtail 5.074,
g2_vtail 26.498 and g1_hand 7.104 unchanged to the gram. Five derived motor ratings moved by rounding (1 KV or
0.1 A). The genset margin fix in the matcher does not move any reference design.
