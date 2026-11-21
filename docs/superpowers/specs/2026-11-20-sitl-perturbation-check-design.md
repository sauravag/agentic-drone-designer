# SITL perturbation check (robustness of the generated gains)

Status: draft for approval, 2026-11-20.

## Why

The ArduPilot gains are derived by inverting the same linear `FlightModel` that the JSBSim plant is built from, so
today's SITL pass shows the parameter file is self-consistent and ArduPilot's loops are wired correctly. It says
nothing about whether the gains survive errors in the VSPAERO and strip-theory derivatives, which the user puts at
20-30 %. The SITL roll and pitch steps should be flown against plants that are deliberately wrong, and the gate
applied to the worst case.

## Goals

- Every SITL attempt flies the nominal plant plus two perturbed plants with the **same** parameter file.
- Roll and pitch overshoot (< 20 %) and settling (< 3 s) are gated on the worst of the three flights.
- The gain retry reflies all three plants at the new gain scale (two-way, see Design).
- The report, `sitl_report.json` and the param-file header show each case and which case governed.

## Non-goals

- Full 16-corner coverage (option A) or a final full sweep (option C): rejected by the user for runtime.
- Perturbing the transition: the transition altitude-loss gate stays on the nominal flight.
- Changing how the gains are derived.

## Approaches (user picked B)

- **A. All 16 corners + nominal per gain attempt:** complete within the ranges, but adds about 10-17 min per design
  run (up to about 1 h with retries).
- **B. Two physical extremes + nominal (chosen):** 3 flights per attempt, about 2-3 min more per design run.
  The risk: it assumes the worst overshoot is the "hot" corner and the worst settling is the "sluggish" corner.
  ArduPilot's integrators and rate limits could make a mixed corner worse, and that case goes untested.
- **C. B in the loop, A once for the final design:** complete for the result, but slower and more complex.

## Design

**Perturbed plants.** A pure function `perturbed_model(model, case) -> FlightModel` returns a new `FlightModel`
(`dataclasses.replace`, nothing mutated) for two named cases. The factors are named constants:

| Factor | Hot (worst overshoot) | Sluggish (worst settling) |
|---|---|---|
| Control power `cl_delta_a`, `cm_delta_e` | x1.3 | x0.7 |
| Damping `cl_p`, `cm_q` | x0.7 | x1.3 |
| Inertia `ixx`, `iyy`, `izz`, `ixz` | x0.8 | x1.2 |
| Static margin | SM_MIN (0.08, aft CG) | SM_MAX (0.14, forward CG) |

**CG shift.** Moving the CG from the as-built margin SM0 to SM' changes the moment about the CG by
CL x (SM0 - SM'). So:
- `cm_alpha' = -cl_alpha * SM'`
- `cm0' = cm0 + cl0 * (SM0 - SM')`

The tail incidence stays at its nominal setting, so the perturbed aircraft needs elevator to trim, as a real
mis-balanced aircraft would. SM0 must be carried on `FlightModel` (new field `static_margin`) instead of being
inferred. `SM_MIN`/`SM_MAX` move from `cad_packer` (which imports Build123d) to `stability.py`; `cad_packer` re-exports them so existing imports keep working.

**Flow.** In `generate_and_verify_fcs`, each attempt generates `params` from the nominal model, then flies
nominal, hot and sluggish in turn. All three go through the existing `_attempt`, one SITL at a time under the
existing lock, the nominal flight still in `attempt_N/` (the viewer replay and existing tests read it there) and the perturbed flights in `attempt_N/hot/` and `attempt_N/sluggish/`. An attempt passes only when all three flights complete
and:
- the worst roll and pitch steps pass, and
- the nominal transition loss is below 15 m.

A perturbed flight that crashes or times out fails the attempt, the same as a nominal one does today.

**Two-way retry (user decision 2026-11-21, after the first Docker run).**
The first Docker run showed the 0.75x-down-only retry (SRS 5.6) can't fix a too-slow plant: g1's sluggish roll
settling went 3.12 s at gain 1.0, then 3.40 / 3.84 / 4.40 s as the gains dropped. The retry is now direction-aware,
based on the governing (worst) case of the failed attempt:
- If any completed plant's overshoot fails, or a plant crashed, the next attempt multiplies the gain scale by 0.75
  (as before).
- If only settling fails, the next attempt divides it by 0.75.
- If the direction would reverse relative to the previous step, no gain scale on the 0.75 step grid satisfies
  both corners. The loop stops and reports the failure.

`MAX_ATTEMPTS` (4) is unchanged.

**Gain ceiling (R41, 2026-11-21).** Raising the gains is bounded by `MAX_GAIN_SCALE = 1/0.75**2` (about 1.78), a
named constant in `sitl_validator.py`. An up-step that would exceed it stops the loop the same way a reversal does,
so the reachable up path is 1.0, 1.333, 1.778. The quadplane rate P is clamped to `Q_RATE_P_BOUNDS` after the scale
is applied, so the scale can never push it past the design's own clamp. The fixed-wing rate gains get no new bound
beyond `MAX_GAIN_SCALE`. One `gain_scale` still drives both the fixed-wing and the VTOL gains: a fixed-wing settling
retry also raises the hover gains (up to about 1.78x, within the Q clamp). Splitting them would be a new design.

**Failed-run selection (R42, 2026-11-21).** When an attempt passes, it is the one shipped. When none passes, the
shipped attempt is the flown attempt (nominal flight completed) with the lowest severity: the governing case's
largest limit ratio, `max(overshoot / 0.20, settling / 3 s)` over roll and pitch, where a crashed plant counts as
infinite. Ties go to the lower gain scale. `replay_attempt`, `ardupilot.param` and `FCSArtifacts` all follow that
attempt, so a failed two-way run never hands over a more aggressive set than a gentler attempt that failed by less.

**Outputs.**
- `FCSArtifacts` overshoot and settling fields become the worst case. New fields:
  - `governing_case`, defaulting to `"nominal"` so old records validate;
  - `perturbation_cases`, the per-case metrics.
- `sitl_report.json` lists every case of every attempt.
- The replay stays the nominal flight.
- The design report and UI show a small per-case table.
- The param-file header names the governing case.
- The README and SRS wording moves from "consistent in simulation" to "checked against two extreme corners of the
  roll/pitch aerodynamics: plus or minus 30 % aileron/elevator power and roll/pitch damping, plus or minus 20 %
  inertia and both static-margin limits". Yaw/rudder terms and the VTOL rotor terms are not perturbed (the hover
  sees the inertia change but no CG imbalance), the transition gate uses the nominal flight, and mixed corners are
  not flown.

## Testing

- **Unit:** `perturbed_model` scales exactly the listed fields, leaves the input untouched, and gives
  `cm_alpha' = -cl_alpha * SM'` and the `cm0` shift above.
- **Unit:** worst-case scoring picks the larger overshoot and settling across cases.
- **Unit:** a crashed perturbed flight fails the attempt.
- **Unit:** the retry reflies all three plants at the new gain scale, raises the gains after a settling-only
  failure, stops at a reversal or at `MAX_GAIN_SCALE`, and ships the least severe attempt when none passes.
- **Unit:** the quadplane rate P stays within `Q_RATE_P_BOUNDS` at the maximum gain scale.
- **Docker:** the existing SITL acceptance tests run g1 and a quadplane through all three plants, and
  `sitl_report.json` holds three cases per attempt.
- **Docker:** a plant pushed far beyond the ranges (control power x0.3, too slow to settle in 3 s) fails the gate, which proves the check can
  fail.

## Left to the builder

The JSBSim-side mapping of the scaled fields comes for free, because the exporter reads them from `FlightModel`.
The builder also decides the exact table layout in the report and UI.
