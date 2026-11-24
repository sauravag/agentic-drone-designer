"""Outer sizing loop around propulsion matching (spec 6.5).

Pass 0 sizes with the seed efficiencies. After each converged sizing the propulsion set is matched at that design's
flight points, and the design is re-sized with that set frozen: its efficiencies per phase, its part masses and its
propeller and rotor diameters. The loop stops when the matched set repeats. A cycle (A -> B -> A) takes the heavier
set, which is feasible at both masses, re-sizes once with it and notes it; no settled set within
MAX_PROPULSION_PASSES matches is a readable sizing failure.

Ruling R2: this module never imports mission_solver. The solver hands in `size_pass` and `context_for` callables, so
the loop stays generic in the solver's evaluation type.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final, Generic, Optional, TypeVar

from src.schemas.assumptions import DesignAssumptions
from src.schemas.propulsion import PropulsionMatch, SetKey
from src.schemas.state import PhaseResult
from src.tools.database.seed_data import catalog_rows
from src.tools.propulsion.match_types import FlightPoint, MatchContext, PropulsionMatchError
from src.tools.propulsion.matching import match_propulsion
from src.tools.sizing.aero_model import air_properties, speed_of_sound_ms
from src.tools.sizing.layouts import LayoutSpec
from src.tools.sizing.mass_model import MatchedHardware, lift_unit_mass_kg
from src.tools.sizing.mission_profile import (FORWARD_PHASES, VTOL_THRUST_TO_WEIGHT, PhaseSpec,
                                              PropulsionEfficiencies)
from src.units import G0

logger: logging.Logger = logging.getLogger(__name__)

MAX_PROPULSION_PASSES: Final[int] = 6
# Largest cruise propeller the layout's clearance allows, as a multiple of the sized fuselage diameter (controller's
# call D3; tunable per spec 6.8). The matched diameter then drives the gear, pusher gap and clearance checks.
PROP_MAX_TO_FUSELAGE_DIAMETER_RATIO: Final[float] = 3.0
HOVER_POINT: Final[str] = "Hover"
HOVER_PEAK_POINT: Final[str] = "Hover_Peak"
HYBRID_BUFFER_CHEMISTRY: Final[str] = "lipo"  # the hybrid buffer needs LiPo discharge rates

E = TypeVar("E")


@dataclass(frozen=True)
class MatchedSizing(Generic[E]):
    evaluation: E  # the design sized with `match` frozen
    match: PropulsionMatch
    passes: int  # sizing passes with a matched set (pass 0 not counted)


# ------------------------------------------------------------------ flight points


def _forward_point(spec: PhaseSpec, phase: PhaseResult, efficiency: float, isa_offset_c: float) -> FlightPoint:
    """Thrust = the phase's mean propulsive power x the efficiency it was sized with / its airspeed."""
    _, mu = air_properties(spec.altitude_m, isa_offset_c)
    return FlightPoint(
        name=spec.name, is_hover=False, airspeed_ms=spec.speed_ms, rho=spec.rho, mu=mu,
        speed_of_sound_ms=speed_of_sound_ms(spec.altitude_m, isa_offset_c),
        thrust_n=phase.propulsive_power_w * efficiency / spec.speed_ms, duration_s=spec.duration_s,
        check_sagged=spec.climb_rate_ms > 0.0,
    )


def _hover_points(hover: Sequence[PhaseSpec], weight_n: float, rotor_count: int,
                  isa_offset_c: float) -> tuple[FlightPoint, ...]:
    """Per rotor at T/W 1 (the hover phases' energy) and at VTOL_THRUST_TO_WEIGHT (installed peak, no duration),
    both at the take-off site and both re-checked on the sagged pack."""
    site: PhaseSpec = hover[0]
    _, mu = air_properties(site.altitude_m, isa_offset_c)
    sound: float = speed_of_sound_ms(site.altitude_m, isa_offset_c)
    per_rotor: float = weight_n / rotor_count
    duration: float = sum(s.duration_s for s in hover)
    return tuple(
        FlightPoint(name=name, is_hover=True, airspeed_ms=0.0, rho=site.rho, mu=mu, speed_of_sound_ms=sound,
                    thrust_n=ratio * per_rotor, duration_s=seconds, check_sagged=True)
        for name, ratio, seconds in ((HOVER_POINT, 1.0, duration), (HOVER_PEAK_POINT, VTOL_THRUST_TO_WEIGHT, 0.0))
    )


def flight_points(plan: Sequence[PhaseSpec], phases: Sequence[PhaseResult], efficiencies: PropulsionEfficiencies,
                  mtow_kg: float, rotor_count: int, isa_offset_c: float) -> tuple[FlightPoint, ...]:
    """The points the propulsion must meet for one sized design (spec 6.4): each forward phase at its mean power,
    and for lift rotors the hover points at the take-off weight."""
    results: dict[str, PhaseResult] = {p.name: p for p in phases}
    forward: tuple[FlightPoint, ...] = tuple(
        _forward_point(spec, results[spec.name], efficiencies.forward_for(spec.name), isa_offset_c)
        for spec in plan if not spec.is_hover)
    hover: list[PhaseSpec] = [spec for spec in plan if spec.is_hover]
    if not hover or rotor_count == 0:
        return forward
    return forward + _hover_points(hover, mtow_kg * G0, rotor_count, isa_offset_c)


def match_context(points: tuple[FlightPoint, ...], *, is_hybrid: bool, lift_rotor_count: int,
                  rotor_diameter_m: float, fuselage_diameter_m: float, is_hand_launch: bool, spec: LayoutSpec,
                  hotel_w: float, assumptions: DesignAssumptions) -> MatchContext:
    """The matching inputs (controller's rulings D3): cruise D up to PROP_MAX_TO_FUSELAGE_DIAMETER_RATIO x the
    fuselage; folding propellers for a hand launch; lift rotors at least the disk-loading diameter; the pusher
    installation factor for any pusher layout; LiPo for a hybrid buffer, else the search picks the chemistry."""
    return MatchContext(
        points=points, chemistry=HYBRID_BUFFER_CHEMISTRY if is_hybrid else None, is_hybrid=is_hybrid,
        lift_rotor_count=lift_rotor_count, min_lift_rotor_diameter_m=rotor_diameter_m,
        max_cruise_prop_diameter_m=PROP_MAX_TO_FUSELAGE_DIAMETER_RATIO * fuselage_diameter_m,
        require_folding_prop=is_hand_launch,
        pusher_factor=1.0 if spec.propeller == "tractor_nose" else assumptions.pusher_efficiency_factor,
        hotel_w=hotel_w, assumptions=assumptions,
    )


# ------------------------------------------------------------------ what a matched set feeds back


def efficiencies_from(match: PropulsionMatch, seed: PropulsionEfficiencies) -> PropulsionEfficiencies:
    """The matched set's efficiency per phase. Without lift rotors the hover values are never used: the seed's are
    kept so the record stays valid."""
    eff = match.efficiencies
    return PropulsionEfficiencies({name: eff[name] for name in FORWARD_PHASES},
                                  eff.get(HOVER_POINT, seed.hover), eff.get(HOVER_PEAK_POINT, seed.hover_peak))


def _part(table: str, name: str) -> Any:
    found = next((row for row in catalog_rows(table) if row.name == name), None)
    if found is None:
        raise PropulsionMatchError(f"matched part {name!r} is not in the {table} catalogue")
    return found


def _stored_or_catalogue(stored: Optional[float], table: str, name: str, field: str) -> float:
    return stored if stored is not None else getattr(_part(table, name), field)


def _cell_chemistry(match: PropulsionMatch) -> str:
    return match.cell_chemistry or _part("cells", match.cell_name).chemistry


def matched_hardware(match: PropulsionMatch, lift_rotor_count: int) -> MatchedHardware:
    """Part masses, diameters and cell chemistry of the matched set: as stored on the match, or (records written
    before they were stored) from the catalogue by name."""
    m = match
    cruise_kg: float = (_stored_or_catalogue(m.cruise_motor_kg, "cruise_motors", m.cruise_motor, "mass_kg")
                        + _stored_or_catalogue(m.cruise_propeller_kg, "propellers", m.cruise_propeller, "mass_kg")
                        + _stored_or_catalogue(m.cruise_esc_kg, "escs", m.cruise_esc, "mass_kg"))
    cruise_d: float = _stored_or_catalogue(m.cruise_propeller_diameter_m, "propellers", m.cruise_propeller,
                                           "diameter_m")
    chemistry: str = _cell_chemistry(m)
    if m.lift_motor is None or m.lift_rotor is None or m.lift_esc is None or lift_rotor_count == 0:
        return MatchedHardware(cruise_kg, 0.0, cruise_d, cell_chemistry=chemistry)
    unit: float = lift_unit_mass_kg(_stored_or_catalogue(m.lift_motor_kg, "vtol_motors", m.lift_motor, "mass_kg"),
                                    _stored_or_catalogue(m.lift_rotor_kg, "propellers", m.lift_rotor, "mass_kg"),
                                    _stored_or_catalogue(m.lift_esc_kg, "escs", m.lift_esc, "mass_kg"))
    rotor_d: float = _stored_or_catalogue(m.lift_rotor_diameter_m, "propellers", m.lift_rotor, "diameter_m")
    return MatchedHardware(cruise_kg, lift_rotor_count * unit, cruise_d, rotor_d, chemistry)


# ------------------------------------------------------------------ the loop


def _describe(match: PropulsionMatch) -> str:
    lift: str = f", {match.lift_motor} on {match.lift_rotor}" if match.lift_motor else ""
    return f"{match.series}S {match.cruise_motor} on {match.cruise_propeller}{lift} ({match.propulsion_mass_kg:.3f} kg)"


def _cell_note(recorded: PropulsionMatch, rematched: PropulsionMatch) -> PropulsionMatch:
    """The recorded set, noting when the final re-match (same set) picked another cell for the pack."""
    if rematched.cell_name == recorded.cell_name:
        return recorded
    note: str = (f"the final re-match picked cell {rematched.cell_name} for the same set; the design was sized "
                 f"with {recorded.cell_name}")
    return recorded.model_copy(update={"notes": (*recorded.notes, note)})


def _cycle_choice(found: PropulsionMatch, history: tuple[PropulsionMatch, ...]) -> PropulsionMatch:
    """The heaviest set of the cycle (from the earlier visit of `found`'s set to now), with a note."""
    start: int = next(i for i, m in enumerate(history) if m.set_key == found.set_key)
    latest: dict[SetKey, PropulsionMatch] = {m.set_key: m for m in (*history[start:], found)}
    heaviest: PropulsionMatch = max(latest.values(), key=lambda m: (m.propulsion_mass_kg, m.set_key))
    note: str = (f"propulsion sets cycled between {' / '.join(_describe(m) for m in latest.values())}; using the "
                 f"heavier set {_describe(heaviest)}, which is feasible at both masses")
    logger.warning(note)
    return heaviest.model_copy(update={"notes": (*heaviest.notes, note)})


def size_with_matching(size_pass: Callable[[PropulsionEfficiencies, Optional[PropulsionMatch]], E],
                       context_for: Callable[[E], MatchContext], seed: PropulsionEfficiencies,
                       match: Callable[[MatchContext], PropulsionMatch] = match_propulsion) -> MatchedSizing[E]:
    """Size, match, re-size with the set frozen, until the set repeats (spec 6.5). PropulsionMatchError when the
    matching finds no set, or when no set settles within MAX_PROPULSION_PASSES matches."""
    evaluation: E = size_pass(seed, None)
    history: tuple[PropulsionMatch, ...] = ()
    for pass_no in range(1, MAX_PROPULSION_PASSES + 1):
        found: PropulsionMatch = match(context_for(evaluation))
        if history and found.set_key == history[-1].set_key:
            return MatchedSizing(evaluation, _cell_note(history[-1], found), pass_no - 1)
        if any(m.set_key == found.set_key for m in history):
            chosen: PropulsionMatch = _cycle_choice(found, history)
            return MatchedSizing(size_pass(efficiencies_from(chosen, seed), chosen), chosen, pass_no)
        if pass_no == MAX_PROPULSION_PASSES:
            break
        logger.info("Propulsion pass %d: %s", pass_no, _describe(found))
        evaluation = size_pass(efficiencies_from(found, seed), found)
        history = (*history, found)
    tried: str = "; ".join(_describe(m) for m in (*history, found))
    raise PropulsionMatchError(f"the propulsion set did not settle in {MAX_PROPULSION_PASSES} passes "
                               f"(sets tried: {tried})")
