"""Module C orchestration: OML + VSPAERO -> structure + packing, iterating the wing station until the
static margin lands inside the layout's static-margin band (the `build_aero_cad` tool node of SRS 5).

Static margin has two levers. The packer slides the battery and avionics tray (moves the CG). When
the fuselage runs out of room for that, this loop slides the wing+tail group (moves the neutral
point, and with it the fuel tank) with a discontinuity-aware scan + bracketed secant on wing station. The aero analyser is injected so the loop is
testable without OpenVSP.
"""

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Final

from src.schemas.components import SelectedBOM
from src.schemas.fabrication import DEFAULT_FABRICATION, FabricationSettings
from src.schemas.state import AeroStabilityOutput, CADArtifacts, SizingOutput
from src.tools.aero_cad.cad_packer import (
    CADPackingError,
    PackingResult,
    PackingSlotError,
    export_artifacts,
    log_strike,
    pack_layout,
)
from src.tools.aero_cad.components import required_fuselage_length_m
from src.tools.aero_cad.layout import DEFAULT_WING_STATION, WING_STATION_SCAN, AircraftLayout, layout_from_sizing
from src.tools.fabrication.kit_pod import kit_fuselage_diameter_m
from src.tools.sizing.layouts import PusherBoomClearanceError

logger = logging.getLogger(__name__)

MAX_BALANCE_ITERATIONS: Final[int] = 12
SCAN_STATIONS: Final[tuple[float, ...]] = WING_STATION_SCAN
STATION_TOLERANCE: Final[float] = 1e-4
STATION_DECIMALS: Final[int] = 6  # evaluated stations are keyed rounded to this
# Bisecting across a packing jump looks for a balanced station beside it, not for the jump itself: at most this
# many halvings, and never of a gap narrower than JUMP_MIN_WIDTH (fraction of fuselage length).
MAX_JUMP_BISECTIONS: Final[int] = 3
JUMP_MIN_WIDTH: Final[float] = 0.01


@dataclass(frozen=True)
class OmlAnalysis:
    aero: AeroStabilityOutput
    vsp3_path: Path
    step_path: Path


@dataclass(frozen=True)
class BalancedDesign:
    layout: AircraftLayout
    analysis: OmlAnalysis
    packing: PackingResult
    iterations: int  # evaluation that produced this (best) design
    total_iterations: int

    def score(self) -> tuple[int, float]:
        """Lexicographic: fewest interferences first, then smallest static-margin error."""
        return self.packing.interference_count, abs(self.packing.static_margin - self.packing.band.target)

    def signature(self) -> tuple[str, ...]:
        """Packing configuration: internal parts in fore-to-aft order. SM(station) is continuous
        only while this stays the same."""
        return tuple(c.name for c in sorted(self.packing.components, key=lambda c: c.x_min) if c.is_internal)


Analyser = Callable[[AircraftLayout], OmlAnalysis]


class _Search:
    """Evaluates wing stations (each = one OpenVSP + packing run) and keeps the best design."""

    def __init__(self, sizing: SizingOutput, bom: SelectedBOM, analyse: Analyser,
                 settings: FabricationSettings = DEFAULT_FABRICATION) -> None:
        self.sizing, self.bom, self.analyse, self.settings = sizing, bom, analyse, settings
        self.min_diameter: float = kit_fuselage_diameter_m(sizing, bom, settings)  # R23: one pod for aero and kit
        self.min_length: float = required_fuselage_length_m(sizing, bom, max(self.min_diameter, sizing.fuselage_diameter_m))
        self.evaluated: dict[float, BalancedDesign | None] = {}
        self.best: BalancedDesign | None = None
        self.reasons: dict[str, str] = {}  # last infeasibility message per exception type

    @property
    def budget_left(self) -> bool:
        return len(self.evaluated) < MAX_BALANCE_ITERATIONS

    @property
    def done(self) -> bool:
        return self.best is not None and self.best.packing.is_balanced

    def _infeasible(self, n: int, station: float, exc: Exception) -> None:
        """Book `station` as infeasible (no packing slot, or no pusher boom clearance) and keep its reason."""
        logger.info("[Tool] build_aero_cad evaluation %d: wing station %.3f -> %s", n, station, exc)
        self.evaluated[station] = None
        self.reasons[type(exc).__name__] = str(exc)
        return None

    def evaluate(self, station: float) -> BalancedDesign | None:
        station = round(station, STATION_DECIMALS)
        if station in self.evaluated:
            return self.evaluated[station]
        n: int = len(self.evaluated) + 1
        try:
            layout = layout_from_sizing(self.sizing, station, self.min_diameter, self.min_length,
                                        propeller_diameter_m=self.bom.cruise_propeller.diameter_m)
        except PusherBoomClearanceError as exc:  # floored pod, forward station: the boom cannot reach past its root
            return self._infeasible(n, station, exc)
        analysis = self.analyse(layout)
        try:
            packing = pack_layout(self.sizing, self.bom, analysis.aero, layout, analysis.step_path, self.settings)
        except PackingSlotError as exc:
            return self._infeasible(n, station, exc)
        logger.info("[Tool] build_aero_cad evaluation %d: wing station %.3f -> SM %.3f, %d interferences",
                    n, station, packing.static_margin, packing.interference_count)
        design = BalancedDesign(layout, analysis, packing, n, n)
        self.evaluated[station] = design
        if self.best is None or design.score() < self.best.score():
            self.best = design
        return design


def _straddles(a: BalancedDesign, b: BalancedDesign) -> bool:
    """SM on opposite sides of the band target at the two designs."""
    return (a.packing.static_margin - a.packing.band.target) * (b.packing.static_margin - b.packing.band.target) < 0


def _adjacent_straddles(search: _Search, same_configuration: bool) -> list[tuple[BalancedDesign, BalancedDesign]]:
    """Adjacent evaluated stations whose SM straddles the target, with the same (or a different) packing
    configuration at the two ends."""
    feasible = sorted((d for d in search.evaluated.values() if d is not None), key=lambda d: d.layout.wing_station)
    return [(a, b) for a, b in zip(feasible, feasible[1:])
            if _straddles(a, b) and (a.signature() == b.signature()) == same_configuration]


def _bracket(search: _Search) -> tuple[BalancedDesign, BalancedDesign] | None:
    """Adjacent evaluated stations with SM on opposite sides of the target and the same packing
    configuration (so a root exists between them), narrowest first."""
    pairs = _adjacent_straddles(search, same_configuration=True)
    return min(pairs, key=lambda p: p[1].layout.wing_station - p[0].layout.wing_station) if pairs else None


def _jump(search: _Search) -> tuple[BalancedDesign, BalancedDesign] | None:
    """Adjacent stations that straddle the target across a configuration change, at least JUMP_MIN_WIDTH apart:
    the band may still be reached on either side of the jump, so the gap is bisected (never extrapolated)."""
    pairs = [(a, b) for a, b in _adjacent_straddles(search, same_configuration=False)
             if b.layout.wing_station - a.layout.wing_station >= JUMP_MIN_WIDTH]
    return min(pairs, key=lambda p: p[1].layout.wing_station - p[0].layout.wing_station) if pairs else None


def _seen(search: _Search, station: float) -> bool:
    """`station` was already evaluated (as `_Search.evaluate` keys it): evaluating it again adds nothing."""
    return round(station, STATION_DECIMALS) in search.evaluated


def _refine(search: _Search, allow_jump: bool = False) -> None:
    """Secant (regula falsi) inside same-configuration brackets, until balanced or out of budget. With
    `allow_jump` (the final, post-scan refine only, so the scan's budget is never spent on it), a straddle that
    crosses a configuration change is bisected up to MAX_JUMP_BISECTIONS times to look for a balanced station
    beside the jump."""
    jumps_left: int = MAX_JUMP_BISECTIONS if allow_jump else 0
    while search.budget_left and not search.done:
        pair = _bracket(search)
        if pair is None:
            jump = _jump(search) if jumps_left > 0 else None
            if jump is None:
                return
            mid: float = 0.5 * (jump[0].layout.wing_station + jump[1].layout.wing_station)
            if _seen(search, mid):  # the midpoint had no packable slot: no progress possible here
                return
            jumps_left -= 1
            search.evaluate(mid)
            continue
        (a, b) = pair
        sa, sb = a.packing.static_margin - a.packing.band.target, b.packing.static_margin - b.packing.band.target
        xa, xb = a.layout.wing_station, b.layout.wing_station
        x_new: float = xa - sa * (xb - xa) / (sb - sa)
        if min(abs(x_new - xa), abs(x_new - xb)) < STATION_TOLERANCE or _seen(search, x_new):
            return
        search.evaluate(x_new)


def _scan(search: _Search, wing_station: float) -> None:
    """Try `wing_station`, then the scan stations nearest-first (refining same-configuration brackets after
    each), then one final refine that may also bisect across a packing jump."""
    search.evaluate(wing_station)
    for station in sorted(SCAN_STATIONS, key=lambda s: abs(s - wing_station)):
        if search.done or not search.budget_left:
            break
        search.evaluate(station)
        _refine(search)
    _refine(search, allow_jump=True)


def _infeasibility(search: _Search) -> str:
    """Why every station failed: no free battery/tray slot, and/or the pusher boom ending ahead of its root."""
    causes: list[str] = []
    if PackingSlotError.__name__ in search.reasons:
        causes.append("no free battery/tray slot")
    if PusherBoomClearanceError.__name__ in search.reasons:
        causes.append(f"pusher boom clearance ({search.reasons[PusherBoomClearanceError.__name__]})")
    return "; ".join(causes) or "no station evaluated"


def balance_design(sizing: SizingOutput, bom: SelectedBOM, analyse: Analyser,
                   wing_station: float = DEFAULT_WING_STATION,
                   settings: FabricationSettings = DEFAULT_FABRICATION) -> BalancedDesign:
    """Find a wing station whose packing is balanced and interference-free.

    SM(station) jumps whenever a slider changes slot (e.g. battery ahead of / behind the tank), so a
    plain secant can bounce across the jump forever. Instead: try the default station, scan the
    station range nearest-first, and refine inside brackets whose two ends share a packing
    configuration; only after the scan, bisect (a capped number of times) across a jump that straddles the
    target. Returns the best attempt (fewest interferences, then smallest SM error) even if
    unbalanced, so the critic can act on it. The pod is floored so the internal COTS parts fit."""
    search = _Search(sizing, bom, analyse, settings)
    _scan(search, wing_station)
    if search.best is None:
        raise CADPackingError(f"no wing station produced a packable layout: {_infeasibility(search)}")
    total: int = len(search.evaluated)
    log_strike(search.best.packing.clearance, logging.WARNING)
    if not search.best.packing.is_balanced:
        logger.warning("design not balanced after %d evaluation(s): SM %.3f, %d interferences at wing station %.3f",
                       total, search.best.packing.static_margin, search.best.packing.interference_count,
                       search.best.layout.wing_station)
    return replace(search.best, total_iterations=total)


def build_aero_cad(sizing: SizingOutput, bom: SelectedBOM, cruise_altitude_m: float, out_dir: Path,
                   settings: FabricationSettings = DEFAULT_FABRICATION,
                   temperature_offset_c: float = 0.0) -> tuple[AeroStabilityOutput, CADArtifacts]:
    """Full Module C run with OpenVSP/VSPAERO (Docker toolchain); temperature_offset_c is the ISA offset."""
    from src.tools.aero_cad import openvsp_builder  # needs the compiled OpenVSP API

    def analyse(layout: AircraftLayout) -> OmlAnalysis:
        aero, vsp3, step = openvsp_builder.analyse_layout(layout, sizing, cruise_altitude_m, out_dir,
                                                          temperature_offset_c)
        return OmlAnalysis(aero, vsp3, step)

    design: BalancedDesign = balance_design(sizing, bom, analyse, settings=settings)
    if design.iterations != design.total_iterations:  # best attempt was not the last: rewrite its OML files
        design = replace(design, analysis=analyse(design.layout))
    artifacts: CADArtifacts = export_artifacts(design.packing, design.analysis.vsp3_path,
                                               design.analysis.step_path, out_dir)
    return design.analysis.aero, artifacts
