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
    pack_layout,
)
from src.tools.aero_cad.components import required_fuselage_length_m
from src.tools.aero_cad.layout import DEFAULT_WING_STATION, WING_STATION_SCAN, AircraftLayout, layout_from_sizing
from src.tools.fabrication.kit_pod import kit_fuselage_diameter_m

logger = logging.getLogger(__name__)

MAX_BALANCE_ITERATIONS: Final[int] = 12
SCAN_STATIONS: Final[tuple[float, ...]] = WING_STATION_SCAN
STATION_TOLERANCE: Final[float] = 1e-4


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

    @property
    def budget_left(self) -> bool:
        return len(self.evaluated) < MAX_BALANCE_ITERATIONS

    @property
    def done(self) -> bool:
        return self.best is not None and self.best.packing.is_balanced

    def evaluate(self, station: float) -> BalancedDesign | None:
        station = round(station, 6)
        if station in self.evaluated:
            return self.evaluated[station]
        n: int = len(self.evaluated) + 1
        layout = layout_from_sizing(self.sizing, station, self.min_diameter, self.min_length)
        analysis = self.analyse(layout)
        try:
            packing = pack_layout(self.sizing, self.bom, analysis.aero, layout, analysis.step_path, self.settings)
        except PackingSlotError as exc:
            logger.info("[Tool] build_aero_cad evaluation %d: wing station %.3f -> %s", n, station, exc)
            self.evaluated[station] = None
            return None
        logger.info("[Tool] build_aero_cad evaluation %d: wing station %.3f -> SM %.3f, %d interferences",
                    n, station, packing.static_margin, packing.interference_count)
        design = BalancedDesign(layout, analysis, packing, n, n)
        self.evaluated[station] = design
        if self.best is None or design.score() < self.best.score():
            self.best = design
        return design


def _bracket(search: _Search) -> tuple[BalancedDesign, BalancedDesign] | None:
    """Adjacent evaluated stations with SM on opposite sides of the target and the same packing
    configuration (so a root exists between them), narrowest first."""
    feasible = sorted((d for d in search.evaluated.values() if d is not None), key=lambda d: d.layout.wing_station)
    pairs = [
        (a, b) for a, b in zip(feasible, feasible[1:])
        if a.signature() == b.signature()
        and (a.packing.static_margin - a.packing.band.target) * (b.packing.static_margin - b.packing.band.target) < 0
    ]
    return min(pairs, key=lambda p: p[1].layout.wing_station - p[0].layout.wing_station) if pairs else None


def _refine(search: _Search) -> None:
    """Secant (regula falsi) inside same-configuration brackets until balanced or out of budget."""
    while search.budget_left and not search.done:
        pair = _bracket(search)
        if pair is None:
            return
        (a, b) = pair
        sa, sb = a.packing.static_margin - a.packing.band.target, b.packing.static_margin - b.packing.band.target
        xa, xb = a.layout.wing_station, b.layout.wing_station
        x_new: float = xa - sa * (xb - xa) / (sb - sa)
        if min(abs(x_new - xa), abs(x_new - xb)) < STATION_TOLERANCE:
            return
        search.evaluate(x_new)


def balance_design(sizing: SizingOutput, bom: SelectedBOM, analyse: Analyser,
                   wing_station: float = DEFAULT_WING_STATION,
                   settings: FabricationSettings = DEFAULT_FABRICATION) -> BalancedDesign:
    """Find a wing station whose packing is balanced and interference-free.

    SM(station) jumps whenever a slider changes slot (e.g. battery ahead of / behind the tank), so a
    plain secant can bounce across the jump forever. Instead: try the default station, scan the
    station range nearest-first, and refine only inside brackets whose two ends share a packing
    configuration. Returns the best attempt (fewest interferences, then smallest SM error) even if
    unbalanced, so the critic can act on it. The pod is floored so the internal COTS parts fit."""
    search = _Search(sizing, bom, analyse, settings)
    search.evaluate(wing_station)
    for station in sorted(SCAN_STATIONS, key=lambda s: abs(s - wing_station)):
        if search.done or not search.budget_left:
            break
        search.evaluate(station)
        _refine(search)
    _refine(search)
    if search.best is None:
        raise CADPackingError("no wing station produced a packable layout (no free battery/tray slot)")
    total: int = len(search.evaluated)
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
