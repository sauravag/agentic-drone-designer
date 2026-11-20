"""Kit-aware pod sizing and slider seating (R23; SI, aircraft frame: x aft, y right, z up).

A laser-cut kit cuts every component an opening of its box + half the component gap in each former and plate it
crosses, and keeps a MIN_WEB_M web (plus room for the stringer notches) inside the former rim. The battery and
avionics tray therefore sit as low as their former openings allow (`seat_low`). The wing-box plates and socket
tubes fill the upper pod at the wing; for Group 1-2 designs (the ones that get a kit) the pod is widened until the
seated battery passes beneath them with STRUCTURE_CLEARANCE_M to spare, so it can sit at the CG under the wing.
Every slider also leaves crutch side rails of at least crutch_rail_min_m beside its crutch opening (R28,
`crutch_rails_fit`), and the wing box's floor includes the centreline overwrap at its wrapped OD (R24) and the
socket bosses when the plate loads need them (R32).
Group 3+ designs keep the COTS-fit diameter. One diameter flows through the whole pipeline (OpenVSP included).
"""

import dataclasses
import logging
import math
from typing import Final, Optional

from src.schemas.components import SelectedBOM
from src.schemas.fabrication import DEFAULT_FABRICATION, FabricationSettings
from src.schemas.state import SizingOutput
from src.tools.aero_cad import components
from src.tools.aero_cad.airfoil_geometry import load_airfoil
from src.tools.aero_cad.components import COMPONENT_GAP_M, PlacedComponent
from src.tools.aero_cad.layout import (
    DEFAULT_WING_STATION,
    WING_STATION_SCAN,
    AircraftLayout,
    fuselage_radius_at,
    layout_from_sizing,
)
from src.tools.fabrication.build_kit import MAX_DOD_GROUP, kit_spar_plan
from src.tools.fabrication.fuselage_parts import wing_box_floor_m
from src.tools.fabrication.tubes import FIT_EPS_M, MIN_WEB_M, KitUnavailable, plate_loads

logger = logging.getLogger(__name__)

STRUCTURE_CLEARANCE_M: Final[float] = COMPONENT_GAP_M / 2.0  # = the kit's opening allowance round a component
POD_GROWTH_STEP: Final[float] = 0.02  # fractional diameter step while looking for a pod the battery fits under
MAX_POD_GROWTH: Final[float] = 1.5  # never widen a pod by more than half for the kit
POD_DIAMETER_TOLERANCE_M: Final[float] = 0.5e-3  # bisection stops once the bracket is this narrow
SIZING_STATIONS: Final[tuple[float, ...]] = (DEFAULT_WING_STATION, *WING_STATION_SCAN)  # every station balancing tries


def _half_opening(item: PlacedComponent) -> tuple[float, float]:
    """(y, z) half sizes of the item's former opening: its box + half the component gap each side."""
    return (item.size_m[1] + COMPONENT_GAP_M) / 2.0, (item.size_m[2] + COMPONENT_GAP_M) / 2.0


def _opening_radius_m(radius_m: float, settings: FabricationSettings) -> float:
    """Largest radius an opening corner may reach in a former of skin radius `radius_m`."""
    return radius_m - settings.fuselage_cover_thickness_m - MIN_WEB_M - settings.stringer_side_m


def seated_z_m(item: PlacedComponent, diameter_m: float, settings: FabricationSettings) -> float:
    """Centre z that puts the opening's lower corners on the opening circle of the constant section (never above
    the axis; 0 when the opening is too wide to drop at all)."""
    hy, hz = _half_opening(item)
    reach = _opening_radius_m(diameter_m / 2.0, settings)
    if hy >= reach:
        return 0.0
    return min(0.0, hz - math.sqrt(reach**2 - hy**2))


def seat_low(item: PlacedComponent, diameter_m: float, settings: FabricationSettings) -> PlacedComponent:
    x, y, _ = item.center_m
    return dataclasses.replace(item, center_m=(x, y, seated_z_m(item, diameter_m, settings)))


def fits_kit_formers(item: PlacedComponent, layout: AircraftLayout, settings: FabricationSettings) -> bool:
    """The item's former openings keep their web and stringer room at both ends of the opening (the pod profile is
    unimodal, so the ends bound every former in between)."""
    hy, hz = _half_opening(item)
    gap = COMPONENT_GAP_M / 2.0
    radius = min(fuselage_radius_at(layout, item.x_min - gap), fuselage_radius_at(layout, item.x_max + gap))
    corner = math.hypot(abs(item.center_m[1]) + hy, abs(item.center_m[2]) + hz)
    return corner <= _opening_radius_m(radius, settings) + FIT_EPS_M


def crutch_rails_fit(item: PlacedComponent, layout: AircraftLayout, settings: FabricationSettings) -> bool:
    """The crutch opening of an internal item (its footprint + half the component gap each side) leaves side rails
    of at least crutch_rail_min_m to the crutch edge (the skin less the cover) at both ends of the opening (R28)."""
    gap = COMPONENT_GAP_M / 2.0
    edge = min(fuselage_radius_at(layout, item.x_min - gap), fuselage_radius_at(layout, item.x_max + gap))
    edge -= settings.fuselage_cover_thickness_m
    half_width = (item.size_m[1] + COMPONENT_GAP_M) / 2.0
    return edge - abs(item.center_m[1]) - half_width >= settings.crutch_rail_min_m - FIT_EPS_M


def fits_kit(item: PlacedComponent, layout: AircraftLayout, settings: FabricationSettings) -> bool:
    """Both kit rules for a slider: its former openings keep their webs and its crutch rails keep their width."""
    return fits_kit_formers(item, layout, settings) and crutch_rails_fit(item, layout, settings)


def _wing_box_floor_m(sizing: SizingOutput, bom: SelectedBOM, diameter_m: float,
                     settings: FabricationSettings) -> Optional[float]:
    """Lowest wing-box floor over every wing station the balance search may pick (stations where no wing box can be
    built at all are skipped: no kit there anyway); None when there is none."""
    length = components.required_fuselage_length_m(sizing, bom, diameter_m)
    floors: list[float] = []
    for station in SIZING_STATIONS:
        layout = layout_from_sizing(sizing, station, diameter_m, length)
        try:
            spar = kit_spar_plan(sizing, layout, load_airfoil(layout.airfoil), settings)
            floors.append(wing_box_floor_m(layout, spar, settings, plate_loads(sizing, settings)))
        except KitUnavailable:
            continue
    return min(floors) if floors else None


def pod_fits_kit(sizing: SizingOutput, bom: SelectedBOM, diameter_m: float, settings: FabricationSettings) -> bool:
    """Every seated slider opens in the formers of the constant section with crutch rails to spare, and the seated
    battery passes under the wing box (its overwrap included) with STRUCTURE_CLEARANCE_M, at the worst wing station
    the balance search may choose."""
    length = components.required_fuselage_length_m(sizing, bom, diameter_m)
    layout = layout_from_sizing(sizing, DEFAULT_WING_STATION, diameter_m, length)
    seated = [seat_low(s, diameter_m, settings) for s in components.sliders(sizing, bom)]
    middle = 0.5 * layout.fuselage_length_m
    if not all(fits_kit(s.at_x(middle), layout, settings) for s in seated):
        return False
    floor = _wing_box_floor_m(sizing, bom, diameter_m, settings)
    battery = seated[0]
    return floor is not None and battery.center_m[2] + battery.size_m[2] / 2.0 + STRUCTURE_CLEARANCE_M <= floor


def _first_feasible(sizing: SizingOutput, bom: SelectedBOM, start_m: float,
                    settings: FabricationSettings) -> Optional[tuple[float, float]]:
    """(last infeasible, first feasible) diameter stepping up from `start_m`, or None up to MAX_POD_GROWTH."""
    previous, d = start_m, start_m * (1.0 + POD_GROWTH_STEP)
    while d <= start_m * MAX_POD_GROWTH:
        if pod_fits_kit(sizing, bom, d, settings):
            return previous, d
        previous, d = d, d * (1.0 + POD_GROWTH_STEP)
    return None


def kit_fuselage_diameter_m(sizing: SizingOutput, bom: SelectedBOM,
                            settings: FabricationSettings = DEFAULT_FABRICATION) -> float:
    """Pod diameter floor: the COTS-fit diameter, widened for Group 1-2 designs to the smallest pod whose seated
    battery passes under the wing box (module docstring). Unchanged when already so, for Group 3+, or when no pod
    up to MAX_POD_GROWTH works (the kit then falls back with a warning)."""
    base = components.required_fuselage_diameter_m(sizing, bom)
    start = max(base, sizing.fuselage_diameter_m)
    if sizing.dod_group > MAX_DOD_GROUP or pod_fits_kit(sizing, bom, start, settings):
        return base
    bracket = _first_feasible(sizing, bom, start, settings)
    if bracket is None:
        logger.warning("no pod up to %.0f %% wider seats the battery under the wing box", 100 * (MAX_POD_GROWTH - 1))
        return base
    lo, hi = bracket
    while hi - lo > POD_DIAMETER_TOLERANCE_M:
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if pod_fits_kit(sizing, bom, mid, settings) else (mid, hi)
    return hi
