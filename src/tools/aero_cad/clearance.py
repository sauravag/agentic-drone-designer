"""Cruise-propeller clearance checks of a packed layout (SI, structural axes: x aft of the nose, z up).

Three checks: the disc against the tail (axial), the disc tip against the ground on the landing gear (at the take-off
rotation angle for a runway launch, level for a vertical one), and, for a gearless hand-launched aircraft, the disc tip
against the belly line it lands on. A tail or ground strike is a collision-class failure (the packer counts it);
a tip below the belly line only calls for a folding propeller.
"""

import math
from dataclasses import dataclass
from typing import Final, Optional

from src.tools.aero_cad.layout import FUSELAGE_PROFILE, AircraftLayout, PropellerStation, tail_trailing_edge_m
from src.tools.gear import PROP_TIP_CLEARANCE_FRACTION, ROTATION_ANGLE_DEG, GearContact, GearGeometry

BELLY_LANDING_PITCH_DEG: Final[float] = 5.0  # nose-up attitude of a gearless aircraft touching down on its belly


@dataclass(frozen=True)
class PropClearance:
    """`tail_gap_m`: axial gap between the disc and the airframe next to it (a pusher's disc behind the tail
    trailing edge and the pod's tail, a tractor's ahead of the tail leading edge); negative is a strike. `ground_m`: tip to ground on the gear (None
    without gear). `belly_m`: tip above the lowest belly point at the landing pitch (only without gear)."""

    tail_gap_m: float
    ground_m: Optional[float]
    belly_m: Optional[float]
    needs_folding_prop: bool

    @property
    def strikes(self) -> bool:
        """The propeller hits the tail or the ground: a collision-class failure."""
        return self.tail_gap_m < 0 or (self.ground_m is not None and self.ground_m < 0)


def _tail_gap_m(layout: AircraftLayout) -> float:
    prop: PropellerStation = layout.propeller
    if prop.is_pusher:
        return prop.x_m - max(tail_trailing_edge_m(layout.tail_surfaces), layout.fuselage_length_m)
    return min(t.planform.x_le_root_m for t in layout.tail_surfaces) - prop.x_m


def _main_contact(gear: GearGeometry) -> GearContact:
    """A main-gear contact: the aircraft rotates about the main wheels."""
    mains = [c for c in gear.contacts if not c.steerable]
    if not mains:
        raise ValueError(f"a {gear.kind!r} gear has no main-gear contact to rotate about")
    return mains[0]


def _ground_m(prop: PropellerStation, gear: GearGeometry, launch: str) -> Optional[float]:
    """Tip-to-ground distance on the gear: rotated ROTATION_ANGLE_DEG about the main wheels for a runway launch
    (a disc aft of the mains drops), level for a vertical launch; None without gear."""
    if gear.kind == "none":
        return None
    main: GearContact = _main_contact(gear)
    tip_z: float = prop.z_m - prop.diameter_m / 2
    if launch == "vertical":
        return tip_z - main.z_m
    if launch != "runway":
        raise ValueError(f"no ground-clearance rule for a {launch!r} launch with a {gear.kind!r} gear")
    theta: float = math.radians(ROTATION_ANGLE_DEG)
    return (tip_z - main.z_m) * math.cos(theta) - (prop.x_m - main.x_m) * math.sin(theta)


def _pitched_height_m(x_m: float, z_m: float, pitch_rad: float) -> float:
    """Height of a body point with the aircraft pitched nose-up by `pitch_rad` (aft points drop)."""
    return z_m * math.cos(pitch_rad) - x_m * math.sin(pitch_rad)


def _belly_m(layout: AircraftLayout) -> float:
    """Disc tip above the lowest point of the belly line (pod profile and tail-boom bottoms; the profile is
    piecewise linear, so its lowest point is a vertex) at BELLY_LANDING_PITCH_DEG nose-up."""
    pitch: float = math.radians(BELLY_LANDING_PITCH_DEG)
    length, radius = layout.fuselage_length_m, layout.fuselage_diameter_m / 2
    hull: list[tuple[float, float]] = [(0.0, 0.0), *((f * length, -d * radius) for f, d in FUSELAGE_PROFILE),
                                       (length, 0.0)]
    for boom in layout.tail_booms:
        bottom: float = boom.z_m - boom.diameter_m / 2
        hull += [(boom.x_start_m, bottom), (boom.x_end_m, bottom)]
    lowest: float = min(_pitched_height_m(x, z, pitch) for x, z in hull)
    prop: PropellerStation = layout.propeller
    return _pitched_height_m(prop.x_m, prop.z_m - prop.diameter_m / 2, pitch) - lowest


def prop_clearance(layout: AircraftLayout, gear: GearGeometry, x_cg_m: float, launch: str) -> PropClearance:
    """Clearances of the layout's propeller with `gear`, which was built at the CG `x_cg_m` (its main wheels sit
    there; the CG is checked, the contacts carry it). A hand-launched pusher always folds its propeller; any
    propeller folds when its tip is within PROP_TIP_CLEARANCE_FRACTION x radius of the belly line."""
    if not math.isfinite(x_cg_m):
        raise ValueError(f"x_cg_m must be finite, got {x_cg_m}")
    belly: Optional[float] = _belly_m(layout) if gear.kind == "none" else None
    radius: float = layout.propeller.diameter_m / 2
    folds: bool = (launch == "hand" and layout.propeller.is_pusher) or (
        belly is not None and belly < PROP_TIP_CLEARANCE_FRACTION * radius)
    return PropClearance(tail_gap_m=_tail_gap_m(layout), ground_m=_ground_m(layout.propeller, gear, launch),
                         belly_m=belly, needs_folding_prop=folds)
