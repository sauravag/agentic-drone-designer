"""One landing-gear description shared by sizing (mass), the fabrication kit (residual mass) and the JSBSim model
(contact points). Depends on nothing else in the repo."""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, Mapping

GEAR_CLEARANCE_FRACTION: Final[float] = 0.25  # extra ground clearance, fraction of the larger radius
NOSE_GEAR_STATION: Final[float] = 0.12  # fraction of fuselage length
MAIN_GEAR_AFT_OF_CG_CHORDS: Final[float] = 0.15
TIP_BACK_ANGLE_DEG: Final[float] = 15.0  # main gear aft of the CG by at least h_cg * tan(15 deg)
MAIN_GEAR_TRACK_FRACTION: Final[float] = 0.15  # of wingspan, each side
ROTATION_ANGLE_DEG: Final[float] = 12.0  # runway take-off rotation the propeller must clear the ground at
PROP_TIP_CLEARANCE_FRACTION: Final[float] = 0.25  # propeller tip to ground, fraction of propeller radius
GEAR_FRACTION: Final[Mapping[str, float]] = MappingProxyType(
    {"fixed_wing": 0.04, "quadplane_vtol": 0.015})  # of MTOW, by topology


@dataclass(frozen=True)
class GearContact:
    name: str
    x_m: float
    y_m: float
    z_m: float
    steerable: bool


@dataclass(frozen=True)
class GearGeometry:
    """`leg_height_m` is the contact depth below the fuselage axis (0 for none); `base_leg_height_m` the height the
    pre-clearance rule gave, so the mass scales with the leg relative to it."""
    kind: Literal["none", "tricycle"]
    contacts: tuple[GearContact, ...]
    leg_height_m: float
    base_leg_height_m: float = 0.0


@dataclass(frozen=True)
class PropellerDisc:
    """The cruise propeller in structural axes: hub position and diameter."""
    x_m: float
    z_m: float
    diameter_m: float


@dataclass(frozen=True)
class GearInputs:
    launch: Literal["runway", "hand", "vertical"]
    fuselage_length_m: float
    fuselage_diameter_m: float
    mean_chord_m: float
    wingspan_m: float
    x_cg_m: float
    propeller: PropellerDisc


def _rotation_leg_height_m(inputs: GearInputs, x_main_min_m: float) -> float:
    """Smallest leg height whose propeller tip clears the ground by PROP_TIP_CLEARANCE_FRACTION x radius with the
    aircraft rotated ROTATION_ANGLE_DEG about the main-gear contact. The main gear sits at
    max(x_main_min, x_cg + h tan(tip-back)); the clearance rises with h, so the answer is the smaller of the
    heights that satisfy each branch of that max."""
    prop = inputs.propeller
    radius: float = prop.diameter_m / 2
    theta: float = math.radians(ROTATION_ANGLE_DEG)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    tan_back: float = math.tan(math.radians(TIP_BACK_ANGLE_DEG))
    # clearance = (z_tip + h) cos - (x_prop - x_main) sin >= c
    required: float = PROP_TIP_CLEARANCE_FRACTION * radius - (prop.z_m - radius) * cos_t
    fixed_main: float = (required + (prop.x_m - x_main_min_m) * sin_t) / cos_t
    swung_main: float = (required + (prop.x_m - inputs.x_cg_m) * sin_t) / (cos_t + tan_back * sin_t)
    return min(fixed_main, swung_main)


def landing_gear(inputs: GearInputs) -> GearGeometry:
    """The one landing-gear rule, in structural axes (x aft of nose, y right, z up; fuselage axis at z = 0).
    hand launch: no gear. Otherwise a tricycle whose leg is the tallest of: the ground clearance of the larger of
    fuselage and propeller (`base`), the level propeller clearance (vertical), or the clearance at the rotation
    angle (runway)."""
    if inputs.launch == "hand":
        return GearGeometry("none", (), 0.0)
    radius: float = max(inputs.fuselage_diameter_m, inputs.propeller.diameter_m) / 2
    base: float = radius * (1 + GEAR_CLEARANCE_FRACTION)
    x_main_min: float = inputs.x_cg_m + MAIN_GEAR_AFT_OF_CG_CHORDS * inputs.mean_chord_m
    prop = inputs.propeller
    if inputs.launch == "vertical":
        needed: float = (1 + PROP_TIP_CLEARANCE_FRACTION) * prop.diameter_m / 2 - prop.z_m
    else:
        needed = _rotation_leg_height_m(inputs, x_main_min)
    height: float = max(base, needed)
    aft: float = max(MAIN_GEAR_AFT_OF_CG_CHORDS * inputs.mean_chord_m,
                     height * math.tan(math.radians(TIP_BACK_ANGLE_DEG)))
    x_main: float = inputs.x_cg_m + aft
    track: float = MAIN_GEAR_TRACK_FRACTION * inputs.wingspan_m
    z: float = -height
    return GearGeometry("tricycle", (GearContact("NOSE", NOSE_GEAR_STATION * inputs.fuselage_length_m, 0.0, z, True),
                                     GearContact("LEFT_MAIN", x_main, -track, z, False),
                                     GearContact("RIGHT_MAIN", x_main, track, z, False)), height, base)


def gear_mass_kg(gear: GearGeometry, airframe_topology: str, mtow_kg: float) -> float:
    """Landing-gear mass: GEAR_FRACTION of MTOW for the airframe topology, scaled by the leg height over its base
    height (so a tractor's mass is unchanged and a taller leg weighs proportionally more); 0 without gear."""
    if airframe_topology not in GEAR_FRACTION:
        raise ValueError(f"unknown airframe topology {airframe_topology!r}; expected one of {sorted(GEAR_FRACTION)}")
    if gear.kind == "none":
        return 0.0
    return GEAR_FRACTION[airframe_topology] * mtow_kg * gear.leg_height_m / gear.base_leg_height_m
