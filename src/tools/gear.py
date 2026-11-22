"""One landing-gear description shared by sizing (mass), the fabrication kit (residual mass) and the JSBSim model
(contact points). Depends on nothing else in the repo."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

GEAR_CLEARANCE_FRACTION: Final[float] = 0.25  # extra ground clearance, fraction of the larger radius
NOSE_GEAR_STATION: Final[float] = 0.12  # fraction of fuselage length
MAIN_GEAR_AFT_OF_CG_CHORDS: Final[float] = 0.15
TIP_BACK_ANGLE_DEG: Final[float] = 15.0  # main gear aft of the CG by at least h_cg * tan(15 deg)
MAIN_GEAR_TRACK_FRACTION: Final[float] = 0.15  # of wingspan, each side
GEAR_FRACTION: Final[dict[str, float]] = {"fixed_wing": 0.04, "quadplane_vtol": 0.015}  # of MTOW, by topology


@dataclass(frozen=True)
class GearContact:
    name: str
    x_m: float
    y_m: float
    z_m: float
    steerable: bool


@dataclass(frozen=True)
class GearGeometry:
    kind: Literal["none", "tricycle"]
    contacts: tuple[GearContact, ...]


def tricycle_gear(fuselage_length_m: float, fuselage_diameter_m: float, prop_diameter_m: float,
                  mean_chord_m: float, wingspan_m: float, x_cg_m: float) -> GearGeometry:
    """Tricycle gear in structural axes (x aft of nose, y right, z up; fuselage axis at z = 0)."""
    radius: float = max(fuselage_diameter_m, prop_diameter_m) / 2
    z: float = -radius * (1 + GEAR_CLEARANCE_FRACTION)
    aft: float = max(MAIN_GEAR_AFT_OF_CG_CHORDS * mean_chord_m, -z * math.tan(math.radians(TIP_BACK_ANGLE_DEG)))
    x_main: float = x_cg_m + aft
    track: float = MAIN_GEAR_TRACK_FRACTION * wingspan_m
    return GearGeometry("tricycle", (GearContact("NOSE", NOSE_GEAR_STATION * fuselage_length_m, 0.0, z, True),
                                     GearContact("LEFT_MAIN", x_main, -track, z, False),
                                     GearContact("RIGHT_MAIN", x_main, track, z, False)))


def gear_mass_kg(airframe_topology: str, mtow_kg: float) -> float:
    """Landing-gear mass: GEAR_FRACTION of MTOW for the airframe topology."""
    return GEAR_FRACTION[airframe_topology] * mtow_kg
