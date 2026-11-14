"""DoD group classification and airframe / powertrain selection rules (SRS §4.1.1)."""

import math
from typing import Final, Literal

from src.schemas.requirements import OperationalRequirements

Airframe = Literal["fixed_wing", "quadplane_vtol"]
Propulsion = Literal["electric", "series_hybrid"]

GROUP1_UPPER_KG: Final[float] = 9.07  # 20 lb
GROUP2_UPPER_KG: Final[float] = 24.95  # 55 lb
GROUP3_UPPER_KG: Final[float] = 599.0  # 1,320 lb

ELECTRIC_MAX_MTOW_KG: Final[float] = 15.0
ELECTRIC_MAX_ENDURANCE_H: Final[float] = 3.0



class GroupClassificationError(ValueError):
    """The converged design falls outside DoD Groups 1-3 (a mission-level infeasibility)."""

def classify_group(mtow_kg: float) -> Literal[1, 2, 3]:
    """Map MTOW to DoD UAS group. Raises ValueError outside Group 1-3 or for non-physical mass."""
    if not math.isfinite(mtow_kg) or mtow_kg <= 0:
        raise ValueError(f"MTOW must be finite and positive, got {mtow_kg!r}")
    if mtow_kg < GROUP1_UPPER_KG:
        return 1
    if mtow_kg < GROUP2_UPPER_KG:
        return 2
    if mtow_kg < GROUP3_UPPER_KG:
        return 3
    raise GroupClassificationError(f"MTOW {mtow_kg:.1f} kg exceeds the Group 3 limit of {GROUP3_UPPER_KG} kg")


def select_airframe(req: OperationalRequirements) -> Airframe:
    if req.airframe_topology != "auto":
        return req.airframe_topology
    return "fixed_wing" if req.runway_available else "quadplane_vtol"


def select_propulsion(req: OperationalRequirements, estimated_mtow_kg: float) -> Propulsion:
    if req.propulsion_type != "auto":
        return req.propulsion_type
    is_small: bool = estimated_mtow_kg < ELECTRIC_MAX_MTOW_KG
    is_short: bool = req.endurance_hours <= ELECTRIC_MAX_ENDURANCE_H
    return "electric" if is_small and is_short else "series_hybrid"
