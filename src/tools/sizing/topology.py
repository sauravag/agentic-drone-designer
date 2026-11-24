"""DoD group classification and airframe / powertrain selection rules (SRS §4.1.1)."""

import math
from typing import Final, Literal, Optional

from src.schemas.requirements import OperationalRequirements

Airframe = Literal["fixed_wing", "quadplane_vtol"]
Propulsion = Literal["electric", "series_hybrid"]
LaunchMethod = Literal["runway", "hand", "vertical"]

GROUP1_UPPER_KG: Final[float] = 9.07  # 20 lb
GROUP2_UPPER_KG: Final[float] = 24.95  # 55 lb
GROUP3_UPPER_KG: Final[float] = 599.0  # 1,320 lb

HAND_LAUNCH_MAX_MTOW_KG: Final[float] = GROUP1_UPPER_KG  # a thrown aircraft is Group 1
HAND_LAUNCH_SPEED_MS: Final[float] = 10.0  # throw speed along the body x axis
HAND_LAUNCH_HEIGHT_M: Final[float] = 2.0  # release height above the ground
HAND_LAUNCH_PITCH_DEG: Final[float] = 10.0  # nose-up attitude at release
HAND_LAUNCH_STALL_MARGIN: Final[float] = 1.2  # take-off safety convention: throw speed >= 1.2 x stall speed
# Laminar share of the wing and tail boundary layers a hand-launched design is drag-sized for (the user's ruling,
# 2026-11-24): sizing trips NeuralFoil at this chord fraction and VSPAERO's ParasiteDrag runs this laminar fraction,
# so both models describe the same aircraft. Runway and VTOL designs keep their models' defaults.
HAND_LAUNCH_LAMINAR_FRACTION: Final[float] = 0.30
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
    if req.launch_method == "hand":
        return "fixed_wing"
    return "fixed_wing" if req.runway_available else "quadplane_vtol"


def resolved_launch_method(req: OperationalRequirements) -> LaunchMethod:
    """How the resolved aircraft launches: a quadplane vertically, a fixed-wing by hand or off a runway."""
    if select_airframe(req) == "quadplane_vtol":
        return "vertical"
    return "hand" if req.launch_method == "hand" else "runway"


def select_propulsion(req: OperationalRequirements, estimated_mtow_kg: float) -> Propulsion:
    if req.propulsion_type != "auto":
        return req.propulsion_type
    is_small: bool = estimated_mtow_kg < ELECTRIC_MAX_MTOW_KG
    is_short: bool = req.endurance_hours <= ELECTRIC_MAX_ENDURANCE_H
    return "electric" if is_small and is_short else "series_hybrid"


def effective_max_stall_speed_ms(req: OperationalRequirements) -> float:
    """The stall-speed limit sizing designs to at the take-off site: the requirement, or for a hand launch no more
    than the throw speed over HAND_LAUNCH_STALL_MARGIN (a thrown aircraft must fly away from its release speed)."""
    if resolved_launch_method(req) == "hand":
        return min(req.max_stall_speed_ms, HAND_LAUNCH_SPEED_MS / HAND_LAUNCH_STALL_MARGIN)
    return req.max_stall_speed_ms


def surface_laminar_fraction(launch: str) -> Optional[float]:
    """Laminar boundary-layer fraction (of chord) assumed on the lifting surfaces for a resolved launch method:
    HAND_LAUNCH_LAMINAR_FRACTION for a hand launch, None (each model's own transition) otherwise."""
    return HAND_LAUNCH_LAMINAR_FRACTION if launch == "hand" else None
