"""Reference missions shared by the sizing (M2), aero/CAD (M3) and UI tests (defined once in src.ui.examples)."""

from src.schemas.requirements import OperationalRequirements
from src.ui.examples import (G1_ELECTRIC_FW, G1_HAND_LAUNCH, G1_VTAIL_PUSHER_FW, G2_ELECTRIC_VTOL,
                             G2_VTAIL_PUSHER_VTOL, G3_HYBRID_VTOL)

# Electric quadplane VTOL that fails to close at defaults (copy of the failing g1 run's requirements).
REQ_FAILED: OperationalRequirements = OperationalRequirements(
    mission_name="G1 electric fixed-wing", propulsion_type="electric", airframe_topology="quadplane_vtol",
    payload_mass_kg=5.0, payload_power_w=25.0, endurance_hours=6.0, combat_radius_km=100.0,
    cruise_speed_ms=30.0, service_ceiling_m=500.0, max_stall_speed_ms=13.0, runway_available=False,
)

CASES: dict[str, OperationalRequirements] = {
    "g1": G1_ELECTRIC_FW, "g2": G2_ELECTRIC_VTOL, "g3": G3_HYBRID_VTOL,
    "g1_vtail": G1_VTAIL_PUSHER_FW, "g2_vtail": G2_VTAIL_PUSHER_VTOL, "g1_hand": G1_HAND_LAUNCH,
}

__all__ = ["CASES", "REQ_FAILED", "G1_ELECTRIC_FW", "G2_ELECTRIC_VTOL", "G3_HYBRID_VTOL", "G1_VTAIL_PUSHER_FW",
           "G2_VTAIL_PUSHER_VTOL", "G1_HAND_LAUNCH"]
