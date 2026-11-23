"""Example missions: presets for the web form and the reference designs used by the test suite."""

from typing import Final

from src.schemas.requirements import OperationalRequirements

G1_ELECTRIC_FW: Final = OperationalRequirements(
    mission_name="G1 electric fixed-wing", payload_mass_kg=1.0, payload_power_w=15.0, endurance_hours=2.0,
    combat_radius_km=15.0, cruise_speed_ms=18.0, service_ceiling_m=500.0, max_stall_speed_ms=13.0,
    runway_available=True,
)
# SRS fixes payload (3 kg) and endurance (3 h); the remaining inputs describe a short-range mission.
# With a 20 km radius / 1000 m ceiling the same aircraft sizes to ~26 kg (Group 3).
G2_ELECTRIC_VTOL: Final = OperationalRequirements(
    mission_name="G2 electric quadplane", payload_mass_kg=3.0, payload_power_w=30.0, endurance_hours=3.0,
    combat_radius_km=10.0, cruise_speed_ms=22.0, service_ceiling_m=500.0, max_stall_speed_ms=16.0,
    runway_available=False, propulsion_type="electric",
)
G3_HYBRID_VTOL: Final = OperationalRequirements(
    mission_name="G3 hybrid quadplane", payload_mass_kg=20.0, payload_power_w=300.0, endurance_hours=12.0,
    combat_radius_km=150.0, cruise_speed_ms=40.0, service_ceiling_m=4000.0, max_stall_speed_ms=25.0,
    runway_available=False,
)

VTAIL_PUSHER: Final[str] = "single_boom_vtail_pusher"


def _variant(base: OperationalRequirements, **changes: object) -> OperationalRequirements:
    """`base` with `changes`, validated (model_copy would skip the layout and launch checks)."""
    return OperationalRequirements.model_validate({**base.model_dump(), **changes})


# The G1 and G2 missions flown on the single-boom V-tail pusher (fixed-wing off a runway; electric quadplane).
G1_VTAIL_PUSHER_FW: Final = _variant(G1_ELECTRIC_FW, mission_name="G1 V-tail pusher fixed-wing",
                                     airframe_layout=VTAIL_PUSHER)
G2_VTAIL_PUSHER_VTOL: Final = _variant(G2_ELECTRIC_VTOL, mission_name="G2 V-tail pusher quadplane",
                                       airframe_layout=VTAIL_PUSHER)
# The G1 mission thrown by hand: no runway and no VTOL. The hand launch caps the stall speed (topology.py), so the
# wing is larger than the runway G1's.
G1_HAND_LAUNCH: Final = _variant(G1_ELECTRIC_FW, mission_name="G1 hand-launched fixed-wing", runway_available=False,
                                 launch_method="hand")

EXAMPLE_MISSIONS: Final[dict[str, OperationalRequirements]] = {
    "G1 electric fixed-wing (reference)": G1_ELECTRIC_FW,
    "G2 electric quadplane (reference)": G2_ELECTRIC_VTOL,
    "G3 hybrid quadplane (reference)": G3_HYBRID_VTOL,
    "G1 V-tail pusher fixed-wing (reference)": G1_VTAIL_PUSHER_FW,
    "G2 V-tail pusher quadplane (reference)": G2_VTAIL_PUSHER_VTOL,
    "G1 hand-launched fixed-wing (reference)": G1_HAND_LAUNCH,
}
