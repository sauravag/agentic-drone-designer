"""Reference missions shared by the sizing (M2) and aero/CAD (M3) acceptance tests."""

from src.schemas.requirements import OperationalRequirements

G1_ELECTRIC_FW = OperationalRequirements(
    mission_name="G1 electric fixed-wing", payload_mass_kg=1.0, payload_power_w=15.0, endurance_hours=2.0,
    combat_radius_km=15.0, cruise_speed_ms=18.0, service_ceiling_m=500.0, max_stall_speed_ms=13.0,
    runway_available=True,
)
# SRS fixes payload (3 kg) and endurance (3 h); the remaining inputs describe a short-range mission.
# With a 20 km radius / 1000 m ceiling the same aircraft sizes to ~26 kg (Group 3) - see report.
G2_ELECTRIC_VTOL = OperationalRequirements(
    mission_name="G2 electric quadplane", payload_mass_kg=3.0, payload_power_w=30.0, endurance_hours=3.0,
    combat_radius_km=10.0, cruise_speed_ms=22.0, service_ceiling_m=500.0, max_stall_speed_ms=16.0,
    runway_available=False, propulsion_type="electric",
)
G3_HYBRID_VTOL = OperationalRequirements(
    mission_name="G3 hybrid quadplane", payload_mass_kg=20.0, payload_power_w=300.0, endurance_hours=12.0,
    combat_radius_km=150.0, cruise_speed_ms=40.0, service_ceiling_m=4000.0, max_stall_speed_ms=25.0,
    runway_available=False,
)
CASES: dict[str, OperationalRequirements] = {"g1": G1_ELECTRIC_FW, "g2": G2_ELECTRIC_VTOL, "g3": G3_HYBRID_VTOL}
