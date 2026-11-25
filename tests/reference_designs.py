"""Reference missions shared by the sizing (M2), aero/CAD (M3) and UI tests (defined once in src.ui.examples)."""

import json
from pathlib import Path

from src.schemas.components import SelectedBOM
from src.schemas.requirements import OperationalRequirements
from src.schemas.state import SizingOutput
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

# Lighter variants that still build a laser-cut kit, so the kit code stays under test while the reference designs'
# kits are refused (user, 2026-11-25: "accept, kit fixes later"). Not reference missions: test inputs only.
G1_KIT_ENDURANCE_H: float = 1.75  # G1 at 1.75 h (4.12 kg) builds the tractor kit; at 2 h (4.45 kg) it does not
KIT_VARIANTS: dict[str, OperationalRequirements] = {
    "g1_kit": G1_ELECTRIC_FW.model_copy(update={"mission_name": "G1 kit variant (1.75 h)",
                                                "endurance_hours": G1_KIT_ENDURANCE_H}),
}

# Saved records that still build the laser-cut kit: the reference missions as master df94b76 sized them (before
# propulsion matching). The kit, CAD-packing and fabrication tests run on these while the matched references' kits
# are refused (provenance and the command that wrote them: tests/data/kit_designs/README.txt).
KIT_DESIGN_CASES: tuple[str, ...] = ("g1", "g2", "g2_vtail", "g1_hand")
KIT_DESIGN_DIR: Path = Path(__file__).parent / "data" / "kit_designs"


def saved_kit_design(case: str) -> tuple[OperationalRequirements, SizingOutput, SelectedBOM]:
    """(requirements, sizing, bom) of a saved kit design under tests/data/kit_designs/<case>/."""
    folder: Path = KIT_DESIGN_DIR / case

    def read(name: str) -> dict:
        return json.loads((folder / f"{name}.json").read_text())

    return (OperationalRequirements.model_validate(read("requirements")), SizingOutput.model_validate(read("sizing")),
            SelectedBOM.model_validate(read("bom")))


__all__ = ["CASES", "KIT_VARIANTS", "KIT_DESIGN_CASES", "KIT_DESIGN_DIR", "saved_kit_design", "REQ_FAILED",
           "G1_ELECTRIC_FW", "G2_ELECTRIC_VTOL", "G3_HYBRID_VTOL", "G1_VTAIL_PUSHER_FW", "G2_VTAIL_PUSHER_VTOL",
           "G1_HAND_LAUNCH"]
