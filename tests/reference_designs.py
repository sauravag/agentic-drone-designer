"""Reference missions shared by the sizing (M2), aero/CAD (M3) and UI tests (defined once in src.ui.examples)."""

from src.schemas.requirements import OperationalRequirements
from src.ui.examples import G1_ELECTRIC_FW, G2_ELECTRIC_VTOL, G3_HYBRID_VTOL

CASES: dict[str, OperationalRequirements] = {"g1": G1_ELECTRIC_FW, "g2": G2_ELECTRIC_VTOL, "g3": G3_HYBRID_VTOL}

__all__ = ["CASES", "G1_ELECTRIC_FW", "G2_ELECTRIC_VTOL", "G3_HYBRID_VTOL"]
