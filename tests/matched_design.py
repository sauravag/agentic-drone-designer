"""A G1 design sized with the propulsion match (the stored `load_design` records predate it)."""

import tempfile
from functools import cache
from pathlib import Path

from src.schemas.components import SelectedBOM
from src.schemas.state import SizingOutput
from src.tools.database.matcher import select_bom
from src.tools.database.seed_data import seed_database
from src.tools.sizing.mission_solver import component_needs, size_aircraft
from tests.reference_designs import G1_ELECTRIC_FW


@cache
def matched_g1() -> tuple[SizingOutput, SelectedBOM]:
    """G1 electric fixed-wing sized with the match, and its BOM from a freshly seeded catalogue."""
    sizing: SizingOutput = size_aircraft(G1_ELECTRIC_FW)
    db_path: Path = Path(tempfile.mkdtemp(prefix="matched_g1_")) / "catalog.db"
    seed_database(db_path)
    return sizing, select_bom(db_path, component_needs(sizing, G1_ELECTRIC_FW))
