"""Shared test fixtures, plus test isolation: no test sees the developer's real LLM keys or
provider/model settings.

The design fixtures (`catalog`, `designs`, `g1_layout`, `g1_kit_inputs`,
`g2_kit_inputs`) are shared by the CAD and
fabrication tests; plain helpers live in `tests/cad_helpers.py`. The LLM-env snapshot is taken at import (before any fixture runs) so the opt-in live test
(tests/test_agent_e2e.py with ISR_FORGE_LIVE_LLM=1) can restore it explicitly.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest

from src.schemas.components import SelectedBOM
from src.schemas.fabrication import DEFAULT_FABRICATION, FabricationSettings
from src.schemas.state import SizingOutput
from src.tools.aero_cad.airfoil_geometry import AirfoilShape, load_airfoil
from src.tools.aero_cad.layout import AircraftLayout, layout_from_sizing
from src.tools.database.matcher import select_bom
from src.tools.database.seed_data import seed_database
from src.tools.sizing.mission_solver import component_needs, size_aircraft
from tests.reference_designs import CASES

if TYPE_CHECKING:  # the kit fixtures need build123d; sizing-only tests must collect without it (M5)
    from src.tools.fabrication.tubes import SparPlan

LLM_ENV_VARS: Final[tuple[str, ...]] = ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "ISR_FORGE_LLM_PROVIDER",
                                        "ISR_FORGE_MODEL")
LIVE_LLM_ENV: Final[dict[str, str]] = {k: os.environ[k] for k in LLM_ENV_VARS if os.environ.get(k)}


@pytest.fixture(autouse=True)
def _no_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in LLM_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------- shared design fixtures ---
@pytest.fixture(scope="module")
def catalog(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return seed_database(tmp_path_factory.mktemp("db") / "cots_catalog.db")


@pytest.fixture(scope="module")
def designs(catalog: Path) -> dict[str, tuple[SizingOutput, SelectedBOM]]:
    out: dict[str, tuple[SizingOutput, SelectedBOM]] = {}
    for key, req in CASES.items():
        sizing: SizingOutput = size_aircraft(req)
        out[key] = (sizing, select_bom(catalog, component_needs(sizing, req)))
    return out


@pytest.fixture()
def g1_layout(designs: dict) -> AircraftLayout:
    return layout_from_sizing(designs["g1"][0])


KitInputs = tuple[AircraftLayout, AirfoilShape, "SparPlan", FabricationSettings]


def _kit_inputs(designs: dict, case: str) -> KitInputs:
    """(layout, airfoil, spar plan, settings) for a reference design, so `wing_ribs(*inputs)` works. The CAD-stack
    imports live here, not at module level (M5)."""
    from src.tools.fabrication.build_kit import stiffened_spar_plan
    from tests.cad_helpers import _fitted_layout

    sizing, bom = designs[case]
    layout = _fitted_layout(sizing, bom)
    airfoil = load_airfoil(layout.airfoil)
    spar = stiffened_spar_plan(sizing, layout, airfoil, DEFAULT_FABRICATION)  # as build_kit plans it (torsion, pin)
    return layout, airfoil, spar, DEFAULT_FABRICATION


@pytest.fixture(scope="module")
def g1_kit_inputs(designs: dict) -> KitInputs:
    return _kit_inputs(designs, "g1")


@pytest.fixture(scope="module")
def g2_kit_inputs(designs: dict) -> KitInputs:
    return _kit_inputs(designs, "g2")


@pytest.fixture(scope="module")
def vtail_g1(designs: dict) -> tuple[SizingOutput, SelectedBOM]:
    """The G1 mission sized as a single-boom V-tail pusher with its matched BOM."""
    return designs["g1_vtail"]


@pytest.fixture(scope="module")
def vtail_g2(designs: dict) -> tuple[SizingOutput, SelectedBOM]:
    """The G2 mission (quadplane) sized as a single-boom V-tail pusher with its matched BOM."""
    return designs["g2_vtail"]


@pytest.fixture(scope="module")
def vtail_g1_sizing(vtail_g1: tuple[SizingOutput, SelectedBOM]) -> SizingOutput:
    return vtail_g1[0]
