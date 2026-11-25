"""Write the saved flight-control fixtures under tests/data/flight_control/<case>/ (requirements, sizing, bom, aero,
cad JSON, as `tests.test_flight_control.load_design` reads them), each from one sizing + BOM + aero/CAD run of the
current code. Needs the Docker toolchain (OpenVSP/VSPAERO):

    docker run --rm -v "$PWD":/app -w /app isr-forge:latest python scripts/generate_flight_control_fixtures.py [case ...]

Cases: g1 and g2 (the reference missions) and g1_vtail (the G1 mission, with its original mission name, on the
single-boom V-tail pusher). No case named: all three. Last regenerated 2026-11-25 on the matched propulsion.
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel  # noqa: E402

from src.schemas.requirements import OperationalRequirements  # noqa: E402
from src.tools.aero_cad.build import build_aero_cad  # noqa: E402
from src.tools.database.matcher import select_bom  # noqa: E402
from src.tools.database.seed_data import seed_database  # noqa: E402
from src.tools.sizing.mission_solver import component_needs, size_aircraft  # noqa: E402
from tests.reference_designs import CASES  # noqa: E402

VTAIL_LAYOUT: Final[str] = "single_boom_vtail_pusher"
FIXTURE_CASES: Final[tuple[str, ...]] = ("g1", "g2", "g1_vtail")
FIXTURE_PREFIX: Final[str] = "fixture"
DATA_DIR: Final[Path] = Path(__file__).resolve().parents[1] / "tests" / "data" / "flight_control"


def _requirements(case: str) -> OperationalRequirements:
    """The fixture's mission: a reference case, or (g1_vtail) G1 on the V-tail pusher layout."""
    if case == "g1_vtail":
        base = CASES["g1"]
        return type(base).model_validate({**base.model_dump(), "airframe_layout": VTAIL_LAYOUT})
    return CASES[case]


def _write(folder: Path, name: str, record: BaseModel) -> None:
    path: Path = folder / f"{name}.json"
    computed: set[str] = set(type(record).model_computed_fields)  # derived on load, refused as input
    path.write_text(json.dumps(record.model_dump(mode="json", exclude=computed), indent=2) + "\n")
    print(f"[Tool] fixture: wrote {path}")


def generate(case: str) -> None:
    """Size, select, build and balance `case`, then write its five records."""
    req = _requirements(case)
    sizing = size_aircraft(req)
    with tempfile.TemporaryDirectory() as tmp:
        bom = select_bom(seed_database(Path(tmp) / "cots_catalog.db"), component_needs(sizing, req))
        aero, cad = build_aero_cad(sizing, bom, req.service_ceiling_m, Path(tmp) / "aero_cad")
        # the run's files are not kept: their paths become "fixture/<relative path>"
        cad = cad.model_copy(update={name: f"{FIXTURE_PREFIX}/{Path(value).relative_to(Path(tmp) / 'aero_cad')}"
                                     for name, value in cad.model_dump().items()
                                     if isinstance(value, str) and value.startswith(tmp)})
    folder: Path = DATA_DIR / case
    folder.mkdir(parents=True, exist_ok=True)
    for name, record in (("requirements", req), ("sizing", sizing), ("bom", bom), ("aero", aero), ("cad", cad)):
        _write(folder, name, record)
    print(f"[Tool] fixture {case}: as-built {cad.as_built_mass_kg:.3f} kg, SM {cad.static_margin:.3f}, "
          f"kit: {cad.kit_warning or 'built'}")


def main(cases: list[str]) -> None:
    unknown = sorted(set(cases) - set(FIXTURE_CASES))
    if unknown:
        raise SystemExit(f"unknown fixture case(s) {unknown}; known: {list(FIXTURE_CASES)}")
    for case in cases or FIXTURE_CASES:
        generate(case)


if __name__ == "__main__":
    main(sys.argv[1:])
