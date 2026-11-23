"""Write the V-tail G1 flight-control fixture under tests/data/flight_control/g1_vtail/ (requirements, sizing, bom,
aero, cad JSON, as `tests.test_flight_control.load_design` reads them) from one pipeline run. Needs the Docker toolchain
(OpenVSP/VSPAERO):

    docker run --rm -v "$PWD":/app -w /app isr-forge:latest python scripts/generate_vtail_g1_fixture.py

The requirements are the G1 reference mission (with its original mission name) on the single-boom V-tail pusher.
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel  # noqa: E402

from src.tools.aero_cad.build import build_aero_cad  # noqa: E402
from src.tools.database.matcher import select_bom  # noqa: E402
from src.tools.database.seed_data import seed_database  # noqa: E402
from src.tools.sizing.mission_solver import component_needs, size_aircraft  # noqa: E402
from tests.reference_designs import CASES  # noqa: E402

LAYOUT: Final[str] = "single_boom_vtail_pusher"
FIXTURE_PREFIX: Final[str] = "fixture"
OUT_DIR: Final[Path] = Path(__file__).resolve().parents[1] / "tests" / "data" / "flight_control" / "g1_vtail"


def _write(name: str, record: BaseModel) -> None:
    path: Path = OUT_DIR / f"{name}.json"
    computed: set[str] = set(type(record).model_computed_fields)  # derived on load, refused as input
    path.write_text(json.dumps(record.model_dump(mode="json", exclude=computed), indent=2) + "\n")
    print(f"[Tool] fixture: wrote {path}")


def main() -> None:
    base = CASES["g1"]
    req = type(base).model_validate({**base.model_dump(), "airframe_layout": LAYOUT})
    sizing = size_aircraft(req)
    with tempfile.TemporaryDirectory() as tmp:
        bom = select_bom(seed_database(Path(tmp) / "cots_catalog.db"), component_needs(sizing, req))
        aero, cad = build_aero_cad(sizing, bom, req.service_ceiling_m, Path(tmp) / "aero_cad")
        # the run's files are not kept: their paths become "fixture/<relative path>", like the g1/g2 fixtures
        cad = cad.model_copy(update={name: f"{FIXTURE_PREFIX}/{Path(value).relative_to(Path(tmp) / 'aero_cad')}"
                                     for name, value in cad.model_dump().items()
                                     if isinstance(value, str) and value.startswith(tmp)})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, record in (("requirements", req), ("sizing", sizing), ("bom", bom), ("aero", aero), ("cad", cad)):
        _write(name, record)


if __name__ == "__main__":
    main()
