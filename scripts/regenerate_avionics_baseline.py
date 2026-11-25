"""Write tests/data/avionics_baseline.json (the G1/G2/G3 sizing fields `tests/test_avionics.py` pins) from the current
code. Only for a deliberate re-baseline (the last: 2026-11-25, matched propulsion); otherwise a changed value means
the code is wrong.

    .venv/bin/python scripts/regenerate_avionics_baseline.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tools.sizing.mission_solver import size_aircraft  # noqa: E402
from tests.test_avionics import BASELINE_FIELDS, BASELINE_MISSIONS, BASELINE_PATH  # noqa: E402


def main() -> None:
    baseline: dict[str, dict[str, float]] = {}
    for req in BASELINE_MISSIONS:
        sizing = size_aircraft(req)
        baseline[req.mission_name] = {field: getattr(sizing, field) for field in BASELINE_FIELDS}
        print(f"[Tool] avionics baseline: {req.mission_name} MTOW {sizing.mtow_kg:.3f} kg")
    BASELINE_PATH.write_text(json.dumps(baseline, indent=1) + "\n")


if __name__ == "__main__":
    main()
