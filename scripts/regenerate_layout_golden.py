"""Write the golden snapshots under tests/data/golden/propulsion_matching/ (run on the host and again in Docker, which
adds `vsp_geometry`). Only for a deliberate re-baseline (the last: 2026-11-25, propulsion matching phase 1); otherwise
a changed snapshot means the code is wrong.

    .venv/bin/python scripts/regenerate_layout_golden.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.golden import GOLDEN_DIR, KIT_CASES, SIZING_CASES, snapshot  # noqa: E402


def main() -> None:
    for case in (*SIZING_CASES, *KIT_CASES):
        folder = GOLDEN_DIR / case
        folder.mkdir(parents=True, exist_ok=True)
        for key, text in snapshot(case).items():
            (folder / f"{key}.json").write_text(text)
            print(f"[golden] {case}/{key}.json ({len(text)} chars)")


if __name__ == "__main__":
    main()
