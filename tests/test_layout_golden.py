"""Phase 1 gate: every reference output is unchanged by the airframe-layout refactor.

The stored snapshots come from the code before the refactor (scripts/regenerate_layout_golden.py). If one of these
fails, the code is wrong: fix the code, never the snapshot.
"""

import pytest

from tests.golden import GOLDEN_DIR, snapshot

VSP_KEYS = ("sizing", "layout", "params", "jsbsim", "kit", "vsp_geometry")
EXPECTED = {**{case: set(VSP_KEYS) for case in ("g1", "g2")}, "g3": {"sizing", "layout"}}
STORED = sorted(GOLDEN_DIR.glob("*/*.json"))


@pytest.fixture(scope="module")
def snapshots() -> dict[str, dict[str, str]]:
    return {}  # case -> snapshot, filled lazily so each case is computed once


@pytest.mark.parametrize("path", STORED, ids=[f"{p.parent.name}-{p.stem}" for p in STORED])
def test_reference_outputs_unchanged(path, snapshots):
    case, key = path.parent.name, path.stem
    if key == "vsp_geometry":
        pytest.importorskip("openvsp")
    if case not in snapshots:
        snapshots[case] = snapshot(case)
    assert snapshots[case][key] == path.read_text()


def test_every_expected_snapshot_is_stored():
    stored = {case: {p.stem for p in (GOLDEN_DIR / case).glob("*.json")} for case in EXPECTED}
    assert stored == EXPECTED
    assert {p.name for p in GOLDEN_DIR.iterdir()} == set(EXPECTED)
