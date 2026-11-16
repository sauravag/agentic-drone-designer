"""Viewer page builder: the data block, self-containment, and which run files feed the page."""

import json
import re
from pathlib import Path
from typing import Any, Optional

import pytest

from src.viewer import build
from src.viewer.build import ViewerError, build_viewer_html, load_trace, write_viewer

MODEL: dict[str, Any] = {
    "frame": "x_aft_y_right_z_up", "cg_m": [0.5, 0, 0], "mass_kg": 10.0,
    "parts": [{"name": "</script><img src=x onerror=alert(1)>", "group": "component", "mass_kg": 1.0,
               "positions_b64": "", "indices_b64": ""}],
}
TRACE: list[dict[str, float]] = [{"t_s": 0.0, "agl_m": 0.2, "roll_deg": 0.0, "pitch_deg": 0.0, "airspeed_ms": 0.0,
                                  "heading_deg": 90.0, "north_m": 0.0, "east_m": 0.0}]


def _data_block(html: str) -> dict[str, Any]:
    return json.loads(html.split('id="viewer-data">', 1)[1].split("</script>", 1)[0])


def _write_run(run_dir: Path, replay_attempt: Optional[int], passed: bool = True) -> None:
    """A run folder with a model, two SITL attempts whose traces differ by agl_m, and a report."""
    (run_dir / build.VIEWER_MODEL_FILE).write_text(json.dumps(MODEL))
    for attempt in (1, 2):
        attempt_dir: Path = run_dir / "sitl" / f"attempt_{attempt}"
        attempt_dir.mkdir(parents=True)
        (attempt_dir / build.TRUTH_FILE).write_text(json.dumps([{**TRACE[0], "agl_m": float(attempt)}]))
    report: dict[str, Any] = {"passed": passed, "replay_attempt": replay_attempt}
    (run_dir / build.SITL_REPORT_FILE).write_text(json.dumps(report))


def test_data_cannot_close_the_script_block() -> None:
    html: str = build_viewer_html(MODEL, TRACE, True)
    assert html.count('id="viewer-data"') == 1 and "</script><img" not in html
    assert _data_block(html)["model"]["parts"][0]["name"].startswith("</script>")


def test_the_page_is_self_contained() -> None:
    html: str = build_viewer_html(MODEL, None, None)
    assert not re.search(r'(src|href)=["\']https?://|from\s*["\']https?://', html)
    assert "importmap" in html and len(html.encode()) < 8 * 1024 * 1024


def test_the_page_inlines_the_vendored_library() -> None:
    html: str = build_viewer_html(MODEL, None, None)
    assert '"three": "data:text/javascript;base64,' in html
    assert '"three/addons/controls/OrbitControls.js": "data:text/javascript;base64,' in html


def test_no_trace_means_no_replay_data() -> None:
    data: dict[str, Any] = _data_block(build_viewer_html(MODEL, None, None))
    assert data["trace"] is None and data["sitl_passed"] is None


def test_write_viewer_uses_the_reported_attempt(tmp_path: Path) -> None:
    _write_run(tmp_path, replay_attempt=2, passed=False)
    path: Optional[Path] = write_viewer(tmp_path)
    assert path == tmp_path / build.VIEWER_FILE
    data: dict[str, Any] = _data_block(path.read_text())
    assert data["trace"][0]["agl_m"] == 2.0 and data["sitl_passed"] is False


def test_a_null_replay_attempt_means_no_trace(tmp_path: Path) -> None:
    _write_run(tmp_path, replay_attempt=None)
    assert load_trace(tmp_path) == (None, None)


def test_a_missing_report_or_trace_means_no_trace(tmp_path: Path) -> None:
    assert load_trace(tmp_path) == (None, None)
    _write_run(tmp_path, replay_attempt=3)  # attempt_3 was never written
    assert load_trace(tmp_path) == (None, None)


def test_write_viewer_without_a_model_returns_none(tmp_path: Path) -> None:
    assert write_viewer(tmp_path) is None
    assert not (tmp_path / build.VIEWER_FILE).exists()


@pytest.mark.parametrize("content", ["{not json", json.dumps({"frame": "x_aft_y_right_z_up"}), json.dumps([1, 2])])
def test_a_malformed_model_raises_viewer_error(tmp_path: Path, content: str) -> None:
    (tmp_path / build.VIEWER_MODEL_FILE).write_text(content)
    with pytest.raises(ViewerError):
        write_viewer(tmp_path)


def test_a_malformed_trace_raises_viewer_error(tmp_path: Path) -> None:
    _write_run(tmp_path, replay_attempt=1)
    (tmp_path / "sitl" / "attempt_1" / build.TRUTH_FILE).write_text('{"t_s": 0}')
    with pytest.raises(ViewerError):
        write_viewer(tmp_path)


def test_file_names_match_their_sources() -> None:
    from src.tools.aero_cad import mesh_export

    assert build.VIEWER_MODEL_FILE == mesh_export.VIEWER_MODEL_FILE
    sitl_validator = pytest.importorskip("src.tools.flight_control.sitl_validator")
    assert build.TRUTH_FILE == sitl_validator.TRUTH_FILE
    assert build.SITL_REPORT_FILE == sitl_validator.REPORT_FILE


def test_a_non_finite_trace_value_raises_viewer_error(tmp_path: Path) -> None:
    _write_run(tmp_path, replay_attempt=1)
    (tmp_path / "sitl" / "attempt_1" / build.TRUTH_FILE).write_text(json.dumps([{**TRACE[0], "agl_m": float("nan")}]))
    with pytest.raises(ViewerError):
        write_viewer(tmp_path)


def test_a_non_finite_unchecked_value_raises_viewer_error() -> None:
    with pytest.raises(ViewerError):
        build_viewer_html(MODEL, [{**TRACE[0], "alpha_deg": float("inf")}], True)
