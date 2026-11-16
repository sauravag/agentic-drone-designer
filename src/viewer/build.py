"""Builds `viewer.html`: one self-contained page showing the packed aircraft and replaying its SITL flight.

The page inlines Three.js (as `data:` URLs in an import map), the page script and the run data, so it opens
from disk, inside a Streamlit iframe or as an e-mail attachment with no network access. Run data reaches the
page only through one `<script type="application/json" id="viewer-data">` block.
"""

import base64
import json
import math
import re
from pathlib import Path
from typing import Any, Final, Mapping, Optional, Sequence

VIEWER_FILE: Final[str] = "viewer.html"
# Copies of the producers' file names, so this module (and the UI importing it) never imports build123d or the
# Docker-only SITL stack. tests/test_viewer.py asserts they stay equal to their sources.
VIEWER_MODEL_FILE: Final[str] = "viewer_model.json"  # src/tools/aero_cad/mesh_export.py VIEWER_MODEL_FILE
TRUTH_FILE: Final[str] = "jsbsim_truth.json"  # src/tools/flight_control/sitl_validator.py TRUTH_FILE
SITL_REPORT_FILE: Final[str] = "sitl_report.json"  # src/tools/flight_control/sitl_validator.py REPORT_FILE
SITL_DIR: Final[str] = "sitl"
ATTEMPT_DIR_PREFIX: Final[str] = "attempt_"

PAGE_DIR: Final[Path] = Path(__file__).parent
TEMPLATE_FILE: Final[Path] = PAGE_DIR / "template.html"
SCRIPT_FILE: Final[Path] = PAGE_DIR / "viewer.js"
# Import-map specifier -> vendored file (three@0.170.0; three.module.min.js imports nothing else).
VENDOR_MODULES: Final[Mapping[str, Path]] = {
    "three": PAGE_DIR / "vendor" / "three.module.min.js",
    "three/addons/controls/OrbitControls.js": PAGE_DIR / "vendor" / "OrbitControls.js",
}
PLACEHOLDER: Final[re.Pattern[str]] = re.compile(r"\{\{(IMPORT_MAP|VIEWER_JS|VIEWER_DATA)\}\}")
PART_GROUPS: Final[frozenset[str]] = frozenset({"skin", "structure", "component"})
PART_STRING_KEYS: Final[tuple[str, ...]] = ("name", "positions_b64", "indices_b64")
CG_AXES: Final[int] = 3
TRACE_KEYS: Final[tuple[str, ...]] = ("t_s", "agl_m", "roll_deg", "pitch_deg", "airspeed_ms")


class ViewerError(RuntimeError):
    """The run files the viewer is built from are unreadable or malformed."""


def _script_safe_json(value: Any) -> str:
    """JSON that cannot end its <script> element: every `</` becomes `<\\/` and `<!--` is escaped too."""
    return json.dumps(value, allow_nan=False).replace("</", "<\\/").replace("<!--", "\\u003c!--")


def _import_map() -> str:
    imports: dict[str, str] = {
        name: f"data:text/javascript;base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"
        for name, path in VENDOR_MODULES.items()
    }
    return json.dumps({"imports": imports}, indent=1)


def build_viewer_html(model: Mapping[str, Any], trace: Optional[Sequence[Mapping[str, float]]],
                      sitl_passed: Optional[bool]) -> str:
    """The complete page for `model` and, when given, the replayed SITL `trace`."""
    data: dict[str, Any] = {
        "model": model,
        "trace": list(trace) if trace is not None else None,
        "sitl_passed": sitl_passed,
    }
    try:
        data_json: str = _script_safe_json(data)
    except ValueError as exc:  # NaN/Infinity anywhere in the data, e.g. an unchecked trace key
        raise ViewerError(f"viewer data is not finite JSON: {exc}") from exc
    values: dict[str, str] = {
        "IMPORT_MAP": _import_map(),
        "VIEWER_JS": SCRIPT_FILE.read_text(encoding="utf-8"),
        "VIEWER_DATA": data_json,
    }
    # One pass, so a placeholder spelled inside the data (e.g. a part name) is never expanded.
    return PLACEHOLDER.sub(lambda match: values[match.group(1)], TEMPLATE_FILE.read_text(encoding="utf-8"))


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ViewerError(f"cannot read {path}: {exc}") from exc


def _is_number(value: Any) -> bool:
    # Finite only: NaN/Infinity would serialise to tokens the browser's JSON.parse rejects.
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_part(part: Any, path: Path) -> None:
    if not isinstance(part, dict) or part.get("group") not in PART_GROUPS:
        raise ViewerError(f"{path}: a part is not an object with a known group")
    if not all(isinstance(part.get(key), str) for key in PART_STRING_KEYS):
        raise ViewerError(f"{path}: part {part.get('name')!r} lacks one of {PART_STRING_KEYS}")
    if part.get("mass_kg") is not None and not _is_number(part["mass_kg"]):
        raise ViewerError(f"{path}: part {part['name']!r} has a non-numeric mass_kg")


def _check_model(model: Any, path: Path) -> dict[str, Any]:
    if not isinstance(model, dict) or not isinstance(model.get("parts"), list):
        raise ViewerError(f"{path}: expected an object with a 'parts' list")
    cg: Any = model.get("cg_m")
    if not (isinstance(cg, list) and len(cg) == CG_AXES and all(_is_number(c) for c in cg)):
        raise ViewerError(f"{path}: 'cg_m' must be three numbers")
    if not _is_number(model.get("mass_kg")):
        raise ViewerError(f"{path}: 'mass_kg' must be a number")
    for part in model["parts"]:
        _check_part(part, path)
    return model


def _check_trace(trace: Any, path: Path) -> list[dict[str, float]]:
    if not isinstance(trace, list):
        raise ViewerError(f"{path}: expected a list of samples")
    for sample in trace:
        if not (isinstance(sample, dict) and all(_is_number(sample.get(key)) for key in TRACE_KEYS)):
            raise ViewerError(f"{path}: every sample needs numeric {TRACE_KEYS}")
    return trace


def load_trace(run_dir: Path) -> tuple[Optional[list[dict[str, float]]], Optional[bool]]:
    """The replayed SITL attempt's truth trace and whether SITL passed; `(None, None)` when there is none.

    `sitl_report.json`'s `replay_attempt` names the attempt (null when no flight completed); the trace is read
    from that attempt's folder only, never by globbing.
    """
    report_path: Path = run_dir / SITL_REPORT_FILE
    if not report_path.is_file():
        return None, None
    report: Any = _read_json(report_path)
    if not isinstance(report, dict):
        raise ViewerError(f"{report_path}: expected an object")
    attempt: Any = report.get("replay_attempt")
    if attempt is None:
        return None, None
    if not isinstance(attempt, int) or isinstance(attempt, bool):
        raise ViewerError(f"{report_path}: 'replay_attempt' must be an integer or null")
    trace_path: Path = run_dir / SITL_DIR / f"{ATTEMPT_DIR_PREFIX}{attempt}" / TRUTH_FILE
    if not trace_path.is_file():
        return None, None
    trace: list[dict[str, float]] = _check_trace(_read_json(trace_path), trace_path)
    if not trace:
        return None, None
    passed: Any = report.get("passed")
    return trace, (passed if isinstance(passed, bool) else None)


def write_viewer(run_dir: Path) -> Optional[Path]:
    """Writes `viewer.html` into `run_dir`; `None` (and no file) when the run has no `viewer_model.json`."""
    model_path: Path = run_dir / VIEWER_MODEL_FILE
    if not model_path.is_file():
        return None
    model: dict[str, Any] = _check_model(_read_json(model_path), model_path)
    trace, sitl_passed = load_trace(run_dir)
    out_path: Path = run_dir / VIEWER_FILE
    try:
        out_path.write_text(build_viewer_html(model, trace, sitl_passed), encoding="utf-8")
    except OSError as exc:
        raise ViewerError(f"cannot write {out_path}: {exc}") from exc
    return out_path
