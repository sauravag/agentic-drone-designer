"""design_report.pdf: the Markdown report typeset with reportlab, plus generated figures after their sections."""

import base64
import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("reportlab")

from reportlab.platypus import Image, Paragraph, Table  # noqa: E402

from src.agent import report, report_figures, report_pdf  # noqa: E402
from src.schemas.state import FCSArtifacts  # noqa: E402
from tests.test_flight_control import load_design  # noqa: E402

PNG_MAGIC: bytes = b"\x89PNG"
PDF_IMAGE_MARKER: bytes = b"/SMask "  # once per figure: matplotlib PNGs are RGBA, so each image carries a soft mask


def _box_part(name: str, group: str, x0: float, x1: float, half_span: float, half_height: float) -> dict[str, object]:
    """A closed box as a viewer-model part (positions float32, indices uint32, base64), like mesh_export writes."""
    xs, ys, zs = (x0, x1), (-half_span, half_span), (-half_height, half_height)
    corners = np.array([(x, y, z) for x in xs for y in ys for z in zs], dtype="<f4")
    faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
             (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    return {"name": name, "group": group, "mass_kg": None,
            "positions_b64": base64.b64encode(corners.tobytes()).decode(),
            "indices_b64": base64.b64encode(np.array(faces, dtype="<u4").tobytes()).decode()}


@pytest.fixture()
def viewer_model(tmp_path: Path) -> Path:
    path = tmp_path / "viewer_model.json"
    parts = [_box_part("skin", "skin", 0.0, 1.0, 0.06, 0.06), _box_part("wing", "skin", 0.3, 0.5, 1.2, 0.02),
             _box_part("battery", "component", 0.2, 0.35, 0.04, 0.03)]
    path.write_text(json.dumps({"frame": "x_aft_y_right_z_up", "cg_m": [0.4, 0.0, 0.0], "mass_kg": 3.0,
                                "parts": parts}))
    return path


@pytest.fixture()
def sitl_report(tmp_path: Path) -> Path:
    t = np.arange(0.0, 20.0, 0.04)
    nav_roll = np.where(t > 5.0, 15.0, 0.0)
    nav_pitch = np.where(t > 12.0, 10.0, 0.0)
    telemetry = [{"t_s": float(ti), "roll_deg": float(r * 0.95), "pitch_deg": float(p * 0.9), "nav_roll_deg": float(r),
                  "nav_pitch_deg": float(p), "alt_m": 50.0, "airspeed_ms": 18.0, "throttle_pct": 20.0}
                 for ti, r, p in zip(t, nav_roll, nav_pitch)]
    path = tmp_path / "sitl_report.json"
    path.write_text(json.dumps({"passed": True, "telemetry": telemetry}))
    return path


@pytest.fixture()
def inputs(tmp_path: Path, viewer_model: Path, sitl_report: Path) -> report.ReportInputs:
    req, sizing, bom, aero, cad = load_design("g1")
    fcs = FCSArtifacts(param_path=str(tmp_path / "ardupilot.param"), sitl_report_path=str(sitl_report), passed=True,
                       roll_overshoot=0.05, pitch_overshoot=0.04, roll_settling_s=1.5, pitch_settling_s=1.2,
                       max_transition_altitude_loss_m=0.0)
    return report.ReportInputs(run_id="r1", out_dir=tmp_path, requirements=req, sizing=sizing, bom=bom, aero=aero,
                               cad=cad.model_copy(update={"viewer_model_path": str(viewer_model)}), fcs=fcs,
                               converged=True, iterations=1)


def _write_markdown(inputs: report.ReportInputs) -> Path:
    return report.write_report(inputs, report.fallback_narrative(inputs, "no LLM configured"))


# --------------------------------------------------------------------- markup ---


def test_markup_undoes_markdown_escapes_and_escapes_xml() -> None:
    text = r"a \[b\] &lt;script> & `x.step` **CONVERGED** c\|d"
    assert report_pdf.markup(text) == (f"a [b] &lt;script&gt; &amp; <font face=\"{report_pdf.MONO_FONT}\">x.step</font> "
                                       "<b>CONVERGED</b> c|d")


def test_table_lines_become_one_table_with_a_header_row() -> None:
    (flowable,) = report_pdf.markdown_flowables("| Quantity | Value |\n|---|---|\n| Span | a\\|b |\n", {})
    assert isinstance(flowable, Table)
    cells = [[cell.text for cell in row] for row in flowable._cellvalues]
    assert cells == [["<b>Quantity</b>", "<b>Value</b>"], ["Span", "a|b"]]


def test_figures_close_the_section_they_belong_to() -> None:
    png = report_figures.power_profile(load_design("g1")[1])
    flowables = report_pdf.markdown_flowables("## Alpha\n\nfirst\n\n## Beta\n\nsecond\n", {"Alpha": [png]})
    kinds = [type(f).__name__ for f in flowables]
    assert kinds == ["Paragraph", "Paragraph", "Image", "Paragraph", "Paragraph"]
    assert isinstance(flowables[3], Paragraph) and flowables[3].text == "Beta"


def test_a_figure_for_the_last_section_still_appears() -> None:
    png = report_figures.power_profile(load_design("g1")[1])
    flowables = report_pdf.markdown_flowables("## Only\n\ntext\n", {"Only": [png]})
    assert isinstance(flowables[-1], Image)


# -------------------------------------------------------------------- figures ---


def test_every_figure_renders_a_png(viewer_model: Path, sitl_report: Path) -> None:
    _, sizing, _, aero, _ = load_design("g1")
    for png in (report_figures.three_view(viewer_model), report_figures.power_profile(sizing),
                report_figures.drag_polar(aero, sizing), report_figures.step_responses(sitl_report)):
        assert png is not None and png.startswith(PNG_MAGIC)


def test_figures_without_their_input_are_skipped(tmp_path: Path) -> None:
    empty_flight = tmp_path / "sitl_report.json"
    empty_flight.write_text(json.dumps({"passed": False, "telemetry": []}))  # every attempt crashed before flying
    assert report_figures.three_view(None) is None
    assert report_figures.three_view(tmp_path / "missing.json") is None
    assert report_figures.step_responses(empty_flight) is None


def test_the_figure_set_follows_what_the_run_produced(inputs: report.ReportInputs) -> None:
    assert set(report_pdf.section_figures(inputs)) == {report.SECTION_SIZING, report.SECTION_ENERGY,
                                                       report.SECTION_POLAR, report.SECTION_SITL}
    sizing_only = replace(inputs, aero=None, cad=None, fcs=None)
    assert set(report_pdf.section_figures(sizing_only)) == {report.SECTION_ENERGY}


# ------------------------------------------------------------------- the file ---


def test_the_pdf_holds_the_report_and_all_four_figures(inputs: report.ReportInputs) -> None:
    path = report_pdf.write_pdf_report(inputs, _write_markdown(inputs))
    assert path == inputs.out_dir / report_pdf.PDF_FILE
    data = path.read_bytes()
    assert data.startswith(b"%PDF") and data.count(PDF_IMAGE_MARKER) == 4


def test_a_failing_figure_is_left_out_with_a_warning(inputs: report.ReportInputs, monkeypatch: pytest.MonkeyPatch,
                                                     caplog: pytest.LogCaptureFixture) -> None:
    def broken(*_: object) -> bytes:
        raise ValueError("bad mesh")

    monkeypatch.setattr(report_figures, "three_view", broken)
    with caplog.at_level(logging.WARNING):
        path = report_pdf.write_pdf_report(inputs, _write_markdown(inputs))
    assert path is not None and path.read_bytes().count(PDF_IMAGE_MARKER) == 3
    assert "three-view" in caplog.text and "bad mesh" in caplog.text


def test_a_failed_pdf_leaves_the_markdown_report_and_says_why(inputs: report.ReportInputs,
                                                              monkeypatch: pytest.MonkeyPatch,
                                                              caplog: pytest.LogCaptureFixture) -> None:
    markdown = _write_markdown(inputs)

    def broken_build(*_: object, **__: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(report_pdf.SimpleDocTemplate, "build", broken_build)
    with caplog.at_level(logging.ERROR):
        assert report_pdf.write_pdf_report(inputs, markdown) is None
    assert markdown.is_file() and "disk full" in caplog.text
