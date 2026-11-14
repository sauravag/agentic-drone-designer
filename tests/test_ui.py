"""Local web form: form parsing, result presenters, and a headless Streamlit smoke test."""

from pathlib import Path

import pytest

from src import pipeline
from src.ui import presenters
from src.ui.examples import EXAMPLE_MISSIONS
from tests.reference_designs import G1_ELECTRIC_FW

APP_PATH = Path(__file__).resolve().parents[1] / "src" / "ui" / "app.py"


def test_examples_are_valid_requirements() -> None:
    assert "G1 electric fixed-wing (reference)" in EXAMPLE_MISSIONS
    assert all(req.mission_name for req in EXAMPLE_MISSIONS.values())


def _schema_limits(field: str) -> tuple[float, float]:
    from src.schemas.requirements import OperationalRequirements

    lo, hi = float("-inf"), float("inf")
    for rule in OperationalRequirements.model_fields[field].metadata:
        for attr in ("gt", "ge"):
            lo = getattr(rule, attr, None) if getattr(rule, attr, None) is not None else lo
        for attr in ("lt", "le"):
            hi = getattr(rule, attr, None) if getattr(rule, attr, None) is not None else hi
    return lo, hi


def test_form_bounds_sit_inside_the_schema_and_fit_every_preset() -> None:
    for fields in presenters.NUMBER_FIELDS.values():
        for name, lo, hi, _, _ in fields:
            s_lo, s_hi = _schema_limits(name)
            assert s_lo <= lo < hi <= s_hi, name
            for preset in EXAMPLE_MISSIONS.values():
                assert lo <= getattr(preset, name) <= hi, (name, preset.mission_name)


def test_parse_form_accepts_valid_values() -> None:
    req, errors = presenters.parse_form(G1_ELECTRIC_FW.model_dump())
    assert errors == [] and req == G1_ELECTRIC_FW


def test_parse_form_maps_auto_group_to_none() -> None:
    req, _ = presenters.parse_form({**G1_ELECTRIC_FW.model_dump(), "group_constraint": "auto"})
    assert req is not None and req.group_constraint is None


def test_parse_form_reports_friendly_field_errors() -> None:
    req, errors = presenters.parse_form({**G1_ELECTRIC_FW.model_dump(), "payload_mass_kg": -1.0, "mission_name": ""})
    assert req is None
    assert any(e.startswith("Payload mass") for e in errors)
    assert any(e.startswith("Mission name") for e in errors)


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> pipeline.DesignRun:
    root = tmp_path_factory.mktemp("ui")
    return pipeline.run_design(G1_ELECTRIC_FW, outputs_root=root / "o", db_path=root / "c.db", with_aero_cad=False)


def test_sizing_summary_has_headline_numbers(run: pipeline.DesignRun) -> None:
    summary = dict(presenters.sizing_summary(run.sizing))
    assert summary["MTOW"] == f"{run.sizing.mtow_kg:.2f} kg"
    assert summary["DoD group"] == "Group 1"
    assert "Wingspan" in summary and "Cruise L/D" in summary


def test_phase_rows_cover_the_mission(run: pipeline.DesignRun) -> None:
    rows = presenters.phase_rows(run.sizing)
    assert [r["Phase"] for r in rows] == [p.name for p in run.sizing.phases]


def test_bom_rows_list_parts_and_total(run: pipeline.DesignRun) -> None:
    rows = presenters.bom_rows(run.bom)
    parts = [r["Part"] for r in rows]
    assert {"Cruise motor", "Battery", "Payload"} <= set(parts)
    assert rows[-1]["Part"] == "Total" and rows[-1]["Mass (kg)"] == pytest.approx(run.bom.total_hardware_mass_kg)


def test_downloadable_files_stay_inside_the_run_folder(run: pipeline.DesignRun) -> None:
    files = presenters.downloadable_files(run.out_dir)
    assert {f.name for f in files} >= {"requirements.json", "sizing.json", "bom.json"}
    assert all(f.resolve().parent == run.out_dir.resolve() for f in files)


def test_app_runs_a_sizing_only_design_headlessly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("ISR_FORGE_OUTPUTS", str(tmp_path / "outputs"))
    monkeypatch.setenv("ISR_FORGE_DB", str(tmp_path / "catalog.db"))
    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    assert not at.exception
    aero_box = at.checkbox(key="with_aero_cad")
    if not aero_box.disabled:  # enabled only where OpenVSP exists; keep this smoke test fast
        aero_box.uncheck()
    assert aero_box.disabled == (not pipeline.aero_cad_available())
    at.button(key="submit").click()
    at.run()
    assert not at.exception
    assert any(m.label == "MTOW" for m in at.metric)
    assert any((tmp_path / "outputs").glob("*/bom.json"))


def test_app_shows_validation_errors_instead_of_running(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("ISR_FORGE_OUTPUTS", str(tmp_path / "outputs"))
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    at.text_input(key="mission_name").set_value("")
    at.button(key="submit").click()
    at.run()
    assert not at.exception
    assert any("Mission name" in e.value for e in at.error)
    assert not (tmp_path / "outputs").exists()
