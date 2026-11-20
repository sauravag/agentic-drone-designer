"""Design-pipeline service used by the web form (and later the CLI): sizing -> BOM -> aero/CAD -> files."""

import json
import re
from pathlib import Path

import pytest

from src import pipeline
from src.schemas.requirements import OperationalRequirements
from src.schemas.state import CADArtifacts, FCSArtifacts
from tests.reference_designs import G1_ELECTRIC_FW

RUN_ID_PATTERN = re.compile(r"^\d{8}T\d{6}Z-[a-z0-9-]{1,40}$")


@pytest.fixture(scope="module")
def sizing_only_run(tmp_path_factory: pytest.TempPathFactory) -> pipeline.DesignRun:
    root = tmp_path_factory.mktemp("runs")
    return pipeline.run_design(G1_ELECTRIC_FW, outputs_root=root / "outputs", db_path=root / "db" / "catalog.db",
                               with_aero_cad=False)


def test_sizing_only_run_writes_its_artifacts(sizing_only_run: pipeline.DesignRun) -> None:
    run = sizing_only_run
    assert RUN_ID_PATTERN.match(run.run_id)
    assert run.out_dir.name == run.run_id
    for name in ("requirements.json", "sizing.json", "bom.json"):
        assert (run.out_dir / name).is_file()
    assert run.aero is None and run.cad is None
    bom = json.loads((run.out_dir / "bom.json").read_text())
    assert bom["total_hardware_mass_kg"] == pytest.approx(run.bom.total_hardware_mass_kg)
    assert OperationalRequirements.model_validate_json((run.out_dir / "requirements.json").read_text()) == G1_ELECTRIC_FW


def test_missing_catalog_is_seeded_on_first_run(sizing_only_run: pipeline.DesignRun) -> None:
    assert sizing_only_run.bom.payload.mass_kg >= G1_ELECTRIC_FW.payload_mass_kg


def test_progress_callback_reports_each_stage(tmp_path: Path) -> None:
    stages: list[str] = []
    pipeline.run_design(G1_ELECTRIC_FW, outputs_root=tmp_path / "o", db_path=tmp_path / "c.db",
                        with_aero_cad=False, progress=stages.append)
    assert stages[0].startswith("Sizing") and any(s.startswith("Selecting COTS") for s in stages)


@pytest.mark.parametrize("name", ["../../etc/passwd", "Recon / Mk II!!", "   ", "x" * 300])
def test_run_id_is_filesystem_safe(name: str) -> None:
    run_id = pipeline.new_run_id(name)
    assert RUN_ID_PATTERN.match(run_id)
    assert "/" not in run_id and ".." not in run_id


def test_infeasible_mission_raises_a_design_error_and_marks_the_run_failed(tmp_path: Path) -> None:
    impossible = G1_ELECTRIC_FW.model_copy(update={"endurance_hours": 40.0, "combat_radius_km": 900.0,
                                                   "propulsion_type": "electric"})
    with pytest.raises(pipeline.DESIGN_ERRORS):
        pipeline.run_design(impossible, outputs_root=tmp_path / "o", db_path=tmp_path / "c.db", with_aero_cad=False)
    (summary,) = (tmp_path / "o").glob("*/run_summary.json")
    assert json.loads(summary.read_text())["status"] == "failed"


def test_plain_value_errors_are_not_treated_as_design_errors() -> None:
    assert ValueError not in pipeline.DESIGN_ERRORS


def test_same_second_runs_with_the_same_name_get_distinct_folders(tmp_path: Path) -> None:
    first = pipeline.create_run_dir(tmp_path, "20260101T000000Z-x")
    second = pipeline.create_run_dir(tmp_path, "20260101T000000Z-x")
    assert first[0] != second[0] and second[1].is_dir()


def test_aero_cad_without_openvsp_is_rejected(tmp_path: Path) -> None:
    if pipeline.aero_cad_available():
        pytest.skip("OpenVSP is installed here")
    with pytest.raises(pipeline.PipelineError, match="OpenVSP"):
        pipeline.run_design(G1_ELECTRIC_FW, outputs_root=tmp_path / "o", db_path=tmp_path / "c.db", with_aero_cad=True)


def _cad(as_built: float, sm: float = 0.11, collisions: int = 0) -> CADArtifacts:
    return CADArtifacts(oml_vsp3_path="a", oml_step_path="b", assembly_step_path="c", x_cg_m=0.3,
                        static_margin=sm, collision_count=collisions, as_built_mass_kg=as_built,
                        ixx_kgm2=1.0, iyy_kgm2=1.0, izz_kgm2=1.0, ixz_kgm2=0.0, viewer_model_path="d")


def test_warnings_flag_mass_growth_margin_and_collisions(sizing_only_run: pipeline.DesignRun) -> None:
    sizing = sizing_only_run.sizing
    assert pipeline.design_warnings(sizing, None, _cad(sizing.mtow_kg)) == ()
    text = " ".join(pipeline.design_warnings(sizing, None, _cad(1.27 * sizing.mtow_kg, sm=0.02, collisions=2)))
    assert "27%" in text and "static margin" in text and "2 interference" in text


def test_full_run_with_openvsp_writes_cad_and_flies_sitl(tmp_path: Path) -> None:
    pytest.importorskip("openvsp")
    run = pipeline.run_design(G1_ELECTRIC_FW, outputs_root=tmp_path / "o", db_path=tmp_path / "c.db", with_aero_cad=True)
    assert run.aero is not None and run.cad is not None
    for name in ("aero.json", "cad.json", "oml.step", "oml.vsp3", "assembly.step"):
        assert (run.out_dir / name).is_file()
    if pipeline.sitl_available():
        assert run.fcs is not None and run.fcs.passed
        for name in ("ardupilot.param", "sitl_report.json", "fcs.json"):
            assert (run.out_dir / name).is_file()


def test_sitl_requires_aero_cad(tmp_path: Path) -> None:
    with pytest.raises(pipeline.PipelineError, match="aero/CAD"):
        pipeline.run_design(G1_ELECTRIC_FW, outputs_root=tmp_path / "o", db_path=tmp_path / "c.db",
                            with_aero_cad=False, with_sitl=True)


def test_sitl_without_arduplane_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "aero_cad_available", lambda: True)
    monkeypatch.setattr(pipeline, "sitl_available", lambda: False)
    with pytest.raises(pipeline.PipelineError, match="arduplane"):
        pipeline.run_design(G1_ELECTRIC_FW, outputs_root=tmp_path / "o", db_path=tmp_path / "c.db",
                            with_aero_cad=True, with_sitl=True)
    assert not (tmp_path / "o").exists()  # rejected before a run folder is created


def test_stage_defaults_follow_the_installed_toolchain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "aero_cad_available", lambda: True)
    monkeypatch.setattr(pipeline, "sitl_available", lambda: True)
    assert pipeline.resolve_stages(None, None) == {"aero_cad": True, "sitl": True}
    assert pipeline.resolve_stages(False, None) == {"aero_cad": False, "sitl": False}
    monkeypatch.setattr(pipeline, "sitl_available", lambda: False)
    assert pipeline.resolve_stages(None, None) == {"aero_cad": True, "sitl": False}


def test_payload_warnings_flag_a_hungrier_or_heavier_catalog_sensor(sizing_only_run: pipeline.DesignRun) -> None:
    req, bom = G1_ELECTRIC_FW, sizing_only_run.bom
    assert pipeline.payload_warnings(req, None) == ()
    text = " ".join(pipeline.payload_warnings(req, bom))
    assert bom.payload.power_w > req.payload_power_w  # 1 kg request snaps to a 25 W unit vs 15 W budgeted
    assert f"draws {bom.payload.power_w:.0f} W" in text and "15 W" in text
    exact = req.model_copy(update={"payload_power_w": bom.payload.power_w, "payload_mass_kg": bom.payload.mass_kg})
    assert pipeline.payload_warnings(exact, bom) == ()
    assert any("payload" in w for w in sizing_only_run.warnings)  # surfaced by run_design too


def test_payload_warnings_cover_the_power_and_mass_branches_separately(sizing_only_run: pipeline.DesignRun) -> None:
    bom = sizing_only_run.bom
    fits = G1_ELECTRIC_FW.model_copy(update={"payload_power_w": bom.payload.power_w,
                                             "payload_mass_kg": bom.payload.mass_kg})
    (power_only,) = pipeline.payload_warnings(fits.model_copy(update={"payload_power_w": 1.0}), bom)
    assert "draws" in power_only and "weighs" not in power_only
    (mass_only,) = pipeline.payload_warnings(fits.model_copy(update={"payload_mass_kg": bom.payload.mass_kg / 2}), bom)
    assert "weighs" in mass_only and "heavier" in mass_only


def test_warnings_flag_a_failed_sitl_test(sizing_only_run: pipeline.DesignRun) -> None:
    fcs = FCSArtifacts(param_path="a", sitl_report_path="b", passed=False, roll_overshoot=0.31, pitch_overshoot=0.05,
                       roll_settling_s=4.2, pitch_settling_s=1.0, max_transition_altitude_loss_m=2.0, attempts=4)
    text = " ".join(pipeline.design_warnings(sizing_only_run.sizing, None, None, fcs))
    assert "SITL doublet test failed after 4 attempt(s)" in text and "31%" in text


def test_sitl_failure_warning_names_the_governing_case(sizing_only_run: pipeline.DesignRun) -> None:
    fcs = FCSArtifacts(param_path="a", sitl_report_path="b", passed=False, roll_overshoot=0.31, pitch_overshoot=0.05,
                       roll_settling_s=4.2, pitch_settling_s=1.0, max_transition_altitude_loss_m=2.0, attempts=4,
                       governing_case="hot")
    text = " ".join(pipeline.design_warnings(sizing_only_run.sizing, None, None, fcs))
    assert "worst case: hot" in text


# ------------------------------------------------------------------ 3D viewer ---
VIEWER_MODEL = {"frame": "x_aft_y_right_z_up", "cg_m": [0.5, 0, 0], "mass_kg": 10.0,
                "parts": [{"name": "fuselage", "group": "component", "mass_kg": 1.0,
                           "positions_b64": "", "indices_b64": ""}]}


_CAD_WITH_MODEL = CADArtifacts(oml_vsp3_path="a", oml_step_path="b", assembly_step_path="c", x_cg_m=0.3,
                               static_margin=0.11, collision_count=0, as_built_mass_kg=10.0, ixx_kgm2=1.0,
                               iyy_kgm2=1.0, izz_kgm2=1.0, ixz_kgm2=0.0, viewer_model_path="viewer_model.json")


@pytest.fixture
def fake_aero_cad(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend OpenVSP is installed; the stage drops a viewer model like the real CAD export would."""
    def fake(sizing, bom, req, out_dir, progress, settings):  # type: ignore[no-untyped-def]
        (out_dir / "viewer_model.json").write_text(json.dumps(VIEWER_MODEL))
        return None, _CAD_WITH_MODEL

    monkeypatch.setattr(pipeline, "aero_cad_available", lambda: True)
    monkeypatch.setattr(pipeline, "run_aero_cad", fake)


def test_run_design_builds_the_viewer_when_a_model_was_exported(tmp_path: Path, fake_aero_cad: None) -> None:
    run = pipeline.run_design(G1_ELECTRIC_FW, outputs_root=tmp_path / "o", db_path=tmp_path / "c.db",
                              with_aero_cad=True, with_sitl=False)
    assert (run.out_dir / "viewer.html").is_file()
    assert not any("3D viewer" in w for w in run.warnings)


def test_a_viewer_failure_is_a_run_warning_not_a_failed_run(tmp_path: Path, fake_aero_cad: None,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    from src.viewer.build import ViewerError

    def broken(run_dir: Path) -> Path:
        raise ViewerError("bad model")

    monkeypatch.setattr("src.viewer.build.write_viewer", broken)
    run = pipeline.run_design(G1_ELECTRIC_FW, outputs_root=tmp_path / "o", db_path=tmp_path / "c.db",
                              with_aero_cad=True, with_sitl=False)
    assert "3D viewer not built: bad model" in run.warnings
    summary = json.loads((run.out_dir / "run_summary.json").read_text())
    assert summary["status"] == "ok" and "3D viewer not built: bad model" in summary["warnings"]


def test_a_run_without_a_model_has_no_viewer_and_no_warning(sizing_only_run: pipeline.DesignRun) -> None:
    assert not (sizing_only_run.out_dir / "viewer.html").exists()
    assert not any("3D viewer" in w for w in sizing_only_run.warnings)


def test_a_stale_model_without_cad_artifacts_is_not_built(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def stale(sizing, bom, req, out_dir, progress, settings):  # type: ignore[no-untyped-def]
        (out_dir / "viewer_model.json").write_text(json.dumps(VIEWER_MODEL))
        return None, None  # the stage failed after an earlier pass left a mesh behind

    monkeypatch.setattr(pipeline, "aero_cad_available", lambda: True)
    monkeypatch.setattr(pipeline, "run_aero_cad", stale)
    run = pipeline.run_design(G1_ELECTRIC_FW, outputs_root=tmp_path / "o", db_path=tmp_path / "c.db",
                              with_aero_cad=True, with_sitl=False)
    assert not (run.out_dir / "viewer.html").exists()
    assert not any("3D viewer" in w for w in run.warnings)
