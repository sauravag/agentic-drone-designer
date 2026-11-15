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
                        ixx_kgm2=1.0, iyy_kgm2=1.0, izz_kgm2=1.0, ixz_kgm2=0.0)


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
