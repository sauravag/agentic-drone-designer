"""Design export: the versioned design.json record and the run-folder zip (spec 2026-11-19-export-design-zip)."""

import io
import shutil
import zipfile
from pathlib import Path
from typing import Any

import pytest

from src.schemas.assumptions import DEFAULT_ASSUMPTIONS
from src.schemas.changes import AssumptionChanges, ChangeSet, KnobChanges, RequirementChanges
from src.schemas.state import CADArtifacts, FCSArtifacts, RunLineage
from src.tools.sizing.mission_solver import size_aircraft
from src.ui.agent_service import AgentRunView
from src.ui.design_archive import DesignRecord, build_zip, design_record, zip_name
from tests.reference_designs import G1_ELECTRIC_FW


def _cad(run_dir: Path) -> CADArtifacts:
    return CADArtifacts(oml_vsp3_path=str(run_dir / "oml.vsp3"), oml_step_path=str(run_dir / "oml.step"),
                        assembly_step_path=str(run_dir / "assembly.step"), x_cg_m=0.4, static_margin=0.1,
                        collision_count=0, as_built_mass_kg=5.0, ixx_kgm2=1.0, iyy_kgm2=1.0, izz_kgm2=1.5,
                        ixz_kgm2=0.0, viewer_model_path=str(run_dir / "viewer_model.json"))


def _fcs(run_dir: Path) -> FCSArtifacts:
    return FCSArtifacts(param_path=str(run_dir / "ardupilot.param"),
                        sitl_report_path=str(run_dir / "sitl" / "sitl_report.json"), passed=True,
                        roll_overshoot=0.1, pitch_overshoot=0.1, roll_settling_s=1.0, pitch_settling_s=1.0,
                        max_transition_altitude_loss_m=0.0)


def _view(run_dir: Path, **fields: Any) -> AgentRunView:
    base: dict[str, Any] = {"status": "converged", "run_id": "r1", "out_dir": run_dir,
                            "requirements": G1_ELECTRIC_FW, "cad": _cad(run_dir), "fcs": _fcs(run_dir),
                            "report_path": run_dir / "design_report.md"}
    return AgentRunView(**{**base, **fields})


def test_record_round_trips_and_keeps_requirements_exact(tmp_path: Path) -> None:
    assumptions = DEFAULT_ASSUMPTIONS.model_copy(update={"battery_reserve_fraction": 0.25})
    change = ChangeSet(label="longer loiter", requirements=RequirementChanges(endurance_hours=3.0),
                       assumptions=AssumptionChanges(), knobs=KnobChanges())
    view = _view(tmp_path, sizing=size_aircraft(G1_ELECTRIC_FW), design_assumptions=assumptions,
                 gates=({"name": "closure", "passed": True, "detail": "ok"},),
                 critic_log=({"iteration": 1, "failed": [], "changes": []},),
                 lineage=RunLineage(parent_run_id="r0", change_set=change, change_source="manual"))
    rec = design_record(view)
    assert DesignRecord.model_validate_json(rec.model_dump_json()) == rec
    assert rec.requirements == view.requirements
    assert rec.design_assumptions == assumptions
    assert (rec.format, rec.format_version) == ("isr-forge-design", 1)
    assert "out_dir" not in rec.model_dump()


def test_record_paths_are_relative_to_the_run_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rec = design_record(_view(tmp_path))
    assert rec.cad is not None and rec.fcs is not None
    assert (rec.cad.oml_step_path, rec.cad.viewer_model_path) == ("oml.step", "viewer_model.json")
    assert rec.fcs.sitl_report_path == "sitl/sitl_report.json"
    assert rec.report_path == "design_report.md"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    relative = _view(Path("run"), cad=_cad(Path("run")), fcs=_fcs(Path("run")))
    assert design_record(relative).cad.oml_step_path == "oml.step"  # type: ignore[union-attr]


def test_record_rejects_a_path_outside_the_run_folder(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    cad = _cad(run_dir).model_copy(update={"oml_step_path": str(tmp_path / "elsewhere" / "oml.step")})
    with pytest.raises(ValueError, match="outside the run folder"):
        design_record(_view(run_dir, cad=cad))


def test_record_without_artifacts_builds(tmp_path: Path) -> None:
    rec = design_record(_view(tmp_path, cad=None, fcs=None, report_path=None, sizing=None, status="not_converged"))
    assert (rec.cad, rec.fcs, rec.report_path, rec.status) == (None, None, None, "not_converged")


def test_record_needs_a_run_folder(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run folder"):
        design_record(_view(tmp_path, out_dir=None))


def _zip(view: AgentRunView) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(build_zip(view)))


def _run_folder(tmp_path: Path) -> Path:
    run_dir = tmp_path / "r1"
    (run_dir / "sitl").mkdir(parents=True)
    (run_dir / "vspaero" / "a").mkdir(parents=True)
    (run_dir / "oml.step").write_bytes(b"step")
    (run_dir / "sitl" / "log.bin").write_bytes(b"\x00\x01")
    (run_dir / "vspaero" / "a" / "b.txt").write_text("polar")
    return run_dir


def test_zip_holds_every_file_under_the_run_id_folder(tmp_path: Path) -> None:
    view = _view(_run_folder(tmp_path))
    archive = _zip(view)
    assert sorted(archive.namelist()) == ["r1/design.json", "r1/oml.step", "r1/sitl/log.bin", "r1/vspaero/a/b.txt"]
    assert DesignRecord.model_validate_json(archive.read("r1/design.json")).run_id == "r1"
    assert archive.read("r1/sitl/log.bin") == b"\x00\x01"
    assert zip_name(view) == "r1.zip"


def test_zip_skips_symlinks_and_outside_files(tmp_path: Path) -> None:
    run_dir = _run_folder(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    (run_dir / "linked.txt").symlink_to(outside / "secret.txt")
    (run_dir / "linked_dir").symlink_to(outside, target_is_directory=True)
    names = _zip(_view(run_dir)).namelist()
    assert not any("linked" in name or "secret" in name for name in names)


def test_zip_replaces_a_stray_design_json(tmp_path: Path) -> None:
    run_dir = _run_folder(tmp_path)
    (run_dir / "design.json").write_text("stale")
    archive = _zip(_view(run_dir))
    assert archive.namelist().count("r1/design.json") == 1
    assert DesignRecord.model_validate_json(archive.read("r1/design.json")).run_id == "r1"


def test_zip_needs_an_existing_folder(tmp_path: Path) -> None:
    run_dir = _run_folder(tmp_path)
    shutil.rmtree(run_dir)
    with pytest.raises(FileNotFoundError):
        build_zip(_view(run_dir))


def test_design_json_is_not_written_to_the_run_folder(tmp_path: Path) -> None:
    run_dir = _run_folder(tmp_path)
    build_zip(_view(run_dir))
    assert not (run_dir / "design.json").exists()


ZIP_BUTTON = "Download all files (.zip)"


def _app_with(view: AgentRunView, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from src.ui import agent_service
    from tests.test_ui import _llm_app

    at: Any = _llm_app(tmp_path, monkeypatch)
    monkeypatch.setattr(agent_service, "run", lambda *a, **k: view)
    at.button(key="submit_prompt").click()
    at.run()
    assert not at.exception
    return at


def _zip_buttons(at: Any) -> list[Any]:
    return [b for b in at.download_button if b.label == ZIP_BUTTON]


def _failed_view(run_dir: Path, **fields: Any) -> AgentRunView:
    return _view(run_dir, status="not_converged", cad=None, fcs=None, report_path=None,
                 warnings=("NON-CONVERGENCE: sizing: mass spiral",), **fields)


def test_files_tab_offers_the_zip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app_with(_failed_view(_run_folder(tmp_path)), tmp_path, monkeypatch)
    assert len(_zip_buttons(at)) == 1


def test_files_tab_warns_when_the_run_cannot_be_zipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    view = _view(_run_folder(tmp_path), status="not_converged", cad=None, fcs=None,
                 report_path=tmp_path / "elsewhere.md", warnings=("NON-CONVERGENCE: sizing: mass spiral",))
    at = _app_with(view, tmp_path, monkeypatch)
    assert any("cannot be zipped" in w.value for w in at.warning)
    assert not _zip_buttons(at)


def test_files_tab_has_no_zip_for_a_deleted_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app_with(_failed_view(tmp_path / "gone"), tmp_path, monkeypatch)
    assert not _zip_buttons(at)
