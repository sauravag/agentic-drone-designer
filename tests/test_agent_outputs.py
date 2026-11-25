"""Agent run outputs (split from test_agent.py): the 3D viewer and the fabrication settings a run writes and threads
through the CAD stage and the CLI. Host only: scripted LLM, aero/CAD off."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from src import pipeline
from src.agent import nodes, report
from src.agent.graph import run_agent
from src.main import EXIT_CONVERGED, EXIT_ERROR, EXIT_NOT_CONVERGED, main
from src.schemas.fabrication import DEFAULT_FABRICATION, FabricationSettings
from src.schemas.state import CADArtifacts
from src.viewer.build import ViewerError
from tests.agent_fakes import ScriptedLLM, agent_context, seeded_catalog
from tests.reference_designs import G1_ELECTRIC_FW

@pytest.fixture(scope="module")
def db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return seeded_catalog(tmp_path_factory.mktemp("db"))


_ctx = agent_context


# ------------------------------------------------------------------ 3D viewer ---
VIEWER_MODEL: dict[str, Any] = {"frame": "x_aft_y_right_z_up", "cg_m": [0.5, 0, 0], "mass_kg": 10.0,
                                "parts": [{"name": "fuselage", "group": "component", "mass_kg": 1.0,
                                           "positions_b64": "", "indices_b64": ""}]}


def _cad_with_model(model_path: Optional[str]) -> CADArtifacts:
    return CADArtifacts(oml_vsp3_path="a.vsp3", oml_step_path="b.step", assembly_step_path="c.step", x_cg_m=0.3,
                        static_margin=0.11, collision_count=0, as_built_mass_kg=10.0, ixx_kgm2=1.0, iyy_kgm2=1.0,
                        izz_kgm2=1.0, ixz_kgm2=0.0, viewer_model_path=model_path)


def test_compile_report_builds_the_viewer_from_the_exported_model(tmp_path: Path, db: Path) -> None:
    ctx = _ctx(tmp_path, db, ScriptedLLM())
    state = run_agent(ctx, requirements=G1_ELECTRIC_FW)
    out = Path(state["out_dir"])
    assert not (out / "viewer.html").exists()  # the aero/CAD-free run exported no mesh
    (out / "viewer_model.json").write_text(json.dumps(VIEWER_MODEL))
    update = nodes.compile_report(
        {**state, "cad_artifacts": _cad_with_model(str(out / "viewer_model.json"))}, ctx)
    assert (out / "viewer.html").is_file() and update["viewer_error"] is None


@pytest.mark.parametrize("cad", [None, _cad_with_model(None)], ids=["no_cad", "no_mesh_path"])
def test_a_stale_viewer_model_never_becomes_the_viewer(tmp_path: Path, db: Path, cad: Optional[CADArtifacts]) -> None:
    ctx = _ctx(tmp_path, db, ScriptedLLM())
    state = run_agent(ctx, requirements=G1_ELECTRIC_FW)
    out = Path(state["out_dir"])
    (out / "viewer.html").unlink(missing_ok=True)
    (out / "viewer_model.json").write_text(json.dumps(VIEWER_MODEL))  # left behind by an earlier iteration
    update = nodes.compile_report({**state, "cad_artifacts": cad}, ctx)
    assert not (out / "viewer.html").exists()
    assert update["viewer_error"] is None
    assert not any("3D viewer not built" in w for w in nodes.run_warnings({**state, **update}))


def test_a_viewer_failure_is_a_warning_not_a_failed_run(tmp_path: Path, db: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(run_dir: Path) -> Path:
        raise ViewerError("bad model")

    monkeypatch.setattr("src.viewer.build.write_viewer", broken)
    ctx = _ctx(tmp_path, db, ScriptedLLM())
    state = run_agent(ctx, requirements=G1_ELECTRIC_FW)
    update = nodes.compile_report({**state, "cad_artifacts": _cad_with_model("viewer_model.json")}, ctx)
    assert Path(update["final_report_path"]).is_file()
    assert update["viewer_error"] == "3D viewer not built: bad model"
    assert "bad model" in Path(update["final_report_path"]).read_text()


def test_the_report_lists_the_viewer_file_only_when_it_was_built(tmp_path: Path) -> None:
    cad = CADArtifacts(oml_vsp3_path="a.vsp3", oml_step_path="b.step", assembly_step_path="c.step", x_cg_m=0.3,
                       static_margin=0.11, collision_count=0, as_built_mass_kg=10.0, ixx_kgm2=1.0, iyy_kgm2=1.0,
                       izz_kgm2=1.0, ixz_kgm2=0.0, viewer_model_path="viewer_model.json")
    inputs = report.ReportInputs(run_id="r", out_dir=tmp_path, requirements=G1_ELECTRIC_FW, sizing=None,
                                 bom=None, aero=None, cad=cad, fcs=None, converged=True, iterations=0)
    assert "viewer.html" not in "\n".join(report._sections(inputs))  # mesh exported, but the build failed
    no_mesh = report.ReportInputs(run_id="r", out_dir=tmp_path, requirements=G1_ELECTRIC_FW, sizing=None, bom=None,
                                  aero=None, cad=_cad_with_model(None), fcs=None, converged=True, iterations=0)
    assert "viewer.html" not in "\n".join(report._sections(no_mesh))  # no mesh exported: nothing to list
    (tmp_path / "viewer.html").write_text("<html></html>")
    assert "`viewer.html` (3D viewer)" in "\n".join(report._sections(inputs))


# ------------------------------------------------------------ fabrication ---
CUSTOM_FABRICATION = FabricationSettings(bed_width_m=0.5, bed_height_m=0.3)


def test_fabrication_settings_are_written_to_the_run_folder(tmp_path: Path, db: Path) -> None:
    state = run_agent(_ctx(tmp_path, db, None), requirements=G1_ELECTRIC_FW, fabrication=CUSTOM_FABRICATION)
    written = json.loads((Path(state["out_dir"]) / "fabrication.json").read_text())
    assert FabricationSettings.model_validate(written) == CUSTOM_FABRICATION
    assert state["fabrication"] == CUSTOM_FABRICATION


def test_a_run_without_settings_uses_the_default(tmp_path: Path, db: Path) -> None:
    state = run_agent(_ctx(tmp_path, db, None), requirements=G1_ELECTRIC_FW)
    assert state["fabrication"] == DEFAULT_FABRICATION
    written = json.loads((Path(state["out_dir"]) / "fabrication.json").read_text())
    assert FabricationSettings.model_validate(written) == DEFAULT_FABRICATION


def test_the_cad_stage_receives_the_run_settings(tmp_path: Path, db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = run_agent(_ctx(tmp_path, db, None), requirements=G1_ELECTRIC_FW, fabrication=CUSTOM_FABRICATION)
    seen: list[FabricationSettings] = []

    def fake(sizing: Any, bom: Any, req: Any, out_dir: Path, progress: Any,
             settings: FabricationSettings = DEFAULT_FABRICATION) -> Any:
        seen.append(settings)
        raise pipeline.PipelineError("stop here")

    monkeypatch.setattr("src.agent.nodes.pipeline.run_aero_cad", fake)
    monkeypatch.setattr("src.agent.nodes.pipeline.DESIGN_ERRORS", (pipeline.PipelineError,))
    nodes.build_aero_cad(state, replace(_ctx(tmp_path, db, None), with_aero_cad=True))
    assert seen == [CUSTOM_FABRICATION]


def _kit_cad(warning: Optional[str]) -> CADArtifacts:
    return _cad_with_model(None).model_copy(update={"kit_warning": warning})


def test_a_kit_warning_is_a_design_warning_exactly_once(tmp_path: Path, db: Path) -> None:
    state = run_agent(_ctx(tmp_path, db, None), requirements=G1_ELECTRIC_FW)
    with_kit_warning = {**state, "cad_artifacts": _kit_cad("no kit: shell model used")}
    hits = [w for w in nodes.run_warnings(with_kit_warning) if "no kit: shell model used" in w]
    assert len(hits) == 1
    assert not any("kit" in w.lower() for w in nodes.run_warnings({**state, "cad_artifacts": _kit_cad(None)}))


def test_run_aero_cad_hands_the_settings_and_design_day_to_build_aero_cad(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Any] = []
    monkeypatch.setitem(__import__("sys").modules, "src.tools.aero_cad.build", SimpleNamespace(
        build_aero_cad=lambda *args: seen.append(args[-2:]) or ("aero", "cad")))
    req = SimpleNamespace(service_ceiling_m=1.0, isa_temperature_offset_c=7.0)
    out = pipeline.run_aero_cad("sizing", "bom", req, Path("."), lambda _: None, CUSTOM_FABRICATION)
    assert out == ("aero", "cad") and seen == [(CUSTOM_FABRICATION, 7.0)]


def test_cli_fabrication_file_reaches_the_run(tmp_path: Path, db: Path) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text(G1_ELECTRIC_FW.model_dump_json())
    fab = tmp_path / "fab.json"
    fab.write_text(CUSTOM_FABRICATION.model_dump_json())
    args = [f"--spec={spec}", f"--outputs={tmp_path / 'o'}", f"--db={db}", "--no-aero", "--quiet"]
    assert main([*args, f"--fabrication={fab}"]) in (EXIT_CONVERGED, EXIT_NOT_CONVERGED)
    (written,) = (tmp_path / "o").glob("*/fabrication.json")
    assert FabricationSettings.model_validate_json(written.read_text()) == CUSTOM_FABRICATION


@pytest.mark.parametrize("content", [None, "not json", '{"bed_width_m": -1}', '{"bogus": 1}'],
                         ids=["missing", "not_json", "out_of_bounds", "unknown_field"])
def test_cli_rejects_a_bad_fabrication_file(tmp_path: Path, db: Path, content: Optional[str],
                                            capsys: pytest.CaptureFixture[str]) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text(G1_ELECTRIC_FW.model_dump_json())
    fab = tmp_path / "fab.json"
    if content is not None:
        fab.write_text(content)
    code = main([f"--spec={spec}", f"--outputs={tmp_path / 'o'}", f"--db={db}", "--no-aero", "--quiet",
                 f"--fabrication={fab}"])
    assert code == EXIT_ERROR and "fabrication file" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


def test_cli_rejects_a_non_utf8_fabrication_file_naming_the_path(tmp_path: Path, db: Path,
                                                                 capsys: pytest.CaptureFixture[str]) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text(G1_ELECTRIC_FW.model_dump_json())
    fab = tmp_path / "fab.json"
    fab.write_bytes(b'{"bed_width_m": \xff\xfe}')
    code = main([f"--spec={spec}", f"--outputs={tmp_path / 'o'}", f"--db={db}", "--no-aero", "--quiet",
                 f"--fabrication={fab}"])
    err = capsys.readouterr().err
    assert code == EXIT_ERROR and "fabrication file" in err and str(fab) in err
    assert not (tmp_path / "o").exists()


def test_run_design_writes_and_threads_the_settings(tmp_path: Path, db: Path) -> None:
    run = pipeline.run_design(G1_ELECTRIC_FW, tmp_path / "o", db, with_aero_cad=False, with_sitl=False,
                              settings=CUSTOM_FABRICATION)
    written = (run.out_dir / "fabrication.json").read_text()
    assert FabricationSettings.model_validate_json(written) == CUSTOM_FABRICATION
