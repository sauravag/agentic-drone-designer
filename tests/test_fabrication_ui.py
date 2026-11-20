"""Laser-kit UI: the settings panel, the kit summary in the results, and the kit files in the Files tab."""

from pathlib import Path
from typing import Any, Optional

import pytest

from src import pipeline
from src.schemas.fabrication import DEFAULT_FABRICATION, FabricationSettings
from src.schemas.state import AeroStabilityOutput, CADArtifacts
from src.ui import agent_service, fabrication_panel, presenters
from tests.reference_designs import G1_ELECTRIC_FW

APP_PATH = Path(__file__).resolve().parents[1] / "src" / "ui" / "app.py"
NON_SCALAR_FIELDS = ("sheets", "tubes", "rods_m")
CUSTOM_BED_MM = 800.0


def _app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(agent_service, "llm_problem", lambda: None)
    monkeypatch.setenv("ISR_FORGE_OUTPUTS", str(tmp_path / "outputs"))
    monkeypatch.setenv("ISR_FORGE_DB", str(tmp_path / "catalog.db"))
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    return at


def _record_runs(monkeypatch: pytest.MonkeyPatch, views: Optional[list[Any]] = None) -> list[dict[str, Any]]:
    """Replace agent_service.run: record its keyword arguments, return the next of `views` (or a stop view)."""
    calls: list[dict[str, Any]] = []
    queue = list(views or [])

    def fake_run(*args: object, **kwargs: object) -> Any:
        calls.append(dict(kwargs))
        return queue.pop(0) if queue else agent_service.AgentRunView("error", message="stop")

    monkeypatch.setattr(agent_service, "run", fake_run)
    return calls


def _converged(run_id: str, fabrication: FabricationSettings, **extra: Any) -> Any:
    return agent_service.AgentRunView("converged", run_id=run_id, requirements=G1_ELECTRIC_FW,
                                      fabrication=fabrication, **extra)


def _rebuild(values: Optional[dict[str, float]] = None, sheets: Optional[list[dict[str, Any]]] = None,
             tubes: Optional[list[dict[str, Any]]] = None,
             rods: Optional[str] = None) -> tuple[Optional[FabricationSettings], list[str]]:
    """build_settings from the default panel values, with any part replaced."""
    d = DEFAULT_FABRICATION
    return fabrication_panel.build_settings(
        {**fabrication_panel.display_values(d), **(values or {})},
        fabrication_panel.sheet_rows(d) if sheets is None else sheets,
        fabrication_panel.tube_rows(d) if tubes is None else tubes,
        fabrication_panel.rods_text(d) if rods is None else rods)


# ---------------------------------------------------------------- panel ---
def test_kit_settings_expander_sits_under_design_assumptions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(tmp_path, monkeypatch)
    labels = [e.label for e in at.expander]
    assert "Laser kit settings" in labels
    assert labels.index("Laser kit settings") == labels.index("Design assumptions") + 1


def test_every_scalar_setting_has_an_input() -> None:
    scalars = [name for name in FabricationSettings.model_fields if name not in NON_SCALAR_FIELDS]
    assert sorted(scalars) == sorted(fabrication_panel.SCALAR_FIELDS)


def test_every_scalar_setting_is_drawn_and_the_defaults_round_trip(tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(tmp_path, monkeypatch)
    for name in fabrication_panel.SCALAR_FIELDS:
        assert at.number_input(key=f"fab_{name}") is not None
    calls = _record_runs(monkeypatch)
    at.button(key="submit_prompt").click()
    at.run()
    assert not at.exception
    assert calls[0]["fabrication"] == DEFAULT_FABRICATION


def test_values_are_shown_in_millimetres_and_megapascals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(tmp_path, monkeypatch)
    assert at.number_input(key="fab_bed_width_m").value == pytest.approx(600.0)
    assert at.number_input(key="fab_kerf_m").value == pytest.approx(0.15)
    assert at.number_input(key="fab_carbon_modulus_pa").value == pytest.approx(70_000.0)


def test_custom_bed_reaches_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(tmp_path, monkeypatch)
    calls = _record_runs(monkeypatch)
    at.number_input(key="fab_bed_width_m").set_value(CUSTOM_BED_MM)
    at.button(key="submit_prompt").click()
    at.run()
    assert not at.exception
    assert calls[0]["fabrication"].bed_width_m == 0.8


def test_an_advanced_allowable_reaches_the_structured_form_run(tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(tmp_path, monkeypatch)
    calls = _record_runs(monkeypatch)
    at.number_input(key="fab_carbon_modulus_pa").set_value(90_000.0)
    at.button(key="submit").click()
    at.run()
    assert not at.exception
    assert calls[0]["fabrication"].carbon_modulus_pa == pytest.approx(90e9)


def test_invalid_bed_blocks_the_run_and_names_the_field(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(tmp_path, monkeypatch)
    calls = _record_runs(monkeypatch)
    at.number_input(key="fab_bed_width_m").set_value(0.0)
    at.button(key="submit_prompt").click()
    at.run()
    assert not at.exception
    assert calls == []
    assert any("Laser kit" in e.value and "Bed width" in e.value for e in at.error)


def test_invalid_cloth_layer_blocks_the_structured_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(tmp_path, monkeypatch)
    calls = _record_runs(monkeypatch)
    at.number_input(key="fab_overwrap_layer_thickness_m").set_value(0.1)  # mm: a layer too thin to hold the 200 g/m^2 cloth
    at.button(key="submit").click()
    at.run()
    assert not at.exception
    assert calls == []
    assert any("Laser kit" in e.value and "overwrap_" not in e.value for e in at.error)


def test_reset_restores_the_default_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(tmp_path, monkeypatch)
    at.number_input(key="fab_bed_width_m").set_value(CUSTOM_BED_MM)
    at.run()
    at.button(key="reset_fabrication").click()
    at.run()
    assert at.number_input(key="fab_bed_width_m").value == pytest.approx(600.0)


def test_an_opened_design_shows_its_own_settings_in_the_panel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    custom = DEFAULT_FABRICATION.model_copy(update={"bed_width_m": 0.8})
    at = _app(tmp_path, monkeypatch)
    _record_runs(monkeypatch, [_converged("first", custom), _converged("second", DEFAULT_FABRICATION)])
    for _ in range(2):
        at.button(key="submit_prompt").click()
        at.run()
    assert at.number_input(key="fab_bed_width_m").value == pytest.approx(600.0)
    at.button(key="open_run_first").click()
    at.run()
    assert not at.exception
    assert at.number_input(key="fab_bed_width_m").value == pytest.approx(CUSTOM_BED_MM)


def test_a_new_design_with_other_settings_loads_them_into_the_panel(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """A re-run keeps its parent's settings, whatever the panel holds now; the panel follows the new design."""
    custom = DEFAULT_FABRICATION.model_copy(update={"kerf_m": 0.0003})
    at = _app(tmp_path, monkeypatch)
    _record_runs(monkeypatch, [_converged("rerun", custom)])
    at.button(key="submit_prompt").click()
    at.run()
    at.run()
    assert at.number_input(key="fab_kerf_m").value == pytest.approx(0.3)


def test_a_design_with_other_sheets_fills_the_tables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sheet = DEFAULT_FABRICATION.sheets[0].model_copy(update={"thickness_m": 0.004})
    custom = DEFAULT_FABRICATION.model_copy(update={"sheets": (sheet, DEFAULT_FABRICATION.sheets[1])})
    at = _app(tmp_path, monkeypatch)
    _record_runs(monkeypatch, [_converged("rerun", custom)])
    at.button(key="submit_prompt").click()
    at.run()
    at.run()
    assert at.session_state[fabrication_panel.SHEETS_KEY][0]["Thickness (mm)"] == pytest.approx(4.0)


@pytest.mark.parametrize("view", [agent_service.AgentRunView("error", message="stop"),
                                  agent_service.AgentRunView("clarification", message="which payload?")],
                         ids=["error", "clarification"])
def test_a_result_without_a_design_keeps_the_designers_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                view: Any) -> None:
    at = _app(tmp_path, monkeypatch)
    _record_runs(monkeypatch, [view])
    at.number_input(key="fab_bed_width_m").set_value(CUSTOM_BED_MM)
    at.button(key="submit_prompt").click()
    at.run()
    at.run()
    assert not at.exception
    assert at.number_input(key="fab_bed_width_m").value == pytest.approx(CUSTOM_BED_MM)


def test_an_edit_after_a_restore_survives_the_next_interaction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    custom = DEFAULT_FABRICATION.model_copy(update={"kerf_m": 0.0003})
    at = _app(tmp_path, monkeypatch)
    _record_runs(monkeypatch, [_converged("rerun", custom)])
    at.button(key="submit_prompt").click()
    at.run()
    assert at.number_input(key="fab_kerf_m").value == pytest.approx(0.3)  # applied at once, not at the next click
    at.number_input(key="fab_bed_width_m").set_value(700.0)
    at.run()
    assert at.number_input(key="fab_bed_width_m").value == pytest.approx(700.0)
    assert at.number_input(key="fab_kerf_m").value == pytest.approx(0.3)


def test_reset_wins_over_a_queued_design(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(tmp_path, monkeypatch)
    at.session_state[fabrication_panel.PENDING_KEY] = DEFAULT_FABRICATION.model_copy(update={"kerf_m": 0.0003})
    at.button(key="reset_fabrication").click()
    at.run()
    assert at.number_input(key="fab_kerf_m").value == pytest.approx(0.15)


# ------------------------------------------------- pure conversions ---
def test_default_panel_values_rebuild_the_default_settings() -> None:
    settings, errors = _rebuild()
    assert errors == [] and settings == DEFAULT_FABRICATION


def test_out_of_bounds_scalar_is_reported_by_its_label() -> None:
    settings, errors = _rebuild(values={"kerf_m": -1.0})
    assert settings is None
    assert len(errors) == 1 and errors[0].startswith("Laser kit: Kerf")


def test_tube_with_id_not_below_od_is_reported_in_user_terms() -> None:
    rows = fabrication_panel.tube_rows(DEFAULT_FABRICATION)
    settings, errors = _rebuild(tubes=[{**rows[0], "ID (mm)": rows[0]["OD (mm)"]}, *rows[1:]])
    assert settings is None
    assert any("Tube" in e and "inner diameter" in e for e in errors)


def test_an_uncovered_part_family_is_reported() -> None:
    rows = fabrication_panel.sheet_rows(DEFAULT_FABRICATION)
    settings, errors = _rebuild(sheets=[{**rows[1], "Families": "rib"}])
    assert settings is None and errors


def test_blank_or_non_numeric_cells_are_reported_not_raised() -> None:
    rows = fabrication_panel.sheet_rows(DEFAULT_FABRICATION)
    settings, errors = _rebuild(sheets=[{**rows[0], "Thickness (mm)": None}, rows[1]], rods="3, four")
    assert settings is None
    assert any("Sheet" in e for e in errors) and any("Rod" in e for e in errors)


# --------------------------------------------------------------- results ---
def _cad(kit_dir: Optional[Path], **extra: Any) -> CADArtifacts:
    return CADArtifacts(oml_vsp3_path="a.vsp3", oml_step_path="b.step", assembly_step_path="c.step", x_cg_m=0.3,
                        static_margin=0.11, collision_count=0, as_built_mass_kg=10.0, ixx_kgm2=1.0, iyy_kgm2=1.0,
                        izz_kgm2=1.0, ixz_kgm2=0.0, kit_dir=None if kit_dir is None else str(kit_dir), **extra)


def _kit(parent: Path, sheets: dict[str, int]) -> Path:
    kit = parent / "kit"
    (kit / "sheets").mkdir(parents=True)
    for stem, count in sheets.items():
        for n in range(1, count + 1):
            (kit / "sheets" / f"{stem}-{n}.dxf").write_text("0\nEOF\n")
    (kit / "build_sheet.md").write_text("# Build sheet\n")
    (kit / "parts.csv").write_text("label\n")
    return kit


def test_kit_summary_counts_the_real_sheet_files_by_stock(tmp_path: Path) -> None:
    kit = _kit(tmp_path, {"birch-ply-1-5-mm": 4, "birch-ply-3-mm": 2})
    text = presenters.kit_summary(_cad(kit, kit_mass_kg=1.84), DEFAULT_FABRICATION)
    assert text == "Laser kit: 1.84 kg, 6 sheets (4 × Birch ply 1.5 mm, 2 × Birch ply 3 mm)"


def test_kit_summary_without_settings_humanises_the_file_names(tmp_path: Path) -> None:
    kit = _kit(tmp_path, {"birch-ply-3-mm": 1})
    assert presenters.kit_summary(_cad(kit, kit_mass_kg=0.5)) == "Laser kit: 0.50 kg, 1 sheet (1 × birch ply 3 mm)"


def test_kit_summary_is_none_without_a_kit() -> None:
    assert presenters.kit_summary(_cad(None)) is None


def test_kit_summary_survives_a_missing_sheet_folder(tmp_path: Path) -> None:
    text = presenters.kit_summary(_cad(tmp_path / "gone", kit_mass_kg=0.5))
    assert text == "Laser kit: 0.50 kg, 0 sheets"


def test_cad_rows_list_the_kit_next_to_the_as_built_mass(tmp_path: Path) -> None:
    kit = _kit(tmp_path, {"birch-ply-3-mm": 1})
    labels = [label for label, _ in presenters.cad_rows(_cad(kit, kit_mass_kg=0.5), DEFAULT_FABRICATION)]
    assert labels[labels.index("As-built mass") + 1] == "Laser kit"
    assert "Laser kit" not in [label for label, _ in presenters.cad_rows(_cad(None))]


def test_downloadable_files_include_the_build_sheet_and_parts_list(tmp_path: Path) -> None:
    out = tmp_path / "run"
    out.mkdir()
    _kit(out, {"birch-ply-3-mm": 1})
    (out / "design.json").write_text("{}")
    names = [p.name for p in presenters.downloadable_files(out)]
    assert {"design.json", "build_sheet.md", "parts.csv"} <= set(names)
    assert names[-1] == "birch-ply-3-mm-1.dxf"  # M2: the cut sheets are offered too, after the build sheet


def test_downloadable_files_skip_a_symlinked_sheet_or_sheet_folder(tmp_path: Path) -> None:
    out = tmp_path / "run"
    (out / "kit" / "sheets").mkdir(parents=True)
    secret = tmp_path / "secret.dxf"
    secret.write_text("x")
    (out / "kit" / "sheets" / "birch-ply-3-mm-1.dxf").symlink_to(secret)
    assert presenters.downloadable_files(out) == []
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "birch-ply-3-mm-1.dxf").write_text("x")
    (out / "kit" / "sheets" / "birch-ply-3-mm-1.dxf").unlink()
    (out / "kit" / "sheets").rmdir()
    (out / "kit" / "sheets").symlink_to(elsewhere)
    assert presenters.downloadable_files(out) == []


def test_downloadable_files_skip_a_symlinked_kit_file(tmp_path: Path) -> None:
    out = tmp_path / "run"
    (out / "kit").mkdir(parents=True)
    secret = tmp_path / "secret.md"
    secret.write_text("x")
    (out / "kit" / "build_sheet.md").symlink_to(secret)
    assert presenters.downloadable_files(out) == []


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> pipeline.DesignRun:
    root = tmp_path_factory.mktemp("kitui")
    return pipeline.run_design(G1_ELECTRIC_FW, outputs_root=root / "o", db_path=root / "c.db", with_aero_cad=False)


def _kit_view(tmp_path: Path, run: pipeline.DesignRun, warning: Optional[str]) -> Any:
    out = tmp_path / "run"
    out.mkdir()
    kit = _kit(out, {"birch-ply-1-5-mm": 4, "birch-ply-3-mm": 2})
    aero = AeroStabilityOutput(cl0=0.2, cl_alpha_per_rad=5.0, cd0=0.03, ld_cruise=12.0, cm_alpha_per_rad=-1.0,
                               cn_beta_per_rad=0.1, cl_beta_per_rad=-0.1, cl_delta_a_per_rad=0.2,
                               cm_delta_e_per_rad=-1.0, cn_delta_r_per_rad=-0.1, x_np_m=0.4)
    notes = (f"Laser-cut kit: {warning}",) if warning else ()
    return _converged("kitrun", DEFAULT_FABRICATION, out_dir=out, sizing=run.sizing, aero=aero,
                      cad=_cad(kit, kit_mass_kg=1.84, kit_warning=warning), warnings=notes)


def test_kit_summary_and_warning_are_shown_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                run: pipeline.DesignRun) -> None:
    at = _app(tmp_path, monkeypatch)
    _record_runs(monkeypatch, [_kit_view(tmp_path, run, "2 parts did not fit the bed")])
    at.button(key="submit_prompt").click()
    at.run()
    assert not at.exception
    cells = " ".join(str(df.value) for df in at.dataframe)
    assert "Laser kit" in cells and "6 sheets" in cells
    assert sum("2 parts did not fit the bed" in w.value for w in at.warning) == 1


def test_build_sheet_and_parts_list_are_offered_for_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                             run: pipeline.DesignRun) -> None:
    at = _app(tmp_path, monkeypatch)
    _record_runs(monkeypatch, [_kit_view(tmp_path, run, None)])
    at.button(key="submit_prompt").click()
    at.run()
    labels = {b.label for b in at.get("download_button")}
    assert {"build_sheet.md", "parts.csv"} <= labels


def test_kit_file_names_come_from_one_dependency_free_module() -> None:
    """M4: the exporter and the UI share src/schemas/kit_files.py; the UI side needs no fabrication import."""
    import subprocess
    import sys

    from src.schemas import kit_files

    code = ("import sys; sys.modules['build123d'] = None; import src.schemas.kit_files, src.ui.presenters; "
            "assert not [m for m in sys.modules if m.startswith('src.tools.fabrication')]")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).resolve().parents[1])
    assert presenters.stock_slug is kit_files.stock_slug
    assert kit_files.sheet_file_name("Birch ply 1.5 mm", 2) == "birch-ply-1-5-mm-2.dxf"


def test_empty_tube_and_rod_lists_are_reported_in_user_terms() -> None:
    _, tube_errors = _rebuild(tubes=[])
    _, rod_errors = _rebuild(rods="")
    assert tube_errors == ["Laser kit: Tube stock needs at least one entry"]
    assert rod_errors == ["Laser kit: Rod stock needs at least one entry"]


def test_a_cleared_number_input_is_a_field_error_not_a_crash() -> None:
    settings, errors = _rebuild(values={"bed_width_m": None})
    assert settings is None
    assert errors == ["Laser kit: Bed width (mm): enter a number"]


def test_default_tubes_carry_no_float_dust() -> None:
    assert all(round(t.od_m, 9) == t.od_m and round(t.id_m, 9) == t.id_m for t in DEFAULT_FABRICATION.tubes)


# ---------------------------------------------------------------- re-runs (R37) ---
def _view_with_an_option(tmp_path: Path) -> Any:
    change = {"label": "shorter", "requirements": {"endurance_hours": 0.5}, "assumptions": {}, "knobs": {}}
    result = {"closes": True, "mtow_kg": 4.0, "dod_group": 1, "propulsion": "electric", "battery_mass_kg": 0.8,
              "fuel_mass_kg": 0.0, "reason": ""}
    option = {"lever": "endurance", "change": change, "result": result, "relative_change": 0.0, "closes": True,
              "mtow_saved_kg": None, "group_drop": False}
    return agent_service.AgentRunView("not_converged", run_id="r1", out_dir=tmp_path, requirements=G1_ELECTRIC_FW,
                                      explorer_mode="closure", options=(option,))


def _app_at_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[dict[str, Any]]]:
    """The app showing a run with one option loaded in the review panel; agent_service.rerun records its calls."""
    at = _app(tmp_path, monkeypatch)
    monkeypatch.setattr(agent_service, "run", lambda *a, **k: _view_with_an_option(tmp_path))
    at.button(key="submit_prompt").click()
    at.run()
    at.button(key="select_option_0").click()
    at.run()
    calls: list[dict[str, Any]] = []

    def fake_rerun(*args: object, **kwargs: object) -> Any:
        calls.append(dict(kwargs))
        return agent_service.AgentRunView("error", message="stop")

    monkeypatch.setattr(agent_service, "rerun", fake_rerun)
    return at, calls


def test_a_rerun_uses_the_panels_edited_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at, calls = _app_at_review(tmp_path, monkeypatch)
    at.number_input(key="fab_bed_width_m").set_value(CUSTOM_BED_MM)
    at.button(key="rerun").click()
    at.run()
    assert not at.exception
    assert len(calls) == 1 and calls[0]["fabrication"].bed_width_m == pytest.approx(CUSTOM_BED_MM / 1e3)


def test_invalid_panel_settings_block_a_rerun(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at, calls = _app_at_review(tmp_path, monkeypatch)
    at.number_input(key="fab_bed_width_m").set_value(0.0)
    at.button(key="rerun").click()
    at.run()
    assert not at.exception
    assert calls == []
    assert any("Laser kit" in e.value and "Bed width" in e.value for e in at.error)
    assert at.session_state["pending_change"] is not None  # the panel stays open to fix the value


def test_a_rerun_with_edited_settings_says_so(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.schemas.changes import ChangeSet
    from src.schemas.state import RunLineage

    change = ChangeSet.model_validate({"label": "shorter", "requirements": {"endurance_hours": 0.5},
                                       "assumptions": {}, "knobs": {}})
    lineage = RunLineage(parent_run_id="r1", change_set=change, change_source="option",
                         fabrication_changes=("bed_width_m", "tubes"))
    edited = DEFAULT_FABRICATION.model_copy(update={"bed_width_m": 0.8})
    at = _app(tmp_path, monkeypatch)
    _record_runs(monkeypatch, [_converged("r2", edited, lineage=lineage)])
    at.button(key="submit_prompt").click()
    at.run()
    assert not at.exception
    assert any("edited laser kit settings" in i.value and "Bed width (mm)" in i.value and "Tubes" in i.value
               for i in at.info)


def test_a_rerun_with_the_parents_settings_says_nothing_about_them(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(tmp_path, monkeypatch)
    _record_runs(monkeypatch, [_converged("r2", DEFAULT_FABRICATION)])
    at.button(key="submit_prompt").click()
    at.run()
    assert not any("edited laser kit settings" in i.value for i in at.info)
