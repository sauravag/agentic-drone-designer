# Export Design Zip Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One "Download all files (.zip)" button per design that zips every run file plus a versioned `design.json`.

**Architecture:** A Streamlit-free module `src/ui/design_archive.py` builds a frozen pydantic `DesignRecord` from the
in-memory `AgentRunView` (paths made relative to the run folder) and zips it with the run folder's regular files
under a top `<run_id>/` folder. `app.py` adds one button to the Files tab.

**Tech Stack:** Python 3.11, pydantic v2, stdlib `zipfile`/`io`, Streamlit 1.67 (AppTest for the UI test).

**Spec:** `docs/superpowers/specs/2026-11-19-export-design-zip-design.md`

## Global Constraints

- `format = "isr-forge-design"`, `format_version = 1`, `created_utc` ISO 8601 UTC.
- Zip file name `<run_id>.zip`, mime `application/zip`, every entry under `<run_id>/`; `ZIP_DEFLATED`.
- `design.json` exists only in the zip, never written to the run folder; no absolute paths in it.
- Symlinks and anything resolving outside the run folder are excluded.
- Project rules: TDD, functions < 50 lines, files < 800 lines, named constants, immutable data, type hints.
- `tests/test_ui.py` is at 770 lines: new tests go in a new `tests/test_design_archive.py`.

## Review Focus

1. A symlinked directory inside the run folder (e.g. `sitl -> /elsewhere`) — its files must not be zipped.
2. Run paths given relative to the CWD (`outputs/<id>/oml.step`) vs absolute — both must become `oml.step`.
3. The run folder deleted while the result is on screen — no button, no exception.
4. A run with `cad=None`/`fcs=None`/`report_path=None` (sizing-only, or not converged) — record still builds.
5. A stray `design.json` already in the run folder — the zip holds exactly one, the fresh record.

---

### Task 1: `DesignRecord` and `design_record(view)`

**Files:**
- Create: `src/ui/design_archive.py`
- Test: `tests/test_design_archive.py`

**Interfaces:**
- Produces: `DESIGN_FORMAT: Final[str] = "isr-forge-design"`, `FORMAT_VERSION: Final[int] = 1`,
  `RECORD_FILE: Final[str] = "design.json"`, `class DesignRecord(BaseModel)` (frozen, `extra="forbid"`),
  `design_record(view: AgentRunView, now: Optional[datetime] = None) -> DesignRecord`.
- `DesignRecord` fields: `format: Literal["isr-forge-design"]`, `format_version: Literal[1]`, `created_utc: datetime`,
  then the AgentRunView fields with the same names and types except paths: `status: str`, `message: Optional[str]`,
  `run_id: str`, `requirements: Optional[OperationalRequirements]`, `sizing: Optional[SizingOutput]`,
  `bom: Optional[SelectedBOM]`, `aero: Optional[AeroStabilityOutput]`, `cad: Optional[CADArtifacts]`,
  `fcs: Optional[FCSArtifacts]`, `iterations: int`, `warnings: tuple[str, ...]`, `assumptions: tuple[str, ...]`,
  `gates: tuple[dict[str, Any], ...]`, `critic_log: tuple[dict[str, Any], ...]`, `report_path: Optional[str]`,
  `used_llm: bool`, `design_assumptions: DesignAssumptions`, `explorer_mode: str`, `options: tuple[dict[str, Any], ...]`,
  `options_explanation: Optional[str]`, `explorer_note: Optional[str]`, `lineage: Optional[RunLineage]`. No `out_dir`.
- `design_record` raises `ValueError` when the view has no `run_id`/`out_dir`, or when `report_path`, any
  `cad.*_path` (including `viewer_model_path` when set) or `fcs.*_path` does not resolve inside `out_dir`. Stored
  paths are POSIX strings relative to `out_dir` (via `Path(p).resolve().relative_to(out_dir.resolve())`).

- [ ] **Step 1: Write the failing tests** in `tests/test_design_archive.py`:
  - `test_record_round_trips_and_keeps_requirements_exact`: view with `requirements=G1_ELECTRIC_FW`, a
    non-default `DesignAssumptions` (e.g. `battery_reserve_fraction` changed), `sizing=size_aircraft(G1_ELECTRIC_FW)`,
    gates/critic_log dicts, a `RunLineage`; `DesignRecord.model_validate_json(rec.model_dump_json()) == rec`;
    `rec.requirements == view.requirements`; `rec.design_assumptions == view.design_assumptions`;
    `rec.format == "isr-forge-design"`, `rec.format_version == 1`; `"out_dir" not in rec.model_dump()`.
  - `test_record_paths_are_relative_to_the_run_folder`: CAD/FCS paths and `report_path` absolute inside `tmp_path`
    (one in a sub-folder, `sitl/sitl_report.json`) → `"oml.step"`, `"sitl/sitl_report.json"`, `"design_report.md"`;
    plus a CWD-relative form (`monkeypatch.chdir(tmp_path.parent)`, path `run/oml.step`) → `"oml.step"`.
  - `test_record_rejects_a_path_outside_the_run_folder`: `cad.oml_step_path` in a sibling folder → `ValueError`.
  - `test_record_without_artifacts_builds`: `cad=None, fcs=None, report_path=None, sizing=None` → builds.
  - `test_record_needs_a_run_folder`: `out_dir=None` → `ValueError`.

- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_design_archive.py -q`
  Expected: FAIL / collection error — `src.ui.design_archive` does not exist.

- [ ] **Step 3: Implement** `DesignRecord` and `design_record` in `src/ui/design_archive.py` (module docstring;
  rewrite artifact paths with `model_copy(update=...)`, never mutating the view).

- [ ] **Step 4: Run** the same command. Expected: 5 passed.

- [ ] **Step 5: Commit** `feat: versioned design record for exported designs`.

### Task 2: `build_zip(view)`

**Files:**
- Modify: `src/ui/design_archive.py`
- Test: `tests/test_design_archive.py`

**Interfaces:**
- Consumes: `design_record`, `RECORD_FILE` (Task 1).
- Produces: `ZIP_MIME: Final[str] = "application/zip"`, `zip_name(view: AgentRunView) -> str` (`f"{run_id}.zip"`),
  `build_zip(view: AgentRunView, now: Optional[datetime] = None) -> bytes`. Raises `FileNotFoundError` when
  `out_dir` is not a directory, `ValueError` as `design_record` does.
- Entries: `<run_id>/design.json` (`rec.model_dump_json(indent=2)`) first, then every regular, non-symlink file
  under `out_dir` (walk with `os.walk(followlinks=False)` or `rglob` filtering `is_symlink()` on the file *and*
  requiring `resolve()` inside the resolved root), sorted, at `<run_id>/<relative posix path>`; a top-level
  `design.json` in the folder is skipped.

- [ ] **Step 1: Write the failing tests:**
  - `test_zip_holds_every_file_under_the_run_id_folder`: files `oml.step`, `sitl/log.bin`, `vspaero/a/b.txt` →
    `namelist()` == sorted set `{"r1/design.json", "r1/oml.step", "r1/sitl/log.bin", "r1/vspaero/a/b.txt"}`; the
    zipped `design.json` parses with `DesignRecord.model_validate_json`; file bytes match; `zip_name(view) == "r1.zip"`.
  - `test_zip_skips_symlinks_and_outside_files`: a symlinked file to `tmp_path/secret.txt` and a symlinked dir to
    an outside dir with a file → neither appears.
  - `test_zip_replaces_a_stray_design_json`: folder has `design.json` with `"stale"` → exactly one
    `r1/design.json`, and it is the fresh record.
  - `test_zip_needs_an_existing_folder`: deleted folder → `FileNotFoundError`.
  - `test_design_json_is_not_written_to_the_run_folder`: after `build_zip`, `not (out_dir / "design.json").exists()`.

- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_design_archive.py -q`
  Expected: the 5 new tests FAIL (`build_zip` / `zip_name` missing).

- [ ] **Step 3: Implement** `zip_name` and `build_zip` (stdlib `zipfile.ZipFile(io.BytesIO(), "w", ZIP_DEFLATED)`).

- [ ] **Step 4: Run** the same command. Expected: 10 passed.

- [ ] **Step 5: Commit** `feat: zip a run folder with its design record`.

### Task 3: Files-tab button and README

**Files:**
- Modify: `src/ui/app.py` (`_agent_tabs`, files tab), `README.md` ("What a run produces")
- Test: `tests/test_design_archive.py`

**Interfaces:**
- Consumes: `build_zip`, `zip_name`, `ZIP_MIME` (Task 2).
- Produces: `app.py` constant `ZIP_LABEL: Final[str] = "Download all files (.zip)"`, a helper
  `_zip_download(view: AgentRunView) -> None` called first in the files tab: when `view.out_dir` is a directory,
  `st.download_button(ZIP_LABEL, data=build_zip(view), file_name=zip_name(view), mime=ZIP_MIME, type="primary",
  key=f"dl-zip-{view.run_id}")`; a `ValueError` shows `st.warning("This run's files cannot be zipped: <reason>")`
  and logs it; a missing folder renders nothing.

- [ ] **Step 1: Write the failing tests** (AppTest, reusing `tests.test_ui._llm_app` and the
  `monkeypatch.setattr(agent_service, "run", ...)` + `submit_prompt` click pattern used in `tests/test_ui.py`):
  - `test_files_tab_offers_the_zip`: a `not_converged` view with `out_dir=tmp_path/"r1"` (exists, one file) →
    no exception and a download button whose label is `"Download all files (.zip)"` (find it via
    `at.get("download_button")` and its `proto.label`).
  - `test_files_tab_warns_when_the_run_cannot_be_zipped`: `report_path` outside the folder → a warning containing
    `"cannot be zipped"`, no zip button.
  - `test_files_tab_has_no_zip_for_a_deleted_folder`: `out_dir` that does not exist → no zip button, no exception.

- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_design_archive.py -q`
  Expected: the 3 UI tests FAIL (no zip button / no warning).

- [ ] **Step 3: Implement** `ZIP_LABEL` and `_zip_download` in `app.py`; add README rows: the zip
  (`<run_id>.zip` from the Files tab: every file plus `design.json`) and what `design.json` is (versioned record of
  the run's results, paths relative to the run folder; used to reopen designs once sharing exists).

- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q -x` (full suite). Expected: all pass.

- [ ] **Step 5: Commit** `feat: download a design's files as one zip`.
