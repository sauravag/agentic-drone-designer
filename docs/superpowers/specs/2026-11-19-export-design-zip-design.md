# Export a design as a zip — design (short)

## Intent

One click gives the designer every file of a run in a single zip. Opening a zip in the app ("sharing") is parked
(`2026-11-19-share-designs-design.md`), but the zip already carries the `design.json` record that the future
opener will read, so designs exported now stay openable later.

## Goals / non-goals

- Goal: a "Download all files (.zip)" button per design that holds every run file plus `design.json`.
- Non-goals: opening/importing a zip, restoring session history, zipping several designs at once.
- Anti-goals: the zip leaks files from outside the run folder (symlinks); `design.json` contains absolute paths
  from the sender's machine; `design.json` is written into the run folder itself (it exists only in the zip).

## Approaches

1. Zip the run folder only — rejected by the user: exported designs could never be opened later.
2. **Zip the run folder plus a versioned `design.json` built at download time from the in-memory view — chosen.**
3. Write `design.json` into the run folder at the end of every run — rejected: older runs would lack it, and the
   view in memory at download time is already the complete record.

## Design

- **Where:** a primary `st.download_button` "Download all files (.zip)" at the top of the Files tab
  (`src/ui/app.py` `_agent_tabs`), above the per-file buttons, which stay. File name `<run_id>.zip`, mime
  `application/zip`. Shown only when the run folder exists.
- **Zip layout:** everything under one top folder `<run_id>/`:
  - `<run_id>/design.json`;
  - every regular file under the run folder, recursively (`sitl/`, `vspaero/`, `airfoils/`, `viewer.html`, …),
    at its path relative to the run folder. Symlinks and anything resolving outside the folder are skipped
    (same rule as `presenters.downloadable_files`). A stray `design.json` already in the folder is replaced.
  - stdlib `zipfile`, `ZIP_DEFLATED`, built in memory.
- **`design.json`:** a frozen, `extra="forbid"` pydantic model (`DesignRecord`) dumped with
  `model_dump(mode="json")`:
  - `format = "isr-forge-design"`, `format_version = 1`, `created_utc` (ISO 8601 UTC);
  - the `AgentRunView` fields `status`, `message`, `run_id`, `requirements`, `sizing`, `bom`, `aero`, `cad`,
    `fcs`, `iterations`, `warnings`, `assumptions`, `gates`, `critic_log`, `report_path`, `used_llm`,
    `design_assumptions`, `explorer_mode`, `options`, `options_explanation`, `explorer_note`, `lineage`;
  - `out_dir` omitted; `report_path`, `cad.*_path` and `fcs.*_path` stored **relative to the run folder**. A
    path that is not inside the run folder raises `ValueError`, and the Files tab shows a warning instead of the
    zip button (it would mean a broken run, not something to paper over).
- **Code:** a new Streamlit-free module `src/ui/design_archive.py` with `design_record(view) -> DesignRecord` and
  `build_zip(view) -> bytes`; one widget block in `app.py`; README: the zip and `design.json` under
  "What a run produces".
- **Tests:** the record round-trips (`DesignRecord.model_validate(json)` equals the original) and the
  requirements/assumptions survive exactly; paths are relative; a path outside the folder raises; the zip holds
  nested files under `<run_id>/` plus `design.json`; symlinks and outside files are excluded; the button is
  absent when the folder is gone.

## Calls I made

- The button lives in the Files tab; per-file buttons stay.
- A top `<run_id>/` folder inside the zip.
- `DesignRecord` is a pydantic model now, so the future opener validates with the same class.
- Building the zip on every render vs caching per run is left to the builder (the per-file buttons already read
  files on every render).
