# Share designs as a zip — design

> **Status: PARKED (2026-11-19).** Only the export half was built; see
> `2026-11-19-export-design-zip-design.md`. Opening a shared zip in the app is deferred.

## Intent

A designer wants to hand a finished design to someone else. One download button per design produces a zip of
every file the run wrote. The recipient uploads that zip into **their own copy of the app** and the design opens
there as a full design they can keep working on: re-run it with changes, explore improvements, and ask the design
chat about it, exactly as if they had run it themselves (user choice **B**).

## Goals

- One "Download design (.zip)" button per design in the results view, carrying every file in the run folder
  (including sub-folders such as `airfoils/`, `sitl/`, `vspaero/`) plus a `design.json` record.
- An "Open a shared design" uploader that validates the zip, unpacks it into the recipient's outputs folder,
  rebuilds the design and shows it in the results view, listed in the session's design history.
- An opened design behaves like a native one: every results tab, the Files tab downloads, the 3D viewer, the
  report, the review/re-run panel, improvement exploration and the design chat all work.
- A re-run of an opened design is recorded as a child of it (`RunLineage.parent_run_id` = the opened run id).
- The opened design is saved to disk (`outputs/<run id>/`), so it survives a restart like the recipient's own runs.

## Non-goals

- Restoring the whole session history after a reload, or browsing old run folders from the UI.
- A zip of several designs at once (one zip = one design).
- Restoring the sender's chat conversation into the chat panel. The sender's `design_chat.json` travels in the
  zip and is extracted as `sender_chat.json` (downloadable from the Files tab), because `chat_panel._save`
  rewrites `design_chat.json` on every message and would otherwise overwrite it. The recipient's chat starts
  fresh.
- Links, accounts, or server-side sharing.

## Anti-goals (a failure even if it "works")

- An uploaded zip writes outside its own new run folder, overwrites an existing run, or exhausts the disk.
- Sender-supplied HTML or script runs in the recipient's app. (`viewer.html` is embedded with
  `components.html`, which executes scripts — so the zip's copy must never be shown.)
- A zip opens "successfully" but a re-run silently starts from different requirements or assumptions than the
  sender's design used.
- Loading uses `pickle` or anything else that can execute code from the file.

## Approaches considered

1. **Zip the folder as-is; rebuild the design from the existing JSON reports.** Cheapest download, but those
   files are reports, not a complete record: the critic gate results and the explorer options shown in the
   results view are not obviously in any of them, so opening would depend on reassembling files never meant to
   be complete, and would break silently when a report changes.
2. **Zip the folder plus one complete `design.json` written at download time — chosen by the user.** The
   results view already holds the full design in memory (`AgentRunView`); serialise it, version it, and validate
   it on open with the same pydantic models the app uses for its own runs. Costs one serialised record to keep
   in step with `AgentRunView`.
3. **Pickle the view.** Rejected: unpickling an untrusted file can run arbitrary code.

## How it works

### Download

- In the Files tab (`src/ui/app.py` `_agent_tabs`, above the per-file buttons), one primary
  `st.download_button` "Download design (.zip)", file name `<run_id>.zip`, mime `application/zip`.
- Shown only for designs with a run folder (`view.has_design and view.out_dir is not None`).
- The zip contains:
  - `design.json` at the root (below);
  - every **regular file** under the run folder, recursively, at its path relative to the run folder. Symlinks
    and anything resolving outside the run folder are skipped. A `design.json` already in the folder is replaced
    by the freshly written one.
- `design.json`:
  - `format`: a fixed identifier string (e.g. `"isr-forge-design"`) and `format_version`: integer, starting at 1;
  - `created_utc` (ISO 8601, informational only);
  - the `AgentRunView` fields: `status`, `message`, `run_id`, `requirements`, `sizing`, `bom`, `aero`, `cad`,
    `fcs`, `iterations`, `warnings`, `assumptions`, `gates`, `critic_log`, `report_path`, `used_llm`,
    `design_assumptions`, `explorer_mode`, `options`, `options_explanation`, `explorer_note`, `lineage`;
  - **every path stored relative to the run folder** (`report_path`, `cad.*_path`, `fcs.*_path`); `out_dir` is
    omitted. Absolute paths from the sender's machine (e.g. `/app/outputs/...`) never appear in the record.

### Open

- An "Open a shared design" `st.file_uploader` (type `zip`), placed with the design history in `main()`.
- Steps, all-or-nothing:
  1. Check that the upload is a valid zip, and that the outputs root has at least the uncompressed-size cap
     free (`shutil.disk_usage`); otherwise refuse with "not enough disk space".
  2. Inspect every entry *before* extracting: reject absolute names, `..` components, drive letters or
     backslashes, symlink entries (from `external_attr`), duplicate names, more entries than a cap, and a total
     declared uncompressed size over a cap. Header sizes can be forged, so extraction also counts the bytes
     actually written and aborts once the cap is passed.
  3. Read and validate `design.json` into a frozen, `extra="forbid"` pydantic model. A missing file, unknown
     `format`, newer `format_version`, a stored path that is absolute or escapes the folder, or any validation
     error fails with a clear message. Statuses `clarification` and `error` are rejected (they have no design);
     `converged`, `not_converged` and `converged_sitl_failed` all open and show like native runs.
     - `gates`, `critic_log` and `options` get small typed pydantic models matching the shapes the app writes
       (`gates`: `{name, passed, detail}`; `critic_log`: the critic's entry keys; `options`:
       `ExploreOption.to_json()`, whose change set validates as a `ChangeSet`), not loose dicts. Free text
       (`message`, `warnings`, `assumptions`, `options_explanation`, `explorer_note`) is length-capped and shown
       through the same rendering paths as native runs.
     - The BOM and sizing are self-contained records; they are not looked up in the recipient's component
       database, so a different catalogue does not block opening. The one catalogue check is the existing
       `OperationalRequirements.payload_model` validator: an unknown payload model is rejected, because a
       re-run could not use it.
  4. Claim a fresh folder with the existing `pipeline.create_run_dir(OUTPUTS_ROOT, run_id)` — it already
     suffixes the id when the name is taken, so opening the same zip twice never overwrites anything. The stored
     id is first reduced to one safe path component (same character rule as `pipeline.new_run_id`).
  5. Extract the checked entries into that folder, **except `viewer.html`, which is never extracted**, and with
     `design_chat.json` renamed to `sender_chat.json`.
  6. **Regenerate `viewer.html`** with `src.viewer.build.write_viewer(run_dir)`, which builds it from the
     validated `viewer_model.json`. If there is no model or the rebuild fails, there is no `viewer.html` at all
     (the 3D tab already explains a missing viewer); the open still succeeds.
  7. Rebuild the `AgentRunView` with paths re-joined onto the new folder and `run_id` = the claimed folder name,
     then `design_loop.record_run(view)`.
  8. On any failure after the folder is claimed, delete the folder. The user sees one clear error; details go to
     the server log.
- Streamlit re-runs the script on every interaction and the uploader keeps its file, so each upload is opened
  **once**: the uploader's `file_id` is remembered in session state, and a repeat sees it and does nothing.
- An opened design's `lineage` may name a parent run the recipient does not have; it is shown as-is
  ("from `<parent id>`").
- The opened design's report, Files tab and 3D viewer read from the new folder through the existing guarded
  readers (`presenters.read_report`, `presenters.downloadable_files`, `viewer_panel.render_viewer`).

### Re-run fidelity

`agent_service.rerun` needs only `requirements`, `design_assumptions` and `run_id` from the view (the critic
restarts from default knobs). These round-trip exactly through `model_dump(mode="json")` / `model_validate`. A
test proves an opened design and its source give identical `apply_changes` results for the same change set.

## Constraints

- Python, Streamlit 1.67, pydantic v2 frozen models, stdlib `zipfile` (no new dependency).
- Project rules: TDD, 80%+ coverage, functions < 50 lines, files < 800 lines, named constants for every limit,
  immutable data, SI units, no secrets in code.
- Streamlit's upload limit (`server.maxUploadSize`, default 200 MB) already exceeds a typical run (about 22 MB,
  about 55 files); no change needed.
- The new logic lives outside Streamlit in its own module (e.g. `src/ui/design_archive.py` with
  `build_zip(view) -> bytes` and `open_zip(data, outputs_root) -> AgentRunView`) so it is unit-testable like
  `agent_service`. `app.py` (381 lines) and `design_loop.py` (333) only gain the two widgets.

## Decided by the user

- Sharing is a zip per design that the recipient opens in their own copy of the app.
- The recipient can keep designing from it (B), not view-only.
- Approach 2: the zip carries a complete, versioned `design.json` written at download time.

## Calls I made (correct any)

- One zip per design; the button sits in the Files tab.
- Opened designs are saved to the recipient's outputs folder and listed in the session history.
- The sender's chat is not restored into the chat panel (the file is still included).
- `viewer.html` is always regenerated on open; the sender's copy is never displayed.
- Caps, as named constants: 500 MB uncompressed in total, 5,000 entries, enforced on bytes actually written;
  plus a free-space check. No cumulative cap on the outputs folder (the recipient's own runs have none either).
- `viewer.html` is never extracted from a zip; `design_chat.json` is renamed `sender_chat.json`.
- Each upload opens once per `file_id`.
- All three design statuses can be shared and opened; clarification/error records are rejected.
- `gates`, `critic_log` and `options` are validated with typed models; free text is length-capped.
- A different component catalogue does not block opening; an unknown payload model does.
- Version 1 is the only format; a lower version would need a migration, none exists yet.

## Left to the builder

- Exact constant names, error wording, and where the uploader sits relative to the history expander.
- Whether the zip bytes are built on each render or cached per `run_id` (the existing per-file buttons already
  read files on every render; caching is fine if it stays simple).
- README: a short "Sharing a design" section and a `design.json` note in "What a run produces".
