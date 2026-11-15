"""ISR-Forge local web form: enter mission requirements, run the design pipeline, inspect and download.

    Full pipeline (OpenVSP aero + CAD + ArduPilot SITL):  docker compose up ui  ->  http://localhost:8501
    Sizing + parts list only (host):     streamlit run src/ui/app.py

Environment: ISR_FORGE_OUTPUTS (run folders, default ./outputs), ISR_FORGE_DB (COTS catalog path).
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # `streamlit run` only puts src/ui on sys.path
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st  # noqa: E402

from src import pipeline  # noqa: E402
from src.tools.database.seed_data import DEFAULT_DB_PATH  # noqa: E402
from src.ui import presenters  # noqa: E402
from src.ui.examples import EXAMPLE_MISSIONS  # noqa: E402

logger = logging.getLogger(__name__)

OUTPUTS_ROOT: Final[Path] = Path(os.environ.get("ISR_FORGE_OUTPUTS", str(pipeline.DEFAULT_OUTPUTS_ROOT)))
DB_PATH: Final[Path] = Path(os.environ.get("ISR_FORGE_DB", str(DEFAULT_DB_PATH)))
AIRFRAMES: Final[tuple[str, ...]] = ("auto", "fixed_wing", "quadplane_vtol")
PROPULSION: Final[tuple[str, ...]] = ("auto", "electric", "series_hybrid")
GROUPS: Final[tuple[int | str, ...]] = ("auto", 1, 2, 3)


def _load_preset() -> None:
    for key, value in presenters.form_defaults(EXAMPLE_MISSIONS[st.session_state["preset"]]).items():
        st.session_state[key] = value


def _init_state() -> None:
    if "mission_name" not in st.session_state:
        st.session_state["preset"] = next(iter(EXAMPLE_MISSIONS))
        _load_preset()
    st.session_state.setdefault("with_aero_cad", pipeline.aero_cad_available())
    st.session_state.setdefault("with_sitl", pipeline.aero_cad_available() and pipeline.sitl_available())


def _number_inputs() -> None:
    for column, (title, fields) in zip(st.columns(len(presenters.NUMBER_FIELDS)), presenters.NUMBER_FIELDS.items()):
        with column:
            st.subheader(title)
            for key, lo, hi, step, fmt in fields:
                st.number_input(presenters.FIELD_LABELS[key], min_value=lo, max_value=hi, step=step, format=fmt, key=key)


def _configuration_inputs() -> None:
    st.subheader("Configuration")
    a, p, g, r = st.columns(4)
    a.selectbox(presenters.FIELD_LABELS["airframe_topology"], AIRFRAMES, key="airframe_topology",
                help="auto: fixed-wing with a runway, VTOL quadplane without")
    p.selectbox(presenters.FIELD_LABELS["propulsion_type"], PROPULSION, key="propulsion_type",
                help="auto: battery-electric, falling back to series-hybrid when electric cannot close")
    g.selectbox(presenters.FIELD_LABELS["group_constraint"], GROUPS, key="group_constraint",
                format_func=lambda v: "auto" if v == "auto" else f"Group {v}")
    r.checkbox(presenters.FIELD_LABELS["runway_available"], key="runway_available")


def _requirements_form() -> bool:
    with st.form("requirements"):
        st.text_input(presenters.FIELD_LABELS["mission_name"], key="mission_name", max_chars=200)
        _number_inputs()
        _configuration_inputs()
        available: bool = pipeline.aero_cad_available()
        st.checkbox("Run OpenVSP aerodynamics + 3D CAD (minutes)", key="with_aero_cad", disabled=not available,
                    help=None if available else "Needs the Docker image: docker compose up ui")
        sitl: bool = available and pipeline.sitl_available()
        st.checkbox("Fly the ArduPilot SITL take-off and doublet test (needs aero + CAD)", key="with_sitl",
                    disabled=not sitl, help=None if sitl else "Needs the Docker image: docker compose up ui")
        return st.form_submit_button("Design aircraft", type="primary", key="submit")


def _run(values: dict[str, Any]) -> None:
    req, errors = presenters.parse_form(values)
    if req is None:
        for message in errors:
            st.error(message)
        st.session_state.pop("last_run", None)
        return
    with st.status("Designing…", expanded=True) as status:
        try:
            do_aero: bool = st.session_state["with_aero_cad"]
            run = pipeline.run_design(req, OUTPUTS_ROOT, DB_PATH, with_aero_cad=do_aero, progress=status.write,
                                      with_sitl=do_aero and st.session_state["with_sitl"])
        except pipeline.DESIGN_ERRORS as exc:
            logger.warning("design run failed: %s", exc)
            status.update(label="Design failed", state="error")
            st.error(f"The mission could not be designed: {exc}")
            st.session_state.pop("last_run", None)
            return
        status.update(label=f"Design complete: {run.run_id}", state="complete", expanded=False)
    st.session_state["last_run"] = run


def _headline(run: pipeline.DesignRun) -> None:
    s = run.sizing
    cols = st.columns(6)
    cols[0].metric("MTOW", f"{s.mtow_kg:.2f} kg")
    cols[1].metric("DoD group", f"Group {s.dod_group}")
    cols[2].metric("Wingspan", f"{s.wingspan_m:.2f} m")
    ld = run.aero.ld_cruise if run.aero else s.assumed_ld_cruise
    cols[3].metric("Cruise L/D", f"{ld:.1f}", help="VSPAERO" if run.aero else "sizing estimate")
    cols[4].metric("Static margin", f"{run.cad.static_margin:.1%}" if run.cad else "not run")
    cols[5].metric("SITL flight test", ("passed" if run.fcs.passed else "FAILED") if run.fcs else "not run")


def _table(rows: list[tuple[str, str]]) -> None:
    st.dataframe([{"Quantity": k, "Value": v} for k, v in rows], hide_index=True, use_container_width=True)


def _render(run: pipeline.DesignRun) -> None:
    st.header(run.requirements.mission_name)
    st.caption(f"Run `{run.run_id}` · files in `{run.out_dir}`")
    for note in run.warnings:
        st.warning(note)
    _headline(run)
    sizing_tab, parts_tab, aero_tab, fcs_tab, files_tab = st.tabs(
        ["Sizing", "Components", "Aero & CAD", "Flight test", "Files"])
    with sizing_tab:
        _table(presenters.sizing_summary(run.sizing))
        st.subheader("Mission phases")
        st.dataframe(presenters.phase_rows(run.sizing), hide_index=True, use_container_width=True)
    with parts_tab:
        st.dataframe(presenters.bom_rows(run.bom), hide_index=True, use_container_width=True)
        st.caption("Catalog parts are representative; verify against manufacturer datasheets.")
    with aero_tab:
        if run.aero is None or run.cad is None:
            st.info("OpenVSP aerodynamics and CAD were not run for this design.")
        else:
            st.dataframe(presenters.aero_rows(run.aero), hide_index=True, use_container_width=True)
            _table(presenters.cad_rows(run.cad))
    with fcs_tab:
        if run.fcs is None:
            st.info("The ArduPilot SITL flight test was not run for this design.")
        else:
            st.dataframe(presenters.fcs_rows(run.fcs), hide_index=True, use_container_width=True)
            st.caption("ardupilot.param was flown in arduplane + JSBSim: take-off (VTOL climb and transition for a "
                       "quadplane), then +15° roll and +10° pitch steps in FBWA. Details in sitl_report.json.")
    with files_tab:
        for path in presenters.downloadable_files(run.out_dir):
            st.download_button(path.name, data=path.read_bytes(), file_name=path.name, key=f"dl-{run.run_id}-{path.name}")


def main() -> None:
    st.set_page_config(page_title="ISR-Forge", layout="wide")
    st.title("ISR-Forge")
    st.caption("Design a Group 1-3 ISR UAV from mission requirements. All engineering numbers come from "
               "deterministic physics tools.")
    _init_state()
    st.selectbox("Start from an example", list(EXAMPLE_MISSIONS), key="preset", on_change=_load_preset)
    if _requirements_form():
        _run({key: st.session_state[key] for key in presenters.FIELD_LABELS})
    if "last_run" in st.session_state:
        _render(st.session_state["last_run"])


main()
