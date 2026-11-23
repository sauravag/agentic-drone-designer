"""ISR-Forge local web form: describe a mission (or fill in its requirements), run the design agent with its
critic loop, then inspect the design, the critic's decisions and the report, and download every file.

    Full pipeline (OpenVSP aero + CAD + ArduPilot SITL):  docker compose up ui  ->  http://localhost:8501
    Sizing + parts list only (host):     streamlit run src/ui/app.py

Environment: ISR_FORGE_OUTPUTS (run folders, default ./outputs), ISR_FORGE_DB (COTS catalog path),
OPENROUTER_API_KEY or ANTHROPIC_API_KEY (optional: enables designing from a description, the LLM critic and the
report narrative), ISR_FORGE_LLM_PROVIDER and ISR_FORGE_MODEL (see src/agent/providers.py).
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Final, Optional

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # `streamlit run` only puts src/ui on sys.path
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st  # noqa: E402

from src import pipeline  # noqa: E402
from src.agent.providers import LLMConfigError  # noqa: E402
from src.agent.report import defang, inline  # noqa: E402
from src.schemas.requirements import (ARK_JETSON_PAB, ENABLED_LAYOUTS, LAYOUT_LABELS,  # noqa: E402
                                      OperationalRequirements)
from src.tools.database.seed_data import DEFAULT_DB_PATH  # noqa: E402
from src.schemas.assumptions import DesignAssumptions  # noqa: E402
from src.schemas.fabrication import FabricationSettings  # noqa: E402
from src.schemas.payload_catalog import payload_choices, payload_label, payload_spec  # noqa: E402
from src.ui import agent_service, chat_panel, design_archive, design_loop, presenters, viewer_panel  # noqa: E402
from src.ui.agent_service import AgentRunView  # noqa: E402
from src.ui.assumptions_panel import current_assumptions, render_assumptions_panel  # noqa: E402
from src.ui.examples import EXAMPLE_MISSIONS  # noqa: E402
from src.ui.fabrication_panel import current_fabrication, render_fabrication_panel  # noqa: E402

logger = logging.getLogger(__name__)

OUTPUTS_ROOT: Final[Path] = Path(os.environ.get("ISR_FORGE_OUTPUTS", str(pipeline.DEFAULT_OUTPUTS_ROOT)))
DB_PATH: Final[Path] = Path(os.environ.get("ISR_FORGE_DB", str(DEFAULT_DB_PATH)))
AIRFRAMES: Final[tuple[str, ...]] = ("auto", "fixed_wing", "quadplane_vtol")
LAUNCHES: Final[tuple[str, ...]] = ("auto", "runway", "hand")
# Layouts built for at least one topology, in LAYOUT_LABELS order; validation explains a topology it lacks.
LAYOUTS: Final[tuple[str, ...]] = tuple(name for name in LAYOUT_LABELS
                                        if any(name == built for built, _ in ENABLED_LAYOUTS))
PROPULSION: Final[tuple[str, ...]] = ("auto", "electric", "series_hybrid")
GROUPS: Final[tuple[int | str, ...]] = ("auto", 1, 2, 3)
MAX_PROMPT_CHARS: Final[int] = 2000
EXAMPLE_PROMPT: Final[str] = ("Design a runway-independent Group 2 VTOL ISR drone with a 3.5 kg EO/IR gimbal, "
                              "6-hour endurance, and 80 km datalink radius")
DESIGN_TABS: Final[tuple[str, ...]] = ("Sizing", "Components", "Aero & CAD", "3D", "Flight test")
COMPANION_CUBE_KEY: Final[str] = "companion_computer_cube"  # editable box (Cube); writes "companion_computer"
COMPANION_ARK_KEY: Final[str] = "companion_computer_ark"  # greyed, ticked box shown for the ARK stack
PENDING_DESIGN_KEY: Final[str] = "pending_design"  # set by Design aircraft inside the box; main() runs it
AGENT_TABS: Final[tuple[str, ...]] = ("Critic", "Report", "Files")
HEADLINE_COLUMNS: Final[int] = 6
GENERIC_FAILURE: Final[str] = ("The design run failed unexpectedly; details are in the server log and the run "
                               "folder's run_summary.json.")
STATUS_TEXT: Final[dict[str, str]] = {
    agent_service.STATUS_CONVERGED: "Converged: every critic gate passed",
    "converged_sitl_failed": "Converged on the critic gates, but the SITL flight test FAILED",
    "not_converged": "NOT converged: the critic could not satisfy every gate (see the Critic tab)",
}


def _load_preset() -> None:
    for key, value in presenters.form_defaults(EXAMPLE_MISSIONS[st.session_state["preset"]]).items():
        if key not in presenters.HARDWARE_KEYS:  # presets describe missions, not the designer's hardware choice
            st.session_state[key] = value


def _init_state() -> None:
    if "mission_name" not in st.session_state:
        st.session_state["preset"] = next(iter(EXAMPLE_MISSIONS))
        _load_preset()
    for key, value in presenters.NEW_DESIGN_HARDWARE.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("prompt", EXAMPLE_PROMPT)
    st.session_state.setdefault("with_aero_cad", pipeline.aero_cad_available())
    st.session_state.setdefault("with_sitl", pipeline.aero_cad_available() and pipeline.sitl_available())


# ------------------------------------------------------------------ inputs ---
def _stage_options() -> None:
    available: bool = pipeline.aero_cad_available()
    st.checkbox("Run OpenVSP aerodynamics + 3D CAD (minutes)", key="with_aero_cad", disabled=not available,
                help=None if available else "Needs the Docker image: docker compose up ui")
    sitl: bool = available and pipeline.sitl_available()
    st.checkbox("Fly the ArduPilot SITL take-off and doublet test (needs aero + CAD)", key="with_sitl",
                disabled=not (sitl and st.session_state["with_aero_cad"]),
                help=None if sitl else "Needs the Docker image: docker compose up ui")


def _llm_info() -> None:
    """Read-only: the provider and model are set only in .env (ISR_FORGE_LLM_PROVIDER, ISR_FORGE_MODEL)."""
    if agent_service.llm_problem() is not None:
        return
    try:
        st.caption(f"LLM: {defang(agent_service.llm_label())} (set in .env)")
    except LLMConfigError as exc:
        st.error(defang(str(exc)))


def _description_form() -> bool:
    problem: Optional[str] = agent_service.llm_problem()
    with st.form("describe"):
        st.subheader("Describe the mission")
        st.text_area("Design request", key="prompt", max_chars=MAX_PROMPT_CHARS, disabled=problem is not None,
                     help="The LLM extracts the requirements; every number is then computed by the physics tools.")
        if problem:
            st.caption(defang(problem))
        return st.form_submit_button("Design from description", type="primary", key="submit_prompt", disabled=problem is not None)


def _number_inputs(title: str) -> None:
    for key, lo, hi, step, fmt in presenters.NUMBER_FIELDS[title]:
        st.number_input(presenters.FIELD_LABELS[key], min_value=lo, max_value=hi, step=step, format=fmt, key=key)


def _fill_payload() -> None:
    """on_change: a catalogue model brings its datasheet mass and power; Custom keeps the current numbers."""
    spec = payload_spec(st.session_state["payload_model"])
    if spec is not None:
        st.session_state["payload_mass_kg"] = spec.mass_kg
        st.session_state["payload_power_w"] = spec.power_w


def _payload_inputs() -> None:
    st.subheader("Payload")
    st.selectbox(presenters.FIELD_LABELS["payload_model"], payload_choices(), key="payload_model",
                 format_func=payload_label, on_change=_fill_payload)
    _number_inputs("Payload")
    st.caption(presenters.PAYLOAD_CAPTION)


def _configuration_inputs() -> None:
    st.subheader("Configuration")
    a, layout, launch, p, g, r = st.columns(6)
    a.selectbox(presenters.FIELD_LABELS["airframe_topology"], AIRFRAMES, key="airframe_topology",
                help="auto: fixed-wing with a runway or a hand launch, VTOL quadplane without")
    layout.selectbox(presenters.FIELD_LABELS["airframe_layout"], LAYOUTS, key="airframe_layout",
                     format_func=LAYOUT_LABELS.get,
                     help="Pod-and-boom tractor: nose propeller, conventional tail. Single-boom V-tail pusher: "
                          "propeller at the boom end behind a V-tail (fixed-wing or quadplane)")
    launch.selectbox(presenters.FIELD_LABELS["launch_method"], LAUNCHES, key="launch_method",
                     help="auto: off a runway if there is one, else VTOL. hand: a hand-launched fixed-wing "
                          "(Group 1, no runway)")
    p.selectbox(presenters.FIELD_LABELS["propulsion_type"], PROPULSION, key="propulsion_type",
                help="auto: battery-electric, falling back to series-hybrid when electric cannot close")
    g.selectbox(presenters.FIELD_LABELS["group_constraint"], GROUPS, key="group_constraint",
                format_func=lambda v: "auto" if v == "auto" else f"Group {v}")
    r.checkbox(presenters.FIELD_LABELS["runway_available"], key="runway_available")


def _store_companion() -> None:
    """on_change: keep the Cube's Jetson choice in a plain key that survives runs where the box is not drawn."""
    st.session_state["companion_computer"] = st.session_state[COMPANION_CUBE_KEY]


def _avionics_inputs() -> None:
    """The Avionics column: the Jetson box greys out the moment the ARK stack is chosen (the box is a fragment)."""
    st.subheader("Avionics")
    st.selectbox(presenters.FIELD_LABELS["flight_controller"], tuple(presenters.FLIGHT_CONTROLLER_NAMES),
                 key="flight_controller", format_func=presenters.FLIGHT_CONTROLLER_NAMES.get,
                 help="Cube Orange+: flight controller only, Jetson optional. ARK Jetson PAB: flight controller and "
                      "Jetson Orin NX 16 GB on one board stack.")
    label: str = presenters.FIELD_LABELS["companion_computer"]
    if st.session_state["flight_controller"] == ARK_JETSON_PAB:
        st.checkbox(label, value=True, disabled=True, key=COMPANION_ARK_KEY)
        st.caption("Built into the ARK stack.")
    else:
        st.checkbox(label, value=st.session_state["companion_computer"], key=COMPANION_CUBE_KEY,
                    on_change=_store_companion, help="Jetson Orin NX on its own carrier board (see Avionics "
                                                     "in Design assumptions for its mass and power).")
    st.caption("A description sets its own avionics.")


@st.fragment
def _requirements_box() -> None:
    """A fragment, not a form: each edit reruns only this box, so the payload prefill and the ARK greying are live.
    Design aircraft flags a run and reruns the whole app, where main() starts it below the box."""
    with st.container(border=True):
        st.text_input(presenters.FIELD_LABELS["mission_name"], key="mission_name", max_chars=200)
        payload, mission, envelope, avionics = st.columns(4)
        with payload:
            _payload_inputs()
        for column, title in ((mission, "Mission"), (envelope, "Envelope")):
            with column:
                st.subheader(title)
                _number_inputs(title)
        with avionics:
            _avionics_inputs()
        _configuration_inputs()
        if st.button("Design aircraft", type="primary", key="submit"):
            st.session_state[PENDING_DESIGN_KEY] = True
            st.rerun(scope="app")


# --------------------------------------------------------------------- run ---
def _fail(status: Any, label: str, message: str) -> None:
    status.update(label=label, state="error")
    st.error(message)
    st.session_state.pop("last_run", None)


def _status_label(view: AgentRunView) -> str:
    if view.status == agent_service.STATUS_CLARIFY:
        return "Needs more information"
    if view.status == agent_service.STATUS_ERROR:
        return "Design failed"
    return f"Run {view.run_id} finished: {view.status.replace('_', ' ')}"


def _execute(assumptions: DesignAssumptions, fabrication: FabricationSettings, prompt: str = "",
             requirements: Optional[OperationalRequirements] = None) -> None:
    with st.status("Designing…", expanded=True) as status:
        try:
            do_aero: bool = st.session_state["with_aero_cad"]
            view = agent_service.run(OUTPUTS_ROOT, DB_PATH, with_aero_cad=do_aero,
                                     with_sitl=do_aero and st.session_state["with_sitl"], progress=status.write,
                                     prompt=prompt, requirements=requirements,
                                     design_assumptions=assumptions, fabrication=fabrication)
        except (pipeline.PipelineError, LLMConfigError) as exc:
            logger.warning("design run rejected: %s", exc)
            _fail(status, "Design not started", defang(str(exc)))
            return
        except Exception:  # a bug or I/O failure: log the traceback, never leave a stale result on screen
            logger.exception("design run failed")
            _fail(status, "Design failed", GENERIC_FAILURE)
            return
        failed: bool = view.status == agent_service.STATUS_ERROR
        status.update(label=_status_label(view), state="error" if failed else "complete", expanded=False)
    design_loop.record_run(view)


def _assumptions_or_stop() -> Optional[DesignAssumptions]:
    """The panel's validated assumptions; None (messages shown, no run) when a value is out of bounds."""
    assumptions, errors = current_assumptions()
    if assumptions is None:
        for message in errors:
            st.error(f"Design assumptions: {message}")
        st.session_state.pop("last_run", None)
    return assumptions


def _settings_or_stop() -> Optional[tuple[DesignAssumptions, FabricationSettings]]:
    """Both panels' validated settings; None (every message shown, no run) when either has an invalid value."""
    assumptions = _assumptions_or_stop()
    fabrication, errors = current_fabrication()
    if fabrication is None:
        for message in errors:  # already worded "Laser kit: <field>: <problem>"
            st.error(message)
        st.session_state.pop("last_run", None)
    return None if assumptions is None or fabrication is None else (assumptions, fabrication)


def _run_description(prompt: str) -> None:
    settings = _settings_or_stop()
    if settings is None:
        return
    if not prompt.strip():
        st.error("Describe the mission first.")
        st.session_state.pop("last_run", None)
        return
    _execute(*settings, prompt=prompt)


def _run_form(values: dict[str, Any]) -> None:
    settings = _settings_or_stop()
    if settings is None:
        return
    req, errors = presenters.parse_form(values)
    if req is None:
        for message in errors:
            st.error(message)
        st.session_state.pop("last_run", None)
        return
    _execute(*settings, requirements=req)


# ------------------------------------------------------------------ output ---
def _headline(view: AgentRunView) -> None:
    s = view.sizing
    cols = st.columns(HEADLINE_COLUMNS)
    cols[0].metric("MTOW", f"{s.mtow_kg:.2f} kg")
    cols[1].metric("DoD group", f"Group {s.dod_group}")
    cols[2].metric("Critic iterations", str(view.iterations))
    ld = view.aero.ld_cruise if view.aero else s.assumed_ld_cruise
    cols[3].metric("Cruise L/D", f"{ld:.1f}", help="VSPAERO" if view.aero else "sizing estimate")
    cols[4].metric("Static margin", f"{view.cad.static_margin:.1%}" if view.cad else "not run")
    cols[5].metric("SITL flight test", ("passed" if view.fcs.passed else "FAILED") if view.fcs else "not run")
    if view.requirements is not None:
        st.caption("Payload: " + presenters.payload_summary(view.requirements))
        st.caption("Avionics: " + presenters.avionics_summary(view.requirements, view.design_assumptions))


def _table(rows: list[tuple[str, str]]) -> None:
    st.dataframe([{"Quantity": k, "Value": v} for k, v in rows], hide_index=True, width="stretch")


def _design_tabs(view: AgentRunView, sizing_tab: Any, parts_tab: Any, aero_tab: Any, viewer_tab: Any,
                  fcs_tab: Any) -> None:
    with sizing_tab:
        _table(presenters.sizing_summary(view.sizing))
        st.subheader("Mission phases")
        st.dataframe(presenters.phase_rows(view.sizing), hide_index=True, width="stretch")
    with parts_tab:
        if view.bom is None:
            st.info("No parts list: the catalog step did not complete for the final iteration.")
        else:
            st.dataframe(presenters.bom_rows(view.bom), hide_index=True, width="stretch")
            st.caption("Catalog parts are representative; verify against manufacturer datasheets.")
    with aero_tab:
        if view.aero is None or view.cad is None:
            st.info("OpenVSP aerodynamics and CAD were not run for this design.")
        else:
            st.dataframe(presenters.aero_rows(view.aero), hide_index=True, width="stretch")
            _table(presenters.cad_rows(view.cad, view.fabrication))
    with viewer_tab:
        viewer_panel.render_viewer(view)
    with fcs_tab:
        if view.fcs is None:
            st.info("The ArduPilot SITL flight test was not run for this design.")
        else:
            st.dataframe(presenters.fcs_rows(view.fcs), hide_index=True, width="stretch")
            st.caption("ardupilot.param was flown in arduplane + JSBSim: take-off (VTOL climb and transition for a "
                       "quadplane), then +15° roll and +10° pitch steps in FBWA. Details in sitl_report.json.")


def _zip_download(view: AgentRunView) -> None:
    """Every file of the run plus design.json in one zip; nothing when the run folder is gone.

    The record is checked now (cheap) so a broken run shows a warning; the archive itself (tens of MB) is built
    only when the button is clicked, on Streamlit's download thread, not on every rerun of the page."""
    if view.out_dir is None or not Path(view.out_dir).is_dir():
        return
    try:
        design_archive.design_record(view)
    except ValueError as exc:
        logger.warning("run %s cannot be zipped: %s", view.run_id, exc)
        st.warning(f"This run's files cannot be zipped: {exc}")
        return
    st.download_button(design_archive.ZIP_LABEL, data=lambda: design_archive.build_zip(view),
                       file_name=design_archive.zip_name(view), mime=design_archive.ZIP_MIME,
                       type="primary", key=f"dl-zip-{view.run_id}")


def _agent_tabs(view: AgentRunView, critic_tab: Any, report_tab: Any, files_tab: Any) -> None:
    with critic_tab:
        st.subheader("Final critic gates")
        st.dataframe(list(view.gates), hide_index=True, width="stretch")
        st.subheader("Critic iterations")
        st.dataframe([{"Iteration": c.get("iteration"), "Failed gates": ", ".join(c.get("failed", [])) or "-",
                       "Changes": "; ".join(c.get("changes", [])) or "-", "Summary": c.get("summary", "")}
                      for c in view.critic_log], hide_index=True, width="stretch")
        st.caption("Critic: " + ("the LLM proposes knob changes; " if view.used_llm else "")
                   + "bounded rule-table steps are applied and always win on failed gates.")
    with report_tab:
        text: Optional[str] = presenters.read_report(view.out_dir, view.report_path)
        if text is None:
            st.info("No report is available for this run.")
        else:
            st.markdown(text)  # LLM prose in it is defanged by report.render; raw HTML stays disabled
    with files_tab:
        _zip_download(view)
        for path in presenters.downloadable_files(view.out_dir) if view.out_dir else []:
            st.download_button(path.name, data=path.read_bytes(), file_name=path.name, key=f"dl-{view.run_id}-{path.name}")


def _render(view: AgentRunView) -> None:
    if view.status == agent_service.STATUS_CLARIFY:
        st.info(f"Clarification needed: {defang(view.message or '')}")
        return
    if view.status == agent_service.STATUS_ERROR or view.requirements is None:
        st.error(defang(view.message or GENERIC_FAILURE))
        return
    st.header(defang(view.requirements.mission_name))
    (st.success if view.status == agent_service.STATUS_CONVERGED else st.warning)(
        STATUS_TEXT.get(view.status, view.status))
    st.caption(f"Run `{view.run_id}` · files in `{view.out_dir}`")
    design_loop.render_assumptions_used(view)
    design_loop.render_kit_settings_note(view)
    for note in view.warnings:
        st.warning(note)
    if view.assumptions:
        with st.expander("Assumptions made while reading the request"):
            st.markdown("\n".join(f"- {inline(a)}" for a in view.assumptions))
    if view.sizing is None:
        st.error("Sizing did not converge for the final iteration; see the Critic and Report tabs.")
    else:
        _headline(view)
    design_loop.render_options(view, DB_PATH)
    design_loop.render_review_panel(view, OUTPUTS_ROOT, DB_PATH)
    chat_panel.render_chat(view, DB_PATH)
    tabs = st.tabs([*DESIGN_TABS, *AGENT_TABS])
    if view.sizing is not None:
        _design_tabs(view, *tabs[:len(DESIGN_TABS)])
    _agent_tabs(view, *tabs[len(DESIGN_TABS):])


def main() -> None:
    st.set_page_config(page_title="ISR-Forge", layout="wide")
    st.title("ISR-Forge")
    st.caption("Design a Group 1-3 ISR UAV from mission requirements. A critic loop checks static margin, L/D, "
               "packaging, group mass and energy reserve; all engineering numbers come from deterministic tools.")
    _init_state()
    _stage_options()
    _llm_info()
    render_assumptions_panel()
    render_fabrication_panel()
    if _description_form():
        _run_description(st.session_state["prompt"])
    st.subheader("Start from an example")
    st.selectbox("Start from an example", list(EXAMPLE_MISSIONS), key="preset", on_change=_load_preset,
                 label_visibility="collapsed")
    st.subheader("…or enter the requirements")
    _requirements_box()
    if st.session_state.pop(PENDING_DESIGN_KEY, False):
        _run_form({key: st.session_state[key] for key in presenters.FIELD_LABELS})
    design_loop.render_history()
    if "last_run" in st.session_state:
        _render(st.session_state["last_run"])


main()
