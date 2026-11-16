"""Designer in the loop (spec 6.4 and 8): option cards, the review panel with its linked re-run, and the session's
design history. Rendering only; every number shown comes from the explorer / solver, never from an LLM.

Session keys: `runs` (list of AgentRunView, newest last), `last_run` (the view in the results area),
`pending_change` (tuple[ChangeSet, source] | None), `explored_options` (run_id -> improvement options as JSON).
"""

import logging
from pathlib import Path
from typing import Any, Final, Literal, Optional, get_args

import streamlit as st
from pydantic import BaseModel

from src import pipeline
from src.agent.providers import LLMConfigError
from src.agent.report import defang, inline
from src.schemas.changes import (FIELD_LABELS, LABEL_MAX_LEN, AssumptionChanges, ChangeSet, KnobChanges,
                                 RequirementChanges)
from src.ui import agent_service
from src.ui.agent_service import AgentRunView, NoChangeError

logger = logging.getLogger(__name__)

RUNS_KEY: Final[str] = "runs"
LAST_RUN_KEY: Final[str] = "last_run"
PENDING_KEY: Final[str] = "pending_change"
EXPLORED_KEY: Final[str] = "explored_options"
REVIEW_PREFIX: Final[str] = "review_"
CLOSURE_MODE: Final[str] = "closure"
CHECKED_NOTE: Final[str] = "checked with sizing + catalog only"
EDITED_SUFFIX: Final[str] = " (edited)"
MINUS: Final[str] = "−"  # the typographic minus for mass savings
HISTORY_COLUMNS: Final[list[int]] = [4, 1]
RERUN_FAILURE: Final[str] = "The re-run failed unexpectedly; details are in the server log."
EXPLORE_FAILURE: Final[str] = "Exploring alternatives failed unexpectedly; details are in the server log."
SECTIONS: Final[dict[str, type[BaseModel]]] = {
    "requirements": RequirementChanges, "assumptions": AssumptionChanges, "knobs": KnobChanges,
}
Source = Literal["option", "chat", "manual"]


# ------------------------------------------------------------- session ---
def record_run(view: AgentRunView) -> None:
    """Show `view` and add it to the history (views without a run folder are shown but not listed)."""
    st.session_state[LAST_RUN_KEY] = view
    st.session_state[PENDING_KEY] = None
    if view.run_id is not None:
        st.session_state[RUNS_KEY] = [*st.session_state.get(RUNS_KEY, []), view]


def _review_key(section: str, field: str) -> str:
    return f"{REVIEW_PREFIX}{section}_{field}"


def _set_fields(model: BaseModel) -> dict[str, Any]:
    return {name: value for name, value in model.model_dump().items() if value is not None}


def _empty_change(label: str) -> dict[str, Any]:
    return {"label": label, "requirements": {}, "assumptions": {}, "knobs": {}}


def select_change(change: ChangeSet, source: Source) -> None:
    """on_click: load `change` into the review panel, seeding its editable widgets with the proposed values."""
    for key in [k for k in st.session_state if str(k).startswith(REVIEW_PREFIX)]:
        del st.session_state[key]
    for section in SECTIONS:
        for field, value in _set_fields(getattr(change, section)).items():
            st.session_state[_review_key(section, field)] = value
    st.session_state[PENDING_KEY] = (change, source)


def _cancel() -> None:
    st.session_state[PENDING_KEY] = None


def _open(view: AgentRunView) -> None:
    st.session_state[LAST_RUN_KEY] = view
    st.session_state[PENDING_KEY] = None


# ------------------------------------------------------------- options ---
def change_text(change: ChangeSet, view: AgentRunView) -> str:
    """'Endurance (h) 6 → 3.2' for every field the change actually moves."""
    rows = change.describe(view.requirements, view.design_assumptions) if view.requirements else []
    return "; ".join(f"{label} {old} → {new}" for label, old, new in rows) or inline(change.label)


def _as_number(text: str) -> Optional[float]:
    try:
        return float(text)
    except ValueError:
        return None


def _tried_row(label: str, old: str, new: str) -> str:
    """'Endurance (h): tried down to 0.5' for a scanned number; 'Airframe quadplane_vtol → fixed_wing' otherwise."""
    old_num, new_num = _as_number(old), _as_number(new)
    if old_num is None or new_num is None:
        return f"{label} {old} → {new}"
    return f"{label}: tried {'down' if new_num < old_num else 'up'} to {new}"


def _tried_text(change: ChangeSet, view: AgentRunView) -> str:
    """What a non-closing option tried, worded so it does not read as a proposal."""
    rows = change.describe(view.requirements, view.design_assumptions) if view.requirements else []
    return "; ".join(_tried_row(*row) for row in rows) or inline(change.label)


def _outcome_text(result: dict[str, Any]) -> str:
    parts: list[str] = [f"MTOW {result['mtow_kg']:.1f} kg", f"Group {result['dod_group']}"]
    if result.get("battery_mass_kg"):
        parts.append(f"battery {result['battery_mass_kg']:.1f} kg")
    if result.get("fuel_mass_kg"):
        parts.append(f"fuel {result['fuel_mass_kg']:.1f} kg")
    return " · ".join(parts)


def saving_text(mtow_saved_kg: float) -> str:
    """'MTOW − 3.1 kg' for a lighter design, 'MTOW + 0.1 kg' when the step makes it heavier."""
    sign = MINUS if mtow_saved_kg >= 0 else "+"
    return f"MTOW {sign} {abs(mtow_saved_kg):.1f} kg"


def _closing_card(index: int, option: dict[str, Any], view: AgentRunView, saving: bool) -> None:
    change = ChangeSet.model_validate(option["change"])
    with st.container(border=True):
        text = change_text(change, view)
        if saving and option.get("mtow_saved_kg") is not None:
            drop = ", drops a DoD group" if option.get("group_drop") else ""
            st.markdown(f"**{text}**: {saving_text(option['mtow_saved_kg'])}{drop}")
        else:
            st.markdown(f"**{text}**")
        st.caption(f"{_outcome_text(option['result'])} — {CHECKED_NOTE}")
        st.button("Select", key=f"select_option_{index}", on_click=select_change, args=(change, "option"))


def _option_cards(options: tuple[dict[str, Any], ...], view: AgentRunView, saving: bool) -> None:
    """Closing options as cards (in the explorer's ranked order), then the rest compactly with their reason."""
    for index, option in enumerate(o for o in options if o.get("closes")):
        _closing_card(index, option, view, saving)
    others = [o for o in options if not o.get("closes")]
    if others:
        st.caption("Checked, does not close:")
        st.markdown("\n".join(
            f"- {_tried_text(ChangeSet.model_validate(o['change']), view)} — {inline(str(o['result']['reason']))}"
            for o in others))


def _explore(view: AgentRunView, db_path: Path) -> None:
    try:
        with st.spinner("Checking alternatives with sizing + catalog…"):
            options = agent_service.explore(view, db_path)
    except ValueError as exc:
        st.error(defang(str(exc)))
        return
    except Exception:  # a bug or I/O failure: log the traceback, keep the page usable
        logger.exception("improvement explorer failed")
        st.error(EXPLORE_FAILURE)
        return
    explored: dict[str, tuple[dict[str, Any], ...]] = st.session_state.get(EXPLORED_KEY, {})
    st.session_state[EXPLORED_KEY] = {**explored, view.run_id: tuple(o.to_json() for o in options)}


def render_options(view: AgentRunView, db_path: Path) -> None:
    """Closure cards for a design that did not close; on demand, improvement cards for a converged one."""
    if view.requirements is None:
        return
    if view.explorer_mode == CLOSURE_MODE and view.options:
        st.subheader("Options that would close the design")
        if view.options_explanation:
            st.markdown(defang(view.options_explanation))
        _option_cards(view.options, view, saving=False)
    if view.status != agent_service.STATUS_CONVERGED:
        return
    if st.button("Explore alternatives", key="explore_alternatives",
                 help="What would make this design lighter (sizing + catalog checks, no full run)"):
        _explore(view, db_path)
    improved: Optional[tuple[dict[str, Any], ...]] = st.session_state.get(EXPLORED_KEY, {}).get(view.run_id)
    if improved is not None:
        st.subheader("Alternatives")
        if not improved:
            st.info("No alternative within the explored ranges makes this design lighter.")
        _option_cards(improved, view, saving=True)


# -------------------------------------------------------------- review ---
def _choices(section: str, field: str) -> Optional[tuple[Any, ...]]:
    """The allowed values of a Literal field (e.g. propulsion type), or None for a free number / flag."""
    annotation = SECTIONS[section].model_fields[field].annotation
    literal = next((arg for arg in get_args(annotation) if get_args(arg)), None)
    return get_args(literal) if literal is not None else None


def _edit_widget(section: str, field: str, value: Any) -> None:
    key, label = _review_key(section, field), FIELD_LABELS[field]
    choices = _choices(section, field)
    if isinstance(value, bool):
        st.checkbox(label, key=key)
    elif choices is not None:
        st.selectbox(label, choices, key=key)
    else:
        st.number_input(label, key=key, format="%g")


def _old_value(change: ChangeSet, section: str, field: str, value: Any, view: AgentRunView) -> str:
    """The run's current value of one changed field, formatted by ChangeSet.describe."""
    single = ChangeSet.model_validate({**_empty_change(change.label), section: {field: value}})
    rows = single.describe(view.requirements, view.design_assumptions) if view.requirements else []
    return rows[0][1] if rows else "unchanged"


def _review_rows(change: ChangeSet, view: AgentRunView) -> None:
    for section in SECTIONS:
        for field, value in _set_fields(getattr(change, section)).items():
            old_col, new_col = st.columns(2)
            old_col.markdown(f"{FIELD_LABELS[field]}: **{_old_value(change, section, field, value, view)}** →")
            with new_col:
                _edit_widget(section, field, value)


def _edited_change(change: ChangeSet) -> ChangeSet:
    """The pending change with the review panel's (possibly edited) values; raises ValidationError on bad input."""
    sections = {section: {field: st.session_state[_review_key(section, field)]
                          for field in _set_fields(getattr(change, section))} for section in SECTIONS}
    edited = ChangeSet.model_validate({**_empty_change(change.label), **sections})
    if edited == change:
        return change
    label = change.label[:LABEL_MAX_LEN - len(EDITED_SUFFIX)] + EDITED_SUFFIX
    return edited.model_copy(update={"label": label})


def _rerun(view: AgentRunView, change: ChangeSet, source: Source, outputs_root: Path, db_path: Path) -> None:
    do_aero: bool = st.session_state["with_aero_cad"]
    with st.status("Re-running with the change…", expanded=True) as status:
        try:
            new_view = agent_service.rerun(view, _edited_change(change), source, outputs_root, db_path,
                                           with_aero_cad=do_aero, with_sitl=do_aero and st.session_state["with_sitl"],
                                           progress=status.write)
        except (NoChangeError, ValueError, pipeline.PipelineError, LLMConfigError) as exc:
            # ValueError also covers pydantic's out-of-bounds ValidationError
            status.update(label="Re-run not started", state="error")
            st.error(defang(str(exc)))
            return
        except Exception:  # a bug or I/O failure: log the traceback, keep the current run on screen
            logger.exception("re-run failed")
            status.update(label="Re-run failed", state="error")
            st.error(RERUN_FAILURE)
            return
        status.update(label=f"Run {new_view.run_id} finished", state="complete", expanded=False)
    record_run(new_view)
    # The results header, cards and this panel were already drawn for the parent run in this script pass;
    # start a fresh pass so the page shows the new run (and the panel closes, pending_change is now None).
    st.rerun()


def render_review_panel(view: AgentRunView, outputs_root: Path, db_path: Path) -> None:
    """Old → new for the selected change; the designer may edit the new values, then Re-run or Cancel."""
    pending: Optional[tuple[ChangeSet, Source]] = st.session_state.get(PENDING_KEY)
    if pending is None or view.requirements is None:
        return
    change, source = pending
    with st.container(border=True):
        st.subheader("Review the change")
        st.caption(f"Applied to run `{view.run_id}` · source: {source}. Edit the new values if needed.")
        _review_rows(change, view)
        run_col, cancel_col = st.columns(2)
        pressed: bool = run_col.button("Re-run", key="rerun", type="primary")
        cancel_col.button("Cancel", key="cancel_change", on_click=_cancel)
    if pressed:
        _rerun(view, change, source, outputs_root, db_path)


# ------------------------------------------------------------- history ---
def _change_label(view: AgentRunView) -> str:
    if view.lineage is None:
        return "initial run"
    lineage = view.lineage
    return f"{inline(lineage.change_set.label)} (from `{lineage.parent_run_id}`, {lineage.change_source})"


def render_history() -> None:
    """The session's runs, oldest first; any of them can be reopened in the results view."""
    runs: list[AgentRunView] = st.session_state.get(RUNS_KEY, [])
    if not runs:
        return
    with st.expander("Design history", expanded=len(runs) > 1):
        for view in runs:
            text_col, button_col = st.columns(HISTORY_COLUMNS)
            text_col.markdown(f"`{view.run_id}` · {view.status.replace('_', ' ')} · {_change_label(view)}")
            button_col.button("Open", key=f"open_run_{view.run_id}", on_click=_open, args=(view,))


def render_assumptions_used(view: AgentRunView) -> None:
    """The run's own design assumptions (not the panel's current values)."""
    changed = view.design_assumptions.non_default()
    if not changed:
        st.caption("Assumptions used: defaults")
        return
    st.caption("Assumptions used: " + ", ".join(
        f"{FIELD_LABELS.get(name, name)} {value:g} (default {default:g})" for name, (default, value) in changed.items()))
