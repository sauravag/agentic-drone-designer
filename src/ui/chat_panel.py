"""Design chat panel (spec 7.1, 7.3): the designer asks, the assistant checks what-ifs with the solver and may
propose a change. A proposal never runs by itself: its button only fills the review panel (design_loop).

Session key: `chat_<run_id>` (tuple of ChatMessage, designer and agent turns, oldest first). Every reply is
also saved to the run folder as `design_chat.json`. All LLM-written (and designer-typed) text is rendered
through `report.defang` / `inline`; every number in the what-if table comes from the evaluator.
"""

import json
import logging
from pathlib import Path
from typing import Any, Final, Optional

import streamlit as st

from src.agent.design_chat import ChatMessage
from src.agent.llm import LLMError
from src.agent.providers import LLMConfigError
from src.agent.report import defang, inline
from src.ui import agent_service, design_loop
from src.ui.agent_service import AgentRunView

logger = logging.getLogger(__name__)

CHAT_INPUT_KEY: Final[str] = "design_chat"
HISTORY_PREFIX: Final[str] = "chat_"
PROPOSAL_PREFIX: Final[str] = "use_proposal_"
CHAT_FILE: Final[str] = "design_chat.json"
AUTO_OPEN_MODES: Final[tuple[str, ...]] = ("closure", "airframe")  # non-convergence opens the chat (spec 7.1)
MAX_MESSAGE_CHARS: Final[int] = 2000
ROLE_AVATARS: Final[dict[str, str]] = {"designer": "user", "agent": "assistant"}
PLACEHOLDER: Final[str] = "Ask about this design, e.g. what would close it while staying electric?"
NO_LLM_CAPTION: Final[str] = "The design chat needs an LLM (the option cards work without one). {problem}"
LLM_FAILURE: Final[str] = "The design assistant could not answer: {error}"
CHAT_FAILURE: Final[str] = "The design chat failed unexpectedly; details are in the server log."
SAVE_FAILURE: Final[str] = "The chat could not be saved to the run folder; details are in the server log."
CHECKED_NOTE: Final[str] = "what-ifs checked with sizing + catalog only"


# ------------------------------------------------------------- history ---
def _history_key(view: AgentRunView) -> str:
    return f"{HISTORY_PREFIX}{view.run_id}"


def _history(view: AgentRunView) -> tuple[ChatMessage, ...]:
    return st.session_state.get(_history_key(view), ())


def _message_json(message: ChatMessage) -> dict[str, object]:
    """role, text and what-if results (change, solver facts or error), plus the cost and audit fields."""
    return {**message.to_json(), "rounds": message.rounds, "stray_numbers": list(message.stray_numbers),
            "proposal": None if message.proposal is None else message.proposal.model_dump(mode="json")}


def _save(view: AgentRunView, history: tuple[ChatMessage, ...]) -> None:
    """Rewrite the run folder's design_chat.json (temp file + rename, so a reader never sees half a file)."""
    if view.out_dir is None:
        return
    target = Path(view.out_dir) / CHAT_FILE
    temp = target.with_suffix(".json.tmp")
    try:
        temp.write_text(json.dumps({"run_id": view.run_id, "messages": [_message_json(m) for m in history]},
                                   indent=2), encoding="utf-8")
        temp.replace(target)
    except OSError:
        logger.exception("saving %s failed", target)
        st.warning(SAVE_FAILURE)


# ---------------------------------------------------------------- reply ---
def _ask(view: AgentRunView, message: str, db_path: Path) -> None:
    """Get the assistant's answer; on success append both turns and save. Errors keep the history as it was."""
    history = _history(view)
    try:
        with st.spinner("Checking what-ifs with sizing + catalog…"):
            reply = agent_service.chat(view, history, message, db_path)
    except LLMError as exc:
        st.error(LLM_FAILURE.format(error=inline(str(exc))))
        return
    except (ValueError, LLMConfigError) as exc:
        st.error(defang(str(exc)))
        return
    except Exception:  # a bug or I/O failure: log the traceback, keep the page and the options usable
        logger.exception("design chat failed")
        st.error(CHAT_FAILURE)
        return
    updated = (*history, ChatMessage(role="designer", text=message), reply)
    st.session_state[_history_key(view)] = updated
    _save(view, updated)


# --------------------------------------------------------------- render ---
def _cell(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.1f}"


def _what_if_rows(message: ChatMessage, view: AgentRunView) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for what_if in message.what_ifs:
        result = what_if.result
        rows.append({
            "What-if": inline(what_if.change.label), "Change": design_loop.change_text(what_if.change, view),
            "Closes": "-" if result is None else ("yes" if result.closes else "no"),
            "MTOW (kg)": _cell(None if result is None else result.mtow_kg),
            "Group": "-" if result is None or result.dod_group is None else str(result.dod_group),
            "Note": inline(what_if.error or (result.reason if result else "")) or "-",
        })
    return rows


def _has_effect(message: ChatMessage, view: AgentRunView) -> bool:
    """A proposal worth reviewing: it moves at least one field of the shown run."""
    change = message.proposal
    if change is None or change.is_empty() or view.requirements is None:
        return False
    return bool(change.describe(view.requirements, view.design_assumptions))


def _agent_extras(message: ChatMessage, ordinal: int, view: AgentRunView) -> None:
    """Under an answer: the solver's what-if table, stray-number warning, cost, and the proposal button."""
    if message.what_ifs:
        st.dataframe(_what_if_rows(message, view), hide_index=True)
    if message.stray_numbers:
        st.caption("Warning: numbers in this answer that no solver result backs: "
                   + ", ".join(inline(n) for n in message.stray_numbers))
    st.caption(f"{message.rounds} LLM round{'s' if message.rounds != 1 else ''} · {CHECKED_NOTE}")
    if message.proposal is not None and _has_effect(message, view):
        st.markdown(f"Proposed: **{design_loop.change_text(message.proposal, view)}**")
        st.button("Review this change", key=f"{PROPOSAL_PREFIX}{ordinal}", on_click=design_loop.select_change,
                  args=(message.proposal, "chat"), help="Fills the review panel; nothing runs until you press Re-run")


def _render_history(view: AgentRunView) -> None:
    agent_ordinal = 0
    for message in _history(view):
        with st.chat_message(ROLE_AVATARS[message.role]):
            st.markdown(defang(message.text))
            if message.role == "agent":
                _agent_extras(message, agent_ordinal, view)
                agent_ordinal += 1


def render_chat(view: AgentRunView, db_path: Path) -> None:
    """The chat below the options; opens by itself when the design did not close (spec 7.1)."""
    if view.requirements is None or view.run_id is None:
        return
    with st.expander("Design chat", expanded=view.explorer_mode in AUTO_OPEN_MODES):
        problem: Optional[str] = agent_service.llm_problem()
        if problem is not None:
            st.caption(NO_LLM_CAPTION.format(problem=defang(problem)))
        history_box = st.container()  # drawn after the input is handled, so a new reply shows at once
        message: Optional[str] = st.chat_input(PLACEHOLDER, key=CHAT_INPUT_KEY, disabled=problem is not None,
                                               max_chars=MAX_MESSAGE_CHARS)
        if message and message.strip():
            _ask(view, message.strip(), db_path)
        with history_box:
            _render_history(view)
