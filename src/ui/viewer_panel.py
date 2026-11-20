"""The 3D tab: the run's self-contained viewer.html embedded in the page, plus a download button."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Final

import streamlit as st
import streamlit.components.v1 as components

from src.ui.agent_service import AgentRunView
from src.viewer.build import VIEWER_FILE

VIEWER_HEIGHT_PX: Final = 640
VIEWER_CACHE_ENTRIES: Final = 4  # a few runs' pages; each can be a few MB
DOWNLOAD_LABEL: Final = f"Download {VIEWER_FILE}"
NO_VIEWER_TEXT: Final = (
    "No 3D viewer for this run. The viewer is built by the aero/CAD stage, which needs the Docker image "
    "(`docker compose up ui`); runs made before the 3D viewer existed do not have one."
)


@st.cache_data(max_entries=VIEWER_CACHE_ENTRIES, show_spinner=False)
def read_viewer_page(path: str, mtime_ns: int, size: int) -> str:
    """The viewer page's text, cached per file version (path, mtime, size) so a rerun does not re-read it."""
    return Path(path).read_text(encoding="utf-8")


def _page_bytes(path: Path) -> Callable[[], bytes]:
    """Deferred download data: the page is read and sent only when the user clicks download, not on every rerun."""
    return path.read_bytes


def render_viewer(view: AgentRunView) -> None:
    """Embed the run's viewer.html and offer it as a download, or explain why there is none."""
    path = view.out_dir / VIEWER_FILE if view.out_dir is not None else None
    if path is None or not path.is_file():
        st.info(NO_VIEWER_TEXT)
        return
    try:
        stat = path.stat()
        html = read_viewer_page(str(path), stat.st_mtime_ns, stat.st_size)
    except (OSError, UnicodeError) as exc:
        st.warning(f"The 3D viewer could not be read: {exc}")
        return
    components.html(html, height=VIEWER_HEIGHT_PX)
    st.download_button(DOWNLOAD_LABEL, data=_page_bytes(path), file_name=VIEWER_FILE, mime="text/html",
                       key="download_viewer")
    st.caption(f"{VIEWER_FILE} is self-contained: open it offline in any browser or share it as is.")
