"""The 3D tab: the run's self-contained viewer.html embedded in the page, plus a download button."""
from __future__ import annotations

from typing import Final

import streamlit as st
import streamlit.components.v1 as components

from src.ui.agent_service import AgentRunView
from src.viewer.build import VIEWER_FILE

VIEWER_HEIGHT_PX: Final = 640
DOWNLOAD_LABEL: Final = f"Download {VIEWER_FILE}"
NO_VIEWER_TEXT: Final = (
    "No 3D viewer for this run. The viewer is built by the aero/CAD stage, which needs the Docker image "
    "(`docker compose up ui`); runs made before the 3D viewer existed do not have one."
)


def render_viewer(view: AgentRunView) -> None:
    """Embed the run's viewer.html and offer it as a download, or explain why there is none."""
    path = view.out_dir / VIEWER_FILE if view.out_dir is not None else None
    if path is None or not path.is_file():
        st.info(NO_VIEWER_TEXT)
        return
    try:
        html = path.read_text(encoding="utf-8")
    except OSError as exc:
        st.warning(f"The 3D viewer could not be read: {exc}")
        return
    components.html(html, height=VIEWER_HEIGHT_PX)
    st.download_button(DOWNLOAD_LABEL, data=html, file_name=VIEWER_FILE, mime="text/html", key="download_viewer")
    st.caption("viewer.html is self-contained: open it offline in any browser or share it as is.")
