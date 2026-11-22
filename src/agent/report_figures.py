"""Figures for design_report.pdf, drawn from the run's own outputs: a three-view of the packed aircraft (from the 3D
viewer's display mesh, so it follows any airframe layout), the mission power profile, the drag polars and the SITL
attitude responses. Each returns PNG bytes, or None when its input is missing (no CAD, no SITL flight).

Uses matplotlib's object API on the Agg canvas, never pyplot: no global figure state, safe in a server process.
"""

import base64
import io
import json
import math
from pathlib import Path
from typing import Any, Final, Optional, Sequence

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure

from src.schemas.state import AeroStabilityOutput, SizingOutput

DPI: Final[int] = 150
WIDTH_IN: Final[float] = 6.5  # the PDF text width (A4 less margins)
SECONDS_PER_MINUTE: Final[float] = 60.0
POLAR_ALPHA_DEG: Final[tuple[float, float]] = (-4.0, 12.0)  # the report's VSPAERO sweep range
POLAR_POINTS: Final[int] = 60
LABEL_HEADROOM: Final[float] = 1.45  # power-profile y-limit over the peak: room for the rotated phase names
LABEL_OFFSET: Final[float] = 1.04
GROUP_STYLE: Final[dict[str, tuple[str, float]]] = {  # (face colour, alpha) per viewer part group
    "skin": ("#9db4c8", 0.35), "structure": ("#8a6d3b", 0.5), "component": ("#c0504d", 0.8)}
CG_MARKER: Final[dict[str, Any]] = {"marker": "+", "color": "black", "markersize": 10, "markeredgewidth": 1.5}
X_AXIS, Y_AXIS, Z_AXIS = 0, 1, 2  # CAD frame: x aft, y right, z up
VIEW_GAP: Final[float] = 0.35  # gridspec spacing between the views, as a fraction of the average axis size
MIN_VIEW_HEIGHT_IN: Final[float] = 0.8  # a flat aircraft's front and side views stay readable


def _png(fig: Figure) -> bytes:
    FigureCanvasAgg(fig)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=DPI, bbox_inches="tight")
    return buffer.getvalue()


def _triangles(part: dict[str, Any]) -> np.ndarray:
    """(n, 3, 3) triangle vertices of one viewer-model part (mesh_export encoding: float32 / uint32, base64)."""
    positions = np.frombuffer(base64.b64decode(part["positions_b64"]), dtype="<f4").reshape(-1, 3)
    indices = np.frombuffer(base64.b64decode(part["indices_b64"]), dtype="<u4").reshape(-1, 3)
    return positions[indices]


def _draw_view(ax: Any, parts: Sequence[tuple[str, np.ndarray]], cg: Sequence[float], u: int, v: int,
               title: str) -> None:
    for group, tris in parts:
        face, alpha = GROUP_STYLE.get(group, GROUP_STYLE["skin"])
        ax.add_collection(PolyCollection(tris[:, :, [u, v]], facecolors=face, alpha=alpha, edgecolors="none"))
    ax.plot(cg[u], cg[v], **CG_MARKER)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)


def three_view(viewer_model: Optional[Path]) -> Optional[bytes]:
    """A drawing-sheet three-view of every display-mesh part, CG marked: the top view (nose up) across the full
    width, the front and side views below it, each panel sized from the aircraft's own extents at one scale."""
    if viewer_model is None or not viewer_model.is_file():
        return None
    model: dict[str, Any] = json.loads(viewer_model.read_text())
    parts = [(p.get("group", "skin"), _triangles(p)) for p in model["parts"]]
    points = np.concatenate([tris.reshape(-1, 3) for _, tris in parts])
    length, span, height = np.ptp(points, axis=0)
    top_height_in: float = WIDTH_IN * length / span
    lower_height_in: float = max(WIDTH_IN * height / (span + length), MIN_VIEW_HEIGHT_IN)
    fig = Figure(figsize=(WIDTH_IN, top_height_in + lower_height_in + 1.0))
    grid = fig.add_gridspec(2, 2, height_ratios=(top_height_in, lower_height_in), width_ratios=(span, length),
                            hspace=VIEW_GAP, wspace=VIEW_GAP)
    top = fig.add_subplot(grid[0, :])
    _draw_view(top, parts, model["cg_m"], Y_AXIS, X_AXIS, "Top (nose up)")
    top.invert_yaxis()
    _draw_view(fig.add_subplot(grid[1, 0]), parts, model["cg_m"], Y_AXIS, Z_AXIS, "Front")
    _draw_view(fig.add_subplot(grid[1, 1]), parts, model["cg_m"], X_AXIS, Z_AXIS, "Side (nose left)")
    fig.suptitle("+ centre of gravity · axes in metres", fontsize=8, y=0.02)
    return _png(fig)


def power_profile(sizing: SizingOutput) -> bytes:
    """Total and propulsive electrical power through the mission, one step per phase, against elapsed time."""
    edges = np.concatenate(([0.0], np.cumsum([p.duration_s for p in sizing.phases]))) / SECONDS_PER_MINUTE
    peak: float = max(p.total_power_w for p in sizing.phases)
    fig = Figure(figsize=(WIDTH_IN, WIDTH_IN * 0.42))
    ax = fig.add_subplot()
    ax.stairs([p.total_power_w for p in sizing.phases], edges, label="Total (incl. payload and avionics)")
    ax.stairs([p.propulsive_power_w for p in sizing.phases], edges, label="Propulsive", linestyle="--")
    for phase, start, end in zip(sizing.phases, edges[:-1], edges[1:]):
        ax.text((start + end) / 2.0, peak * LABEL_OFFSET, phase.name.replace("_", " "), ha="center", va="bottom",
                fontsize=7, rotation=90)
    ax.set_ylim(0.0, peak * LABEL_HEADROOM)
    ax.set_xlabel("Mission time (min)")
    ax.set_ylabel("Power (W)")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    return _png(fig)


def drag_polar(aero: AeroStabilityOutput, sizing: SizingOutput) -> bytes:
    """VSPAERO and sizing drag polars (CL vs CD, L/D vs CL) with the sizing cruise and loiter CLs marked."""
    cl = aero.cl0 + aero.cl_alpha_per_rad * np.radians(np.linspace(*POLAR_ALPHA_DEG, POLAR_POINTS))
    k_sizing: float = 1.0 / (math.pi * sizing.oswald_e * sizing.aspect_ratio)
    curves = (("VSPAERO", aero.cd0, aero.induced_drag_factor or k_sizing, "-"), ("Sizing", sizing.cd0, k_sizing, "--"))
    fig = Figure(figsize=(WIDTH_IN, WIDTH_IN * 0.42))
    ax_polar, ax_ld = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)
    for label, cd0, k, style in curves:
        cd = cd0 + k * cl**2
        ax_polar.plot(cd, cl, style, label=label)
        ax_ld.plot(cl, cl / cd, style, label=label)
    for name, value in (("cruise", sizing.cl_cruise), ("loiter", sizing.cl_loiter)):
        if value is not None:
            ax_ld.axvline(value, color="grey", linewidth=0.8)
            ax_ld.text(value, ax_ld.get_ylim()[0], f" {name}", fontsize=7, rotation=90, va="bottom")
    ax_polar.set(xlabel="CD", ylabel="CL")
    ax_ld.set(xlabel="CL", ylabel="L/D")
    for ax in (ax_polar, ax_ld):
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    return _png(fig)


def _series(telemetry: Sequence[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([sample[key] for sample in telemetry], dtype=float)


def step_responses(sitl_report: Path) -> Optional[bytes]:
    """Roll and pitch against their commands through the shipped SITL flight (nominal plant)."""
    telemetry: list[dict[str, Any]] = json.loads(sitl_report.read_text()).get("telemetry", [])
    if not telemetry:
        return None
    t = _series(telemetry, "t_s")
    fig = Figure(figsize=(WIDTH_IN, WIDTH_IN * 0.55))
    axes = [fig.add_subplot(2, 1, slot + 1) for slot in range(2)]
    for ax, axis in zip(axes, ("roll", "pitch")):
        ax.plot(t, _series(telemetry, f"nav_{axis}_deg"), "--", color="grey", linewidth=1.0, label="Commanded")
        ax.plot(t, _series(telemetry, f"{axis}_deg"), linewidth=1.0, label="Flown")
        ax.set_ylabel(f"{axis.capitalize()} (deg)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper left")
    axes[-1].set_xlabel("Time (s)", fontsize=8)
    fig.tight_layout()
    return _png(fig)
