"""Lightweight display mesh of the packed aircraft for the 3D viewer (`viewer_model.json`).

Frame is the CAD frame: SI metres, x aft from the nose, y right, z up. Positions are float32 rounded to
0.1 mm and indices uint32, both little-endian and base64 encoded so the file stays compact.
"""

import base64
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Optional, Sequence

import build123d as bd
import numpy as np

if TYPE_CHECKING:  # type-only: cad_packer imports this module, so a runtime import would be circular
    from src.tools.aero_cad.cad_packer import PackingResult

logger = logging.getLogger(__name__)

VIEWER_MODEL_FILE: Final[str] = "viewer_model.json"
FRAME: Final[str] = "x_aft_y_right_z_up"
LINEAR_TOLERANCE_M: Final[float] = 0.002  # 2 mm chord deviation: smooth skin at ~1 m scale, small file
ANGULAR_TOLERANCE_RAD: Final[float] = 0.3
POSITION_DECIMALS: Final[int] = 4  # 0.1 mm

PartGroup = Literal["skin", "structure", "component"]


@dataclass(frozen=True)
class MeshPart:
    name: str
    group: PartGroup
    mass_kg: Optional[float]
    positions: tuple[float, ...]  # flat x, y, z per vertex
    indices: tuple[int, ...]  # flat vertex triples


def tessellate(solid: bd.Shape, name: str, group: PartGroup, mass_kg: Optional[float]) -> MeshPart:
    vertices, triangles = solid.tessellate(LINEAR_TOLERANCE_M, ANGULAR_TOLERANCE_RAD)
    return MeshPart(
        name=name, group=group, mass_kg=mass_kg,
        positions=tuple(float(c) for v in vertices for c in (v.X, v.Y, v.Z)),
        indices=tuple(int(i) for tri in triangles for i in tri),
    )


def _unique_names(names: Sequence[str]) -> tuple[str, ...]:
    """Second and later repeats of a name become 'name #2', 'name #3'."""
    seen: dict[str, int] = {}
    unique: list[str] = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        unique.append(name if seen[name] == 1 else f"{name} #{seen[name]}")
    return tuple(unique)


def model_parts(result: "PackingResult") -> tuple[MeshPart, ...]:
    """Skin, one merged structure part (ribs and spars), then one part per component."""
    structure = bd.Compound([*result.structure.ribs, *result.structure.spars])
    names = _unique_names([c.name for c in result.components])
    return (
        tessellate(result.oml, "skin", "skin", None),
        tessellate(structure, "structure", "structure", None),
        *(tessellate(c.solid(), n, "component", c.mass_kg) for c, n in zip(result.components, names)),
    )


def _encode(values: Sequence[float] | Sequence[int], dtype: str) -> str:
    return base64.b64encode(np.asarray(values, dtype=dtype).tobytes()).decode("ascii")


def _part_json(part: MeshPart) -> dict[str, object]:
    rounded: np.ndarray = np.round(np.asarray(part.positions, dtype=np.float64), POSITION_DECIMALS)
    return {"name": part.name, "group": part.group, "mass_kg": part.mass_kg,
            "positions_b64": _encode(rounded, "<f4"), "indices_b64": _encode(part.indices, "<u4")}


def write_viewer_model(parts: Sequence[MeshPart], cg_m: Sequence[float], mass_kg: float, path: Path) -> Path:
    payload: dict[str, object] = {"frame": FRAME, "cg_m": [float(c) for c in cg_m], "mass_kg": float(mass_kg),
                                  "parts": [_part_json(p) for p in parts]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def export_viewer_model(result: "PackingResult", out_dir: Path) -> Optional[Path]:
    """Write `viewer_model.json`; never raises - a failure is logged and reported as None."""
    try:
        return write_viewer_model(model_parts(result), result.mass.cg_m, result.mass.mass_kg,
                                  out_dir / VIEWER_MODEL_FILE)
    except Exception:
        logger.exception("viewer mesh export failed; the run continues without a 3D viewer")
        return None
