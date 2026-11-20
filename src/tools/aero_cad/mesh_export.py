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
from OCP.BRep import BRep_Tool
from OCP.TopAbs import TopAbs_Orientation
from OCP.TopLoc import TopLoc_Location

from src.tools.fabrication.kit_export import carbon_solids, ply_solids
from src.tools.fabrication.kit_mass import kit_mass_items

if TYPE_CHECKING:  # type-only: cad_packer imports this module, so a runtime import would be circular
    from src.tools.aero_cad.cad_packer import PackingResult

logger = logging.getLogger(__name__)

VIEWER_MODEL_FILE: Final[str] = "viewer_model.json"
FRAME: Final[str] = "x_aft_y_right_z_up"
LINEAR_TOLERANCE_M: Final[float] = 0.002  # 2 mm chord deviation: smooth skin at ~1 m scale, small file
ANGULAR_TOLERANCE_RAD: Final[float] = 0.3
POSITION_DECIMALS: Final[int] = 4  # 0.1 mm
SHELL_STRUCTURE_NAME: Final[str] = "structure"
PLY_STRUCTURE_NAME: Final[str] = "structure (ply)"  # every laser-cut sheet part
CARBON_STRUCTURE_NAME: Final[str] = "structure (carbon)"  # every stock tube and rod, plus the centreline overwrap

PartGroup = Literal["skin", "structure", "component"]


@dataclass(frozen=True)
class MeshPart:
    name: str
    group: PartGroup
    mass_kg: Optional[float]
    positions: tuple[float, ...]  # flat x, y, z per vertex
    indices: tuple[int, ...]  # flat vertex triples


def _triangulate(shape: bd.Shape) -> tuple[list[float], list[int]]:
    """Flat positions and indices of the shape's mesh, face by face, exactly as `Shape.tessellate` orders them, but
    reading triangles by index: iterating OCP's triangle array costs ~10 ms per face (a laser-cut rib has ~90)."""
    shape.mesh(LINEAR_TOLERANCE_M, ANGULAR_TOLERANCE_RAD)
    positions: list[float] = []
    indices: list[int] = []
    offset = 0
    for face in shape.faces():
        loc = TopLoc_Location()
        poly = BRep_Tool.Triangulation_s(face.wrapped, loc)
        trsf = loc.Transformation()
        for i in range(1, poly.NbNodes() + 1):
            node = poly.Node(i).Transformed(trsf)
            positions += (node.X(), node.Y(), node.Z())
        order = (1, 3, 2) if face.wrapped.Orientation() == TopAbs_Orientation.TopAbs_REVERSED else (1, 2, 3)
        for i in range(1, poly.NbTriangles() + 1):
            triangle = poly.Triangle(i)
            indices += (triangle.Value(k) + offset - 1 for k in order)
        offset += poly.NbNodes()
    return positions, indices


def tessellate(solid: bd.Shape, name: str, group: PartGroup, mass_kg: Optional[float]) -> MeshPart:
    positions, indices = _triangulate(solid)
    return MeshPart(name=name, group=group, mass_kg=mass_kg, positions=tuple(float(c) for c in positions),
                    indices=tuple(int(i) for i in indices))


def _unique_names(names: Sequence[str]) -> tuple[str, ...]:
    """Second and later repeats of a name become 'name #2', 'name #3'."""
    seen: dict[str, int] = {}
    unique: list[str] = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        unique.append(name if seen[name] == 1 else f"{name} #{seen[name]}")
    return tuple(unique)


def _kit_parts(result: "PackingResult") -> tuple[MeshPart, ...]:
    """The laser-cut kit as two structure parts, ply then carbon, each carrying its kit mass (overwrap in carbon)."""
    kit, settings = result.kit, result.settings
    masses = {item.name: item.mass_kg for item in kit_mass_items(kit, settings)}
    ply_kg = sum(masses[p.label] for p in kit.parts)
    carbon_kg = sum(masses[t.label] for t in kit.tubes) + kit.overwrap.mass_kg
    return (tessellate(bd.Compound(ply_solids(kit, settings)), PLY_STRUCTURE_NAME, "structure", ply_kg),
            tessellate(bd.Compound(carbon_solids(kit)), CARBON_STRUCTURE_NAME, "structure", carbon_kg))


def _structure_parts(result: "PackingResult") -> tuple[MeshPart, ...]:
    """The kit's ply and carbon parts, or one merged shell-model structure part (ribs and spars) without a kit."""
    if result.kit is not None:
        return _kit_parts(result)
    if result.structure is None:
        return ()
    structure = bd.Compound([*result.structure.ribs, *result.structure.spars])
    return (tessellate(structure, SHELL_STRUCTURE_NAME, "structure", None),)


def model_parts(result: "PackingResult") -> tuple[MeshPart, ...]:
    """Skin, the structure (kit: ply and carbon; shell model: one part), then one part per component."""
    names = _unique_names([c.name for c in result.components])
    return (
        tessellate(result.oml, "skin", "skin", None),
        *_structure_parts(result),
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
    target: Path = out_dir / VIEWER_MODEL_FILE
    try:
        # Drop any file from an earlier critic iteration first: a failed export must not leave stale geometry.
        target.unlink(missing_ok=True)
        return write_viewer_model(model_parts(result), result.mass.cg_m, result.mass.mass_kg, target)
    except Exception:
        logger.exception("viewer mesh export failed; the run continues without a 3D viewer")
        return None
