"""Display-mesh export: tessellation, the viewer_model.json encoding, and failure handling."""

import base64
import json
import logging
import types
from pathlib import Path

import numpy as np
import pytest

build123d = pytest.importorskip("build123d")

from src.pipeline import design_warnings  # noqa: E402
from src.schemas.fabrication import DEFAULT_FABRICATION  # noqa: E402
from src.schemas.state import CADArtifacts  # noqa: E402
from src.tools.aero_cad import mesh_export  # noqa: E402
from src.tools.fabrication.kit_export import carbon_solids, ply_solids  # noqa: E402
from src.tools.fabrication.kit_mass import kit_mass_items  # noqa: E402
from tests.test_fabrication_kit import _build  # noqa: E402


def _box(size: float = 0.1) -> "build123d.Solid":
    return build123d.Solid.make_box(size, size, size)


def _fake_result(component_names: tuple[str, ...] = ("battery",), ribs: int = 3, spars: int = 2) -> types.SimpleNamespace:
    comps = tuple(types.SimpleNamespace(name=n, mass_kg=1.0, solid=lambda: _box()) for n in component_names)
    structure = types.SimpleNamespace(ribs=tuple(_box() for _ in range(ribs)), spars=tuple(_box() for _ in range(spars)))
    return types.SimpleNamespace(oml=_box(0.5), structure=structure, components=comps, kit=None,
                                 settings=DEFAULT_FABRICATION)


def test_a_box_tessellates_inside_its_bounds() -> None:
    part = mesh_export.tessellate(build123d.Solid.make_box(1.0, 0.5, 0.25), "battery", "component", 2.0)
    xyz = np.array(part.positions).reshape(-1, 3)
    assert part.name == "battery" and part.mass_kg == 2.0 and len(part.indices) % 3 == 0
    assert xyz.min(axis=0) == pytest.approx([0, 0, 0], abs=1e-6)
    assert xyz.max(axis=0) == pytest.approx([1, 0.5, 0.25], abs=1e-6)
    assert max(part.indices) < len(xyz)


def test_tessellation_matches_build123d_on_a_many_faced_solid() -> None:
    """The indexed triangle read gives exactly build123d's mesh (its iterator is ~10 ms per face on OCP)."""
    plate = build123d.Rectangle(0.2, 0.1) - build123d.Pos(0.05, 0) * build123d.Circle(0.01)
    solid = build123d.extrude(plate.faces()[0], amount=0.003).solids()[0]
    vertices, triangles = build123d.Solid(solid.wrapped).tessellate(mesh_export.LINEAR_TOLERANCE_M,
                                                                    mesh_export.ANGULAR_TOLERANCE_RAD)
    part = mesh_export.tessellate(solid, "plate", "structure", None)
    assert part.positions == tuple(c for v in vertices for c in (v.X, v.Y, v.Z))
    assert part.indices == tuple(i for tri in triangles for i in tri)


def test_the_model_file_round_trips(tmp_path: Path) -> None:
    part = mesh_export.tessellate(build123d.Solid.make_cylinder(0.1, 0.4), "tank", "component", 1.5)
    path = mesh_export.write_viewer_model([part], (0.5, 0.0, 0.01), 17.5, tmp_path / "viewer_model.json")
    data = json.loads(path.read_text())
    assert data["frame"] == "x_aft_y_right_z_up" and data["cg_m"] == [0.5, 0.0, 0.01] and data["mass_kg"] == 17.5
    positions = np.frombuffer(base64.b64decode(data["parts"][0]["positions_b64"]), "<f4")
    assert positions == pytest.approx(np.round(part.positions, 4), abs=1e-6)
    indices = np.frombuffer(base64.b64decode(data["parts"][0]["indices_b64"]), "<u4")
    assert tuple(indices) == part.indices


def test_repeated_component_names_get_suffixes() -> None:
    parts = mesh_export.model_parts(_fake_result(("generator", "generator", "battery")))
    assert [p.name for p in parts if p.group == "component"] == ["generator", "generator #2", "battery"]
    assert parts[0].group == "skin" and parts[0].mass_kg is None


def test_shell_structure_is_one_merged_part() -> None:
    parts = mesh_export.model_parts(_fake_result())
    structure = [p for p in parts if p.group == "structure"]
    assert len(structure) == 1 and structure[0].mass_kg is None and len(structure[0].indices) > 0


@pytest.fixture(scope="module")
def g1_kit(designs: dict, g1_kit_inputs: tuple) -> object:
    return _build(designs, g1_kit_inputs, "g1")


def _bounds(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return points.min(axis=0), points.max(axis=0)


def test_viewer_structure_part_comes_from_the_kit(g1_kit: object) -> None:
    result = _fake_result(spars=0, ribs=0)
    result = types.SimpleNamespace(**{**vars(result), "structure": None, "kit": g1_kit})
    parts = mesh_export.model_parts(result)
    structure = {p.name: p for p in parts if p.group == "structure"}
    assert set(structure) == {"structure (ply)", "structure (carbon)"}
    items = {i.name: i.mass_kg for i in kit_mass_items(g1_kit, DEFAULT_FABRICATION)}
    ply_kg = sum(items[p.label] for p in g1_kit.parts)
    carbon_kg = sum(items[t.label] for t in g1_kit.tubes) + g1_kit.overwrap.mass_kg
    assert structure["structure (ply)"].mass_kg == pytest.approx(ply_kg)
    assert structure["structure (carbon)"].mass_kg == pytest.approx(carbon_kg)
    for name, solids in (("structure (ply)", ply_solids(g1_kit, DEFAULT_FABRICATION)),
                         ("structure (carbon)", carbon_solids(g1_kit))):
        mesh = np.array(structure[name].positions).reshape(-1, 3)
        box = build123d.Compound(solids).bounding_box()
        assert len(structure[name].indices) > 0
        assert np.allclose(_bounds(mesh), ([box.min.X, box.min.Y, box.min.Z], [box.max.X, box.max.Y, box.max.Z]),
                           atol=mesh_export.LINEAR_TOLERANCE_M)


def test_export_failure_returns_none_and_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                              caplog: pytest.LogCaptureFixture) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("tessellation failed")

    monkeypatch.setattr(mesh_export, "tessellate", boom)
    with caplog.at_level(logging.ERROR):
        assert mesh_export.export_viewer_model(_fake_result(), tmp_path) is None
    assert any(r.levelno == logging.ERROR for r in caplog.records)


def test_failed_export_removes_a_stale_viewer_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stale = tmp_path / "viewer_model.json"
    stale.write_text("{}")

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("tessellation failed")

    monkeypatch.setattr(mesh_export, "tessellate", boom)
    assert mesh_export.export_viewer_model(_fake_result(), tmp_path) is None
    assert not stale.exists()


def test_design_warnings_flag_a_missing_viewer_model() -> None:
    cad = CADArtifacts(oml_vsp3_path="a", oml_step_path="b", assembly_step_path="c", x_cg_m=1.0, static_margin=0.1,
                       collision_count=0, as_built_mass_kg=10.0, ixx_kgm2=1.0, iyy_kgm2=1.0, izz_kgm2=1.0,
                       ixz_kgm2=0.0, viewer_model_path=None)
    sizing = types.SimpleNamespace(mtow_kg=10.0, assumed_ld_cruise=10.0)
    assert any("3D viewer" in w for w in design_warnings(sizing, None, cad))
