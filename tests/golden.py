"""Golden snapshots of the reference designs' outputs, re-baselined once in phase 2 (2026-11-23: explicit tail/lift
boom tubes, boom drag and a stopped-rotor term) after the phase-1 airframe-layout refactor.

`snapshot(case)` returns one canonical text per output key; `tests/test_layout_golden.py` compares each with the file
stored under `tests/data/golden/layouts_phase2/<case>/<key>.json`. A changed snapshot means the code is wrong.
"""

import dataclasses
import json
import tempfile
from pathlib import Path
from typing import Any, Final, Optional

from pydantic import BaseModel

from src.tools.aero_cad import components
from src.tools.aero_cad.layout import AircraftLayout, layout_from_sizing
from src.tools.database.matcher import select_bom
from src.tools.database.seed_data import seed_database
from src.tools.flight_control.flight_model import build_flight_model
from src.tools.flight_control.jsbsim_exporter import export_jsbsim
from src.tools.flight_control.param_generator import generate_params
from src.tools.sizing.mission_solver import component_needs, size_aircraft
from tests.reference_designs import CASES

GOLDEN_DIR: Final[Path] = Path(__file__).parent / "data" / "golden" / "layouts_phase2"
SIG_DIGITS: Final[int] = 12  # `canonical` default
SNAPSHOT_DIGITS: Final[int] = 9  # snapshots: the last digits of a float differ between macOS and Linux libm/BLAS
KIT_DECIMALS: Final[int] = 6  # micron: the CAD kernel differs from the 12th digit between macOS and Linux
# SizingOutput keys added after the stored snapshots were taken (left out of the comparison until the next deliberate
# re-baseline). Empty since the phase-2 re-baseline stored every key.
NEW_SIZING_KEYS: Final[frozenset[str]] = frozenset({"gear_kind"})
SIZING_CASES: Final[tuple[str, ...]] = ("g1", "g2", "g3")
DESIGN_CASES: Final[tuple[str, ...]] = ("g1", "g2")  # the cases with a saved flight-control design
VSP_PARMS: Final[tuple[tuple[str, str], ...]] = (
    ("Span", "XSec_1"), ("Root_Chord", "XSec_1"), ("Tip_Chord", "XSec_1"), ("Sweep", "XSec_1"),
    ("Dihedral", "XSec_1"), ("X_Rel_Location", "XForm"), ("Z_Rel_Location", "XForm"))
LAYOUT_FIELDS: Final[tuple[str, ...]] = (
    "fuselage_length_m", "fuselage_diameter_m", "wing", "htail", "vtail", "tail_boom", "vtol_booms",
    "rotor_positions", "rotor_diameter_m", "airfoil", "wing_station")


def _round_float(value: float, decimals: Optional[int], digits: int) -> float:
    """`digits` significant digits, or fixed `decimals` (for CAD-kernel output)."""
    rounded = round(value, decimals) if decimals is not None else float(f"{value:.{digits - 1}e}")
    return rounded + 0.0  # -0.0 -> 0.0


def _plain(obj: Any, decimals: Optional[int], digits: int) -> Any:
    """Recursively reduce to JSON types: rounded floats, tuples as lists."""
    if isinstance(obj, BaseModel):
        return _plain(obj.model_dump(mode="json"), decimals, digits)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _plain(dataclasses.asdict(obj), decimals, digits)
    if isinstance(obj, dict):
        return {str(k): _plain(v, decimals, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v, decimals, digits) for v in obj]
    if isinstance(obj, float):
        return _round_float(obj, decimals, digits)
    return obj


def canonical(obj: object, decimals: Optional[int] = None, digits: int = SIG_DIGITS) -> str:
    """Deterministic JSON text of `obj`: sorted keys, floats rounded to 12 significant digits (or to `decimals`)."""
    return json.dumps(_plain(obj, decimals, digits), sort_keys=True, indent=1)


PLANFORM_FIELDS: Final[tuple[str, ...]] = (
    "x_le_root_m", "z_root_m", "semi_span_m", "root_chord_m", "tip_chord_m", "dihedral_deg", "is_symmetric")
BOOM_FIELDS: Final[tuple[str, ...]] = ("x_start_m", "length_m", "diameter_m", "y_m", "z_m")


def _fields(obj: Any, names: tuple[str, ...]) -> Optional[dict[str, Any]]:
    """`obj`'s `names` by attribute access (fields added later are ignored, renamed or removed ones raise)."""
    return None if obj is None else {name: getattr(obj, name) for name in names}


def layout_view(layout: AircraftLayout) -> dict[str, Any]:
    """The layout's geometry by attribute access and fixed field lists, so the snapshot survives a reshuffle of the
    dataclass fields and the addition of new ones."""
    view: dict[str, Any] = {name: getattr(layout, name) for name in LAYOUT_FIELDS}
    for name in ("wing", "htail", "vtail"):
        view[name] = _fields(view[name], PLANFORM_FIELDS)
    view["tail_boom"] = _fields(view["tail_boom"], BOOM_FIELDS)
    view["vtol_booms"] = [_fields(b, BOOM_FIELDS) for b in view["vtol_booms"]]
    return view


def _sizing(case: str) -> Any:
    return size_aircraft(CASES[case])


def _sizing_text(case: str) -> str:
    return canonical(_sizing(case).model_dump(mode="json", exclude=NEW_SIZING_KEYS), digits=SNAPSHOT_DIGITS)


def _layout_text(case: str) -> str:
    return canonical(layout_view(layout_from_sizing(_sizing(case))), digits=SNAPSHOT_DIGITS)


def _saved_design(case: str) -> Any:
    from tests.test_flight_control import load_design  # lazy: pulls in the whole flight-control test module
    return load_design(case)


def _params_text(case: str) -> str:
    _, sizing, bom, _, _ = inputs = _saved_design(case)
    return canonical(generate_params(build_flight_model(*inputs), sizing, bom), digits=SNAPSHOT_DIGITS)


def _jsbsim_text(case: str) -> str:
    model = build_flight_model(*_saved_design(case))
    with tempfile.TemporaryDirectory() as tmp:
        return canonical(export_jsbsim(model, Path(tmp)).xml_path.read_text(), digits=SNAPSHOT_DIGITS)


def _kit_text(case: str) -> str:
    from src.tools.fabrication.build_kit import build_kit
    from src.schemas.fabrication import DEFAULT_FABRICATION
    from tests.cad_helpers import _fitted_layout
    from tests.test_fabrication_fuselage import _real_components
    from src.tools.aero_cad.airfoil_geometry import load_airfoil

    with tempfile.TemporaryDirectory() as tmp:
        req, sizing = CASES[case], _sizing(case)
        bom = select_bom(seed_database(Path(tmp) / "cots_catalog.db"), component_needs(sizing, req))
    layout = _fitted_layout(sizing, bom)
    placed = (*_real_components({case: (sizing, bom)}, case, layout), *components.vtol_motors(bom, layout))
    kit = build_kit(sizing, layout, load_airfoil(layout.airfoil), placed, DEFAULT_FABRICATION)
    parts = [{"name": p.label, "role": p.family, "quantity": p.quantity,
              "vertices": sorted((round(v.X, KIT_DECIMALS), round(v.Y, KIT_DECIMALS), round(v.Z, KIT_DECIMALS))
                               for v in p.outline.vertices()),  # kernel vertex order differs per platform
              "position": tuple(p.location.position), "orientation": tuple(p.location.orientation)}
             for p in kit.parts]
    tubes = [{"name": t.label, "role": t.role, "length_m": t.length_m, "start_m": t.start_m, "axis": t.axis}
             for t in kit.tubes]
    return canonical({"parts": parts, "tubes": tubes}, KIT_DECIMALS)


def _vsp_text(case: str) -> str:
    import openvsp as vsp
    from src.tools.aero_cad.openvsp_builder import build_model

    layout = layout_from_sizing(_sizing(case))
    with tempfile.TemporaryDirectory() as tmp:
        build_model(layout, Path(tmp))
    comps = []
    for gid in vsp.FindGeoms():
        is_wing = vsp.GetGeomTypeName(gid) == "Wing"  # only wings have the XSec_1 planform parameters
        parms = {f"{group}.{name}": vsp.GetParmVal(vsp.FindParm(gid, name, group))
                 for name, group in VSP_PARMS if is_wing or group == "XForm"}
        comps.append({"name": vsp.GetGeomName(gid), "type": vsp.GetGeomTypeName(gid), "parms": parms})
    return canonical(comps, digits=SNAPSHOT_DIGITS)


def snapshot(case: str) -> dict[str, str]:
    """Canonical text of every output key available for `case` (`vsp_geometry` only where OpenVSP imports)."""
    out: dict[str, str] = {}
    if case in SIZING_CASES:
        out["sizing"], out["layout"] = _sizing_text(case), _layout_text(case)
    if case in DESIGN_CASES:
        out["params"], out["jsbsim"], out["kit"] = _params_text(case), _jsbsim_text(case), _kit_text(case)
        try:
            out["vsp_geometry"] = _vsp_text(case)
        except ImportError:
            pass
    return out
