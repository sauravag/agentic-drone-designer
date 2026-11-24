"""Boom-end pusher balance search with a floored pod (whole-branch review I1, phase 2): a wing station whose tail
plus propeller gap ends ahead of the boom root is infeasible, like a station with no packing slot, not fatal."""

from pathlib import Path

import pytest

pytest.importorskip("build123d")

from src.schemas.components import SelectedBOM  # noqa: E402
from src.schemas.state import SizingOutput  # noqa: E402
from src.tools.aero_cad import build  # noqa: E402
from src.tools.aero_cad.cad_packer import CADPackingError  # noqa: E402
from src.tools.aero_cad.layout import AircraftLayout, layout_from_sizing  # noqa: E402
from src.tools.sizing.layouts import PusherBoomClearanceError  # noqa: E402
from tests.cad_helpers import _synthetic_aero, _synthetic_oml  # noqa: E402

FLOORED_POD_FACTOR: float = 1.5  # pod floored 50 % over its sized length: the stations up to 0.30 cannot hold the boom
UNBUILDABLE_POD_FACTOR: float = 4.0  # no scan station puts the tail and propeller gap behind the boom root
INFEASIBLE_STATION: float = 0.30


def _synthetic_analyser(tmp_path: Path):
    def analyse(layout: AircraftLayout) -> build.OmlAnalysis:
        step = _synthetic_oml(layout, tmp_path / f"oml_{layout.wing_station:.4f}.step")
        return build.OmlAnalysis(_synthetic_aero(layout), tmp_path / "oml.vsp3", step)
    return analyse


def _floor_pod(monkeypatch: pytest.MonkeyPatch, sizing: SizingOutput, factor: float) -> None:
    monkeypatch.setattr(build, "required_fuselage_length_m", lambda *_: factor * sizing.fuselage_length_m)


def test_boom_ending_ahead_of_its_root_is_a_clearance_error(vtail_g1: tuple[SizingOutput, SelectedBOM]) -> None:
    sizing, bom = vtail_g1
    with pytest.raises(PusherBoomClearanceError, match="ahead of its root"):
        layout_from_sizing(sizing, INFEASIBLE_STATION, 0.0, FLOORED_POD_FACTOR * sizing.fuselage_length_m,
                           propeller_diameter_m=bom.cruise_propeller.diameter_m)


def test_infeasible_forward_station_does_not_abort_the_search(vtail_g1: tuple[SizingOutput, SelectedBOM],
                                                              tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    sizing, bom = vtail_g1
    _floor_pod(monkeypatch, sizing, FLOORED_POD_FACTOR)
    design = build.balance_design(sizing, bom, _synthetic_analyser(tmp_path))
    assert design.layout.wing_station > INFEASIBLE_STATION
    assert design.packing.is_balanced and design.packing.interference_count == 0


def test_no_feasible_station_is_a_readable_packing_error(vtail_g1: tuple[SizingOutput, SelectedBOM], tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    sizing, bom = vtail_g1
    _floor_pod(monkeypatch, sizing, UNBUILDABLE_POD_FACTOR)
    with pytest.raises(CADPackingError, match="no wing station produced a packable layout.*boom"):
        build.balance_design(sizing, bom, _synthetic_analyser(tmp_path))
