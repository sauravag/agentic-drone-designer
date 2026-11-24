"""Propeller strike and folding-propeller reporting on the CAD record (whole-branch review M1, M2 and the Task 6
deferred minor, phase 2): what the propeller hits, and the folding-propeller requirement with or without a kit."""

import logging
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("build123d")

from src.schemas.components import SelectedBOM  # noqa: E402
from src.schemas.state import CADArtifacts, SizingOutput  # noqa: E402
from src.tools.aero_cad import cad_packer, components  # noqa: E402
from src.tools.aero_cad.clearance import PropClearance  # noqa: E402
from src.tools.aero_cad.layout import AircraftLayout, layout_from_sizing  # noqa: E402
from src.tools.fabrication.kit import KitUnavailable  # noqa: E402
from src.tools.fabrication.kit_pod import kit_fuselage_diameter_m  # noqa: E402
from src.tools.gear import GearGeometry, PropellerDisc  # noqa: E402
from tests.cad_helpers import _synthetic_aero, _synthetic_oml  # noqa: E402


def test_clear_propeller_has_no_strike_description() -> None:
    assert PropClearance(0.02, 0.03, None, False).strike_description is None


def test_strike_description_names_the_tail_and_the_ground() -> None:
    assert PropClearance(-0.01, 0.03, None, False).strike_description == (
        "propeller strikes the tail (disc 0.010 m into the airframe ahead of it)")
    assert PropClearance(0.02, -0.012, None, False).strike_description == (
        "propeller strikes the ground (tip 0.012 m below the ground on the gear)")
    both = PropClearance(-0.01, -0.012, None, False).strike_description
    assert both is not None and "tail" in both and "ground" in both


@pytest.fixture()
def no_kit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the shell-model fallback: the kit build sheet (and its folding-prop note) is not written."""
    def unavailable(*_args: object, **_kwargs: object) -> None:
        raise KitUnavailable("forced by the test")
    monkeypatch.setattr(cad_packer, "build_kit", unavailable)


@pytest.fixture()
def short_legs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gear legs cut to the pod radius: the pusher's disc reaches the ground at rotation."""
    real = cad_packer.as_built_gear

    def stub(sizing: SizingOutput, x_cg_m: float, propeller: PropellerDisc) -> GearGeometry:
        gear = real(sizing, x_cg_m, propeller)
        leg = sizing.fuselage_diameter_m / 2
        return replace(gear, contacts=tuple(replace(c, z_m=-leg) for c in gear.contacts), leg_height_m=leg)

    monkeypatch.setattr(cad_packer, "as_built_gear", stub)


def _pusher_layout(sizing: SizingOutput, bom: SelectedBOM) -> AircraftLayout:
    diameter: float = max(sizing.fuselage_diameter_m, kit_fuselage_diameter_m(sizing, bom))
    return layout_from_sizing(sizing, 0.30, diameter, components.required_fuselage_length_m(sizing, bom, diameter),
                              propeller_diameter_m=bom.cruise_propeller.diameter_m)


def _pack_and_export(vtail_g1: tuple[SizingOutput, SelectedBOM], tmp_path: Path) -> CADArtifacts:
    sizing, bom = vtail_g1
    lay = _pusher_layout(sizing, bom)
    step = _synthetic_oml(lay, tmp_path / "oml.step")
    return cad_packer.pack_aircraft(sizing, bom, _synthetic_aero(lay), lay, tmp_path / "oml.vsp3", step,
                                    tmp_path / "out")


def test_artifacts_carry_the_strike_and_pack_aircraft_warns_once(vtail_g1: tuple[SizingOutput, SelectedBOM],
                                                                 tmp_path: Path, no_kit: None, short_legs: None,
                                                                 caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger=cad_packer.logger.name):
        cad = _pack_and_export(vtail_g1, tmp_path)
    assert cad.collision_count >= 1
    assert cad.prop_strike is not None and cad.prop_strike.startswith("propeller strikes the ground")
    warned = [r for r in caplog.records if "propeller strike" in r.getMessage() and r.levelno >= logging.WARNING]
    assert [r.levelno for r in warned] == [logging.WARNING]


def test_folding_prop_reaches_the_artifacts_without_a_kit(vtail_g1: tuple[SizingOutput, SelectedBOM],
                                                          tmp_path: Path, no_kit: None,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    real = cad_packer.prop_clearance
    monkeypatch.setattr(cad_packer, "prop_clearance",
                        lambda layout, gear, launch: replace(real(layout, gear, launch), needs_folding_prop=True))
    cad = _pack_and_export(vtail_g1, tmp_path)
    assert cad.kit_dir is None and cad.needs_folding_prop and cad.prop_strike is None
