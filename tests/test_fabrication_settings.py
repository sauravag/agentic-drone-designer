import math

import pytest
from pydantic import ValidationError

from src.schemas.fabrication import DEFAULT_FABRICATION, FabricationSettings, TubeStock


def test_defaults_match_the_spec() -> None:
    s = DEFAULT_FABRICATION
    assert (s.bed_width_m, s.bed_height_m, s.margin_m, s.kerf_m) == (0.600, 0.400, 0.005, 0.00015)
    assert s.sheet_for("rib").thickness_m == 0.0015 and s.sheet_for("former").thickness_m == 0.003
    assert {(round(t.od_m * 1e3), round(t.id_m * 1e3)) for t in s.tubes} >= {
        (25, 23),
        (36, 32),
        (40, 36),
        (6, 4),
    }


def test_every_family_needs_exactly_one_sheet() -> None:
    three_mm = DEFAULT_FABRICATION.sheet_for("former")
    with pytest.raises(ValidationError, match="rib"):
        FabricationSettings(sheets=(three_mm,))  # the 3 mm sheet does not list "rib"


def test_family_on_two_sheets_is_rejected() -> None:
    three_mm = DEFAULT_FABRICATION.sheet_for("former")
    one_five = DEFAULT_FABRICATION.sheet_for("rib")
    overlapping = one_five.model_copy(update={"families": ("rib", "former")})
    with pytest.raises(ValidationError, match="former"):
        FabricationSettings(sheets=(three_mm, overlapping))


def test_tubes_are_sorted_by_od() -> None:
    tubes = (TubeStock(od_m=0.020, id_m=0.018), TubeStock(od_m=0.010, id_m=0.008))
    s = FabricationSettings(tubes=tubes)
    assert [t.od_m for t in s.tubes] == [0.010, 0.020]


def test_tube_id_must_be_below_od() -> None:
    with pytest.raises(ValidationError):
        TubeStock(od_m=0.010, id_m=0.010)


def test_round_trips_json() -> None:
    assert FabricationSettings.model_validate_json(DEFAULT_FABRICATION.model_dump_json()) == DEFAULT_FABRICATION


def test_second_moment_of_a_tube() -> None:
    t = TubeStock(od_m=0.025, id_m=0.023)
    assert t.second_moment_m4 == pytest.approx(math.pi / 64 * (0.025**4 - 0.023**4))


def test_area_of_solid_rod() -> None:
    assert TubeStock(od_m=0.004, id_m=0.0).area_m2 == pytest.approx(math.pi / 4 * 0.004**2)
