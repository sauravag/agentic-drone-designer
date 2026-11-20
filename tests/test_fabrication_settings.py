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


def test_minimum_insertion_defaults_to_three_spar_diameters() -> None:
    assert DEFAULT_FABRICATION.min_insertion_diameters == 3.0
    assert not hasattr(DEFAULT_FABRICATION, "insertion_fraction")
    with pytest.raises(ValidationError, match="min_insertion_diameters"):
        FabricationSettings(min_insertion_diameters=0.0)


def test_structural_check_defaults_match_the_spec() -> None:
    s = FabricationSettings()
    assert s.carbon_shear_allow_pa == 60e6
    assert s.ply_bearing_allow_pa == 15e6
    assert s.overwrap_layer_thickness_m == 0.25e-3
    assert s.overwrap_allow_pa == 250e6
    assert s.overwrap_density_kg_m3 == 1500.0
    assert s.overwrap_max_layers == 12  # R29
    assert s.epoxy_lap_shear_pa == 15e6
    assert s.crutch_rail_min_m == 0.008
    assert s.ply_shear_allow_pa == 7e6  # R33


def test_polar_moment_of_a_tube() -> None:
    tube = TubeStock(od_m=0.025, id_m=0.023)
    assert tube.polar_moment_m4 == pytest.approx(2.0 * tube.second_moment_m4)


def test_overwrap_cloth_defaults_to_200_g_m2_and_must_fit_its_cured_layer():
    from src.schemas.fabrication import FabricationSettings

    assert FabricationSettings().overwrap_cloth_g_m2 == 200.0
    with pytest.raises(ValueError, match="overwrap_cloth_g_m2"):  # 0.6 kg/m2 of fibre in a 0.375 kg/m2 layer
        FabricationSettings(overwrap_cloth_g_m2=600.0)


def test_usable_bed_is_the_bed_inside_the_margin() -> None:
    assert DEFAULT_FABRICATION.usable_bed_m == pytest.approx((0.590, 0.390))
    s = FabricationSettings(bed_width_m=0.3, bed_height_m=0.5, margin_m=0.01)
    assert s.usable_bed_m == pytest.approx((0.28, 0.48))


def test_shared_fixtures_load_without_the_cad_stack() -> None:
    """M5: conftest imports nothing that needs build123d, so sizing-only tests collect on a host without it."""
    import subprocess
    import sys
    from pathlib import Path

    code = "import sys; sys.modules['build123d'] = None; import tests.conftest"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).resolve().parents[1])


def test_every_default_is_a_named_constant() -> None:
    """D1: each field's default is the module's DEFAULT_<FIELD> Final constant, not a literal in Field(...)."""
    import src.schemas.fabrication as fabrication

    for name, field in FabricationSettings.model_fields.items():
        assert field.default == getattr(fabrication, f"DEFAULT_{name.upper()}"), name


def test_all_families_is_the_part_family_literal() -> None:
    """D2: one source of truth for the part families."""
    from typing import get_args

    from src.schemas.fabrication import ALL_FAMILIES, PartFamily

    assert ALL_FAMILIES == get_args(PartFamily)


@pytest.mark.parametrize("rods", [(0.0, 0.004), (-0.003,)])
def test_rods_must_be_positive(rods: tuple[float, ...]) -> None:
    """D3: a zero or negative rod is a settings error, not an unexpected kit error later."""
    with pytest.raises(ValidationError, match="rod diameters must be positive"):
        FabricationSettings(rods_m=rods)


def test_duplicate_rods_are_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate rod diameter 4 mm"):
        FabricationSettings(rods_m=(0.004, 0.003, 0.004))


def test_duplicate_tubes_are_rejected() -> None:
    """D5."""
    tube = TubeStock(od_m=0.006, id_m=0.004)
    with pytest.raises(ValidationError, match="duplicate tube 6 x 4 mm"):
        FabricationSettings(tubes=(tube, TubeStock(od_m=0.008, id_m=0.006), tube))


@pytest.mark.parametrize("margin", [0.2, 0.25])
def test_margins_must_leave_a_usable_bed(margin: float) -> None:
    """D4: 2 x margin must be below the bed's smaller side."""
    with pytest.raises(ValidationError, match="margin_m .* leaves no usable bed"):
        FabricationSettings(bed_width_m=0.6, bed_height_m=0.4, margin_m=margin)
    FabricationSettings(bed_width_m=0.6, bed_height_m=0.4, margin_m=0.19)
