"""Packs built from catalogue cells: parallel count, voltages, mass from the editable assumption, legacy packs."""

import math
from pathlib import Path

import pytest

from src.schemas.assumptions import DEFAULT_ASSUMPTIONS, DesignAssumptions
from src.schemas.components import Battery, Cell
from src.tools.database.matcher import load_catalog
from src.tools.database.packs import (LI_ION_LOADED_MIN_V_CELL, LIPO_LOADED_MIN_V_CELL, build_pack,
                                      bus_window_v, charged_voltage_v, loaded_min_voltage_v)
from src.tools.database.seed_data import catalog_rows, seed_database
from src.units import WH_TO_J

MIN_CELL_ROWS: int = 15
LI_ION_21700: Cell = Cell(name="Test 21700", chemistry="li_ion", nominal_voltage_v=3.6, max_voltage_v=4.2,
                          capacity_wh=18.0, max_current_a=10.0, mass_kg=0.069, length_m=0.070, width_m=0.021,
                          height_m=0.021)


def _cell(name_part: str) -> Cell:
    return next(c for c in catalog_rows("cells") if name_part in c.name)


def test_cells_catalogue_has_li_ion_and_lipo_classes(tmp_path: Path) -> None:
    cells: list[Cell] = load_catalog(seed_database(tmp_path / "c.db"), "cells")
    assert len(cells) >= MIN_CELL_ROWS
    assert {c.chemistry for c in cells} == {"li_ion", "lipo"}
    c21700 = _cell("21700")
    assert c21700.nominal_voltage_v == pytest.approx(3.6) and c21700.max_voltage_v == pytest.approx(4.2)
    assert max(c.max_current_a for c in cells if c.chemistry == "lipo") > 100.0
    for cell in cells:
        assert cell.max_voltage_v > cell.nominal_voltage_v
        assert 100.0 <= cell.capacity_wh / cell.mass_kg <= 300.0, cell.name


def test_pack_parallel_count_meets_energy_and_current() -> None:
    # Energy-limited: 6S x 18 Wh = 108 Wh per string; 500 Wh -> 5 strings (4 would give 432 Wh).
    energy_pack = build_pack(LI_ION_21700, 6, 500.0 * WH_TO_J, 100.0, DEFAULT_ASSUMPTIONS)
    assert energy_pack.parallel == 5
    assert energy_pack.capacity_wh == pytest.approx(6 * 5 * 18.0)
    # Current-limited: 2000 W at 21.6 V = 92.6 A -> 10 strings of 10 A (9 give 90 A).
    current_pack = build_pack(LI_ION_21700, 6, 100.0 * WH_TO_J, 2000.0, DEFAULT_ASSUMPTIONS)
    assert current_pack.parallel == math.ceil(2000.0 / 21.6 / 10.0) == 10
    assert current_pack.capacity_j >= 100.0 * WH_TO_J
    assert current_pack.max_power_w >= 2000.0


def test_exact_fit_does_not_round_up_on_float_noise() -> None:
    pack = build_pack(LI_ION_21700, 6, 6 * 3 * 18.0 * WH_TO_J, 1.0, DEFAULT_ASSUMPTIONS)
    assert pack.parallel == 3


def test_pack_voltage_is_series_times_nominal() -> None:
    pack = build_pack(LI_ION_21700, 12, 400.0 * WH_TO_J, 500.0, DEFAULT_ASSUMPTIONS)
    assert pack.voltage_v == pytest.approx(12 * 3.6)
    assert (pack.cell_name, pack.series) == ("Test 21700", 12)
    assert pack.name == f"Test 21700 12S{pack.parallel}P"
    assert pack.chemistry == "li_ion"


def test_pack_mass_and_box_follow_the_assumption() -> None:
    base = build_pack(LI_ION_21700, 6, 500.0 * WH_TO_J, 100.0, DEFAULT_ASSUMPTIONS)
    light = build_pack(LI_ION_21700, 6, 500.0 * WH_TO_J, 100.0, DesignAssumptions(liion_specific_energy_wh_kg=300))
    assert base.mass_kg == pytest.approx(base.capacity_wh / DEFAULT_ASSUMPTIONS.specific_energy_wh_kg("li_ion"),
                                         rel=1e-3)
    assert light.mass_kg == pytest.approx(light.capacity_wh / 300.0, rel=1e-3)
    assert base.length_m > base.width_m > base.height_m > 0


def test_pack_c_rate_comes_from_cell_current() -> None:
    pack = build_pack(LI_ION_21700, 6, 500.0 * WH_TO_J, 100.0, DEFAULT_ASSUMPTIONS)
    assert pack.max_c_rate == pytest.approx(10.0 / (18.0 / 3.6))  # 10 A from a 5 Ah cell = 2C
    assert pack.max_power_w == pytest.approx(pack.parallel * 10.0 * pack.voltage_v)


def test_hybrid_buffer_pack_is_lipo() -> None:
    # A hybrid buffer: little energy (the generator carries cruise) but the full hover power.
    energy_j, peak_w, series = 150.0 * WH_TO_J, 6_000.0, 12
    packs: list[Battery] = [build_pack(c, series, energy_j, peak_w, DEFAULT_ASSUMPTIONS) for c in catalog_rows("cells")]
    lightest = min(packs, key=lambda b: b.mass_kg)
    assert lightest.chemistry == "lipo"
    assert lightest.max_power_w >= peak_w


def test_loaded_min_and_charged_voltages() -> None:
    assert LI_ION_LOADED_MIN_V_CELL == 3.3 and LIPO_LOADED_MIN_V_CELL == 3.5
    assert loaded_min_voltage_v("li_ion", 12) == pytest.approx(39.6)
    assert loaded_min_voltage_v("lipo", 6) == pytest.approx(21.0)
    assert charged_voltage_v(LI_ION_21700, 12) == pytest.approx(50.4)
    assert bus_window_v(LI_ION_21700, 12) == pytest.approx((39.6, 50.4))


@pytest.mark.parametrize("series, energy_j, peak_w", [(0, 1e5, 100.0), (6, 0.0, 100.0), (6, 1e5, -1.0),
                                                      (6, math.nan, 100.0)])
def test_build_pack_rejects_bad_inputs(series: int, energy_j: float, peak_w: float) -> None:
    with pytest.raises(ValueError):
        build_pack(LI_ION_21700, series, energy_j, peak_w, DEFAULT_ASSUMPTIONS)


def test_voltage_helpers_reject_bad_series() -> None:
    with pytest.raises(ValueError):
        loaded_min_voltage_v("li_ion", 0)
    with pytest.raises(ValueError):
        charged_voltage_v(LI_ION_21700, -2)


def test_legacy_battery_without_cells_loads(tmp_path: Path) -> None:
    legacy = Battery.model_validate({"name": "LiPo 6S 10000mAh", "chemistry": "lipo", "capacity_wh": 222.0,
                                     "max_c_rate": 25.0, "voltage_v": 22.2, "mass_kg": 1.48, "length_m": 0.2,
                                     "width_m": 0.1, "height_m": 0.066})
    assert (legacy.cell_name, legacy.series, legacy.parallel) == (None, None, None)
    seeded: list[Battery] = load_catalog(seed_database(tmp_path / "c.db"), "batteries")
    assert seeded and all(b.cell_name is None for b in seeded)  # fixed packs stay until Task 6's switch-over
