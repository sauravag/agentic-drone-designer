"""Propeller CT/CP tables: accuracy against the BEMT model, range handling and the on-disk cache."""

import json
import logging
import math
from pathlib import Path

import numpy as np
import pytest

from src.tools.propulsion import bemt, prop_tables
from src.tools.propulsion.bemt import PROP_AIRFOIL, PROP_MODEL_VERSION, bemt_point, generic_blade
from src.tools.propulsion.prop_tables import PropellerTable, propeller_table

RHO: float = 1.225
MU: float = 1.81e-5
SOUND: float = 340.3
ACCURACY: float = 0.02
SMALL_PROP: tuple[float, float] = (0.254, 0.1143)  # 10 x 4.5 inch
BIG_PROP: tuple[float, float] = (2.0, 1.0)


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the disk cache at a temp dir and drop the in-memory cache around each test."""
    monkeypatch.setenv("PROP_TABLE_CACHE_DIR", str(tmp_path))
    propeller_table.cache_clear()
    yield tmp_path
    propeller_table.cache_clear()


def _cache_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.json"))


def test_table_matches_bemt_within_two_percent() -> None:
    # Arrange
    diameter, pitch = SMALL_PROP
    table: PropellerTable = propeller_table(diameter, pitch)
    blade = generic_blade(diameter, pitch)
    rng = np.random.default_rng(7)
    checked: int = 0
    # Act / Assert: random (rpm, V) inside the grid, away from the zero-thrust crossing
    while checked < 12:
        rpm: float = float(rng.uniform(4000.0, 16000.0))
        j: float = float(rng.uniform(0.05, 0.45))
        airspeed: float = j * rpm / 60.0 * diameter
        point = bemt_point(blade, rpm, airspeed, RHO, MU, SOUND)
        assert point.thrust_n > 0.0
        assert table.thrust_n(rpm, airspeed, RHO, MU) == pytest.approx(point.thrust_n, rel=ACCURACY)
        assert table.torque_nm(rpm, airspeed, RHO, MU) == pytest.approx(point.torque_nm, rel=ACCURACY)
        checked += 1


def test_coefficients_outside_j_range_raise() -> None:
    table: PropellerTable = propeller_table(*SMALL_PROP)
    with pytest.raises(ValueError, match="J"):
        table.coefficients(table.j[-1] * 1.01, table.reynolds75[len(table.reynolds75) // 2])
    with pytest.raises(ValueError, match="J"):
        table.coefficients(-0.01, table.reynolds75[len(table.reynolds75) // 2])


def test_coefficients_outside_reynolds_range_raise() -> None:
    table: PropellerTable = propeller_table(*SMALL_PROP)
    with pytest.raises(ValueError, match="Re"):
        table.coefficients(0.2, table.reynolds75[0] * 0.9)
    with pytest.raises(ValueError, match="Re"):
        table.coefficients(0.2, table.reynolds75[-1] * 1.1)
    with pytest.raises(ValueError, match="Re"):
        table.thrust_n(20.0, 0.0, RHO, MU)  # far below the Reynolds floor


def test_thrust_never_negative() -> None:
    table: PropellerTable = propeller_table(*SMALL_PROP)
    rpm: float = 9000.0
    airspeed: float = table.j[-1] * rpm / 60.0 * table.diameter_m  # top of the J axis: CT <= 0
    assert table.thrust_n(rpm, airspeed, RHO, MU) == 0.0
    ct, _ = table.coefficients(table.j[-1], table.reynolds75[3])
    assert ct <= 0.0


def test_j_axis_reaches_zero_thrust_at_every_rpm() -> None:
    for diameter, pitch in (SMALL_PROP, BIG_PROP):
        table: PropellerTable = propeller_table(diameter, pitch)
        assert all(row[-1] <= 0.0 for row in table.ct)
        assert all(row[0] > 0.0 for row in table.ct)  # static thrust positive


def test_pitch_speed() -> None:
    table: PropellerTable = propeller_table(*SMALL_PROP)
    assert table.pitch_speed_ms(6000.0) == pytest.approx(SMALL_PROP[1] * 100.0)


def test_cache_round_trip(isolated_cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first: PropellerTable = propeller_table(*SMALL_PROP)
    assert len(_cache_files(isolated_cache)) == 1
    propeller_table.cache_clear()
    calls: list[int] = []
    real = prop_tables.bemt_point
    monkeypatch.setattr(prop_tables, "bemt_point", lambda *a, **k: calls.append(1) or real(*a, **k))
    second: PropellerTable = propeller_table(*SMALL_PROP)
    assert calls == []
    assert second == first


def test_memory_cache_returns_same_object() -> None:
    assert propeller_table(*SMALL_PROP) is propeller_table(*SMALL_PROP)


def test_corrupt_cache_file_is_rebuilt(isolated_cache: Path, caplog: pytest.LogCaptureFixture) -> None:
    propeller_table(*SMALL_PROP)
    (path,) = _cache_files(isolated_cache)
    path.write_text(path.read_text()[:100])  # truncated JSON
    propeller_table.cache_clear()
    with caplog.at_level(logging.WARNING):
        table: PropellerTable = propeller_table(*SMALL_PROP)
    assert any("rebuil" in r.getMessage().lower() for r in caplog.records)
    assert math.isfinite(table.ct[0][0])
    assert json.loads(path.read_text())["version"] == PROP_MODEL_VERSION  # healed on disk


def test_stale_version_not_used(isolated_cache: Path) -> None:
    propeller_table(*SMALL_PROP)
    (path,) = _cache_files(isolated_cache)
    payload = json.loads(path.read_text())
    payload["version"] = PROP_MODEL_VERSION + 1
    payload["ct"] = [[999.0 for _ in row] for row in payload["ct"]]
    path.write_text(json.dumps(payload))
    propeller_table.cache_clear()
    table: PropellerTable = propeller_table(*SMALL_PROP)
    assert table.ct[0][0] < 100.0


def test_cache_file_name_carries_key(isolated_cache: Path) -> None:
    propeller_table(*SMALL_PROP)
    (path,) = _cache_files(isolated_cache)
    assert f"v{PROP_MODEL_VERSION}" in path.name and PROP_AIRFOIL in path.name
    assert "0.254" in path.name and "0.1143" in path.name


def test_default_cache_dir_is_repo_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROP_TABLE_CACHE_DIR")
    assert prop_tables.cache_dir() == Path(prop_tables.__file__).resolve().parents[3] / ".cache" / "prop_tables"


def test_hole_before_zero_thrust_propagates_with_prop_size(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing(*_args: object, **_kwargs: object) -> object:
        raise bemt.PropModelError("synthetic failure")
    monkeypatch.setattr(prop_tables, "bemt_point", failing)
    with pytest.raises(bemt.PropModelError, match="0.254"):
        propeller_table(*SMALL_PROP)


def test_invalid_size_rejected() -> None:
    with pytest.raises(ValueError):
        propeller_table(-1.0, 0.1)
    with pytest.raises(ValueError):
        propeller_table(0.254, float("nan"))


@pytest.mark.parametrize("size", [(0.2, 0.1), (2.5, 1.2), BIG_PROP])
def test_extreme_props_build_and_match(size: tuple[float, float]) -> None:
    diameter, pitch = size
    table: PropellerTable = propeller_table(diameter, pitch)
    rpm: float = prop_tables.rpm_for_reynolds75(diameter, pitch, math.sqrt(table.reynolds75[0] * table.reynolds75[-1]),
                                                RHO, MU)
    airspeed: float = 0.2 * rpm / 60.0 * diameter
    point = bemt_point(generic_blade(diameter, pitch), rpm, airspeed, RHO, MU, SOUND)
    assert table.thrust_n(rpm, airspeed, RHO, MU) == pytest.approx(point.thrust_n, rel=ACCURACY)
