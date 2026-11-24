"""Propeller CT/CP tables: accuracy against the BEMT model, range handling, failure rules and the on-disk cache."""

import json
import logging
import math
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, Final

import aerosandbox as asb
import numpy as np
import pytest

from src.tools.propulsion import prop_tables
from src.tools.propulsion.bemt import (PROP_AIRFOIL, PROP_MODEL_VERSION, PropBatch, PropModelError, bemt_point,
                                       generic_blade)
from src.tools.propulsion.prop_tables import PropellerTable, propeller_table
from src.tools.sizing.aero_model import air_properties

SECONDS_PER_MINUTE: Final[float] = 60.0
ACCURACY: Final[float] = 0.02
SMALL_PROP: Final[tuple[float, float]] = (0.254, 0.1143)  # 10 x 4.5 inch
BIG_PROP: Final[tuple[float, float]] = (2.0, 1.0)
SAMPLES_PER_CASE: Final[int] = 20
MIN_STATIC_THRUST_FRACTION: Final[float] = 0.10  # envelope: thrust still >= 10 % of static
MAX_SAMPLE_TIP_MACH: Final[float] = 0.75
MAX_SAMPLE_TRIES: Final[int] = 3000
ISA_PLUS_20: Final[float] = 20.0
HIGH_ALTITUDE_M: Final[float] = 4000.0
Atmosphere = tuple[float, float, float]  # rho, mu, speed of sound


def _atmosphere(altitude_m: float, offset_c: float) -> Atmosphere:
    rho, mu = air_properties(altitude_m, offset_c)
    sound: float = float(asb.Atmosphere(altitude=altitude_m, temperature_deviation=offset_c).speed_of_sound())
    return rho, mu, sound


SEA_LEVEL: Final[Atmosphere] = _atmosphere(0.0, 0.0)
FOUR_KM_HOT: Final[Atmosphere] = _atmosphere(HIGH_ALTITUDE_M, ISA_PLUS_20)


@pytest.fixture(scope="module")
def small_table() -> PropellerTable:
    return prop_tables._build_table(*SMALL_PROP)


@pytest.fixture(scope="module")
def big_table() -> PropellerTable:
    return prop_tables._build_table(*BIG_PROP)


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point the disk cache at a temp dir and drop the in-memory cache around each test."""
    monkeypatch.setenv("PROP_TABLE_CACHE_DIR", str(tmp_path))
    propeller_table.cache_clear()
    yield tmp_path
    propeller_table.cache_clear()


@pytest.fixture
def counted_build(small_table: PropellerTable, monkeypatch: pytest.MonkeyPatch) -> list[tuple[float, float]]:
    """Replace the slow table build with the shared small table and record each call."""
    calls: list[tuple[float, float]] = []

    def fake_build(diameter_m: float, pitch_m: float) -> PropellerTable:
        calls.append((diameter_m, pitch_m))
        return small_table
    monkeypatch.setattr(prop_tables, "_build_table", fake_build)
    return calls


def _cache_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.json"))


def _operating_point(table: PropellerTable, tip_mach: float, reynolds75: float, j: float,
                     sound: float = SEA_LEVEL[2]) -> tuple[float, float, float, float]:
    """(rpm, airspeed, rho, mu) of a point at this tip Mach, Re75 and J (density follows from Re75)."""
    d: float = table.diameter_m
    rpm: float = tip_mach * sound / (math.pi * d) * SECONDS_PER_MINUTE
    rho: float = reynolds75 / prop_tables.reynolds75_at(d, table.pitch_m, rpm, 1.0, SEA_LEVEL[1]) * 1.0
    return rpm, j * rpm / SECONDS_PER_MINUTE * d, rho, SEA_LEVEL[1]


def _middle(table: PropellerTable) -> tuple[float, float]:
    return table.tip_mach[len(table.tip_mach) // 2], table.reynolds75[len(table.reynolds75) // 2]


def _sample_envelope(table: PropellerTable, atmosphere: Atmosphere) -> Iterator[tuple[float, float, Any]]:
    """Random (rpm, airspeed, BEMT point) inside the table where thrust is >= 10 % of static, rpm sweep included."""
    rho, mu, sound = atmosphere
    d, p = table.diameter_m, table.pitch_m
    blade = generic_blade(d, p)
    rng = np.random.default_rng(3)
    found: int = 0
    for _ in range(MAX_SAMPLE_TRIES):
        rpm: float = rng.uniform(table.tip_mach[0], MAX_SAMPLE_TIP_MACH) * sound / (math.pi * d) * SECONDS_PER_MINUTE
        re75: float = prop_tables.reynolds75_at(d, p, rpm, rho, mu)
        if not table.reynolds75[0] <= re75 <= table.reynolds75[-1]:
            continue
        airspeed: float = rng.uniform(0.0, table.j[-1]) * rpm / SECONDS_PER_MINUTE * d
        try:
            point = bemt_point(blade, rpm, airspeed, rho, mu, sound)
            static = bemt_point(blade, rpm, 0.0, rho, mu, sound)
        except PropModelError:
            continue
        if point.thrust_n >= MIN_STATIC_THRUST_FRACTION * static.thrust_n:
            found += 1
            yield rpm, airspeed, point
        if found == SAMPLES_PER_CASE:
            return
    raise AssertionError(f"only {found} envelope samples found")


@pytest.mark.parametrize("atmosphere", [SEA_LEVEL, FOUR_KM_HOT], ids=["sea_level", "4000m_isa_plus_20"])
@pytest.mark.parametrize("which", ["small", "big"])
def test_table_matches_bemt_within_two_percent(which: str, atmosphere: Atmosphere, small_table: PropellerTable,
                                               big_table: PropellerTable) -> None:
    table: PropellerTable = small_table if which == "small" else big_table
    rho, mu, sound = atmosphere
    for rpm, airspeed, point in _sample_envelope(table, atmosphere):
        assert table.thrust_n(rpm, airspeed, rho, mu, sound) == pytest.approx(point.thrust_n, rel=ACCURACY)
        assert table.torque_nm(rpm, airspeed, rho, mu, sound) == pytest.approx(point.torque_nm, rel=ACCURACY)


def test_coefficients_outside_each_axis_raise(small_table: PropellerTable) -> None:
    mach, re75 = _middle(small_table)
    j: float = small_table.j[len(small_table.j) // 2]
    with pytest.raises(ValueError, match="J="):
        small_table.coefficients(small_table.j[-1] * 1.01, mach, re75)
    with pytest.raises(ValueError, match="J="):
        small_table.coefficients(-0.01, mach, re75)
    with pytest.raises(ValueError, match="tip Mach"):
        small_table.coefficients(j, small_table.tip_mach[-1] * 1.01, re75)
    with pytest.raises(ValueError, match="tip Mach"):
        small_table.coefficients(j, small_table.tip_mach[0] * 0.9, re75)
    with pytest.raises(ValueError, match="Re75"):
        small_table.coefficients(j, mach, small_table.reynolds75[0] * 0.9)
    with pytest.raises(ValueError, match="Re75"):
        small_table.coefficients(j, mach, small_table.reynolds75[-1] * 1.1)


def test_thrust_and_torque_propagate_range_errors(small_table: PropellerTable) -> None:
    rho, mu, sound = SEA_LEVEL
    with pytest.raises(ValueError, match="Re75"):
        small_table.thrust_n(9000.0, 0.0, 1e-3, mu, sound)  # near-vacuum density: Re75 far below the floor
    with pytest.raises(ValueError, match="tip Mach"):
        small_table.torque_nm(60000.0, 0.0, rho, mu, sound)  # supersonic tip
    with pytest.raises(ValueError, match="finite"):
        small_table.thrust_n(math.nan, 0.0, rho, mu, sound)


def test_altitude_point_with_same_tip_mach_is_inside_the_table(small_table: PropellerTable) -> None:
    rho, mu, sound = FOUR_KM_HOT
    rpm: float = 0.7 * sound / (math.pi * small_table.diameter_m) * SECONDS_PER_MINUTE
    assert small_table.thrust_n(rpm, 0.0, rho, mu, sound) > 0.0


def test_thrust_never_negative_and_zero_at_top_of_j_axis(small_table: PropellerTable) -> None:
    mach, re75 = _middle(small_table)
    rpm, airspeed, rho, mu = _operating_point(small_table, mach, re75, small_table.j[-1])
    assert small_table.thrust_n(rpm, airspeed, rho, mu, SEA_LEVEL[2]) == 0.0
    ct, _ = small_table.coefficients(small_table.j[-1], mach, re75)
    assert ct <= 0.0


@pytest.mark.parametrize("which", ["small", "big"])
def test_j_axis_reaches_zero_thrust_in_every_row(which: str, small_table: PropellerTable,
                                                 big_table: PropellerTable) -> None:
    table: PropellerTable = small_table if which == "small" else big_table
    ct: np.ndarray = np.asarray(table.ct)
    assert np.all(ct[..., -1] <= 0.0)
    assert np.all(ct[..., 0] > 0.0)  # static thrust positive everywhere
    assert np.all(np.isfinite(np.asarray(table.cp)))
    assert table.reynolds75[0] > 1e4 and table.tip_mach[-1] == pytest.approx(0.80)


def test_pitch_speed(small_table: PropellerTable) -> None:
    assert small_table.pitch_speed_ms(6000.0) == pytest.approx(SMALL_PROP[1] * 100.0)


def _failing_above(monkeypatch: pytest.MonkeyPatch, j_fail: float) -> None:
    """Make every BEMT point with J > j_fail fail (values untouched, error recorded), as the real model does."""
    real: Callable[..., PropBatch] = prop_tables.bemt_points

    def spy(blade: Any, rpm: np.ndarray, airspeed: np.ndarray, *args: Any) -> PropBatch:
        batch: PropBatch = real(blade, rpm, airspeed, *args)
        j: np.ndarray = airspeed / (rpm / SECONDS_PER_MINUTE * blade.diameter_m)
        errors = tuple("synthetic failure" if jj > j_fail else e for jj, e in zip(j, batch.errors))
        return replace(batch, errors=errors)
    monkeypatch.setattr(prop_tables, "bemt_points", spy)


def test_failure_after_zero_thrust_holds_last_point_and_limits_torque(
        small_table: PropellerTable, monkeypatch: pytest.MonkeyPatch) -> None:
    j: tuple[float, ...] = small_table.j
    _failing_above(monkeypatch, (j[-2] + j[-1]) / 2.0)  # only the last J node fails, past zero thrust
    table: PropellerTable = prop_tables._build_table(*SMALL_PROP)
    assert np.allclose(np.asarray(table.j_limit), j[-2])
    assert np.all(np.asarray(table.ct)[..., -1] <= 0.0) and np.all(np.isfinite(np.asarray(table.cp)))
    assert np.array_equal(np.asarray(table.cp)[..., -1], np.asarray(table.cp)[..., -2])
    mach, re75 = _middle(table)
    rpm, airspeed, rho, mu = _operating_point(table, mach, re75, j[-1])
    assert table.thrust_n(rpm, airspeed, rho, mu, SEA_LEVEL[2]) == 0.0
    with pytest.raises(ValueError, match="beyond the last computed J"):
        table.torque_nm(rpm, airspeed, rho, mu, SEA_LEVEL[2])
    inside = _operating_point(table, mach, re75, j[-2] * 0.99)
    assert math.isfinite(table.torque_nm(*inside, SEA_LEVEL[2]))


def test_failure_before_zero_thrust_in_scan_propagates_with_prop_size(monkeypatch: pytest.MonkeyPatch) -> None:
    _failing_above(monkeypatch, 0.2)  # CT is still positive there
    with pytest.raises(PropModelError, match=r"0\.254 x 0\.1143.*zero-thrust search"):
        prop_tables._build_table(*SMALL_PROP)


def test_failure_at_every_point_propagates_with_prop_size(monkeypatch: pytest.MonkeyPatch) -> None:
    _failing_above(monkeypatch, -1.0)
    with pytest.raises(PropModelError, match=r"0\.254 x 0\.1143"):
        prop_tables._build_table(*SMALL_PROP)


def test_clamp_row_holds_last_point_after_zero_thrust() -> None:
    j_axis: np.ndarray = np.array([0.0, 0.1, 0.2, 0.3])
    ct, cp = np.array([1.0, -0.1, math.nan, math.nan]), np.array([2.0, 1.5, math.nan, math.nan])
    held_ct, held_cp, limit = prop_tables._clamp_row(ct, cp, np.array([True, True, False, False]), j_axis, "row")
    assert held_ct.tolist() == [1.0, -0.1, -0.1, -0.1] and held_cp.tolist() == [2.0, 1.5, 1.5, 1.5]
    assert limit == 0.1


def test_clamp_row_never_holds_a_positive_ct() -> None:
    j_axis: np.ndarray = np.array([0.0, 0.1, 0.2])
    ct, cp = np.array([1.0, 0.5, math.nan]), np.array([2.0, 1.5, math.nan])
    with pytest.raises(PropModelError, match="before zero thrust"):  # failure just above the last positive point
        prop_tables._clamp_row(ct, cp, np.array([True, True, False]), j_axis, "row")
    with pytest.raises(PropModelError, match="before zero thrust"):
        prop_tables._clamp_row(ct, cp, np.array([False, True, True]), j_axis, "row")


def test_clamp_row_leaves_complete_rows_alone() -> None:
    j_axis: np.ndarray = np.array([0.0, 0.1])
    ct, cp = np.array([1.0, -0.1]), np.array([2.0, 1.5])
    out_ct, out_cp, limit = prop_tables._clamp_row(ct, cp, np.array([True, True]), j_axis, "row")
    assert out_ct is ct and out_cp is cp and limit == 0.1


def test_cache_round_trip(isolated_cache: Path, counted_build: list[tuple[float, float]],
                          small_table: PropellerTable, monkeypatch: pytest.MonkeyPatch) -> None:
    first: PropellerTable = propeller_table(*SMALL_PROP)
    assert len(_cache_files(isolated_cache)) == 1 and len(counted_build) == 1
    propeller_table.cache_clear()
    bemt_calls: list[int] = []
    monkeypatch.setattr(prop_tables, "bemt_points", lambda *a, **k: bemt_calls.append(1))
    second: PropellerTable = propeller_table(*SMALL_PROP)
    assert len(counted_build) == 1 and bemt_calls == []  # read from disk: no build, no BEMT
    assert second == first == small_table


def test_memory_cache_returns_same_object(counted_build: list[tuple[float, float]]) -> None:
    assert propeller_table(*SMALL_PROP) is propeller_table(*SMALL_PROP)
    assert len(counted_build) == 1


def test_corrupt_cache_file_is_rebuilt(isolated_cache: Path, counted_build: list[tuple[float, float]],
                                       caplog: pytest.LogCaptureFixture) -> None:
    propeller_table(*SMALL_PROP)
    (path,) = _cache_files(isolated_cache)
    path.write_text(path.read_text()[:100])  # truncated JSON
    propeller_table.cache_clear()
    with caplog.at_level(logging.WARNING):
        table: PropellerTable = propeller_table(*SMALL_PROP)
    assert any("rebuilding" in r.getMessage() for r in caplog.records)
    assert len(counted_build) == 2 and math.isfinite(table.ct[0][0][0])
    assert json.loads(path.read_text())["version"] == PROP_MODEL_VERSION  # healed on disk


def _tamper(path: Path, change: Callable[[dict[str, Any]], None]) -> None:
    payload: dict[str, Any] = json.loads(path.read_text())
    change(payload)
    path.write_text(json.dumps(payload))


def _stale_version(payload: dict[str, Any]) -> None:
    payload["version"] = PROP_MODEL_VERSION + 1


def _stale_grid(payload: dict[str, Any]) -> None:
    payload["grid"] = "0000000000"


def _decreasing_re(payload: dict[str, Any]) -> None:
    payload["reynolds75"] = payload["reynolds75"][::-1]


def _j_not_from_zero(payload: dict[str, Any]) -> None:
    payload["j"] = [x + 0.1 for x in payload["j"]]


def _mach_too_high(payload: dict[str, Any]) -> None:
    payload["tip_mach"][-1] = 1.5


def _limit_outside_axis(payload: dict[str, Any]) -> None:
    payload["j_limit"][0][0] = payload["j"][-1] * 2.0


def _non_finite_cp(payload: dict[str, Any]) -> None:
    payload["cp"][0][0][0] = float("nan")


@pytest.mark.parametrize("change", [_stale_version, _stale_grid, _decreasing_re, _j_not_from_zero, _mach_too_high,
                                    _limit_outside_axis, _non_finite_cp])
def test_stale_or_invalid_cache_is_rebuilt_with_warning_and_healed(
        change: Callable[[dict[str, Any]], None], isolated_cache: Path, counted_build: list[tuple[float, float]],
        caplog: pytest.LogCaptureFixture) -> None:
    propeller_table(*SMALL_PROP)
    (path,) = _cache_files(isolated_cache)
    _tamper(path, change)
    propeller_table.cache_clear()
    with caplog.at_level(logging.WARNING):
        table: PropellerTable = propeller_table(*SMALL_PROP)
    assert any("rebuilding" in r.getMessage() for r in caplog.records)
    assert len(counted_build) == 2
    healed = json.loads(path.read_text())
    assert healed["version"] == PROP_MODEL_VERSION and healed["grid"] == prop_tables.GRID_SPEC
    assert math.isfinite(table.ct[0][0][0])


def test_grid_constants_change_the_cache_key(isolated_cache: Path, counted_build: list[tuple[float, float]],
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    propeller_table(*SMALL_PROP)
    (path,) = _cache_files(isolated_cache)
    assert prop_tables.GRID_SPEC in path.name
    monkeypatch.setattr(prop_tables, "GRID_SPEC", "abcdef0123")
    propeller_table.cache_clear()
    propeller_table(*SMALL_PROP)
    assert len(counted_build) == 2 and len(_cache_files(isolated_cache)) == 2


def test_cache_file_name_carries_key_and_ignores_float_type(isolated_cache: Path,
                                                            counted_build: list[tuple[float, float]]) -> None:
    propeller_table(*SMALL_PROP)
    (path,) = _cache_files(isolated_cache)
    assert f"v{PROP_MODEL_VERSION}" in path.name and PROP_AIRFOIL in path.name
    assert "D0.254_" in path.name and "P0.1143" in path.name
    assert prop_tables._cache_path(np.float64(0.254), np.float64(0.1143)) == path
    assert prop_tables._cache_path(1, 1).name.endswith("D1.0_P1.0.json")


def test_default_cache_dir_is_repo_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROP_TABLE_CACHE_DIR")
    assert prop_tables.cache_dir() == Path(prop_tables.__file__).resolve().parents[3] / ".cache" / "prop_tables"


def test_failed_cache_write_leaves_no_temp_file_and_still_returns_table(
        isolated_cache: Path, counted_build: list[tuple[float, float]], small_table: PropellerTable,
        monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def broken_dump(*_args: object, **_kwargs: object) -> None:
        raise TypeError("not serialisable")
    monkeypatch.setattr(prop_tables.json, "dump", broken_dump)
    with caplog.at_level(logging.WARNING):
        table: PropellerTable = propeller_table(*SMALL_PROP)
    assert table is small_table and list(isolated_cache.iterdir()) == []
    assert any("Could not write" in r.getMessage() for r in caplog.records)


def test_invalid_size_rejected() -> None:
    with pytest.raises(ValueError):
        propeller_table(-1.0, 0.1)
    with pytest.raises(ValueError):
        propeller_table(0.254, float("nan"))


@pytest.mark.parametrize("size", [(0.2, 0.1), (2.5, 1.2)])
def test_extreme_props_build_and_match(size: tuple[float, float]) -> None:
    table: PropellerTable = prop_tables._build_table(*size)
    rho, mu, sound = FOUR_KM_HOT
    for rpm, airspeed, point in list(_sample_envelope(table, FOUR_KM_HOT))[:5]:
        assert table.thrust_n(rpm, airspeed, rho, mu, sound) == pytest.approx(point.thrust_n, rel=ACCURACY)
