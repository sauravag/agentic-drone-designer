"""Cached propeller CT/CP tables built from the BEMT model (spec 6.1).

A table holds CT = T / (rho n^2 D^4) and CP = P / (rho n^3 D^5) for one diameter and pitch, indexed by advance
ratio J = V / (n D) and the Reynolds number at r/R 0.75 (Re75 = rho U75 c75 / mu with U75 = 0.75 pi D n and c75 the
chord there). The sizing loop and the motor search evaluate it thousands of times, so one table is built once per
prop (a few hundred BEMT points), kept in memory and written to disk as JSON.

Grid. Tables are generated at sea-level ISA (rho 1.225, mu 1.81e-5, a 340.3) over an rpm sweep that runs from just
above the BEMT Reynolds floor (Re75 >= 1e4) up to the rpm where the tip Mach number reaches 0.80, and a J axis from
0 up to where CT <= 0 at every rpm, so thrust rolls off to zero. Bilinear interpolation in (J, log Re75).

KNOWN APPROXIMATION: Mach number is not a table axis. Compressibility is taken at the sea-level Mach that goes with
each Re75 (rpm and tip speed follow from Re75 at sea-level density and viscosity). At altitude or in hot air the
real Mach for a given Re75 differs; the error is second order below the 0.70 tip-Mach limit.

Out of range is an error, not a clamp: `coefficients` (and so `thrust_n` and `torque_nm`) raise ValueError for J or
Re75 outside the table. Thrust never goes below zero (CT is clamped at 0 in `thrust_n`).
"""

import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any, Final

import numpy as np

from src.tools.propulsion.bemt import (PROP_AIRFOIL, PROP_MODEL_VERSION, REPRESENTATIVE_RE_MIN,
                                       REPRESENTATIVE_STATION, BladeGeometry, PropModelError, bemt_point,
                                       generic_blade)

logger: logging.Logger = logging.getLogger(__name__)

# Sea-level ISA the tables are generated at.
TABLE_RHO: Final[float] = 1.225
TABLE_MU: Final[float] = 1.81e-5
TABLE_SPEED_OF_SOUND: Final[float] = 340.3
TABLE_MAX_TIP_MACH: Final[float] = 0.80
RE_FLOOR_MARGIN: Final[float] = 1.1  # lowest table Re75 sits this far above the BEMT floor
N_RPM_NODES: Final[int] = 14  # geometric in rpm, which is geometric in Re75
N_J_NODES: Final[int] = 33
J_SCAN_STEP: Final[float] = 0.05  # step of the search for the zero-thrust J
J_SCAN_MAX: Final[float] = 6.0
J_TOP_MARGIN: Final[float] = 1.1
J_EXTEND_FACTOR: Final[float] = 1.25
MAX_J_EXTENSIONS: Final[int] = 6
PROBE_RPM_FRACTIONS: Final[tuple[float, ...]] = (0.0, 0.5, 1.0)  # where along the rpm sweep the J search runs
SECONDS_PER_MINUTE: Final[float] = 60.0
CACHE_DIR_ENV: Final[str] = "PROP_TABLE_CACHE_DIR"
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class PropellerTable:
    """CT and CP at `j[i]` and `reynolds75[k]` stored as `ct[k][i]` (one row per Re75)."""
    diameter_m: float
    pitch_m: float
    j: tuple[float, ...]
    reynolds75: tuple[float, ...]
    ct: tuple[tuple[float, ...], ...]
    cp: tuple[tuple[float, ...], ...]

    @cached_property
    def _arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (np.asarray(self.j), np.log(np.asarray(self.reynolds75)), np.asarray(self.ct), np.asarray(self.cp))

    def pitch_speed_ms(self, rpm: float) -> float:
        """Airspeed at which the geometric pitch gives zero advance slip: P n."""
        return self.pitch_m * rpm / SECONDS_PER_MINUTE

    def coefficients(self, j: float, reynolds75: float) -> tuple[float, float]:
        """(CT, CP) by bilinear interpolation; ValueError when J or Re75 lies outside the table."""
        j_axis, log_re_axis, ct, cp = self._arrays
        if not (math.isfinite(j) and j_axis[0] <= j <= j_axis[-1]):
            raise ValueError(f"J={j:.4g} outside the table range [{j_axis[0]:.4g}, {j_axis[-1]:.4g}] "
                             f"for {self.diameter_m} x {self.pitch_m} m")
        if not (math.isfinite(reynolds75) and reynolds75 > 0.0
                and log_re_axis[0] <= math.log(reynolds75) <= log_re_axis[-1]):
            raise ValueError(f"Re75={reynolds75:.4g} outside the table range [{self.reynolds75[0]:.4g}, "
                             f"{self.reynolds75[-1]:.4g}] for {self.diameter_m} x {self.pitch_m} m")
        log_re: float = math.log(reynolds75)
        i, fj = _cell(j_axis, j)
        k, fr = _cell(log_re_axis, log_re)
        return _blend(ct, k, i, fr, fj), _blend(cp, k, i, fr, fj)

    def _state(self, rpm: float, airspeed_ms: float, rho: float, mu: float) -> tuple[float, float, float]:
        """(n rev/s, CT, CP) at the operating point."""
        if not all(math.isfinite(x) for x in (rpm, airspeed_ms, rho, mu)) or min(rpm, rho, mu) <= 0.0 \
                or airspeed_ms < 0.0:
            raise ValueError(f"table needs finite rpm, rho, mu > 0 and airspeed >= 0, got rpm={rpm}, "
                             f"V={airspeed_ms}, rho={rho}, mu={mu}")
        n: float = rpm / SECONDS_PER_MINUTE
        re75: float = reynolds75_at(self.diameter_m, self.pitch_m, rpm, rho, mu)
        ct, cp = self.coefficients(airspeed_ms / (n * self.diameter_m), re75)
        return n, ct, cp

    def thrust_n(self, rpm: float, airspeed_ms: float, rho: float, mu: float) -> float:
        """Thrust in N, never negative; ValueError outside the table."""
        n, ct, _ = self._state(rpm, airspeed_ms, rho, mu)
        return max(ct, 0.0) * rho * n ** 2 * self.diameter_m ** 4

    def torque_nm(self, rpm: float, airspeed_ms: float, rho: float, mu: float) -> float:
        """Shaft torque in N m from CP as tabulated; ValueError outside the table."""
        n, _, cp = self._state(rpm, airspeed_ms, rho, mu)
        return cp * rho * n ** 2 * self.diameter_m ** 5 / (2.0 * math.pi)


def _cell(axis: np.ndarray, x: float) -> tuple[int, float]:
    """Lower node index and the fraction of the way to the next node."""
    idx: int = min(int(np.searchsorted(axis, x, side="right")) - 1, len(axis) - 2)
    return idx, float((x - axis[idx]) / (axis[idx + 1] - axis[idx]))


def _blend(values: np.ndarray, k: int, i: int, fr: float, fj: float) -> float:
    lo: float = values[k, i] * (1.0 - fj) + values[k, i + 1] * fj
    hi: float = values[k + 1, i] * (1.0 - fj) + values[k + 1, i + 1] * fj
    return float(lo * (1.0 - fr) + hi * fr)


@lru_cache(maxsize=None)
def _chord75_m(diameter_m: float, pitch_m: float) -> float:
    blade: BladeGeometry = generic_blade(diameter_m, pitch_m)
    return float(np.interp(REPRESENTATIVE_STATION, blade.stations, blade.chord_over_radius)) * diameter_m / 2.0


def reynolds75_at(diameter_m: float, pitch_m: float, rpm: float, rho: float, mu: float) -> float:
    """Reynolds number of the 0.75 R section from blade speed alone (no inflow)."""
    blade_speed: float = REPRESENTATIVE_STATION * math.pi * diameter_m * rpm / SECONDS_PER_MINUTE
    return rho * blade_speed * _chord75_m(diameter_m, pitch_m) / mu


def rpm_for_reynolds75(diameter_m: float, pitch_m: float, reynolds75: float, rho: float, mu: float) -> float:
    """Inverse of `reynolds75_at` (Re75 is proportional to rpm)."""
    return reynolds75 / reynolds75_at(diameter_m, pitch_m, 1.0, rho, mu)


def _rpm_sweep(diameter_m: float, pitch_m: float) -> np.ndarray:
    """Geometric rpm nodes from just above the Reynolds floor to tip Mach TABLE_MAX_TIP_MACH."""
    low: float = rpm_for_reynolds75(diameter_m, pitch_m, REPRESENTATIVE_RE_MIN * RE_FLOOR_MARGIN, TABLE_RHO,
                                    TABLE_MU)
    high: float = TABLE_MAX_TIP_MACH * TABLE_SPEED_OF_SOUND / (math.pi * diameter_m) * SECONDS_PER_MINUTE
    if high <= low:
        raise ValueError(f"{diameter_m} m propeller cannot span Re75 >= {REPRESENTATIVE_RE_MIN:.0e} to tip Mach "
                         f"{TABLE_MAX_TIP_MACH}")
    return np.geomspace(low, high, N_RPM_NODES)


def _coefficients_at(blade: BladeGeometry, rpm: float, j: float) -> tuple[float, float]:
    """(CT, CP) from one BEMT point at sea-level ISA."""
    n: float = rpm / SECONDS_PER_MINUTE
    d: float = blade.diameter_m
    point = bemt_point(blade, rpm, j * n * d, TABLE_RHO, TABLE_MU, TABLE_SPEED_OF_SOUND)
    return (point.thrust_n / (TABLE_RHO * n ** 2 * d ** 4), point.shaft_power_w / (TABLE_RHO * n ** 3 * d ** 5))


def _row(blade: BladeGeometry, rpm: float, j_axis: np.ndarray) -> tuple[list[float], list[float]]:
    """CT and CP along J at one rpm. A BEMT failure after CT has reached zero repeats the last achievable
    point with CT clamped to <= 0; a failure before that propagates with the prop size."""
    ct_row: list[float] = []
    cp_row: list[float] = []
    for j in j_axis:
        try:
            ct, cp = _coefficients_at(blade, rpm, float(j))
        except PropModelError as exc:
            if not ct_row or ct_row[-1] > 0.0:
                raise PropModelError(f"{blade.diameter_m} x {blade.pitch_m} m propeller: BEMT failed at "
                                     f"rpm {rpm:.0f}, J {j:.3f} before zero thrust: {exc}") from exc
            ct, cp = min(ct_row[-1], 0.0), cp_row[-1]
        ct_row.append(ct)
        cp_row.append(cp)
    return ct_row, cp_row


def _zero_thrust_j(blade: BladeGeometry, rpm: float) -> float:
    """First scanned J where CT <= 0 at this rpm (a failure after the last positive point ends the scan)."""
    scan: np.ndarray = np.arange(0.0, J_SCAN_MAX + J_SCAN_STEP, J_SCAN_STEP)
    last_j: float = 0.0
    for j in scan:
        try:
            ct, _ = _coefficients_at(blade, rpm, float(j))
        except PropModelError as exc:
            if j == 0.0:
                raise PropModelError(f"{blade.diameter_m} x {blade.pitch_m} m propeller: BEMT failed at rpm "
                                     f"{rpm:.0f}, J 0: {exc}") from exc
            return last_j + J_SCAN_STEP  # inflow breaks down only once thrust has gone
        if ct <= 0.0:
            return float(j)
        last_j = float(j)
    raise PropModelError(f"{blade.diameter_m} x {blade.pitch_m} m propeller: CT > 0 up to J {J_SCAN_MAX}")


def _build_rows(blade: BladeGeometry, rpms: np.ndarray, j_axis: np.ndarray) -> tuple[tuple[tuple[float, ...], ...],
                                                                                     tuple[tuple[float, ...], ...]]:
    rows = [_row(blade, float(rpm), j_axis) for rpm in rpms]
    return tuple(tuple(r[0]) for r in rows), tuple(tuple(r[1]) for r in rows)


def _build_table(diameter_m: float, pitch_m: float) -> PropellerTable:
    """Run BEMT over the (rpm, J) grid; widen the J axis until every rpm row ends at CT <= 0."""
    blade: BladeGeometry = generic_blade(diameter_m, pitch_m)
    rpms: np.ndarray = _rpm_sweep(diameter_m, pitch_m)
    probes: list[float] = [float(rpms[round(f * (len(rpms) - 1))]) for f in PROBE_RPM_FRACTIONS]
    j_top: float = J_TOP_MARGIN * max(_zero_thrust_j(blade, rpm) for rpm in probes)
    for _ in range(MAX_J_EXTENSIONS):
        j_axis: np.ndarray = np.linspace(0.0, j_top, N_J_NODES)
        ct, cp = _build_rows(blade, rpms, j_axis)
        if all(row[-1] <= 0.0 for row in ct):
            re75: tuple[float, ...] = tuple(reynolds75_at(diameter_m, pitch_m, float(r), TABLE_RHO, TABLE_MU)
                                            for r in rpms)
            return PropellerTable(diameter_m, pitch_m, tuple(float(x) for x in j_axis), re75, ct, cp)
        j_top *= J_EXTEND_FACTOR
    raise PropModelError(f"{diameter_m} x {pitch_m} m propeller: CT still positive at J {j_top:.2f}")


def cache_dir() -> Path:
    """Directory of the JSON tables: $PROP_TABLE_CACHE_DIR, else <repo>/.cache/prop_tables."""
    override: str | None = os.environ.get(CACHE_DIR_ENV)
    return Path(override) if override else REPO_ROOT / ".cache" / "prop_tables"


def _cache_path(diameter_m: float, pitch_m: float) -> Path:
    name: str = f"prop_v{PROP_MODEL_VERSION}_{PROP_AIRFOIL}_D{diameter_m!r}_P{pitch_m!r}.json"
    return cache_dir() / name


def _to_payload(table: PropellerTable) -> dict[str, Any]:
    return {"version": PROP_MODEL_VERSION, "airfoil": PROP_AIRFOIL, "diameter_m": table.diameter_m,
            "pitch_m": table.pitch_m, "j": list(table.j), "reynolds75": list(table.reynolds75),
            "ct": [list(r) for r in table.ct], "cp": [list(r) for r in table.cp]}


def _from_payload(payload: dict[str, Any], diameter_m: float, pitch_m: float) -> PropellerTable:
    """Validate a loaded payload against this prop and model; ValueError when stale or malformed."""
    expected: dict[str, Any] = {"version": PROP_MODEL_VERSION, "airfoil": PROP_AIRFOIL, "diameter_m": diameter_m,
                                "pitch_m": pitch_m}
    stale: list[str] = [key for key, value in expected.items() if payload.get(key) != value]
    if stale:
        raise ValueError(f"stale table, mismatched {', '.join(stale)}")
    table = PropellerTable(diameter_m, pitch_m, tuple(float(x) for x in payload["j"]),
                           tuple(float(x) for x in payload["reynolds75"]),
                           tuple(tuple(float(x) for x in r) for r in payload["ct"]),
                           tuple(tuple(float(x) for x in r) for r in payload["cp"]))
    shape: tuple[int, int] = (len(table.reynolds75), len(table.j))
    grids = (np.asarray(table.ct), np.asarray(table.cp))
    if min(shape) < 2 or any(g.shape != shape for g in grids) or not all(np.all(np.isfinite(g)) for g in grids):
        raise ValueError("malformed table grid")
    return table


def _load(path: Path, diameter_m: float, pitch_m: float) -> PropellerTable | None:
    """The cached table, or None (with a logged warning when the file exists but cannot be used)."""
    if not path.is_file():
        return None
    try:
        return _from_payload(json.loads(path.read_text(encoding="utf-8")), diameter_m, pitch_m)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:  # JSONDecodeError is a ValueError
        logger.warning("Propeller table cache %s unusable (%s); rebuilding", path, exc)
        return None


def _save(path: Path, table: PropellerTable) -> None:
    """Atomic write: temp file in the same directory, then os.replace. A failure is logged, not fatal."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, suffix=".tmp",
                                         delete=False) as handle:
            json.dump(_to_payload(table), handle)
        os.replace(handle.name, path)
    except OSError as exc:
        logger.warning("Could not write propeller table cache %s (%s); continuing without it", path, exc)


@lru_cache(maxsize=None)
def propeller_table(diameter_m: float, pitch_m: float) -> PropellerTable:
    """The table for this diameter and pitch: memory, else disk, else built (and written to disk)."""
    if not all(math.isfinite(x) and x > 0.0 for x in (diameter_m, pitch_m)):
        raise ValueError(f"propeller needs finite positive diameter and pitch, got D={diameter_m}, P={pitch_m}")
    path: Path = _cache_path(diameter_m, pitch_m)
    table: PropellerTable | None = _load(path, diameter_m, pitch_m)
    if table is None:
        table = _build_table(diameter_m, pitch_m)
        _save(path, table)
    return table
