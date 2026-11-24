"""Cached propeller CT/CP tables built from the BEMT model (spec 6.1).

A table holds CT = T / (rho n^2 D^4) and CP = P / (rho n^3 D^5) for one diameter and pitch, indexed by advance ratio
J = V / (n D), tip Mach number M = pi D n / a, and the equivalent density rho_eq = rho x (TABLE_MU / mu) x (a /
TABLE_SPEED_OF_SOUND): the density that gives the same Reynolds numbers at the table's reference viscosity and speed
of sound (at fixed M and J every section speed scales with a, so Re ~ rho M a / mu). The sizing loop and the motor
search evaluate it thousands of times, so one table is built once per prop (a few thousand BEMT points, solved as
one vectorised batch), kept in memory and written to disk as JSON.

Grid. Tip Mach and equivalent density are independent axes, so altitude and temperature are free, and EVERY cell is
a physical operating state: rpm from M (at the reference speed of sound 340.3 m/s, which fixes the section Mach
numbers exactly) and a real equivalent density between TABLE_THIN_RHO_EQ (thin, hot air: about 5 km ISA+20) and the
cold-day value (sea level at -30 C: rho 1.46, mu 1.55e-5, a 312.6 m/s) with a margin. An earlier (M, Re75) grid
implied rho = Re75 / (k rpm), so its corners asked BEMT for densities of 0.1 or 45 kg/m^3 and failed there. M runs
from the tip Mach where Re75 at the thinnest density sits just above the BEMT floor of 1e4 up to 0.80; J runs from 0
up to where CT <= 0 in every (M, rho_eq) row, so thrust rolls off to zero. Interpolation is trilinear in (J, log M,
log rho_eq).

Out of range is an error, not a clamp: `coefficients` (and so `thrust_n` and `torque_nm`) raise ValueError for J, M
or density outside the table. Thrust never goes below zero (CT is clamped at 0 in `thrust_n`). Past the J where BEMT
stops converging (beyond zero thrust) a row repeats its last computed point with CT <= 0; that extent is recorded
(`j_limit`) and `torque_nm` raises beyond it rather than return a made-up CP.
"""

import contextlib
import hashlib
import json
import logging
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any, Final, Optional, Sequence

import numpy as np

from src.tools.propulsion.bemt import (PROP_AIRFOIL, PROP_MODEL_VERSION, REPRESENTATIVE_RE_MIN,
                                       REPRESENTATIVE_STATION, BladeGeometry, PropBatch, PropModelError,
                                       bemt_points, generic_blade)

logger: logging.Logger = logging.getLogger(__name__)

# Reference air the (M, Re75) cells are solved in: only viscosity and the speed of sound are fixed, density follows
# from Re75, so the result is valid for any atmosphere inside the axes.
TABLE_MU: Final[float] = 1.81e-5
TABLE_SPEED_OF_SOUND: Final[float] = 340.3
# Thinnest air tabulated (equivalent density): 4 km ISA+20 is about 0.85, 5 km ISA+20 about 0.72.
TABLE_THIN_RHO_EQ: Final[float] = 0.6
# Cold dense air (sea level at -30 C) sets the top of the density axis.
TABLE_COLD_RHO: Final[float] = 1.46
TABLE_COLD_MU: Final[float] = 1.55e-5
TABLE_COLD_SPEED_OF_SOUND: Final[float] = 312.6
DENSE_AXIS_MARGIN: Final[float] = 1.05  # the density axis reaches this far past the cold day
# Equivalent density of the cold day (about 1.566) with the margin: the top of the density axis.
TABLE_DENSE_RHO_EQ: Final[float] = (DENSE_AXIS_MARGIN * TABLE_COLD_RHO * TABLE_MU / TABLE_COLD_MU
                                    * TABLE_COLD_SPEED_OF_SOUND / TABLE_SPEED_OF_SOUND)
TABLE_MAX_TIP_MACH: Final[float] = 0.80
RE_FLOOR_MARGIN: Final[float] = 1.1  # lowest table Re75 sits this far above the BEMT floor
N_RHO_NODES: Final[int] = 8  # geometric in equivalent density (about 15 % apart)
N_J_NODES: Final[int] = 33
# Geometric tip-Mach nodes about this far apart: along a fixed-density row Re75 is proportional to rpm, so this axis
# also carries the Reynolds effect (strongest at low rpm) and needs the old Re75 axis's spacing.
MACH_NODE_RATIO: Final[float] = 1.18
J_SCAN_STEP: Final[float] = 0.05  # step of the search for the zero-thrust J
J_SCAN_CHUNK: Final[int] = 30  # J values per scan batch
J_SCAN_MAX: Final[float] = 6.0
J_TOP_MARGIN: Final[float] = 1.1
J_EXTEND_FACTOR: Final[float] = 1.25
MAX_J_EXTENSIONS: Final[int] = 6
SECONDS_PER_MINUTE: Final[float] = 60.0
SPEC_HASH_LENGTH: Final[int] = 10
CACHE_DIR_ENV: Final[str] = "PROP_TABLE_CACHE_DIR"
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
AXIS_TOLERANCE: Final[float] = 1e-9

# Everything that shapes the grid: a change to any of these changes the hash in the cache key and payload.
GRID_SPEC: Final[str] = hashlib.sha1(repr((
    TABLE_MU, TABLE_SPEED_OF_SOUND, TABLE_THIN_RHO_EQ, TABLE_COLD_RHO, TABLE_COLD_MU, TABLE_COLD_SPEED_OF_SOUND,
    DENSE_AXIS_MARGIN, TABLE_MAX_TIP_MACH,
    RE_FLOOR_MARGIN, N_RHO_NODES, N_J_NODES, MACH_NODE_RATIO, J_SCAN_STEP, J_SCAN_CHUNK,
    J_SCAN_MAX, J_TOP_MARGIN, J_EXTEND_FACTOR, MAX_J_EXTENSIONS, REPRESENTATIVE_RE_MIN,
    REPRESENTATIVE_STATION)).encode()).hexdigest()[:SPEC_HASH_LENGTH]


@dataclass(frozen=True)
class PropellerTable:
    """CT and CP at `j[i]`, `tip_mach[m]` and `rho_eq[k]` stored as `ct[m][k][i]`. `j_limit[m][k]` is the last J
    computed by BEMT in that (M, rho_eq) row; beyond it CT is held at <= 0 and CP at its last computed value."""
    diameter_m: float
    pitch_m: float
    j: tuple[float, ...]
    tip_mach: tuple[float, ...]
    rho_eq: tuple[float, ...]  # equivalent density rho x TABLE_MU / mu, kg/m^3
    ct: tuple[tuple[tuple[float, ...], ...], ...]
    cp: tuple[tuple[tuple[float, ...], ...], ...]
    j_limit: tuple[tuple[float, ...], ...]

    @cached_property
    def _arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (np.asarray(self.j), np.asarray(self.tip_mach), np.log(np.asarray(self.rho_eq)),
                np.asarray(self.ct), np.asarray(self.cp), np.asarray(self.j_limit))

    def pitch_speed_ms(self, rpm: float) -> float:
        """Airspeed at which the geometric pitch gives zero advance slip: P n."""
        return self.pitch_m * rpm / SECONDS_PER_MINUTE

    def _locate(self, j: float, tip_mach: float, rho_eq: float) -> tuple[tuple[int, float], ...]:
        """Cells and fractions along (M, rho_eq, J); ValueError when any coordinate lies outside its axis."""
        j_axis, mach_axis, log_rho_axis = self._arrays[:3]
        where: str = f"for {self.diameter_m} x {self.pitch_m} m"
        if not (math.isfinite(j) and j_axis[0] <= j <= j_axis[-1]):
            raise ValueError(f"J={j:.4g} outside the table range [{j_axis[0]:.4g}, {j_axis[-1]:.4g}] {where}")
        if not (math.isfinite(tip_mach) and mach_axis[0] <= tip_mach <= mach_axis[-1]):
            raise ValueError(f"tip Mach {tip_mach:.4g} outside the table range [{mach_axis[0]:.4g}, "
                             f"{mach_axis[-1]:.4g}] {where}")
        if not (math.isfinite(rho_eq) and rho_eq > 0.0
                and log_rho_axis[0] - AXIS_TOLERANCE <= math.log(rho_eq) <= log_rho_axis[-1] + AXIS_TOLERANCE):
            raise ValueError(f"equivalent density {rho_eq:.4g} kg/m^3 outside the table range "
                             f"[{self.rho_eq[0]:.4g}, {self.rho_eq[-1]:.4g}] {where}")
        log_rho: float = min(max(math.log(rho_eq), log_rho_axis[0]), log_rho_axis[-1])
        # log M: along a fixed-density row Re75 is proportional to M, and the Reynolds effect is ~linear in log Re
        return (_cell(np.log(mach_axis), math.log(tip_mach)), _cell(log_rho_axis, log_rho), _cell(j_axis, j))

    def coefficients(self, j: float, tip_mach: float, rho_eq: float) -> tuple[float, float]:
        """(CT, CP) by trilinear interpolation; ValueError when J, tip Mach or density lies outside the table."""
        (m, fm), (k, fr), (i, fj) = self._locate(j, tip_mach, rho_eq)
        ct, cp = self._arrays[3], self._arrays[4]
        return _trilinear(ct, m, k, i, fm, fr, fj), _trilinear(cp, m, k, i, fm, fr, fj)

    def _computed_j_limit(self, tip_mach: float, rho_eq: float) -> float:
        """The smallest last-computed J among the four (M, rho_eq) rows around the point."""
        (m, _), (k, _), _ = self._locate(self.j[0], tip_mach, rho_eq)
        limits: np.ndarray = self._arrays[5]
        return float(limits[m:m + 2, k:k + 2].min())

    def _state(self, rpm: float, airspeed_ms: float, rho: float, mu: float,
               speed_of_sound_ms: float) -> tuple[float, float, float, float, float]:
        """(n rev/s, J, tip Mach, CT, CP) at the operating point."""
        values: tuple[float, ...] = (rpm, airspeed_ms, rho, mu, speed_of_sound_ms)
        if not all(math.isfinite(x) for x in values) or min(rpm, rho, mu, speed_of_sound_ms) <= 0.0 \
                or airspeed_ms < 0.0:
            raise ValueError(f"table needs finite rpm, rho, mu, a > 0 and airspeed >= 0, got rpm={rpm}, "
                             f"V={airspeed_ms}, rho={rho}, mu={mu}, a={speed_of_sound_ms}")
        n: float = rpm / SECONDS_PER_MINUTE
        j: float = airspeed_ms / (n * self.diameter_m)
        tip_mach: float = math.pi * self.diameter_m * n / speed_of_sound_ms
        ct, cp = self.coefficients(j, tip_mach, equivalent_density(rho, mu, speed_of_sound_ms))
        return n, j, tip_mach, ct, cp

    def thrust_n(self, rpm: float, airspeed_ms: float, rho: float, mu: float, speed_of_sound_ms: float) -> float:
        """Thrust in N, never negative (zero beyond the computed J); ValueError outside the table."""
        n, _, _, ct, _ = self._state(rpm, airspeed_ms, rho, mu, speed_of_sound_ms)
        return max(ct, 0.0) * rho * n ** 2 * self.diameter_m ** 4

    def past_computed_j(self, rpm: float, airspeed_ms: float, rho: float, mu: float,
                        speed_of_sound_ms: float) -> bool:
        """True where `torque_nm` would refuse the point for lying beyond the last J BEMT computed (past zero
        thrust); ValueError outside the table axes."""
        _, j, tip_mach, _, _ = self._state(rpm, airspeed_ms, rho, mu, speed_of_sound_ms)
        return j > self._computed_j_limit(tip_mach, equivalent_density(rho, mu, speed_of_sound_ms))

    def torque_nm(self, rpm: float, airspeed_ms: float, rho: float, mu: float, speed_of_sound_ms: float) -> float:
        """Shaft torque in N m from CP as tabulated; ValueError outside the table or beyond the J BEMT computed."""
        n, j, tip_mach, _, cp = self._state(rpm, airspeed_ms, rho, mu, speed_of_sound_ms)
        limit: float = self._computed_j_limit(tip_mach, equivalent_density(rho, mu, speed_of_sound_ms))
        if j > limit:
            raise ValueError(f"J={j:.4g} is beyond the last computed J {limit:.4g} (past zero thrust) for "
                             f"{self.diameter_m} x {self.pitch_m} m: torque is not tabulated there")
        return cp * rho * n ** 2 * self.diameter_m ** 5 / (2.0 * math.pi)


def _cell(axis: np.ndarray, x: float) -> tuple[int, float]:
    """Lower node index and the fraction of the way to the next node."""
    idx: int = min(int(np.searchsorted(axis, x, side="right")) - 1, len(axis) - 2)
    return idx, float((x - axis[idx]) / (axis[idx + 1] - axis[idx]))


def _trilinear(values: np.ndarray, m: int, k: int, i: int, fm: float, fr: float, fj: float) -> float:
    """Blend the eight corners of cell (m, k, i) of `values[mach][re][j]` with fractions (fm, fr, fj)."""
    cell: np.ndarray = values[m:m + 2, k:k + 2, i:i + 2]
    along_j: np.ndarray = cell[..., 0] * (1.0 - fj) + cell[..., 1] * fj
    along_re: np.ndarray = along_j[:, 0] * (1.0 - fr) + along_j[:, 1] * fr
    return float(along_re[0] * (1.0 - fm) + along_re[1] * fm)


@lru_cache(maxsize=None)
def _chord75_m(diameter_m: float, pitch_m: float) -> float:
    blade: BladeGeometry = generic_blade(diameter_m, pitch_m)
    return float(np.interp(REPRESENTATIVE_STATION, blade.stations, blade.chord_over_radius)) * diameter_m / 2.0


def equivalent_density(rho: float, mu: float, speed_of_sound_ms: float) -> float:
    """The density that gives the same Reynolds numbers at the same tip Mach and J in the table's reference air
    (TABLE_MU, TABLE_SPEED_OF_SOUND): rho x (TABLE_MU / mu) x (a / TABLE_SPEED_OF_SOUND)."""
    return rho * TABLE_MU / mu * speed_of_sound_ms / TABLE_SPEED_OF_SOUND


def reynolds75_at(diameter_m: float, pitch_m: float, rpm: float, rho: float, mu: float) -> float:
    """Reynolds number of the 0.75 R section from blade speed alone (no inflow)."""
    blade_speed: float = REPRESENTATIVE_STATION * math.pi * diameter_m * rpm / SECONDS_PER_MINUTE
    return rho * blade_speed * _chord75_m(diameter_m, pitch_m) / mu


def rpm_for_reynolds75(diameter_m: float, pitch_m: float, reynolds75: float, rho: float, mu: float) -> float:
    """Inverse of `reynolds75_at` (Re75 is proportional to rpm)."""
    return reynolds75 / reynolds75_at(diameter_m, pitch_m, 1.0, rho, mu)


def _rpm_for_tip_mach(diameter_m: float, tip_mach: float) -> float:
    return tip_mach * TABLE_SPEED_OF_SOUND / (math.pi * diameter_m) * SECONDS_PER_MINUTE


def _axes(diameter_m: float, pitch_m: float) -> tuple[np.ndarray, np.ndarray]:
    """(tip Mach nodes, equivalent-density nodes). The lowest Mach is where Re75 at the thinnest density sits just
    above the BEMT floor, so every cell is inside the airfoil polars; density spans thin hot air to a cold day."""
    re_low: float = REPRESENTATIVE_RE_MIN * RE_FLOOR_MARGIN
    rpm_low: float = rpm_for_reynolds75(diameter_m, pitch_m, re_low, TABLE_THIN_RHO_EQ, TABLE_MU)
    rpm_high: float = _rpm_for_tip_mach(diameter_m, TABLE_MAX_TIP_MACH)
    if rpm_high <= rpm_low:
        raise ValueError(f"{diameter_m} m propeller: tip Mach at Re75 {re_low:.3g} already reaches "
                         f"{TABLE_MAX_TIP_MACH}; no Mach range to tabulate")
    mach_low: float = math.pi * diameter_m * rpm_low / SECONDS_PER_MINUTE / TABLE_SPEED_OF_SOUND
    nodes: int = max(2, math.ceil(math.log(TABLE_MAX_TIP_MACH / mach_low) / math.log(MACH_NODE_RATIO)) + 1)
    mach: np.ndarray = np.geomspace(mach_low, TABLE_MAX_TIP_MACH, nodes)  # >= 2 nodes, ends exactly at the limit
    return mach, np.geomspace(TABLE_THIN_RHO_EQ, TABLE_DENSE_RHO_EQ, N_RHO_NODES)


@dataclass(frozen=True)
class _Cells:
    """(M, Re75) rows of one batch: rpm and density per row (shape (rows,))."""
    rpm: np.ndarray
    rho: np.ndarray


def _cells(diameter_m: float, mach: np.ndarray, densities: np.ndarray) -> _Cells:
    """rpm and density for every (M, rho_eq) pair, flattened Mach-major (solved at TABLE_MU, so rho = rho_eq)."""
    rpm: np.ndarray = np.repeat(_rpm_for_tip_mach(diameter_m, mach), len(densities))
    return _Cells(rpm=rpm, rho=np.tile(densities, len(mach)))


def _solve(blade: BladeGeometry, cells: _Cells, j_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(CT, CP, ok) with shape (rows, J): one BEMT batch over every cell row and J."""
    rpm: np.ndarray = np.repeat(cells.rpm, len(j_axis))
    rho: np.ndarray = np.repeat(cells.rho, len(j_axis))
    n: np.ndarray = rpm / SECONDS_PER_MINUTE
    d: float = blade.diameter_m
    airspeed: np.ndarray = np.tile(j_axis, len(cells.rpm)) * n * d
    batch: PropBatch = bemt_points(blade, rpm, airspeed, rho, TABLE_MU, TABLE_SPEED_OF_SOUND)
    shape: tuple[int, int] = (len(cells.rpm), len(j_axis))
    ct: np.ndarray = batch.thrust_n / (rho * n ** 2 * d ** 4)
    cp: np.ndarray = batch.shaft_power_w / (rho * n ** 3 * d ** 5)
    return ct.reshape(shape), cp.reshape(shape), batch.ok.reshape(shape)


def _clamp_row(ct: np.ndarray, cp: np.ndarray, ok: np.ndarray, j_axis: np.ndarray,
               label: str) -> tuple[np.ndarray, np.ndarray, float]:
    """(CT, CP, j_limit) for one row. If BEMT failed from some J on and CT had already reached <= 0 at the point
    before, the rest repeats that last computed point with CT held <= 0 and j_limit is its J; a failure before
    zero thrust raises PropModelError (no made-up number fills a hole)."""
    if ok.all():
        return ct, cp, float(j_axis[-1])
    first_bad: int = int(np.argmin(ok))
    if first_bad == 0 or ct[first_bad - 1] > 0.0:
        raise PropModelError(f"{label}: BEMT failed at J {j_axis[first_bad]:.3f} before zero thrust "
                             f"(last computed CT {ct[first_bad - 1] if first_bad else math.nan:.4g})")
    held_ct: np.ndarray = np.concatenate([ct[:first_bad], np.full(len(ct) - first_bad, min(ct[first_bad - 1], 0.0))])
    held_cp: np.ndarray = np.concatenate([cp[:first_bad], np.full(len(cp) - first_bad, cp[first_bad - 1])])
    return held_ct, held_cp, float(j_axis[first_bad - 1])


def _label(diameter_m: float, pitch_m: float, cells: _Cells, row: int) -> str:
    return (f"{diameter_m} x {pitch_m} m propeller, row {row} (rpm {cells.rpm[row]:.0f}, "
            f"rho {cells.rho[row]:.4g})")


def _zero_thrust_j(blade: BladeGeometry, cells: _Cells) -> float:
    """Largest first J with CT <= 0 over the probe rows, scanned in chunks of J_SCAN_CHUNK steps. A BEMT failure
    before a row's CT reaches zero raises PropModelError (the same rule as `_clamp_row`)."""
    zero: np.ndarray = np.full(len(cells.rpm), math.nan)
    start: int = 0
    while np.isnan(zero).any():
        if start * J_SCAN_STEP > J_SCAN_MAX:
            raise PropModelError(f"{blade.diameter_m} x {blade.pitch_m} m propeller: CT > 0 up to J {J_SCAN_MAX}")
        scan: np.ndarray = (start + np.arange(J_SCAN_CHUNK)) * J_SCAN_STEP
        ct, _, ok = _solve(blade, cells, scan)
        for row in np.flatnonzero(np.isnan(zero)):
            bad: np.ndarray = ~ok[row]
            first_bad: int = int(np.argmax(bad)) if bad.any() else len(scan)
            nonpositive: np.ndarray = ok[row] & (ct[row] <= 0.0)
            first_zero: int = int(np.argmax(nonpositive)) if nonpositive.any() else len(scan)
            if first_bad < first_zero:
                raise PropModelError(f"{_label(blade.diameter_m, blade.pitch_m, cells, int(row))}: BEMT failed at "
                                     f"J {scan[first_bad]:.3f} before zero thrust (zero-thrust search)")
            if first_zero < len(scan):
                zero[row] = scan[first_zero]
        start += J_SCAN_CHUNK
    return float(zero.max())


def _probe_rows(n_mach: int, n_re: int) -> np.ndarray:
    """Row indices (Mach-major) of the four corners and the centre of the (M, Re75) grid."""
    return np.array(sorted({m * n_re + k for m in (0, n_mach - 1, n_mach // 2) for k in (0, n_re - 1, n_re // 2)}))


def _build_table(diameter_m: float, pitch_m: float) -> PropellerTable:
    """Run BEMT over the (M, Re75, J) grid; widen the J axis until every row ends at CT <= 0."""
    blade: BladeGeometry = generic_blade(diameter_m, pitch_m)
    mach, densities = _axes(diameter_m, pitch_m)
    cells: _Cells = _cells(diameter_m, mach, densities)
    probes: np.ndarray = _probe_rows(len(mach), len(densities))
    j_top: float = J_TOP_MARGIN * _zero_thrust_j(blade, _Cells(cells.rpm[probes], cells.rho[probes]))
    for _ in range(MAX_J_EXTENSIONS):
        j_axis: np.ndarray = np.linspace(0.0, j_top, N_J_NODES)
        ct, cp, ok = _solve(blade, cells, j_axis)
        rows = [_clamp_row(ct[r], cp[r], ok[r], j_axis, _label(diameter_m, pitch_m, cells, r))
                for r in range(len(cells.rpm))]
        if all(row[0][-1] <= 0.0 for row in rows):
            return _assemble(diameter_m, pitch_m, j_axis, mach, densities, rows)
        j_top *= J_EXTEND_FACTOR
    raise PropModelError(f"{diameter_m} x {pitch_m} m propeller: CT still positive at J {j_top:.2f}")


def _assemble(diameter_m: float, pitch_m: float, j_axis: np.ndarray, mach: np.ndarray, densities: np.ndarray,
              rows: list[tuple[np.ndarray, np.ndarray, float]]) -> PropellerTable:
    shape: tuple[int, int, int] = (len(mach), len(densities), len(j_axis))
    ct: np.ndarray = np.stack([r[0] for r in rows]).reshape(shape)
    cp: np.ndarray = np.stack([r[1] for r in rows]).reshape(shape)
    limit: np.ndarray = np.array([r[2] for r in rows]).reshape(shape[:2])
    return PropellerTable(diameter_m, pitch_m, _floats(j_axis), _floats(mach), _floats(densities),
                          tuple(tuple(_floats(r) for r in plane) for plane in ct),
                          tuple(tuple(_floats(r) for r in plane) for plane in cp),
                          tuple(_floats(r) for r in limit))


def _floats(values: Any) -> tuple[float, ...]:
    return tuple(float(x) for x in values)


def cache_dir() -> Path:
    """Directory of the JSON tables: $PROP_TABLE_CACHE_DIR, else <repo>/.cache/prop_tables."""
    override: str | None = os.environ.get(CACHE_DIR_ENV)
    return Path(override) if override else REPO_ROOT / ".cache" / "prop_tables"


def _cache_path(diameter_m: float, pitch_m: float) -> Path:
    name: str = (f"prop_v{PROP_MODEL_VERSION}_{PROP_AIRFOIL}_g{GRID_SPEC}_D{float(diameter_m)!r}_"
                 f"P{float(pitch_m)!r}.json")
    return cache_dir() / name


def _to_payload(table: PropellerTable) -> dict[str, Any]:
    return {"version": PROP_MODEL_VERSION, "airfoil": PROP_AIRFOIL, "grid": GRID_SPEC,
            "diameter_m": table.diameter_m, "pitch_m": table.pitch_m, "j": list(table.j),
            "tip_mach": list(table.tip_mach), "rho_eq": list(table.rho_eq),
            "ct": [[list(r) for r in plane] for plane in table.ct],
            "cp": [[list(r) for r in plane] for plane in table.cp], "j_limit": [list(r) for r in table.j_limit]}


def _check_axes(table: PropellerTable) -> None:
    """ValueError unless the axes are finite and strictly increasing, J starts at 0, density > 0 and Mach is in
    range."""
    for name, axis in (("j", table.j), ("tip_mach", table.tip_mach), ("rho_eq", table.rho_eq)):
        values: np.ndarray = np.asarray(axis)
        if len(values) < 2 or not np.all(np.isfinite(values)) or not np.all(np.diff(values) > 0.0):
            raise ValueError(f"axis {name} must be finite and strictly increasing with >= 2 nodes")
    if table.j[0] != 0.0 or table.rho_eq[0] <= 0.0:
        raise ValueError("axis j must start at 0 and rho_eq must be positive")
    if table.tip_mach[0] <= 0.0 or table.tip_mach[-1] > TABLE_MAX_TIP_MACH + AXIS_TOLERANCE:
        raise ValueError(f"axis tip_mach must lie in (0, {TABLE_MAX_TIP_MACH}]")


def _from_payload(payload: dict[str, Any], diameter_m: float, pitch_m: float) -> PropellerTable:
    """Validate a loaded payload against this prop and grid; ValueError when stale or malformed."""
    expected: dict[str, Any] = {"version": PROP_MODEL_VERSION, "airfoil": PROP_AIRFOIL, "grid": GRID_SPEC,
                                "diameter_m": float(diameter_m), "pitch_m": float(pitch_m)}
    stale: list[str] = [key for key, value in expected.items() if payload.get(key) != value]
    if stale:
        raise ValueError(f"stale table, mismatched {', '.join(stale)}")
    table = PropellerTable(
        float(diameter_m), float(pitch_m), _floats(payload["j"]), _floats(payload["tip_mach"]),
        _floats(payload["rho_eq"]), tuple(tuple(_floats(r) for r in plane) for plane in payload["ct"]),
        tuple(tuple(_floats(r) for r in plane) for plane in payload["cp"]),
        tuple(_floats(r) for r in payload["j_limit"]))
    _check_axes(table)
    shape: tuple[int, int, int] = (len(table.tip_mach), len(table.rho_eq), len(table.j))
    grids: tuple[np.ndarray, ...] = (np.asarray(table.ct), np.asarray(table.cp))
    limits: np.ndarray = np.asarray(table.j_limit)
    if any(g.shape != shape for g in grids) or limits.shape != shape[:2]:
        raise ValueError("malformed table grid")
    if not all(np.all(np.isfinite(g)) for g in (*grids, limits)) or np.any(limits < 0.0) \
            or np.any(limits > table.j[-1]):
        raise ValueError("table values must be finite and j_limit inside the J axis")
    return table


def _load(path: Path, diameter_m: float, pitch_m: float) -> PropellerTable | None:
    """The cached table, or None (with a logged warning when the file exists but cannot be used)."""
    if not path.is_file():
        return None
    try:
        return _from_payload(json.loads(path.read_text(encoding="utf-8")), diameter_m, pitch_m)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as exc:
        logger.warning("Propeller table cache %s unusable (%s); rebuilding", path, exc)
        return None


def _save(path: Path, table: PropellerTable) -> None:
    """Atomic write: temp file in the same directory, then os.replace; the temp file never outlives a failure.
    A failure is logged, not fatal."""
    temp_name: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, suffix=".tmp",
                                         delete=False) as handle:
            temp_name = handle.name
            json.dump(_to_payload(table), handle)
        os.replace(temp_name, path)
        temp_name = None
    except (OSError, TypeError, ValueError, OverflowError) as exc:
        logger.warning("Could not write propeller table cache %s (%s); continuing without it", path, exc)
    finally:
        if temp_name is not None:
            with contextlib.suppress(OSError):
                Path(temp_name).unlink(missing_ok=True)


@lru_cache(maxsize=None)
def propeller_table(diameter_m: float, pitch_m: float) -> PropellerTable:
    """The table for this diameter and pitch: memory, else disk, else built (and written to disk)."""
    if not all(math.isfinite(x) and x > 0.0 for x in (diameter_m, pitch_m)):
        raise ValueError(f"propeller needs finite positive diameter and pitch, got D={diameter_m}, P={pitch_m}")
    path: Path = _cache_path(diameter_m, pitch_m)
    table: PropellerTable | None = _load(path, diameter_m, pitch_m)
    if table is None:
        table = _build_table(float(diameter_m), float(pitch_m))
        _save(path, table)
    return table


# ---------------------------------------------------------------- cache warming (Docker / CI / first use)

WARM_FLAG: Final[str] = "--warm"


@dataclass(frozen=True)
class WarmSummary:
    distinct: int  # distinct (diameter, pitch) pairs in the catalogue
    failed: tuple[str, ...]  # "<D>x<P> m: <reason>" per table that could not be built
    seconds: float


def _catalogue_propellers() -> Sequence[Any]:
    from src.tools.database.seed_data import catalog_rows  # lazy: the catalogue is not a table-model dependency

    return catalog_rows("propellers")


def warm_cache() -> WarmSummary:
    """Build (or load) the table of every distinct catalogue propeller, so a first sizing run starts warm."""
    start: float = time.perf_counter()
    sizes: list[tuple[float, float]] = sorted({(p.diameter_m, p.pitch_m) for p in _catalogue_propellers()})
    failed: list[str] = []
    for diameter_m, pitch_m in sizes:
        try:
            propeller_table(diameter_m, pitch_m)
        except (PropModelError, ValueError) as exc:
            failed.append(f"{diameter_m:g}x{pitch_m:g} m: {exc}")
    return WarmSummary(len(sizes), tuple(failed), time.perf_counter() - start)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """`python -m src.tools.propulsion.prop_tables --warm`: warm the table cache; exit 1 if any table failed."""
    args: list[str] = list(sys.argv[1:] if argv is None else argv)
    if args != [WARM_FLAG]:
        raise SystemExit(f"usage: python -m src.tools.propulsion.prop_tables {WARM_FLAG}")
    summary: WarmSummary = warm_cache()
    print(f"[Tool] prop_tables {WARM_FLAG}: {summary.distinct} distinct propeller tables in {cache_dir()} "
          f"({summary.seconds:.1f} s, {len(summary.failed)} failed)")
    for line in summary.failed:
        print(f"[Tool] prop_tables failed: {line}")
    if summary.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
