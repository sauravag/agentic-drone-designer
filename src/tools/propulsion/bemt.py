"""Blade-element momentum (BEMT) propeller model on a generic blade (spec 6.1).

The blade is defined by diameter and pitch only: a fixed chord distribution (c/R vs r/R) and twist from constant
geometric pitch, beta(r) = atan(P / (2 pi r)), with one generic thin propeller airfoil. Section lift and drag come
from NeuralFoil, swept once over alpha per (Reynolds, Mach) node of a small grid and interpolated, so NeuralFoil is
never called per section or per iteration. Prandtl tip and hub losses. All SI.

The inflow angle phi at each station solves the momentum/blade-element residual of Ning (2014), multiplied through
by sin(phi) so it stays finite at zero airspeed (hover):
    R(phi) = sin^2(phi) - lambda sin(phi) cos(phi) - sigma' (cn + lambda ct) / (4 F),   lambda = V / (Omega r)
For a working propeller R < 0 as phi -> 0 and R > 0 at phi = pi/2, and bisection on (0, pi/2) converges; the
bracket is checked at every station and a missing sign change raises `PropModelError`.

Every failure (bad input, no bracket, Reynolds or Mach outside the polar grid, non-finite or non-positive result)
raises a readable error instead of returning a silently wrong point.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Final

import aerosandbox as asb
import numpy as np

PROP_AIRFOIL: Final[str] = "naca4409"  # generic thin cambered propeller section (chosen against the UIUC gate)
PROP_MODEL_VERSION: Final[int] = 1  # bump when the blade, airfoil or solver changes (keys the on-disk tables)

HUB_FRACTION: Final[float] = 0.15  # r/R where the blade starts (spinner / hub)
N_ELEMENTS: Final[int] = 30  # equal-width blade elements from hub to tip; stations are their midpoints
MIN_STATIONS: Final[int] = 2  # element widths come from neighbouring midpoints
# Generic planform, c/R at r/R knots: a typical thin-electric blade (widest near 0.4 R, tapering to a narrow tip),
# set against the UIUC validation gate (tests/test_propeller_bemt.py) together with the airfoil.
CHORD_KNOT_STATIONS: Final[tuple[float, ...]] = (0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
CHORD_KNOT_VALUES: Final[tuple[float, ...]] = (0.154, 0.171, 0.209, 0.220, 0.215, 0.193, 0.160, 0.121, 0.088, 0.044)

# Polar grid: one NeuralFoil alpha sweep per (Reynolds, Mach) node. A section outside the Reynolds or Mach range
# raises PropModelError (no extrapolation, no clamping). The top Mach node sits above the tip-Mach limit (0.70).
# NeuralFoil is credible from Re 1e4: the representative section (r/R 0.75) must reach it, while inner stations,
# which carry ~1 % of the load, may use the 1e3 and 3e3 nodes (controller ruling, ledger Task 1, option A).
REPRESENTATIVE_STATION: Final[float] = 0.75
REPRESENTATIVE_RE_MIN: Final[float] = 1e4
POLAR_RE_NODES: Final[tuple[float, ...]] = (1e3, 3e3, 1e4, 2e4, 4e4, 7e4, 1.2e5, 2e5, 4e5, 1e6, 3e6, 1e7, 3e7)
POLAR_MACH_NODES: Final[tuple[float, ...]] = (0.0, 0.25, 0.5, 0.7, 0.85)
POLAR_ALPHA_DEG: Final[np.ndarray] = np.linspace(-90.0, 90.0, 181)  # spans every alpha = beta - phi possible

PHI_MIN_RAD: Final[float] = 1e-6  # lower bisection bound (phi = 0 is singular in the loss factors)
BISECTION_STEPS: Final[int] = 60  # halves (0, pi/2) to ~1e-18 rad
# Reynolds/Mach use the relative speed W, which depends on the solution: iterate W to a fixed point.
SPEED_REL_TOLERANCE: Final[float] = 1e-6
MAX_SPEED_PASSES: Final[int] = 20
LOSS_FACTOR_MIN: Final[float] = 1e-9  # keeps the residual finite where the Prandtl factor reaches zero
SECONDS_PER_MINUTE: Final[float] = 60.0

SectionCoefficients = Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


class PropModelError(RuntimeError):
    """The BEMT model could not produce a valid operating point."""


@dataclass(frozen=True)
class BladeGeometry:
    """Blade by diameter and pitch; `stations` are element midpoints (r/R), chord as c/R at each station."""
    diameter_m: float
    pitch_m: float
    n_blades: int
    stations: tuple[float, ...]
    chord_over_radius: tuple[float, ...]

    def __post_init__(self) -> None:
        _validate_blade(self)

    @property
    def twist_rad(self) -> tuple[float, ...]:
        """Section pitch angle from constant geometric pitch: beta = atan(P / (2 pi r))."""
        radius: float = self.diameter_m / 2.0
        return tuple(math.atan(self.pitch_m / (2.0 * math.pi * s * radius)) for s in self.stations)


@dataclass(frozen=True)
class PropPoint:
    """One operating point. `efficiency` = T V / P_shaft when V > 0 and P_shaft > 0, else 0.0.
    `is_windmilling` is True when the air drives the propeller (V > 0 and P_shaft <= 0); such a point is not a
    propulsive operating point and its efficiency 0.0 is a placeholder."""
    thrust_n: float
    torque_nm: float
    shaft_power_w: float
    efficiency: float
    is_windmilling: bool


def _element_edges(stations: np.ndarray) -> np.ndarray:
    """Element edges (r/R) from midpoints: halfway between stations, extrapolated at hub and tip."""
    interior: np.ndarray = (stations[1:] + stations[:-1]) / 2.0
    return np.concatenate([[2.0 * stations[0] - interior[0]], interior, [2.0 * stations[-1] - interior[-1]]])


def _validate_blade(blade: BladeGeometry) -> None:
    sizes: tuple[float, float] = (blade.diameter_m, blade.pitch_m)
    if not all(math.isfinite(x) and x > 0.0 for x in sizes) or blade.n_blades < 1:
        raise ValueError(f"blade needs finite positive diameter and pitch and >= 1 blade, got "
                         f"D={blade.diameter_m} m, P={blade.pitch_m} m, B={blade.n_blades}")
    stations: np.ndarray = np.asarray(blade.stations, dtype=float)
    chords: np.ndarray = np.asarray(blade.chord_over_radius, dtype=float)
    if len(stations) < MIN_STATIONS or len(stations) != len(chords):
        raise ValueError(f"blade needs >= {MIN_STATIONS} stations and one chord per station, got "
                         f"{len(stations)} stations and {len(chords)} chords")
    if not (np.all(np.isfinite(chords)) and np.all(chords > 0.0)):
        raise ValueError(f"blade chords (c/R) must be finite and positive, got {blade.chord_over_radius}")
    edges: np.ndarray = _element_edges(stations)
    if not (np.all(np.isfinite(stations)) and np.all(np.diff(stations) > 0.0) and edges[0] > 0.0
            and edges[-1] <= 1.0 + 1e-9):
        raise ValueError(f"blade stations (r/R) must increase and their elements lie in (0, 1], got "
                         f"{blade.stations}")


def generic_blade(diameter_m: float, pitch_m: float, n_blades: int = 2) -> BladeGeometry:
    """The generic blade at this diameter and pitch (same planform and airfoil for every propeller)."""
    width: float = (1.0 - HUB_FRACTION) / N_ELEMENTS
    stations: np.ndarray = HUB_FRACTION + width * (np.arange(N_ELEMENTS) + 0.5)
    chords: np.ndarray = np.interp(stations, CHORD_KNOT_STATIONS, CHORD_KNOT_VALUES)
    return BladeGeometry(diameter_m=diameter_m, pitch_m=pitch_m, n_blades=n_blades,
                         stations=tuple(float(s) for s in stations),
                         chord_over_radius=tuple(float(c) for c in chords))


@lru_cache(maxsize=None)
def _alpha_sweep(airfoil: str, reynolds: float, mach: float) -> tuple[np.ndarray, np.ndarray]:
    """(CL, CD) over POLAR_ALPHA_DEG from one NeuralFoil call (360-degree effects blended in)."""
    aero: dict = asb.Airfoil(airfoil).get_aero_from_neuralfoil(alpha=POLAR_ALPHA_DEG, Re=reynolds, mach=mach)
    return np.asarray(aero["CL"], dtype=float), np.asarray(aero["CD"], dtype=float)


def _check_in_grid(reynolds: np.ndarray, mach: np.ndarray) -> None:
    """Raise when any section's Reynolds or Mach number lies outside the polar grid."""
    if not (np.all(reynolds >= POLAR_RE_NODES[0]) and np.all(reynolds <= POLAR_RE_NODES[-1])):
        raise PropModelError(f"section Reynolds number {float(np.min(reynolds)):.3g}..{float(np.max(reynolds)):.3g}"
                             f" is outside the polar grid {POLAR_RE_NODES[0]:.0e}..{POLAR_RE_NODES[-1]:.0e}")
    if not np.all(mach <= POLAR_MACH_NODES[-1]):
        raise PropModelError(f"section Mach number {float(np.max(mach)):.3f} exceeds the polar grid top "
                             f"{POLAR_MACH_NODES[-1]} (compressible tip; lower the rpm)")


@lru_cache(maxsize=None)
def _section_coefficients(airfoil: str) -> SectionCoefficients:
    """Trilinear (log Re, Mach, alpha) -> (CL, CD) over the polar grid, for arrays of any shape; out-of-grid
    Reynolds or Mach raises. Alpha is on a uniform grid, so it is indexed arithmetically."""
    sweeps: list[list[tuple[np.ndarray, np.ndarray]]] = [[_alpha_sweep(airfoil, re, m) for m in POLAR_MACH_NODES]
                                                         for re in POLAR_RE_NODES]
    cl_flat: np.ndarray = np.array([[s[0] for s in row] for row in sweeps]).ravel()
    cd_flat: np.ndarray = np.array([[s[1] for s in row] for row in sweeps]).ravel()
    log_re_nodes: np.ndarray = np.log10(POLAR_RE_NODES)
    mach_nodes: np.ndarray = np.asarray(POLAR_MACH_NODES)
    n_alpha: int = len(POLAR_ALPHA_DEG)
    mach_stride: int = n_alpha
    re_stride: int = len(mach_nodes) * n_alpha
    alpha_step: float = float(POLAR_ALPHA_DEG[1] - POLAR_ALPHA_DEG[0])

    def coefficients(reynolds: np.ndarray, mach: np.ndarray, alpha_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _check_in_grid(reynolds, mach)
        ir, fr = _node_fraction(log_re_nodes, np.log10(reynolds))
        im, fm = _node_fraction(mach_nodes, mach)
        ia, fa = _node_fraction(POLAR_ALPHA_DEG, np.clip(alpha_deg, POLAR_ALPHA_DEG[0], POLAR_ALPHA_DEG[-1]),
                                step=alpha_step)
        base: np.ndarray = ir * re_stride + im * mach_stride + ia
        return (_trilinear(cl_flat, base, re_stride, mach_stride, fr, fm, fa),
                _trilinear(cd_flat, base, re_stride, mach_stride, fr, fm, fa))

    return coefficients


def _node_fraction(nodes: np.ndarray, x: np.ndarray, step: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Lower node index and fraction towards the next node (arithmetic when the nodes are uniform with `step`)."""
    if step is None:
        index: np.ndarray = np.clip(np.searchsorted(nodes, x, side="right") - 1, 0, len(nodes) - 2)
    else:
        index = np.clip(np.floor((x - nodes[0]) / step).astype(int), 0, len(nodes) - 2)
    return index, (x - nodes[index]) / (nodes[index + 1] - nodes[index])


def _trilinear(flat: np.ndarray, base: np.ndarray, re_stride: int, mach_stride: int, fr: np.ndarray, fm: np.ndarray,
               fa: np.ndarray) -> np.ndarray:
    """Blend the eight table corners around flat index `base` (alpha stride 1)."""
    def along_alpha(offset: int) -> np.ndarray:
        return flat[base + offset] * (1.0 - fa) + flat[base + offset + 1] * fa

    low: np.ndarray = along_alpha(0) * (1.0 - fm) + along_alpha(mach_stride) * fm
    high: np.ndarray = along_alpha(re_stride) * (1.0 - fm) + along_alpha(re_stride + mach_stride) * fm
    return low * (1.0 - fr) + high * fr


@dataclass(frozen=True)
class _Stations:
    """Per-station quantities. Shapes: (stations,) except `inflow_ratio`, which is (points, stations)."""
    r_frac: np.ndarray
    radius_m: np.ndarray
    hub_frac: float
    width_m: np.ndarray
    chord_m: np.ndarray
    twist: np.ndarray
    solidity: np.ndarray  # local solidity sigma' = B c / (2 pi r)
    inflow_ratio: np.ndarray  # lambda = V / (Omega r)


@dataclass(frozen=True)
class _Section:
    """Section state at an inflow angle: force coefficients along thrust (cn) and torque (ct), and the loss F."""
    cn: np.ndarray
    ct: np.ndarray
    loss: np.ndarray


class _Failures:
    """Per-point failure reasons for a batch: the first reason wins and the point is excluded from later checks."""

    def __init__(self, count: int) -> None:
        self._reasons: list[str | None] = [None] * count

    def flag(self, mask: np.ndarray, reason: Callable[[int], str]) -> None:
        """Record `reason(point)` for every point in `mask` (full-batch boolean) that has not failed yet."""
        for point in np.flatnonzero(mask):
            if self._reasons[point] is None:
                self._reasons[point] = reason(int(point))

    @property
    def mask(self) -> np.ndarray:
        return np.array([r is not None for r in self._reasons], dtype=bool)

    @property
    def reasons(self) -> tuple[str | None, ...]:
        return tuple(self._reasons)


@dataclass(frozen=True)
class PropBatch:
    """Operating points of `bemt_points`. A point that failed has `errors[i]` set and NaN in every value array;
    `ok[i]` is False for it. `is_windmilling` follows `PropPoint`."""
    thrust_n: np.ndarray
    torque_nm: np.ndarray
    shaft_power_w: np.ndarray
    efficiency: np.ndarray
    is_windmilling: np.ndarray
    errors: tuple[str | None, ...]

    @property
    def ok(self) -> np.ndarray:
        return np.array([e is None for e in self.errors], dtype=bool)


def _stations(blade: BladeGeometry, omega: np.ndarray, airspeed_ms: np.ndarray) -> _Stations:
    tip_radius: float = blade.diameter_m / 2.0
    r_frac: np.ndarray = np.asarray(blade.stations)
    edges: np.ndarray = _element_edges(r_frac)
    radius: np.ndarray = r_frac * tip_radius
    chord: np.ndarray = np.asarray(blade.chord_over_radius) * tip_radius
    return _Stations(r_frac=r_frac, radius_m=radius, hub_frac=float(edges[0]), width_m=np.diff(edges) * tip_radius,
                     chord_m=chord, twist=np.asarray(blade.twist_rad),
                     solidity=blade.n_blades * chord / (2.0 * math.pi * radius),
                     inflow_ratio=airspeed_ms[:, None] / (omega[:, None] * radius))


def _prandtl_loss(n_blades: int, stations: _Stations, sin_phi: np.ndarray) -> np.ndarray:
    """Combined Prandtl tip and hub loss factor F."""
    r_frac: np.ndarray = stations.r_frac
    f_tip: np.ndarray = n_blades / 2.0 * (1.0 - r_frac) / (r_frac * sin_phi)
    f_hub: np.ndarray = n_blades / 2.0 * (r_frac - stations.hub_frac) / (stations.hub_frac * sin_phi)
    loss: np.ndarray = (2.0 / math.pi) ** 2 * np.arccos(np.exp(-f_tip)) * np.arccos(np.exp(-f_hub))
    return np.maximum(loss, LOSS_FACTOR_MIN)


def _section(stations: _Stations, n_blades: int, phi: np.ndarray, reynolds: np.ndarray, mach: np.ndarray,
             coefficients: SectionCoefficients) -> _Section:
    cl, cd = coefficients(reynolds, mach, np.degrees(stations.twist - phi))
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    return _Section(cn=cl * cos_phi - cd * sin_phi, ct=cl * sin_phi + cd * cos_phi,
                    loss=_prandtl_loss(n_blades, stations, sin_phi))


def _residual(stations: _Stations, phi: np.ndarray, section: _Section) -> np.ndarray:
    lam: np.ndarray = stations.inflow_ratio
    return (np.sin(phi) ** 2 - lam * np.sin(phi) * np.cos(phi)
            - stations.solidity * (section.cn + lam * section.ct) / (4.0 * section.loss))


def _solve_inflow(stations: _Stations, n_blades: int, reynolds: np.ndarray, mach: np.ndarray,
                  coefficients: SectionCoefficients) -> tuple[np.ndarray, np.ndarray]:
    """(phi, bracketed): inflow angle per (point, station) by vectorised bisection on (0, pi/2), and where the
    residual changes sign on that interval (phi is meaningless where it does not)."""
    def residual(phi: np.ndarray) -> np.ndarray:
        return _residual(stations, phi, _section(stations, n_blades, phi, reynolds, mach, coefficients))

    low: np.ndarray = np.full(reynolds.shape, PHI_MIN_RAD)
    high: np.ndarray = np.full(reynolds.shape, math.pi / 2.0)
    bracketed: np.ndarray = (residual(low) < 0.0) & (residual(high) > 0.0)
    for _ in range(BISECTION_STEPS):
        mid: np.ndarray = (low + high) / 2.0
        negative: np.ndarray = residual(mid) < 0.0
        low, high = np.where(negative, mid, low), np.where(negative, high, mid)
    return (low + high) / 2.0, bracketed


def _relative_speed(stations: _Stations, omega: np.ndarray, phi: np.ndarray, section: _Section) -> np.ndarray:
    """Relative speed W from the tangential balance W (cos phi + sigma' ct / (4 F sin phi)) = Omega r;
    `omega` is a (points, 1) column."""
    swirl_term: np.ndarray = stations.solidity * section.ct / (4.0 * section.loss * np.sin(phi))
    return omega * stations.radius_m / (np.cos(phi) + swirl_term)


def _validate_conditions(rpm: np.ndarray, airspeed_ms: np.ndarray, rho: np.ndarray, mu: np.ndarray,
                         speed_of_sound_ms: np.ndarray) -> None:
    values: tuple[np.ndarray, ...] = (rpm, airspeed_ms, rho, mu, speed_of_sound_ms)
    positive: bool = all(np.all(x > 0.0) for x in (rpm, rho, mu, speed_of_sound_ms))
    if not all(np.all(np.isfinite(x)) for x in values) or not positive or np.any(airspeed_ms < 0.0):
        raise ValueError(f"bemt needs finite rpm, rho, mu, a > 0 and airspeed >= 0, got rpm={rpm}, "
                         f"V={airspeed_ms}, rho={rho}, mu={mu}, a={speed_of_sound_ms}")


def _grid_reason(reynolds: np.ndarray, mach: np.ndarray) -> str:
    """The `_check_in_grid` message for one point (a row of stations)."""
    try:
        _check_in_grid(reynolds, mach)
    except PropModelError as exc:
        return str(exc)
    return "section Reynolds or Mach number is not finite"


def _bracket_reason(stations: _Stations, bracketed_row: np.ndarray) -> str:
    bad: float = float(stations.r_frac[np.argmin(bracketed_row)])
    return (f"BEMT inflow residual has no sign change on (0, pi/2) at r/R = {bad:.3f} "
            f"({int(np.sum(~bracketed_row))} station(s)); no momentum solution there")


def _speed_reason(stations: _Stations, speed_row: np.ndarray) -> str:
    bad: int = int(np.argmin(np.isfinite(speed_row) & (speed_row > 0.0)))
    return (f"BEMT relative speed is {float(speed_row[bad])} m/s at r/R = {float(stations.r_frac[bad]):.3f}; "
            f"the swirl balance has no physical solution")


def _representative_reynolds(stations: _Stations, reynolds: np.ndarray) -> np.ndarray:
    """Reynolds number at r/R 0.75 per point (linear between the stations around it)."""
    r_frac: np.ndarray = stations.r_frac
    lower: int = int(np.clip(np.searchsorted(r_frac, REPRESENTATIVE_STATION, side="right") - 1, 0, len(r_frac) - 2))
    weight: float = float(np.clip((REPRESENTATIVE_STATION - r_frac[lower]) / (r_frac[lower + 1] - r_frac[lower]),
                                  0.0, 1.0))
    return reynolds[:, lower] * (1.0 - weight) + reynolds[:, lower + 1] * weight


def _subset(stations: _Stations, rows: np.ndarray) -> _Stations:
    return replace(stations, inflow_ratio=stations.inflow_ratio[rows])


@dataclass(frozen=True)
class _Conditions:
    """Per-point operating conditions as (points, 1) columns."""
    omega: np.ndarray
    rho: np.ndarray
    mu: np.ndarray
    sound: np.ndarray

    def rows(self, rows: np.ndarray) -> "_Conditions":
        return _Conditions(self.omega[rows], self.rho[rows], self.mu[rows], self.sound[rows])


def _scatter(size: int, rows: np.ndarray, sub_mask: np.ndarray) -> np.ndarray:
    full: np.ndarray = np.zeros(size, dtype=bool)
    full[rows] = sub_mask
    return full


def _converged_state(blade: BladeGeometry, stations: _Stations, omega: np.ndarray, airspeed_ms: np.ndarray,
                     rho: np.ndarray, mu: np.ndarray, speed_of_sound_ms: np.ndarray,
                     failures: _Failures) -> tuple[_Section, np.ndarray]:
    """(section, W) per point with Reynolds and Mach from the converged W: start from the geometric speed, solve
    phi, update W, repeat until W changes by less than SPEED_REL_TOLERANCE. The returned section was evaluated at
    that converged W, so loads built from it are consistent. Points that fail are recorded in `failures` and left
    at zero; converged points drop out of later passes."""
    coefficients: SectionCoefficients = _section_coefficients(PROP_AIRFOIL)
    count: int = len(omega)
    cond: _Conditions = _Conditions(omega[:, None], rho[:, None], mu[:, None], speed_of_sound_ms[:, None])
    speed: np.ndarray = np.hypot(airspeed_ms[:, None], cond.omega * stations.radius_m)
    shape: tuple[int, int] = speed.shape
    out: dict[str, np.ndarray] = {key: np.zeros(shape) for key in ("cn", "ct", "loss", "speed")}
    done: np.ndarray = np.zeros(count, dtype=bool)
    for _ in range(MAX_SPEED_PASSES):
        rows: np.ndarray = np.flatnonzero(~(done | failures.mask))
        if len(rows) == 0:
            break
        converged: np.ndarray = _speed_pass(blade, _subset(stations, rows), cond.rows(rows), speed, rows, coefficients,
                                            failures, out)
        done |= _scatter(count, rows, converged)
    failures.flag(~done, lambda _p: f"BEMT relative speed did not converge in {MAX_SPEED_PASSES} Reynolds/Mach passes")
    return _Section(cn=out["cn"], ct=out["ct"], loss=out["loss"]), out["speed"]


def _speed_pass(blade: BladeGeometry, stations: _Stations, cond: _Conditions, speed: np.ndarray, rows: np.ndarray,
                coefficients: SectionCoefficients, failures: _Failures, out: dict[str, np.ndarray]) -> np.ndarray:
    """One Reynolds/Mach pass for the points `rows`: updates `speed` in place for them, stores converged results in
    `out`, flags failures; returns which of `rows` converged this pass."""
    count: int = len(speed)
    reynolds: np.ndarray = cond.rho * speed[rows] * stations.chord_m / cond.mu
    mach: np.ndarray = speed[rows] / cond.sound
    in_grid: np.ndarray = (np.all(reynolds >= POLAR_RE_NODES[0], axis=1) & np.all(reynolds <= POLAR_RE_NODES[-1], axis=1)
                           & np.all(mach <= POLAR_MACH_NODES[-1], axis=1))
    failures.flag(_scatter(count, rows, ~in_grid), lambda p: _grid_reason(reynolds[_at(rows, p)], mach[_at(rows, p)]))
    safe_re: np.ndarray = np.clip(np.nan_to_num(reynolds, nan=POLAR_RE_NODES[0]), POLAR_RE_NODES[0], POLAR_RE_NODES[-1])
    safe_mach: np.ndarray = np.clip(np.nan_to_num(mach), 0.0, POLAR_MACH_NODES[-1])
    phi, bracketed = _solve_inflow(stations, blade.n_blades, safe_re, safe_mach, coefficients)
    all_bracketed: np.ndarray = np.all(bracketed, axis=1)
    failures.flag(_scatter(count, rows, ~all_bracketed),
                  lambda p: _bracket_reason(stations, bracketed[_at(rows, p)]))
    section: _Section = _section(stations, blade.n_blades, phi, safe_re, safe_mach, coefficients)
    new_speed: np.ndarray = _relative_speed(stations, cond.omega, phi, section)
    valid: np.ndarray = np.all(np.isfinite(new_speed) & (new_speed > 0.0), axis=1)
    failures.flag(_scatter(count, rows, ~valid), lambda p: _speed_reason(stations, new_speed[_at(rows, p)]))
    alive: np.ndarray = ~failures.mask[rows]
    change: np.ndarray = np.abs(new_speed / speed[rows] - 1.0)
    converged: np.ndarray = alive & np.all(change < SPEED_REL_TOLERANCE, axis=1)
    _flag_low_reynolds(failures, stations, reynolds, rows, converged, count)
    final: np.ndarray = converged & ~failures.mask[rows]
    for key, value in (("cn", section.cn), ("ct", section.ct), ("loss", section.loss), ("speed", speed[rows])):
        out[key][rows[final]] = value[final]
    speed[rows[valid]] = new_speed[valid]
    return final


def _at(rows: np.ndarray, point: int) -> int:
    """Position of batch point `point` within the active `rows`."""
    return int(np.searchsorted(rows, point))


def _flag_low_reynolds(failures: _Failures, stations: _Stations, reynolds: np.ndarray, rows: np.ndarray,
                       converged: np.ndarray, count: int) -> None:
    """Flag converged points whose r/R 0.75 section is below the Reynolds number where the polars are credible."""
    re_75: np.ndarray = _representative_reynolds(stations, reynolds)
    low: np.ndarray = converged & (re_75 < REPRESENTATIVE_RE_MIN)
    failures.flag(_scatter(count, rows, low), lambda p: (
        f"Reynolds number at r/R {REPRESENTATIVE_STATION} is {float(re_75[_at(rows, p)]):.3g}, below "
        f"{REPRESENTATIVE_RE_MIN:.0e} where the airfoil polars are credible (raise rpm or size)"))


def bemt_points(blade: BladeGeometry, rpm: np.ndarray, airspeed_ms: np.ndarray, rho: float | np.ndarray,
                mu: float | np.ndarray, speed_of_sound_ms: float | np.ndarray) -> PropBatch:
    """Thrust, torque, shaft power and efficiency at many operating points at once (1-D arrays; rho, mu and the
    speed of sound may be scalars or per-point). Invalid inputs raise ValueError for the whole batch; a point the
    model cannot solve is reported in `PropBatch.errors` (NaN values) without affecting the others."""
    columns = np.broadcast_arrays(*(np.atleast_1d(np.asarray(x, dtype=float))
                                    for x in (rpm, airspeed_ms, rho, mu, speed_of_sound_ms)))
    rpm_a, speed_a, rho_a, mu_a, sound_a = (np.array(c) for c in columns)
    _validate_conditions(rpm_a, speed_a, rho_a, mu_a, sound_a)
    omega: np.ndarray = rpm_a * 2.0 * math.pi / SECONDS_PER_MINUTE
    stations: _Stations = _stations(blade, omega, speed_a)
    failures: _Failures = _Failures(len(omega))
    section, speed = _converged_state(blade, stations, omega, speed_a, rho_a, mu_a, sound_a, failures)
    load: np.ndarray = 0.5 * rho_a[:, None] * speed ** 2 * blade.n_blades * stations.chord_m * stations.width_m
    with np.errstate(invalid="ignore", over="ignore"):
        thrust: np.ndarray = np.sum(load * section.cn, axis=-1)
        torque: np.ndarray = np.sum(load * section.ct * stations.radius_m, axis=-1)
    finite: np.ndarray = np.isfinite(thrust) & np.isfinite(torque)
    failures.flag(~finite, lambda p: f"BEMT produced non-finite loads: thrust {thrust[p]} N, torque {torque[p]} N m")
    return _batch_result(failures, thrust, torque, omega, speed_a)


def _batch_result(failures: _Failures, thrust: np.ndarray, torque: np.ndarray, omega: np.ndarray,
                  airspeed_ms: np.ndarray) -> PropBatch:
    failed: np.ndarray = failures.mask
    power: np.ndarray = torque * omega
    moving: np.ndarray = (airspeed_ms > 0.0) & (power > 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        efficiency: np.ndarray = np.where(moving, thrust * airspeed_ms / power, 0.0)
    windmilling: np.ndarray = (airspeed_ms > 0.0) & (power <= 0.0)

    def masked(values: np.ndarray) -> np.ndarray:
        return np.where(failed, math.nan, values)

    return PropBatch(thrust_n=masked(thrust), torque_nm=masked(torque), shaft_power_w=masked(power),
                     efficiency=masked(efficiency), is_windmilling=windmilling & ~failed,
                     errors=failures.reasons)


def bemt_point(blade: BladeGeometry, rpm: float, airspeed_ms: float, rho: float, mu: float,
               speed_of_sound_ms: float) -> PropPoint:
    """Thrust, torque, shaft power and efficiency of `blade` at `rpm` and axial `airspeed_ms` (>= 0)."""
    batch: PropBatch = bemt_points(blade, np.array([rpm]), np.array([airspeed_ms]), rho, mu, speed_of_sound_ms)
    if batch.errors[0] is not None:
        raise PropModelError(batch.errors[0])
    return PropPoint(thrust_n=float(batch.thrust_n[0]), torque_nm=float(batch.torque_nm[0]),
                     shaft_power_w=float(batch.shaft_power_w[0]), efficiency=float(batch.efficiency[0]),
                     is_windmilling=bool(batch.is_windmilling[0]))
