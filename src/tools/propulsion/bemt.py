"""Blade-element momentum (BEMT) propeller model on a generic blade (spec 6.1).

The blade is defined by diameter and pitch only: a fixed chord distribution (c/R vs r/R) and twist from constant
geometric pitch, beta(r) = atan(P / (2 pi r)), with one generic thin propeller airfoil. Section lift and drag come
from NeuralFoil, swept once over alpha per (Reynolds, Mach) node of a small grid and interpolated, so NeuralFoil is
never called per section or per iteration. Prandtl tip and hub losses. All SI.

The inflow angle phi at each station solves the momentum/blade-element residual of Ning (2014), multiplied through
by sin(phi) so it stays finite at zero airspeed (hover):
    R(phi) = sin^2(phi) - lambda sin(phi) cos(phi) - sigma' (cn + lambda ct) / (4 F),   lambda = V / (Omega r)
R < 0 as phi -> 0 and R > 0 at phi = pi/2, so bisection on (0, pi/2) always converges.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

import aerosandbox as asb
import numpy as np
from scipy.interpolate import RegularGridInterpolator

PROP_AIRFOIL: Final[str] = "naca4409"  # generic thin cambered propeller section (chosen against the UIUC gate)
PROP_MODEL_VERSION: Final[int] = 1  # bump when the blade, airfoil or solver changes (keys the on-disk tables)

HUB_FRACTION: Final[float] = 0.15  # r/R where the blade starts (spinner / hub)
N_ELEMENTS: Final[int] = 30  # equal-width blade elements from hub to tip; stations are their midpoints
# Generic planform, c/R at r/R knots: a typical thin-electric blade (widest near 0.4 R, tapering to a narrow tip),
# set against the UIUC validation gate (tests/test_propeller_bemt.py) together with the airfoil.
CHORD_KNOT_STATIONS: Final[tuple[float, ...]] = (0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
CHORD_KNOT_VALUES: Final[tuple[float, ...]] = (0.154, 0.171, 0.209, 0.220, 0.215, 0.193, 0.160, 0.121, 0.088, 0.044)

# Polar grid: one NeuralFoil alpha sweep per (Reynolds, Mach) node; queries outside the grid are clamped to it.
POLAR_RE_NODES: Final[tuple[float, ...]] = (1e4, 2e4, 4e4, 7e4, 1.2e5, 2e5, 4e5, 1e6, 3e6, 1e7)
POLAR_MACH_NODES: Final[tuple[float, ...]] = (0.0, 0.25, 0.5, 0.75)
POLAR_ALPHA_DEG: Final[np.ndarray] = np.linspace(-90.0, 90.0, 181)

PHI_MIN_RAD: Final[float] = 1e-6  # lower bisection bound (phi = 0 is singular in the loss factors)
BISECTION_STEPS: Final[int] = 60  # halves (0, pi/2) to ~1e-18 rad
REYNOLDS_PASSES: Final[int] = 2  # pass 1: geometric speed; pass 2: induced relative speed
LOSS_FACTOR_MIN: Final[float] = 1e-9  # keeps the residual finite where the Prandtl factor reaches zero
SECONDS_PER_MINUTE: Final[float] = 60.0

SectionCoefficients = Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class BladeGeometry:
    """Blade by diameter and pitch; `stations` are element midpoints (r/R), chord as c/R at each station."""
    diameter_m: float
    pitch_m: float
    n_blades: int
    stations: tuple[float, ...]
    chord_over_radius: tuple[float, ...]

    @property
    def twist_rad(self) -> tuple[float, ...]:
        """Section pitch angle from constant geometric pitch: beta = atan(P / (2 pi r))."""
        radius: float = self.diameter_m / 2.0
        return tuple(math.atan(self.pitch_m / (2.0 * math.pi * s * radius)) for s in self.stations)


@dataclass(frozen=True)
class PropPoint:
    """One operating point. `efficiency` = T V / P_shaft; 0.0 at zero airspeed or non-positive shaft power."""
    thrust_n: float
    torque_nm: float
    shaft_power_w: float
    efficiency: float


def generic_blade(diameter_m: float, pitch_m: float, n_blades: int = 2) -> BladeGeometry:
    """The generic blade at this diameter and pitch (same planform and airfoil for every propeller)."""
    if diameter_m <= 0.0 or pitch_m <= 0.0 or n_blades < 1:
        raise ValueError(f"blade needs positive diameter and pitch and >= 1 blade, got D={diameter_m} m, "
                         f"P={pitch_m} m, B={n_blades}")
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


@lru_cache(maxsize=None)
def _section_coefficients(airfoil: str) -> SectionCoefficients:
    """Interpolator (Reynolds, Mach, alpha deg) -> (CL, CD) over the polar grid (log Re), inputs clamped to it."""
    sweeps: list[list[tuple[np.ndarray, np.ndarray]]] = [[_alpha_sweep(airfoil, re, m) for m in POLAR_MACH_NODES]
                                                         for re in POLAR_RE_NODES]
    log_re: np.ndarray = np.log10(POLAR_RE_NODES)
    mach_nodes: np.ndarray = np.asarray(POLAR_MACH_NODES)
    grid: tuple[np.ndarray, np.ndarray, np.ndarray] = (log_re, mach_nodes, POLAR_ALPHA_DEG)
    cl_table = RegularGridInterpolator(grid, np.array([[s[0] for s in row] for row in sweeps]))
    cd_table = RegularGridInterpolator(grid, np.array([[s[1] for s in row] for row in sweeps]))

    def coefficients(reynolds: np.ndarray, mach: np.ndarray, alpha_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        query: np.ndarray = np.column_stack([np.clip(np.log10(reynolds), log_re[0], log_re[-1]),
                                             np.clip(mach, mach_nodes[0], mach_nodes[-1]),
                                             np.clip(alpha_deg, POLAR_ALPHA_DEG[0], POLAR_ALPHA_DEG[-1])])
        return cl_table(query), cd_table(query)

    return coefficients


@dataclass(frozen=True)
class _Stations:
    """Per-station quantities that do not depend on the inflow angle."""
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


def _stations(blade: BladeGeometry, omega: float, airspeed_ms: float) -> _Stations:
    """Element widths come from the midpoints: edges halfway between stations, extrapolated at hub and tip."""
    tip_radius: float = blade.diameter_m / 2.0
    r_frac: np.ndarray = np.asarray(blade.stations)
    interior: np.ndarray = (r_frac[1:] + r_frac[:-1]) / 2.0
    edges: np.ndarray = np.concatenate([[2.0 * r_frac[0] - interior[0]], interior, [2.0 * r_frac[-1] - interior[-1]]])
    radius: np.ndarray = r_frac * tip_radius
    chord: np.ndarray = np.asarray(blade.chord_over_radius) * tip_radius
    return _Stations(r_frac=r_frac, radius_m=radius, hub_frac=float(edges[0]), width_m=np.diff(edges) * tip_radius,
                     chord_m=chord, twist=np.asarray(blade.twist_rad),
                     solidity=blade.n_blades * chord / (2.0 * math.pi * radius),
                     inflow_ratio=airspeed_ms / (omega * radius))


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
                  coefficients: SectionCoefficients) -> np.ndarray:
    """Inflow angle per station by vectorised bisection on (0, pi/2), where the residual changes sign."""
    low: np.ndarray = np.full_like(stations.r_frac, PHI_MIN_RAD)
    high: np.ndarray = np.full_like(stations.r_frac, math.pi / 2.0)
    for _ in range(BISECTION_STEPS):
        mid: np.ndarray = (low + high) / 2.0
        section: _Section = _section(stations, n_blades, mid, reynolds, mach, coefficients)
        negative: np.ndarray = _residual(stations, mid, section) < 0.0
        low, high = np.where(negative, mid, low), np.where(negative, high, mid)
    return (low + high) / 2.0


def _relative_speed(stations: _Stations, omega: float, phi: np.ndarray, section: _Section) -> np.ndarray:
    """Relative speed W from the tangential balance W (cos phi + sigma' ct / (4 F sin phi)) = Omega r."""
    swirl_term: np.ndarray = stations.solidity * section.ct / (4.0 * section.loss * np.sin(phi))
    return omega * stations.radius_m / (np.cos(phi) + swirl_term)


def _validate_conditions(rpm: float, airspeed_ms: float, rho: float, mu: float, speed_of_sound_ms: float) -> None:
    if rpm <= 0.0 or airspeed_ms < 0.0 or rho <= 0.0 or mu <= 0.0 or speed_of_sound_ms <= 0.0:
        raise ValueError(f"bemt_point needs rpm, rho, mu, a > 0 and airspeed >= 0, got rpm={rpm}, "
                         f"V={airspeed_ms}, rho={rho}, mu={mu}, a={speed_of_sound_ms}")


def bemt_point(blade: BladeGeometry, rpm: float, airspeed_ms: float, rho: float, mu: float,
               speed_of_sound_ms: float) -> PropPoint:
    """Thrust, torque, shaft power and efficiency of `blade` at `rpm` and axial `airspeed_ms` (>= 0)."""
    _validate_conditions(rpm, airspeed_ms, rho, mu, speed_of_sound_ms)
    omega: float = rpm * 2.0 * math.pi / SECONDS_PER_MINUTE
    stations: _Stations = _stations(blade, omega, airspeed_ms)
    coefficients: SectionCoefficients = _section_coefficients(PROP_AIRFOIL)
    speed: np.ndarray = np.hypot(airspeed_ms, omega * stations.radius_m)
    for _ in range(REYNOLDS_PASSES):
        reynolds: np.ndarray = rho * speed * stations.chord_m / mu
        mach: np.ndarray = speed / speed_of_sound_ms
        phi: np.ndarray = _solve_inflow(stations, blade.n_blades, reynolds, mach, coefficients)
        section: _Section = _section(stations, blade.n_blades, phi, reynolds, mach, coefficients)
        speed = _relative_speed(stations, omega, phi, section)
    load: np.ndarray = 0.5 * rho * speed ** 2 * blade.n_blades * stations.chord_m * stations.width_m
    thrust: float = float(np.sum(load * section.cn))
    torque: float = float(np.sum(load * section.ct * stations.radius_m))
    power: float = torque * omega
    efficiency: float = thrust * airspeed_ms / power if airspeed_ms > 0.0 and power > 0.0 else 0.0
    return PropPoint(thrust_n=thrust, torque_nm=torque, shaft_power_w=power, efficiency=efficiency)
