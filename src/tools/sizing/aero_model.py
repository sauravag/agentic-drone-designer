"""Aerodynamic building blocks for conceptual sizing: ISA atmosphere and NeuralFoil airfoil polars
via AeroSandbox, plus a component drag build-up (Raymer-style). All SI."""

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

import aerosandbox as asb
import numpy as np

RE_LOW_MAX: Final[float] = 3e5  # below: SD7037
RE_MID_MAX: Final[float] = 1e6  # up to (inclusive): MH 32; above: NACA 4415
CL_MAX_3D_FACTOR: Final[float] = 0.90  # finite-wing / 2D CLmax knockdown
ALPHA_SWEEP_DEG: Final[np.ndarray] = np.linspace(-5.0, 20.0, 51)
POLAR_MACH: Final[float] = 0.05

VNE_TO_CRUISE: Final[float] = 1.4  # design dive speed / cruise (CS-23 style margin)

QUADPLANE_DRAG_PENALTY: Final[float] = 0.008  # booms + stopped lift rotors, referenced to wing area
MISC_DRAG_FACTOR: Final[float] = 1.15  # gimbal, antennas, cooling, interference
FUSELAGE_WETTED_FACTOR: Final[float] = 0.75  # tapered pod vs. full cylinder


@dataclass(frozen=True)
class AirfoilPolar:
    name: str
    reynolds: float
    cl_max_2d: float
    cd_min: float

    @property
    def cl_max_3d(self) -> float:
        return CL_MAX_3D_FACTOR * self.cl_max_2d


def select_airfoil(reynolds: float) -> str:
    """Reynolds-matched airfoil per SRS §4.1.2."""
    if reynolds < RE_LOW_MAX:
        return "sd7037"
    if reynolds <= RE_MID_MAX:
        return "mh32"
    return "naca4415"


@lru_cache(maxsize=64)
def airfoil_polar(name: str, reynolds: float) -> AirfoilPolar:
    """NeuralFoil sweep -> 2D CLmax and minimum profile drag at the given Reynolds number."""
    aero: dict = asb.Airfoil(name).get_aero_from_neuralfoil(alpha=ALPHA_SWEEP_DEG, Re=reynolds, mach=POLAR_MACH)
    return AirfoilPolar(name=name, reynolds=reynolds,
                        cl_max_2d=float(np.max(aero["CL"])), cd_min=float(np.min(aero["CD"])))


def air_properties(altitude_m: float, temperature_offset_c: float = 0.0) -> tuple[float, float]:
    """(density kg/m^3, dynamic viscosity Pa*s) at altitude in an ISA atmosphere shifted by temperature_offset_c
    (an ISA+20 day is 20 degC warmer at every altitude, at the same pressure)."""
    atm = asb.Atmosphere(altitude=altitude_m, temperature_deviation=temperature_offset_c)
    return float(atm.density()), float(atm.dynamic_viscosity())


def reynolds_number(rho: float, speed_ms: float, length_m: float, mu: float) -> float:
    return rho * speed_ms * length_m / mu


def oswald_efficiency(aspect_ratio: float) -> float:
    """Raymer straight-wing estimate."""
    return 1.78 * (1.0 - 0.045 * aspect_ratio**0.68) - 0.64


def induced_drag_factor(aspect_ratio: float) -> float:
    """k in CD = CD0 + k*CL^2."""
    return 1.0 / (math.pi * oswald_efficiency(aspect_ratio) * aspect_ratio)


def turbulent_skin_friction(reynolds: float) -> float:
    return 0.455 / math.log10(reynolds) ** 2.58


def parasite_drag_coefficient(
    wing_area_m2: float,
    tail_area_m2: float,
    profile_cd: float,
    fuselage_length_m: float,
    fuselage_diameter_m: float,
    fuselage_reynolds: float,
    is_quadplane: bool,
) -> float:
    """CD0 referenced to wing area: lifting surfaces + fuselage + VTOL penalty, times misc factor."""
    lifting: float = profile_cd * (wing_area_m2 + tail_area_m2) / wing_area_m2
    fineness: float = fuselage_length_m / fuselage_diameter_m
    form_factor: float = 1.0 + 60.0 / fineness**3 + fineness / 400.0
    wetted: float = math.pi * fuselage_diameter_m * fuselage_length_m * FUSELAGE_WETTED_FACTOR
    fuselage: float = turbulent_skin_friction(fuselage_reynolds) * form_factor * wetted / wing_area_m2
    vtol: float = QUADPLANE_DRAG_PENALTY if is_quadplane else 0.0
    return MISC_DRAG_FACTOR * (lifting + fuselage + vtol)


def lift_to_drag(cl: float, cd0: float, k: float) -> float:
    return cl / (cd0 + k * cl**2)
