"""Aerodynamic building blocks for conceptual sizing: ISA atmosphere and NeuralFoil airfoil polars
via AeroSandbox, plus a component drag build-up (Raymer-style). All SI."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Final

import aerosandbox as asb
import numpy as np

if TYPE_CHECKING:  # mass_model imports this module; the boom type is only needed for annotations
    from src.tools.sizing.mass_model import BoomSet

RE_LOW_MAX: Final[float] = 3e5  # below: SD7037
RE_MID_MAX: Final[float] = 1e6  # up to (inclusive): MH 32; above: NACA 4415
CL_MAX_3D_FACTOR: Final[float] = 0.90  # finite-wing / 2D CLmax knockdown
ALPHA_SWEEP_DEG: Final[np.ndarray] = np.linspace(-5.0, 20.0, 51)
POLAR_MACH: Final[float] = 0.05

VNE_TO_CRUISE: Final[float] = 1.4  # design dive speed / cruise (CS-23 style margin)

# Drag area of one stopped (parked) lift rotor per diameter squared: CD0 += n * C * D^2 / S_wing, inside the misc
# factor. It is what remains of the old lump QUADPLANE_DRAG_PENALTY (0.008, booms + stopped rotors, which also sat
# inside the misc factor) once the lift booms have their own wetted-area drag. Calibrated once (2026-11-23) on G2 so
# that lift-boom drag + this term take the lump's place and sum to 0.008 before the misc factor (0.0092 in total CD0):
#     C = (0.008 - lift_boom_term) * S_wing / (4 * D_rotor^2)
# Sizing depends on CD0, so C was iterated to a fixed point (size G2 with C_k, take C_{k+1} from the formula):
# C converged in 6 steps to 0.0048069 at S_wing = 1.1196 m^2, D_rotor = 0.6027 m, lift_boom_term = 1.7626e-3,
# MTOW 19.88 kg. Rounded to 0.00481 (G2 increment 0.0080001 before the factor). Each parked rotor then has a drag
# area of 0.0017 m^2, about a 0.6 m two-blade propeller edge-on.
STOPPED_ROTOR_DRAG_COEFF: Final[float] = 0.00481
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


def body_form_factor(fineness: float) -> float:
    """Raymer fuselage form factor at length/diameter `fineness` (used for the pod and the boom tubes)."""
    return 1.0 + 60.0 / fineness**3 + fineness / 400.0


def boom_drag_area_ratio(boom: "BoomSet", reynolds_per_m: float, wing_area_m2: float) -> float:
    """count x Cf(Re on the boom length) x FF(boom fineness) x wetted area / S_wing (before the misc factor)."""
    wetted: float = math.pi * boom.diameter_m * boom.length_m
    skin: float = turbulent_skin_friction(reynolds_per_m * boom.length_m)
    return boom.count * skin * body_form_factor(boom.length_m / boom.diameter_m) * wetted / wing_area_m2


def parasite_drag_coefficient(
    wing_area_m2: float,
    tail_area_m2: float,
    profile_cd: float,
    fuselage_length_m: float,
    fuselage_diameter_m: float,
    fuselage_reynolds: float,
    booms: Sequence["BoomSet"],
    rotor_count: int,
    rotor_diameter_m: float,
) -> float:
    """CD0 referenced to wing area: lifting surfaces + fuselage + boom tubes + stopped lift rotors, times misc
    factor. The booms see the fuselage's Reynolds number per metre, applied to their own length."""
    lifting: float = profile_cd * (wing_area_m2 + tail_area_m2) / wing_area_m2
    form_factor: float = body_form_factor(fuselage_length_m / fuselage_diameter_m)
    wetted: float = math.pi * fuselage_diameter_m * fuselage_length_m * FUSELAGE_WETTED_FACTOR
    fuselage: float = turbulent_skin_friction(fuselage_reynolds) * form_factor * wetted / wing_area_m2
    reynolds_per_m: float = fuselage_reynolds / fuselage_length_m
    boom_drag: float = sum(boom_drag_area_ratio(b, reynolds_per_m, wing_area_m2) for b in booms)
    rotors: float = rotor_count * STOPPED_ROTOR_DRAG_COEFF * rotor_diameter_m**2 / wing_area_m2
    return MISC_DRAG_FACTOR * (lifting + fuselage + boom_drag + rotors)


def lift_to_drag(cl: float, cd0: float, k: float) -> float:
    return cl / (cd0 + k * cl**2)
