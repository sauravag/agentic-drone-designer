"""Geometry sizing and semi-empirical empty-mass build-up for Group 1-3 UAVs (SI).

Empty mass = structure + propulsion hardware + systems; it excludes payload, battery and fuel.
Propulsion hardware carries the matcher's 15% margin so sized MTOW tracks the discrete COTS BOM.
"""

import math
from dataclasses import dataclass
from typing import Final

from src.tools.sizing.aero_model import FUSELAGE_WETTED_FACTOR
from src.schemas.state import TailSurface
from src.tools.sizing.layouts import (
    HTAIL_ASPECT_RATIO,
    HTAIL_TAPER_RATIO,
    VTAIL_ASPECT_RATIO,
    VTAIL_TAPER_RATIO,
    LayoutSpec,
    tail_arm_m,
)
from src.tools.sizing.topology import Airframe
from src.units import G0


FUSELAGE_LENGTH_COEFF: Final[float] = 0.55  # m per kg^(1/3)
FUSELAGE_MIN_LENGTH_M: Final[float] = 0.6
FUSELAGE_FINENESS: Final[float] = 7.0

WING_AREAL_DENSITY_BASE: Final[float] = 0.9  # kg/m^2
WING_AREAL_DENSITY_PER_LOADING: Final[float] = 0.006  # kg/m^2 per N/m^2 of wing loading
WING_AR_REFERENCE: Final[float] = 10.0
WING_AR_EXPONENT: Final[float] = 0.6  # higher AR -> heavier spar
TAIL_DENSITY_FACTOR: Final[float] = 0.6
FUSELAGE_AREAL_DENSITY_BASE: Final[float] = 1.0  # kg/m^2 of wetted area
FUSELAGE_AREAL_DENSITY_PER_KG: Final[float] = 0.002

GEAR_FRACTION: Final[dict[str, float]] = {"fixed_wing": 0.04, "quadplane_vtol": 0.015}
SYSTEMS_FRACTION: Final[float] = 0.05  # wiring, servos, harness, fasteners

MATCH_MARGIN: Final[float] = 1.15
CRUISE_PROPULSION_W_PER_KG: Final[float] = 3_000.0  # motor + ESC + prop
VTOL_THRUST_N_PER_KG: Final[float] = 280.0  # lift motor
VTOL_INSTALL_FACTOR: Final[float] = 1.3  # + props, ESCs, mounts
BOOM_FRACTION: Final[float] = 0.04
GENSET_W_PER_KG: Final[float] = 700.0
FUEL_SYSTEM_FRACTION: Final[float] = 0.15  # tank + lines + pump, of fuel mass


@dataclass(frozen=True)
class Geometry:
    wing_area_m2: float
    aspect_ratio: float
    wingspan_m: float
    mean_chord_m: float
    fuselage_length_m: float
    fuselage_diameter_m: float
    horizontal_tail_area_m2: float
    vertical_tail_area_m2: float
    tail_arm_m: float
    tail_surfaces: tuple[TailSurface, ...] = ()

    @property
    def tail_area_m2(self) -> float:
        return self.horizontal_tail_area_m2 + self.vertical_tail_area_m2


@dataclass(frozen=True)
class PropulsionLoads:
    cruise_motor_power_w: float
    vtol_total_thrust_n: float = 0.0
    generator_power_w: float = 0.0
    fuel_mass_kg: float = 0.0


def required_wing_area_m2(mtow_kg: float, rho: float, speed_ms: float, cl_limit: float) -> float:
    """S >= 2*W / (rho * V^2 * CL_limit)."""
    return 2.0 * mtow_kg * G0 / (rho * speed_ms**2 * cl_limit)


def size_geometry(
    mtow_kg: float,
    wing_area: float,
    aspect_ratio: float,
    tail_volume_h: float,
    tail_volume_v: float,
    fuselage_length_factor: float,
    spec: LayoutSpec,
) -> Geometry:
    """Planform from wing area + AR; fuselage from MTOW scaling; tail from volume coefficients."""
    span: float = math.sqrt(wing_area * aspect_ratio)
    chord: float = wing_area / span
    fus_length: float = max(FUSELAGE_MIN_LENGTH_M, FUSELAGE_LENGTH_COEFF * mtow_kg ** (1 / 3)) * fuselage_length_factor
    tail_arm: float = tail_arm_m(spec, fus_length)
    h_area: float = tail_volume_h * wing_area * chord / tail_arm
    v_area: float = tail_volume_v * wing_area * span / tail_arm
    return Geometry(
        wing_area_m2=wing_area, aspect_ratio=aspect_ratio, wingspan_m=span, mean_chord_m=chord,
        fuselage_length_m=fus_length, fuselage_diameter_m=fus_length / FUSELAGE_FINENESS,
        horizontal_tail_area_m2=h_area, vertical_tail_area_m2=v_area, tail_arm_m=tail_arm,
        tail_surfaces=_conventional_tail(h_area, v_area, tail_arm),
    )


def _conventional_tail(h_area: float, v_area: float, arm: float) -> tuple[TailSurface, TailSurface]:
    """H-tail plus a single fin; span = sqrt(AR * area), the planform the CAD layout draws."""
    return (
        TailSurface(name="htail", control="elevator", area_m2=h_area, span_m=math.sqrt(HTAIL_ASPECT_RATIO * h_area),
                    taper_ratio=HTAIL_TAPER_RATIO, dihedral_deg=0.0, arm_m=arm, is_symmetric=True),
        TailSurface(name="vtail", control="rudder", area_m2=v_area, span_m=math.sqrt(VTAIL_ASPECT_RATIO * v_area),
                    taper_ratio=VTAIL_TAPER_RATIO, dihedral_deg=90.0, arm_m=arm, is_symmetric=False),
    )


def _structure_mass(mtow_kg: float, geom: Geometry) -> dict[str, float]:
    wing_loading: float = mtow_kg * G0 / geom.wing_area_m2
    sigma_wing: float = (WING_AREAL_DENSITY_BASE + WING_AREAL_DENSITY_PER_LOADING * wing_loading) * (
        geom.aspect_ratio / WING_AR_REFERENCE
    ) ** WING_AR_EXPONENT
    fus_wetted: float = math.pi * geom.fuselage_diameter_m * geom.fuselage_length_m * FUSELAGE_WETTED_FACTOR
    return {
        "wing": sigma_wing * geom.wing_area_m2,
        "tail": TAIL_DENSITY_FACTOR * sigma_wing * geom.tail_area_m2,
        "fuselage": (FUSELAGE_AREAL_DENSITY_BASE + FUSELAGE_AREAL_DENSITY_PER_KG * mtow_kg) * fus_wetted,
    }


def _propulsion_mass(mtow_kg: float, loads: PropulsionLoads) -> dict[str, float]:
    vtol: float = 0.0
    if loads.vtol_total_thrust_n > 0:
        motors: float = MATCH_MARGIN * loads.vtol_total_thrust_n / VTOL_THRUST_N_PER_KG
        vtol = motors * VTOL_INSTALL_FACTOR + BOOM_FRACTION * mtow_kg
    return {
        "cruise_propulsion": MATCH_MARGIN * loads.cruise_motor_power_w / CRUISE_PROPULSION_W_PER_KG,
        "vtol_system": vtol,
        "generator": MATCH_MARGIN * loads.generator_power_w / GENSET_W_PER_KG,
        "fuel_system": FUEL_SYSTEM_FRACTION * loads.fuel_mass_kg,
    }


def airframe_mass_kg(mtow_kg: float, empty_mass_kg: float, loads: PropulsionLoads, avionics_mass_kg: float) -> float:
    """Structure (incl. VTOL booms) + landing gear of a converged design: empty mass minus
    propulsion hardware, avionics and systems. The CAD shell carries this mass; booms are part of
    the OML, so their share stays in the airframe."""
    non_structural: float = (
        sum(_propulsion_mass(mtow_kg, loads).values())
        + avionics_mass_kg
        + SYSTEMS_FRACTION * mtow_kg
    )
    booms: float = BOOM_FRACTION * mtow_kg if loads.vtol_total_thrust_n > 0 else 0.0
    return empty_mass_kg - non_structural + booms


def empty_mass_breakdown(
    mtow_kg: float, geom: Geometry, airframe: Airframe, loads: PropulsionLoads, avionics_mass_kg: float
) -> dict[str, float]:
    """Itemised empty mass [kg]. `avionics_mass_kg` comes from DesignAssumptions.avionics_mass_kg."""
    return {
        **_structure_mass(mtow_kg, geom),
        **_propulsion_mass(mtow_kg, loads),
        "landing_gear": GEAR_FRACTION[airframe] * mtow_kg,
        "avionics": avionics_mass_kg,
        "systems": SYSTEMS_FRACTION * mtow_kg,
    }
