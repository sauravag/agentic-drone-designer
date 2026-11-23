"""Geometry sizing and semi-empirical empty-mass build-up for Group 1-3 UAVs (SI).

Empty mass = structure + propulsion hardware + systems; it excludes payload, battery and fuel.
Propulsion hardware carries the matcher's 15% margin so sized MTOW tracks the discrete COTS BOM.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from src.schemas.state import SizingOutput, TailSurface
from src.tools.gear import GearGeometry, GearInputs, PropellerDisc, gear_mass_kg, landing_gear
from src.tools.sizing.aero_model import FUSELAGE_WETTED_FACTOR
from src.tools.sizing.layouts import (
    DEFAULT_WING_STATION,
    HTAIL_ASPECT_RATIO,
    HTAIL_TAPER_RATIO,
    QUARTER_CHORD,
    TAIL_BOOM_START_FRACTION,
    VTAIL_ASPECT_RATIO,
    VTAIL_TAPER_RATIO,
    WING_TAPER_RATIO,
    BoomSet,
    LayoutSpec,
    layout_spec,
    lift_boom_geometry,
    tail_arm_m,
    tail_boom_diameter_m,
    trapezoid_mac_m,
    trapezoid_root_chord_m,
)
from src.tools.sizing.topology import Airframe, LaunchMethod
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

SYSTEMS_FRACTION: Final[float] = 0.05  # wiring, servos, harness, fasteners

MATCH_MARGIN: Final[float] = 1.15
CRUISE_PROPULSION_W_PER_KG: Final[float] = 3_000.0  # motor + ESC + prop
VTOL_THRUST_N_PER_KG: Final[float] = 280.0  # lift motor
VTOL_INSTALL_FACTOR: Final[float] = 1.3  # + props, ESCs, mounts
GENSET_W_PER_KG: Final[float] = 700.0
FUEL_SYSTEM_FRACTION: Final[float] = 0.15  # tank + lines + pump, of fuel mass
# Boom tube wall mass per area: 1 mm carbon at 1600 kg/m^3 (= aero_cad.structure.COMPOSITE_DENSITY_KG_M3).
CF_TUBE_WALL_KG_M2: Final[float] = 1.6
LIFT_BOOM_COUNT: Final[int] = 2  # two longitudinal booms, two lift rotors each


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


def _default_station_wing_m(geom: Geometry) -> tuple[float, float, float]:
    """(root chord, root LE, AC) of the CAD wing (taper WING_TAPER_RATIO) at the default wing station."""
    root: float = trapezoid_root_chord_m(geom.wing_area_m2, geom.wingspan_m, WING_TAPER_RATIO)
    x_le: float = DEFAULT_WING_STATION * geom.fuselage_length_m
    return root, x_le, x_le + QUARTER_CHORD * trapezoid_mac_m(root, WING_TAPER_RATIO)


def _surface_trailing_edge_m(surface: TailSurface, wing_x_ac_m: float) -> float:
    """Root trailing edge of a tail surface whose MAC quarter-chord sits its arm behind the wing AC."""
    root: float = trapezoid_root_chord_m(surface.area_m2, surface.span_m, surface.taper_ratio)
    x_le: float = wing_x_ac_m + surface.arm_m - QUARTER_CHORD * trapezoid_mac_m(root, surface.taper_ratio)
    return x_le + root


def boom_mass_kg(boom: BoomSet) -> float:
    """count x length x tube wall mass per area x circumference."""
    return boom.count * boom.length_m * CF_TUBE_WALL_KG_M2 * math.pi * boom.diameter_m


def sizing_booms(geom: Geometry, spec: LayoutSpec, rotor_diameter_m: float) -> tuple[BoomSet, ...]:
    """The boom tubes the sizing books, estimated at DEFAULT_WING_STATION exactly as the CAD layout places them:
    a centre tail boom from TAIL_BOOM_START_FRACTION of the pod to the tail trailing edge (none when the tail sits
    on the pod), and two lift booms when there are lift rotors (`rotor_diameter_m` > 0)."""
    if spec.booms != "centre":
        raise ValueError(f"no boom estimate for {spec.booms!r} booms yet")
    if not geom.tail_surfaces:
        raise ValueError("geometry has no tail surfaces to size a tail boom for")
    root, x_le, x_ac = _default_station_wing_m(geom)
    tail_end: float = max(_surface_trailing_edge_m(s, x_ac) for s in geom.tail_surfaces)
    start: float = TAIL_BOOM_START_FRACTION * geom.fuselage_length_m
    tail: tuple[BoomSet, ...] = (
        (BoomSet(1, tail_end - start, tail_boom_diameter_m(geom.fuselage_diameter_m), "tail_boom"),)
        if tail_end > geom.fuselage_length_m else ()
    )
    if rotor_diameter_m <= 0:
        return tail
    length, diameter = lift_boom_geometry(rotor_diameter_m, root, x_le, x_ac)
    return (*tail, BoomSet(LIFT_BOOM_COUNT, length, diameter, "lift_booms"))


def sizing_gear(geom: Geometry, spec: LayoutSpec, launch: LaunchMethod) -> GearGeometry:
    """Landing gear from sizing's own estimates: the wing at DEFAULT_WING_STATION, the CG at the layout's target
    static margin ahead of the wing AC, and a propeller no larger than the fuselage (the propeller is not chosen
    until the BOM; the CAD stage recomputes the gear from the real one)."""
    if spec.propeller != "tractor_nose":
        raise ValueError(f"no propeller-disc estimate for {spec.propeller!r} propellers yet")
    _, _, x_ac = _default_station_wing_m(geom)
    x_cg: float = x_ac - spec.static_margin.target * geom.mean_chord_m
    return landing_gear(GearInputs(launch, geom.fuselage_length_m, geom.fuselage_diameter_m, geom.mean_chord_m,
                                   geom.wingspan_m, x_cg, PropellerDisc(0.0, 0.0, geom.fuselage_diameter_m)))


def sizing_output_gear(sizing: SizingOutput) -> GearGeometry:
    """`sizing_gear` for a finished design (the same geometry, read back from the record)."""
    geom = Geometry(
        wing_area_m2=sizing.wing_area_m2, aspect_ratio=sizing.aspect_ratio, wingspan_m=sizing.wingspan_m,
        mean_chord_m=sizing.mean_chord_m, fuselage_length_m=sizing.fuselage_length_m,
        fuselage_diameter_m=sizing.fuselage_diameter_m, horizontal_tail_area_m2=sizing.horizontal_tail_area_m2,
        vertical_tail_area_m2=sizing.vertical_tail_area_m2, tail_arm_m=sizing.tail_arm_m,
        tail_surfaces=sizing.tail_surfaces)
    return sizing_gear(geom, layout_spec(sizing.airframe_layout), sizing.launch_method)


def _structure_mass(mtow_kg: float, geom: Geometry, booms: Sequence[BoomSet]) -> dict[str, float]:
    wing_loading: float = mtow_kg * G0 / geom.wing_area_m2
    sigma_wing: float = (WING_AREAL_DENSITY_BASE + WING_AREAL_DENSITY_PER_LOADING * wing_loading) * (
        geom.aspect_ratio / WING_AR_REFERENCE
    ) ** WING_AR_EXPONENT
    fus_wetted: float = math.pi * geom.fuselage_diameter_m * geom.fuselage_length_m * FUSELAGE_WETTED_FACTOR
    return {
        "wing": sigma_wing * geom.wing_area_m2,
        "tail": TAIL_DENSITY_FACTOR * sigma_wing * geom.tail_area_m2,
        "fuselage": (FUSELAGE_AREAL_DENSITY_BASE + FUSELAGE_AREAL_DENSITY_PER_KG * mtow_kg) * fus_wetted,
        **{boom.name: boom_mass_kg(boom) for boom in booms},
    }


def _propulsion_mass(loads: PropulsionLoads) -> dict[str, float]:
    """Lift motors with their props/ESCs/mounts; the lift booms are structure (`sizing_booms`)."""
    vtol: float = MATCH_MARGIN * loads.vtol_total_thrust_n / VTOL_THRUST_N_PER_KG * VTOL_INSTALL_FACTOR
    return {
        "cruise_propulsion": MATCH_MARGIN * loads.cruise_motor_power_w / CRUISE_PROPULSION_W_PER_KG,
        "vtol_system": vtol,
        "generator": MATCH_MARGIN * loads.generator_power_w / GENSET_W_PER_KG,
        "fuel_system": FUEL_SYSTEM_FRACTION * loads.fuel_mass_kg,
    }


def airframe_mass_kg(mtow_kg: float, empty_mass_kg: float, loads: PropulsionLoads, avionics_mass_kg: float) -> float:
    """Structure (incl. the tail and lift boom tubes) + landing gear of a converged design: empty mass minus
    propulsion hardware, avionics and systems. The CAD shell carries this mass; the booms are part of the OML and
    already sit in the structure breakdown, so their share stays in the airframe."""
    non_structural: float = (
        sum(_propulsion_mass(loads).values())
        + avionics_mass_kg
        + SYSTEMS_FRACTION * mtow_kg
    )
    return empty_mass_kg - non_structural


def empty_mass_breakdown(
    mtow_kg: float, geom: Geometry, airframe: Airframe, loads: PropulsionLoads, avionics_mass_kg: float,
    booms: Sequence[BoomSet], gear: GearGeometry,
) -> dict[str, float]:
    """Itemised empty mass [kg]. `avionics_mass_kg` comes from DesignAssumptions.avionics_mass_kg; `booms` from
    `sizing_booms` (booked under their names, e.g. "tail_boom" and "lift_booms"); `gear` from `sizing_gear`."""
    return {
        **_structure_mass(mtow_kg, geom, booms),
        **_propulsion_mass(loads),
        "landing_gear": gear_mass_kg(gear, airframe, mtow_kg),
        "avionics": avionics_mass_kg,
        "systems": SYSTEMS_FRACTION * mtow_kg,
    }
