"""Converged MTOW sizing for electric and series-hybrid Group 1-3 ISR UAVs (SRS §4.1).

  outer loop  - Reynolds-matched airfoil (NeuralFoil polar via AeroSandbox)
  inner loop  - damped fixed-point iteration on MTOW
  evaluation  - geometry -> drag polar -> time-stepped mission -> energy storage -> empty mass
Every number is deterministic physics; no LLM estimates (SRS rule 2).
"""

import logging
import math
from dataclasses import dataclass, replace
from typing import Final, Optional

from scipy.optimize import brentq

from src.schemas.assumptions import DEFAULT_ASSUMPTIONS, DesignAssumptions
from src.schemas.components import ComponentNeeds
from src.schemas.knobs import OVERRIDE_BOUNDS, OVERRIDE_DEFAULTS
from src.schemas.requirements import CUSTOM_PAYLOAD, OperationalRequirements
from src.schemas.state import SizingOutput
from src.tools.sizing.aero_model import (
    AirfoilPolar,
    air_properties,
    airfoil_polar,
    induced_drag_factor,
    lift_to_drag,
    oswald_efficiency,
    parasite_drag_coefficient,
    reynolds_number,
    select_airfoil,
)
from src.tools.gear import GearGeometry
from src.tools.sizing.layouts import BoomSet, layout_spec
from src.tools.sizing.mass_model import (
    PropulsionLoads,
    empty_mass_breakdown,
    required_wing_area_m2,
    size_geometry,
    sizing_booms,
    sizing_gear,
)
from src.tools.sizing.mission_profile import (
    GENERATOR_MARGIN,
    SPEED_MARGIN_OVER_STALL,
    VTOL_THRUST_TO_WEIGHT,
    AeroState,
    EnergyStorage,
    MissionTrace,
    disk_loading_n_m2,
    electric_storage,
    forward_power_w,
    hotel_power_w,
    hybrid_storage,
    loiter_speed,
    phase_plan,
    simulate,
)
from src.tools.sizing.topology import (HAND_LAUNCH_MAX_MTOW_KG, Airframe, LaunchMethod, Propulsion, classify_group,
                                        resolved_launch_method, select_airframe, select_propulsion)
from src.units import G0, KWH_TO_J

logger = logging.getLogger(__name__)

RHO_SEA_LEVEL: Final[float] = air_properties(0.0)[0]  # SizingOutput.v_stall_ms is an equivalent (sea-level) speed
MTOW_REL_TOL: Final[float] = 1e-9
ROOT_ABS_TOL_KG: Final[float] = 1e-12  # brentq absolute tolerance on MTOW
MAX_ITERATIONS: Final[int] = 1_000
DAMPING: Final[float] = 0.5
MIN_DAMPING: Final[float] = 0.05
OSCILLATION_WINDOW: Final[int] = 6
BRACKET_FACTOR: Final[float] = 1.5
MIN_MTOW_KG: Final[float] = 0.1
MAX_BRACKET_STEPS: Final[int] = 60
DIVERGENCE_MTOW_KG: Final[float] = 3 * 599.0
MAX_AIRFOIL_PASSES: Final[int] = 5
RE_REFRESH_TOL: Final[float] = 0.10
INITIAL_AIRFOIL: Final[tuple[str, float]] = ("sd7037", 2e5)
VTOL_ROTOR_COUNT: Final[int] = 4



class SizingDivergenceError(RuntimeError):
    """The mission cannot close with this powertrain/airframe (mass spiral or beyond Group 3)."""


@dataclass(frozen=True)
class _Design:
    req: OperationalRequirements
    airframe: Airframe
    propulsion: Propulsion
    params: dict[str, float]  # validated overrides merged over defaults
    assumptions: DesignAssumptions  # technology constants (efficiencies, specific energy, reserves)
    cruise_polar: AirfoilPolar  # profile drag + high-altitude CLmax, at cruise Re
    stall_polar: AirfoilPolar  # CLmax for the stall constraint, at the take-off-site stall Re (same airfoil)


@dataclass(frozen=True)
class _Evaluation:
    mtow_in: float
    mtow_out: float
    design: _Design
    aero: AeroState
    trace: MissionTrace
    storage: EnergyStorage
    generator_power_w: float
    cruise_total_w: float
    wing_reynolds: float
    stall_reynolds: float
    empty_breakdown: dict[str, float]
    gear: GearGeometry


def reserve_fraction(
    params: dict[str, float], propulsion: Propulsion, assumptions: DesignAssumptions = DEFAULT_ASSUMPTIONS
) -> float:
    """Critic override if given, else the assumption (default 20% SoC electric, 10% fuel series-hybrid)."""
    return params.get("reserve_fraction", assumptions.reserve_for(propulsion))


def _validate_overrides(overrides: dict[str, float]) -> dict[str, float]:
    for key, value in overrides.items():
        if key not in OVERRIDE_BOUNDS:
            raise ValueError(f"Unknown sizing override {key!r}; allowed: {sorted(OVERRIDE_BOUNDS)}")
        low, high = OVERRIDE_BOUNDS[key]
        if not (isinstance(value, (int, float)) and math.isfinite(value) and low <= value <= high):
            raise ValueError(f"Sizing override {key}={value!r} outside physical bounds [{low}, {high}]")
    return {**OVERRIDE_DEFAULTS, **{k: float(v) for k, v in overrides.items()}}


def _wing_area_m2(mtow: float, design: _Design, rho_to: float, rho_c: float) -> float:
    """Binding of two constraints: stall at max_stall_speed at the take-off site, and cruise at altitude flown
    no slower than SPEED_MARGIN_OVER_STALL x local stall (CL <= CLmax / margin^2)."""
    req = design.req
    by_stall: float = required_wing_area_m2(mtow, rho_to, req.max_stall_speed_ms, design.stall_polar.cl_max_3d)
    cruise_cl_limit: float = design.cruise_polar.cl_max_3d / SPEED_MARGIN_OVER_STALL**2
    by_cruise: float = required_wing_area_m2(mtow, rho_c, req.cruise_speed_ms, cruise_cl_limit)
    return max(by_stall, by_cruise)


def _rotor_diameter_m(disk_area_m2: float) -> float:
    """Diameter of each of the VTOL_ROTOR_COUNT lift rotors sharing `disk_area_m2` (0 without rotors)."""
    return math.sqrt(4.0 * disk_area_m2 / VTOL_ROTOR_COUNT / math.pi)


def _aero_state(mtow: float, design: _Design) -> tuple[AeroState, float, tuple[BoomSet, ...]]:
    """Geometry + drag polar at this MTOW. Returns (aero, mu at cruise altitude, the boom tubes): the booms are built
    once here and feed both the drag build-up and the empty mass."""
    req, p = design.req, design.params
    spec = layout_spec(req.airframe_layout)
    rho_to, _ = air_properties(req.takeoff_altitude_m, req.isa_temperature_offset_c)
    rho_c, mu_c = air_properties(req.service_ceiling_m, req.isa_temperature_offset_c)
    geom = size_geometry(mtow, _wing_area_m2(mtow, design, rho_to, rho_c), p["aspect_ratio"],
                         p["tail_volume_h"], p["tail_volume_v"], p["fuselage_length_factor"], spec)
    is_quad: bool = design.airframe == "quadplane_vtol"
    disk: float = mtow * G0 / disk_loading_n_m2(mtow) if is_quad else 0.0
    rotor_d: float = _rotor_diameter_m(disk)
    booms = sizing_booms(geom, spec, rotor_d)
    cd0: float = parasite_drag_coefficient(
        geom.wing_area_m2, geom.tail_area_m2, design.cruise_polar.cd_min, geom.fuselage_length_m,
        geom.fuselage_diameter_m, reynolds_number(rho_c, req.cruise_speed_ms, geom.fuselage_length_m, mu_c),
        booms, VTOL_ROTOR_COUNT if is_quad else 0, rotor_d,
    )
    return AeroState(geom, cd0, induced_drag_factor(p["aspect_ratio"]), rho_to, rho_c, disk), mu_c, booms


def _evaluate(mtow: float, design: _Design) -> _Evaluation:
    req, assumptions = design.req, design.assumptions
    eta: float = assumptions.drivetrain_efficiency
    aero, mu_c, booms = _aero_state(mtow, design)
    _, mu_to = air_properties(req.takeoff_altitude_m, req.isa_temperature_offset_c)
    hotel: float = hotel_power_w(req, mtow, assumptions)
    cruise_total: float = forward_power_w(mtow * G0, req.cruise_speed_ms, aero.rho_cruise, aero, eta) + hotel
    is_hybrid: bool = design.propulsion == "series_hybrid"
    gen_w: float = GENERATOR_MARGIN * cruise_total if is_hybrid else 0.0
    trace = simulate(
        phase_plan(req, design.airframe, design.cruise_polar, aero, mtow), mtow, aero, hotel, gen_w,
        assumptions.bsfc_kg_per_kwh / KWH_TO_J, eta, assumptions.hover_figure_of_merit,
        assumptions.lift_motor_efficiency,
    )
    reserve: float = reserve_fraction(design.params, design.propulsion, assumptions)
    storage = (hybrid_storage(trace, gen_w, reserve, assumptions) if is_hybrid
               else electric_storage(trace, reserve, assumptions))
    vtol_thrust: float = VTOL_THRUST_TO_WEIGHT * mtow * G0 if design.airframe == "quadplane_vtol" else 0.0
    loads = PropulsionLoads(trace.max_forward_propulsive_w, vtol_thrust, gen_w, storage.fuel_mass_kg)
    avionics: float = assumptions.avionics_mass_kg(mtow, req.flight_controller, req.companion_computer)
    gear: GearGeometry = sizing_gear(aero.geom, layout_spec(req.airframe_layout), resolved_launch_method(req))
    breakdown: dict[str, float] = empty_mass_breakdown(mtow, aero.geom, design.airframe, loads, avionics, booms, gear)
    empty: float = sum(breakdown.values())
    return _Evaluation(
        mtow_in=mtow, mtow_out=empty + req.payload_mass_kg + storage.battery_mass_kg + storage.fuel_mass_kg,
        design=design, aero=aero, trace=trace, storage=storage, generator_power_w=gen_w,
        cruise_total_w=cruise_total,
        wing_reynolds=reynolds_number(aero.rho_cruise, req.cruise_speed_ms, aero.geom.mean_chord_m, mu_c),
        stall_reynolds=reynolds_number(aero.rho_takeoff, req.max_stall_speed_ms, aero.geom.mean_chord_m, mu_to),
        empty_breakdown=breakdown, gear=gear,
    )


def _is_oscillating(signs: list[int]) -> bool:
    """True when the last OSCILLATION_WINDOW residual signs strictly alternate."""
    return len(signs) == OSCILLATION_WINDOW and all(a != b for a, b in zip(signs, signs[1:]))


def _checked_eval(mtow: float, design: _Design) -> _Evaluation:
    """Evaluate at `mtow`, raising the mass-spiral error when the result is non-finite or runaway."""
    ev = _evaluate(mtow, design)
    if not math.isfinite(ev.mtow_out) or ev.mtow_out > DIVERGENCE_MTOW_KG:
        raise SizingDivergenceError(
            f"{design.propulsion} {design.airframe} mass spiral: MTOW exceeded {DIVERGENCE_MTOW_KG:.0f} kg"
        )
    return ev


def _bracket(mtow: float, design: _Design) -> tuple[float, float]:
    """Step geometrically from `mtow` in the direction of g = mtow_out - mtow until g changes sign."""
    g0: float = _checked_eval(mtow, design).mtow_out - mtow
    factor: float = BRACKET_FACTOR if g0 > 0 else 1.0 / BRACKET_FACTOR
    lo: float = mtow
    for _ in range(MAX_BRACKET_STEPS):
        hi: float = lo * factor
        if hi > DIVERGENCE_MTOW_KG:
            raise SizingDivergenceError(
                f"{design.propulsion} {design.airframe} mass spiral: no MTOW root below {DIVERGENCE_MTOW_KG:.0f} kg"
            )
        if hi < MIN_MTOW_KG:
            raise SizingDivergenceError(
                f"{design.propulsion} {design.airframe} mass collapse: no MTOW root above {MIN_MTOW_KG} kg"
            )
        g1: float = _checked_eval(hi, design).mtow_out - hi
        if g0 * g1 <= 0.0:
            return (min(lo, hi), max(lo, hi))
        lo = hi
    raise SizingDivergenceError(
        f"{design.propulsion} {design.airframe} no MTOW root bracketed in {MAX_BRACKET_STEPS} steps"
    )


def _root_find(mtow: float, design: _Design) -> _Evaluation:
    """Bracketed root find of g(m) = mtow_out(m) - m, used when the damped iteration is too slow."""
    lo, hi = _bracket(mtow, design)
    root, info = brentq(lambda m: _checked_eval(m, design).mtow_out - m, lo, hi, xtol=ROOT_ABS_TOL_KG,
                        rtol=MTOW_REL_TOL, maxiter=MAX_ITERATIONS, full_output=True, disp=False)
    if not info.converged:
        raise SizingDivergenceError(f"MTOW root find did not converge: {info.flag}")
    return _checked_eval(root, design)


def _converge(design: _Design) -> _Evaluation:
    """Damped fixed point MTOW_{k+1} = MTOW_k + damping * (f(MTOW_k) - MTOW_k). If the residual sign
    alternates over a whole window the iteration is overshooting, so damping is halved (floor MIN_DAMPING).
    If it still has not converged after MAX_ITERATIONS (slow contraction), fall back to a bracketed root find."""
    mtow: float = 4.0 * design.req.payload_mass_kg + 1.0
    damping: float = DAMPING
    signs: list[int] = []
    for _ in range(MAX_ITERATIONS):
        ev = _checked_eval(mtow, design)
        residual: float = ev.mtow_out - mtow
        if abs(residual) <= MTOW_REL_TOL * mtow:
            return ev
        signs = [*signs, 1 if residual > 0 else -1][-OSCILLATION_WINDOW:]
        if _is_oscillating(signs):
            damping = max(MIN_DAMPING, damping / 2.0)
            signs = []
        mtow += damping * residual
    return _root_find(mtow, design)


def _polar_at(name: str, reynolds: float) -> AirfoilPolar:
    # Rounding Re to 2 significant figures keeps results deterministic and lets the cache hit.
    return airfoil_polar(name, float(f"{reynolds:.2g}"))


def _is_self_consistent(ev: _Evaluation, name: str) -> bool:
    d = ev.design
    shifts = (abs(ev.wing_reynolds - d.cruise_polar.reynolds) / d.cruise_polar.reynolds,
              abs(ev.stall_reynolds - d.stall_polar.reynolds) / d.stall_polar.reynolds)
    return name == d.cruise_polar.name and max(shifts) <= RE_REFRESH_TOL


def _size_powertrain(req: OperationalRequirements, airframe: Airframe, propulsion: Propulsion,
                     params: dict[str, float], assumptions: DesignAssumptions) -> _Evaluation:
    """Outer loop: re-pick the Reynolds-matched airfoil and re-evaluate its cruise/stall polars until
    airfoil and Re are self-consistent. If it never settles (e.g. flip-flopping across the Re = 3e5
    boundary), fall back to the visited design with the lowest stall CLmax - the conservative wing."""
    initial: AirfoilPolar = airfoil_polar(*INITIAL_AIRFOIL)
    design = _Design(req, airframe, propulsion, params, assumptions, initial, initial)
    visited: list[_Design] = []
    for _ in range(MAX_AIRFOIL_PASSES):
        visited.append(design)
        ev = _converge(design)
        name: str = select_airfoil(ev.wing_reynolds)
        if _is_self_consistent(ev, name):
            return ev
        design = replace(design, cruise_polar=_polar_at(name, ev.wing_reynolds),
                         stall_polar=_polar_at(name, ev.stall_reynolds))
    conservative = min(visited, key=lambda d: d.stall_polar.cl_max_3d)
    logger.warning("Airfoil selection did not settle in %d passes; using conservative %s (CLmax %.3f)",
                   MAX_AIRFOIL_PASSES, conservative.stall_polar.name, conservative.stall_polar.cl_max_3d)
    return _converge(conservative)


def _to_output(ev: _Evaluation) -> SizingOutput:
    req, design, aero, storage = ev.design.req, ev.design, ev.aero, ev.storage
    geom = aero.geom
    mtow: float = ev.mtow_out
    try:
        group = classify_group(mtow)
    except ValueError as exc:
        raise SizingDivergenceError(str(exc)) from exc
    launch: LaunchMethod = resolved_launch_method(req)
    if launch == "hand" and mtow > HAND_LAUNCH_MAX_MTOW_KG:
        raise SizingDivergenceError(f"hand launch needs MTOW <= {HAND_LAUNCH_MAX_MTOW_KG:g} kg (Group 1); "
                                    f"this design sizes to {mtow:.1f} kg")
    weight: float = mtow * G0
    cl_max: float = design.stall_polar.cl_max_3d
    v_loiter: float = loiter_speed(weight, aero, design.cruise_polar.cl_max_3d)
    cl_cruise: float = weight / (0.5 * aero.rho_cruise * req.cruise_speed_ms**2 * geom.wing_area_m2)
    cl_loiter: float = weight / (0.5 * aero.rho_cruise * v_loiter**2 * geom.wing_area_m2)
    is_quad: bool = design.airframe == "quadplane_vtol"
    return SizingOutput(
        mtow_kg=mtow,
        empty_mass_kg=mtow - req.payload_mass_kg - storage.battery_mass_kg - storage.fuel_mass_kg,
        empty_mass_breakdown=ev.empty_breakdown,
        payload_mass_kg=req.payload_mass_kg, fuel_mass_kg=storage.fuel_mass_kg,
        battery_mass_kg=storage.battery_mass_kg,
        wing_area_m2=geom.wing_area_m2, wingspan_m=geom.wingspan_m, mean_chord_m=geom.mean_chord_m,
        aspect_ratio=geom.aspect_ratio, fuselage_length_m=geom.fuselage_length_m,
        fuselage_diameter_m=geom.fuselage_diameter_m, horizontal_tail_area_m2=geom.horizontal_tail_area_m2,
        vertical_tail_area_m2=geom.vertical_tail_area_m2, tail_arm_m=geom.tail_arm_m,
        launch_method=launch, gear_kind=ev.gear.kind, airframe_layout=req.airframe_layout, tail_surfaces=geom.tail_surfaces,
        cruise_power_w=ev.cruise_total_w, hover_power_w=ev.trace.peak_hover_w,
        v_stall_ms=math.sqrt(2.0 * weight / (RHO_SEA_LEVEL * geom.wing_area_m2 * cl_max)),
        v_stall_takeoff_ms=math.sqrt(2.0 * weight / (aero.rho_takeoff * geom.wing_area_m2 * cl_max)),
        v_cruise_ms=req.cruise_speed_ms, cl_cruise=cl_cruise, cl_loiter=cl_loiter,
        v_loiter_ms=v_loiter,
        assumed_ld_cruise=lift_to_drag(cl_cruise, aero.cd0, aero.k),
        dod_group=group, propulsion_type=design.propulsion, airframe_topology=design.airframe,
        airfoil=design.cruise_polar.name, cl_max=cl_max, cd0=aero.cd0, oswald_e=oswald_efficiency(geom.aspect_ratio),
        battery_energy_j=storage.battery_energy_j, battery_peak_power_w=storage.battery_peak_w,
        generator_power_w=ev.generator_power_w, cruise_motor_power_w=ev.trace.max_forward_propulsive_w,
        vtol_thrust_per_motor_n=VTOL_THRUST_TO_WEIGHT * weight / VTOL_ROTOR_COUNT if is_quad else 0.0,
        rotor_diameter_m=_rotor_diameter_m(aero.disk_area_m2),
        energy_reserve_fraction=reserve_fraction(design.params, design.propulsion, design.assumptions),
        phases=ev.trace.phases,
        avionics_mass_kg=design.assumptions.avionics_mass_kg(mtow, req.flight_controller, req.companion_computer),
        avionics_power_w=design.assumptions.avionics_power_w(mtow, req.jetson_fitted),
    )


def size_aircraft(
    req: OperationalRequirements,
    overrides: Optional[dict[str, float]] = None,
    assumptions: DesignAssumptions = DEFAULT_ASSUMPTIONS,
) -> SizingOutput:
    """Converge a design for `req`. With propulsion "auto", size battery-electric first and fall back
    to series-hybrid when electric diverges or breaks the SRS electric envelope (MTOW/endurance)."""
    params: dict[str, float] = _validate_overrides(overrides or {})
    airframe: Airframe = select_airframe(req)
    if req.propulsion_type != "auto":
        return _to_output(_size_powertrain(req, airframe, req.propulsion_type, params, assumptions))
    electric_failure: Optional[SizingDivergenceError] = None
    try:
        electric = _to_output(_size_powertrain(req, airframe, "electric", params, assumptions))
        if select_propulsion(req, electric.mtow_kg) == "electric":
            return electric
        logger.info("Electric design (%.1f kg) is outside the electric envelope; sizing series-hybrid",
                    electric.mtow_kg)
    except SizingDivergenceError as exc:
        electric_failure = exc
        logger.info("electric sizing failed (%s); falling back to series-hybrid", exc)
    try:
        return _to_output(_size_powertrain(req, airframe, "series_hybrid", params, assumptions))
    except SizingDivergenceError as exc:
        raise SizingDivergenceError(f"Neither powertrain closes the mission: {exc}") from (electric_failure or exc)


def component_needs(sizing: SizingOutput, req: OperationalRequirements) -> ComponentNeeds:
    """Translate a converged design into SI hardware requirements for the COTS matcher."""
    is_quad: bool = sizing.airframe_topology == "quadplane_vtol"
    is_hybrid: bool = sizing.propulsion_type == "series_hybrid"
    return ComponentNeeds(
        cruise_power_w=sizing.cruise_motor_power_w,
        battery_energy_j=sizing.battery_energy_j,
        battery_peak_power_w=sizing.battery_peak_power_w,
        battery_chemistry="lipo" if is_hybrid else None,  # hybrid buffer needs 25C discharge
        payload_mass_kg=req.payload_mass_kg,
        vtol_thrust_per_motor_n=sizing.vtol_thrust_per_motor_n if is_quad else None,
        vtol_motor_count=VTOL_ROTOR_COUNT,
        generator_power_w=sizing.generator_power_w if is_hybrid else None,
        flight_controller=req.flight_controller,
        companion_computer=req.companion_computer,
        payload_model=req.payload_model,
        # A named model carries the designer's power; Custom keeps matching by mass alone, as before.
        payload_power_w=None if req.payload_model == CUSTOM_PAYLOAD else req.payload_power_w,
    )
