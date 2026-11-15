"""Mission phase physics: the 6-phase plan, time-stepped power/energy/fuel integration, and
battery / fuel storage sizing for electric and series-hybrid powertrains (SRS §4.1.2). All SI."""

import math
from dataclasses import dataclass
from typing import Final, Literal

from src.schemas.requirements import OperationalRequirements
from src.schemas.state import PhaseResult
from src.tools.sizing.aero_model import AirfoilPolar, air_properties
from src.tools.sizing.mass_model import Geometry
from src.tools.sizing.topology import Airframe
from src.units import G0, KWH_TO_J, WH_TO_J

CLIMB_RATE_MS: Final[float] = 2.5
VTOL_PHASE_S: Final[float] = 60.0
VTOL_THRUST_TO_WEIGHT: Final[float] = 1.45
FIGURE_OF_MERIT: Final[float] = 0.72
VTOL_MOTOR_EFFICIENCY: Final[float] = 0.85
DRIVETRAIN_EFFICIENCY: Final[float] = 0.68
SPEED_MARGIN_OVER_STALL: Final[float] = 1.2
AVIONICS_POWER_BASE_W: Final[float] = 10.0
AVIONICS_POWER_PER_KG: Final[float] = 0.4

LIION_WH_PER_KG: Final[float] = 200.0
LIION_MAX_C: Final[float] = 3.0
LIPO_WH_PER_KG: Final[float] = 155.0
LIPO_MAX_C: Final[float] = 25.0
MIN_SOC_RESERVE: Final[float] = 0.20
GENERATOR_MARGIN: Final[float] = 1.15
BSFC_KG_PER_J: Final[float] = 0.50 / KWH_TO_J
FUEL_RESERVE: Final[float] = 0.10
HOVER_BUFFER_S: Final[float] = 180.0
ABORT_RESERVE_S: Final[float] = 60.0
STEPS_PER_PHASE: Final[int] = 20

PhaseName = Literal["VTOL_Takeoff", "Climb", "Ingress", "Loiter", "Egress", "VTOL_Land"]


@dataclass(frozen=True)
class AeroState:
    geom: Geometry
    cd0: float
    k: float
    rho_sl: float
    rho_cruise: float
    disk_area_m2: float  # total lift-rotor disk area; 0 for fixed-wing


@dataclass(frozen=True)
class PhaseSpec:
    name: PhaseName
    duration_s: float
    is_hover: bool
    speed_ms: float
    rho: float
    climb_rate_ms: float = 0.0


@dataclass(frozen=True)
class MissionTrace:
    phases: tuple[PhaseResult, ...]
    fuel_burned_kg: float
    peak_total_w: float  # installed peak electrical demand (hover at T/W 1.45 for VTOL)
    peak_hover_w: float  # installed peak hover demand incl. hotel
    sustained_hover_w: float  # T = W hover demand incl. hotel (energy basis)
    peak_deficit_w: float  # max demand the generator cannot cover
    non_hover_deficit_j: float
    max_forward_propulsive_w: float


@dataclass(frozen=True)
class EnergyStorage:
    battery_energy_j: float
    battery_mass_kg: float
    battery_peak_w: float
    fuel_mass_kg: float


# -------------------------------------------------------------- physics ---


def disk_loading_n_m2(mtow_kg: float) -> float:
    """Empirical lift-rotor disk loading: small quads ~100 N/m^2, Group 3 quadplanes ~500 N/m^2."""
    return min(max(60.0 * mtow_kg**0.35, 80.0), 600.0)


def hotel_power_w(req: OperationalRequirements, mtow_kg: float) -> float:
    return req.payload_power_w + AVIONICS_POWER_BASE_W + AVIONICS_POWER_PER_KG * mtow_kg


def forward_power_w(weight_n: float, speed: float, rho: float, aero: AeroState, climb_rate: float = 0.0) -> float:
    """Electrical propulsive power: (D*V + W*ROC) / eta_drivetrain."""
    q: float = 0.5 * rho * speed**2
    cl: float = weight_n / (q * aero.geom.wing_area_m2)
    drag: float = q * aero.geom.wing_area_m2 * (aero.cd0 + aero.k * cl**2)
    return (drag * speed + weight_n * climb_rate) / DRIVETRAIN_EFFICIENCY


def hover_power_w(weight_n: float, aero: AeroState, thrust_to_weight: float = 1.0) -> float:
    """Momentum theory: P = T^1.5 / sqrt(2*rho*A) / FM / eta_motor.
    Sustained hover (energy) uses T/W = 1; installed peak (motor & C-rate sizing) uses T/W = 1.45."""
    thrust: float = thrust_to_weight * weight_n
    induced: float = thrust**1.5 / math.sqrt(2.0 * aero.rho_sl * aero.disk_area_m2)
    return induced / FIGURE_OF_MERIT / VTOL_MOTOR_EFFICIENCY


def speed_for_cl(weight_n: float, rho: float, aero: AeroState, cl: float) -> float:
    return math.sqrt(2.0 * weight_n / (rho * aero.geom.wing_area_m2 * cl))


def _min_speed(weight_n: float, rho: float, aero: AeroState, cl_max: float) -> float:
    return SPEED_MARGIN_OVER_STALL * speed_for_cl(weight_n, rho, aero, cl_max)


def loiter_speed(weight_n: float, aero: AeroState, cl_max: float) -> float:
    """Max-L/D speed (CL* = sqrt(CD0/k)), floored at 1.2 x local stall speed."""
    v_ld: float = speed_for_cl(weight_n, aero.rho_cruise, aero, math.sqrt(aero.cd0 / aero.k))
    return max(v_ld, _min_speed(weight_n, aero.rho_cruise, aero, cl_max))


# ------------------------------------------------------------ mission ---


def phase_plan(
    req: OperationalRequirements, airframe: Airframe, polar: AirfoilPolar, aero: AeroState, mtow_kg: float
) -> tuple[PhaseSpec, ...]:
    weight: float = mtow_kg * G0
    rho_climb, _ = air_properties(req.service_ceiling_m / 2.0)
    # Best rate of climb for a prop aircraft ~ minimum-power CL = sqrt(3*CD0/k).
    v_climb: float = max(speed_for_cl(weight, rho_climb, aero, math.sqrt(3.0 * aero.cd0 / aero.k)),
                         _min_speed(weight, rho_climb, aero, polar.cl_max_3d))
    transit_s: float = req.combat_radius_m / req.cruise_speed_ms
    plan: list[PhaseSpec] = [
        PhaseSpec("Climb", req.service_ceiling_m / CLIMB_RATE_MS, False, v_climb, rho_climb, CLIMB_RATE_MS),
        PhaseSpec("Ingress", transit_s, False, req.cruise_speed_ms, aero.rho_cruise),
        PhaseSpec("Loiter", req.endurance_s, False, loiter_speed(weight, aero, polar.cl_max_3d), aero.rho_cruise),
        PhaseSpec("Egress", transit_s, False, req.cruise_speed_ms, aero.rho_cruise),
    ]
    if airframe == "quadplane_vtol":
        plan.insert(0, PhaseSpec("VTOL_Takeoff", VTOL_PHASE_S, True, 0.0, aero.rho_sl))
        plan.append(PhaseSpec("VTOL_Land", VTOL_PHASE_S, True, 0.0, aero.rho_sl))
    return tuple(plan)


def _step_power_w(spec: PhaseSpec, weight_n: float, aero: AeroState) -> tuple[float, float]:
    """(sustained propulsive power for energy, installed peak propulsive power for sizing)."""
    if spec.is_hover:
        return hover_power_w(weight_n, aero), hover_power_w(weight_n, aero, VTOL_THRUST_TO_WEIGHT)
    power: float = forward_power_w(weight_n, spec.speed_ms, spec.rho, aero, spec.climb_rate_ms)
    return power, power


def simulate(plan: tuple[PhaseSpec, ...], mtow_kg: float, aero: AeroState, hotel_w: float, gen_w: float) -> MissionTrace:
    """Integrate each phase in STEPS_PER_PHASE steps. The generator (if any) supplies demand up to
    its rating and burns fuel, lightening the aircraft; the battery covers any deficit."""
    fuel = peak_total = peak_hover = sustained_hover = peak_deficit = peak_forward = non_hover_deficit_j = 0.0
    results: list[PhaseResult] = []
    for spec in plan:
        dt: float = spec.duration_s / STEPS_PER_PHASE
        prop_sum = total_sum = energy = phase_fuel = 0.0
        for _ in range(STEPS_PER_PHASE):
            prop, prop_peak = _step_power_w(spec, (mtow_kg - fuel) * G0, aero)
            total, total_peak = prop + hotel_w, prop_peak + hotel_w
            from_gen: float = min(total, gen_w)
            burned: float = BSFC_KG_PER_J * from_gen * dt
            fuel += burned
            phase_fuel += burned
            prop_sum += prop
            total_sum += total
            energy += total * dt
            peak_total = max(peak_total, total_peak)
            peak_deficit = max(peak_deficit, total_peak - min(total_peak, gen_w))
            if spec.is_hover:
                peak_hover, sustained_hover = max(peak_hover, total_peak), max(sustained_hover, total)
            else:
                peak_forward = max(peak_forward, prop)
                non_hover_deficit_j += (total - from_gen) * dt
        results.append(PhaseResult(
            name=spec.name, duration_s=spec.duration_s, propulsive_power_w=prop_sum / STEPS_PER_PHASE,
            total_power_w=total_sum / STEPS_PER_PHASE, energy_j=energy, fuel_burned_kg=phase_fuel,
        ))
    return MissionTrace(tuple(results), fuel, peak_total, peak_hover, sustained_hover, peak_deficit,
                        non_hover_deficit_j, peak_forward)


# ------------------------------------------------------ energy storage ---


def electric_storage(trace: MissionTrace, reserve: float = MIN_SOC_RESERVE) -> EnergyStorage:
    """Li-Ion pack: capacity keeps >= `reserve` SoC (default 20%) at mission end; mass also satisfies the
    3C limit."""
    energy_j: float = sum(p.energy_j for p in trace.phases) / (1.0 - reserve)
    by_energy: float = energy_j / WH_TO_J / LIION_WH_PER_KG
    by_power: float = trace.peak_total_w / (LIION_MAX_C * LIION_WH_PER_KG)
    return EnergyStorage(energy_j, max(by_energy, by_power), trace.peak_total_w, 0.0)


def hybrid_storage(trace: MissionTrace, gen_w: float, fuel_reserve: float = FUEL_RESERVE) -> EnergyStorage:
    """Buffer LiPo: energy for the sustained-hover deficit (P_hover - P_gen) over 180 s + 60 s abort
    reserve plus any forward-flight deficit (e.g. climb); C-rate covers the installed-peak deficit.
    Fuel = burned fuel + `fuel_reserve` (default 10%)."""
    hover_deficit_w: float = max(trace.sustained_hover_w - gen_w, 0.0)
    needed_j: float = hover_deficit_w * (HOVER_BUFFER_S + ABORT_RESERVE_S) + trace.non_hover_deficit_j
    energy_j: float = max(needed_j, WH_TO_J) / (1.0 - MIN_SOC_RESERVE)  # floor: 1 Wh avionics backup
    peak_w: float = max(trace.peak_deficit_w, 1.0)
    by_energy: float = energy_j / WH_TO_J / LIPO_WH_PER_KG
    by_power: float = peak_w / (LIPO_MAX_C * LIPO_WH_PER_KG)
    return EnergyStorage(energy_j, max(by_energy, by_power), peak_w, trace.fuel_burned_kg * (1.0 + fuel_reserve))
