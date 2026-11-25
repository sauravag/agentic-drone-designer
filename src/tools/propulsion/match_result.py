"""Turn the chosen options into a `PropulsionMatch` (spec 6.4): the operating points through the same `Propulsor`
the simulator uses, the efficiencies sizing consumes, the pack cell and the notes."""

import math
from typing import Final, Optional

from src.schemas.components import Battery, Cell
from src.schemas.propulsion import OperatingPoint, PropulsionMatch
from src.tools.database.packs import PackLimitError, build_pack
from src.tools.propulsion.match_search import Option, ideal_power_w
from src.tools.propulsion.match_types import FlightPoint, MatchContext, PropulsionMatchError
from src.tools.propulsion.prop_tables import propeller_table
from src.tools.propulsion.propulsor import Propulsor
from src.units import WH_TO_J

ESTIMATE_NOTE: Final[str] = ("estimates: generic blade shape (real propellers of the same D x P differ by several "
                             "%), motor Rm / I0 / max current, ESC efficiency and cell data")


def _propulsor(option: Option, installation: float, is_lift: bool) -> Propulsor:
    motor = option.drive.motor
    return Propulsor(
        table=propeller_table(option.drive.propeller.diameter_m, option.drive.propeller.pitch_m), kv=motor.kv,
        rm_ohm=motor.rm_ohm, i0_a=motor.i0_a, max_current_a=motor.max_current_a,
        esc_efficiency=option.esc.efficiency, bus_voltage_v=option.bus.nominal_v, installation_factor=installation,
        role="lift" if is_lift else "cruise",
    )


def _solve(propulsor: Propulsor, point: FlightPoint, bus_v: float, sagged: bool) -> OperatingPoint:
    op: Optional[OperatingPoint] = propulsor.point_for_thrust(
        point.name, point.thrust_n, point.airspeed_ms, point.rho, point.mu, point.speed_of_sound_ms,
        bus_voltage_v=bus_v, sagged=sagged)
    if op is None:  # the search accepted this point with the same tables and motor model: a bug, never silent
        raise PropulsionMatchError(f"internal: the matched {propulsor.role} propulsor cannot reach "
                                   f"{point.thrust_n:.4g} N at {point.name} ({'sagged' if sagged else 'nominal'})")
    return op


def _operating_points(option: Option, installation: float, is_lift: bool) -> tuple[OperatingPoint, ...]:
    """Every point at the nominal bus, and the check_sagged points again at the end-of-reserve loaded bus."""
    propulsor: Propulsor = _propulsor(option, installation, is_lift)
    ops: list[OperatingPoint] = []
    for shaft in option.drive.shafts:
        ops.append(_solve(propulsor, shaft.point, option.bus.nominal_v, sagged=False))
        if shaft.point.check_sagged:
            ops.append(_solve(propulsor, shaft.point, option.bus.sagged_v, sagged=True))
    return tuple(ops)


def _efficiency(point: FlightPoint, op: OperatingPoint, diameter_m: float) -> float:
    """Forward: thrust power / electrical power. Hover: ideal induced power / electrical power."""
    useful_w: float = (ideal_power_w(point.thrust_n, 0.0, point.rho, diameter_m) if point.is_hover
                       else point.thrust_n * point.airspeed_ms)
    return useful_w / op.electrical_power_w


def _efficiencies(options: tuple[Option, ...], ops: tuple[OperatingPoint, ...]) -> dict[str, float]:
    nominal: dict[str, OperatingPoint] = {op.flight_point: op for op in ops if not op.sagged}
    return {s.point.name: _efficiency(s.point, nominal[s.point.name], o.drive.propeller.diameter_m)
            for o in options for s in o.drive.shafts}


def _pack_energy_j(context: MatchContext, options: tuple[Option, ...]) -> float:
    """Electric: the mission energy with the hotel load. Hybrid buffer: the climb and hover (check_sagged) energy,
    or the mission energy when those points have no duration (an estimate; sizing owns the buffer)."""
    total: float = sum(o.energy_j for o in options) + context.hotel_w * context.mission_duration_s
    if not context.is_hybrid:
        return total
    peaks: float = sum(o.electrical_w(i) * s.point.duration_s
                       for o in options for i, s in enumerate(o.drive.shafts) if s.point.check_sagged)
    return peaks if peaks > 0.0 else total


def _peak_power_w(context: MatchContext, options: tuple[Option, ...]) -> float:
    return context.hotel_w + max(o.electrical_w(i) for o in options for i in range(len(o.drive.shafts)))


def _cell(context: MatchContext, options: tuple[Option, ...], cells: tuple[Cell, ...]) -> tuple[Cell, Battery]:
    """The cell of the bus's voltage class whose pack for this energy and peak power is lightest (then by name)."""
    series: int = options[0].bus.series
    energy, peak = _pack_energy_j(context, options), _peak_power_w(context, options)
    packs: list[tuple[Battery, Cell]] = []
    refused: list[PackLimitError] = []
    for c in cells:
        try:
            packs.append((build_pack(c, series, energy, peak, context.assumptions), c))
        except PackLimitError as exc:  # this cell's pack breaks the part limit; another cell's may not
            refused.append(exc)
    if not packs:
        raise refused[0]
    pack, cell = min(packs, key=lambda item: (item[0].mass_kg, item[1].name))
    return cell, pack


def _notes(context: MatchContext, options: tuple[Option, ...], hotel_kg: float, pack: Battery,
           skipped: tuple[str, ...]) -> tuple[str, ...]:
    storage_kg: float = sum(o.storage_kg for o in options) + hotel_kg
    energy_wh: float = (sum(o.energy_j for o in options) + context.hotel_w * context.mission_duration_s) / WH_TO_J
    kind: str = "fuel (with fuel system)" if context.is_hybrid else "battery"
    return (f"objective: hardware + estimated {kind} {storage_kg:.3f} kg for {energy_wh:.1f} Wh of mission energy",
            f"pack estimate {pack.name}: {pack.capacity_wh:.0f} Wh, {pack.mass_kg:.3f} kg",
            ESTIMATE_NOTE) + ((f"catalogue propellers skipped (their table cannot be built): {', '.join(skipped)}",)
                              if skipped else ())


def assemble_match(context: MatchContext, cruise: Option, lift: Optional[Option], cells: tuple[Cell, ...],
                   hotel_kg: float, skipped: tuple[str, ...] = ()) -> PropulsionMatch:
    options: tuple[Option, ...] = (cruise,) if lift is None else (cruise, lift)
    ops: tuple[OperatingPoint, ...] = _operating_points(cruise, context.pusher_factor, is_lift=False) + (
        _operating_points(lift, 1.0, is_lift=True) if lift is not None else ())
    cell, pack = _cell(context, options, cells)
    hardware: float = sum(o.hardware_kg for o in options)
    if not math.isfinite(hardware) or hardware <= 0.0:
        raise PropulsionMatchError(f"internal: matched hardware mass {hardware!r} kg is not positive")
    return PropulsionMatch(
        cell_name=cell.name, series=cruise.bus.series, cruise_motor=cruise.drive.motor.name,
        cruise_propeller=cruise.drive.propeller.name, cruise_esc=cruise.esc.name,
        lift_motor=lift.drive.motor.name if lift else None, lift_rotor=lift.drive.propeller.name if lift else None,
        lift_esc=lift.esc.name if lift else None, operating_points=ops, efficiencies=_efficiencies(options, ops),
        propulsion_mass_kg=hardware, notes=_notes(context, options, hotel_kg, pack, skipped),
        **_part_records(cruise, lift), cell_chemistry=cell.chemistry,
    )


def _part_records(cruise: Option, lift: Optional[Option]) -> dict[str, Optional[float]]:
    """Each matched part's mass and the propeller/rotor diameters, stored on the match."""
    return {
        "cruise_motor_kg": cruise.drive.motor.mass_kg, "cruise_propeller_kg": cruise.drive.propeller.mass_kg,
        "cruise_esc_kg": cruise.esc.mass_kg, "cruise_propeller_diameter_m": cruise.drive.propeller.diameter_m,
        "lift_motor_kg": lift.drive.motor.mass_kg if lift else None,
        "lift_rotor_kg": lift.drive.propeller.mass_kg if lift else None,
        "lift_esc_kg": lift.esc.mass_kg if lift else None,
        "lift_rotor_diameter_m": lift.drive.propeller.diameter_m if lift else None,
    }
