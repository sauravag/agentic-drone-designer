"""Propulsion matching search (spec 6.3-6.4): the series count, motor winding, propeller and ESC for the cruise
propulsor and the lift rotors that meet every rule at every flight point, at the least propulsion + energy-storage
mass.

Rules: throttle <= MAX_MATCH_THROTTLE at the nominal bus, and <= 1.0 at the end-of-reserve loaded bus for
check_sagged points; current <= the motor's max; shaft power within the motor's and the propeller's ratings; tip Mach
<= PROP_TIP_MACH_MAX; cruise diameter <= the layout's limit; the lift rotor is the smallest catalogue diameter at or
above the disk-loading minimum; the lightest ESC with continuous current >= ESC_CURRENT_MARGIN x the highest matched
current and voltage >= the charged pack; series hybrid: only series counts with a generator voltage variant inside
the bus window. Ranking uses electrical power only (never prop_efficiency at climb points).

Objective (controller's ruling 3): hardware mass + energy-storage mass for the mission energy at this set's electrical
powers (+ hotel load): battery = energy / specific energy; hybrid fuel = energy x BSFC x (1 + FUEL_SYSTEM_FRACTION).
Ties: lower hardware mass, then lower series count, then names (deterministic).
"""

import math
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Final, Optional, Union

from src.schemas.components import Cell, Esc, IceGenerator, Propeller
from src.schemas.propulsion import PropulsionMatch
from src.tools.database.packs import LOADED_MIN_V_CELL, bus_window_v
from src.tools.database.seed_data import catalog_rows
from src.tools.propulsion.match_result import assemble_match
from src.tools.propulsion.match_search import (
    MOTOR_OVERSIZE_FACTOR, PROP_OVERSIZE_FACTOR, RULE_MODEL, RULE_OVERSIZE, Bus, Drive, Motor, Option, Rejection,
    drive_for, hardest_ideal_w, option_at, propeller_shafts,
)
from src.tools.propulsion.match_types import (  # noqa: F401  (re-exported: the matching API lives here)
    ESC_CURRENT_MARGIN, MAX_MATCH_THROTTLE, MAX_SAGGED_THROTTLE, PROP_TIP_MACH_MAX, FlightPoint, MatchContext,
    PropulsionMatchError,
)
from src.tools.sizing.mass_model import FUEL_SYSTEM_FRACTION
from src.units import KWH_TO_J, WH_TO_J

DIAMETER_TOL_M: Final[float] = 1e-9
SERIES_EPS: Final[float] = 1e-6  # floor(rated voltage / 4.2 V) must not lose a cell to float noise
TOP_LIMITS_REPORTED: Final[int] = 3
MOTOR_CONSTANTS: Final[tuple[str, ...]] = ("kv", "rm_ohm", "i0_a", "max_current_a")
BusKey = tuple[str, float, float, int]
VoltageClass = tuple[str, float, float]


@dataclass(frozen=True)
class _RoleResult:
    best: dict[BusKey, Option]  # the best option per bus
    rejections: tuple[Rejection, ...]  # one per candidate that met no bus


def match_propulsion(context: MatchContext) -> PropulsionMatch:
    """The lightest feasible set (hardware + estimated energy storage), or PropulsionMatchError naming the limit."""
    classes: dict[VoltageClass, tuple[Cell, ...]] = _voltage_classes(context)
    buses: tuple[Bus, ...] = _buses(context, classes)
    cruise: _RoleResult = _role(context, False, buses)
    lift: Optional[_RoleResult] = _role(context, True, buses) if context.lift_rotor_count else None
    chosen: Optional[tuple[Option, Optional[Option]]] = _best_set(context, buses, cruise, lift)
    if chosen is None:
        raise _failure(context, cruise, lift)
    cruise_option, lift_option = chosen
    cells = classes[cruise_option.bus.key[:3]]
    return assemble_match(context, cruise_option, lift_option, cells, _hotel_kg(context, cruise_option.bus),
                          _skipped_propellers(cruise, lift))


def _skipped_propellers(cruise: _RoleResult, lift: Optional[_RoleResult]) -> tuple[str, ...]:
    """Catalogue propellers left out because their table could not be built (kept loud in the match notes)."""
    prefix: str = f"{RULE_MODEL}: "
    rejections = cruise.rejections + (lift.rejections if lift is not None else ())
    return tuple(sorted({r.rule[len(prefix):] for r in rejections if r.rule.startswith(prefix)}))


# ---------------------------------------------------------------- buses


def _storage_kg_per_j(context: MatchContext, chemistry: str) -> float:
    a = context.assumptions
    if context.is_hybrid:
        return a.bsfc_kg_per_kwh / KWH_TO_J * (1.0 + FUEL_SYSTEM_FRACTION)
    return 1.0 / (a.specific_energy_wh_kg(chemistry) * WH_TO_J)  # type: ignore[arg-type]


def _voltage_classes(context: MatchContext) -> dict[VoltageClass, tuple[Cell, ...]]:
    """Cells grouped by (chemistry, nominal, max voltage): a class fixes every bus voltage of a series count."""
    groups: dict[VoltageClass, tuple[Cell, ...]] = {}
    for cell in catalog_rows("cells"):
        if context.chemistry is None or cell.chemistry == context.chemistry:
            key: VoltageClass = (cell.chemistry, cell.nominal_voltage_v, cell.max_voltage_v)
            groups = {**groups, key: groups.get(key, ()) + (cell,)}
    if not groups:
        wanted: str = f"{context.chemistry} " if context.chemistry is not None else ""
        raise PropulsionMatchError(f"no {wanted}cells in the catalogue to build the pack from")
    return groups


def _max_series(max_cell_v: float) -> int:
    ratings: list[float] = [m.voltage_v for table in ("cruise_motors", "vtol_motors") for m in catalog_rows(table)]
    return max(int(math.floor(v / max_cell_v + SERIES_EPS)) for v in ratings)


def _generator_cap_w(cell: Cell, series: int, generators: tuple[IceGenerator, ...]) -> float:
    """Most continuous power among generator variants whose output sits inside the bus window (0 when none)."""
    low, high = bus_window_v(cell, series)
    return max((g.continuous_power_w for g in generators
                if g.output_voltage_v is not None and low <= g.output_voltage_v <= high), default=0.0)


def _buses(context: MatchContext, classes: dict[VoltageClass, tuple[Cell, ...]]) -> tuple[Bus, ...]:
    generators: tuple[IceGenerator, ...] = tuple(catalog_rows("ice_generators"))
    buses: list[Bus] = []
    for (chem, nominal, top), cells in classes.items():
        storage: float = _storage_kg_per_j(context, chem)
        for series in range(1, _max_series(top) + 1):
            cap: float = _generator_cap_w(cells[0], series, generators) if context.is_hybrid else math.inf
            if cap > 0.0:
                buses.append(Bus(chem, nominal, top, series, storage, cap))
    if not buses:
        raise PropulsionMatchError(_no_bus_message(classes, generators))
    return tuple(buses)


def _no_bus_message(classes: dict[VoltageClass, tuple[Cell, ...]], generators: tuple[IceGenerator, ...]) -> str:
    offered = sorted({g.output_voltage_v for g in generators if g.output_voltage_v is not None})
    windows = ", ".join(f"{chem}: S x {LOADED_MIN_V_CELL[chem]:g}-{top:g} V for S = 1-{_max_series(top)}"
                        for chem, _, top in classes)
    return (f"series hybrid: no generator output voltage lies inside the bus window (end-of-reserve loaded to "
            f"charged pack voltage) of any series count ({windows}); generator voltages offered: "
            f"{', '.join(f'{v:g} V' for v in offered) or 'none'}")


# ---------------------------------------------------------------- candidates


def _cruise_propellers(context: MatchContext) -> tuple[Propeller, ...]:
    props = tuple(p for p in catalog_rows("propellers") if p.application == "cruise"
                  and p.is_folding == context.require_folding_prop
                  and p.diameter_m <= context.max_cruise_prop_diameter_m + DIAMETER_TOL_M)
    if not props:
        kind: str = "folding " if context.require_folding_prop else ""
        raise PropulsionMatchError(f"diameter: no {kind}cruise propeller in the catalogue fits the layout's "
                                   f"{context.max_cruise_prop_diameter_m:.3f} m diameter limit")
    return props


def _lift_rotors(context: MatchContext) -> tuple[Propeller, ...]:
    rotors = [p for p in catalog_rows("propellers") if p.application == "vtol"
              and p.diameter_m >= context.min_lift_rotor_diameter_m - DIAMETER_TOL_M]
    if not rotors:
        raise PropulsionMatchError(f"diameter: no lift rotor in the catalogue reaches the disk-loading diameter "
                                   f"{context.min_lift_rotor_diameter_m:.3f} m")
    smallest: float = min(p.diameter_m for p in rotors)
    return tuple(p for p in rotors if p.diameter_m <= smallest + DIAMETER_TOL_M)


def _matchable(motor: Motor) -> bool:
    return all(getattr(motor, name) is not None for name in MOTOR_CONSTANTS)


def _prop_drives(prop: Propeller, motors: tuple[Motor, ...], points: tuple[FlightPoint, ...],
                 installation: float) -> tuple[tuple[Drive, ...], tuple[Rejection, ...]]:
    """Every motor winding on one propeller: drives, and one rejection per failed candidate. Pruned by power
    BEFORE the propeller's table is built."""
    ideal_w, hardest = hardest_ideal_w(prop, points, installation)
    if prop.max_power_w < ideal_w:
        return (), (Rejection("power (propeller rating)", hardest),)
    if prop.max_power_w > PROP_OVERSIZE_FACTOR * ideal_w:
        return (), (Rejection(RULE_OVERSIZE, ""),)  # skipped, not a limit (reported only if nothing else is left)
    sized = [m for m in motors if m.max_power_w <= MOTOR_OVERSIZE_FACTOR * ideal_w]
    weak = tuple(Rejection("power (motor rating)", hardest) for m in sized if m.max_power_w < ideal_w)
    usable = [m for m in sized if m.max_power_w >= ideal_w]
    if not usable:
        return (), weak or (Rejection(RULE_OVERSIZE, ""),)
    shafts = propeller_shafts(prop, points, installation)
    if isinstance(shafts, Rejection):
        return (), weak + (shafts,)
    results = [drive_for(prop, m, shafts) for m in usable]
    return (tuple(r for r in results if isinstance(r, Drive)),
            weak + tuple(r for r in results if isinstance(r, Rejection)))


def _drive_options(drive: Drive, buses: tuple[Bus, ...], escs: tuple[Esc, ...], count: int,
                   cap_w: Callable[[Bus], float]) -> tuple[tuple[Option, ...], Optional[Rejection]]:
    """The drive on every bus (buses run by rising series count). Without any, one rejection for the drive: the
    LAST one that is not the winding's voltage rating (which only says S went too high), i.e. the limit at the
    highest S the winding allows; if every bus hit the voltage rating, the first of those."""
    options: list[Option] = []
    telling: Optional[Rejection] = None
    for bus in buses:
        result: Union[Option, Rejection] = option_at(drive, bus, escs, count, cap_w(bus))
        if isinstance(result, Option):
            options.append(result)
        elif telling is None or not result.rule.startswith("motor voltage"):
            telling = result
    return tuple(options), (None if options else telling)


def _rank(option: Option) -> tuple[float, float, int, tuple[str, str, str]]:
    return option.objective_kg, option.hardware_kg, option.bus.series, option.names


def _role(context: MatchContext, is_lift: bool, buses: tuple[Bus, ...]) -> _RoleResult:
    """The best option per bus for the cruise propulsor (is_lift False) or the lift rotor set."""
    points = context.hover_points if is_lift else context.forward_points
    props = _lift_rotors(context) if is_lift else _cruise_propellers(context)
    motors = tuple(m for m in catalog_rows("vtol_motors" if is_lift else "cruise_motors") if _matchable(m))
    installation: float = 1.0 if is_lift else context.pusher_factor
    count: int = context.lift_rotor_count if is_lift else 1
    escs: tuple[Esc, ...] = tuple(sorted(catalog_rows("escs"), key=lambda e: (e.mass_kg, e.name)))

    def cap_w(bus: Bus) -> float:  # a hybrid's generator carries the cruise propulsor and the hotel load
        return math.inf if is_lift else bus.generator_cap_w - context.hotel_w

    best: dict[BusKey, Option] = {}
    rejections: list[Rejection] = []
    for prop in props:
        drives, rejected = _prop_drives(prop, motors, points, installation)
        rejections.extend(rejected)
        for drive in drives:
            options, why = _drive_options(drive, buses, escs, count, cap_w)
            rejections.extend([why] if why is not None else [])
            better = {o.bus.key: o for o in options if o.bus.key not in best or _rank(o) < _rank(best[o.bus.key])}
            best = {**best, **better}
    return _RoleResult(best, tuple(rejections))


# ---------------------------------------------------------------- choice and failure


def _hotel_kg(context: MatchContext, bus: Bus) -> float:
    return context.hotel_w * context.mission_duration_s * bus.storage_kg_per_j


def _best_set(context: MatchContext, buses: tuple[Bus, ...], cruise: _RoleResult,
              lift: Optional[_RoleResult]) -> Optional[tuple[Option, Optional[Option]]]:
    choices: list[tuple[tuple[object, ...], Option, Optional[Option]]] = []
    for bus in buses:
        c: Optional[Option] = cruise.best.get(bus.key)
        lo: Optional[Option] = lift.best.get(bus.key) if lift is not None else None
        if c is None or (lift is not None and lo is None):
            continue
        total: float = c.objective_kg + (lo.objective_kg if lo else 0.0) + _hotel_kg(context, bus)
        hardware: float = c.hardware_kg + (lo.hardware_kg if lo else 0.0)
        choices.append(((total, hardware, bus.series, c.names, lo.names if lo else ()), c, lo))
    if not choices:
        return None
    _, c, lo = min(choices, key=lambda item: item[0])
    return c, lo


def _limits(rejections: tuple[Rejection, ...]) -> str:
    limits = tuple(r for r in rejections if r.rule != RULE_OVERSIZE)
    if not limits and rejections:
        return (f"all candidates rated above {PROP_OVERSIZE_FACTOR:g}x ideal power (oversize prune): the thrust is "
                f"below what the catalogue's smallest propellers and motors are meant for")
    if not limits:
        return "no catalogue motor with matching constants or propeller survived pruning"
    unmodelled = sorted({r.rule[len(RULE_MODEL) + 2:] for r in limits if r.rule.startswith(RULE_MODEL)})
    others = [r for r in limits if not r.rule.startswith(RULE_MODEL)]
    counted = Counter((r.rule, r.point) for r in others).most_common(TOP_LIMITS_REPORTED)
    parts = [f"{rule}{' at ' + point if point else ''} ({n} of {len(limits)} candidates)"
             for (rule, point), n in counted]
    if unmodelled:  # one entry, so table failures never crowd the physical limits out of the report
        model_count: int = len(limits) - len(others)
        parts.append(f"{RULE_MODEL} for {', '.join(unmodelled)} ({model_count} of {len(limits)} candidates)")
    return "; ".join(parts)


def _failure(context: MatchContext, cruise: _RoleResult, lift: Optional[_RoleResult]) -> PropulsionMatchError:
    if cruise.best and lift is not None and lift.best:
        cruise_s = sorted({k[3] for k in cruise.best})
        lift_s = sorted({k[3] for k in lift.best})
        return PropulsionMatchError(f"no common bus voltage: the cruise propulsor works at S={cruise_s}, the lift "
                                    f"rotors at S={lift_s}")
    scopes = ((f"cruise propulsor (propellers up to {context.max_cruise_prop_diameter_m:.3f} m diameter)", cruise),
              (f"lift rotors (at least {context.min_lift_rotor_diameter_m:.3f} m diameter)", lift))
    parts: list[str] = [f"{scope}: {_limits(result.rejections)}" for scope, result in scopes
                        if result is not None and not result.best]
    return PropulsionMatchError("no propulsion set meets every matching rule; limiting constraints - "
                                + " | ".join(parts))
